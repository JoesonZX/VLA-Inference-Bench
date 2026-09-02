"""SmolVLA-450M adapter (lerobot >= 0.4.4 processor pipeline).

The lerobot 0.4.x checkpoint format splits inference into three stages:

    raw frame dict -> preprocess (normalize + tokenize) -> policy.predict_action_chunk
                   -> postprocess (unnormalize)

Precision modes and why there are two code paths:

* fp32  : the as-shipped path (`policy.predict_action_chunk`). The lerobot
  checkpoint is stored bf16 but is upcast on load, and upstream evaluation runs
  it in fp32; the flow-matching integration contains a hard `suffix_out.to(float32)`.
  This is the honest baseline.
* bf16 / int8 / int4 : lerobot has no low-precision deployment path of its own, and a
  plain `.to(bfloat16)` crashes on that hard fp32 cast. We replicate
  `VLAFlowMatching.sample_actions` / `denoise_step` verbatim (lerobot 0.4.4)
  minus the fp32 cast, with explicit dtype control: weights are cast/quantized
  uniformly, noise/x_t/timestep follow the weight dtype, and the Euler update
  casts explicitly to avoid in-place promotion errors. The numeric drift this
  introduces (bf16 flow integration) is exactly what a full low-precision
  deployment produces, and it shows up in the deviation metrics.

Phase decomposition (verified against modeling_smolvla.py on 2026-09-03):

  vision      : vision tower forward inside embed_prefix (vlm_with_expert.embed_image)
  prefill_lm  : vlm_with_expert.forward(fill_kv_cache=True) -- computes the prefix KV
                cache over (image + language + state) tokens
  decode_step : each vlm_with_expert.forward(fill_kv_cache=False) -- one flow-matching
                Euler denoising step through the action expert, reusing the cached
                prefix; config.num_steps steps (default 10) per control step

The base config declares three cameras; our pusht fixture feeds camera1 and
prepare_images zero-pads the missing slots (the model's own missing-camera path).

Reproducibility: explicit per-run `noise` from a CPU generator seeded by
(args.seed, run_idx); quantized runs see bit-identical noise to the reference.
"""
import time

import torch
from torch import nn

from benchmarks.latency import HookTimer

MODEL_ID = "lerobot/smolvla_base"


def find_first_module(root, class_substr: str):
    for name, mod in root.named_modules():
        if mod is root:
            continue
        if class_substr.lower() in type(mod).__name__.lower():
            return name, mod
    return None, None


def classify_vlm_call(args, kwargs):
    fill = kwargs.get("fill_kv_cache")
    if fill is None:
        return None
    return "prefill_lm" if bool(fill) else "decode_step"


class SmolVLAAdapter:
    name = "smolvla-450m"
    supports_batch = True

    def __init__(self, args):
        self.args = args
        self.policy = None
        self._hooks = []
        self._param_dtype = None

    # ---- lifecycle ----
    def load(self):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
        if self.args.chunk:
            self.policy.config.chunk_size = int(self.args.chunk)
        self.policy.to("cuda:0")
        self.policy.eval()

        if self.args.precision == "bf16":
            self.policy.model.to(torch.bfloat16)
        elif self.args.precision in ("int8", "int4"):
            self.policy.model.to(torch.bfloat16)  # bnb kernels take bf16/fp16 inputs
            self._quantize_(self.args.precision)
        self._param_dtype = next(self.policy.model.parameters()).dtype

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": "cuda:0"}},
        )

    def _quantize_(self, precision: str):
        """Replace nn.Linear with bnb layers, EXCEPT q/k/v/o attention projections.

        lerobot's SmolVLMWithExpertModel casts hidden/key/value states with
        `.to(dtype=...q_proj.weight.dtype)` glue; bnb Params4bit/Int8Params report
        integer storage dtypes there (uint8/int8), which would poison the cast.
        Keeping attention projections in bf16 keeps the glue correct; MLPs and all
        other Linears are quantized. The same exclusion set is used for int8 and
        int4 so the two precisions remain comparable.
        """
        import bitsandbytes as bnb

        n_replaced, n_skipped = 0, 0
        skip_substr = ("q_proj", "k_proj", "v_proj", "o_proj")

        def repl(m: nn.Module):
            nonlocal n_replaced, n_skipped
            for name, child in m.named_children():
                if isinstance(child, nn.Linear):
                    if any(s in name for s in skip_substr):
                        n_skipped += 1
                        continue
                    if precision == "int8":
                        # threshold=0.0: no outlier decomposition. With threshold=6.0
                        # bnb int8 was ~17x slower than bf16 on H100 (see LOG.md).
                        new = bnb.nn.Linear8bitLt(
                            child.in_features, child.out_features,
                            bias=child.bias is not None,
                            has_fp16_weights=False, threshold=0.0,
                        )
                        new.weight = bnb.nn.Int8Params(
                            child.weight.data, requires_grad=False, has_fp16_weights=False
                        )
                    else:
                        new = bnb.nn.Linear4bit(
                            child.in_features, child.out_features,
                            bias=child.bias is not None,
                            quant_type="nf4", compute_dtype=child.weight.dtype,
                        )
                        new.weight = bnb.nn.Params4bit(
                            child.weight.data, requires_grad=False,
                            quant_type="nf4", compute_dtype=child.weight.dtype,
                        ).to(child.weight.device)
                    if child.bias is not None:
                        new.bias = child.bias
                    setattr(m, name, new)
                    n_replaced += 1
                else:
                    repl(child)

        repl(self.policy.model)
        self.n_quantized_linears = n_replaced
        self.n_skipped_linears = n_skipped
        # bnb quantizes lazily; warmup steps in bench.py settle it before measuring.

    def attach_hooks(self, profiler):
        vwe = self.policy.model.vlm_with_expert
        vis_name, vis_mod = (
            find_first_module(vwe, "vision")
            or find_first_module(vwe.vlm, "vision")
            or find_first_module(vwe.vlm, "encoder")
        )
        assert vis_mod is not None, "vision module not found in SmolVLA tree"
        self.vision_module_path = vis_name
        t1 = HookTimer(profiler, "vision")
        t1.attach(vis_mod)
        # lerobot calls vlm_with_expert.forward(...) directly (bypassing nn.Module.__call__),
        # so forward hooks never fire there; time it with an instance-level patch instead.
        orig_fwd = vwe.forward

        def _timed(*a, **k):
            fill = k.get("fill_kv_cache")
            if fill is None:
                return orig_fwd(*a, **k)
            name = "prefill_lm" if bool(fill) else "decode_step"
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            out = orig_fwd(*a, **k)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            profiler.add(name, s, e)
            return out

        vwe.forward = _timed
        self._orig_vwe_forward = orig_fwd
        self._hooks = [t1]

    def detach_hooks(self):
        for h in self._hooks:
            h.detach()
        self._hooks = []
        if getattr(self, "_orig_vwe_forward", None) is not None:
            self.policy.model.vlm_with_expert.forward = self._orig_vwe_forward
            self._orig_vwe_forward = None

    # ---- low-precision replicas of VLAFlowMatching.sample_actions / denoise_step ----
    def _denoise_step_lowp(self, m, prefix_pad_masks, past_key_values, x_t, timestep):
        mod = self._modeling
        suffix_embs, suffix_pad_masks, suffix_att_masks = m.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = mod.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = m.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=m.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -m.config.chunk_size :]
        # upstream: suffix_out.to(float32) -- dropped for the low-precision path
        v_t = m.action_out_proj(suffix_out)
        return v_t

    def _sample_actions_lowp(self, m, images, img_masks, lang_tokens, lang_masks, state, noise):
        mod = self._modeling
        bsize = state.shape[0]
        device = state.device
        dt = torch.tensor(-1.0 / m.config.num_steps, dtype=noise.dtype, device=device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = mod.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = m.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=m.config.use_cache,
            fill_kv_cache=True,
        )

        x_t = noise
        time = torch.tensor(1.0, dtype=noise.dtype, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step_lowp(m, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + (dt * v_t).to(x_t.dtype)  # explicit cast: avoid in-place promotion
            time = time + dt
        return x_t

    # ---- one control step ----
    def _make_noise(self, bsize: int, run_idx: int) -> torch.Tensor:
        shape = (bsize, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        g = torch.Generator(device="cpu").manual_seed(self.args.seed * 100003 + run_idx)
        noise = torch.randn(shape, generator=g, dtype=torch.float32)
        if self.args.precision != "fp32" and self._param_dtype != torch.float32:
            noise = noise.to(self._param_dtype)  # low-precision path runs a uniform-dtype model
        return noise.to("cuda:0")

    def build_raw_batch(self, fixture, batch_size: int, run_idx: int = 0) -> dict:
        # rotate input frames across runs so deviation metrics cover every fixture
        # frame; batch rows stay distinct (B consecutive frames, wrapped)
        n = fixture["images"].shape[0]
        sel = [(run_idx + i) % n for i in range(batch_size)]
        img_key = next(iter(self.policy.config.image_features))
        state_key = next(k for k in self.policy.config.input_features if "state" in k)
        imgs = fixture["images"][sel].to("cuda:0", dtype=torch.float32) / 255.0
        st = fixture["states"][sel].to("cuda:0", dtype=torch.float32)
        feat_shape = tuple(self.policy.config.input_features[state_key].shape)
        if st.shape[-1] < feat_shape[0]:
            pad = torch.zeros((*st.shape[:-1], feat_shape[0] - st.shape[-1]), device=st.device, dtype=st.dtype)
            st = torch.cat([st, pad], dim=-1)
        return {img_key: imgs, state_key: st, "task": [fixture["task"]] * batch_size}

    def step(self, fixture, run_idx: int) -> dict:
        import lerobot.policies.smolvla.modeling_smolvla as modeling

        self._modeling = modeling
        raw = self.build_raw_batch(fixture, self.args.batch, run_idx)
        noise = self._make_noise(self.args.batch, run_idx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = self.preprocess(raw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            if self.args.precision == "fp32":
                actions = self.policy.predict_action_chunk(batch, noise=noise)
            else:
                # replicate _get_action_chunk's glue around sample_actions
                pol = self.policy
                if self._param_dtype != torch.float32:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor) and v.is_floating_point():
                            batch[k] = v.to(self._param_dtype)
                images, img_masks = pol.prepare_images(batch)
                state = pol.prepare_state(batch)
                lang_tokens = batch[getattr(modeling, "OBS_LANGUAGE_TOKENS", "observation.language_tokens")]
                lang_masks = batch[getattr(modeling, "OBS_LANGUAGE_ATTENTION_MASK", "observation.language_attention_mask")]
                actions = self._sample_actions_lowp(
                    pol.model, images, img_masks, lang_tokens, lang_masks, state, noise=noise
                )
                original_action_dim = pol.config.action_feature.shape[0]
                actions = actions[:, :, :original_action_dim]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        actions = self.postprocess(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.float().cpu().numpy()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return {
            "actions": actions,
            "tokens": None,
            "wall_ms": {
                "preprocess": (t1 - t0) * 1e3,
                "policy": (t2 - t1) * 1e3,
                "postprocess": (t3 - t2) * 1e3,
                "e2e": (t3 - t0) * 1e3,
            },
        }
