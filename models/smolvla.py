"""SmolVLA-450M adapter.

Phase decomposition (matches lerobot 0.3.2 modeling_smolvla.py, read from
site-packages on 2026-09-03):

  vision      : vision tower forward inside embed_prefix (via
                vlm_with_expert.embed_image -> vlm vision module)
  prefill_lm  : vlm_with_expert.forward(fill_kv_cache=True) -- computes the
                prefix KV cache over (image + language + state) tokens
  decode_step : each vlm_with_expert.forward(fill_kv_cache=False) -- one
                flow-matching Euler denoising step through the action expert,
                attending to the cached prefix; config.num_steps steps total

So SmolVLA literally runs prefill + KV cache + decode inside a single control
step, with the cache reused across denoise steps (not across control steps,
because the image changes). This mapping feeds the README's serving section.

Chunk length: policy.config.chunk_size drives the generated horizon
(noise shape and suffix length); n_action_steps is irrelevant here because we
call predict_action_chunk directly.

Reproducibility: predict_action_chunk accepts an explicit `noise` tensor; we
build per-run noise from a CPU generator seeded by (args.seed, run_idx) so
quantized runs see bit-identical noise to the BF16 reference.
"""
import torch
from torch import nn

from benchmarks.latency import HookTimer


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
        return None  # not a path we instrument
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
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        if self.args.chunk:
            self.policy.config.chunk_size = int(self.args.chunk)
        if self.args.precision == "bf16":
            self.policy.model.to(torch.bfloat16)  # deployment-style cast; fp32 fallback logged if unstable
        self.policy.to("cuda:0")
        self.policy.eval()
        self._param_dtype = next(self.policy.model.parameters()).dtype

        if self.args.precision in ("int8", "int4"):
            self._quantize_(self.args.precision)

    def _quantize_(self, precision: str):
        import bitsandbytes as bnb

        n_replaced = 0

        def repl(m: nn.Module):
            nonlocal n_replaced
            for name, child in m.named_children():
                if isinstance(child, nn.Linear):
                    dt = child.weight.dtype
                    if precision == "int8":
                        new = bnb.nn.Linear8bitLt(
                            child.in_features, child.out_features,
                            bias=child.bias is not None,
                            has_fp16_weights=False, threshold=6.0,
                        )
                        new.weight = bnb.nn.Int8Params(
                            child.weight.data, requires_grad=False, has_fp16_weights=False
                        )
                    else:
                        new = bnb.nn.Linear4bit(
                            child.in_features, child.out_features,
                            bias=child.bias is not None,
                            quant_type="nf4", compute_dtype=dt,
                        )
                        new.weight = bnb.nn.Params4bit(
                            child.weight.data, requires_grad=False,
                            quant_type="nf4", compute_dtype=dt,
                        ).to(child.weight.device)
                    if child.bias is not None:
                        new.bias = child.bias
                    setattr(m, name, new)
                    n_replaced += 1
                else:
                    repl(child)

        repl(self.policy.model)
        # bnb quantizes lazily (int8 on first forward, 4bit on device move);
        # warmup steps in bench.py settle it before any measurement.

    def attach_hooks(self, profiler):
        vis_name, vis_mod = find_first_module(self.policy.model.vlm_with_expert, "vision")
        if vis_mod is None:
            vis_name, vis_mod = find_first_module(self.policy.model.vlm_with_expert.vlm, "vision")
        if vis_mod is None:
            vis_name, vis_mod = find_first_module(self.policy.model.vlm_with_expert.vlm, "encoder")
        assert vis_mod is not None, "vision module not found in SmolVLA tree"
        self.vision_module_path = vis_name
        t1 = HookTimer(profiler, "vision")
        t1.attach(vis_mod)
        t2 = HookTimer(profiler, "vlm", classify=classify_vlm_call)
        t2.attach(self.policy.model.vlm_with_expert)
        self._hooks = [t1, t2]

    def detach_hooks(self):
        for h in self._hooks:
            h.detach()
        self._hooks = []

    # ---- one control step ----
    def _make_noise(self, bsize: int, run_idx: int) -> torch.Tensor:
        shape = (bsize, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        g = torch.Generator(device="cpu").manual_seed(self.args.seed * 100003 + run_idx)
        noise = torch.randn(shape, generator=g, dtype=torch.float32)
        return noise.to("cuda:0", dtype=self._param_dtype)

    def build_batch(self, fixture, batch_size: int) -> dict:
        """Map fixture tensors onto the policy's expected input feature keys."""
        feats = dict(self.policy.config.input_features)
        img_keys = [k for k in feats if "image" in k.lower()]
        state_keys = [k for k in feats if "state" in k.lower()]
        batch = {}
        imgs = fixture["images"][:batch_size].to("cuda:0", dtype=self._param_dtype)
        for k in img_keys:
            batch[k] = imgs
        st = fixture["states"][:batch_size].to("cuda:0", dtype=torch.float32)
        feat_shape = None
        for k in state_keys:
            feat_shape = tuple(feats[k].shape)
            if len(feat_shape) == 1 and st.shape[-1] < feat_shape[0]:
                pad = torch.zeros((*st.shape[:-1], feat_shape[0] - st.shape[-1]), device=st.device, dtype=st.dtype)
                st = torch.cat([st, pad], dim=-1)
            elif len(feat_shape) == 1 and st.shape[-1] > feat_shape[0]:
                st = st[..., : feat_shape[0]]
            batch[k] = st
        batch["task"] = [fixture["task"]] * batch_size
        return batch

    def step(self, fixture, run_idx: int) -> dict:
        import time

        batch = self.build_batch(fixture, self.args.batch)
        noise = self._make_noise(self.args.batch, run_idx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch, noise=noise)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return {
            "actions": actions.float().cpu().numpy(),
            "tokens": None,
            "wall_ms": {"e2e": (t1 - t0) * 1e3},
        }
