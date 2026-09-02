"""OpenVLA-7B adapter.

Phase decomposition (matches modeling_prismatic.py source, read from the
openvla/openvla-7b snapshot on 2026-09-03):

  vision_backbone : PrismaticVisionBackbone forward (SigLIP towers)  -- prefill only
  vision_projector: PrismaticProjector forward                        -- prefill only
  prefill_lm      : language_model forward on multimodal embeds (256 img + text)
  decode          : 6 x language_model forward on the last token (7 tokens generated,
                    the 1st comes out of the prefill forward)

`predict_action` is replicated here (instead of calling the upstream method) so
we can capture the generated action token ids for deviation metrics and time
the post-processing separately. Logic copied verbatim from
OpenVLAForActionPrediction.predict_action, including its quirk of appending
token 29871 to input_ids without extending attention_mask.

NOTE: the upstream remote code raises for batch > 1 in
prepare_inputs_for_generation -- batched generation is unsupported by the
model's own generation path. The batch dimension here can only be used by
looping; bench.py refuses --batch > 1 for openvla.
"""
import time

import numpy as np
import torch

PROMPT_TMPL = "In: What action should the robot take to {task}?\nOut:"
EMPTY_TOKEN_ID = 29871  # Llama tokenizer empty token '' appended after "Out:"


def classify_lm_call(args, kwargs):
    """language_model forward -> 'prefill_lm' | 'decode'."""
    from benchmarks.latency import seq_len_of

    n = seq_len_of(args, kwargs)
    if n is None:
        return "prefill_lm"
    return "decode" if n == 1 else "prefill_lm"


class OpenVLAAdapter:
    name = "openvla-7b"
    supports_batch = False

    def __init__(self, args):
        self.args = args
        self.unnorm_key = "bridge_orig"  # stats key for un-normalization (semantic validity not required for latency/deviation)
        self.model = None
        self.processor = None
        self._hooks = []

    # ---- lifecycle ----
    def load(self):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        kws = dict(trust_remote_code=True, device_map={"": 0}, torch_dtype=torch.bfloat16)
        if self.args.precision == "int8":
            from transformers import BitsAndBytesConfig

            kws["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=0.0,  # outlier decomposition at 6.0 was ~34x slower on H100 (see LOG.md)
            )
        elif self.args.precision == "int4":
            from transformers import BitsAndBytesConfig

            kws["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            if self.args.precision != "bf16":
                raise ValueError(self.args.precision)

        self.model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", **kws)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    def attach_hooks(self, profiler):
        from benchmarks.latency import HookTimer

        t1 = HookTimer(profiler, "vision_backbone")
        t1.attach(self.model.vision_backbone)
        t2 = HookTimer(profiler, "vision_projector")
        t2.attach(self.model.projector)
        t3 = HookTimer(profiler, "lm", classify=classify_lm_call)
        t3.attach(self.model.language_model)
        self._hooks = [t1, t2, t3]

    def detach_hooks(self):
        for h in self._hooks:
            h.detach()
        self._hooks = []

    # ---- one control step ----
    def step(self, fixture, run_idx: int) -> dict:
        from PIL import Image

        # rotate input frames across runs so deviation metrics cover every fixture
        # frame, not one deterministic image
        n = fixture["images"].shape[0]
        img_np = fixture["images"][run_idx % n].numpy()
        image = Image.fromarray(np.transpose(img_np, (1, 2, 0)).astype(np.uint8)).convert("RGB")
        prompt = PROMPT_TMPL.format(task=fixture["task"])

        # -- CPU preprocessing (tokenize + image transform) --
        t0 = time.perf_counter()
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        # replicate upstream: append the empty token '' if missing
        if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([EMPTY_TOKEN_ID]).long(), dim=0).to(input_ids.device)),
                dim=1,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # -- generate (prefill + decode; phase times come from hooks) --
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.model.get_action_dim(self.unnorm_key),
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # -- postprocess: detokenize + unnormalize (replicated from predict_action) --
        n_act = self.model.get_action_dim(self.unnorm_key)
        predicted_action_token_ids = generated_ids[0, -n_act:].cpu().numpy()
        discretized_actions = self.model.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.model.bin_centers.shape[0] - 1)
        normalized_actions = self.model.bin_centers[discretized_actions]

        action_norm_stats = self.model.get_action_stats(self.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        t3 = time.perf_counter()

        return {
            "actions": actions.astype(np.float32),
            "tokens": predicted_action_token_ids.astype(np.int64),
            "wall_ms": {
                "preprocess": (t1 - t0) * 1e3,
                "generate": (t2 - t1) * 1e3,
                "postprocess": (t3 - t2) * 1e3,
                "e2e": (t3 - t0) * 1e3,
            },
        }
