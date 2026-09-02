# VLA-Inference-Bench

A single-GPU benchmark that profiles and compares the **inference cost of Vision-Language-Action models**, and quantifies what post-training INT8/INT4 quantization and action chunking buy — and what they cost.

**Models:** OpenVLA-7B (single-action, autoregressive decode) · SmolVLA-450M (action-chunk, flow-matching decode)

**Measured per configuration:** end-to-end per-step latency with a phase decomposition (vision encoding / prefill / decode), weight + peak activation memory, and numerical deviation of quantized outputs vs BF16 (action-token mismatch + action L2 / chunk MSE).

## Headline results

*(filled in as formal runs complete on the H100 — see REPORT.md)*

## Repo layout

```
bench.py              # single CLI entry for every measurement -> results/*.json
models/               # per-model adapters incl. phase-timing hooks (read the source notes in each docstring)
benchmarks/           # latency hooks, memory, deviation metrics
data/fixtures.py      # builds the fixed-frame input fixture from lerobot/pusht (committed)
plots/make_plots.py   # JSON -> figures; never edits data by hand
results/              # one JSON per configuration (committed)
LOG.md                # chronological build log with every error and fix
REPORT.md             # the smooth narrative + results
```

## Protocol

- All formal data is collected on **one physical GPU** (H100 80GB) with a same-card idle check (`nvidia-smi`) recorded in each JSON's meta.
- Fixed inputs: 8 frames of episode 0 of `lerobot/pusht`, indices baked into the fixture file; identical seeded noise per run index for SmolVLA's flow matching.
- Warmup 5, then ≥30 timed steps per configuration; mean ± std + p50/p99 reported; CUDA events for phase timing, `torch.cuda.synchronize()` at step boundaries.
- Every number lives in `results/*.json`; figures are regenerated from JSONs only.

## Reproduce

```bash
# server, GPU 1, HF_HOME=/storage/xuan/vlabench/hf_cache
python data/fixtures.py                     # build fixed inputs (once)
python bench.py --model smolvla --precision bf16 --chunk 50 --save-ref
python bench.py --model smolvla --precision int4 --chunk 50 --deviation-ref results/ref_smolvla_bf16_chunk50.pt
python bench.py --model openvla --precision int4
python plots/make_plots.py
```

## Non-goals

Training/fine-tuning, custom CUDA kernels, multi-node serving, simulation success rates (numeric deviation only — future work), cross-GPU comparisons (every table is same-card).

## Known limitations

- OpenVLA's upstream generation code does not support batch > 1; batching is benchmarked on SmolVLA only.
- Shared cluster GPU: clocks are not locked; idle-at-start is verified and recorded instead.
