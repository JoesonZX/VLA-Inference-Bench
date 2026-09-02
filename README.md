# VLA-Inference-Bench

A single-GPU benchmark that profiles the **inference cost of Vision-Language-Action models** and quantifies what post-training INT8/INT4 quantization and action chunking buy — and what they cost.

**Models:** OpenVLA-7B (single-action, autoregressive decode) · SmolVLA-450M (50-action chunks, flow-matching decode)
**Hardware for all numbers below:** one physical H100 80GB, GPU-idle verified before every run, warmup 5 + 30 timed steps, fixed inputs (8 frames of `lerobot/pusht`, rotated per run), bit-identical seeded noise across precisions.

## Headline results

1. **OpenVLA-7B emits one action in ~195 ms (BF16, 14.1 GB weights). NF4 INT4 cuts weights to 4.1 GB (−71%) at essentially unchanged latency (+3%)** — the memory-bound decode phase gets *faster* (127→101 ms; half the weight traffic) while the compute-bound prefill gets slower (31→53 ms; dequant overhead). Whether quantization speeds up or slows down VLA inference depends on which side of the roofline the load sits.
2. **Action chunk length is nearly free for chunked policies.** SmolVLA chunk 1→50 raises step latency only 168→176 ms (+4.6%) because its decode is parallel flow-matching, not autoregression — amortized per-action cost drops **47×** (168 ms → 3.5 ms). Chunking is a cleaner lever than quantization; its cost is control reactivity, not latency.
3. **bitsandbytes INT8 is a bad trade at single-step VLA loads:** the default outlier-decomposition setting is 17–34× slower than baseline; even with `threshold=0` it is 52–58% slower than BF16/FP32 on both models — while INT4's output deviation is real and grows with chunk length (chunk MSE 0.035→0.244 for chunk 1→50). Pick NF4 over INT8 on this path, and validate deviation at the task level before deployment.

| model | precision | e2e (ms) | weights (GB) | deviation vs reference |
|---|---|---|---|---|
| OpenVLA-7B | BF16 | 195.0 ± 2.0 | 14.09 | — |
| OpenVLA-7B | INT8 | 308.2 ± 4.8 | 7.43 | action L2 0.147 |
| OpenVLA-7B | INT4 | 201.3 ± 2.0 | 4.08 | action L2 0.523 |
| SmolVLA-450M | FP32 (as shipped) | 176.2 ± 1.9 | 0.90 | — |
| SmolVLA-450M | INT8 | 267.1 ± 2.1 | 0.65 | chunk MSE 0.035 |
| SmolVLA-450M | INT4 | 213.9 ± 5.7 | 0.46 | chunk MSE 0.244 |

## Figures

| figure | what it shows |
|---|---|
| `plots/fig_openvla_precision.png` | OpenVLA per-phase latency stack by precision (decode dominates; INT4 shifts cost prefill↔decode) |
| `plots/fig_smolvla_precision.png` | SmolVLA per-phase stack by precision (chunk=50) |
| `plots/fig_smolvla_chunk.png` | SmolVLA phase stack vs chunk length (prefill/vision flat, decode barely grows) |
| `plots/fig_quant_tradeoff.png` | memory-vs-latency scatter: what each precision buys and costs |
| `plots/fig_chunk_curve.png` | chunk length vs latency per precision + amortized ms/action |
| `plots/fig_deviation.png` | token mismatch / action L2 / chunk MSE vs reference |
| `plots/fig_batch.png` | SmolVLA batching: ×6.6 aggregate throughput at batch 8 (decode is bandwidth-bound → batches almost for free) |

Full narrative with methodology, the LLM-serving concept mapping (prefill/decode, KV-cache reuse across control steps, continuous batching ↔ multi-robot), and limitations: **[REPORT.md](REPORT.md)**. Build log with every error and fix: **[LOG.md](LOG.md)**.

## Repo layout

```
bench.py              # single CLI entry for every measurement -> results/*.json
models/               # per-model adapters incl. phase-timing instrumentation (docstrings cite the exact upstream source lines)
benchmarks/           # CUDA-event phase timing, memory, deviation metrics
data/fixtures.py      # builds the fixed-frame input fixture from lerobot/pusht (committed)
plots/make_plots.py   # JSON -> all figures + results/summary.md
plots/summarize.py    # console table of every JSON
results/              # one JSON per configuration (committed; figures regenerate from these only)
scripts/run_formal.sh # the full formal sweep as run on the H100
```

## Protocol

- All formal data on **one physical GPU**; `nvidia-smi` idle check recorded in each JSON's meta (all runs idle=True).
- Fixed inputs: 8 frames of `lerobot/pusht` episode 0 + fixed instruction, frame indices baked into the fixture; frames rotate per run index so deviation covers all 8 inputs.
- Warmup 5, then 30 timed steps (one config re-measured with 50 after a transient cluster blip); mean ± std, p50/p99; CUDA events for phases, `torch.cuda.synchronize()` at step boundaries.
- Deviation is measured against a saved reference run (OpenVLA: BF16; SmolVLA: FP32 as shipped by lerobot) with bit-identical seeded noise and paired run indices.
- Versions: torch 2.14.0+cu130, transformers 4.46.3/4.57.1, lerobot 0.4.4, bitsandbytes 0.50.2, timm 0.9.16 (hard requirement of OpenVLA's remote code).

## Reproduce

```bash
# one H100 (same physical GPU for every config), HF_HOME pointing at a weight cache
python data/fixtures.py
bash scripts/run_formal.sh
python plots/make_plots.py
```

## Non-goals

Training/fine-tuning, custom CUDA kernels, multi-node serving, simulation success rates (numeric deviation only — future work), cross-GPU comparisons (every table is same-card).

## Known limitations

- OpenVLA's upstream generation code does not support batch > 1; batching is benchmarked on SmolVLA only.
- Shared cluster GPU: clocks are not locked; idle-at-start verified and recorded instead. Two configs showed transient bimodal latency once and were re-measured clean (LOG.md).
- SmolVLA loads mixed-precision as shipped (bf16 towers + fp32 action heads); INT8/INT4 quantization excludes q/k/v/o projections (lerobot's internal dtype glue reads their `.weight.dtype`) — the exclusion set is identical for both precisions.
- bitsandbytes INT8 kernels cast bf16 inputs to fp16; the "INT8" path is not pure int8 arithmetic.
