# VLA-Inference-Bench

A single-GPU benchmark that profiles the **inference cost of Vision-Language-Action models** and quantifies what post-training INT8/INT4 quantization and action chunking buy — and what they cost.

> **Follow-up project:** [vla-quant-robust](https://github.com/JoesonZX/vla-quant-robust) — asks whether the INT4 deployment recommendation carries a security price: white-box PGD across BF16/INT8/INT4 finds that OpenVLA's own action binning caps attack deviation, and that quantization sets the *relative* noise floor attacks are judged against (INT4 masks, INT8 exposes).

**Models:** OpenVLA-7B (single-action, autoregressive decode) · SmolVLA-450M (50-action chunks, flow-matching decode)
**Hardware for all numbers below:** one physical H100 80GB, GPU-idle verified before every run, warmup 5 + 30 timed steps, fixed inputs (8 frames of `lerobot/pusht`, rotated per run), bit-identical seeded noise across precisions.

## Headline results

1. **OpenVLA-7B emits one action in ~195 ms (BF16, 14.1 GB weights). NF4 INT4 cuts weights to 4.1 GB (−71%) at essentially unchanged latency (+3%)** — the memory-bound decode phase gets *faster* (127→101 ms; half the weight traffic) while the compute-bound prefill gets slower (31→53 ms; dequant overhead). Whether quantization speeds up or slows down VLA inference depends on which side of the roofline the load sits.
2. **Action chunk length is nearly free for chunked policies.** SmolVLA chunk 1→50 raises step latency only 168→176 ms (+4.6%) because its decode is parallel flow-matching, not autoregression — amortized per-action cost drops **47×** (168 ms → 3.5 ms). Chunking is a cleaner lever than quantization; its cost is control reactivity, not latency.
3. **bitsandbytes INT8 is a bad trade at single-step VLA loads:** the default outlier-decomposition setting is 17–34× slower than baseline; even with `threshold=0` it is 52–58% slower than BF16/FP32 on both models — while INT4's output deviation is real and grows with chunk length (chunk MSE 0.035→0.244 for chunk 1→50). Pick NF4 over INT8 on this path, and validate deviation at the task level before deployment.
4. **The first serving wins are in the glue and the executor, not the model — and they are free.** Profiling SmolVLA's step shows 11,253 kernel launches and 32 stream syncs (146 ms of launch gap per ~211 ms step). Two pure-execution rewrites, both with **bit-identical outputs**: (a) hoisting the recomputed-per-denoise-step time embeddings/masks and de-tensorizing loop control (−24%), and (b) capturing the whole 10-step flow-matching denoise loop in a **CUDA graph** with a static KV buffer (−45% total: 176→96 ms at chunk=50, latency now near-flat across chunk lengths, 1.9 ms amortized per action). Zero quantization, zero algorithm changes — the LLM-serving playbook (static KV pool, decode CUDA graphs) transfers to VLA inference directly. See REPORT §4.5.

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
| `plots/fig_hoist.png` | execution-layer ladder: upstream → hoist (−24%) → CUDA-graph decode (−45%), bit-identical outputs |

Full narrative with methodology and limitations: **[REPORT.md](REPORT.md)**. Build log with every error and fix: **[LOG.md](LOG.md)**.

## From LLM Serving to VLA Inference

LLM serving engines (reference: LMSYS's teaching engine [Mini-SGLang](https://github.com/sgl-project/mini-sglang), ~9k lines — guided reading notes in [docs/minisglang-notes.md](docs/minisglang-notes.md)) are organized around prefill/decode two-phase scheduling, a paged KV cache with radix prefix reuse, and continuous batching. Mapping each concept onto a VLA control loop — with our measurements — tells you exactly which serving tricks transfer:

| Serving concept | Mini-SGLang locus | VLA counterpart (measured here) |
|---|---|---|
| Prefill (compute-bound) | `scheduler/prefill.py` — token-budget batching, chunked prefill | OpenVLA: image+instruction forward each control step, 31 ms (16% of e2e). SmolVLA: prefix fill once per chunk, 9 ms |
| Decode (bandwidth-bound) | `scheduler/decode.py` — decode batch = all runnable requests, every iteration | OpenVLA: 6 autoregressive token steps, 127 ms (**65%** — the bandwidth wall; INT4 cuts it to 101 ms by halving weight traffic). SmolVLA: 10 parallel flow-matching steps sharing one prefix KV cache, 88 ms — no per-token wall, which is why **chunk length is nearly free** |
| Radix prefix cache | `kvcache/radix_cache.py` — token-prefix tree, page-aligned, leaf-LRU | Cross-step reuse is impossible for image tokens (they change every step); only the frozen instruction prefix is cacheable. Our split shows vision is 9–14 ms of prefill while the LM dominates → prefix-cache the instruction, recompute the image. SmolVLA already reuses KV *within* a step across all 10 denoise iterations (`fill_kv_cache` in its source) |
| Chunked prefill + reserved budget | `prefill.py` `PrefillAdder.reserved_size = inflight_tokens` | A new robot's prefill must not eat the KV budget of robots mid-decode — the same reservation logic applies verbatim |
| Continuous batching | `scheduler/decode.py` (39 lines — admission granularity is one iteration) | Multi-robot on one GPU. We measure ×6.6 aggregate throughput at batch 8 with decode time nearly flat (bandwidth-bound) → the physical headroom is real; chunk policies are the ideal fit (fixed 10-step decode, predictable lengths). OpenVLA's upstream code does not support batched generation — it wants one stream per robot |
| Overlap scheduling + decode CUDA graphs | `scheduler.py` dual-stream loop; `engine/graph.py` bs∈{1,2,4,8..256} | SmolVLA spends ~56 ms/step (~⅓ of e2e) in Python glue ("other" phase) — exactly the overhead class these engine techniques eliminate. The first serving win for VLAs is not in the model, it's in the glue |

One-sentence version: **a single-action VLA (OpenVLA) is a textbook autoregressive serving workload — decode-dominated, quantization-friendly, batch-hostile; a chunked VLA (SmolVLA) is prefix-heavy with parallel decode — chunking is its batching, and its fixed decode length makes it easier to schedule than any LLM.**


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
scripts/run_hoist_ab.sh # M4: baseline vs glue-optimized A/B pairs (back-to-back, same session)
benchmarks/profile_glue.py # M4: torch.profiler glue-layer decomposition (run before optimizing)
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
bash scripts/run_formal.sh          # baseline matrix (31 configs)
bash scripts/run_hoist_ab.sh        # M4 glue-optimization A/B pairs
python benchmarks/profile_glue.py   # glue-layer decomposition behind REPORT 4.5
python plots/make_plots.py
```

## Non-goals

Training/fine-tuning, custom CUDA kernels, multi-node serving, simulation success rates (numeric deviation only — future work), cross-GPU comparisons (every table is same-card).

## Known limitations

- OpenVLA's upstream generation code does not support batch > 1; batching is benchmarked on SmolVLA only.
- Shared cluster GPU: clocks are not locked; idle-at-start verified and recorded instead. Two configs showed transient bimodal latency once and were re-measured clean (LOG.md).
- SmolVLA loads mixed-precision as shipped (bf16 towers + fp32 action heads); INT8/INT4 quantization excludes q/k/v/o projections (lerobot's internal dtype glue reads their `.weight.dtype`) — the exclusion set is identical for both precisions.
- bitsandbytes INT8 kernels cast bf16 inputs to fp16; the "INT8" path is not pure int8 arithmetic.
