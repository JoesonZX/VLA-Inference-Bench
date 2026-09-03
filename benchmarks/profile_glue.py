"""M4-T1: decompose SmolVLA's glue ("other") phase before optimizing anything.

Two views:
  1. component wall-time: CUDA-event timing wrapped around embed_prefix,
     embed_suffix, make_att_2d_masks (the mask/position glue), plus the existing
     vision/prefill/decode hooks -> a table that sums against the policy wall time.
  2. torch.profiler on one warm step: top operators by self CUDA / self CPU time,
     and total CUDA kernel time vs wall (the launch-gap number that compile/graphs attack).

Usage (server, smolvla env):
  python benchmarks/profile_glue.py --precision fp32 --chunk 50 --runs 10
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    from benchmarks.latency import PhaseProfiler
    from models.smolvla import SmolVLAAdapter
    import lerobot.policies.smolvla.modeling_smolvla as modeling

    class A:  # minimal args stand-in
        model = "smolvla"
        precision = args.precision
        chunk = args.chunk
        batch = 1
        seed = 0
    adapter = SmolVLAAdapter(A())
    adapter.load()
    fixture = torch.load(Path(__file__).parent.parent / "data" / "fixtures_pusht.pt", weights_only=False)

    # --- component timing via CUDA events + instance patches ---
    prof = PhaseProfiler()
    adapter.attach_hooks(prof)  # vision + prefill_lm/decode_step patches

    def wrap_module_method(obj, name, label):
        orig = getattr(obj, name)
        def timed(*a, **k):
            s = torch.cuda.Event(enable_timing=True); s.record()
            out = orig(*a, **k)
            e = torch.cuda.Event(enable_timing=True); e.record()
            prof.add(label, s, e)
            return out
        setattr(obj, name, timed)

    m = adapter.policy.model
    wrap_module_method(m, "embed_prefix", "embed_prefix")
    wrap_module_method(m, "embed_suffix", "embed_suffix")
    orig_masks = modeling.make_att_2d_masks
    mask_calls = [0]
    def timed_masks(*a, **k):
        mask_calls[0] += 1
        s = torch.cuda.Event(enable_timing=True); s.record()
        out = orig_masks(*a, **k)
        e = torch.cuda.Event(enable_timing=True); e.record()
        prof.add("make_att_2d_masks", s, e)
        return out
    modeling.make_att_2d_masks = timed_masks

    # warmup
    for i in range(args.warmup):
        adapter.step(fixture, run_idx=i)
    torch.cuda.synchronize()
    mask_calls[0] = 0

    walls = []
    for i in range(args.runs):
        prof.reset()
        mask_calls[0] = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = adapter.step(fixture, run_idx=i)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        walls.append((t1 - t0) * 1e3)
        sums = prof.finish()

    print(f"\n=== component decomposition ({args.runs} runs, {args.precision}, chunk={args.chunk}) ===")
    print(f"wall e2e: {statistics.fmean(walls):.1f} ± {statistics.stdev(walls):.1f} ms")
    print(f"make_att_2d_masks calls per step: {mask_calls[0] // args.runs}")
    for k in ["vision", "prefill_lm", "decode_step", "embed_prefix", "embed_suffix", "make_att_2d_masks"]:
        if k in sums:
            print(f"  {k:20s} {sums[k]:7.2f} ms (sum over calls)")

    # --- operator-level profile of one step ---
    print("\n=== torch.profiler (1 warm step) ===")
    from torch.profiler import ProfilerActivity, profile

    adapter.step(fixture, run_idx=0)  # ensure warm
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        adapter.step(fixture, run_idx=0)
        torch.cuda.synchronize()

    ev = p.key_averages()
    total_cuda = sum(e.self_device_time_total for e in ev) / 1e3
    total_cpu = sum(e.self_cpu_time_total for e in ev) / 1e3
    print(f"total self CUDA time: {total_cuda:.1f} ms | total self CPU time: {total_cpu:.1f} ms | wall ~{walls[0]:.1f} ms")
    print(f"launch-gap estimate (wall - cuda busy): {walls[0] - total_cuda:.1f} ms")

    print("\n-- top 15 by self CUDA time (ms) --")
    rows = sorted(ev, key=lambda e: e.self_device_time_total, reverse=True)[:15]
    for e in rows:
        if e.self_device_time_total > 0:
            print(f"  {e.key[:60]:60s} {e.self_device_time_total/1e3:7.2f}  x{e.count}")
    print("\n-- top 15 by self CPU time (ms) --")
    rows = sorted(ev, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]
    for e in rows:
        print(f"  {e.key[:60]:60s} {e.self_cpu_time_total/1e3:7.2f}  x{e.count}")

    out_dir = Path(__file__).parent.parent / "results" / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    p.export_chrome_trace(str(out_dir / f"glue_{args.precision}_chunk{args.chunk}.json"))
    print(f"\nchrome trace -> {out_dir}")


if __name__ == "__main__":
    main()
