"""VLA-Inference-Bench: single-GPU VLA inference benchmark.

One entry point for every measurement. Emits one JSON per configuration
into results/; plots/make_plots.py turns JSONs into figures.

Examples:
  python bench.py --model smolvla --precision bf16 --chunk 50 --save-ref
  python bench.py --model smolvla --precision int4 --chunk 50 --deviation-ref results/ref_smolvla_bf16_chunk50.pt
  python bench.py --model openvla --precision int8 --save-ref
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["smolvla", "openvla"], required=True)
    p.add_argument("--precision", choices=["bf16", "int8", "int4"], default="bf16")
    p.add_argument("--chunk", type=int, default=None, help="action chunk length (smolvla only)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=1, help="PHYSICAL gpu index; all formal data must use the same one")
    p.add_argument("--fixture", type=str, default="data/fixtures_pusht.pt")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--save-ref", action="store_true", help="save per-run outputs as the deviation reference")
    p.add_argument("--deviation-ref", type=str, default=None, help="path to a saved reference .pt")
    return p.parse_args()


def gpu_idle_check(phys_gpu: int) -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(phys_gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        foreign = [l for l in out.splitlines() if l.strip()]
        return {"foreign_gpu_processes": foreign, "gpu_idle": len(foreign) == 0}
    except Exception as e:  # noqa: BLE001
        return {"foreign_gpu_processes": None, "gpu_idle": None, "error": str(e)}


def pkg_versions(pkgs) -> dict:
    import importlib.metadata as md

    v = {}
    for p in pkgs:
        try:
            v[p] = md.version(p)
        except Exception:  # noqa: BLE001
            v[p] = None
    return v


def series(vals):
    vals = sorted(vals)
    n = len(vals)
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if n > 1 else 0.0,
        "p50": vals[n // 2],
        "p99": vals[min(n - 1, int(round(0.99 * (n - 1))))],
        "min": vals[0],
        "max": vals[-1],
    }


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    import torch

    from benchmarks.latency import PhaseProfiler
    from benchmarks.memory import allocated_gb, peak_snapshot, reset_peak, reserved_gb
    from models.openvla import OpenVLAAdapter
    from models.smolvla import SmolVLAAdapter

    if args.model == "openvla" and args.batch > 1:
        sys.exit("openvla upstream generation code does not support batch > 1 (see models/openvla.py docstring)")

    adapter = OpenVLAAdapter(args) if args.model == "openvla" else SmolVLAAdapter(args)

    idle = gpu_idle_check(args.gpu)
    versions = pkg_versions(["torch", "transformers", "lerobot", "bitsandbytes", "timm", "tokenizers"])

    t_start = time.strftime("%Y-%m-%dT%H:%M:%S")
    adapter.load()
    torch.cuda.synchronize()
    weights_after_load = allocated_gb()

    fixture = torch.load(args.fixture, weights_only=False)

    # warmup (also settles lazy bnb quantization)
    for i in range(args.warmup):
        adapter.step(fixture, run_idx=i)
    torch.cuda.synchronize()
    weights_after_warmup = allocated_gb()
    reset_peak()

    profiler = PhaseProfiler()
    adapter.attach_hooks(profiler)

    records, phase_records, outputs = [], [], {"actions": [], "tokens": []}
    for i in range(args.runs):
        profiler.reset()
        out = adapter.step(fixture, run_idx=i)
        phases = profiler.finish()
        records.append(out["wall_ms"])
        phase_records.append(phases)
        outputs["actions"].append(out["actions"])
        if out["tokens"] is not None:
            outputs["tokens"].append(out["tokens"])

    adapter.detach_hooks()

    # aggregate
    wall_keys = records[0].keys()
    latency = {k: series([r[k] for r in records]) for k in wall_keys}
    phase_names = sorted({k for p in phase_records for k in p})
    phases = {k: series([p.get(k, 0.0) for p in phase_records]) for k in phase_names}

    result = {
        "meta": {
            "model": adapter.name,
            "precision": args.precision,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_index_physical": args.gpu,
            "gpu_idle_check": idle,
            "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(args.gpu)], capture_output=True, text=True).stdout.strip(),
            "versions": versions,
            "timestamp_start": t_start,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "vision_module_path": getattr(adapter, "vision_module_path", None),
            "param_dtype": str(getattr(adapter, "_param_dtype", torch.bfloat16)),
        },
        "config": {
            "batch": args.batch,
            "chunk": args.chunk if args.model == "smolvla" else None,
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
            "fixture": args.fixture,
        },
        "latency_ms": {"wall": latency, "phases": phases},
        "memory_gb": {
            "weights_after_load": round(weights_after_load, 4),
            "weights_after_warmup": round(weights_after_warmup, 4),
            "reserved_after_warmup": round(reserved_gb(), 4),
            **peak_snapshot(),
        },
    }

    # deviation vs saved reference (or save this run as the reference)
    ref_path = None
    if args.save_ref:
        ref_path = Path(args.out) / f"ref_{args.model}_{args.precision}_chunk{args.chunk or 'na'}.pt"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actions": outputs["actions"], "tokens": outputs["tokens"], "meta": result["meta"], "config": result["config"]}, ref_path)
        result["deviation_ref_saved"] = str(ref_path)
    elif args.deviation_ref:
        ref = torch.load(args.deviation_ref, weights_only=False)
        from benchmarks import deviation as dev

        if args.model == "openvla":
            result["deviation"] = dev.openvla_deviation(
                outputs["tokens"], outputs["actions"], ref["tokens"], ref["actions"]
            )
        else:
            result["deviation"] = dev.smolvla_deviation(outputs["actions"], ref["actions"])

    out_name = f"{args.model}_{args.precision}"
    if args.model == "smolvla" and args.chunk:
        out_name += f"_chunk{args.chunk}"
    if args.batch > 1:
        out_name += f"_batch{args.batch}"
    out_path = Path(args.out) / f"{out_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    # console summary
    print(f"\n=== {adapter.name} | {args.precision} | chunk={args.chunk} | batch={args.batch} ===")
    print(f"gpu_idle_at_start: {idle.get('gpu_idle')}")
    for k in ["e2e", "preprocess", "generate", "postprocess"]:
        if k in latency:
            s = latency[k]
            print(f"wall {k:>12}: {s['mean']:8.2f} ± {s['std']:6.2f} ms  (p99 {s['p99']:.2f})")
    for k, s in phases.items():
        print(f"phase {k:>17}: {s['mean']:8.2f} ± {s['std']:6.2f} ms")
    m = result["memory_gb"]
    print(f"memory: weights {m['weights_after_warmup']:.2f} GB | peak_alloc {m['peak_allocated_gb']:.2f} GB | peak_reserved {m['peak_reserved_gb']:.2f} GB")
    if "deviation" in result:
        print("deviation:", json.dumps(result["deviation"], indent=1))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
