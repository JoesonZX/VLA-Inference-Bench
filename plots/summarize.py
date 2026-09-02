"""Print a compact summary of every result JSON (works on py3.10+)."""
import glob
import json
import sys


def fmt(r):
    w = r["latency_ms"]["wall"]
    ph = r["latency_ms"]["phases"]
    m = r["memory_gb"]
    d = r.get("deviation", {})
    vis = ph.get("vision", ph.get("vision_backbone", {})).get("mean", 0)
    vis += ph.get("vision_projector", {}).get("mean", 0)
    pre = ph.get("prefill_lm", {}).get("mean", 0)
    dec = ph.get("decode_step", ph.get("decode", {})).get("mean", 0)
    dev = "-"
    if d:
        k = "token_mismatch_rate" if "token_mismatch_rate" in d else "chunk_mse"
        dev = "%.4g" % d[k]
    return "e2e %7.1f±%5.1f | vis %5.1f pre %5.1f dec %6.1f | w %5.2fGB peak %5.2f | dev %s" % (
        w["e2e"]["mean"], w["e2e"]["std"], vis, pre, dec,
        m["weights_after_warmup"], m["peak_allocated_gb"], dev)


for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    name = f.replace("\\", "/").split("/")[-1].replace(".json", "")
    idle = r["meta"]["gpu_idle_check"].get("gpu_idle")
    print("%-30s idle=%-5s %s" % (name, idle, fmt(r)))
