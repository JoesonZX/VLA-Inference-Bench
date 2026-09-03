"""Generate every figure + a markdown summary table from results/*.json.

Rerun-safe: JSONs are the only input. Phase names as produced by the adapters:

  openvla : vision_backbone, vision_projector, prefill_lm, decode
            wall segments: preprocess, generate, postprocess, e2e
  smolvla : vision, prefill_lm, decode_step (sum over num_steps denoise calls)
            wall segments: preprocess, policy, postprocess, e2e
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).parent  # figures live next to this script in plots/

PHASE_COLORS = {
    "vision": "#8ecae6",
    "prefill_lm": "#ffb703",
    "decode": "#fb8500",
    "cpu_pre": "#e0e0e0",
    "cpu_post": "#f5f5f5",
    "other": "#cccccc",
}


def load_all():
    return [json.loads(f.read_text()) for f in sorted(RESULTS.glob("*.json"))]


def phases_of(r):
    """Normalized phase -> mean ms for a run dict."""
    ph = r["latency_ms"]["phases"]
    w = r["latency_ms"]["wall"]
    out = {}
    vision = ph.get("vision", {}).get("mean", 0.0) + ph.get("vision_backbone", {}).get("mean", 0.0) + ph.get("vision_projector", {}).get("mean", 0.0)
    if vision:
        out["vision"] = vision
    if "prefill_lm" in ph:
        out["prefill_lm"] = ph["prefill_lm"]["mean"]
    dec = ph.get("decode", ph.get("decode_step"))
    if dec:
        out["decode"] = dec["mean"]
    known = out.get("vision", 0) + out.get("prefill_lm", 0) + out.get("decode", 0)
    pre = w.get("preprocess", {}).get("mean", 0.0)
    post = w.get("postprocess", {}).get("mean", 0.0)
    out["cpu_pre"] = pre
    out["cpu_post"] = post
    other = max(w["e2e"]["mean"] - known - pre - post, 0.0)
    out["other"] = other
    out["_pre"] = pre
    out["_post"] = post
    out["_e2e"] = w["e2e"]["mean"]
    out["_e2e_std"] = w["e2e"]["std"]
    return out


def stacked_ax(ax, rows, labels, title, ylabel="ms"):
    xs = np.arange(len(rows))
    bottoms = np.zeros(len(rows))
    # stack sums to wall e2e: vision + prefill + decode + cpu pre/post + other
    for ph in ["vision", "prefill_lm", "decode", "cpu_pre", "cpu_post", "other"]:
        vals = np.array([r[ph] for r in rows])
        ax.bar(xs, vals, bottom=bottoms, label=ph, color=PHASE_COLORS[ph], width=0.6)
        bottoms += vals
    e2e = [r["_e2e"] for r in rows]
    ax.plot(xs, e2e, "k_", markersize=14, label="wall e2e")
    ax.set_xticks(xs, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.margins(y=0.12)  # headroom so the wall-e2e markers are never clipped
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)


def fig_openvla_precision(runs):
    rs = [r for r in runs if r["meta"]["model"] == "openvla-7b" and r["config"]["batch"] == 1]
    order = ["bf16", "int8", "int4"]
    rs = sorted(rs, key=lambda r: order.index(r["meta"]["precision"]))
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    stacked_ax(ax, [phases_of(r) for r in rs], [r["meta"]["precision"].upper() for r in rs],
               "OpenVLA-7B: per-phase latency by precision (H100, batch=1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_openvla_precision.png", dpi=160)
    plt.close(fig)


def fig_smolvla_chunk(runs):
    rs = [r for r in runs if r["meta"]["model"] == "smolvla-450m" and r["config"]["batch"] == 1 and r["config"].get("chunk") and r["meta"]["precision"] == "fp32"]
    rs = sorted(rs, key=lambda r: r["config"]["chunk"])
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    stacked_ax(ax, [phases_of(r) for r in rs], [f"chunk={r['config']['chunk']}" for r in rs],
               "SmolVLA (fp32): per-phase latency by action chunk length (H100, batch=1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_smolvla_chunk.png", dpi=160)
    plt.close(fig)


def fig_smolvla_precision(runs):
    rs = [r for r in runs if r["meta"]["model"] == "smolvla-450m" and r["config"]["batch"] == 1 and r["config"].get("chunk") == 50]
    order = ["fp32", "bf16", "int8", "int4"]
    rs = sorted(rs, key=lambda r: order.index(r["meta"]["precision"]))
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    stacked_ax(ax, [phases_of(r) for r in rs], [r["meta"]["precision"].upper() for r in rs],
               "SmolVLA (chunk=50): per-phase latency by precision (H100, batch=1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_smolvla_precision.png", dpi=160)
    plt.close(fig)


def fig_quant_tradeoff(runs):
    rs = [r for r in runs if r["config"]["batch"] == 1 and (r["config"].get("chunk") in (None, 50))]
    if len(rs) < 4:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    styles = {
        "openvla-7b": ("o", "#d62828", "OpenVLA-7B"),
        "smolvla-450m": ("s", "#1d3557", "SmolVLA-450M"),
    }
    i_smol = 0
    for r in rs:
        m = r["meta"]["model"]
        marker, color, pretty = styles.get(m, ("^", "#333333", m))
        x = r["memory_gb"]["weights_after_warmup"]
        y = r["latency_ms"]["wall"]["e2e"]["mean"]
        yerr = r["latency_ms"]["wall"]["e2e"]["std"]
        ax.errorbar(x, y, yerr=yerr, fmt=marker, markersize=8, capsize=3, color=color, label=pretty)
        # stagger the two nearly-coincident smolvla point labels (fp32/bf16)
        if m == "smolvla-450m":
            va = "bottom" if i_smol % 2 == 0 else "top"
            i_smol += 1
        else:
            va = "bottom"
        ax.annotate(f"{r['meta']['precision'].upper()}", (x, y), fontsize=7, ha="center", va=va)
    ax.set_xlabel("weights memory (GB)")
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("What quantization buys (memory) and costs (latency) — H100, batch=1")
    # one legend entry per model (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict()
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_quant_tradeoff.png", dpi=160)
    plt.close(fig)


def fig_chunk_curve(runs):
    rs = [r for r in runs if r["meta"]["model"] == "smolvla-450m" and r["config"].get("chunk") and r["config"]["batch"] == 1]
    if len(rs) < 5:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for prec in ["fp32", "bf16", "int4", "int8"]:
        pts = sorted([(r["config"]["chunk"], r["latency_ms"]["wall"]["e2e"]["mean"], r["latency_ms"]["wall"]["e2e"]["std"])
                      for r in rs if r["meta"]["precision"] == prec])
        if not pts:
            continue
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es, capsize=3, marker="o", label=prec.upper())
        # per-action amortized cost
    ax2 = ax.twinx()
    pts = sorted([(r["config"]["chunk"], r["latency_ms"]["wall"]["e2e"]["mean"] / r["config"]["chunk"])
                  for r in rs if r["meta"]["precision"] == "fp32"])
    xs, ys = zip(*pts)
    ax2.plot(xs, ys, "k--", alpha=0.6, label="fp32 ms/action (right)")
    ax2.set_ylabel("ms per action (amortized)")
    ax.set_xlabel("action chunk length")
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("SmolVLA: chunk length vs step latency and amortized per-action cost")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_chunk_curve.png", dpi=160)
    plt.close(fig)


def fig_deviation(runs):
    rs = [r for r in runs if "deviation" in r]
    if not rs:
        return
    metrics = ["token_mismatch_rate", "action_l2_mean", "chunk_mse"]
    titles = ["OpenVLA action-token mismatch rate", "mean action L2 vs reference", "SmolVLA chunk MSE vs fp32"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, metric, title in zip(axes, metrics, titles):
        pts = [(f"{r['meta']['model'].split('-')[0]}\n{r['meta']['precision'].upper()}"
                + (f"\nchunk={r['config']['chunk']}" if r['meta']['model'].startswith('smolvla') and r['config']['chunk'] not in (None, 50) else ""),
                r["deviation"][metric])
               for r in rs if metric in r.get("deviation", {})]
        if not pts:
            ax.axis("off")
            continue
        labels, vals = zip(*pts)
        ax.bar(labels, vals, color=plt.cm.Set2(np.linspace(0, 1, len(vals))))
        ax.set_ylim(0, max(vals) * 1.18)  # headroom for value labels
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_deviation.png", dpi=160)
    plt.close(fig)


def fig_batch(runs):
    rs = [r for r in runs if r["config"]["batch"] > 1]
    base = [r for r in runs if r["config"]["batch"] == 1 and r["meta"]["model"] == "smolvla-450m"
            and r["meta"]["precision"] == "bf16" and r["config"].get("chunk") == 50]
    if not rs or not base:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = [(1, base[0])] + [(r["config"]["batch"], r) for r in rs]
    rows.sort()
    xs = [b for b, _ in rows]
    ys = [r["latency_ms"]["wall"]["e2e"]["mean"] * b for b, r in rows]
    ax.plot(xs, ys, marker="o", color="#1d3557")
    for b, r in rows:
        ax.annotate(f"{r['latency_ms']['wall']['e2e']['mean']*b:.0f}", (b, r["latency_ms"]["wall"]["e2e"]["mean"] * b),
                    textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("batch (robots sharing one GPU, SmolVLA bf16 chunk=50)")
    ax.set_ylabel("aggregate throughput (actions chunk-steps / s)")
    ax.set_title("Batching on SmolVLA: throughput vs batch size")
    fig.tight_layout()
    fig.savefig(OUT / "fig_batch.png", dpi=160)
    plt.close(fig)


def fig_hoist(runs):
    """M4: baseline vs glue-optimized (hoist) e2e, paired per config."""
    pairs = []
    for prec in ["fp32", "bf16"]:
        for chunk in [1, 10, 50]:
            b = [r for r in runs if r["meta"]["model"] == "smolvla-450m" and r["meta"]["precision"] == prec
                 and r["config"].get("chunk") == chunk and r["config"]["batch"] == 1
                 and r["config"].get("variant", "baseline") == "baseline"]
            h = [r for r in runs if r["meta"]["model"] == "smolvla-450m" and r["meta"]["precision"] == prec
                 and r["config"].get("chunk") == chunk and r["config"]["batch"] == 1
                 and r["config"].get("variant") == "hoist"]
            if b and h:
                pairs.append((prec, chunk, b[0], h[0]))
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(pairs))
    w = 0.38
    for off, idx, label, color in [(-w / 2, 2, "baseline (upstream)", "#999999"), (w / 2, 3, "hoist (glue optimized)", "#2a9d8f")]:
        vals = [p[idx]["latency_ms"]["wall"]["e2e"]["mean"] for p in pairs]
        errs = [p[idx]["latency_ms"]["wall"]["e2e"]["std"] for p in pairs]
        ax.bar(xs + off, vals, width=w, yerr=errs, capsize=3, label=label, color=color)
    for i, (_, _, b, h) in enumerate(pairs):
        saving = (b["latency_ms"]["wall"]["e2e"]["mean"] - h["latency_ms"]["wall"]["e2e"]["mean"]) / b["latency_ms"]["wall"]["e2e"]["mean"] * 100
        top = max(b["latency_ms"]["wall"]["e2e"]["mean"], h["latency_ms"]["wall"]["e2e"]["mean"])
        ax.text(i, top * 1.03, f"−{saving:.0f}%", ha="center", fontsize=9, color="#2a9d8f", fontweight="bold")
    ax.set_xticks(xs, [f"{p[0]}\nchunk={p[1]}" for p in pairs])
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("SmolVLA: glue-layer optimization (bit-identical outputs) — H100, batch=1")
    ax.margins(y=0.15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hoist.png", dpi=160)
    plt.close(fig)


def write_summary_md(runs):
    lines = ["# Results summary (auto-generated from results/*.json — do not edit by hand)", ""]
    lines += ["## Latency & memory (H100, GPU 1, batch=1)", "",
              "| model | precision | chunk | e2e ms (mean±std) | vision | prefill | decode | preprocess | weights GB | peak GB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["meta"]["model"], r["meta"]["precision"], r["config"].get("chunk") or 0)):
        if r["config"]["batch"] != 1:
            continue
        ph = phases_of(r)
        w = r["latency_ms"]["wall"]
        lines.append(
            f"| {r['meta']['model']} | {r['meta']['precision']} | {r['config'].get('chunk') or '-'} "
            f"| {w['e2e']['mean']:.1f}±{w['e2e']['std']:.1f} "
            f"| {ph.get('vision', 0):.1f} | {ph.get('prefill_lm', 0):.1f} | {ph.get('decode', 0):.1f} "
            f"| {ph['_pre']:.1f} "
            f"| {r['memory_gb']['weights_after_warmup']:.2f} | {r['memory_gb']['peak_allocated_gb']:.2f} |"
        )
    dev = [r for r in runs if "deviation" in r]
    if dev:
        lines += ["", "## Deviation vs reference (OpenVLA: BF16 ref; SmolVLA: fp32 as-shipped ref)", "",
                  "| model | precision | chunk | metric(s) |", "|---|---|---|---|"]
        for r in dev:
            d = r["deviation"]
            s = ", ".join(f"{k}={v:.4g}" for k, v in d.items())
            lines.append(f"| {r['meta']['model']} | {r['meta']['precision']} | {r['config'].get('chunk') or '-'} | {s} |")
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("wrote", RESULTS / "summary.md")


def main():
    runs = load_all()
    fig_openvla_precision(runs)
    fig_smolvla_precision(runs)
    fig_smolvla_chunk(runs)
    fig_quant_tradeoff(runs)
    fig_chunk_curve(runs)
    fig_deviation(runs)
    fig_batch(runs)
    fig_hoist(runs)
    write_summary_md(runs)
    print(f"{len(runs)} result JSONs processed")


if __name__ == "__main__":
    main()
