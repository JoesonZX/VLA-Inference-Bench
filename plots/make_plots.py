"""Generate every figure from results/*.json. Rerun-safe: JSONs are the only input."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent.parent / "results"
OUT = Path(__file__).parent

PHASE_ORDER = ["vision_backbone", "vision_projector", "vision", "prefill_lm", "decode", "decode_step"]


def load_all():
    runs = []
    for f in sorted(RESULTS.glob("*.json")):
        runs.append(json.loads(f.read_text()))
    return runs


def key_of(r):
    return (
        r["meta"]["model"],
        r["meta"]["precision"],
        r["config"].get("chunk"),
        r["config"]["batch"],
    )


def phase_stack(r):
    """(ordered phase names, means) for a single run."""
    ph = r["latency_ms"]["phases"]
    names = [n for n in PHASE_ORDER if n in ph]
    # merge aliases
    if "vision" in names and "vision_backbone" in names:
        names.remove("vision")
    means = [ph[n]["mean"] for n in names]
    return names, means


def fig_phase_breakdown(runs, model):
    rs = [r for r in runs if r["meta"]["model"].startswith(model) and r["config"]["batch"] == 1]
    if model == "openvla":
        rs = [r for r in rs if r["config"].get("chunk") is None]
        label = lambda r: r["meta"]["precision"].upper()
    else:
        rs = [r for r in rs if r["meta"]["precision"] == "bf16" and r["config"].get("chunk")]
        label = lambda r: f"chunk={r['config']['chunk']}"
    if not rs:
        return False
    rs.sort(key=label)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.arange(len(rs))
    names, _ = phase_stack(rs[0])
    bottoms = np.zeros(len(rs))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(names), 3)))
    for i, n in enumerate(names):
        vals = np.array([r["latency_ms"]["phases"][n]["mean"] for r in rs])
        errs = np.array([r["latency_ms"]["phases"][n]["std"] for r in rs])
        ax.bar(xs, vals, bottom=bottoms, yerr=errs, capsize=3, label=n, color=colors[i])
        bottoms += vals
    e2e = [r["latency_ms"]["wall"]["e2e"]["mean"] for r in rs]
    ax.plot(xs, e2e, "k_", markersize=14, label="wall e2e")
    ax.set_xticks(xs, [label(r) for r in rs])
    ax.set_ylabel("ms")
    ax.set_title(f"{model}: per-phase latency breakdown (H100, batch=1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_phase_breakdown_{model}.png", dpi=160)
    plt.close(fig)
    return True


def fig_quant_tradeoff(runs):
    rs = [r for r in runs if r["config"]["batch"] == 1 and (r["config"].get("chunk") in (None, 50))]
    if len(rs) < 2:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    marks = {"openvla-7b": "o", "smolvla-450m": "s"}
    for r in rs:
        m = r["meta"]["model"]
        x = r["memory_gb"]["peak_allocated_gb"]
        y = r["latency_ms"]["wall"]["e2e"]["mean"]
        ax.scatter(x, y, marker=marks.get(m, "^"), s=70)
        ax.annotate(f"{m.split('-')[0]}\n{r['meta']['precision'].upper()}", (x, y), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("peak allocated memory (GB)")
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("memory vs latency across precisions (H100)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_quant_tradeoff.png", dpi=160)
    plt.close(fig)
    return True


def fig_chunk_curve(runs):
    rs = [r for r in runs if r["meta"]["model"].startswith("smolvla") and r["config"].get("chunk") and r["config"]["batch"] == 1]
    if len(rs) < 3:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for prec in ["bf16", "int8", "int4"]:
        pts = sorted(
            [(r["config"]["chunk"], r["latency_ms"]["wall"]["e2e"]["mean"], r["latency_ms"]["wall"]["e2e"]["std"]) for r in rs if r["meta"]["precision"] == prec]
        )
        if not pts:
            continue
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es, capsize=3, marker="o", label=prec.upper())
    ax.set_xlabel("action chunk length")
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("SmolVLA: chunk length vs latency (H100, batch=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_chunk_curve.png", dpi=160)
    plt.close(fig)
    return True


def fig_deviation(runs):
    rs = [r for r in runs if "deviation" in r]
    if not rs:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric, title in [
        (axes[0], "token_mismatch_rate", "OpenVLA action-token mismatch rate"),
        (axes[1], "action_l2_mean", "mean action L2 vs BF16 reference"),
    ]:
        pts = [(f"{r['meta']['model'].split('-')[0]} {r['meta']['precision'].upper()}", r["deviation"][metric]) for r in rs if metric in r.get("deviation", {})]
        if not pts:
            ax.axis("off")
            continue
        labels, vals = zip(*pts)
        ax.bar(labels, vals, color=plt.cm.Set2(np.linspace(0, 1, len(vals))))
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_deviation.png", dpi=160)
    plt.close(fig)
    return True


def main():
    runs = load_all()
    made = {
        "openvla_phases": fig_phase_breakdown(runs, "openvla"),
        "smolvla_phases": fig_phase_breakdown(runs, "smolvla"),
        "quant_tradeoff": fig_quant_tradeoff(runs),
        "chunk_curve": fig_chunk_curve(runs),
        "deviation": fig_deviation(runs),
    }
    for k, v in made.items():
        print(f"{'OK ' if v else '-- '} {k}")


if __name__ == "__main__":
    main()
