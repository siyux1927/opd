"""Figures and the summary table for the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from opd.divergences import DIVERGENCES, advantage_from_log_ratio

ARM_ORDER = [
    "sft_init",
    "sft_more_data",
    "grpo",
    "reverse_kl_opd",
    "reverse_kl_opd_plus",
    "forward_kl_opd",
    "forward_kl_opd_plus",
    "jsd_opd",
    "jsd_opd_plus",
]

PRETTY = {
    "sft_init": "SFT init (student)",
    "sft_more_data": "+2x off-policy SFT data",
    "grpo": "GRPO (outcome reward RL)",
    "reverse_kl_opd": "Reverse KL / OPD",
    "reverse_kl_opd_plus": "Reverse KL / OPD+",
    "forward_kl_opd": "Forward KL / OPD",
    "forward_kl_opd_plus": "Forward KL / OPD+",
    "jsd_opd": "JSD / OPD",
    "jsd_opd_plus": "JSD / OPD+",
}

COLORS = {
    "reverse_kl": "#2563eb",
    "forward_kl": "#dc2626",
    "jsd": "#16a34a",
}


def plot_weight_functions(out: Path) -> None:
    """Why forward KL collapses under stop-gradient OPD, in one picture."""
    log_u = torch.linspace(-6, 6, 601)
    u = torch.exp(log_u)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, div in zip(axes, DIVERGENCES):
        opd = advantage_from_log_ratio(log_u, div, "opd").numpy()
        plus = advantage_from_log_ratio(log_u, div, "opd_plus").numpy()
        ax.plot(u.numpy(), opd, label=r"OPD:  $-f(u)$", color=COLORS[div], ls="--", lw=2)
        ax.plot(u.numpy(), plus, label=r"OPD+: $w_f(u)$", color=COLORS[div], lw=2.2)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(1, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$u = q_{teacher}/p_{student}$")
        ax.set_title(div.replace("_", " ").upper())
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("per-token advantage")
    fig.suptitle(
        "Advantage as a function of the teacher/student density ratio\n"
        r"$u \gg 1$ = teacher wants this token far more than the student did",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "fig1_weight_functions.png", dpi=160)
    plt.close(fig)


def plot_ratio_histograms(results: Path, out: Path) -> None:
    files = sorted(results.glob("*_log_u.npz"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for f in files:
        run = f.name.replace("_log_u.npz", "")
        data = np.load(f)
        key = sorted(data.files)[0]
        ax.hist(data[key], bins=80, alpha=0.45, density=True, label=PRETTY.get(run, run))
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_xlabel(r"$\log u$ on sampled response tokens (step 1)")
    ax.set_ylabel("density")
    ax.set_title("Where the sampled tokens actually land on the weight curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig2_log_ratio_distribution.png", dpi=160)
    plt.close(fig)


def load_summaries(results: Path) -> dict[str, dict]:
    summaries = {}
    for f in results.glob("*_final.json"):
        payload = json.loads(f.read_text(encoding="utf-8"))
        summaries[payload.get("run", f.stem)] = payload
    return summaries


def _metric(payload: dict, key: str) -> float | None:
    for container in (payload.get("best"), payload.get("final"), payload):
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def plot_training_curves(summaries: dict[str, dict], out: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for run in ARM_ORDER:
        payload = summaries.get(run)
        if not payload or not payload.get("history"):
            continue
        if payload.get("arm") == "sft":
            continue  # no on-policy steps to plot against
        steps = [h["step"] for h in payload["history"]]
        vals = [h[metric] for h in payload["history"] if metric in h]
        if len(vals) != len(steps):
            continue
        div = payload.get("divergence")
        style = "-" if payload.get("mode") == "opd_plus" else "--"
        color = COLORS.get(div, "#6b7280")
        ax.plot(steps, vals, style, color=color, marker="o", ms=3.5, label=PRETTY.get(run, run))
    ax.set_xlabel("on-policy step")
    ax.set_ylabel(f"GSM8K {metric}")
    ax.set_title("Dashed = stop-gradient OPD, solid = gradient-faithful OPD+")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig3_training_curves.png", dpi=160)
    plt.close(fig)


def plot_cost_curve(summaries: dict[str, dict], out: Path, metric: str) -> None:
    """Compute is counted from the shared SFT-init checkpoint, which every arm starts from."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    init = summaries.get("sft_init")
    if init:
        baseline = _metric(init, metric)
        if baseline is not None:
            ax.axhline(baseline, color="0.35", ls=":", lw=1.6)
            ax.text(
                0.01, baseline, " SFT-init student (shared start)",
                va="bottom", ha="left", fontsize=8, color="0.35",
                transform=ax.get_yaxis_transform(),
            )

    for run in ARM_ORDER:
        if run == "sft_init":
            continue
        payload = summaries.get(run)
        if not payload or not payload.get("history"):
            continue
        pts = [(h.get("minutes", 0.0), h[metric]) for h in payload["history"] if metric in h]
        if not pts:
            continue
        xs, ys = zip(*pts)
        highlight = run in ("grpo", "reverse_kl_opd_plus", "sft_more_data")
        ax.plot(
            xs, ys,
            marker="o" if len(xs) > 1 else "D", ms=4 if len(xs) > 1 else 8,
            lw=2.4 if highlight else 1.2,
            alpha=1.0 if highlight else 0.55,
            label=PRETTY.get(run, run),
        )
    ax.set_xlabel("A100 minutes of post-training (from the shared SFT-init checkpoint)")
    ax.set_ylabel(f"GSM8K {metric}")
    ax.set_title("Accuracy per GPU-minute: dense teacher signal vs sparse outcome reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig4_cost_efficiency.png", dpi=160)
    plt.close(fig)


def write_table(summaries: dict[str, dict], out: Path, metric: str) -> None:
    lines = [
        f"| Arm | Divergence | Advantage | {metric} (best) | greedy | A100 min | rollouts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in ARM_ORDER:
        payload = summaries.get(run)
        if not payload:
            continue
        acc = _metric(payload, metric)
        greedy = _metric(payload, "greedy_acc")
        lines.append(
            "| {name} | {div} | {mode} | {acc} | {greedy} | {mins} | {rollouts} |".format(
                name=PRETTY.get(run, run),
                div=payload.get("divergence") or "-",
                mode=payload.get("mode") or "-",
                acc=f"{acc:.3f}" if acc is not None else "-",
                greedy=f"{greedy:.3f}" if greedy is not None else "-",
                mins=f"{payload.get('train_minutes', 0.0):.1f}",
                rollouts=payload.get("rollouts", "-"),
            )
        )
    (out / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--metric", default="avg@4")
    args = p.parse_args()

    results = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_weight_functions(out)
    plot_ratio_histograms(results, out)
    summaries = load_summaries(results)
    if summaries:
        plot_training_curves(summaries, out, args.metric)
        plot_cost_curve(summaries, out, args.metric)
        write_table(summaries, out, args.metric)
    print(f"figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
