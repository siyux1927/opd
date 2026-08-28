"""Sequential orchestration of every arm. Resumable: finished arms are skipped.

    python run_all.py --tier smoke   # ~6 min, proves the pipeline end to end
    python run_all.py --tier all     # full P0 + P1 + P2 grid
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

STUDENT_BASE = "Qwen/Qwen3-0.6B-Base"
TEACHER = "Qwen/Qwen3-4B-Instruct-2507"
SFT_CKPT = "checkpoints/sft_init"

P0_GRID = [("reverse_kl", "opd"), ("reverse_kl", "opd_plus"), ("forward_kl", "opd"), ("forward_kl", "opd_plus")]
P1_GRID = [("jsd", "opd"), ("jsd", "opd_plus")]


def build_jobs(tier: str, steps: int, eval_limit: int, sft_examples: int) -> list[tuple[str, list[str]]]:
    common = ["--teacher", TEACHER, "--student", SFT_CKPT, "--steps", str(steps), "--eval-limit", str(eval_limit)]
    jobs: list[tuple[str, list[str]]] = [
        (
            "sft_init",
            [sys.executable, "train_sft.py", "--model", STUDENT_BASE, "--run-name", "sft_init",
             "--num-examples", str(sft_examples), "--out", SFT_CKPT, "--eval-limit", str(eval_limit)],
        )
    ]

    grid = list(P0_GRID)
    if tier in ("p1", "p2", "all"):
        grid += P1_GRID
    if tier == "smoke":
        grid = [("forward_kl", "opd"), ("forward_kl", "opd_plus")]

    for divergence, mode in grid:
        run = f"{divergence}_{mode}"
        jobs.append(
            (run, [sys.executable, "train_pg.py", "--objective", "opd",
                   "--divergence", divergence, "--mode", mode, "--run-name", run] + common)
        )

    if tier in ("p2", "all"):
        jobs.append(
            ("sft_more_data",
             [sys.executable, "train_sft.py", "--model", STUDENT_BASE, "--init-from", SFT_CKPT,
              "--run-name", "sft_more_data", "--num-examples", str(sft_examples * 2),
              "--skip-examples", str(sft_examples), "--out", "checkpoints/sft_more_data",
              "--eval-limit", str(eval_limit)])
        )
        jobs.append(
            ("grpo",
             [sys.executable, "train_pg.py", "--objective", "grpo", "--run-name", "grpo"] + common)
        )
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["smoke", "p0", "p1", "p2", "all"], default="all")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--sft-examples", type=int, default=2000)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.tier == "smoke":
        args.steps, args.eval_limit, args.sft_examples = 2, 40, 200

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.tier, args.steps, args.eval_limit, args.sft_examples)
    for name, cmd in jobs:
        marker = results / f"{name}_final.json"
        if marker.exists() and not args.force:
            print(f"== skip {name} (already finished)")
            continue
        print(f"\n{'=' * 70}\n== {name}\n== {' '.join(cmd)}\n{'=' * 70}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"!! {name} failed with code {proc.returncode}; stopping")
            sys.exit(proc.returncode)
        print(f"== {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    subprocess.run([sys.executable, "make_plots.py", "--results-dir", args.results_dir])


if __name__ == "__main__":
    main()
