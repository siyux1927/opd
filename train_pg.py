"""On-policy distillation (OPD / OPD+) and the outcome-reward RL baseline.

Both objectives share one policy-gradient loop and differ only in how the per-token
advantage is built:

  --objective opd   advantage = -f(u) (OPD) or w_f(u) (OPD+), dense, one value per token
  --objective grpo  advantage = group-normalised answer correctness, one value per
                    sequence broadcast over its tokens

Keeping them in the same loop with the same rollout budget is what makes the
accuracy-per-GPU-minute comparison honest.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from opd.core import (
    MetricLogger,
    align_to_student,
    batched_token_logprobs,
    generate_rollouts,
    gpu_memory_gb,
    load_model,
    load_tokenizer,
    resolve_device,
    set_seed,
    token_logprobs,
)
from opd.data import (
    assert_same_vocab,
    is_correct,
    load_gsm8k,
    student_prompt,
    teacher_prompt,
)
from opd.divergences import compute_advantages
from opd.evaluate import evaluate_model


def build_grpo_advantages(batch, device: str) -> tuple[torch.Tensor, dict]:
    rewards = torch.tensor(
        [float(is_correct(text, gold)) for text, gold in zip(batch.texts, batch.golds)],
        device=device,
    )
    grouped = rewards.view(-1, batch.group_size)
    centred = grouped - grouped.mean(dim=1, keepdim=True)
    normalised = centred / (grouped.std(dim=1, keepdim=True) + 1e-4)
    per_seq = normalised.view(-1, 1)
    adv = per_seq * batch.student_resp_mask
    return adv, {"train_reward": float(rewards.mean()), "adv_absmax": float(adv.abs().max())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", choices=["opd", "grpo"], default="opd")
    p.add_argument("--divergence", choices=["reverse_kl", "forward_kl", "jsd"], default="reverse_kl")
    p.add_argument("--mode", choices=["opd", "opd_plus"], default="opd")
    p.add_argument("--student", default="checkpoints/sft_init")
    p.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--run-name", default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--groups-per-batch", type=int, default=16)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=288)
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--log-ratio-clip", type=float, default=6.0)
    p.add_argument("--prompt-offset", type=int, default=6000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--eval-k", type=int, default=4)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.run_name is None:
        args.run_name = "grpo" if args.objective == "grpo" else f"{args.divergence}_{args.mode}"

    set_seed(args.seed)
    device = resolve_device()
    results_dir = Path(args.results_dir)
    logger = MetricLogger(results_dir / "metrics.jsonl", args.run_name)

    student_tok = load_tokenizer(args.student)
    student = load_model(args.student, trainable=True, device=device)

    teacher = teacher_tok = None
    if args.objective == "opd":
        teacher_tok = load_tokenizer(args.teacher)
        assert_same_vocab(student_tok, teacher_tok)
        teacher = load_model(args.teacher, trainable=False, device=device)

    prompts_pool = load_gsm8k("train", limit=args.prompt_offset + args.steps * args.groups_per_batch * 2, seed=args.seed)
    prompts_pool = prompts_pool[args.prompt_offset :]
    if not prompts_pool:
        raise ValueError("prompt pool is empty; lower --prompt-offset")
    test_examples = load_gsm8k("test", limit=args.eval_limit, seed=123)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    rng = random.Random(args.seed)
    history: list[dict] = []
    ratio_samples: dict[str, np.ndarray] = {}
    best = {"avg@%d" % args.eval_k: -1.0}

    def run_eval(step: int) -> dict:
        metrics = evaluate_model(student, student_tok, test_examples, k=args.eval_k, device=device)
        metrics["minutes"] = logger.elapsed_minutes
        logger.log(step, phase="eval", **metrics)
        history.append({"step": step, **metrics})
        key = f"avg@{args.eval_k}"
        if metrics[key] > best.get(key, -1.0):
            best.update(metrics)
            best["step"] = step
        return metrics

    run_eval(0)

    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, step / max(1, args.warmup_steps))

        chosen = rng.sample(prompts_pool, min(args.groups_per_batch, len(prompts_pool)))
        batch = generate_rollouts(
            student,
            student_tok,
            teacher_tok or student_tok,
            [e.question for e in chosen],
            [e.answer for e in chosen],
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            gen_batch_size=args.gen_batch_size,
            device=device,
            student_prompt_fn=student_prompt,
            teacher_prompt_fn=teacher_prompt if teacher_tok else (lambda tok, q: student_prompt(q)),
        )

        student_logp_old = batched_token_logprobs(
            student, batch.student_ids, batch.student_mask, micro_batch_size=args.micro_batch_size
        )

        extra: dict = {}
        if args.objective == "opd":
            teacher_logp = batched_token_logprobs(
                teacher, batch.teacher_ids, batch.teacher_mask, micro_batch_size=args.micro_batch_size
            )
            teacher_aligned = align_to_student(
                teacher_logp, batch.teacher_resp_mask, batch.student_resp_mask
            )
            advantages, stats, log_u = compute_advantages(
                student_logp_old,
                teacher_aligned,
                batch.student_resp_mask,
                divergence=args.divergence,
                mode=args.mode,
                log_ratio_clip=args.log_ratio_clip,
            )
            extra = dict(
                teacher_kl=stats.reverse_kl,
                mean_log_u=stats.mean_log_u,
                clip_frac=stats.clip_frac,
                adv_mean=stats.adv_mean,
                adv_std=stats.adv_std,
                adv_absmax=stats.adv_absmax,
                train_reward=float(
                    np.mean([is_correct(t, g) for t, g in zip(batch.texts, batch.golds)])
                ),
            )
            if step in (1, max(1, args.steps // 2), args.steps):
                ratio_samples[f"step{step}"] = (
                    log_u[batch.student_resp_mask.bool()].float().cpu().numpy()
                )
        else:
            advantages, extra = build_grpo_advantages(batch, device)

        total_tokens = batch.student_resp_mask.sum().clamp(min=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for i in range(0, batch.student_ids.size(0), args.micro_batch_size):
            sl = slice(i, i + args.micro_batch_size)
            logp_new = token_logprobs(student, batch.student_ids[sl], batch.student_mask[sl])
            mask = batch.student_resp_mask[sl]
            ratio = torch.exp(logp_new - student_logp_old[sl].detach())
            loss = -(ratio * advantages[sl].detach() * mask).sum() / total_tokens
            loss.backward()
            loss_value += float(loss)

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        logger.log(
            step,
            phase="train",
            loss=loss_value,
            grad_norm=float(grad_norm),
            resp_len=float(np.mean(batch.response_lengths)),
            mem_gb=gpu_memory_gb(),
            **extra,
        )

        if args.eval_every and step % args.eval_every == 0 and step != args.steps:
            run_eval(step)

    final = run_eval(args.steps)

    summary = {
        "run": args.run_name,
        "arm": args.objective,
        "divergence": args.divergence if args.objective == "opd" else None,
        "mode": args.mode if args.objective == "opd" else None,
        "steps": args.steps,
        "rollouts": args.steps * args.groups_per_batch * args.group_size,
        "train_minutes": logger.elapsed_minutes,
        "peak_mem_gb": gpu_memory_gb(),
        "final": final,
        "best": best,
        "history": history,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{args.run_name}_final.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if ratio_samples:
        np.savez_compressed(results_dir / f"{args.run_name}_log_u.npz", **ratio_samples)
    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        student.config.use_cache = True
        student.save_pretrained(out)
        student_tok.save_pretrained(out)

    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
