"""Supervised fine-tuning: student initialisation and the off-policy (continued-SFT) arm.

Stage 0 turns Qwen3-0.6B-Base into a model that answers in the target format at all, so
that on-policy rollouts are meaningful. Running it again with --skip-examples on fresh
data gives the "just add more off-policy data" baseline that OPD is measured against.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from opd.core import MetricLogger, gpu_memory_gb, load_model, load_tokenizer, resolve_device, set_seed
from opd.data import Example, PromptSpec, assert_same_vocab, load_gsm8k
from opd.evaluate import evaluate_model


class SFTDataset(Dataset):
    def __init__(self, tokenizer, examples: list[Example], max_length: int, spec: PromptSpec):
        self.rows = []
        for ex in examples:
            prompt_ids = tokenizer(spec.student_prompt(ex.question), add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(
                spec.target(ex.reference, ex.answer), add_special_tokens=False
            )["input_ids"] + [spec.stop_id]
            ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            self.rows.append((ids, labels))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad_id: int):
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attention[i, : len(ids)] = 1
    return input_ids, attention, labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--init-from", default=None, help="resume from a previous checkpoint dir")
    p.add_argument("--run-name", default="sft_init")
    p.add_argument("--num-examples", type=int, default=2000)
    p.add_argument("--skip-examples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--eval-k", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--prompt-style", choices=["chat", "plain", "split"], default="chat")
    p.add_argument("--template-model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="tokenizer that owns the chat template; must share the student's vocab")
    p.add_argument("--out", default="checkpoints/sft_init")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device()
    results_dir = Path(args.results_dir)
    logger = MetricLogger(results_dir / "metrics.jsonl", args.run_name)

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.init_from or args.model, trainable=True, device=device)

    if args.prompt_style == "plain":
        spec = PromptSpec.build("plain", tokenizer, tokenizer)
    else:
        template_tok = load_tokenizer(args.template_model)
        assert_same_vocab(tokenizer, template_tok)
        spec = PromptSpec.build(args.prompt_style, template_tok, tokenizer)

    train_pool = load_gsm8k("train", limit=args.skip_examples + args.num_examples, seed=args.seed)
    train_examples = train_pool[args.skip_examples :]
    test_examples = load_gsm8k("test", limit=args.eval_limit, seed=123)

    dataset = SFTDataset(tokenizer, train_examples, args.max_length, spec)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total_updates = max(1, math.ceil(len(loader) * args.epochs / args.grad_accum))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_updates, pct_start=0.05, anneal_strategy="cos"
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for micro, (input_ids, attention, labels) in enumerate(loader):
            input_ids, attention, labels = input_ids.to(device), attention.to(device), labels.to(device)
            loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            (loss / args.grad_accum).backward()
            if (micro + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    logger.log(step, loss=float(loss.detach()), lr=scheduler.get_last_lr()[0])

    metrics = evaluate_model(
        model, tokenizer, test_examples, spec, k=args.eval_k, device=device,
        batch_size=args.eval_batch_size,
    )
    metrics["train_minutes"] = logger.elapsed_minutes
    metrics["peak_mem_gb"] = gpu_memory_gb()
    logger.log(step, **metrics)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    summary = {
        "run": args.run_name,
        "arm": "sft",
        "divergence": None,
        "mode": None,
        "rollouts": 0,
        "train_minutes": metrics["train_minutes"],
        "final": metrics,
        "best": metrics,
        # single point on the cost curve: SFT spends compute without any rollouts
        "history": [{"step": step, "minutes": metrics["train_minutes"], **metrics}],
    }
    (results_dir / f"{args.run_name}_final.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"saved {args.run_name} -> {out}: {metrics}")


if __name__ == "__main__":
    main()
