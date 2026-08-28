"""GSM8K evaluation shared by every training arm."""

from __future__ import annotations

import torch

from .data import Example, is_correct, student_prompt


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    examples: list[Example],
    *,
    k: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    greedy: bool = True,
    max_new_tokens: int = 320,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    """avg@k accuracy plus a greedy pass, matching the OPD+ paper's avg@n protocol."""
    was_training = model.training
    cache_flag = model.config.use_cache
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    prompts = [student_prompt(e.question) for e in examples]
    golds = [e.answer for e in examples]

    results = {}
    if greedy:
        gens = _generate(
            model, tokenizer, prompts, 1, max_new_tokens, batch_size, device,
            do_sample=False,
        )
        results["greedy_acc"] = sum(
            is_correct(g[0], gold) for g, gold in zip(gens, golds)
        ) / len(golds)
        results["greedy_len"] = sum(len(g[0]) for g in gens) / len(gens)

    if k > 0:
        gens = _generate(
            model, tokenizer, prompts, k, max_new_tokens, batch_size, device,
            do_sample=True, temperature=temperature, top_p=top_p,
        )
        per_example = [
            sum(is_correct(c, gold) for c in cands) / len(cands)
            for cands, gold in zip(gens, golds)
        ]
        results[f"avg@{k}"] = sum(per_example) / len(per_example)
        results[f"pass@{k}"] = sum(
            any(is_correct(c, gold) for c in cands) for cands, gold in zip(gens, golds)
        ) / len(golds)

    tokenizer.padding_side = "right"
    model.config.use_cache = cache_flag
    if was_training:
        model.gradient_checkpointing_enable()
        model.train()
    return results


def _generate(
    model, tokenizer, prompts, num_return, max_new_tokens, batch_size, device, **gen_kwargs
) -> list[list[str]]:
    out: list[list[str]] = []
    eff_batch = max(1, batch_size // max(1, num_return))
    for i in range(0, len(prompts), eff_batch):
        chunk = prompts[i : i + eff_batch]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        seqs = model.generate(
            **enc,
            num_return_sequences=num_return,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )
        new_tokens = seqs[:, enc["input_ids"].size(1) :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for j in range(len(chunk)):
            out.append(decoded[j * num_return : (j + 1) * num_return])
    return out
