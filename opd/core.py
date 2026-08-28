"""Model loading, memory-bounded log-probability collection, and on-policy rollouts."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.bfloat16

_TF_VERSION = tuple(int(p) for p in transformers.__version__.split(".")[:2] if p.isdigit())
_DTYPE_KWARG = "dtype" if _TF_VERSION >= (4, 56) else "torch_dtype"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name_or_path: str, *, trainable: bool, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        **{_DTYPE_KWARG: DTYPE},
    )
    model.to(device)
    if trainable:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def _hidden_states(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Backbone forward without materialising the vocabulary projection."""
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise AttributeError(f"{type(model).__name__} has no .model backbone")
    out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state


def token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """log p(input_ids[:, t] | input_ids[:, :t]) for t = 1..L-1, shape [B, L-1].

    The lm_head is applied in position chunks so that the [B, L, 151936] logit tensor is
    never materialised in full; that tensor alone would be several GB for a 4B teacher.
    """
    hidden = _hidden_states(model, input_ids, attention_mask)[:, :-1, :]
    targets = input_ids[:, 1:]

    pieces = []
    for start in range(0, hidden.size(1), chunk_size):
        h = hidden[:, start : start + chunk_size, :]
        t = targets[:, start : start + chunk_size]
        logits = model.lm_head(h).float()
        gathered = logits.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        pieces.append(gathered - torch.logsumexp(logits, dim=-1))
        del logits
    return torch.cat(pieces, dim=1)


@torch.no_grad()
def batched_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    micro_batch_size: int = 4,
    chunk_size: int = 256,
) -> torch.Tensor:
    outs = []
    for i in range(0, input_ids.size(0), micro_batch_size):
        outs.append(
            token_logprobs(
                model,
                input_ids[i : i + micro_batch_size],
                attention_mask[i : i + micro_batch_size],
                chunk_size=chunk_size,
            )
        )
    return torch.cat(outs, dim=0)


@dataclass
class RolloutBatch:
    """One on-policy batch. Response ids are shared between student and teacher views."""

    student_ids: torch.Tensor  # [N, Ls] right-padded
    student_mask: torch.Tensor  # [N, Ls] attention mask
    student_resp_mask: torch.Tensor  # [N, Ls-1] 1 on response tokens (aligned to logprobs)
    teacher_ids: torch.Tensor  # [N, Lt]
    teacher_mask: torch.Tensor
    teacher_resp_mask: torch.Tensor
    texts: list[str]
    golds: list[str]
    group_size: int
    response_lengths: list[int]


def _pad_stack(seqs: list[list[int]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def _resp_mask(prompt_lens: list[int], resp_lens: list[int], width: int, device: str) -> torch.Tensor:
    """Mask over logprob positions, which are offset by one from token positions."""
    mask = torch.zeros((len(prompt_lens), width - 1), dtype=torch.float)
    for i, (p, r) in enumerate(zip(prompt_lens, resp_lens)):
        mask[i, p - 1 : p - 1 + r] = 1.0
    return mask.to(device)


@torch.no_grad()
def generate_rollouts(
    student,
    student_tok,
    teacher_tok,
    questions: list[str],
    golds: list[str],
    *,
    group_size: int,
    max_new_tokens: int,
    stop_id: int,
    gen_batch_size: int = 64,
    device: str = "cuda",
    student_prompt_fn,
    teacher_prompt_fn,
) -> RolloutBatch:
    """Sample ``group_size`` completions per question at temperature 1.0.

    Temperature must be exactly 1.0 with no top-k/top-p truncation: any truncation makes
    the sampling distribution differ from p_theta, which silently breaks the on-policy
    assumption behind the policy-gradient estimator.
    """
    was_training = student.training
    cache_flag = student.config.use_cache
    # eval() alone already bypasses checkpointing, but disabling it explicitly keeps
    # transformers from silently forcing use_cache=False and making decoding ~10x slower.
    student.gradient_checkpointing_disable()
    student.eval()
    student.config.use_cache = True

    prompts = [student_prompt_fn(q) for q in questions]
    expanded_q, expanded_gold, expanded_prompt = [], [], []
    for q, g, p in zip(questions, golds, prompts):
        expanded_q += [q] * group_size
        expanded_gold += [g] * group_size
        expanded_prompt += [p] * group_size

    eos_id = stop_id
    pad_id = student_tok.pad_token_id
    responses: list[list[int]] = []

    student_tok.padding_side = "left"
    for i in range(0, len(expanded_prompt), gen_batch_size):
        chunk = expanded_prompt[i : i + gen_batch_size]
        # add_special_tokens=False everywhere, so that the prompt length measured below
        # matches the prompt length the model actually generated from.
        enc = student_tok(
            chunk, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        out = student.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=stop_id,
        )
        new_tokens = out[:, enc["input_ids"].size(1) :]
        for row in new_tokens.tolist():
            trimmed = []
            for tok_id in row:
                if tok_id == pad_id and eos_id != pad_id:
                    break
                trimmed.append(tok_id)
                if tok_id == eos_id:
                    break
            if not trimmed:
                trimmed = [eos_id]
            responses.append(trimmed)
    student_tok.padding_side = "right"

    student_prompt_ids = [student_tok(p, add_special_tokens=False)["input_ids"] for p in expanded_prompt]
    teacher_prompt_ids = [
        teacher_tok(teacher_prompt_fn(teacher_tok, q), add_special_tokens=False)["input_ids"]
        for q in expanded_q
    ]

    s_seqs = [p + r for p, r in zip(student_prompt_ids, responses)]
    t_seqs = [p + r for p, r in zip(teacher_prompt_ids, responses)]
    resp_lens = [len(r) for r in responses]

    s_ids, s_mask = _pad_stack(s_seqs, pad_id, device)
    t_ids, t_mask = _pad_stack(t_seqs, pad_id, device)

    batch = RolloutBatch(
        student_ids=s_ids,
        student_mask=s_mask,
        student_resp_mask=_resp_mask([len(p) for p in student_prompt_ids], resp_lens, s_ids.size(1), device),
        teacher_ids=t_ids,
        teacher_mask=t_mask,
        teacher_resp_mask=_resp_mask([len(p) for p in teacher_prompt_ids], resp_lens, t_ids.size(1), device),
        texts=[student_tok.decode(r, skip_special_tokens=True) for r in responses],
        golds=expanded_gold,
        group_size=group_size,
        response_lengths=resp_lens,
    )

    student.config.use_cache = cache_flag
    if was_training:
        student.gradient_checkpointing_enable()
        student.train()
    return batch


def gather_response_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compact [N, L-1] per-position values down to the ragged response positions."""
    return values[mask.bool()]


def align_to_student(
    teacher_values: torch.Tensor,
    teacher_mask: torch.Tensor,
    student_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter teacher response logprobs onto the student's [N, Ls-1] layout.

    Student and teacher prompts have different lengths, so the same response token sits
    at different absolute positions in the two views. Getting this off by one is the
    classic silent bug in OPD implementations, so it is done once, here.
    """
    out = torch.zeros_like(student_mask)
    flat = teacher_values[teacher_mask.bool()]
    out[student_mask.bool()] = flat.to(out.dtype)
    return out


@dataclass
class MetricLogger:
    path: Path
    run_name: str
    records: list[dict] = field(default_factory=list)
    _t0: float = field(default_factory=time.time)

    def log(self, step: int, **kwargs) -> None:
        record = {"run": self.run_name, "step": step, "elapsed_s": time.time() - self._t0}
        record.update(kwargs)
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        summary = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in kwargs.items()
        )
        print(f"[{self.run_name}] step {step} t={record['elapsed_s']:.0f}s {summary}", flush=True)

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self._t0) / 60.0


def resolve_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("This project expects a CUDA GPU (Colab A100).")
    return "cuda"


def gpu_memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def env_report() -> dict:
    return {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
