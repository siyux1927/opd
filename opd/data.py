"""GSM8K loading, prompt rendering and answer checking.

Prompt style is a first-class knob because it materially changes the distillation signal.

The default, ``chat``, gives teacher and student the *same* token sequence, which is what
the Tinker reference implementation does (it feeds the student's rollout sequence
straight into ``teacher.compute_logprobs``). The student is a base model, so it simply
learns the ChatML format during SFT; both models share the Qwen3 vocabulary, so the
rendered ids are identical.

``split`` (teacher on its chat template, student on a plain completion prompt) looks
appealing because the instruct teacher stays in its natural context, but it injects
format-only noise into the ratio: the student terminates with ``<|endoftext|>`` while the
teacher, in a ChatML context, puts nearly all its terminal mass on ``<|im_end|>``. Under
reverse KL that is a large negative advantage on the stop token of *every* rollout, i.e.
a systematic "never stop" push that has nothing to do with reasoning quality. It is kept
only as an ablation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

ANSWER_PREFIX = "The answer is"

INSTRUCTION = (
    "Solve the following grade school math problem. Reason step by step, keep the "
    "reasoning short, and finish with a single line of the form "
    f"'{ANSWER_PREFIX} <number>.'"
)

PROMPT_STYLES = ("chat", "plain", "split")

_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class Example:
    question: str
    reference: str
    answer: str


def plain_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def chat_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {question.strip()}"}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Qwen3-*-Instruct-2507 is non-thinking only and rejects enable_thinking.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass
class PromptSpec:
    """Bundles prompt rendering, the SFT target format and the stop token.

    ``render_tok`` is the tokenizer that owns the chat template (the teacher's). A base
    student has no template of its own, but since the vocabulary is shared the rendered
    string tokenises identically under either tokenizer.
    """

    style: str
    render_tok: Any
    stop_id: int

    @classmethod
    def build(cls, style: str, render_tok, student_tok) -> "PromptSpec":
        if style not in PROMPT_STYLES:
            raise ValueError(f"unknown prompt style {style!r}, expected one of {PROMPT_STYLES}")
        stop_id = student_tok.eos_token_id
        if style != "plain":
            im_end = render_tok.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end, int) and im_end >= 0:
                stop_id = im_end
        return cls(style=style, render_tok=render_tok, stop_id=stop_id)

    def student_prompt(self, question: str) -> str:
        if self.style in ("chat",):
            return chat_prompt(self.render_tok, question)
        return plain_prompt(question)

    def teacher_prompt(self, tokenizer, question: str) -> str:
        """Signature matches ``generate_rollouts``' teacher_prompt_fn."""
        if self.style == "plain":
            return plain_prompt(question)
        return chat_prompt(tokenizer, question)

    def target(self, reference_cot: str, answer: str) -> str:
        """SFT target, matching exactly what we ask the teacher to produce."""
        cot = _CALC_ANNOTATION.sub("", reference_cot).strip()
        body = f"{cot}\n{ANSWER_PREFIX} {answer}."
        # A plain prompt ends with "Answer:" and needs the separating space; a ChatML
        # prompt ends with "assistant\n" and must not have one.
        return body if self.style == "chat" else f" {body}"


def extract_answer(text: str) -> str | None:
    """Pull the predicted number out of a generation."""
    idx = text.rfind(ANSWER_PREFIX)
    if idx != -1:
        tail = text[idx + len(ANSWER_PREFIX) :]
        match = _NUMBER.search(tail)
        if match:
            return _normalise_number(match.group(0))
    matches = _NUMBER.findall(text)
    return _normalise_number(matches[-1]) if matches else None


def _normalise_number(raw: str) -> str:
    value = raw.replace(",", "").rstrip(".")
    try:
        as_float = float(value)
    except ValueError:
        return value
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:g}"


def is_correct(generation: str, gold: str) -> bool:
    predicted = extract_answer(generation)
    return predicted is not None and predicted == _normalise_number(gold)


def load_gsm8k(split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    raw = load_dataset("openai/gsm8k", "main", split=split)
    raw = raw.shuffle(seed=seed)
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))
    examples = []
    for row in raw:
        solution: str = row["answer"]
        cot, _, final = solution.partition("####")
        examples.append(
            Example(
                question=row["question"],
                reference=cot.strip(),
                answer=_normalise_number(final.strip()),
            )
        )
    return examples


def assert_same_vocab(tok_a, tok_b) -> None:
    """OPD is only well defined when both models tokenise responses identically."""
    if tok_a.vocab_size != tok_b.vocab_size:
        raise ValueError(
            f"tokenizer vocab mismatch: {tok_a.vocab_size} vs {tok_b.vocab_size}. "
            "Student and teacher must come from the same model family."
        )
    probes = [
        "Question: If 3 apples cost $12, what is the price of 7 apples?\nAnswer: 28",
        # Special tokens must agree too, since the chat style feeds ChatML ids rendered
        # by the teacher's tokenizer into the student.
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    ]
    for probe in probes:
        if tok_a(probe)["input_ids"] != tok_b(probe)["input_ids"]:
            raise ValueError(f"tokenizers disagree on probe {probe!r}; cannot share ids")
