"""GSM8K loading, prompt rendering and answer checking.

Teacher and student see *different* prompt renderings (the teacher is an instruct model
and gets its chat template, the student is a base model and gets a plain completion
prompt) but they score the *same* response token ids. That is what on-policy
distillation requires: identical continuation tokens, each model conditioned on its own
natural context. It only works because both models share the Qwen3 tokenizer, which
``assert_same_vocab`` checks at startup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

ANSWER_PREFIX = "The answer is"

TEACHER_INSTRUCTION = (
    "Solve the following grade school math problem. Reason step by step, keep the "
    "reasoning short, and finish with a single line of the form "
    f"'{ANSWER_PREFIX} <number>.'"
)

_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class Example:
    question: str
    reference: str
    answer: str


def student_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def teacher_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "user", "content": f"{TEACHER_INSTRUCTION}\n\nQuestion: {question.strip()}"}
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Qwen3-*-Instruct-2507 is non-thinking only and rejects enable_thinking.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # The student's response starts with a leading space after "Answer:"; mirror that so
    # the teacher scores the same first token in a context where it is natural.
    return text


def render_target(reference_cot: str, answer: str) -> str:
    """SFT target. Matches exactly what we ask the teacher to produce."""
    cot = _CALC_ANNOTATION.sub("", reference_cot).strip()
    return f" {cot}\n{ANSWER_PREFIX} {answer}."


def extract_answer(text: str) -> str | None:
    """Pull the predicted number out of a generation."""
    idx = text.rfind(ANSWER_PREFIX)
    if idx != -1:
        tail = text[idx + len(ANSWER_PREFIX) :]
        match = _NUMBER.search(tail)
        if match:
            return _normalise_number(match.group(0))
    # Fallback: last number anywhere in the completion.
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
    probe = "Question: If 3 apples cost $12, what is the price of 7 apples?\nAnswer: 28"
    if tok_a(probe)["input_ids"] != tok_b(probe)["input_ids"]:
        raise ValueError("tokenizers disagree on a probe string; cannot share response ids")
