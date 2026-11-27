"""Part C per-prompt metrics (protocol C4, C5, C7, C8, AM2.6). Gold codes and evidence offsets enter only here and in
the teacher-forced pass, never in selection or compression."""
from __future__ import annotations

import re

import torch

SCORER = ("task2-kv-scorer-v1: first match of (?<!\\d)\\d{5}(?!\\d) in the generated text equals the gold code; "
          "no match is incorrect and flagged")
CODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")


def extract_code(text: str) -> str | None:
    m = CODE_RE.search(text)
    return m.group(0) if m else None


def exact_match(text: str, code: str) -> dict:
    pred = extract_code(text)
    return {"prediction": pred, "no_match": pred is None, "correct": pred == code}


def kl_rows(ref_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(ref || method) per row over the full vocabulary, FP32."""
    lp = torch.log_softmax(ref_logits.float(), dim=-1)
    lq = torch.log_softmax(logits.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def nll_rows(logits: torch.Tensor, target_ids) -> torch.Tensor:
    """Negative log-likelihood of each target token, FP32."""
    t = torch.as_tensor(list(target_ids), device=logits.device)
    return -torch.log_softmax(logits.float(), dim=-1).gather(-1, t[:, None]).squeeze(-1)


def evidence_tokens(offsets, char_start: int, char_end: int) -> list[int]:
    """Token indices whose character range [a, b) overlaps [char_start, char_end) (protocol C4)."""
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < char_end and b > char_start]


def evidence_fraction(fp16_positions, evidence_positions) -> float:
    kept = set(int(p) for p in fp16_positions)
    ev = [int(p) for p in evidence_positions]
    return sum(p in kept for p in ev) / len(ev)
