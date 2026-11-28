"""Standalone-accuracy scorer (protocol D17): mathematics extraction, short-form exact match and per-template format
compliance.

Responses are scored as decoded text with special tokens skipped (the EOS token is not part of the text).

Mathematics: the first number after the last "####" if the response contains "####", else the last number in the
response; commas are removed; the prediction is correct when it equals the gold number (the number after "####" in
the gold solution) as an exact decimal. A response without any number is unextractable and counts as wrong.

Short form: exact match of the response and the gold answer after stripping surrounding whitespace; format compliance
is a full match of the stripped response against the regex of the prompt's template. A template is (family, form):
family 0 date to month, 1 quantity, 2 code; form "familiar" (the training and familiar test template) or "shifted".
Each regex is derived from the template's instruction:
- 0 familiar "Return only the month." / 0 shifted "Answer with the full month name.": one full English month name.
- 1 familiar "Give the integer.": an optionally signed integer written as digits.
- 1 shifted "Give count as digits, nothing else.": digits only.
- 2 familiar "Output the destination code." (codes are two capital letters in the prompt) / 2 shifted "Extract the
  two-letter code only.": two capital letters.
"""
import re
from decimal import Decimal, InvalidOperation

SCORER_ID = "t3-scorer-v1"

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")
_MONTH_RE = r"(?:" + "|".join(_MONTHS) + r")"

TEMPLATES = {
    (0, "familiar"): {"instruction": "Return only the month.", "regex": _MONTH_RE},
    (0, "shifted"): {"instruction": "Answer with the full month name.", "regex": _MONTH_RE},
    (1, "familiar"): {"instruction": "Give the integer.", "regex": r"-?\d+"},
    (1, "shifted"): {"instruction": "Give count as digits, nothing else.", "regex": r"\d+"},
    (2, "familiar"): {"instruction": "Output the destination code.", "regex": r"[A-Z]{2}"},
    (2, "shifted"): {"instruction": "Extract the two-letter code only.", "regex": r"[A-Z]{2}"},
}
_COMPILED = {k: re.compile(v["regex"]) for k, v in TEMPLATES.items()}

SCORER_SPEC = {
    "scorer_id": SCORER_ID,
    "input": "decoded response text, special tokens skipped",
    "math": "first number after the last '####' if present, else the last number; commas removed; exact decimal "
            "equality with the number after '####' in the gold solution; no number = unextractable = wrong",
    "short_exact": "response.strip() == gold.strip()",
    "short_format": "re.fullmatch(template regex, response.strip())",
    "templates": {f"{f}_{form}": v for (f, form), v in TEMPLATES.items()},
}


def template_key(family, split) -> tuple:
    """(family, form): splits "train" and "familiar" share the familiar template, "shifted" is its own."""
    family = int(family)
    if split in ("train", "familiar"):
        form = "familiar"
    elif split == "shifted":
        form = "shifted"
    else:
        raise ValueError(f"unknown short-form split {split!r}")
    if (family, form) not in TEMPLATES:
        raise ValueError(f"unknown template ({family}, {form})")
    return family, form


def _clean(num: str) -> str:
    return num.replace(",", "")


def extract_math(text: str):
    """The extracted number as a string with commas removed, or None."""
    if "####" in text:
        m = _NUMBER.search(text.rsplit("####", 1)[1])
        if m:
            return _clean(m.group(0))
    found = _NUMBER.findall(text)
    return _clean(found[-1]) if found else None


def _decimal(s):
    try:
        return Decimal(s)
    except (InvalidOperation, TypeError):
        return None


def gold_math(solution: str) -> str:
    """Gold number of a GSM8K solution: the number after its '####' (commas removed)."""
    if "####" not in solution:
        raise ValueError("gold solution without '####'")
    m = _NUMBER.search(solution.rsplit("####", 1)[1])
    if not m:
        raise ValueError("no number after '####' in the gold solution")
    return _clean(m.group(0))


def score_math(response: str, gold: str) -> dict:
    pred = extract_math(response)
    g, p = _decimal(gold), _decimal(pred)
    return {"scorer_id": SCORER_ID, "prediction": pred, "gold": gold, "extractable": pred is not None,
            "has_hash_marker": "####" in response, "correct": bool(p is not None and g is not None and p == g)}


def format_ok(response: str, family, split) -> bool:
    return _COMPILED[template_key(family, split)].fullmatch(response.strip()) is not None


def score_short(response: str, gold: str, family, split) -> dict:
    key = template_key(family, split)
    return {"scorer_id": SCORER_ID, "prediction": response.strip(), "gold": gold.strip(),
            "template": f"{key[0]}_{key[1]}", "correct": response.strip() == gold.strip(),
            "format_ok": format_ok(response, family, split)}
