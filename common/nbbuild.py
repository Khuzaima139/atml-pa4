"""Build a Jupyter notebook from a cells markdown file.

Usage: python -m common.nbbuild --cells FILE.md --out FILE.ipynb [--section TITLE] [--set NAME=VALUE ...]

Format: text before the first "## Cell" heading becomes a markdown cell; every "## Cell ..." section becomes one
markdown cell (heading and prose) followed by one code cell per ```python fenced block. {{NAME}} placeholders are
replaced from --set; an unknown or unreplaced placeholder is an error. The output file must not exist.
--section TITLE uses only the part of the file from the top-level heading line "# TITLE" (outside code blocks) to the
next top-level heading; a missing or repeated title is an error.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_PH = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
_FENCE_OPEN = "```python"
_FENCE_CLOSE = "```"


def _cell_id(kind: str, i: int, text: str) -> str:
    return hashlib.sha256(f"{kind}{i}{text}".encode()).hexdigest()[:12]


def _lines(text: str) -> list:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def parse(md: str) -> list:
    """[(kind, text)] with kind 'markdown' or 'code'."""
    cells = []
    prose, code, in_code = [], [], False

    def flush_prose():
        text = "\n".join(prose).strip("\n")
        if text.strip():
            cells.append(("markdown", text))
        prose.clear()

    for line in md.split("\n"):
        if in_code:
            if line.strip() == _FENCE_CLOSE:
                cells.append(("code", "\n".join(code).rstrip("\n")))
                code.clear()
                in_code = False
            else:
                code.append(line)
            continue
        if line.strip() == _FENCE_OPEN:
            flush_prose()
            in_code = True
            continue
        if line.startswith("## Cell"):
            flush_prose()
        prose.append(line)
    if in_code:
        raise ValueError("unclosed ```python block")
    flush_prose()
    return cells


def sections(md: str) -> dict:
    """{title: text} for every top-level "# " heading outside ```python blocks; text starts at the heading line."""
    out, order, cur, in_code = {}, [], None, False
    for line in md.split("\n"):
        if in_code:
            in_code = line.strip() != _FENCE_CLOSE
        elif line.strip() == _FENCE_OPEN:
            in_code = True
        elif line.startswith("# "):
            cur = line[2:].strip()
            if cur in out:
                raise ValueError(f"repeated section title {cur!r}")
            out[cur] = []
            order.append(cur)
        if cur is not None:
            out[cur].append(line)
    return {t: "\n".join(out[t]).rstrip("\n") + "\n" for t in order}


def section(md: str, title: str) -> str:
    secs = sections(md)
    if title not in secs:
        raise ValueError(f"no section {title!r}; sections: {list(secs)}")
    return secs[title]


def substitute(text: str, values: dict) -> str:
    missing = sorted(set(_PH.findall(text)) - set(values))
    if missing:
        raise ValueError(f"no value for placeholders {missing}")
    return _PH.sub(lambda m: values[m.group(1)], text)


def build(md: str, values: dict) -> dict:
    cells = []
    for i, (kind, text) in enumerate(parse(md)):
        text = substitute(text, values)
        c = {"cell_type": kind, "id": _cell_id(kind, i, text), "metadata": {}, "source": _lines(text)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True,
                                    "language": "python", "sourceType": "notebook"}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--section", default=None, help="title of the top-level section to build")
    a = ap.parse_args(argv)
    values = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not _PH.fullmatch("{{" + k + "}}"):
            ap.error(f"bad placeholder name {k!r}")
        values[k] = v
    md = Path(a.cells).read_text(encoding="utf-8")
    if a.section is not None:
        md = section(md, a.section)
    nb = build(md, values)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    print(f"{out}: {len(nb['cells'])} cells ({n_code} code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
