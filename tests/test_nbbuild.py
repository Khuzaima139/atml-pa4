"""Notebook building from a cells markdown file."""
import ast
import json

import pytest

from common import nbbuild

MD = """# Title

Intro text.

## Cell 1: Setup

Prose for cell 1.

```python
EXPECTED = "{{COMMIT}}"
print(EXPECTED)
```

## Cell 2: Second

```python
x = 1
```
"""


def test_build_cells(tmp_path):
    nb = nbbuild.build(MD, {"COMMIT": "abc123"})
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds == ["markdown", "markdown", "code", "markdown", "code"]
    assert "".join(nb["cells"][2]["source"]) == 'EXPECTED = "abc123"\nprint(EXPECTED)'
    assert "".join(nb["cells"][1]["source"]).startswith("## Cell 1: Setup")
    assert nb["nbformat"] == 4 and all(c["id"] for c in nb["cells"])
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_placeholders_and_refusals(tmp_path):
    with pytest.raises(ValueError, match="COMMIT"):
        nbbuild.build(MD, {})
    with pytest.raises(ValueError, match="unclosed"):
        nbbuild.parse("```python\nx = 1\n")
    md = tmp_path / "c.md"
    md.write_text(MD)
    out = tmp_path / "nb" / "n.ipynb"
    assert nbbuild.main(["--cells", str(md), "--out", str(out), "--set", "COMMIT=f00"]) == 0
    assert 'EXPECTED = "f00"' in "".join(json.loads(out.read_text())["cells"][2]["source"])
    with pytest.raises(FileExistsError):
        nbbuild.main(["--cells", str(md), "--out", str(out), "--set", "COMMIT=f00"])


def test_code_cells_parse():
    for c in nbbuild.build(MD, {"COMMIT": "x"})["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))


def test_sections(tmp_path):
    md = MD + """
# Second title

## Cell 1: Other

```python
# a comment line, not a heading
y = 2
```
"""
    secs = nbbuild.sections(md)
    assert list(secs) == ["Title", "Second title"]
    assert "a comment line" in secs["Second title"] and "Second title" not in secs["Title"]
    nb = nbbuild.build(nbbuild.section(md, "Second title"), {})
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "markdown", "code"]
    with pytest.raises(ValueError, match="no section"):
        nbbuild.section(md, "Missing")
    with pytest.raises(ValueError, match="repeated"):
        nbbuild.sections(md + "\n# Title\n")
    f = tmp_path / "c.md"
    f.write_text(md)
    out = tmp_path / "s.ipynb"
    assert nbbuild.main(["--cells", str(f), "--out", str(out), "--section", "Second title"]) == 0
    assert "y = 2" in "".join(json.loads(out.read_text())["cells"][2]["source"])
