"""requirements-kaggle.txt: exact pins equal to the installed versions, lm_eval direct dependencies, no torch."""
import importlib.metadata as md

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from common.paths import REPO_ROOT


def _pins():
    out = {}
    for line in (REPO_ROOT / "requirements-kaggle.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, ver = line.split("==")
        out[canonicalize_name(name)] = ver
    return out


def test_pins_exact_and_installed():
    pins = _pins()
    assert "torch" not in pins
    assert pins["transformers"] == "4.57.1" and pins["huggingface-hub"] == "0.36.2" and pins["lm-eval"] == "0.4.13"
    for name, ver in pins.items():
        assert md.version(name) == ver, name


def test_lm_eval_direct_dependencies_pinned():
    pins = _pins()
    base = [Requirement(r) for r in md.requires("lm_eval")]
    base = [r for r in base if not r.marker or r.marker.evaluate({"extra": ""})]
    want = {canonicalize_name(r.name) for r in base} - {"torch"}
    assert want <= set(pins), sorted(want - set(pins))
    assert {"datasets", "accelerate", "safetensors"} <= set(pins)
