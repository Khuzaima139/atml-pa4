"""Print run metadata as JSON and assert the pinned library versions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.runmeta import collect_meta  # noqa: E402

PINS = {"transformers": "4.57.1", "huggingface_hub": "0.36.2"}


def main():
    meta = collect_meta()
    print(json.dumps(meta, indent=2, sort_keys=True))
    for pkg, want in PINS.items():
        got = meta["package_versions"].get(pkg)
        assert got == want, f"{pkg}=={got}, expected {want}"
    print("env_check: pins OK")


if __name__ == "__main__":
    main()
