"""Named repository paths."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_DIR = REPO_ROOT / "external"
DATA_REPO_DIR = EXTERNAL_DIR / "atml-pa4-datasets"
DATA_REPO_PUBLIC_LOCKED = DATA_REPO_DIR / "data" / "public_locked"
DATA_LOCK_DIR = REPO_ROOT / "data_lock"
PUBLIC_ASSETS_LOCK = DATA_LOCK_DIR / "public_assets.json"
SOURCE_REVISIONS_LOCK = DATA_LOCK_DIR / "local_source_revisions.json"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
