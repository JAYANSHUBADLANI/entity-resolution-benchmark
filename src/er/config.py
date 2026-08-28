"""Config and path handling. One source of truth for where things live."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def load_config(path: Path | None = None) -> dict:
    path = path or ROOT / "config" / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)
