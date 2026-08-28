"""Load the two sides of a benchmark plus its ground truth mapping.

Ground truth here is a list of matching id pairs. Everything not listed is a non match, which
is the standard reading of these benchmarks and is worth stating out loud: it makes the labels
complete rather than sampled, so recall is measurable against the whole pair space.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DATA_RAW, load_config


@dataclass
class Benchmark:
    key: str
    label: str
    character: str
    left: pd.DataFrame
    right: pd.DataFrame
    matches: pd.DataFrame
    compare_fields: list[str]

    @property
    def pair_space(self) -> int:
        return len(self.left) * len(self.right)

    @property
    def match_rate(self) -> float:
        return len(self.matches) / self.pair_space


def load_benchmark(key: str, config: dict | None = None) -> Benchmark:
    config = config or load_config()
    spec = config["datasets"][key]
    base = DATA_RAW / key
    enc = spec["encoding"]

    left = pd.read_csv(base / spec["left_file"], encoding=enc)
    right = pd.read_csv(base / spec["right_file"], encoding=enc)
    matches = pd.read_csv(base / spec["mapping_file"], encoding=enc)

    if "rename_right" in spec:
        right = right.rename(columns=spec["rename_right"])

    left = left.rename(columns={"id": "record_id"})
    right = right.rename(columns={"id": "record_id"})
    matches = matches.rename(columns={spec["left_id"]: "left_id", spec["right_id"]: "right_id"})

    for frame in (left, right):
        frame["record_id"] = frame["record_id"].astype(str)
    matches["left_id"] = matches["left_id"].astype(str)
    matches["right_id"] = matches["right_id"].astype(str)

    return Benchmark(
        key=key,
        label=spec["label"],
        character=spec["character"],
        left=left,
        right=right,
        matches=matches,
        compare_fields=list(spec["compare_fields"]),
    )
