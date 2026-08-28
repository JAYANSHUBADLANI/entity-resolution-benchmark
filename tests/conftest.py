"""Fixtures built in memory, so the suite runs with no network and no downloaded data."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.er.load import Benchmark


@pytest.fixture
def tiny_benchmark() -> Benchmark:
    """Four records a side. l1-r1 and l2-r2 match, l3 matches both r3 and r4."""
    left = pd.DataFrame({
        "record_id": ["l1", "l2", "l3", "l4"],
        "title": ["red widget", "blue gadget", "green sprocket", "totally unrelated"],
        "year": [2001, 2002, 2003, 2004],
    })
    right = pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4"],
        "title": ["red widget", "blue gadget deluxe", "green sprocket", "green sprocket pack"],
        "year": [2001, 2002, 2003, 2003],
    })
    matches = pd.DataFrame({
        "left_id": ["l1", "l2", "l3", "l3"],
        "right_id": ["r1", "r2", "r3", "r4"],
    })
    return Benchmark(key="tiny", label="Tiny", character="test", left=left, right=right,
                     matches=matches, compare_fields=["title", "year"])
