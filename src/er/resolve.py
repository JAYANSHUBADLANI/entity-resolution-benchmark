"""Turn pair scores into a set of decided matches.

A matcher scores each candidate pair on its own, which means nothing stops it from claiming that
one product matches nine others. Whether that is wrong depends on the data: in a bibliographic
merge of two deduplicated catalogues each paper appears once per source, so a record may be used
at most once. In a product catalogue the same item is genuinely listed several times, so forcing
one match per record deletes true matches rather than false ones.

Both constraints are implemented here, along with the arithmetic that says what each one costs on
a given ground truth, so the choice is made against a measured ceiling rather than a habit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def constraint_ceiling(matches: pd.DataFrame, max_left: int, max_right: int) -> dict:
    """The most true matches any decision rule can keep under these degree caps.

    Greedy on true matches alone: repeatedly keep a match while both of its records still have
    capacity. This is an upper bound on recall, independent of any model.
    """
    left_used: dict[str, int] = {}
    right_used: dict[str, int] = {}
    kept = 0
    for left, right in zip(matches["left_id"], matches["right_id"]):
        if left_used.get(left, 0) < max_left and right_used.get(right, 0) < max_right:
            left_used[left] = left_used.get(left, 0) + 1
            right_used[right] = right_used.get(right, 0) + 1
            kept += 1
    return {
        "max_left": max_left,
        "max_right": max_right,
        "keepable_matches": kept,
        "total_matches": len(matches),
        "recall_ceiling": round(kept / len(matches), 4),
    }


def observed_degrees(matches: pd.DataFrame) -> tuple[int, int]:
    """Degree caps read off a set of labelled matches, used on training folds only."""
    if matches.empty:
        return 1, 1
    return (int(matches.groupby("left_id").size().max()),
            int(matches.groupby("right_id").size().max()))


def degree_capped_greedy(pairs: pd.DataFrame, score: np.ndarray, threshold: float,
                         max_left: int, max_right: int) -> np.ndarray:
    """Accept pairs in descending score order while both records still have capacity."""
    order = np.argsort(-score)
    left = pairs["left_id"].to_numpy()
    right = pairs["right_id"].to_numpy()
    left_used: dict[str, int] = {}
    right_used: dict[str, int] = {}
    decision = np.zeros(len(score), dtype=int)
    for idx in order:
        if score[idx] < threshold:
            break
        l, r = left[idx], right[idx]
        if left_used.get(l, 0) < max_left and right_used.get(r, 0) < max_right:
            left_used[l] = left_used.get(l, 0) + 1
            right_used[r] = right_used.get(r, 0) + 1
            decision[idx] = 1
    return decision


def hungarian_one_to_one(pairs: pd.DataFrame, score: np.ndarray, threshold: float) -> np.ndarray:
    """Globally optimal 1:1 assignment over the candidate scores.

    Included to answer an obvious question: does the greedy rule leave anything on the table
    compared with solving the assignment exactly. Only pairs at or above the threshold are
    eligible, so the assignment cannot be forced to accept a low scoring pair.
    """
    left_ids = pd.Index(sorted(set(pairs["left_id"])))
    right_ids = pd.Index(sorted(set(pairs["right_id"])))
    cost = np.zeros((len(left_ids), len(right_ids)), dtype=float)
    rows = left_ids.get_indexer(pairs["left_id"])
    cols = right_ids.get_indexer(pairs["right_id"])
    eligible = score >= threshold
    cost[rows[eligible], cols[eligible]] = -score[eligible]

    row_idx, col_idx = linear_sum_assignment(cost)
    chosen = {(left_ids[r], right_ids[c]) for r, c in zip(row_idx, col_idx) if cost[r, c] < 0}
    return np.array([1 if pair in chosen else 0
                     for pair in zip(pairs["left_id"], pairs["right_id"])], dtype=int)
