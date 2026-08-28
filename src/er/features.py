"""Similarity features for candidate pairs.

One function per family of comparison, so each feature can be named in the report and read
individually. Nothing here is learned, these are deterministic comparisons, which keeps the
feature step separable from the matcher step and lets the unsupervised matcher use the same
inputs as the supervised ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalise, tokens

NUMERIC_FIELDS = {"year", "price"}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set, b: set) -> float:
    if not a or not b:
        return np.nan if not (a or b) else 0.0
    return len(a & b) / min(len(a), len(b))


def _cosine_for_pairs(left_text, right_text, left_pos, right_pos, analyzer, ngram_range):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1)
    matrix = vec.fit_transform(list(left_text) + list(right_text))
    left_matrix = matrix[: len(left_text)]
    right_matrix = matrix[len(left_text):]
    # Row-wise cosine on just the candidate pairs, never the full matrix product.
    lm = left_matrix[left_pos]
    rm = right_matrix[right_pos]
    return np.asarray(lm.multiply(rm).sum(axis=1)).ravel()


def build_features(bench, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    left = bench.left.set_index("record_id")
    right = bench.right.set_index("record_id")
    left_order = {rid: i for i, rid in enumerate(left.index)}
    right_order = {rid: i for i, rid in enumerate(right.index)}

    left_ids = [p[0] for p in pairs]
    right_ids = [p[1] for p in pairs]
    left_pos = np.array([left_order[i] for i in left_ids])
    right_pos = np.array([right_order[i] for i in right_ids])

    out = pd.DataFrame({"left_id": left_ids, "right_id": right_ids})

    for field in bench.compare_fields:
        if field not in left.columns or field not in right.columns:
            continue

        if field in NUMERIC_FIELDS:
            lv = pd.to_numeric(left[field], errors="coerce").to_numpy()[left_pos]
            rv = pd.to_numeric(right[field], errors="coerce").to_numpy()[right_pos]
            with np.errstate(invalid="ignore", divide="ignore"):
                denom = np.maximum(np.abs(lv), np.abs(rv))
                rel = np.where(denom > 0, np.abs(lv - rv) / denom, np.nan)
            out[f"{field}_equal"] = (lv == rv).astype(float)
            out[f"{field}_rel_diff"] = rel
            continue

        left_text = [normalise(v) for v in left[field]]
        right_text = [normalise(v) for v in right[field]]
        left_tokens = [set(t.split()) if t else set() for t in left_text]
        right_tokens = [set(t.split()) if t else set() for t in right_text]

        out[f"{field}_jaccard"] = [
            jaccard(left_tokens[i], right_tokens[j]) for i, j in zip(left_pos, right_pos)
        ]
        out[f"{field}_containment"] = [
            containment(left_tokens[i], right_tokens[j]) for i, j in zip(left_pos, right_pos)
        ]
        out[f"{field}_cosine_char"] = _cosine_for_pairs(
            left_text, right_text, left_pos, right_pos, "char_wb", (2, 4))
        out[f"{field}_cosine_word"] = _cosine_for_pairs(
            left_text, right_text, left_pos, right_pos, "word", (1, 1))
        # A field that is empty on one side is not evidence of disagreement, it is missing
        # evidence, and the two are different. This flag lets a model tell them apart.
        out[f"{field}_either_missing"] = [
            float(not left_text[i] or not right_text[j]) for i, j in zip(left_pos, right_pos)
        ]

    return out


def label_pairs(features: pd.DataFrame, bench) -> pd.Series:
    truth = set(zip(bench.matches["left_id"], bench.matches["right_id"]))
    return pd.Series(
        [1 if pair in truth else 0 for pair in zip(features["left_id"], features["right_id"])],
        index=features.index, name="is_match")
