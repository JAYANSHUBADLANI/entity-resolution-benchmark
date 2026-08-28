"""Blocking: cut the pair space down before any pair is scored, and measure what that costs.

The measurement is the point. Blocking is usually presented as a performance optimisation, but
it is really a recall decision made before the model exists. Whatever share of true matches a
blocking scheme discards is a ceiling on the recall of every matcher that runs after it, and no
amount of downstream modelling can recover it. So each scheme here reports pairs completeness
(the share of true matches it keeps) next to reduction ratio (the share of the pair space it
removes), and the ceiling is stated explicitly.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .text import normalise, tokens


def _truth_set(bench) -> set[tuple[str, str]]:
    return set(zip(bench.matches["left_id"], bench.matches["right_id"]))


def full_cartesian(bench) -> set[tuple[str, str]]:
    return {(l, r) for l in bench.left["record_id"] for r in bench.right["record_id"]}


def exact_field(bench, field: str) -> set[tuple[str, str]]:
    """Every pair that agrees exactly on one normalised field."""
    index = defaultdict(list)
    for rid, value in zip(bench.right["record_id"], bench.right[field]):
        index[normalise(value)].append(rid)
    pairs = set()
    for lid, value in zip(bench.left["record_id"], bench.left[field]):
        key = normalise(value)
        if not key:
            continue
        for rid in index.get(key, ()):
            pairs.add((lid, rid))
    return pairs


def shared_token(bench, field: str, max_block: int = 500) -> set[tuple[str, str]]:
    """Every pair sharing at least one token in `field`.

    Blocks larger than `max_block` right records are skipped: a token that appears in hundreds of
    records carries no discriminating information and would reintroduce most of the pair space.
    The cap is a design choice with a recall cost, and that cost is measured like everything else.
    """
    index = defaultdict(list)
    for rid, value in zip(bench.right["record_id"], bench.right[field]):
        for token in set(tokens(value)):
            index[token].append(rid)
    oversized = {token for token, ids in index.items() if len(ids) > max_block}

    pairs = set()
    for lid, value in zip(bench.left["record_id"], bench.left[field]):
        for token in set(tokens(value)):
            if token in oversized:
                continue
            for rid in index.get(token, ()):
                pairs.add((lid, rid))
    return pairs


def tfidf_neighbours(bench, field: str, k: int = 20, seed: int = 42) -> set[tuple[str, str]]:
    """Keep each left record's k nearest right records by TF-IDF cosine on `field`.

    This is the scheme that actually scales, because the candidate count grows linearly in the
    number of left records rather than quadratically.
    """
    left_text = [normalise(v) for v in bench.left[field]]
    right_text = [normalise(v) for v in bench.right[field]]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    right_matrix = vec.fit_transform(right_text)
    left_matrix = vec.transform(left_text)

    k = min(k, right_matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(right_matrix)
    _, indices = nn.kneighbors(left_matrix)

    right_ids = bench.right["record_id"].to_numpy()
    left_ids = bench.left["record_id"].to_numpy()
    pairs = set()
    for row, lid in enumerate(left_ids):
        for col in indices[row]:
            pairs.add((lid, right_ids[col]))
    return pairs


def evaluate_blocking(name: str, pairs: set, bench) -> dict:
    truth = _truth_set(bench)
    kept = len(truth & pairs)
    candidates = len(pairs)
    space = bench.pair_space
    return {
        "scheme": name,
        "candidate_pairs": candidates,
        "candidates_pct_of_space": round(100 * candidates / space, 4),
        "true_matches_kept": kept,
        "pairs_completeness": round(kept / len(truth), 4),
        "recall_ceiling": round(kept / len(truth), 4),
        "reduction_ratio": round(1 - candidates / space, 6),
        "pair_quality": round(kept / candidates, 6) if candidates else 0.0,
        "matches_lost": len(truth) - kept,
    }


def run_all(bench, k_values=(1, 5, 10, 20, 50)) -> pd.DataFrame:
    rows = [evaluate_blocking("full cartesian", full_cartesian(bench), bench)]

    for field in bench.compare_fields:
        if field in {"description", "price"}:
            continue
        rows.append(evaluate_blocking(f"exact on {field}", exact_field(bench, field), bench))

    rows.append(evaluate_blocking("shared title token", shared_token(bench, "title"), bench))

    for k in k_values:
        pairs = tfidf_neighbours(bench, "title", k=k)
        rows.append(evaluate_blocking(f"tfidf title top-{k}", pairs, bench))

    return pd.DataFrame(rows).sort_values("candidate_pairs").reset_index(drop=True)
