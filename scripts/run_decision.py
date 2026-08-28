"""Phase 4: the decision layer, which is where a scored candidate becomes a claimed match.

Three things are compared, in the order a real build would reach for them:

1. the default 0.5 cut, which nobody chooses deliberately and everybody ships;
2. a cut tuned on a validation split carved out of the training fold, never on the test fold;
3. the tuned cut plus a cardinality constraint fitted from the training fold's own labels.

Everything runs under the entity level split, since the earlier experiment established that a
pair level split cannot see a whole class of failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.er.blocking import tfidf_neighbours
from src.er.config import ARTIFACTS, REPORTS, load_config
from src.er.features import build_features, label_pairs
from src.er.load import load_benchmark
from src.er.matchers import SupervisedMatcher, feature_columns
from src.er.resolve import (constraint_ceiling, degree_capped_greedy, hungarian_one_to_one,
                            observed_degrees)

FOLDS = 5
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def prf(y_true: np.ndarray, y_pred: np.ndarray, truth_total: int) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / truth_total if truth_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp}


def tune_threshold(score: np.ndarray, y: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for cut in GRID:
        pred = (score >= cut).astype(int)
        result = prf(y, pred, int(y.sum()))
        if result["f1"] > best[0]:
            best = (result["f1"], float(cut))
    return best[1]


def run(key: str, config: dict):
    seed = config["random_seed"]
    bench = load_benchmark(key, config)
    pairs = sorted(tfidf_neighbours(bench, "title", k=20, seed=seed))
    features = build_features(bench, pairs)
    labels = label_pairs(features, bench).to_numpy()
    X = features[feature_columns(features)]
    groups = features["left_id"]
    matches_by_left = bench.matches.groupby("left_id").size().to_dict()

    ceilings = [constraint_ceiling(bench.matches, 1, 1),
                constraint_ceiling(bench.matches, *observed_degrees(bench.matches))]

    rows = []
    curves = []
    for fold, (train_idx, test_idx) in enumerate(
            GroupKFold(n_splits=FOLDS).split(X, labels, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        pairs_test = features.iloc[test_idx][["left_id", "right_id"]]

        test_entities = set(features["left_id"].iloc[test_idx])
        truth_total = int(sum(matches_by_left.get(e, 0) for e in test_entities))

        # Carve a validation split out of the training fold, again by record.
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        inner_train, inner_val = next(inner.split(X_train, y_train,
                                                  groups=groups.iloc[train_idx]))

        tuner = SupervisedMatcher("gradient boosting", seed=seed).fit(
            X_train.iloc[inner_train], y_train[inner_train])
        cut = tune_threshold(tuner.score(X_train.iloc[inner_val]), y_train[inner_val])

        model = SupervisedMatcher("gradient boosting", seed=seed).fit(X_train, y_train)
        score = model.score(X_test)

        train_matches = bench.matches[bench.matches["left_id"].isin(
            set(features["left_id"].iloc[train_idx]))]
        max_left, max_right = observed_degrees(train_matches)

        variants = {
            "default cut 0.50": (score >= 0.5).astype(int),
            f"tuned cut": (score >= cut).astype(int),
            "tuned cut + degree cap (fitted per fold)":
                degree_capped_greedy(pairs_test, score, cut, max_left, max_right),
            "tuned cut + strict 1:1 (greedy)":
                degree_capped_greedy(pairs_test, score, cut, 1, 1),
            "tuned cut + strict 1:1 (hungarian)":
                hungarian_one_to_one(pairs_test, score, cut),
        }
        for name, pred in variants.items():
            result = prf(y_test, pred, truth_total)
            result.update({"dataset": bench.label, "rule": name, "fold": fold,
                           "threshold": cut if "default" not in name else 0.5,
                           "max_left": max_left, "max_right": max_right})
            rows.append(result)

        for cut_value in GRID:
            pred = (score >= cut_value).astype(int)
            point = prf(y_test, pred, truth_total)
            curves.append({"dataset": bench.label, "fold": fold, "threshold": float(cut_value),
                           "precision": point["precision"], "recall": point["recall"]})

    frame = pd.DataFrame(rows)
    summary = (frame.groupby(["dataset", "rule"])
               .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                    f1=("f1", "mean"), f1_std=("f1", "std"), threshold=("threshold", "mean"),
                    caps_left=("max_left", "max"), caps_right=("max_right", "max"))
               .round(4).reset_index())
    return summary, pd.DataFrame(curves), ceilings, bench.label


def main() -> None:
    config = load_config()
    summaries, curves, ceiling_rows = [], [], []
    for key in config["datasets"]:
        summary, curve, ceilings, label = run(key, config)
        summaries.append(summary)
        curves.append(curve)
        for c in ceilings:
            c["dataset"] = label
            ceiling_rows.append(c)
        print(f"\n=== {label} ===")
        print(summary.drop(columns=["dataset"]).to_string(index=False))

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(ARTIFACTS / "decision_rules.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(ARTIFACTS / "pr_curves.csv", index=False)
    ceilings = pd.DataFrame(ceiling_rows)
    ceilings.to_csv(ARTIFACTS / "cardinality_ceilings.csv", index=False)

    print("\n=== What each cardinality constraint can possibly keep ===")
    print(ceilings[["dataset", "max_left", "max_right", "keepable_matches",
                    "total_matches", "recall_ceiling"]].to_string(index=False))


if __name__ == "__main__":
    main()
