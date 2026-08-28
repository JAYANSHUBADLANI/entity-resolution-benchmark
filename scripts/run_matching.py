"""Phase 3: train every matcher under two different splitting regimes and compare.

The two regimes exist to measure one thing. Splitting candidate pairs at random puts pairs
belonging to the same record into both the training and the test fold, so the model sees part of
an entity during training and is then asked about the rest of it. Splitting by record does not.
The gap between the two is the size of the optimism that a pair level split buys you for free.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.er.blocking import tfidf_neighbours
from src.er.config import ARTIFACTS, REPORTS, load_config
from src.er.features import build_features, label_pairs
from src.er.load import load_benchmark
from src.er.matchers import (FellegiSunter, SupervisedMatcher, ThresholdMatcher,
                             feature_columns, score_predictions)

BLOCKING_K = 20
FOLDS = 5


def build_matcher(name, seed):
    if name == "threshold rule":
        return ThresholdMatcher()
    if name == "fellegi-sunter (unsupervised)":
        return FellegiSunter(seed=seed)
    if name == "logistic regression":
        return SupervisedMatcher("logistic", seed=seed)
    return SupervisedMatcher("gradient boosting", seed=seed)


MATCHERS = ["threshold rule", "fellegi-sunter (unsupervised)", "logistic regression",
            "gradient boosting"]


def run_benchmark(key: str, config: dict) -> pd.DataFrame:
    seed = config["random_seed"]
    bench = load_benchmark(key, config)

    pairs = sorted(tfidf_neighbours(bench, "title", k=BLOCKING_K, seed=seed))
    features = build_features(bench, pairs)
    labels = label_pairs(features, bench).to_numpy()
    columns = feature_columns(features)
    X = features[columns]

    truth = bench.matches
    matches_by_left = truth.groupby("left_id").size().to_dict()

    rows = []
    for regime in ("pair level (random)", "entity level (by record)"):
        if regime.startswith("pair"):
            splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
            folds = splitter.split(X)
        else:
            splitter = GroupKFold(n_splits=FOLDS)
            folds = splitter.split(X, labels, groups=features["left_id"])

        collected = {name: [] for name in MATCHERS}
        for train_idx, test_idx in folds:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            if regime.startswith("pair"):
                # A pair level split cannot see a match that blocking never proposed, so the
                # only denominator available to it is the one it can see.
                truth_total = int(y_test.sum())
            else:
                test_entities = set(features["left_id"].iloc[test_idx])
                truth_total = int(sum(matches_by_left.get(e, 0) for e in test_entities))

            for name in MATCHERS:
                model = build_matcher(name, seed)
                if name == "fellegi-sunter (unsupervised)":
                    model.fit(X_train)
                else:
                    model.fit(X_train, y_train)
                pred = model.predict(X_test)
                collected[name].append(score_predictions(y_test, pred, truth_total))

        for name, results in collected.items():
            frame = pd.DataFrame(results)
            rows.append({
                "dataset": bench.label,
                "regime": regime,
                "matcher": name,
                "precision": round(frame["precision"].mean(), 4),
                "recall_in_candidates": round(frame["recall_in_candidates"].mean(), 4),
                "f1_in_candidates": round(frame["f1_in_candidates"].mean(), 4),
                "f1_std": round(frame["f1_in_candidates"].std(), 4),
                "recall_end_to_end": round(frame["recall_end_to_end"].mean(), 4),
                "f1_end_to_end": round(frame["f1_end_to_end"].mean(), 4),
            })

    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frames = []
    for key in config["datasets"]:
        frame = run_benchmark(key, config)
        frames.append(frame)
        print(f"\n=== {frame['dataset'].iloc[0]} ===")
        print(frame.drop(columns=["dataset"]).to_string(index=False))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(ARTIFACTS / "matching_results.csv", index=False)

    # The leakage number: same matcher, same features, only the split changes.
    pivot = combined.pivot_table(index=["dataset", "matcher"], columns="regime",
                                 values="f1_in_candidates")
    pivot["inflation_pp"] = ((pivot["pair level (random)"] - pivot["entity level (by record)"])
                             * 100).round(2)
    pivot.to_csv(ARTIFACTS / "split_leakage.csv")
    print("\n=== F1 inflation from splitting pairs instead of records (percentage points) ===")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
