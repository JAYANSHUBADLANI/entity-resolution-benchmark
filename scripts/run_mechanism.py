"""Two follow up experiments, both prompted by results that did not come out as expected.

Experiment A. The pair level split was expected to inflate F1 and did not, by any margin worth
reporting. The explanation offered is that these features are symmetric similarity functions, so
there is no channel through which a model could memorise a particular record: seeing one pair of
record X in training tells it nothing about record X specifically, only about how similar titles
behave in general. That explanation is testable. Add one feature that does carry record identity,
a target encoding of the left record id, which is a routine thing to add and looks harmless, and
the inflation should appear. If it does not, the explanation is wrong.

Experiment B. Fellegi-Sunter came out with high recall and poor precision. The suspected cause is
its conditional independence assumption meeting four correlated views of the same field. Refitting
it on one feature per field tests that directly.
"""
from __future__ import annotations

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
from src.er.matchers import (FellegiSunter, SupervisedMatcher, feature_columns,
                             score_predictions)

FOLDS = 5


def target_encode(train_ids, y_train, test_ids):
    frame = pd.DataFrame({"id": np.asarray(train_ids), "y": y_train})
    means = frame.groupby("id")["y"].mean()
    prior = float(y_train.mean())
    return np.asarray([means.get(i, prior) for i in test_ids], dtype=float)


def experiment_a(bench, X, features, labels, seed):
    rows = []
    for with_identity in (False, True):
        for regime in ("pair level (random)", "entity level (by record)"):
            if regime.startswith("pair"):
                folds = KFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(X)
            else:
                folds = GroupKFold(n_splits=FOLDS).split(X, labels, groups=features["left_id"])

            scores = []
            for train_idx, test_idx in folds:
                X_train = X.iloc[train_idx].copy()
                X_test = X.iloc[test_idx].copy()
                y_train, y_test = labels[train_idx], labels[test_idx]

                if with_identity:
                    ids_train = features["left_id"].iloc[train_idx]
                    ids_test = features["left_id"].iloc[test_idx]
                    X_train["left_id_target_encoded"] = target_encode(ids_train, y_train, ids_train)
                    X_test["left_id_target_encoded"] = target_encode(ids_train, y_train, ids_test)

                model = SupervisedMatcher("gradient boosting", seed=seed).fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(score_predictions(y_test, pred, int(y_test.sum()))["f1_in_candidates"])

            rows.append({
                "dataset": bench.label,
                "features": "with record identity feature" if with_identity else "similarity only",
                "regime": regime,
                "f1": round(float(np.mean(scores)), 4),
            })
    return rows


def experiment_b(bench, X, labels, seed):
    one_per_field = [c for c in X.columns if c.endswith("_cosine_char") or c.endswith("_equal")]
    rows = []
    for label, columns in (("all features", list(X.columns)), ("one per field", one_per_field)):
        model = FellegiSunter(seed=seed).fit(X[columns])
        pred = model.predict(X[columns])
        result = score_predictions(labels, pred, int(labels.sum()))
        rows.append({
            "dataset": bench.label,
            "feature set": label,
            "n_features": len(columns),
            "precision": result["precision"],
            "recall": result["recall_in_candidates"],
            "f1": result["f1_in_candidates"],
            "em_iterations": model.n_iter_,
        })
    return rows


def main() -> None:
    config = load_config()
    seed = config["random_seed"]
    rows_a, rows_b = [], []

    for key in config["datasets"]:
        bench = load_benchmark(key, config)
        pairs = sorted(tfidf_neighbours(bench, "title", k=20, seed=seed))
        features = build_features(bench, pairs)
        labels = label_pairs(features, bench).to_numpy()
        X = features[feature_columns(features)]

        rows_a.extend(experiment_a(bench, X, features, labels, seed))
        rows_b.extend(experiment_b(bench, X, labels, seed))

    a = pd.DataFrame(rows_a)
    pivot = a.pivot_table(index=["dataset", "features"], columns="regime", values="f1")
    pivot["inflation_pp"] = ((pivot["pair level (random)"]
                              - pivot["entity level (by record)"]) * 100).round(2)
    b = pd.DataFrame(rows_b)

    print("=== Experiment A: does a record identity feature create the leak? ===")
    print(pivot.round(4).to_string())
    print("\n=== Experiment B: Fellegi-Sunter under correlated features ===")
    print(b.to_string(index=False))

    pivot.to_csv(ARTIFACTS / "leakage_mechanism.csv")
    b.to_csv(ARTIFACTS / "fellegi_sunter_independence.csv", index=False)


if __name__ == "__main__":
    main()
