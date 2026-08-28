"""Phase 4b: what the 0.61 F1 on the dirty benchmark is actually failing on.

An aggregate score says a system is wrong without saying how. This splits every missed match into
the stage that lost it, which matters because the fixes are different: a match that blocking never
proposed cannot be recovered by a better model, and a match scored just under the cut is a
threshold decision, not a modelling failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.er.blocking import tfidf_neighbours
from src.er.config import ARTIFACTS, REPORTS, load_config
from src.er.features import build_features, label_pairs
from src.er.load import load_benchmark
from src.er.matchers import SupervisedMatcher, feature_columns
from src.er.resolve import degree_capped_greedy

THRESHOLD = 0.25  # the tuned cut from the decision phase, reused so the analysis matches it
KEY = "amazon_google"


def out_of_fold_scores(X, labels, groups, seed):
    scores = np.zeros(len(X))
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, labels, groups=groups):
        model = SupervisedMatcher("gradient boosting", seed=seed).fit(
            X.iloc[train_idx], labels[train_idx])
        scores[test_idx] = model.score(X.iloc[test_idx])
    return scores


def main() -> None:
    config = load_config()
    seed = config["random_seed"]
    bench = load_benchmark(KEY, config)

    pairs = sorted(tfidf_neighbours(bench, "title", k=20, seed=seed))
    features = build_features(bench, pairs)
    labels = label_pairs(features, bench).to_numpy()
    X = features[feature_columns(features)]

    scores = out_of_fold_scores(X, labels, features["left_id"], seed)
    decision = degree_capped_greedy(features[["left_id", "right_id"]], scores, THRESHOLD, 1, 1)

    candidate_set = set(zip(features["left_id"], features["right_id"]))
    truth = set(zip(bench.matches["left_id"], bench.matches["right_id"]))
    decided = {p for p, d in zip(zip(features["left_id"], features["right_id"]), decision) if d}

    lost_in_blocking = truth - candidate_set
    scored = {p: s for p, s in zip(zip(features["left_id"], features["right_id"]), scores)}
    missed_below_cut = {p for p in (truth & candidate_set) - decided if scored[p] < THRESHOLD}
    missed_by_constraint = (truth & candidate_set) - decided - missed_below_cut

    lines = ["# Where the dirty benchmark loses matches", ""]
    lines.append(f"Ground truth matches: {len(truth):,}")
    lines.append(f"- never proposed by blocking: {len(lost_in_blocking):,} "
                 f"({100*len(lost_in_blocking)/len(truth):.1f} percent), unreachable by any model")
    lines.append(f"- proposed but scored below the {THRESHOLD} cut: {len(missed_below_cut):,} "
                 f"({100*len(missed_below_cut)/len(truth):.1f} percent)")
    lines.append(f"- scored above the cut but dropped by the 1:1 constraint: "
                 f"{len(missed_by_constraint):,} "
                 f"({100*len(missed_by_constraint)/len(truth):.1f} percent)")
    lines.append(f"- correctly returned: {len(truth & decided):,} "
                 f"({100*len(truth & decided)/len(truth):.1f} percent)")
    lines.append("")

    left = bench.left.set_index("record_id")
    right = bench.right.set_index("record_id")

    def show(pair_list, title, n=8, with_score=True):
        lines.append(f"\n## {title}\n")
        rows = []
        for l, r in pair_list[:n]:
            rows.append({
                "score": round(float(scored.get((l, r), float("nan"))), 3) if with_score else "",
                "amazon": str(left.loc[l, "title"])[:70],
                "google": str(right.loc[r, "title"])[:70],
                "amazon_price": left.loc[l, "price"],
                "google_price": right.loc[r, "price"],
            })
        lines.append(pd.DataFrame(rows).to_markdown(index=False))

    false_positives = sorted(decided - truth, key=lambda p: -scored[p])
    show(false_positives, "Highest scoring false positives")

    near_misses = sorted(missed_below_cut, key=lambda p: -scored[p])
    show(near_misses, "True matches that scored just below the cut")

    show(sorted(lost_in_blocking), "True matches blocking never proposed", with_score=False)

    # Feature level contrast, to say something about the failures beyond eyeballing them.
    frame = X.copy()
    frame["outcome"] = np.where(labels == 1, np.where(decision == 1, "true positive", "false negative"),
                                np.where(decision == 1, "false positive", "true negative"))
    interesting = [c for c in ["title_cosine_char", "title_jaccard", "price_rel_diff",
                               "manufacturer_either_missing", "description_cosine_word"]
                   if c in frame.columns]
    contrast = frame.groupby("outcome")[interesting].mean().round(3)
    lines.append("\n## Mean feature values by outcome\n")
    lines.append(contrast.to_markdown())

    # The false positives above do not read like errors. Several are the same product with a
    # trademark symbol or a marketing suffix attached, at an identical price. That is a claim
    # about the labels, not about the model, so it is counted rather than asserted: how many
    # false positives agree closely on title and agree exactly on a non zero price.
    fp_frame = features.loc[[i for i, (l, r) in enumerate(zip(features["left_id"], features["right_id"]))
                             if (l, r) in (decided - truth)]].copy()
    fp_frame["score"] = [scored[(l, r)] for l, r in zip(fp_frame["left_id"], fp_frame["right_id"])]
    left_price = pd.to_numeric(left["price"], errors="coerce")
    right_price = pd.to_numeric(right["price"], errors="coerce")
    fp_frame["lp"] = [left_price.get(l) for l in fp_frame["left_id"]]
    fp_frame["rp"] = [right_price.get(r) for r in fp_frame["right_id"]]
    suspicious = fp_frame[(fp_frame["title_cosine_char"] >= 0.8)
                          & (fp_frame["lp"] > 0) & (fp_frame["lp"] == fp_frame["rp"])]
    lines.append("\n## How many false positives look like missing labels\n")
    lines.append(f"Of {len(fp_frame):,} false positives, {len(suspicious):,} "
                 f"({100*len(suspicious)/max(len(fp_frame),1):.1f} percent) agree on title at a "
                 f"character cosine of 0.8 or higher and carry an identical non zero price on both "
                 f"sides. These are flagged for manual review rather than counted as correct: the "
                 f"benchmark's precision figure is a lower bound if they are genuine matches, and "
                 f"no reported number in this project assumes they are.")

    (REPORTS / "03_error_analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"\nwrote {REPORTS / '03_error_analysis.md'}")


if __name__ == "__main__":
    main()
