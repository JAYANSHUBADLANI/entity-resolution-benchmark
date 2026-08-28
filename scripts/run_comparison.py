"""Phase 4c: put this project's numbers next to the published ones, and say why they differ.

The published figures are not a target to beat. They are measured on a supplied candidate set
with a far friendlier class balance, so a straight comparison of the two headline numbers would
be misleading in this project's favour on one dataset and against it on the other. Both are
printed with the balance they were measured at, which is the only way the comparison means
anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.er.blocking import tfidf_neighbours
from src.er.config import ARTIFACTS, REPORTS, ROOT, load_config
from src.er.features import label_pairs
from src.er.load import load_benchmark


def main() -> None:
    config = load_config()
    published = yaml.safe_load((ROOT / "config" / "published_baselines.yaml").read_text())
    decisions = pd.read_csv(ARTIFACTS / "decision_rules.csv")

    rows = []
    for key in config["datasets"]:
        bench = load_benchmark(key, config)
        pairs = sorted(tfidf_neighbours(bench, "title", k=20, seed=config["random_seed"]))
        positives = int(label_pairs(pd.DataFrame(pairs, columns=["left_id", "right_id"]),
                                    bench).sum())
        mine = decisions[decisions["dataset"] == bench.label]
        best = mine.loc[mine["f1"].idxmax()]
        pub = published["results"][key]

        rows.append({
            "dataset": bench.label,
            "this project, best rule": best["rule"],
            "this project F1": round(float(best["f1"]), 3),
            "candidate pairs here": len(pairs),
            "positive rate here %": round(100 * positives / len(pairs), 1),
            "published Magellan F1": pub["f1"]["Magellan"] / 100,
            "published best DL F1": max(pub["f1"].values()) / 100,
            "published pairs": pub["labelled_pairs"],
            "published positive rate %": pub["positive_rate_pct"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(ARTIFACTS / "published_comparison.csv", index=False)

    text = [
        "# This project next to the published figures",
        "",
        f"Published source: {published['source']['citation']}.",
        f"Tables used: {published['source']['tables']}. Transcribed {published['source']['transcribed']}.",
        "",
        table.to_markdown(index=False),
        "",
        "## Why these columns are not the same measurement",
        "",
        published["comparability"]["why_not_like_for_like"].strip(),
        "",
        f"Published protocol: {published['comparability']['protocol']}",
        "",
        "The reading that survives all of that: on the clean citation benchmark a classical "
        "pipeline lands in the same place as both the published classical baseline and the "
        "published deep models, which agree with each other there. On the dirty product "
        "benchmark the published deep model is well ahead of the published classical baseline, "
        "and the pipeline here sits between the two while being measured on a harder candidate "
        "set. That is consistent with the usual finding, that deep models earn their cost on "
        "dirty text and not on structured records, but it is not evidence of beating anything.",
    ]
    (REPORTS / "04_published_comparison.md").write_text("\n".join(text) + "\n")
    print(table.to_string(index=False))
    print(f"\nwrote {REPORTS / '04_published_comparison.md'}")


if __name__ == "__main__":
    main()
