"""Profile both benchmarks before any modelling, and reconcile the ground truth against them.

The checks here exist because the labels in these files are not automatically trustworthy. A
mapping row that points at an id which does not exist in either source is a broken label, and a
source id that appears twice means the "one record per entity per source" assumption is already
false. Both are checked rather than assumed, and whatever is found is reported.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import REPORTS, load_config
from .load import load_benchmark


def field_profile(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        if field not in frame.columns:
            rows.append({"field": field, "present": False})
            continue
        col = frame[field]
        text = col.astype(str).where(col.notna(), "")
        rows.append({
            "field": field,
            "present": True,
            "missing": int(col.isna().sum()),
            "missing_pct": round(100 * col.isna().mean(), 2),
            "distinct": int(col.nunique(dropna=True)),
            "mean_chars": round(text.str.len().mean(), 1),
            "max_chars": int(text.str.len().max()),
        })
    return pd.DataFrame(rows)


def reconcile(bench) -> dict:
    left_ids = set(bench.left["record_id"])
    right_ids = set(bench.right["record_id"])
    m = bench.matches

    orphan_left = sorted(set(m["left_id"]) - left_ids)
    orphan_right = sorted(set(m["right_id"]) - right_ids)
    dup_pairs = int(m.duplicated(subset=["left_id", "right_id"]).sum())

    # Cardinality: these benchmarks are usually described as one to one, which is worth testing
    # rather than believing, because it decides whether the matcher may emit several matches per
    # record and whether a clustering step is needed at all.
    left_deg = m.groupby("left_id").size()
    right_deg = m.groupby("right_id").size()

    return {
        "left_records": len(bench.left),
        "right_records": len(bench.right),
        "duplicate_left_ids": int(bench.left["record_id"].duplicated().sum()),
        "duplicate_right_ids": int(bench.right["record_id"].duplicated().sum()),
        "labelled_matches": len(m),
        "duplicate_label_rows": dup_pairs,
        "labels_pointing_at_missing_left_id": len(orphan_left),
        "labels_pointing_at_missing_right_id": len(orphan_right),
        "pair_space": bench.pair_space,
        "match_rate_pct": round(100 * bench.match_rate, 5),
        "one_pair_per_2700": round(1 / bench.match_rate),
        "left_max_matches": int(left_deg.max()),
        "right_max_matches": int(right_deg.max()),
        "left_records_with_multiple_matches": int((left_deg > 1).sum()),
        "right_records_with_multiple_matches": int((right_deg > 1).sum()),
        "is_strictly_one_to_one": bool(left_deg.max() == 1 and right_deg.max() == 1),
    }


def main() -> None:
    config = load_config()
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    lines = []

    for key in config["datasets"]:
        bench = load_benchmark(key, config)
        rec = reconcile(bench)
        summary[key] = rec

        lines.append(f"\n## {bench.label} ({bench.character})\n")
        lines.append(f"- left records: {rec['left_records']:,}")
        lines.append(f"- right records: {rec['right_records']:,}")
        lines.append(f"- pair space: {rec['pair_space']:,}")
        lines.append(f"- labelled matches: {rec['labelled_matches']:,} "
                     f"({rec['match_rate_pct']}%, one in {rec['one_pair_per_2700']:,} pairs)")
        lines.append(f"- duplicate ids in sources: left {rec['duplicate_left_ids']}, "
                     f"right {rec['duplicate_right_ids']}")
        lines.append(f"- labels pointing at an id that does not exist: "
                     f"left {rec['labels_pointing_at_missing_left_id']}, "
                     f"right {rec['labels_pointing_at_missing_right_id']}")
        lines.append(f"- strictly one to one: {rec['is_strictly_one_to_one']} "
                     f"(max matches per left record {rec['left_max_matches']}, "
                     f"per right record {rec['right_max_matches']}; "
                     f"{rec['left_records_with_multiple_matches']} left and "
                     f"{rec['right_records_with_multiple_matches']} right records have more than one)")

        for side, frame in (("left", bench.left), ("right", bench.right)):
            prof = field_profile(frame, bench.compare_fields)
            lines.append(f"\n{side} field profile\n")
            lines.append(prof.to_markdown(index=False))

        print(f"{bench.label}: {rec['left_records']:,} x {rec['right_records']:,} = "
              f"{rec['pair_space']:,} pairs, {rec['labelled_matches']:,} matches, "
              f"one to one = {rec['is_strictly_one_to_one']}")

    (REPORTS / "01_data_profile.md").write_text(
        "# Data profile and ground truth reconciliation\n" + "\n".join(lines) + "\n")
    (REPORTS / "01_data_profile.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {REPORTS / '01_data_profile.md'}")


if __name__ == "__main__":
    main()
