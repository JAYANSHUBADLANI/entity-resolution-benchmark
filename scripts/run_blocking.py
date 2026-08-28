"""Phase 2 entry point: evaluate every blocking scheme on every benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.er.blocking import run_all
from src.er.config import ARTIFACTS, REPORTS, load_config
from src.er.load import load_benchmark


def main() -> None:
    config = load_config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    sections = []

    for key in config["datasets"]:
        bench = load_benchmark(key, config)
        table = run_all(bench)
        table.to_csv(ARTIFACTS / f"blocking_{key}.csv", index=False)

        print(f"\n=== {bench.label} ({bench.character}) ===")
        show = table[["scheme", "candidate_pairs", "candidates_pct_of_space",
                      "pairs_completeness", "pair_quality", "matches_lost"]]
        print(show.to_string(index=False))

        sections.append(f"\n## {bench.label} ({bench.character})\n")
        sections.append(f"Pair space {bench.pair_space:,}, true matches {len(bench.matches):,}.\n")
        sections.append(show.to_markdown(index=False))

    (REPORTS / "02_blocking.md").write_text(
        "# Blocking: what each scheme keeps and what it throws away\n" + "\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
