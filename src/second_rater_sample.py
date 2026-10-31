"""Draw the trials for the human second rating.

    python src/second_rater_sample.py

Simple random sample of 10 of the 31 scored trials (32%, above the protocol's
20% minimum), drawn with a fixed seed BEFORE any human label existed, and
written to data/second_rater_sample.csv. Re-running gives the same sample.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEED = 20261031  # the date the sample was drawn
N = 10


def draw(task_ids: list[str], n: int = N, seed: int = SEED) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(task_ids), n))


def main() -> int:
    with (REPO / "data" / "ground_truth.csv").open(encoding="utf-8", newline="") as fh:
        scored = [r["task_id"] for r in csv.DictReader(fh)]
    sample = draw(scored)
    with (REPO / "data" / "second_rater_sample.csv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("task_id\n" + "".join(f"{t}\n" for t in sample))
    print(f"seed={SEED} n={len(sample)}: {', '.join(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
