"""Refresh the generated blocks in README.md from results/results.json.

    python tools/sync_readme.py

Run after `python src/evaluate.py`. The tests fail if the README and the
results disagree, so this is the only way numbers enter the README.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import report  # noqa: E402


def main() -> int:
    res = json.loads((REPO / "results" / "results.json").read_text(encoding="utf-8"))
    readme_path = REPO / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    for block, render in (("findings", report.key_findings), ("results", report.results_table)):
        pattern = re.compile(rf"(<!-- {block}:start -->\n).*?(<!-- {block}:end -->)", re.S)
        if not pattern.search(text):
            raise SystemExit(f"README.md has no {block} markers")
        text = pattern.sub(lambda m, r=render: m.group(1) + r(res) + m.group(2), text)
    readme_path.write_text(text, encoding="utf-8", newline="\n")
    print("README.md blocks refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
