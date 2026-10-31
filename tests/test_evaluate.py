"""Metric maths on hand-made examples with known answers, and a check that
the committed data, results and README all follow from the code."""
from __future__ import annotations

import csv
import json
import re
import sys
from itertools import pairwise
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import derive  # noqa: E402
import evaluate  # noqa: E402
import metrics  # noqa: E402
import report  # noqa: E402
import second_rater_sample  # noqa: E402

# ---------------------------------------------------------------------------
# metrics.py
#   truth: 1 1 1 0 0 0 0 1
#   pred:  1 1 0 1 0 0 1 1   -> TP=3 FN=1 FP=2 TN=2
# ---------------------------------------------------------------------------

Y = [1, 1, 1, 0, 0, 0, 0, 1]
P = [1, 1, 0, 1, 0, 0, 1, 1]


def test_confusion_counts_and_rates():
    c = metrics.confusion(Y, P)
    assert (c["TP"], c["FP"], c["TN"], c["FN"]) == (3, 2, 2, 1)
    assert c["accuracy"] == 0.625
    assert c["precision"] == 0.6
    assert c["recall"] == 0.75
    assert c["specificity"] == 0.5
    assert c["false_positive_rate"] == 0.5
    assert c["false_negative_rate"] == 0.25


def test_undefined_rate_is_none_not_zero():
    c = metrics.confusion([1, 0], [0, 0])
    assert c["precision"] is None
    assert c["false_positive_rate"] == 0.0


def test_regime_variant_uses_achievable_for_negatives():
    r = metrics.regime_confusion([0, 0, 1], [1, 0, 1], [0, 0, 0])
    assert (r["FN_over_deferral"], r["TN_correct_deferral"]) == (2, 1)


def test_auc_known_values():
    assert metrics.auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert metrics.auc([0, 0, 1, 1], [4, 3, 2, 1]) == 0.0
    assert metrics.auc([0, 1], [5, 5]) == 0.5
    # success 3 vs failures 1, 3, 5: win + tie + loss = 1.5 of 3
    assert metrics.auc([1, 0, 0, 0], [3, 1, 3, 5]) == 0.5


def test_auc_equals_trapezoid_area_of_roc_points():
    y = [1, 0, 1, 1, 0, 0, 1, 0]
    s = [15, 15, 35, 72, 0, 5, 15, 15]
    pts = metrics.roc_points(y, s)
    area = sum((b["fpr"] - a["fpr"]) * (a["tpr"] + b["tpr"]) / 2 for a, b in pairwise(pts))
    assert round(area, 4) == round(metrics.auc(y, s), 4)


def test_auc_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    y = [1, 0, 1, 1, 0, 0, 1, 0, 1]
    s = [15, 15, 35, 72, 0, 5, 15, 15, 45]
    assert abs(sk.roc_auc_score(y, s) - metrics.auc(y, s)) < 1e-12


def test_wilson_interval():
    lo, hi = metrics.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert metrics.wilson(0, 0) is None


def test_cohen_kappa_known_values():
    assert metrics.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # both-yes 20, yes/no 5, no/yes 10, both-no 15: po 0.70, pe 0.50 -> 0.40
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert metrics.cohen_kappa(a, b) == pytest.approx(0.40)
    assert metrics.cohen_kappa([1, 1], [1, 1]) is None


def test_bootstrap_is_deterministic():
    y = [1, 0, 1, 1, 0, 0, 1, 0]
    s = [15, 15, 35, 72, 0, 5, 15, 15]
    assert metrics.bootstrap_auc_ci(y, s, n_boot=500) == metrics.bootstrap_auc_ci(y, s, n_boot=500)


# ---------------------------------------------------------------------------
# derive.py: pairing log headers with verdict lines
# ---------------------------------------------------------------------------

def test_parse_pairs_headers_and_verdicts_and_keeps_last_attempt():
    lines = [
        "a.log:1:[1/2] EVAL-HW-01 (homework/happy_path) cap=240s",
        "a.log:5:[task] verdict=failed steps=1 reason=ai_http_413",
        "a.log:9:[2/2] EVAL-HW-02 (homework/happy_path) cap=240s",
        "a.log:12:[task] verdict=ok steps=1",
        "b.log:3:[1/1] EVAL-HW-01 (homework/happy_path) cap=240s",
        "b.log:7:[task] verdict=ok steps=2",
    ]
    final = derive.final_attempts(derive.parse_verdict_lines(lines))
    assert final["EVAL-HW-01"]["verdict"] == "ok"
    assert final["EVAL-HW-01"]["n_attempts"] == 2
    assert final["EVAL-HW-02"]["steps"] == 1


@pytest.mark.parametrize("lines", [
    ["a.log:1:[1/1] EVAL-HW-01 (homework/happy_path) cap=240s",
     "a.log:2:[task] verdict=ok steps=1",
     "a.log:3:[task] verdict=ok steps=1"],
    ["a.log:1:[1/2] EVAL-HW-01 (homework/happy_path) cap=240s",
     "a.log:2:[2/2] EVAL-HW-02 (homework/happy_path) cap=240s",
     "a.log:3:[task] verdict=ok steps=1"],
])
def test_parse_refuses_ambiguous_pairing(lines):
    with pytest.raises(ValueError):
        derive.parse_verdict_lines(lines)


# ---------------------------------------------------------------------------
# The committed data, results and README follow from the code
# ---------------------------------------------------------------------------

def _csv(name: str) -> dict[str, dict]:
    with (REPO / "data" / name).open(encoding="utf-8", newline="") as fh:
        return {r["task_id"]: r for r in csv.DictReader(fh)}


def test_gate_outputs_follow_from_raw_log_lines():
    final = derive.load_final_verdicts(REPO / "data" / "raw" / "verdict_lines.txt")
    turns = {json.loads(x)["task_id"]: json.loads(x)
             for x in (REPO / "data" / "raw" / "turns.jsonl").read_text(encoding="utf-8").splitlines()}
    for tid, row in _csv("gate_outputs.csv").items():
        v = final[tid]
        assert int(row["gate_decision"]) == (v["verdict"] == "ok"), tid
        assert row["gate_reason"] == v["reason"], tid
        assert int(row["n_attempts"]) == v["n_attempts"], tid
        assert int(row["logged_claim"]) == (turns[tid]["logged_outcome"] == "success"), tid
        assert float(row["proxy_a_confidence"]) == float(turns[tid]["proxy_a"]), tid


def test_data_files_cover_the_same_scored_tasks():
    tasks = _csv("tasks.csv")
    scored = {t for t, r in tasks.items() if r["study_status"] == "scored"}
    assert len(tasks) == 44
    assert scored == set(_csv("ground_truth.csv")) == set(_csv("gate_outputs.csv"))


def test_second_rater_sample_is_reproducible():
    assert second_rater_sample.draw(sorted(_csv("ground_truth.csv"))) == sorted(_csv("second_rater_sample.csv"))


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("results")
    evaluate.run(REPO / "data", out)
    return out


@pytest.mark.parametrize("name", ["results.json", "results_table.md", "key_findings.md"])
def test_committed_results_match_a_fresh_run(fresh, name):
    assert (fresh / name).read_text(encoding="utf-8") == (REPO / "results" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("block, render", [("findings", report.key_findings),
                                           ("results", report.results_table)])
def test_readme_blocks_match_generated_text(block, render):
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    found = re.search(rf"<!-- {block}:start -->\n(.*?)<!-- {block}:end -->", readme, re.S)
    assert found, f"README is missing the {block} markers"
    res = json.loads((REPO / "results" / "results.json").read_text(encoding="utf-8"))
    assert found.group(1) == render(res)
