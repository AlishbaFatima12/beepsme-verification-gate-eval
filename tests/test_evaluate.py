"""Tests: metric maths on hand-made examples, and that the committed results
match what the code produces from the committed data."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import derive  # noqa: E402
import evaluate  # noqa: E402

# ---------------------------------------------------------------------------
# Metric maths on a tiny example with known answers
#   truth: 1 1 1 0 0 0 0 1
#   pred:  1 1 0 1 0 0 1 1   -> TP=3 FN=1 FP=2 TN=2
# ---------------------------------------------------------------------------

Y = [1, 1, 1, 0, 0, 0, 0, 1]
P = [1, 1, 0, 1, 0, 0, 1, 1]


def test_confusion_counts_and_rates():
    c = evaluate.confusion(Y, P)
    assert (c["TP"], c["FP"], c["TN"], c["FN"]) == (3, 2, 2, 1)
    assert c["accuracy"] == 0.625          # 5/8
    assert c["precision"] == 0.6           # 3/5
    assert c["recall"] == 0.75             # 3/4
    assert c["specificity"] == 0.5         # 2/4
    assert c["false_positive_rate"] == 0.5  # 2/4
    assert c["false_negative_rate"] == 0.25  # 1/4


def test_constant_predictor_has_undefined_precision_not_zero():
    c = evaluate.confusion([1, 0], [0, 0])
    assert c["precision"] is None
    assert c["false_positive_rate"] == 0.0


def test_regime_variant_uses_achievable_for_negatives():
    # pred 0 on an achievable failure counts as over-deferral, on an
    # unachievable one as a correct deferral.
    r = evaluate.regime_confusion(truth := [0, 0, 1], [1, 0, 1], [0, 0, 0])
    assert truth == [0, 0, 1]
    assert (r["FN_over_deferral"], r["TN_correct_deferral"]) == (2, 1)


def test_auc_known_values():
    assert evaluate.auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert evaluate.auc([0, 0, 1, 1], [4, 3, 2, 1]) == 0.0
    assert evaluate.auc([0, 1], [5, 5]) == 0.5
    # one success (3) vs failures (1, 3, 5): wins 1 + tie 0.5 + loss 0 = 1.5/3
    assert evaluate.auc([1, 0, 0, 0], [3, 1, 3, 5]) == 0.5


def test_auc_equals_trapezoid_area_of_roc_points():
    y = [1, 0, 1, 1, 0, 0, 1, 0]
    s = [15, 15, 35, 72, 0, 5, 15, 15]
    pts = evaluate.roc_points(y, s)
    area = sum((b["fpr"] - a["fpr"]) * (a["tpr"] + b["tpr"]) / 2 for a, b in zip(pts, pts[1:]))
    assert round(area, 4) == round(evaluate.auc(y, s), 4)


def test_auc_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    y = [1, 0, 1, 1, 0, 0, 1, 0, 1]
    s = [15, 15, 35, 72, 0, 5, 15, 15, 45]
    assert abs(sk.roc_auc_score(y, s) - evaluate.auc(y, s)) < 1e-12


def test_wilson_interval():
    lo, hi = evaluate.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert evaluate.wilson(0, 0) is None


def test_cohen_kappa_known_values():
    # perfect agreement with both classes present
    assert evaluate.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # textbook 2x2: both-yes 20, yes/no 5, no/yes 10, both-no 15 (n=50)
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    # po = 0.70, pe = 0.5*0.6 + 0.5*0.4 = 0.50 -> kappa = 0.40
    assert evaluate.cohen_kappa(a, b) == pytest.approx(0.40)
    assert evaluate.cohen_kappa([1, 1], [1, 1]) is None


def test_second_rater_sample_is_reproducible():
    import second_rater_sample
    scored = sorted(_csv("ground_truth.csv"))
    committed = sorted(_csv("second_rater_sample.csv"))
    assert second_rater_sample.draw(scored) == committed


def test_bootstrap_is_deterministic():
    y = [1, 0, 1, 1, 0, 0, 1, 0]
    s = [15, 15, 35, 72, 0, 5, 15, 15]
    a = evaluate.bootstrap_auc_ci(y, s, n_boot=500)
    b = evaluate.bootstrap_auc_ci(y, s, n_boot=500)
    assert a == b


# ---------------------------------------------------------------------------
# Verdict-line parsing
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


def test_parse_refuses_ambiguous_pairing():
    with pytest.raises(ValueError):
        derive.parse_verdict_lines([
            "a.log:1:[1/1] EVAL-HW-01 (homework/happy_path) cap=240s",
            "a.log:2:[task] verdict=ok steps=1",
            "a.log:3:[task] verdict=ok steps=1",
        ])
    with pytest.raises(ValueError):
        derive.parse_verdict_lines([
            "a.log:1:[1/2] EVAL-HW-01 (homework/happy_path) cap=240s",
            "a.log:2:[2/2] EVAL-HW-02 (homework/happy_path) cap=240s",
            "a.log:3:[task] verdict=ok steps=1",
        ])


# ---------------------------------------------------------------------------
# The committed data, results and README agree with the code
# ---------------------------------------------------------------------------

def _csv(name: str) -> dict[str, dict]:
    with (REPO / "data" / name).open(encoding="utf-8", newline="") as fh:
        return {r["task_id"]: r for r in csv.DictReader(fh)}


def test_gate_outputs_match_raw_logs():
    """gate_outputs.csv is derivable from the raw verdict lines and turn log."""
    final = derive.load_final_verdicts(REPO / "data" / "raw" / "verdict_lines.txt")
    turns = {}
    for line in (REPO / "data" / "raw" / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        turns[t["task_id"]] = t
    for tid, row in _csv("gate_outputs.csv").items():
        v = final[tid]
        assert int(row["gate_decision"]) == (1 if v["verdict"] == "ok" else 0), tid
        assert row["gate_reason"] == v["reason"], tid
        assert int(row["n_attempts"]) == v["n_attempts"], tid
        assert int(row["logged_claim"]) == (1 if turns[tid]["logged_outcome"] == "success" else 0), tid
        assert float(row["proxy_a_confidence"]) == float(turns[tid]["proxy_a"]), tid


def test_data_files_cover_the_same_scored_tasks():
    scored = {t for t, r in _csv("tasks.csv").items() if r["study_status"] == "scored"}
    assert scored == set(_csv("ground_truth.csv")) == set(_csv("gate_outputs.csv"))
    assert len(_csv("tasks.csv")) == 44


def test_results_json_matches_fresh_computation():
    fresh = json.loads(json.dumps(evaluate.analyse(evaluate.load_rows())))
    committed = json.loads((REPO / "results" / "results.json").read_text(encoding="utf-8"))
    assert fresh == committed


def test_readme_results_block_matches_generated_table():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- results:start -->\n(.*?)<!-- results:end -->", readme, re.S)
    assert m, "README is missing the results markers"
    res = json.loads((REPO / "results" / "results.json").read_text(encoding="utf-8"))
    assert m.group(1) == evaluate.results_table(res)
