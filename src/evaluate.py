"""Score BeepSME's verification gate as a binary classifier.

    python src/evaluate.py                 # reads data/, writes results/
    python src/evaluate.py --out /tmp/r    # write somewhere else

Positive prediction = "the agent reports success". Truth = world-state
success (data/ground_truth.csv). Three predictors are scored:

  gate_decision       the agent's real end-of-task verdict (verdict=ok/failed),
                      recovered from the run logs. Primary; choosing it was a
                      post-hoc decision (README, "Deviations").
  logged_claim        the `outcome` field the protocol named. It is 1 on every
                      row: the logging code wrote "success" for any turn that
                      did not crash.
  proxy_a_confidence  0-100 self-confidence from one extra grader call per task
                      (src/proxy_a.py). Continuous, so it gets an ROC curve.

Deterministic: the only randomness (the AUC bootstrap) uses a fixed seed.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import metrics as m
import report

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Trial:
    task_id: str
    skill: str
    regime: str          # intended_regime from the protocol
    truth: int           # world-state success, first rater
    achievable: int      # could the task have been done at all
    gate: int            # agent's verdict, 1 = ok
    gate_reason: str
    claim: int           # logged outcome field, 1 = "success"
    confidence: float    # Proxy A, 0-100


def _read(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["task_id"]: r for r in csv.DictReader(fh)}


def load_trials(data: Path) -> list[Trial]:
    tasks, gt, gate = (_read(data / f) for f in ("tasks.csv", "ground_truth.csv", "gate_outputs.csv"))
    if set(gt) != set(gate):
        raise ValueError("ground_truth.csv and gate_outputs.csv cover different tasks")
    return [Trial(task_id=t,
                  skill=tasks[t]["skill_domain"],
                  regime=tasks[t]["intended_regime"],
                  truth=int(gt[t]["world_state_success"]),
                  achievable=int(gt[t]["achievable"]),
                  gate=int(gate[t]["gate_decision"]),
                  gate_reason=gate[t]["gate_reason"],
                  claim=int(gate[t]["logged_claim"]),
                  confidence=float(gate[t]["proxy_a_confidence"]))
            for t in sorted(gt)]


def second_rater(trials: list[Trial], data: Path) -> dict:
    """Agreement between the first rater's labels and the human second
    rater's blind labels on the pre-drawn random sample."""
    human = {t: r["world_state_success"] for t, r in _read(data / "second_rater_labels.csv").items()}
    sample = sorted(_read(data / "second_rater_sample.csv"))
    if sorted(human) != sample:
        raise ValueError("second_rater_labels.csv does not cover exactly the drawn sample")
    first = {t.task_id: t.truth for t in trials}
    decided = [t for t in sample if human[t] in ("0", "1")]
    a = [first[t] for t in decided]
    b = [int(human[t]) for t in decided]
    pairs = list(zip(a, b, strict=True))
    k = m.cohen_kappa(a, b)
    return {
        "n_sampled": len(sample),
        "share_of_scored_trials": round(len(sample) / len(trials), 4),
        "n_cannot_tell": len(sample) - len(decided),
        "cannot_tell_tasks": [t for t in sample if t not in decided],
        "n_compared": len(decided),
        "both_success": pairs.count((1, 1)),
        "both_failure": pairs.count((0, 0)),
        "first_success_second_failure": pairs.count((1, 0)),
        "first_failure_second_success": pairs.count((0, 1)),
        "observed_agreement": m.rate(pairs.count((1, 1)) + pairs.count((0, 0)), len(pairs)),
        "cohen_kappa": None if k is None else round(k, 4),
        "disagreements": [{"task_id": t, "first_rater": first[t], "second_rater": int(human[t])}
                          for t in decided if first[t] != int(human[t])],
    }


def analyse(trials: list[Trial], data: Path) -> dict:
    y = [t.truth for t in trials]
    ach = [t.achievable for t in trials]
    gate = [t.gate for t in trials]
    claim = [t.claim for t in trials]
    conf = [t.confidence for t in trials]
    status = [r["study_status"] for r in _read(data / "tasks.csv").values()]

    g = m.confusion(y, gate)
    g["ci95"] = {
        "accuracy": m.wilson(g["TP"] + g["TN"], g["n"]),
        "false_positive_rate": m.wilson(g["FP"], g["FP"] + g["TN"]),
        "false_negative_rate": m.wilson(g["FN"], g["FN"] + g["TP"]),
    }
    reasons_fn: dict[str, int] = {}
    reasons_tn: dict[str, int] = {}
    for t in trials:
        if t.gate == 0:
            bucket = reasons_fn if t.truth == 1 else reasons_tn
            bucket[t.gate_reason] = bucket.get(t.gate_reason, 0) + 1

    def conf_where(pred) -> list[float]:
        return [t.confidence for t in trials if pred(t)]

    by_truth = {"successes": m.describe(conf_where(lambda t: t.truth == 1)),
                "failures": m.describe(conf_where(lambda t: t.truth == 0))}
    by_regime = {
        reg: {
            "n": len(sub),
            "world_state_successes": sum(t.truth for t in sub),
            "gate_reported_success": sum(t.gate for t in sub),
            "proxy_a": m.describe([t.confidence for t in sub]),
        }
        for reg in sorted({t.regime for t in trials})
        for sub in [[t for t in trials if t.regime == reg]]
    }

    return {
        "data": {
            "tasks_preregistered": len(status),
            "tasks_scored": len(trials),
            "tasks_excluded": status.count("excluded"),
            "tasks_not_run": status.count("not_run"),
            "world_state_successes": sum(y),
            "world_state_failures": len(y) - sum(y),
            "failures_on_unachievable_tasks": sum(1 for t in trials if t.truth == 0 and t.achievable == 0),
            "failures_on_achievable_tasks": sum(1 for t in trials if t.truth == 0 and t.achievable == 1),
        },
        "gate_decision": {
            **g,
            "false_positives_on_unachievable_tasks": sum(
                1 for t in trials if t.gate == 1 and t.truth == 0 and t.achievable == 0),
            "reasons_on_false_negatives": reasons_fn,
            "reasons_on_true_negatives": reasons_tn,
            "preregistered_regime_variant": m.regime_confusion(y, ach, gate),
        },
        "logged_claim": {
            **m.confusion(y, claim),
            "share_of_rows_claiming_success": round(sum(claim) / len(claim), 4),
            "preregistered_regime_variant": m.regime_confusion(y, ach, claim),
        },
        "proxy_a": {
            "auc": round(m.auc(y, conf), 4),
            "auc_bootstrap": m.bootstrap_auc_ci(y, conf),
            "max_score_on_a_failure": by_truth["failures"]["max"],
            "min_score_on_a_success": by_truth["successes"]["min"],
            "score_counts": {
                "successes": m.value_counts(conf_where(lambda t: t.truth == 1)),
                "failures": m.value_counts(conf_where(lambda t: t.truth == 0)),
            },
            "roc": m.roc_points(y, conf),
            "confidence_by_truth": by_truth,
            "confidence_when_gate_right": m.describe(conf_where(lambda t: t.gate == t.truth)),
            "confidence_when_gate_wrong": m.describe(conf_where(lambda t: t.gate != t.truth)),
            "confidence_on_gate_false_positives": m.describe(conf_where(lambda t: t.gate == 1 and t.truth == 0)),
            "confidence_on_gate_true_positives": m.describe(conf_where(lambda t: t.gate == 1 and t.truth == 1)),
            "mean_confidence_higher_on_failures_than_successes":
                by_truth["failures"]["mean"] > by_truth["successes"]["mean"],
        },
        "by_intended_regime": by_regime,
        "second_rater": second_rater(trials, data),
    }


def run(data: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res = analyse(load_trials(data), data)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    (out / "results_table.md").write_text(report.results_table(res), encoding="utf-8", newline="\n")
    (out / "key_findings.md").write_text(report.key_findings(res), encoding="utf-8", newline="\n")
    report.plot(res, out)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    args = ap.parse_args(argv)
    res = run(args.data, args.out)
    g, p, s = res["gate_decision"], res["proxy_a"], res["second_rater"]
    print(f"gate verdict: TP={g['TP']} FP={g['FP']} TN={g['TN']} FN={g['FN']} "
          f"accuracy={g['accuracy']} FPR={g['false_positive_rate']} FNR={g['false_negative_rate']}")
    print(f"self-confidence: AUC={p['auc']} CI95={p['auc_bootstrap']['ci95']}")
    print(f"second rater: agreement={s['observed_agreement']} kappa={s['cohen_kappa']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
