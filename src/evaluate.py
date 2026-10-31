"""Score BeepSME's verification gate as a binary classifier.

    python src/evaluate.py

Reads data/*.csv, writes results/results.json, results/results_table.md,
results/confusion_matrix.png and results/roc_curve.png. Deterministic: the
only randomness (bootstrap CI for the AUC) uses a fixed seed.

Positive prediction = "the agent reports success". Truth = world-state
success (data/ground_truth.csv). Three predictors are scored:

  gate_decision       the agent's real end-of-task verdict (verdict=ok/failed),
                      recovered from the run logs. PRIMARY (a post-hoc choice;
                      see README "Deviations").
  logged_claim        the `outcome` field the pre-registered analysis used.
                      It is 1 on every row because the logging code wrote
                      "success" whenever a turn did not crash.
  proxy_a_confidence  0-100 self-confidence from one extra grader call
                      (src/proxy_a.py). Continuous, so it gets an ROC curve.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RESULTS = REPO / "results"
SEED = 20260820
N_BOOT = 10_000


# ---------------------------------------------------------------------------
# Metric functions (pure; unit-tested in tests/test_evaluate.py)
# ---------------------------------------------------------------------------

def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def confusion(y_true: list[int], y_pred: list[int]) -> dict:
    """Standard 2x2 plus the rates named in the README.

    FP = reported success, task failed (over-confidence).
    FN = reported failure, task succeeded (over-deferral).
    """
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if p == 1 and t == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if p == 1 and t == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if p == 0 and t == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if p == 0 and t == 1)
    n = tp + fp + tn + fn
    return {
        "n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": _rate(tp + tn, n),
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "specificity": _rate(tn, tn + fp),
        "false_positive_rate": _rate(fp, fp + tn),
        "false_negative_rate": _rate(fn, fn + tp),
    }


def regime_confusion(y_true: list[int], achievable: list[int], y_pred: list[int]) -> dict:
    """The pre-registered variant (protocol README): a reported failure counts
    as over-deferral when the task was ACHIEVABLE, even if it was not in fact
    completed; it is a correct deferral only when the task was not achievable."""
    tp = fp = tn = fn = 0
    for t, a, p in zip(y_true, achievable, y_pred, strict=True):
        if p == 1:
            tp += t == 1
            fp += t == 0
        else:
            fn += a == 1
            tn += a == 0
    return {"TP": tp, "FP_over_confidence": fp, "TN_correct_deferral": tn,
            "FN_over_deferral": fn,
            "over_confidence_rate": _rate(fp, tp + fp),
            "over_deferral_rate": _rate(fn, tn + fn)}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 4), round(centre + half, 4)]


def roc_points(y_true: list[int], score: list[float]) -> list[dict]:
    """ROC points for the rule "report success if score >= threshold",
    one point per distinct score plus the all-negative point."""
    pos = sum(y_true)
    neg = len(y_true) - pos
    pts = []
    for thr in [math.inf, *sorted(set(score), reverse=True)]:
        tp = sum(1 for t, s in zip(y_true, score, strict=True) if s >= thr and t == 1)
        fp = sum(1 for t, s in zip(y_true, score, strict=True) if s >= thr and t == 0)
        pts.append({"threshold": "inf" if thr == math.inf else thr,
                    "TP": tp, "FP": fp, "FN": pos - tp, "TN": neg - fp,
                    "tpr": round(tp / pos, 4), "fpr": round(fp / neg, 4)})
    return pts


def auc(y_true: list[int], score: list[float]) -> float:
    """Probability that a random success outscores a random failure (ties
    count one half). Equal to the trapezoidal area under roc_points."""
    pos = [s for t, s in zip(y_true, score, strict=True) if t == 1]
    neg = [s for t, s in zip(y_true, score, strict=True) if t == 0]
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_auc_ci(y_true: list[int], score: list[float], n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.min() == yb.max():
            continue  # a resample with one class has no AUC
        vals.append(auc(yb.tolist(), s[idx].tolist()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_boot": n_boot, "n_valid": len(vals), "seed": seed}


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa for two raters' binary labels on the same items."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return None  # both raters constant and identical; kappa undefined
    return (po - pe) / (1 - pe)


def second_rater(rows: list[dict]) -> dict:
    """Agreement between the first (AI) rater's labels and the human second
    rater's blind labels on the pre-drawn random sample."""
    with (DATA / "second_rater_labels.csv").open(encoding="utf-8", newline="") as fh:
        human = {r["task_id"]: r["world_state_success"] for r in csv.DictReader(fh)}
    with (DATA / "second_rater_sample.csv").open(encoding="utf-8", newline="") as fh:
        sample = [r["task_id"] for r in csv.DictReader(fh)]
    if set(human) != set(sample):
        raise SystemExit("second_rater_labels.csv does not cover exactly the drawn sample")
    first = {r["task_id"]: r["truth"] for r in rows}
    decided = sorted(t for t in sample if human[t] in ("0", "1"))
    a = [first[t] for t in decided]
    b = [int(human[t]) for t in decided]
    k = cohen_kappa(a, b)
    return {
        "n_sampled": len(sample),
        "share_of_scored_trials": round(len(sample) / len(rows), 4),
        "n_cannot_tell": len(sample) - len(decided),
        "cannot_tell_tasks": sorted(t for t in sample if t not in decided),
        "n_compared": len(decided),
        "both_success": sum(1 for x, y in zip(a, b, strict=True) if x == 1 and y == 1),
        "both_failure": sum(1 for x, y in zip(a, b, strict=True) if x == 0 and y == 0),
        "first_success_second_failure": sum(1 for x, y in zip(a, b, strict=True) if x == 1 and y == 0),
        "first_failure_second_success": sum(1 for x, y in zip(a, b, strict=True) if x == 0 and y == 1),
        "observed_agreement": _rate(sum(1 for x, y in zip(a, b, strict=True) if x == y), len(decided)),
        "cohen_kappa": None if k is None else round(k, 4),
        "disagreements": [{"task_id": t, "first_rater": first[t], "second_rater": int(human[t])}
                          for t in decided if first[t] != int(human[t])],
    }


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(float(np.mean(values)), 2),
            "median": float(np.median(values)), "min": min(values), "max": max(values)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows() -> list[dict]:
    def read(name: str) -> dict[str, dict]:
        with (DATA / name).open(encoding="utf-8", newline="") as fh:
            return {r["task_id"]: r for r in csv.DictReader(fh)}

    tasks, gt, gate = read("tasks.csv"), read("ground_truth.csv"), read("gate_outputs.csv")
    if set(gt) != set(gate):
        raise SystemExit("ground_truth.csv and gate_outputs.csv cover different tasks")
    rows = []
    for tid in sorted(gt):
        rows.append({
            "task_id": tid,
            "skill": tasks[tid]["skill_domain"],
            "regime": tasks[tid]["intended_regime"],
            "truth": int(gt[tid]["world_state_success"]),
            "achievable": int(gt[tid]["achievable"]),
            "gate": int(gate[tid]["gate_decision"]),
            "gate_reason": gate[tid]["gate_reason"],
            "claim": int(gate[tid]["logged_claim"]),
            "proxy_a": float(gate[tid]["proxy_a_confidence"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(rows: list[dict]) -> dict:
    y = [r["truth"] for r in rows]
    ach = [r["achievable"] for r in rows]
    gate = [r["gate"] for r in rows]
    claim = [r["claim"] for r in rows]
    conf = [r["proxy_a"] for r in rows]
    tasks = _task_counts()

    g = confusion(y, gate)
    g["ci95"] = {
        "accuracy": wilson(g["TP"] + g["TN"], g["n"]),
        "false_positive_rate": wilson(g["FP"], g["FP"] + g["TN"]),
        "false_negative_rate": wilson(g["FN"], g["FN"] + g["TP"]),
    }
    reasons_fn: dict[str, int] = {}
    reasons_tn: dict[str, int] = {}
    for r in rows:
        if r["gate"] == 0:
            bucket = reasons_fn if r["truth"] == 1 else reasons_tn
            bucket[r["gate_reason"]] = bucket.get(r["gate_reason"], 0) + 1

    pts = roc_points(y, conf)
    proxy_auc = round(auc(y, conf), 4)

    # Self-confidence where the agent was right vs wrong.
    by_truth = {"successes": describe([c for c, t in zip(conf, y, strict=True) if t == 1]),
                "failures": describe([c for c, t in zip(conf, y, strict=True) if t == 0])}
    gate_right = [c for c, t, p in zip(conf, y, gate, strict=True) if t == p]
    gate_wrong = [c for c, t, p in zip(conf, y, gate, strict=True) if t != p]
    fp_conf = [c for c, t, p in zip(conf, y, gate, strict=True) if p == 1 and t == 0]
    tp_conf = [c for c, t, p in zip(conf, y, gate, strict=True) if p == 1 and t == 1]
    regimes = sorted({r["regime"] for r in rows})
    by_regime = {}
    for reg in regimes:
        sub = [r for r in rows if r["regime"] == reg]
        by_regime[reg] = {
            "n": len(sub),
            "world_state_successes": sum(r["truth"] for r in sub),
            "gate_reported_success": sum(r["gate"] for r in sub),
            "proxy_a": describe([r["proxy_a"] for r in sub]),
        }
    most_conf_where_least_certain = by_truth["failures"]["mean"] > by_truth["successes"]["mean"]

    return {
        "data": {
            "tasks_preregistered": tasks["total"],
            "tasks_scored": len(rows),
            "tasks_excluded": tasks["excluded"],
            "tasks_not_run": tasks["not_run"],
            "world_state_successes": sum(y),
            "world_state_failures": len(y) - sum(y),
            "failures_on_unachievable_tasks": sum(1 for t, a in zip(y, ach, strict=True) if t == 0 and a == 0),
            "failures_on_achievable_tasks": sum(1 for t, a in zip(y, ach, strict=True) if t == 0 and a == 1),
        },
        "gate_decision": {
            **g,
            "false_positives_on_unachievable_tasks": sum(
                1 for t, a, p in zip(y, ach, gate, strict=True) if p == 1 and t == 0 and a == 0),
            "reasons_on_false_negatives": reasons_fn,
            "reasons_on_true_negatives": reasons_tn,
            "preregistered_regime_variant": regime_confusion(y, ach, gate),
        },
        "logged_claim": {
            **confusion(y, claim),
            "share_of_rows_claiming_success": round(sum(claim) / len(claim), 4),
            "preregistered_regime_variant": regime_confusion(y, ach, claim),
        },
        "proxy_a": {
            "auc": proxy_auc,
            "auc_bootstrap": bootstrap_auc_ci(y, conf),
            "max_score_on_a_failure": by_truth["failures"]["max"],
            "min_score_on_a_success": by_truth["successes"]["min"],
            "score_counts": {
                "successes": _counts([c for c, t in zip(conf, y, strict=True) if t == 1]),
                "failures": _counts([c for c, t in zip(conf, y, strict=True) if t == 0]),
            },
            "roc": pts,
            "confidence_by_truth": by_truth,
            "confidence_when_gate_right": describe(gate_right),
            "confidence_when_gate_wrong": describe(gate_wrong),
            "confidence_on_gate_false_positives": describe(fp_conf),
            "confidence_on_gate_true_positives": describe(tp_conf),
            "mean_confidence_higher_on_failures_than_successes": most_conf_where_least_certain,
        },
        "by_intended_regime": by_regime,
        "second_rater": second_rater(rows),
    }


def _counts(values: list[float]) -> dict[str, int]:
    """Score -> how many rows had it, keys as integer strings, ascending."""
    out: dict[str, int] = {}
    for v in sorted(values):
        out[f"{v:.0f}"] = out.get(f"{v:.0f}", 0) + 1
    return out


def _task_counts() -> dict:
    with (DATA / "tasks.csv").open(encoding="utf-8", newline="") as fh:
        status = [r["study_status"] for r in csv.DictReader(fh)]
    return {"total": len(status), "excluded": status.count("excluded"),
            "not_run": status.count("not_run")}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def results_table(res: dict) -> str:
    """Markdown block pasted into README.md between the results markers.
    tests/test_evaluate.py checks the README copy is identical."""
    g, c, p, d = res["gate_decision"], res["logged_claim"], res["proxy_a"], res["data"]
    lines = [
        f"Scored trials: {d['tasks_scored']} of {d['tasks_preregistered']} pre-specified tasks "
        f"({d['world_state_successes']} world-state successes, {d['world_state_failures']} failures).",
        "",
        "| Metric | Gate verdict (primary) | Logged claim (pre-registered field) |",
        "|---|---|---|",
    ]
    for key, label in [("TP", "TP"), ("FP", "FP (over-confidence)"), ("TN", "TN"),
                       ("FN", "FN (over-deferral)"), ("accuracy", "Accuracy"),
                       ("precision", "Precision"), ("recall", "Recall"),
                       ("specificity", "Specificity"),
                       ("false_positive_rate", "False-positive rate"),
                       ("false_negative_rate", "False-negative rate")]:
        lines.append(f"| {label} | {_fmt(g[key])} | {_fmt(c[key])} |")
    ci = g["ci95"]
    lines += [
        "",
        f"Gate verdict 95% Wilson intervals: accuracy {ci['accuracy'][0]:.3f} to {ci['accuracy'][1]:.3f}, "
        f"false-positive rate {ci['false_positive_rate'][0]:.3f} to {ci['false_positive_rate'][1]:.3f}, "
        f"false-negative rate {ci['false_negative_rate'][0]:.3f} to {ci['false_negative_rate'][1]:.3f}.",
        "",
        f"Proxy A self-confidence: ROC AUC {p['auc']:.3f} "
        f"(bootstrap 95% CI {p['auc_bootstrap']['ci95'][0]:.3f} to {p['auc_bootstrap']['ci95'][1]:.3f}). "
        f"Mean score {p['confidence_by_truth']['successes']['mean']:.2f} on successes, "
        f"{p['confidence_by_truth']['failures']['mean']:.2f} on failures; "
        f"highest score on any failure {p['max_score_on_a_failure']:.0f}, "
        f"lowest on any success {p['min_score_on_a_success']:.0f}. "
        f"Mean score {p['confidence_when_gate_right']['mean']:.2f} where the gate verdict was right, "
        f"{p['confidence_when_gate_wrong']['mean']:.2f} where it was wrong.",
        "",
        "Over-confidence vs over-deferral under each decision rule (same 31 trials):",
        "",
        "| Decision rule | Reports success on | FP (over-confidence) | FN (over-deferral) |",
        "|---|---|---|---|",
        f"| Logged claim | {c['TP'] + c['FP']} | {c['FP']} | {c['FN']} |",
        f"| Gate verdict | {g['TP'] + g['FP']} | {g['FP']} | {g['FN']} |",
    ]
    for pt in p["roc"]:
        if pt["threshold"] == "inf":
            continue
        lines.append(f"| Proxy A >= {pt['threshold']:.0f} | {pt['TP'] + pt['FP']} | {pt['FP']} | {pt['FN']} |")
    s = res["second_rater"]
    lines += [
        "",
        f"Human second rating: {s['n_sampled']} randomly drawn trials, labelled blind. "
        f"{s['n_cannot_tell']} were marked \"cannot tell from the evidence\". "
        f"On the other {s['n_compared']}, the two raters agreed on "
        f"{s['both_success'] + s['both_failure']} (observed agreement {s['observed_agreement']:.3f}, "
        f"Cohen's kappa {s['cohen_kappa']:.3f}).",
    ]
    return "\n".join(lines) + "\n"


def plot(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = {"Software": None}
    g = res["gate_decision"]
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    cells = np.array([[g["TP"], g["FN"]], [g["FP"], g["TN"]]])
    ax.imshow(cells, cmap="Blues", vmin=0, vmax=cells.max() * 1.6)
    names = [["TP", "FN\nover-deferral"], ["FP\nover-confidence", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cells[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], ["reported\nsuccess", "reported\nfailure"])
    ax.set_yticks([0, 1], ["task\nsucceeded", "task\nfailed"])
    ax.set_title(f"Gate verdict vs world state (n={g['n']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", metadata=meta)
    plt.close(fig)

    p = res["proxy_a"]
    fpr = [pt["fpr"] for pt in p["roc"]]
    tpr = [pt["tpr"] for pt in p["roc"]]
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=150)
    ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls="--", label="chance")
    ax.plot(fpr, tpr, marker="o", ms=4, color="#1f5fa8", label=f"Proxy A score (AUC {p['auc']:.3f})")
    for pt in p["roc"]:
        if pt["threshold"] != "inf":
            ax.annotate(f"≥{pt['threshold']:.0f}", (pt["fpr"], pt["tpr"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=7, color="#1f5fa8")
    gd, lc = res["gate_decision"], res["logged_claim"]
    ax.scatter([gd["false_positive_rate"]], [gd["recall"]], marker="s", s=60, color="#c0392b",
               zorder=3, label="gate verdict")
    ax.scatter([lc["false_positive_rate"]], [lc["recall"]], marker="^", s=70, color="#7f8c8d",
               zorder=3, label="logged claim")
    ax.set_xlabel("False-positive rate (over-confidence)")
    ax.set_ylabel("True-positive rate")
    ax.set_xlim(-0.02, 1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"ROC, n={gd['n']}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "roc_curve.png", metadata=meta)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    res = analyse(load_rows())
    (RESULTS / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")
    (RESULTS / "results_table.md").write_text(results_table(res), encoding="utf-8", newline="\n")
    plot(res)
    g, p = res["gate_decision"], res["proxy_a"]
    print(f"gate verdict: TP={g['TP']} FP={g['FP']} TN={g['TN']} FN={g['FN']} "
          f"accuracy={g['accuracy']} FPR={g['false_positive_rate']} FNR={g['false_negative_rate']}")
    print(f"proxy A: AUC={p['auc']} CI95={p['auc_bootstrap']['ci95']}")
    print(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
