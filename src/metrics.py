"""Statistics used by the evaluation. Pure functions on plain lists, no I/O.

Conventions: labels are 0/1 integers, 1 = success. For a predictor, 1 means
"the agent reports success". Rates that would divide by zero return None,
not 0, so an undefined rate is never mistaken for a measured one.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z_95 = 1.959964  # two-sided 95% normal quantile


def rate(num: int, den: int) -> float | None:
    """num/den rounded to 4 places, or None when den is 0."""
    return round(num / den, 4) if den else None


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """2x2 counts and the derived rates.

    FP = reported success, task failed (over-confidence).
    FN = reported failure, task succeeded (over-deferral).
    """
    pairs = list(zip(y_true, y_pred, strict=True))
    tp = sum(1 for t, p in pairs if p == 1 and t == 1)
    fp = sum(1 for t, p in pairs if p == 1 and t == 0)
    tn = sum(1 for t, p in pairs if p == 0 and t == 0)
    fn = sum(1 for t, p in pairs if p == 0 and t == 1)
    n = len(pairs)
    return {
        "n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": rate(tp + tn, n),
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn),
        "specificity": rate(tn, tn + fp),
        "false_positive_rate": rate(fp, fp + tn),
        "false_negative_rate": rate(fn, fn + tp),
    }


def regime_confusion(y_true: Sequence[int], achievable: Sequence[int],
                     y_pred: Sequence[int]) -> dict:
    """The protocol's own variant: a reported failure is over-deferral when
    the task was achievable (even if not completed) and a correct deferral
    only when it was not achievable."""
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
            "over_confidence_rate": rate(fp, tp + fp),
            "over_deferral_rate": rate(fn, tn + fn)}


def wilson(k: int, n: int, z: float = Z_95) -> list[float] | None:
    """Wilson score interval for the proportion k/n (default 95%)."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 4), round(centre + half, 4)]


def roc_points(y_true: Sequence[int], score: Sequence[float]) -> list[dict]:
    """One ROC point per rule "report success if score >= threshold", for
    every distinct score, plus the all-negative point (threshold inf)."""
    pairs = list(zip(y_true, score, strict=True))
    pos = sum(y_true)
    neg = len(pairs) - pos
    pts = []
    for thr in [math.inf, *sorted(set(score), reverse=True)]:
        tp = sum(1 for t, s in pairs if s >= thr and t == 1)
        fp = sum(1 for t, s in pairs if s >= thr and t == 0)
        pts.append({"threshold": "inf" if thr == math.inf else thr,
                    "TP": tp, "FP": fp, "FN": pos - tp, "TN": neg - fp,
                    "tpr": round(tp / pos, 4), "fpr": round(fp / neg, 4)})
    return pts


def auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Probability that a random success outscores a random failure, ties
    counting one half (Mann-Whitney form). Equals the trapezoidal area under
    roc_points."""
    pos = [s for t, s in zip(y_true, score, strict=True) if t == 1]
    neg = [s for t, s in zip(y_true, score, strict=True) if t == 0]
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_auc_ci(y_true: Sequence[int], score: Sequence[float], *,
                     n_boot: int = 10_000, seed: int = 20260820) -> dict:
    """Percentile bootstrap 95% interval for the AUC. Resamples with a single
    class have no AUC and are skipped; the count kept is reported."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(auc(yb.tolist(), s[idx].tolist()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_boot": n_boot, "n_valid": len(vals), "seed": seed}


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Cohen's kappa for two raters' binary labels on the same items; None
    when chance agreement is 1 (kappa undefined)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def describe(values: Sequence[float]) -> dict:
    """n, mean, median, min and max of a list of scores."""
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(float(np.mean(values)), 2),
            "median": float(np.median(values)), "min": min(values), "max": max(values)}


def value_counts(values: Sequence[float]) -> dict[str, int]:
    """Score -> number of rows with it, ascending, keys as integer strings."""
    out: dict[str, int] = {}
    for v in sorted(values):
        out[f"{v:.0f}"] = out.get(f"{v:.0f}", 0) + 1
    return out
