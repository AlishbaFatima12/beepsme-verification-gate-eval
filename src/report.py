"""Human-readable outputs: the README blocks and the two figures.

Both README blocks are generated from results.json and checked by the tests,
so no number in them is typed by hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fmt(v) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def key_findings(res: dict) -> str:
    """The short findings list at the top of the README."""
    g, c, p, s = res["gate_decision"], res["logged_claim"], res["proxy_a"], res["second_rater"]
    reasons = g["reasons_on_false_negatives"]
    fn_reason = next(iter(reasons)) if len(reasons) == 1 else None
    same_cause = (f"Every over-deferral has the same cause: `{fn_reason}`, a request-size error "
                  "raised after the content had already been delivered."
                  if fn_reason else "The over-deferrals have several causes.")
    at_mode = p["score_counts"]["successes"].get("15", 0) + p["score_counts"]["failures"].get("15", 0)
    # strictest-useful threshold: the lowest one that makes no false-success claim
    safe = min((pt for pt in p["roc"] if pt["threshold"] != "inf" and pt["FP"] == 0),
               key=lambda pt: pt["threshold"])
    lines = [
        f"- **The field the protocol named as the agent's claim reports success on "
        f"{c['TP'] + c['FP']} of {c['n']} trials**, including every impossible task. "
        "It turned out to be hardcoded, so it carries no information.",
        f"- **The agent's real verdict is right on {g['TP'] + g['TN']} of {g['n']} trials "
        f"(accuracy {g['accuracy']:.3f}).** It errs both ways: {g['FP']} over-confident and "
        f"{g['FN']} over-deferring. {same_cause} "
        f"{g['false_positives_on_unachievable_tasks']} of the {g['FP']} over-confident cases "
        "were tasks that could not be done: the gate has no \"declined\" outcome, so an "
        "honest refusal and a silent no-op both end as success.",
        f"- **A post-hoc self-confidence score ranks successes above failures "
        f"(ROC AUC {p['auc']:.3f})**, but coarsely: {at_mode} of {g['n']} trials scored "
        f"exactly 15, and no failure scored above {p['max_score_on_a_failure']:.0f}. Mean "
        f"confidence was lower on failures ({p['confidence_by_truth']['failures']['mean']:.2f}) "
        f"than on successes ({p['confidence_by_truth']['successes']['mean']:.2f}).",
        f"- **Choosing the decision rule chooses the failure mode.** Counting (over-confident, "
        f"over-deferring) errors: the claim field gives ({c['FP']}, {c['FN']}), the verdict "
        f"({g['FP']}, {g['FN']}), and accepting only confidence of {safe['threshold']:.0f} or more "
        f"({safe['FP']}, {safe['FN']}). None of the rules keeps both low.",
        f"- **A blind human second rating** of {s['n_sampled']} random trials agreed with the "
        f"first rater on {s['both_success'] + s['both_failure']} of {s['n_compared']} decidable "
        f"trials (Cohen's kappa {s['cohen_kappa']:.2f}); {s['n_cannot_tell']} were judged "
        "undecidable from the evidence.",
    ]
    return "\n".join(lines) + "\n"


def results_table(res: dict) -> str:
    """The full results block in the README."""
    g, c, p, d = res["gate_decision"], res["logged_claim"], res["proxy_a"], res["data"]
    lines = [
        f"Scored trials: {d['tasks_scored']} of {d['tasks_preregistered']} pre-specified tasks "
        f"({d['world_state_successes']} world-state successes, {d['world_state_failures']} failures).",
        "",
        "| Metric | Gate verdict (primary) | Logged claim (protocol's field) |",
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
        f"Self-confidence score: ROC AUC {p['auc']:.3f} "
        f"(bootstrap 95% CI {p['auc_bootstrap']['ci95'][0]:.3f} to {p['auc_bootstrap']['ci95'][1]:.3f}). "
        f"Mean score {p['confidence_by_truth']['successes']['mean']:.2f} on successes, "
        f"{p['confidence_by_truth']['failures']['mean']:.2f} on failures; "
        f"highest score on any failure {p['max_score_on_a_failure']:.0f}, "
        f"lowest on any success {p['min_score_on_a_success']:.0f}. "
        f"Mean score {p['confidence_when_gate_right']['mean']:.2f} where the gate verdict was right, "
        f"{p['confidence_when_gate_wrong']['mean']:.2f} where it was wrong.",
        "",
        f"Over-confidence vs over-deferral under each decision rule (same {g['n']} trials):",
        "",
        "| Decision rule | Reports success on | FP (over-confidence) | FN (over-deferral) |",
        "|---|---|---|---|",
        f"| Logged claim | {c['TP'] + c['FP']} | {c['FP']} | {c['FN']} |",
        f"| Gate verdict | {g['TP'] + g['FP']} | {g['FP']} | {g['FN']} |",
    ]
    for pt in p["roc"]:
        if pt["threshold"] == "inf":
            continue
        lines.append(f"| Confidence >= {pt['threshold']:.0f} | {pt['TP'] + pt['FP']} | {pt['FP']} | {pt['FN']} |")
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


def plot(res: dict, out_dir: Path) -> None:
    """Write confusion_matrix.png and roc_curve.png into out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = {"Software": None}  # no version stamp, so reruns give identical bytes
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
    fig.savefig(out_dir / "confusion_matrix.png", metadata=meta)
    plt.close(fig)

    p = res["proxy_a"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=150)
    ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls="--", label="chance")
    ax.plot([pt["fpr"] for pt in p["roc"]], [pt["tpr"] for pt in p["roc"]], marker="o", ms=4,
            color="#1f5fa8", label=f"self-confidence score (AUC {p['auc']:.3f})")
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
    fig.savefig(out_dir / "roc_curve.png", metadata=meta)
    plt.close(fig)
