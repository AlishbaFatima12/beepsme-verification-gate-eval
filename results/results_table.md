Scored trials: 31 of 44 pre-specified tasks (17 world-state successes, 14 failures).

| Metric | Gate verdict (primary) | Logged claim (pre-registered field) |
|---|---|---|
| TP | 12 | 17 |
| FP (over-confidence) | 5 | 14 |
| TN | 9 | 0 |
| FN (over-deferral) | 5 | 0 |
| Accuracy | 0.677 | 0.548 |
| Precision | 0.706 | 0.548 |
| Recall | 0.706 | 1.000 |
| Specificity | 0.643 | 0.000 |
| False-positive rate | 0.357 | 1.000 |
| False-negative rate | 0.294 | 0.000 |

Gate verdict 95% Wilson intervals: accuracy 0.501 to 0.814, false-positive rate 0.163 to 0.612, false-negative rate 0.133 to 0.531.

Proxy A self-confidence: ROC AUC 0.798 (bootstrap 95% CI 0.677 to 0.906). Mean score 23.65 on successes, 9.64 on failures; highest score on any failure 15, lowest on any success 15. Mean score 12.14 where the gate verdict was right, 28.20 where it was wrong.

Over-confidence vs over-deferral under each decision rule (same 31 trials):

| Decision rule | Reports success on | FP (over-confidence) | FN (over-deferral) |
|---|---|---|---|
| Logged claim | 31 | 14 | 0 |
| Gate verdict | 17 | 5 | 5 |
| Proxy A >= 72 | 1 | 0 | 16 |
| Proxy A >= 45 | 2 | 0 | 15 |
| Proxy A >= 35 | 5 | 0 | 12 |
| Proxy A >= 15 | 25 | 8 | 0 |
| Proxy A >= 5 | 28 | 11 | 0 |
| Proxy A >= 0 | 31 | 14 | 0 |

Human second rating: 10 randomly drawn trials, labelled blind. 2 were marked "cannot tell from the evidence". On the other 8, the two raters agreed on 6 (observed agreement 0.750, Cohen's kappa 0.467).
