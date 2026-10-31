# BeepSME verification-gate evaluation

A desktop LLM agent's own "it worked / it failed" reports, scored blind against
what actually happened, on 31 trials of a 44-task protocol fixed in advance.

## Research question

Can an LLM agent's own success claims be trusted, and how does the choice of
verification rule change which way it fails: reporting success when the task
failed (over-confidence), or failure when it succeeded (over-deferral)?

## System

BeepSME is a voice-first Windows desktop assistant I built. It routes a request
to a skill (teach a page, solve homework, read aloud, translate, write a
document, guide through an app, send a message) and acts on the computer. At
the end of every task it decides `ok` or `failed` and shows that to the user.
The full agent is private. Project page: <https://beepsme.com>. Portfolio:
<https://beepsme.com/portfolio>.

## Method

- **Tasks:** 44 tasks over 7 skills, each with an oracle (a checkable success
  condition) written before the run. They are stratified into happy path,
  over-confidence bait, over-deferral bait and impossible
  ([`data/tasks.csv`](data/tasks.csv)).
- **Protocol fixed in advance:** committed to the private BeepSME repository
  on 2026-06-05. The run was on 2026-08-20. It is not on a public registry.
  Copies and hashes: [`preregistration/`](preregistration/PROVENANCE.md).
- **Run:** a driver set each precondition and typed the command, with the
  microphone muted.
- **Ground truth:** each trial is labelled 1/0 from world state (saved file,
  answer key, recorded speech, screen), plus `achievable` (could it have been
  done?) ([`data/ground_truth.csv`](data/ground_truth.csv)). The first rater
  was an LLM (Claude), working from a sheet that hid the agent's claim and its
  confidence score. These labels are the ones scored.
- **Human second rating:** I then labelled a random sample of 10 trials
  myself (seed fixed in
  [`src/second_rater_sample.py`](src/second_rater_sample.py)). The sheet hid
  the agent's report, its confidence score and the first rater's labels, and
  showed only evidence from the scored attempt. "Cannot tell from the
  evidence" was an allowed answer. My labels are in
  [`data/second_rater_labels.csv`](data/second_rater_labels.csv), unchanged.
- **Predictors** ([`data/gate_outputs.csv`](data/gate_outputs.csv)):
  - the *gate verdict*, the agent's real end-of-task decision, read from the
    run logs ([`src/GATE.md`](src/GATE.md));
  - the *logged claim*, the field the protocol named;
  - *Proxy A*, a 0-100 self-confidence score from one grader call per task
    ([`src/proxy_a.py`](src/proxy_a.py)).

  "Reports success" is the positive class.

**Deviations.**
1. 31 of 44 tasks were scored. Not run: 7 WhatsApp send and 5 Canva tutor
   tasks. Excluded: TEACH-04.
2. The logged claim turned out to be hardcoded to "success" for any turn that
   did not crash. So the primary predictor is the real verdict, **a choice
   made after seeing the data**.
3. No frozen build. The second rating was done after the first results were
   known to me, not alongside the first rating.

## Results

<!-- results:start -->
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
<!-- results:end -->

![Confusion matrix of the gate verdict](results/confusion_matrix.png)
![ROC curve of Proxy A, with the two binary rules as points](results/roc_curve.png)

**In plain words:**

- **The logged claim carries no information.** It says success on every
  trial, including the impossible ones.
- **The real gate errs both ways.** All 5 over-deferrals were an HTTP 413
  error from the model proxy *after* the lesson, summary or page description
  had been spoken.
  4 of the 5 over-confident cases were tasks that could not be done. The gate
  reports `ok` whenever nothing raised an error, so an honest refusal ("the
  clipboard is empty") and a silent no-op look the same. It has no
  "declined" outcome.
- **Proxy A ranks successes above failures, but coarsely.** 20 of 31 trials
  scored exactly 15. Its scores were higher where the gate was wrong, because
  the gate's false negatives were delivered content.
- **The decision rule decides the failure mode.** Moving from the logged claim
  to the verdict to a Proxy A threshold trades over-confidence for
  over-deferral. No rule tried keeps both low.

The data do **not** show the agent being most confident where it was least
correct: Proxy A was lower on failures.

## Limitations

- **Small sample:** 31 trials, one agent, one day, wide intervals.
- **Rater agreement is moderate:** kappa on 8 trials is a rough estimate.
  The first rater is from the same model family as the agent. The human
  rater built the system and had seen the headline results. Some oracles
  (faithful teaching, a document vs a spoken answer) need judgement. Where
  the raters disagree, the scored label is the first rater's.
- **Declining has no right answer here:** where declining was the right
  behaviour, the world-state label is still 0.
- **Proxy A had a wrong input:** it was always told "Reported outcome:
  success", and it is a post-hoc grader, not the agent's in-loop belief.
- **Partial transcripts:** the reply logged in `data/raw/turns.jsonl` is
  sometimes only an acknowledgement. Speech capture and screenshots are not
  published (they show the operator's desktop).
- **Private run logs:** [`src/extract.py`](src/extract.py) documents how
  `data/` was taken from them. `data/raw/` keeps the exact lines used.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/evaluate.py        # writes results/
pytest -q
```

No API key is needed. The tests check the metric maths on hand-made examples,
that `data/gate_outputs.csv` follows from the raw log lines, and that
`results/results.json` and the table above match a fresh run.

## Related notes

<https://github.com/AlishbaFatima12/agent-reliability-notes>

## Contact and citation

Syeda Alishba Fatima, alishbafatima25@gmail.com, ORCID
[0009-0007-7901-3224](https://orcid.org/0009-0007-7901-3224).
Cite via [`CITATION.cff`](CITATION.cff). MIT license.
