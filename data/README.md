# Data dictionary

All files are UTF-8. One row per task unless stated. `task_id` joins every
file. 1 means success, or "reports success" for a predictor.

## `tasks.csv` (44 rows): the protocol's task list

| Column | Meaning |
|---|---|
| `task_id` | `EVAL-<SKILL>-<n>` |
| `skill_domain` | teach, homework, read, send, translate, write, tutor |
| `difficulty` | easy, ambiguous, adversarial (as set in the protocol) |
| `intended_regime` | happy_path, overconfidence_bait, overdeferral_bait, impossible. This records what the task was designed to provoke; it is not a label. |
| `description` | the command typed into the agent |
| `precondition` | what had to be on screen or on the clipboard first |
| `oracle` | the success condition, written before the run |
| `oracle_method` | how the oracle is checked (answer key, file, speech, screen) |
| `study_status` | `scored` (31), `excluded` (1), `not_run` (12) |

## `ground_truth.csv` (31 rows): first-rater labels, the ones scored

| Column | Meaning |
|---|---|
| `world_state_success` | 1 if the oracle held after the run |
| `achievable` | 1 if the task could have been completed from that state |
| `how_verified` | what the rater checked |
| `rater`, `label_date`, `build` | who labelled, when, and against which build |
| `notes` | rater's remarks |

## `gate_outputs.csv` (31 rows): what the agent reported

| Column | Meaning |
|---|---|
| `gate_decision` | the agent's end-of-task verdict, 1 = `ok` ([`src/GATE.md`](../src/GATE.md)) |
| `gate_reason` | the failure reason it printed (empty when `ok`) |
| `gate_steps` | action steps in the attempt |
| `logged_claim` | the research log's `outcome` field (1 on every row) |
| `proxy_a_confidence` | self-confidence score 0-100 ([`src/proxy_a.py`](../src/proxy_a.py)) |
| `n_attempts` | attempts in the run; the last one is the one scored |

## `second_rater_sample.csv` and `second_rater_labels.csv` (10 rows)

The seeded random sample and the human second rater's blind labels.
`world_state_success` is 1, 0 or `cannot_tell`.

## `raw/`: the exact source lines

| File | Contents |
|---|---|
| `verdict_lines.txt` | each attempt's header and `[task] verdict=` line, copied in order from the console logs. `src/derive.py` rebuilds `gate_decision` from it. |
| `turns.jsonl` | the last logged turn of each scored attempt: command, actions, reply, logged outcome, confidence. Paths are removed. |
| `driver_attempts.csv` | start time, duration and timeout of every attempt in the study window |

## `artifacts/`

The homework solution files the agent wrote during the scored attempts. They
are the world state for the homework oracles.
