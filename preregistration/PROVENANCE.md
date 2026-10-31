# Provenance of the pre-specified protocol

The two files in this folder are byte-for-byte copies of the evaluation
protocol and task manifest as committed to the private BeepSME repository
before the study ran. They have not been edited.

| File | Source path in the private repo | Last change before the run | SHA-256 |
|---|---|---|---|
| `README_protocol.md` | `research/eval/README.md` | commit `7c3e5090040104443ab25eda98dc9d2f1faa936d`, 2026-06-05 12:43 +05:00 | `2e306060ad209e53b55b05df6c3750c8d6b1216299eea64eef4c0a98582b1a3c` |
| `tasks.jsonl` | `research/eval/tasks.jsonl` | commit `b160c746027162f0c353b1f1b3e913e6cc05c5e5`, 2026-06-05 12:19 +05:00 | `b5c6284771b25edc558892f31cadee1f165fa2f159ddfe86ac76cdb6307e3d0c` |

History of the manifest: first committed 2026-06-04 with 40 tasks (commit
`07327396`), retargeted the same evening (`7271cf22`), and finalised at 44
tasks on 2026-06-05 when the tutor family was added (`b160c746`). It has not
changed since. The study ran on 2026-08-20.

## What this is and is not

- It is a protocol fixed in version control about eleven weeks before the run.
- It is NOT registered with an external registry (OSF, AsPredicted, etc.).
  The repository holding the commits is private, so readers cannot check the
  dates independently. Treat it as a pre-specified protocol.

## Deviations from the protocol (all declared in the main README)

1. 31 of the 44 tasks were scored. Not run: 7 send tasks (needed WhatsApp test
   contacts) and 5 Canva tutor tasks. Excluded after running: TEACH-04
   (four attempts, none usable).
2. The protocol asked for a frozen tagged build (`paper_v1_eval`). The run
   used a working tree with uncommitted changes.
3. The protocol asked for a second human rater on at least 20% of trials with
   Cohen's kappa. The first rater was an LLM (Claude). A human second rating
   on a random 10 of the 31 trials (32%) was done on 2026-10-31, after the
   first results were known to the rater. The agreement is reported in
   `results/results.json` under `second_rater`.
4. The protocol's prediction variable (the logged `outcome` field) turned out
   to be a constant because of a logging defect. The primary analysis here
   uses the agent's real verdict line instead. That choice was made after
   seeing the data.
5. Commands were typed into the agent's text-command path with the
   microphone muted, so speech recognition was not part of the test.
