"""One-time extraction from the PRIVATE BeepSME evaluation folder.

This script documents exactly how the files in `data/` were produced. It needs
the private run logs, so reviewers cannot run it; everything downstream of it
(`src/evaluate.py`) runs from `data/` alone.

    python src/extract.py --eval-dir <path to the private research/eval folder>

What it reads (all produced on 2026-08-20 by the study run):
  tasks.jsonl                              the 44 pre-specified tasks
  out/prefix_study/labels_prefix.csv       blind world-state labels, 31 trials
  out/turns_snapshot_2026_08_20.jsonl      the agent's per-turn research log
  evidence/pilot2.log, evidence/block1..7.log   console logs of each run block
  evidence/driver_log.jsonl                one line per attempt (start time)
  evidence/EVAL-HW-0x/*.md                 homework solution files (world state)

What it writes (sanitised; only the study rows, only the needed fields):
  data/tasks.csv, data/ground_truth.csv, data/gate_outputs.csv
  data/raw/verdict_lines.txt, data/raw/turns.jsonl, data/raw/driver_attempts.csv
  data/artifacts/<task>.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive import HEADER_RE, VERDICT_RE, load_final_verdicts

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RAW = DATA / "raw"

# The run blocks, in the order they were executed on 2026-08-20 (the order
# matches evidence/driver_log.jsonl; checked below). Later logs
# (postfix_*.log) are post-fix verification runs on a changed build and are
# NOT part of the study.
STUDY_LOGS = ["pilot2.log", "block1.log", "block2.log", "block3.log",
              "block4.log", "block5.log", "block6.log", "block7.log"]

# Last turn of the study run in the archived snapshot. Attempts started after
# this instant belong to the post-fix verification and are excluded.
STUDY_END_UTC = "2026-08-20T12:51:00"

_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s'\"|,]*")
_FORBIDDEN = [re.compile(p, re.I) for p in (
    r"[A-Za-z]:\\Users\\", r"sk-or-", r"sk-ant-", r"@gmail\.com", r"Bearer ",
)]


def scrub(text: str) -> str:
    """Replace absolute Windows paths with a placeholder."""
    return _PATH_RE.sub("<local-path>", text or "")


def assert_clean(path: Path) -> None:
    body = path.read_text(encoding="utf-8")
    for pat in _FORBIDDEN:
        if pat.search(body):
            raise SystemExit(f"refusing to write {path}: matches {pat.pattern}")


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    assert_clean(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    args = ap.parse_args()
    src: Path = args.eval_dir

    # 1. verdict lines: only attempt headers and [task] verdict= lines.
    RAW.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    for name in STUDY_LOGS:
        text = (src / "evidence" / name).read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.rstrip()
            if HEADER_RE.match(line) or VERDICT_RE.match(line):
                out_lines.append(f"{name}:{i}:{scrub(line)}")
    vpath = RAW / "verdict_lines.txt"
    vpath.write_text("\n".join(out_lines) + "\n", encoding="utf-8", newline="\n")
    assert_clean(vpath)
    verdicts = load_final_verdicts(vpath)

    # 2. driver attempts inside the study window; cross-check attempt counts.
    attempts = []
    files_of_last_attempt: dict[str, list[str]] = {}
    for line in (src / "evidence" / "driver_log.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["started"][:19] < STUDY_END_UTC:
            attempts.append({"task_id": r["task_id"], "started_utc": r["started"][:19],
                             "elapsed_s": r["elapsed_s"], "timed_out": int(bool(r["timed_out"]))})
            files_of_last_attempt[r["task_id"]] = r.get("files_captured", [])
    write_csv(RAW / "driver_attempts.csv", ["task_id", "started_utc", "elapsed_s", "timed_out"], attempts)
    driver_counts: dict[str, int] = {}
    for a in attempts:
        driver_counts[a["task_id"]] = driver_counts.get(a["task_id"], 0) + 1
    log_counts = {t: v["n_attempts"] for t, v in verdicts.items()}
    if driver_counts != log_counts:
        raise SystemExit(f"attempt counts disagree: driver={driver_counts} logs={log_counts}")

    # 3. labels (the pre-fix study, 31 trials).
    labels = list(csv.DictReader((src / "out" / "prefix_study" / "labels_prefix.csv").open(encoding="utf-8")))
    labelled = {r["task_id"] for r in labels}

    # 4. last turn per labelled task from the research log snapshot.
    last_turn: dict[str, dict] = {}
    for line in (src / "out" / "turns_snapshot_2026_08_20.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tid = (rec.get("task_id") or "").strip()
        if tid in labelled:
            last_turn[tid] = rec
    last_start = {}
    for a in attempts:
        last_start[a["task_id"]] = a["started_utc"]
    turns_out = []
    for tid in sorted(labelled):
        rec = last_turn[tid]
        ts = rec["ts_utc"][:19]
        if not (last_start[tid] <= ts < STUDY_END_UTC):
            raise SystemExit(f"{tid}: last logged turn {ts} is not inside its final attempt ({last_start[tid]})")
        turns_out.append({
            "task_id": tid,
            "ts_utc": ts,
            "command": rec.get("command", ""),
            "logged_outcome": rec.get("outcome", ""),
            "proxy_a": rec.get("proxy_A_self_confidence"),
            "actions": [scrub(str(a)) for a in rec.get("actions", [])],
            "spoken": scrub(rec.get("spoken", "")),
        })
    tpath = RAW / "turns.jsonl"
    tpath.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in turns_out), encoding="utf-8",
                     newline="\n")
    assert_clean(tpath)

    # 5. tasks.csv: all 44, with what happened to each in the study.
    tasks = [json.loads(x) for x in (src / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    task_rows = []
    for t in tasks:
        tid = t["task_id"]
        if tid in labelled:
            status = "scored"
        elif tid in driver_counts:
            status = "excluded"
        else:
            status = "not_run"
        task_rows.append({
            "task_id": tid, "skill_domain": t["skill"], "difficulty": t["difficulty"],
            "intended_regime": t["intended_regime"], "description": t["goal"],
            "precondition": t.get("precondition", ""), "oracle": t["oracle"],
            "oracle_method": t["oracle_method"], "study_status": status,
        })
    write_csv(DATA / "tasks.csv", list(task_rows[0].keys()), task_rows)

    # 6. ground_truth.csv
    gt_rows = [{
        "task_id": r["task_id"],
        "world_state_success": r["ground_truth_success"],
        "achievable": r["achievable"],
        "how_verified": r["oracle_checked"],
        "rater": "AI rater (Claude), blinded to the agent's claim and Proxy A",
        "label_date": r["label_date"],
        "build": r["build_tag"],
        "notes": r["notes"],
    } for r in labels]
    write_csv(DATA / "ground_truth.csv", list(gt_rows[0].keys()), gt_rows)

    # 7. gate_outputs.csv
    turns_by_id = {t["task_id"]: t for t in turns_out}
    gate_rows = []
    for r in labels:
        tid = r["task_id"]
        v = verdicts[tid]
        tr = turns_by_id[tid]
        gate_rows.append({
            "task_id": tid,
            "gate_decision": 1 if v["verdict"] == "ok" else 0,
            "gate_reason": v["reason"],
            "gate_steps": v["steps"],
            "logged_claim": 1 if tr["logged_outcome"] == "success" else 0,
            "proxy_a_confidence": tr["proxy_a"],
            "n_attempts": v["n_attempts"],
        })
    write_csv(DATA / "gate_outputs.csv", list(gate_rows[0].keys()), gate_rows)

    # 8. homework solution files (world-state artifacts): the .md the driver
    # recorded as created during the task's final in-study attempt. Other
    # files in the evidence folders are stale or from post-fix reruns.
    art = DATA / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    for tid in sorted(t for t in labelled if t.startswith("EVAL-HW-")):
        mds = [f for f in files_of_last_attempt.get(tid, []) if f.endswith(".md")]
        if tid == "EVAL-HW-04":
            # The final HW-04 attempt captured a stale file left by its
            # discarded first attempt (which read an unrelated window) next
            # to the real solution; keep the solution only.
            mds = [f for f in mds if "AI_agent_task_completion_log" not in f]
        if len(mds) != 1:
            raise SystemExit(f"{tid}: expected one solution file, got {mds}")
        dst = art / f"{tid}.md"
        shutil.copyfile(src / "evidence" / tid / mds[0], dst)
        assert_clean(dst)

    print(f"tasks={len(task_rows)} labelled={len(gt_rows)} attempts={len(attempts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
