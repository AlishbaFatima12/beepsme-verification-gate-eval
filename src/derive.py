"""Derive the gate's per-task decision from the extracted run-log lines.

The agent prints one canonical line when a task finishes:

    [task] verdict=ok steps=1
    [task] verdict=failed steps=1 reason=ai_http_413

and the evaluation driver prints one header line when it starts an attempt:

    [9/23] EVAL-READ-02 (read/happy_path) cap=240s

`data/raw/verdict_lines.txt` holds exactly these two kinds of line, copied in
order from the private run logs (see `src/extract.py`). This module pairs each
attempt header with the verdict that follows it and keeps the LAST attempt per
task, which is the attempt that was labelled.

Pure functions, no third-party imports, so the tests can re-run it.
"""
from __future__ import annotations

import re
from pathlib import Path

HEADER_RE = re.compile(r"^\[(\d+)/(\d+)\] (EVAL-[A-Z]+-\d+) \(([a-z]+)/([a-z_]+)\)")
VERDICT_RE = re.compile(r"^\[task\] verdict=(ok|failed) steps=(\d+)(?: reason=(.*))?$")


def parse_verdict_lines(lines: list[str]) -> list[dict]:
    """Return one dict per attempt, in log order.

    Each input line is ``<log file>:<line no>:<original line>``. Raises
    ValueError if an attempt has no verdict or more than one, because then the
    pairing would be a guess.
    """
    attempts: list[dict] = []
    current: dict | None = None
    for raw in lines:
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        source, lineno, text = raw.split(":", 2)
        h = HEADER_RE.match(text)
        if h:
            if current is not None and current["verdict"] is None:
                raise ValueError(f"attempt without verdict: {current}")
            current = {
                "task_id": h.group(3),
                "source": source,
                "header_line": int(lineno),
                "verdict": None,
                "steps": None,
                "reason": "",
            }
            attempts.append(current)
            continue
        v = VERDICT_RE.match(text)
        if v:
            if current is None:
                raise ValueError(f"verdict before any attempt header: {raw}")
            if current["verdict"] is not None:
                raise ValueError(f"two verdicts for one attempt: {current['task_id']} in {source}")
            current["verdict"] = v.group(1)
            current["steps"] = int(v.group(2))
            current["reason"] = (v.group(3) or "").strip()
            continue
        raise ValueError(f"unexpected line in verdict file: {raw}")
    if current is not None and current["verdict"] is None:
        raise ValueError(f"attempt without verdict: {current}")
    return attempts


def final_attempts(attempts: list[dict]) -> dict[str, dict]:
    """task_id -> its last attempt (later attempts overwrite earlier ones)."""
    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for a in attempts:
        counts[a["task_id"]] = counts.get(a["task_id"], 0) + 1
        out[a["task_id"]] = dict(a)
    for tid, a in out.items():
        a["n_attempts"] = counts[tid]
    return out


def load_final_verdicts(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return final_attempts(parse_verdict_lines(lines))
