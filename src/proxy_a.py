"""Proxy A: the agent's post-hoc self-confidence score (0-100).

This is the scorer that produced `proxy_a_confidence` in data/gate_outputs.csv.
It is a copy of the private module `beepsme/core/research_confidence.py`, with
three changes for publication: the API key and model come from environment
variables, the private repository URL header was removed, and the BeepSME
on/off flag was dropped. The prompt and parsing are unchanged.

It is NOT needed to reproduce the results: the scores were logged during the
2026-08-20 run and `src/evaluate.py` reads them from data/. It is here so a
reader can see exactly what the score measures.

How it was used in the run: after each task finished, ONE extra request was
sent with the user's command, the actions taken, what was said back, and the
reported outcome. It did not use the agent's conversation history or system
prompt. Model during the run: anthropic/claude-haiku-4.5 via OpenRouter, the
same model that ran the agent, at temperature 0.

Known limitation: the agent passed `outcome="success"` for every turn that did
not crash (the same hardcoded value as the logged claim), so the grader was
always told "Reported outcome: success", including on turns where the gate
verdict was "failed".
"""
from __future__ import annotations

import os
import re
from typing import Any

import requests

_OR_URL = "https://openrouter.ai/api/v1/chat/completions"

_SYS = "You are a strict self-assessment grader. Output ONLY an integer 0-100."

_USER_TMPL = (
    "I just attempted this user request and need an honest confidence score.\n"
    "On a scale of 0 to 100, how confident are you that the request was FULLY "
    "and CORRECTLY completed? 0 means certainly not completed, 100 means "
    "certainly completed correctly. Do not be charitable: if anything is "
    "uncertain or you cannot verify it, lower the score.\n\n"
    "User request: {command}\n"
    "Actions taken: {actions}\n"
    "What was said back to the user: {spoken}\n"
    "Reported outcome: {outcome}\n\n"
    "Reply with ONLY the integer."
)


def parse_score(text: str) -> int | None:
    """First integer in ``text``, clamped to [0, 100]; None if no digits."""
    m = re.search(r"\d{1,3}", text or "")
    if not m:
        return None
    return max(0, min(100, int(m.group(0))))


def score(
    command: str,
    *,
    actions: Any = None,
    spoken: str = "",
    outcome: str = "",
    timeout: float = 20.0,
) -> int | None:
    """0-100 self-confidence that the request was completed, or None on any error."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return None
    try:
        action_str = ", ".join(str(a) for a in (actions or []))[:1200] or "(none)"
        user = _USER_TMPL.format(
            command=(command or "")[:600],
            actions=action_str,
            spoken=(spoken or "")[:600],
            outcome=outcome or "unknown",
        )
        payload = {
            "model": os.getenv("PROXY_A_MODEL", "anthropic/claude-haiku-4.5"),
            "messages": [
                {"role": "system", "content": _SYS},
                {"role": "user", "content": user},
            ],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        resp = requests.post(_OR_URL, headers=headers, json=payload, timeout=(10, timeout))
        if not resp.ok:
            return None
        data = resp.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return parse_score(text)
    except Exception:
        return None
