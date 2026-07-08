"""Shared session-extraction logic (DRY across the local-session and
agentic-corpus extractors).

Both sources are Claude Code session transcripts (one JSONL record per turn,
`type` in {user, assistant}, assistant carrying `message.model`/`message.usage`,
the user turn before it carrying the prompt). The only difference is where the
records come from: `extract_local_sessions.py` reads them straight from files;
`extract_agentic_dataset.py` pulls them out of embedded transcript strings in an
SFT corpus. `iter_claude_rows` is the one pairing rule both feed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import yaml

# session model id -> catalog id. Dated ids are normalized (strip -YYYYMMDD);
# a few short aliases the logs use; <synthetic> is not a real serve.
ALIASES = {"opus": "claude-opus-4-8", "sonnet": "claude-sonnet-4-6", "haiku": "claude-haiku-4-5"}
SKIP_MODELS = {"<synthetic>", ""}
_DATE_SUFFIX = re.compile(r"-\d{8}$")

CODE_KW = ["code", "function", "bug", "compile", "stack trace", "refactor"]
MATH_KW = ["prove", "theorem", "integral", "equation", "calculate", "solve for"]
REASON_KW = ["why", "reason", "step by step", "analyze", "explain how", "trade-off"]
CREATIVE_KW = ["write a story", "poem", "creative", "imagine", "brainstorm"]


def classify(text: str, approx_tokens: int, has_media: bool = False) -> str:
    """Port of hanzo-router `Heuristic::classify` (registry Task taxonomy)."""
    if has_media:
        return "vision"
    if approx_tokens >= 32_000:
        return "long_context"
    t = text.lower()
    has = lambda ws: any(w in t for w in ws)  # noqa: E731
    if "```" in t or has(CODE_KW):
        return "code"
    if has(MATH_KW):
        return "math"
    if has(REASON_KW):
        return "reasoning"
    if has(CREATIVE_KW):
        return "creative"
    if approx_tokens <= 16:
        return "cheap_chat"
    return "general"


def load_catalog(path: Path) -> dict[str, float]:
    cat = yaml.safe_load(Path(path).read_text())["models"]
    return {m["id"]: float(m.get("cost_per_1k", 0.0)) for m in cat}


def resolve(model: str, catalog: dict[str, float]) -> str | None:
    if model in SKIP_MODELS:
        return None
    model = ALIASES.get(model, _DATE_SUFFIX.sub("", model))
    return model if model in catalog else None


def user_text(content) -> str:
    """Prompt text from a user-turn `content` (str or block list). Tool-result-
    only turns (no text block) return '' and are ignored by the pairing rule."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b["text"] for b in content
                 if isinstance(b, dict) and b.get("type") in ("text", "input_text") and b.get("text")]
        return "\n".join(parts).strip()
    return ""


def is_noise(text: str) -> bool:
    if len(text) < 12:
        return True
    return text.startswith(("<command-", "<local-command", "Caveat:", "[Request interrupted"))


def _row(prompt: str, model: str, usage: dict, catalog: dict[str, float]) -> dict:
    approx = max(1, len(prompt) // 4)
    tok = int((usage or {}).get("input_tokens", 0) or 0) + int((usage or {}).get("output_tokens", 0) or 0)
    return {
        "prompt": prompt[:4000],
        "model": model,
        "quality": 1.0,  # PROXY
        "cost": round(tok / 1000.0 * catalog[model], 6),
        "latency_ms": 0,
        "task": classify(prompt, approx),
    }


def iter_claude_rows(records: Iterable[dict], catalog: dict[str, float]) -> Iterator[dict]:
    """Pair each real user prompt with the model of the first assistant reply
    that follows it. One row per user prompt."""
    pending = None
    for d in records:
        if not isinstance(d, dict):
            continue
        t = d.get("type")
        msg = d.get("message") or {}
        if t == "user":
            txt = user_text(msg.get("content"))
            if txt and not is_noise(txt):
                pending = txt
        elif t == "assistant" and pending is not None:
            model = resolve(msg.get("model", ""), catalog)
            if model:
                yield _row(pending, model, msg.get("usage") or {}, catalog)
            pending = None


def parse_jsonl_records(lines: Iterable[str]) -> Iterator[dict]:
    """Parse a stream of JSONL text lines into records, skipping bad lines."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
