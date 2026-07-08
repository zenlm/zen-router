"""Build zen-router eval rows from THIS machine's real AI-assistant usage.

Sources
  ~/.claude/projects/**/*.jsonl   Claude Code sessions (type:"assistant" lines
                                  carry message.model + message.usage; the user
                                  turn before them carries the prompt text)
  ~/.codex/sessions/**/*.jsonl    Codex sessions (response_item messages;
                                  turn_context/session_meta carry the model)

Output (one JSON object per line), the eval-row shape build_dataset.py consumes:
  {prompt, model, quality, cost, latency_ms, task}
    quality    -- PROXY, fixed 1.0 (the served model is the only observation we
                  have; we cannot measure counterfactual quality of other models)
    cost       -- (input+output tokens)/1000 * catalog cost_per_1k for the model
    latency_ms -- 0 (Claude/Codex logs do not record wall-clock serve latency)
    task       -- heuristic keyword labeling, a direct port of hanzo-router's
                  `Heuristic::classify` (same taxonomy the serving router uses)

Model ids are kept REAL and mapped 1:1 to catalog ids (no collapsing into
generic tiers), so the trained route head's label space contains the actual
proprietary models this machine used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# session model id -> catalog id. Identity for ids already in the catalog; a few
# short aliases the logs use, plus <synthetic> which is not a real serve.
ALIASES = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "gpt-5.4-mini": "gpt-5.4-mini",
}
SKIP_MODELS = {"<synthetic>"}

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
    cat = yaml.safe_load(path.read_text())["models"]
    return {m["id"]: float(m.get("cost_per_1k", 0.0)) for m in cat}


def resolve(model: str, catalog: dict[str, float]) -> str | None:
    if model in SKIP_MODELS:
        return None
    model = ALIASES.get(model, model)
    return model if model in catalog else None


def user_text(content) -> str:
    """Extract plain prompt text from a user-turn `content` (str or block list).
    Tool-result-only turns (no text block) return '' and are ignored."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("text", "input_text") and b.get("text"):
                parts.append(b["text"])
        return "\n".join(parts).strip()
    return ""


def is_noise(text: str) -> bool:
    if len(text) < 12:
        return True
    # skip Claude Code command/system envelopes and pasted-in tool noise.
    return text.startswith(("<command-", "<local-command", "Caveat:", "[Request interrupted"))


def emit_row(prompt: str, model: str, usage: dict, catalog: dict[str, float], out: list) -> None:
    approx = max(1, len(prompt) // 4)
    tok = 0
    if usage:
        tok = int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)
    cost = tok / 1000.0 * catalog[model]
    out.append({
        "prompt": prompt[:4000],
        "model": model,
        "quality": 1.0,  # PROXY
        "cost": round(cost, 6),
        "latency_ms": 0,
        "task": classify(prompt, approx),
    })


def parse_claude(path: Path, catalog: dict[str, float], out: list) -> None:
    pending = None  # last real user prompt not yet paired with an assistant
    for line in path.open(errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
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
                emit_row(pending, model, msg.get("usage") or {}, catalog, out)
            pending = None  # one row per user prompt (its first serving model)


def parse_codex(path: Path, catalog: dict[str, float], out: list) -> None:
    pending = None
    model = None
    for line in path.open(errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = d.get("payload") or {}
        if isinstance(p, dict) and p.get("model"):
            model = resolve(p["model"], catalog) or model
        if d.get("type") != "response_item" or not isinstance(p, dict):
            continue
        if p.get("type") != "message":
            continue
        role = p.get("role")
        if role == "user":
            txt = user_text(p.get("content"))
            if txt and not is_noise(txt):
                pending = txt
        elif role == "assistant" and pending is not None and model:
            emit_row(pending, model, {}, catalog, out)
            pending = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", type=Path, default=Path.home() / ".claude/projects")
    ap.add_argument("--codex", type=Path, default=Path.home() / ".codex/sessions")
    ap.add_argument("--catalog", type=Path, default=Path("training/catalog.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/evals.jsonl"))
    ap.add_argument("--max-rows", type=int, default=40000)
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    rows: list = []

    for f in sorted(args.codex.rglob("*.jsonl")):
        parse_codex(f, catalog, rows)
    n_codex = len(rows)

    for f in sorted(args.claude.rglob("*.jsonl")):
        if len(rows) >= args.max_rows:
            break
        parse_claude(f, catalog, rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    from collections import Counter
    bym = Counter(r["model"] for r in rows)
    byt = Counter(r["task"] for r in rows)
    print(f"wrote {len(rows)} eval rows -> {args.out}  (codex={n_codex}, claude={len(rows) - n_codex})")
    print("by model:", dict(bym.most_common()))
    print("by task :", dict(byt.most_common()))


if __name__ == "__main__":
    main()
