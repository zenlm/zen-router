"""Build zen-router eval rows from THIS machine's real AI-assistant usage.

Sources
  ~/.claude/projects/**/*.jsonl   Claude Code sessions
  ~/.codex/sessions/**/*.jsonl    Codex sessions (response_item messages)

Output (one JSON object per line), the eval-row shape build_dataset.py consumes:
  {prompt, model, quality, cost, latency_ms, task}
    quality    -- PROXY, fixed 1.0 (only the served model is observed)
    cost       -- (input+output tokens)/1000 * catalog cost_per_1k
    latency_ms -- 0 (not recorded in the logs)
    task       -- keyword heuristic, a port of hanzo-router's Heuristic::classify

Model ids are kept REAL and mapped 1:1 to catalog ids. Shared pairing/mapping
logic lives in session_common.py (also used by extract_agentic_dataset.py).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from session_common import iter_claude_rows, load_catalog, parse_jsonl_records, resolve, user_text


def parse_claude_file(path: Path, catalog, out: list) -> None:
    out.extend(iter_claude_rows(parse_jsonl_records(path.open(errors="ignore")), catalog))


def parse_codex_file(path: Path, catalog, out: list) -> None:
    """Codex logs: response_item messages; model from turn_context/session_meta."""
    pending, model = None, None
    for d in parse_jsonl_records(path.open(errors="ignore")):
        p = d.get("payload") or {}
        if isinstance(p, dict) and p.get("model"):
            model = resolve(p["model"], catalog) or model
        if d.get("type") != "response_item" or not isinstance(p, dict) or p.get("type") != "message":
            continue
        role = p.get("role")
        if role == "user":
            txt = user_text(p.get("content"))
            if txt:
                pending = txt
        elif role == "assistant" and pending is not None and model:
            approx = max(1, len(pending) // 4)
            from session_common import classify
            out.append({"prompt": pending[:4000], "model": model, "quality": 1.0,
                        "cost": 0.0, "latency_ms": 0, "task": classify(pending, approx)})
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
        parse_codex_file(f, catalog, rows)
    n_codex = len(rows)
    for f in sorted(args.claude.rglob("*.jsonl")):
        if len(rows) >= args.max_rows:
            break
        parse_claude_file(f, catalog, rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} eval rows -> {args.out}  (codex={n_codex}, claude={len(rows) - n_codex})")
    print("by model:", dict(Counter(r["model"] for r in rows).most_common()))
    print("by task :", dict(Counter(r["task"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
