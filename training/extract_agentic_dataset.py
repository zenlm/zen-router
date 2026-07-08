"""Build zen-router eval rows from the zen-agentic-dataset-private corpus.

The corpus is SFT-formatted (`{"messages":[system,user,assistant]}`). The routing
signal is not a top-level field: many assistant turns embed a full Claude Code
session transcript (raw JSONL, `parentUuid`/`message.model`/`message.usage`) as
their `content` string. We pull those transcripts out and run the SAME pairing
rule as the local extractor (session_common.iter_claude_rows) over them.

Streaming + stratified: we never hold the corpus in RAM. Per model we keep at
most --cap rows (caps the majority class); minority models almost never reach the
cap, so every minority row is kept. We stop once --target rows are collected or
--max-chunks files are read, whichever comes first, and report the per-family and
overall model-label distribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from session_common import iter_claude_rows, load_catalog, parse_jsonl_records


def transcripts(messages: list) -> list[str]:
    """The embedded raw-transcript strings in a sample's assistant turns."""
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str) and "parentUuid" in c:
                out.append(c)
    return out


def rows_from_line(line: str, catalog: dict[str, float]):
    try:
        messages = json.loads(line).get("messages")
    except (json.JSONDecodeError, AttributeError):
        return
    if not isinstance(messages, list):
        return
    for t in transcripts(messages):
        yield from iter_claude_rows(parse_jsonl_records(t.split("\n")), catalog)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("/Users/a/work/hanzo/hanzoai/zen-agentic-dataset-private"))
    ap.add_argument("--catalog", type=Path, default=Path("training/catalog.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/evals-agentic.jsonl"))
    ap.add_argument("--cap", type=int, default=60000, help="max rows per model (caps the majority)")
    ap.add_argument("--target", type=int, default=300000, help="stop once this many rows collected")
    ap.add_argument("--max-chunks", type=int, default=45, help="bound on I/O: files to read")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    files = sorted((args.corpus / "train_chunks").glob("*.jsonl"))[: args.max_chunks]

    per_model = Counter()
    per_family: dict[str, Counter] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w") as fh:
        for f in files:
            fam = Counter()
            for line in f.open(errors="ignore"):
                for r in rows_from_line(line, catalog):
                    m = r["model"]
                    if per_model[m] >= args.cap:
                        continue
                    per_model[m] += 1
                    fam[m] += 1
                    fh.write(json.dumps(r) + "\n")
                    total += 1
            per_family[f.name] = fam
            print(f"  {f.name}: +{sum(fam.values())} rows  {dict(fam.most_common(4))}")
            if total >= args.target:
                print(f"  reached target {args.target} after {f.name}")
                break

    print(f"\nwrote {total} eval rows -> {args.out}  (from {len(per_family)} chunks, cap={args.cap})")
    print("model-label distribution:", dict(per_model.most_common()))


if __name__ == "__main__":
    main()
