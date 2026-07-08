"""Row-conversion + metric tests for the benchmarks pipeline (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.build_profiles import map_task, rows_from_records  # noqa: E402
from benchmarks.replicate import aiq, upper_hull  # noqa: E402

# Tiny inline fixture mimicking the RouterBench wide frame: 2 prompts x 2 models,
# each with a bare score column and a "<model>|total_cost" column.
SUITE = {
    "schema": {"prompt": "prompt", "task": "eval_name", "cost_suffix": "|total_cost"},
    "models": ["gpt-4-1106-preview", "mistralai/mistral-7b-chat"],
    "model_map": {},
    "task_map": {"grade-school-math": "math", "": "general"},
}
RECORDS = [
    {
        "prompt": "2+2?",
        "eval_name": "grade-school-math",
        "gpt-4-1106-preview": 1.0,
        "gpt-4-1106-preview|total_cost": 0.003,
        "mistralai/mistral-7b-chat": 0.0,
        "mistralai/mistral-7b-chat|total_cost": 0.00004,
    },
    {
        "prompt": "hello",
        "eval_name": "some_chat_bench",
        "gpt-4-1106-preview": 1.0,
        "gpt-4-1106-preview|total_cost": 0.002,
        "mistralai/mistral-7b-chat": 1.0,
        "mistralai/mistral-7b-chat|total_cost": 0.00003,
    },
]


def test_rows_from_records_expands_per_model():
    rows = list(rows_from_records(RECORDS, SUITE))
    assert len(rows) == 4  # 2 prompts x 2 models
    r0 = next(r for r in rows if r["prompt"] == "2+2?" and r["model"] == "gpt-4-1106-preview")
    assert r0 == {
        "prompt": "2+2?",
        "model": "gpt-4-1106-preview",
        "quality": 1.0,
        "cost": 0.003,
        "latency_ms": None,
        "task": "math",
    }


def test_task_mapping_prefix_and_default():
    assert map_task("grade-school-math.dev.3", SUITE["task_map"]) == "math"
    assert map_task("some_chat_bench", SUITE["task_map"]) == "general"


def test_native_cost_preferred_over_price_map():
    rows = {(r["model"], r["prompt"]): r for r in rows_from_records(RECORDS, SUITE, price_map={"gpt-4-1106-preview": 99.0})}
    # native "<model>|total_cost" wins over any price-map fallback rate
    assert rows[("gpt-4-1106-preview", "2+2?")]["cost"] == 0.003


def test_price_map_fallback_when_no_native_cost():
    suite = {**SUITE, "schema": {"prompt": "prompt", "task": "eval_name", "cost_suffix": "|absent"}}
    rows = list(rows_from_records(RECORDS, suite, price_map={"gpt-4-1106-preview": 0.02, "mistralai/mistral-7b-chat": 0.0002}))
    got = next(r for r in rows if r["model"] == "gpt-4-1106-preview")
    assert got["cost"] == 0.02  # fell back to price map (no native cost column)


def test_missing_score_row_is_skipped():
    rec = [{"prompt": "x", "eval_name": "e", "gpt-4-1106-preview": None,
            "gpt-4-1106-preview|total_cost": 0.1,
            "mistralai/mistral-7b-chat": 0.5, "mistralai/mistral-7b-chat|total_cost": 0.01}]
    rows = list(rows_from_records(rec, SUITE))
    assert [r["model"] for r in rows] == ["mistralai/mistral-7b-chat"]  # NaN score dropped


def test_upper_hull_drops_dominated_points():
    # B is dominated (higher cost, lower quality than A); hull keeps A and C.
    pts = [(0.1, 0.5), (0.2, 0.4), (0.3, 0.9)]
    hull = upper_hull(pts)
    assert (0.2, 0.4) not in hull
    assert (0.1, 0.5) in hull and (0.3, 0.9) in hull


def test_aiq_is_average_quality_over_cost_span():
    # flat frontier at quality 0.8 -> average quality 0.8 regardless of cost span
    assert aiq([(0.0, 0.8), (1.0, 0.8)]) == 0.8
    # linear ramp 0.0->1.0 over cost 0..1 -> mean 0.5
    assert abs(aiq([(0.0, 0.0), (1.0, 1.0)]) - 0.5) < 1e-9


def test_upper_hull_single_point():
    assert upper_hull([(0.5, 0.7)]) == [(0.5, 0.7)]
    assert aiq([(0.5, 0.7)]) == 0.7
