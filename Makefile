.PHONY: data train grpo eval quantize test upload bench-data bench-replicate

PYTHON ?= uv run python

# --- benchmarks: replicate published LLM-routing research (see benchmarks/) ---
BENCH_SUITE  ?= routerbench
BENCH_SAMPLE ?= 200
BENCH_DATA   ?= data/profiles-$(BENCH_SUITE).jsonl
BENCH_POLICY ?= oracle

data:
	$(PYTHON) training/build_dataset.py --evals data/evals.jsonl --ledger data/usage-ledger.jsonl --out data/routing-sft.jsonl

train:
	$(PYTHON) training/sft.py --config training/config.yaml

grpo:
	$(PYTHON) training/grpo.py --config training/config.yaml

eval:
	$(PYTHON) eval/eval_routing.py --model out/zen-router --data data/routing-eval.jsonl

quantize:
	scripts/quantize.sh out/zen-router

test:
	uv run pytest -q

upload:
	$(PYTHON) upload_to_huggingface.py --repo zenlm/zen-router --path out/zen-router

bench-data:
	uv run --with datasets,huggingface_hub,pandas python benchmarks/build_profiles.py \
		--suite $(BENCH_SUITE) --sample $(BENCH_SAMPLE) --out $(BENCH_DATA)

bench-replicate:
	$(PYTHON) benchmarks/replicate.py --data $(BENCH_DATA) --policy $(BENCH_POLICY) \
		--out-prefix out/bench/$(BENCH_SUITE)
