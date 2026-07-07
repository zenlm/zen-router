.PHONY: data train grpo eval quantize test upload

PYTHON ?= uv run python

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
