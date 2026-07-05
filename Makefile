PYTHON ?= python
HOST ?= 0.0.0.0
PORT ?= 8000
DATA ?= data/synthetic_eval.jsonl

.PHONY: setup setup-gpu doctor smoke generate official-data quality test eval experiments report demo docker-build-gpu docker-up-gpu docker-down-gpu

setup:
	$(PYTHON) -m pip install -e ".[dev]"

setup-gpu:
	bash scripts/bootstrap_remote.sh

doctor:
	$(PYTHON) scripts/gpu_doctor.py

smoke:
	bash scripts/smoke_remote.sh

generate:
	$(PYTHON) scripts/generate_data.py --config configs/generation.yaml

official-data:
	$(PYTHON) scripts/build_agentdog_data.py --config configs/agentdog_data_flows.yaml --source agentdog10

quality:
	$(PYTHON) scripts/quality_check.py $(DATA)

test:
	pytest

eval:
	$(PYTHON) scripts/evaluate.py $(DATA) --mode layered

experiments:
	$(PYTHON) scripts/run_experiments.py --data $(DATA) --no-api

report:
	$(PYTHON) scripts/generate_report.py --input reports/experiments.json --output reports/experiment_report.md

demo:
	$(PYTHON) scripts/serve_demo.py --host $(HOST) --port $(PORT)

docker-build-gpu:
	docker compose -f docker-compose.gpu.yml build

docker-up-gpu:
	docker compose -f docker-compose.gpu.yml up -d --build

docker-down-gpu:
	docker compose -f docker-compose.gpu.yml down
