# Peel — one venv at the repo root (uv, Python 3.11) and one npm project in web/.
PY := .venv/bin/python
API_PORT ?= 7860

.PHONY: setup dev api web test lint format openapi examples eval build deploy clean

setup:
	uv venv --python 3.11 --allow-existing .venv
	uv pip install --python $(PY) -e "engine[dev]" -r api/requirements.txt -r eval/requirements.txt
	cd web && npm install

api:
	cd api && ../$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port $(API_PORT) --reload --reload-dir app --reload-dir ../engine/peel

web:
	cd web && npm run dev

dev:
	$(MAKE) -j2 api web

test:
	cd engine && ../$(PY) -m pytest
	cd api && ../$(PY) -m pytest
	cd web && npm run typecheck && npm run lint

lint:
	cd engine && ../.venv/bin/ruff check peel tests && ../.venv/bin/ruff format --check peel tests && ../.venv/bin/mypy -p peel
	.venv/bin/ruff check api eval deploy && .venv/bin/ruff format --check api eval deploy
	cd web && npm run typecheck && npm run lint

format:
	cd engine && ../.venv/bin/ruff format peel tests && ../.venv/bin/ruff check peel tests --fix
	.venv/bin/ruff format api eval deploy && .venv/bin/ruff check api eval deploy --fix

openapi:
	cd api && ../$(PY) -m app.export_openapi
	cd web && npm run gen:types

examples:
	$(PY) -m eval.build_examples

eval:
	$(PY) -m eval.run_all
	$(PY) -m eval.report

build:
	cd web && npm run build

deploy:
	$(PY) deploy/deploy_space.py $(SPACE)

clean:
	rm -rf web/dist eval/out .pytest_cache engine/.pytest_cache api/.pytest_cache
