.PHONY: install lint typecheck test format check run

install:
	poetry install --with dev

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

test:
	poetry run pytest -q

format:
	poetry run ruff format .

check: lint typecheck test
	poetry check

run:
	poetry run python -m sensor_intelligence.cli
