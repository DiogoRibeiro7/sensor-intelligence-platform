# Contributing

Thanks for taking the time to improve Sensor Intelligence Platform. This project is
structured as production-style ML software, so changes should preserve typed package code,
tests, and time-ordering guarantees.

## Development Setup

Requirements:

- Python 3.11 or 3.12
- Poetry

```bash
git clone https://github.com/DiogoRibeiro7/sensor-intelligence-platform.git
cd sensor-intelligence-platform
poetry install --with dev
poetry run pre-commit install
```

## Quality Checks

Run these before opening a pull request:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

For a quick local smoke test:

```bash
poetry run python examples/end_to_end.py
```

## Project Rules

- Keep modelling code in reusable modules under `src/sensor_intelligence`.
- Preserve time ordering. Do not use future data in training features.
- Every model or detector change needs evaluation-oriented tests or metrics.
- Every alert must carry a severity and at least one reason code.
- Keep notebooks as reproducible narratives over package code, not the implementation source.
- Prefer small, focused pull requests with clear motivation and test evidence.

## Pull Request Checklist

- The change is documented in README, docs, or docstrings when user-facing behavior changes.
- Tests cover new behavior and regression risk.
- `ruff`, `mypy`, and `pytest` pass locally.
- Dependency changes are justified and reflected in `poetry.lock`.
- Security-sensitive changes mention risks and mitigations in the PR description.
