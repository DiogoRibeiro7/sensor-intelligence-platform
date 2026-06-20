from __future__ import annotations

import importlib.util
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "end_to_end.py"


def test_end_to_end_example_runs(capsys: object) -> None:
    spec = importlib.util.spec_from_file_location("end_to_end_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()  # should run the full pipeline without raising

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "End-to-end demo" in captured.out
    assert "## Forecast" in captured.out
