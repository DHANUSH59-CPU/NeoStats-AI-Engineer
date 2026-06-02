"""Small shared helpers used across modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalar/array types.

    pandas/numpy computations produce numpy types (np.int64, np.float64, ...)
    that the stdlib json module refuses to serialize. This encoder converts
    them to native Python types so we can dump EDA summaries, metrics, etc.
    """

    def default(self, o: Any):  # noqa: D102
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def write_json(path: str | Path, data: Any) -> None:
    """Write `data` to `path` as pretty JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyJSONEncoder)


def read_json(path: str | Path) -> Any:
    """Read JSON from `path`; return None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
