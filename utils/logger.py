from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CSV_DIR, FIGURES_DIR, MODELS_DIR


def ensure_result_dirs() -> None:
    for path in (MODELS_DIR, CSV_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_rows_csv(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
