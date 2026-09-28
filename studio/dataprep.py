"""Dataset loading and profiling for uploaded model data.

Only the profile (column names, types, summary statistics and a handful of
sample rows) is sent to the LLM. The full data never leaves the machine
except through code that the agents write and run locally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json"}


def read_table(path: Path) -> dict[str, pd.DataFrame]:
    """Return {table_name: DataFrame}; Excel files yield one table per sheet."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {path.stem: pd.read_csv(path)}
    if suffix == ".tsv":
        return {path.stem: pd.read_csv(path, sep="\t")}
    if suffix == ".parquet":
        return {path.stem: pd.read_parquet(path)}
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        if len(sheets) == 1:
            return {path.stem: next(iter(sheets.values()))}
        return {f"{path.stem}__{name}": df for name, df in sheets.items()}
    if suffix == ".json":
        return {path.stem: pd.read_json(path)}
    raise ValueError(f"Unsupported data file type: {path.name}")


def _jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, str):
        return value[:80]
    return value if isinstance(value, (int, float, bool)) else str(value)[:80]


def profile_frame(df: pd.DataFrame, sample_rows: int = 5) -> dict[str, Any]:
    columns = []
    for col in df.columns:
        s = df[col]
        info: dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = s.describe()
            for key in ("mean", "std", "min", "25%", "50%", "75%", "max"):
                if key in desc:
                    info[key] = _jsonable(desc[key])
        else:
            top = s.astype(str).value_counts(dropna=True).head(5)
            info["top_values"] = {str(k)[:40]: int(v) for k, v in top.items()}
        columns.append(info)
    sample = df.head(sample_rows).to_dict(orient="records")
    return {
        "rows": int(len(df)),
        "columns": columns,
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": [{str(k): _jsonable(v) for k, v in row.items()} for row in sample],
    }


def profile_directory(data_dir: Path, sample_rows: int = 5) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for path in sorted(data_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        try:
            for name, df in read_table(path).items():
                tables[name] = {"file": path.name, **profile_frame(df, sample_rows)}
        except Exception as exc:  # noqa: BLE001 - report any reader failure
            errors[path.name] = f"{type(exc).__name__}: {exc}"[:300]
    return {"tables": tables, "errors": errors}
