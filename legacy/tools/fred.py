"""FRED (Federal Reserve Economic Data) fetcher.

Works with or without an API key:
  - With key  : uses JSON API (full history, all series, faster)
  - Without   : falls back to the public CSV download endpoint (same data)

Common series IDs:
  GDP        nominal GDP (quarterly)      GDPC1    real GDP (quarterly)
  CPIAUCSL   CPI all-items (monthly)      PCE      personal consumption
  FEDFUNDS   effective fed funds rate     DGS10    10-year Treasury yield
  DGS2       2-year Treasury yield        T10Y2Y   yield-curve spread
  VIXCLS     CBOE VIX (daily)             UNRATE   unemployment rate
  PAYEMS     nonfarm payrolls (monthly)   HOUST    housing starts

Get a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set it as env var FRED_API_KEY or pass api_key= directly.
"""
from __future__ import annotations

import os
import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests

FRED_API_BASE = "https://api.stlouisfed.org/fred"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT = 30
_RETRIES = 3
_BACKOFF = 1.5


def _api_key() -> str:
    return os.getenv("FRED_API_KEY", "")


def _get(url: str, params: dict) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF * (attempt + 1))
    raise RuntimeError(f"FRED request failed after {_RETRIES} attempts: {last_exc}")


def get_series(
    series_id: str,
    start: str = "2000-01-01",
    end: str = "2030-01-01",
    api_key: str | None = None,
) -> pd.Series:
    """Fetch a single FRED series.

    Returns a pd.Series with a DatetimeIndex and the series_id as its name.
    Dots (``.) in the data, which FRED uses for missing values, are coerced
    to NaN and dropped.
    """
    key = api_key or _api_key()
    if key:
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        r = _get(f"{FRED_API_BASE}/series/observations", params)
        obs = r.json().get("observations", [])
        dates = pd.to_datetime([o["date"] for o in obs])
        vals = pd.to_numeric([o["value"] for o in obs], errors="coerce")
        return pd.Series(vals.values, index=dates, name=series_id).dropna()
    else:
        params = {"id": series_id}
        r = _get(FRED_CSV_BASE, params)
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        df.columns = [series_id]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df[series_id].loc[start:end]


def get_multiple(
    series_ids: list[str],
    start: str = "2000-01-01",
    end: str = "2030-01-01",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch multiple FRED series and align them into a single DataFrame.

    Columns are named by series_id; index is a daily DatetimeIndex.
    Columns that differ in frequency (e.g. daily VIX + monthly CPI) are
    aligned with forward-fill from the lower-frequency series.
    """
    frames = {sid: get_series(sid, start, end, api_key) for sid in series_ids}
    return pd.DataFrame(frames)


def series_info(series_id: str, api_key: str | None = None) -> dict:
    """Return metadata for a series (title, units, frequency, etc.).

    Requires an API key; raises RuntimeError without one.
    """
    key = api_key or _api_key()
    if not key:
        raise RuntimeError("FRED API key required for series_info(). Set FRED_API_KEY env var.")
    params = {"series_id": series_id, "api_key": key, "file_type": "json"}
    r = _get(f"{FRED_API_BASE}/series", params)
    seriess = r.json().get("seriess", [])
    return seriess[0] if seriess else {}
