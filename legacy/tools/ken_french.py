"""Fama-French factor data — direct download from Kenneth R. French Data Library.

No extra dependencies beyond requests, zipfile, io, and pandas/numpy (all
already present). Does NOT use pandas-datareader.

Available factor models:
  'ff3'  — Market excess return, SMB, HML, RF (monthly)
  'ff5'  — Market excess return, SMB, HML, RMW, CMA, RF
  'mom'  — Momentum factor (UMD / WML)

Returns are in **decimal** form (divided by 100 so 1% = 0.01).

Example usage in agent code:
    from tools.ken_french import get_factors, get_portfolios, ff3_alpha
    ff3 = get_factors('ff3', start='2010-01-01', end='2023-12-31')
    result = ff3_alpha(my_monthly_returns_series)
    print(result)
"""
from __future__ import annotations

import io
import time
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import requests

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
_TIMEOUT = 30
_RETRIES = 3
_BACKOFF = 1.5

_MODEL_ZIP: dict[str, str] = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}


def _download_zip(filename: str) -> bytes:
    url = f"{_BASE}/{filename}"
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            r = requests.get(url, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF * (attempt + 1))
    raise RuntimeError(f"Ken French download failed ({url}): {last_exc}")


def _parse_french_csv(raw_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a Ken French CSV file from zip bytes.

    Format: free-text header lines, blank line, then CSV data (header starts
    with a comma because the index column has no name). Annual section follows
    after another blank line.

    Returns (monthly_df, annual_df) both with DatetimeIndex.
    Values are divided by 100 (converted to decimal).
    """
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        text = zf.read(csv_name).decode("latin-1")

    lines = text.splitlines()

    # Find every line index where the CSV header starts (line begins with ',')
    header_indices = [i for i, ln in enumerate(lines) if ln.strip().startswith(",")]

    def _read_block(start: int) -> pd.DataFrame:
        """Read from header_line to the next blank line (or EOF)."""
        block_lines = [lines[start]]
        for ln in lines[start + 1 :]:
            if ln.strip() == "":
                break
            block_lines.append(ln)
        block = "\n".join(block_lines)
        df = pd.read_csv(io.StringIO(block), index_col=0)
        df.index = df.index.astype(str).str.strip()
        df.columns = [c.strip() for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce") / 100.0
        return df.dropna(how="all")

    monthly_df = pd.DataFrame()
    annual_df = pd.DataFrame()

    if len(header_indices) >= 1:
        monthly_df = _read_block(header_indices[0])
        try:
            monthly_df.index = pd.to_datetime(monthly_df.index, format="%Y%m")
            monthly_df.index = monthly_df.index + pd.offsets.MonthEnd(0)
        except Exception:
            pass

    if len(header_indices) >= 2:
        annual_df = _read_block(header_indices[1])
        try:
            annual_df.index = pd.to_datetime(annual_df.index.str.strip(), format="%Y")
        except Exception:
            pass

    return monthly_df, annual_df


def get_factors(
    model: str = "ff3",
    start: str = "2000-01-01",
    end: str = "2030-01-01",
    freq: str = "M",
) -> pd.DataFrame:
    """Download Fama-French factor returns.

    Args:
        model: One of 'ff3', 'ff5', 'mom'.
        start: Start date (ISO format).
        end:   End date (ISO format).
        freq:  'M' for monthly (default) or 'A' for annual.

    Returns:
        DataFrame with DatetimeIndex and factor columns in decimal returns.
        FF3 columns: Mkt-RF, SMB, HML, RF
        FF5 adds:    RMW, CMA
        mom column:  Mom
    """
    key = model.lower()
    if key not in _MODEL_ZIP:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODEL_ZIP)}")
    raw = _download_zip(_MODEL_ZIP[key])
    monthly_df, annual_df = _parse_french_csv(raw)
    df = monthly_df if freq.upper() != "A" else annual_df
    df = df.sort_index()
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask]


def get_portfolios(
    name: str = "25_Portfolios_5x5",
    start: str = "2000-01-01",
    end: str = "2030-01-01",
) -> pd.DataFrame:
    """Download a Ken French portfolio-sort dataset (value-weighted monthly returns).

    Args:
        name: Dataset name from the French library, e.g.:
              '25_Portfolios_5x5', '6_Portfolios_2x3',
              '10_Portfolios_Prior_12_2'.
        start: Start date (ISO format).
        end:   End date (ISO format).

    Returns:
        DataFrame of value-weighted monthly returns in decimal form.
    """
    filename = f"{name}_CSV.zip"
    raw = _download_zip(filename)
    monthly_df, _ = _parse_french_csv(raw)
    monthly_df = monthly_df.sort_index()
    mask = (monthly_df.index >= pd.Timestamp(start)) & (monthly_df.index <= pd.Timestamp(end))
    return monthly_df.loc[mask]


def ff3_alpha(
    returns: pd.Series,
    start: str = "2000-01-01",
    end: str = "2030-01-01",
) -> dict[str, float]:
    """Compute Fama-French 3-factor alpha via OLS.

    Regresses ``returns - RF`` on Mkt-RF, SMB, HML.

    Args:
        returns: Monthly total returns as a pd.Series with DatetimeIndex.
                 Must overlap with the FF3 factor data.
        start:   Earliest date for factor data (auto-aligned with `returns`).
        end:     Latest date for factor data.

    Returns:
        dict with keys:
          alpha      — annualised intercept (monthly alpha * 12)
          alpha_t    — t-statistic of the intercept
          mkt_beta   — market factor loading
          smb_beta   — size factor loading
          hml_beta   — value factor loading
          r_squared  — in-sample R²
    """
    ff3 = get_factors("ff3", start=start, end=end)
    df = ff3.join(returns.rename("ret"), how="inner").dropna()
    if len(df) < 12:
        raise ValueError(f"Only {len(df)} overlapping observations; need at least 12.")

    excess = (df["ret"] - df["RF"]).values
    X_raw = df[["Mkt-RF", "SMB", "HML"]].values
    X = np.column_stack([np.ones(len(X_raw)), X_raw])

    b, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    resid = excess - X @ b
    n, k = len(resid), X.shape[1]
    s2 = float(np.sum(resid**2) / (n - k))
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(np.sum((excess - excess.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": float(b[0] * 12),
        "alpha_t": float(b[0] / se[0]),
        "mkt_beta": float(b[1]),
        "smb_beta": float(b[2]),
        "hml_beta": float(b[3]),
        "r_squared": float(r2),
    }
