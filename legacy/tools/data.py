"""Data source abstractions for multi-source fetching.

Supports yfinance, KSH (Hungarian Central Statistics), MNB (National Bank of Hungary),
and EUROSTAT (EU statistics) with pluggable interface.
"""
from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests
except ImportError:
    requests = None


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def fetch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for symbols between start and end dates.

        Args:
            symbols: List of symbol identifiers (e.g., ['AAPL', 'HU_CPI'])
            start: Start date in ISO format (YYYY-MM-DD)
            end: End date in ISO format (YYYY-MM-DD)

        Returns:
            Dict mapping symbol to pandas DataFrame with OHLCV columns
        """
        pass

    @abstractmethod
    def get_metadata(self, symbol: str) -> Dict:
        """Get metadata for a symbol (e.g., name, currency, sector).

        Args:
            symbol: Symbol identifier

        Returns:
            Dict with metadata fields
        """
        pass


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source using yfinance library."""

    def fetch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        if yf is None:
            raise ImportError("yfinance required. Install: pip install yfinance")

        data = {}
        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    continue
                # Flatten potential MultiIndex from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df.columns = [c.lower() for c in df.columns]
                needed = ["open", "high", "low", "close", "volume"]
                if not all(c in df.columns for c in needed):
                    continue
                data[symbol] = df[needed].dropna()
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol} from Yahoo Finance: {e}")
        return data

    def get_metadata(self, symbol: str) -> Dict:
        if yf is None:
            return {}
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', ''),
                'currency': info.get('currency', 'USD'),
                'sector': info.get('sector', ''),
                'country': info.get('country', ''),
                'exchange': info.get('exchange', ''),
            }
        except Exception:
            return {}


class FREDSource(DataSource):
    """Federal Reserve Economic Data (FRED) source.

    Symbols must be prefixed with 'FRED_' to route here, e.g. 'FRED_VIXCLS'.
    Works without an API key (CSV fallback). Set FRED_API_KEY for full access.
    """

    def fetch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        from tools.fred import get_series
        result = {}
        for sym in symbols:
            sid = sym.replace("FRED_", "", 1)
            try:
                s = get_series(sid, start=start, end=end)
                df = s.reset_index()
                df.columns = ["Date", "close"]
                df = df.set_index("Date")
                for col in ["open", "high", "low", "volume"]:
                    df[col] = df["close"]
                result[sym] = df[["open", "high", "low", "close", "volume"]]
            except Exception as e:
                print(f"Warning: FRED fetch failed for {sid}: {e}")
        return result

    def get_metadata(self, symbol: str) -> Dict:
        return {"name": symbol, "source": "FRED", "type": "macroeconomic"}


class KSHSource(DataSource):
    """Hungarian Central Statistical Office (KSH) data source.

    Note: Requires API key and specific dataset codes.
    Currently a stub - implement actual API calls.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def fetch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        # Stub implementation - KSH has REST API for statistical data
        # Would need to map symbols to dataset codes and handle time series
        raise NotImplementedError("KSH API integration not implemented yet")

    def get_metadata(self, symbol: str) -> Dict:
        # Stub
        return {'source': 'KSH', 'type': 'statistical'}


class MNBSource(DataSource):
    """National Bank of Hungary (MNB) data source.

    Note: MNB provides exchange rate and financial data APIs.
    Currently a stub - implement actual API calls.
    """

    def fetch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        # Stub - MNB has APIs for exchange rates, interest rates, etc.
        raise NotImplementedError("MNB API integration not implemented yet")

    def get_metadata(self, symbol: str) -> Dict:
        # Stub
        return {'source': 'MNB', 'type': 'financial'}


class EurostatSource(DataSource):
    """EUROSTAT (EU statistics) data source.

    Note: EUROSTAT provides bulk download and API access.
    Currently a stub - implement actual API calls.
    """

    def fetch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        # Stub - EUROSTAT has SDMX API and bulk downloads
        raise NotImplementedError("EUROSTAT API integration not implemented yet")

    def get_metadata(self, symbol: str) -> Dict:
        # Stub
        return {'source': 'EUROSTAT', 'type': 'economic'}


class MultiSourceFetcher:
    """Multi-source data fetcher that routes symbols to appropriate sources."""

    def __init__(self):
        self.sources = {
            'yahoo': YahooFinanceSource(),
            'fred': FREDSource(),
            'ksh': KSHSource(),
            'mnb': MNBSource(),
            'eurostat': EurostatSource(),
        }

    def fetch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Fetch data from multiple sources based on symbol routing."""
        # Simple routing: if symbol contains country codes or keywords
        routed = self._route_symbols(symbols)

        all_data = {}
        for source_name, syms in routed.items():
            if source_name in self.sources:
                try:
                    data = self.sources[source_name].fetch(syms, start, end)
                    all_data.update(data)
                except NotImplementedError:
                    print(f"Warning: {source_name} source not implemented")
                except Exception as e:
                    print(f"Warning: Failed to fetch from {source_name}: {e}")

        return all_data

    def _route_symbols(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Route symbols to sources based on heuristics."""
        yahoo_symbols = []
        ksh_symbols = []
        mnb_symbols = []
        eurostat_symbols = []

        fred_symbols = []
        for symbol in symbols:
            symbol_upper = symbol.upper()
            if symbol_upper.startswith('FRED_'):
                fred_symbols.append(symbol)
            elif 'HU_' in symbol_upper or 'HUNGARY' in symbol_upper:
                ksh_symbols.append(symbol)
            elif 'HUF' in symbol_upper or 'EUR' in symbol_upper or 'EXCHANGE' in symbol_upper:
                mnb_symbols.append(symbol)
            elif 'EU_' in symbol_upper or 'EUROPE' in symbol_upper:
                eurostat_symbols.append(symbol)
            else:
                # Default to Yahoo Finance for stocks
                yahoo_symbols.append(symbol)

        return {
            'yahoo': yahoo_symbols,
            'fred': fred_symbols,
            'ksh': ksh_symbols,
            'mnb': mnb_symbols,
            'eurostat': eurostat_symbols,
        }

    def get_metadata(self, symbol: str) -> Dict:
        """Get metadata for a symbol by trying all sources."""
        for source in self.sources.values():
            try:
                meta = source.get_metadata(symbol)
                if meta:
                    return meta
            except Exception:
                continue
        return {}


# Convenience function for backward compatibility
def fetch_yahoo(
    symbols: List[str],
    start: str = "2015-01-01",
    end: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Legacy function - use MultiSourceFetcher instead."""
    source = YahooFinanceSource()
    if end is None:
        end = "2030-01-01"  # Far future
    return source.fetch(symbols, start, end)