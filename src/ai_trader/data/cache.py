from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache") / "bars"
PARQUET_ENGINE = "pyarrow"


def _safe_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "_").replace(":", "_")


def cache_path(symbol: str, timeframe: str = "1D") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_safe_symbol(symbol)}_{timeframe}.parquet"


def _legacy_csv_path(symbol: str, timeframe: str = "1D") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_safe_symbol(symbol)}_{timeframe}.csv"


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index, utc=True, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def load_bars(symbol: str, timeframe: str = "1D") -> pd.DataFrame | None:
    parquet_file = cache_path(symbol, timeframe)

    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file, engine=PARQUET_ENGINE)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet cache requires pyarrow. Install project dependencies with Poetry."
            ) from exc
        return _normalize_bars(df)

    legacy_csv = _legacy_csv_path(symbol, timeframe)
    if legacy_csv.exists():
        df = pd.read_csv(legacy_csv, index_col=0, parse_dates=True)
        df = _normalize_bars(df)

        save_bars(symbol, df, timeframe=timeframe)

        try:
            legacy_csv.unlink()
        except OSError:
            logger.warning("Could not delete legacy CSV cache file: %s", legacy_csv)

        return df

    return None


def save_bars(symbol: str, df: pd.DataFrame, timeframe: str = "1D") -> None:
    normalized = _normalize_bars(df)
    parquet_file = cache_path(symbol, timeframe)
    tmp_file = parquet_file.with_suffix(".parquet.tmp")

    try:
        normalized.to_parquet(
            tmp_file,
            engine=PARQUET_ENGINE,
            index=True,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Parquet cache requires pyarrow. Install project dependencies with Poetry."
        ) from exc

    tmp_file.replace(parquet_file)