from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_trader.shared.bars import normalize_bars

CACHE_DIR = Path(".cache") / "bars"
PARQUET_ENGINE = "pyarrow"


def _safe_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "_").replace(":", "_")


def cache_path(symbol: str, timeframe: str = "1D") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_safe_symbol(symbol)}_{timeframe}.parquet"


def load_bars(symbol: str, timeframe: str = "1D") -> pd.DataFrame | None:
    parquet_file = cache_path(symbol, timeframe)

    if not parquet_file.exists():
        return None

    return normalize_bars(pd.read_parquet(parquet_file, engine=PARQUET_ENGINE))


def save_bars(symbol: str, df: pd.DataFrame, timeframe: str = "1D") -> None:
    normalized = normalize_bars(df)
    parquet_file = cache_path(symbol, timeframe)
    tmp_file = parquet_file.with_suffix(".parquet.tmp")

    normalized.to_parquet(tmp_file, engine=PARQUET_ENGINE, index=True)
    tmp_file.replace(parquet_file)
