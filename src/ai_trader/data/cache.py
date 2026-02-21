import os
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(".cache") / "bars"


def _safe_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "_").replace(":", "_")


def cache_path(symbol: str, timeframe: str = "1D", fmt: str = "parquet") -> Path:
    symbol = _safe_symbol(symbol)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{timeframe}.{fmt}"


def load_bars(symbol: str, timeframe: str = "1D") -> pd.DataFrame | None:
    # Prefer parquet, fallback to csv
    p_parquet = cache_path(symbol, timeframe, "parquet")
    p_csv = cache_path(symbol, timeframe, "csv")

    if p_parquet.exists():
        df = pd.read_parquet(p_parquet)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df.sort_index()

    if p_csv.exists():
        df = pd.read_csv(p_csv, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df.sort_index()

    return None


def save_bars(symbol: str, df: pd.DataFrame, timeframe: str = "1D") -> None:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()

    # Try parquet first
    try:
        p_parquet = cache_path(symbol, timeframe, "parquet")
        df.to_parquet(p_parquet)
        return
    except Exception:
        # Fallback to csv if pyarrow/fastparquet isn't available
        p_csv = cache_path(symbol, timeframe, "csv")
        df.to_csv(p_csv)