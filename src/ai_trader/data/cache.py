from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_trader.shared.bars import normalize_bars
from ai_trader.shared.instruments import AssetClass

CACHE_DIR = Path(".cache") / "bars"
PARQUET_ENGINE = "pyarrow"

# El prefijo con el que se guardan las cripto. Existe porque "BTC/USDT" y un ticker de
# renta variable comparten espacio de nombres y podrian colisionar.
CRYPTO_CACHE_PREFIX = "crypto::"


def cache_symbol(symbol: str, asset_class: AssetClass) -> str:
    """
    La clave con la que un simbolo entra en la cache. UNA sola definicion.

    Quien la necesita son tres: el servicio de datos que escribe la cache, el modulo de
    barras 1H que lee la misma cache con otro timeframe, y los estudios que cargan de
    disco sin tocar la red. Con la regla escrita tres veces, que una se despiste crea un
    SEGUNDO fichero para el mismo dato, y entonces el repo tiene dos caches divergentes
    del mismo par sin que ningun test lo note: cada copia seguiria siendo consistente
    consigo misma.
    """
    if asset_class == AssetClass.CRYPTO:
        return f"{CRYPTO_CACHE_PREFIX}{symbol}"
    return symbol


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
