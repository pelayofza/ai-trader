"""
Barras 1H de cripto sobre la MISMA cache en disco que usa el sistema diario.

El sistema entero es diario (barras 1D, purga en dias, anualizacion 365) y esto no lo
cambia: es la puerta de entrada para MEDIR lo que pasa dentro de una barra diaria, no
para operar con ello. Lo consume `backtest.session_study`.

Dos cosas que hacen que reusar la cache sea seguro y no un parche:

1. LA CLAVE LLEVA EL PREFIJO DE CLASE. `MarketDataService._build_cache_symbol` guarda las
   cripto como `crypto::BTC/USDT`, no como `BTC/USDT`. Si aqui se usara el simbolo pelado
   se crearia un SEGUNDO fichero para el mismo par y el repo tendria dos caches
   divergentes del mismo dato. Por eso `cache_key` esta aqui y replica esa regla.

2. EL TIMEFRAME VA EN EL NOMBRE DEL FICHERO. `cache.cache_path` compone
   `<simbolo>_<timeframe>.parquet`, asi que `CRYPTO__BTC_USDT_1H.parquet` y
   `CRYPTO__BTC_USDT_1D.parquet` conviven sin pisarse. Descargar 1H no invalida ni
   contamina el historico diario que consumen el backtest y los estudios ya publicados.

Convenio de ventana: `[start, end)`, con el final EXCLUSIVO. Es distinto del diario
(`MarketDataService` recorta hasta la ultima vela CERRADA restando un dia) porque aqui
la ventana siempre se pide en fronteras de dia UTC y lo natural es que `end` sea la
primera hora que ya NO entra. Un estudio que pida [2020-01-01, 2026-01-01) recibe
exactamente los seis anos completos, sin una hora de mas ni de menos.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ai_trader.data.cache import load_bars, save_bars
from ai_trader.shared.bars import normalize_bars

logger = logging.getLogger(__name__)

# Etiqueta con la que se guarda en la cache (va en el nombre del fichero).
INTRADAY_TIMEFRAME = "1H"
# Lo que entiende CCXT. Son dos constantes distintas a proposito: la etiqueta de la cache
# es nuestra y el timeframe es del exchange; acoplarlas invitaria a renombrar ficheros al
# cambiar de proveedor.
CCXT_TIMEFRAME = "1h"

BAR = pd.Timedelta(hours=1)

# El mismo prefijo que pone `MarketDataService._build_cache_symbol` para AssetClass.CRYPTO.
CACHE_PREFIX = "crypto::"


class HourlyProvider:
    """Lo unico que este modulo necesita de un proveedor de barras (lo cumple CCXTCrypto)."""

    def get_ohlcv(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1h"
    ) -> pd.DataFrame:  # pragma: no cover - protocolo
        ...


def cache_key(symbol: str) -> str:
    """La clave de cache de un par cripto. Replica la regla de `MarketDataService`."""
    return f"{CACHE_PREFIX}{symbol.strip().upper()}"


def to_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def slice_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Recorte `[start, end)`: el final es EXCLUSIVO (ver docstring del modulo)."""
    return df[(df.index >= to_utc(start)) & (df.index < to_utc(end))]


def load_cached_hourly(symbol: str) -> pd.DataFrame | None:
    """Todo lo que hay en disco para el par, sin recortar. None si no hay nada."""
    cached = load_bars(cache_key(symbol), timeframe=INTRADAY_TIMEFRAME)
    if cached is None or cached.empty:
        return None
    return cached


def get_hourly_bars(
    symbol: str,
    start,
    end,
    provider: HourlyProvider | None = None,
) -> pd.DataFrame | None:
    """
    Barras 1H del par en `[start, end)`, sirviendo de cache y descargando solo lo que falte.

    `provider=None` es el modo OFFLINE: se devuelve lo que haya en disco y nada mas. Es lo
    que permite reproducir el estudio sin red y sin depender de que el exchange siga
    sirviendo el mismo historico dentro de un ano.

    La extension se hace por los DOS extremos (cabecera y cola) porque la cache puede
    haberse construido en una ejecucion anterior con una ventana mas corta, y volver a
    bajar los seis anos enteros por haber movido `--start` un mes seria tirar una hora de
    descarga cada vez.
    """
    start_ts, end_ts = to_utc(start), to_utc(end)
    if end_ts <= start_ts:
        raise ValueError(f"Ventana vacia para {symbol}: [{start_ts}, {end_ts})")

    cached = load_cached_hourly(symbol)

    if provider is None:
        if cached is None:
            return None
        window = slice_window(cached, start_ts, end_ts)
        return window if not window.empty else None

    frames: list[pd.DataFrame] = []
    if cached is None:
        fetched = _fetch(provider, symbol, start_ts, end_ts)
        if fetched is None or fetched.empty:
            return None
        frames.append(fetched)
    else:
        frames.append(cached)
        first, last = cached.index.min(), cached.index.max()
        # La ultima barra cubierta es `last`; la ventana pide hasta `end_ts - BAR`.
        if start_ts < first:
            head = _fetch(provider, symbol, start_ts, first)
            if head is not None and not head.empty:
                frames.append(head)
        if end_ts - BAR > last:
            tail = _fetch(provider, symbol, last, end_ts)
            if tail is not None and not tail.empty:
                frames.append(tail)

    merged = merge_frames(frames)
    if merged.empty:
        return None

    save_bars(cache_key(symbol), merged, timeframe=INTRADAY_TIMEFRAME)
    window = slice_window(merged, start_ts, end_ts)
    return window if not window.empty else None


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union normalizada de trozos que pueden solapar. Gana el ultimo, como en el diario."""
    prepared = [normalize_bars(f) for f in frames if f is not None and not f.empty]
    if not prepared:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    merged = pd.concat(prepared).sort_index()
    return merged[~merged.index.duplicated(keep="last")]


def _fetch(
    provider: HourlyProvider, symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame | None:
    logger.info(
        "Descargando 1H de %s: %s -> %s", symbol, start.isoformat(), end.isoformat()
    )
    try:
        fetched = provider.get_ohlcv(
            symbol=symbol,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            timeframe=CCXT_TIMEFRAME,
        )
    except Exception as exc:  # noqa: BLE001 - un par que el exchange no sirve se declara, no tumba
        logger.warning("Sin barras 1H para %s: %s", symbol, exc)
        return None
    if fetched is None or fetched.empty:
        return None
    return normalize_bars(fetched)
