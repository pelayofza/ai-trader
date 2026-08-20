"""
El historico REAL de mercado, tal y como lo consumen los estudios y el scoring.

Vivia dentro del estudio de fidelidad del generador sintetico, que es el sitio donde se
escribio por primera vez. Eso lo dejaba en una posicion absurda: el lado REAL de cuatro
estudios -fidelidad, transferencia, capa tematica y, desde ahora, el ranking que decide-
entraba por un modulo cuyo nombre habla del sintetico. Al aparcar la linea sintetica habia
que separarlos o el dato real se iba de viaje con ella.

Dos piezas y una constante:

- `CachedBarsProvider`: lee SOLO la cache en disco (`--offline`). Reproduce cualquier
  estudio sin red y sin depender de que el exchange siga sirviendo el mismo historico.
  Recorta hasta la ultima vela diaria CERRADA, exactamente igual que MarketDataService: si
  no, el modo offline devolveria una barra mas que el online y no seria el mismo estudio.
- `fetch_real_bars`: historico diario por simbolo, OMITIENDO y DECLARANDO los que el
  exchange no sirve. No se rellenan ni se sustituyen: un par deslistado no tiene contraparte.
- `DEFAULT_REAL_START` / `DEFAULT_REAL_END`: la ventana historica es una constante CERRADA,
  no "hasta hoy". Si el final se moviera con la fecha de ejecucion, ningun informe seria
  reproducible y dos regeneraciones no serian comparables.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from ai_trader.data.cache import load_bars as load_cached_bars

if TYPE_CHECKING:  # solo para anotar: en ejecucion se importa tarde a proposito, porque
    # arrastra los proveedores de stocks y de mercados de prediccion que esto no usa.
    from ai_trader.data.market_data import MarketDataService

logger = logging.getLogger(__name__)

DEFAULT_EXCHANGE = "binance"
# Ventana historica CERRADA: arranca cuando Binance ya tiene profundidad en la mayoria de
# los pares y termina en un corte fijo, no en "hoy".
DEFAULT_REAL_START = "2017-09-01"
DEFAULT_REAL_END = "2026-01-01"


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class CachedBarsProvider:
    """
    Fuente de barras que SOLO lee la cache en disco (`--offline`).

    Permite reproducir el estudio sin red -y sin depender de que el exchange siga
    sirviendo el mismo historico- una vez los datos se han descargado alguna vez.
    """

    def get_daily_bars(self, symbol: str, start: datetime, end: datetime):
        cached = load_cached_bars(f"crypto::{symbol.strip().upper()}", timeframe="1D")
        if cached is None or cached.empty:
            return None
        # Mismos limites que MarketDataService: hasta la ultima vela diaria CERRADA. Si
        # no, el modo offline devolveria una barra mas que el online y el estudio no
        # seria el mismo estudio.
        first = _utc(start).normalize()
        last = _utc(end).normalize() - pd.Timedelta(days=1)
        return cached.loc[first : max(first, last)]


def build_service(exchange: str, *, offline: bool) -> MarketDataService | CachedBarsProvider:
    """El origen de barras reales del estudio: la cache a secas si `offline`, y si no el
    servicio completo contra el exchange."""
    if offline:
        return CachedBarsProvider()
    # Import tardio: construir el servicio arrastra los proveedores de stocks y de
    # mercados de prediccion, que este estudio no usa.
    from ai_trader.data.market_data import MarketDataService
    from ai_trader.data.providers.ccxt_crypto import CCXTCrypto, CCXTCryptoConfig

    return MarketDataService(
        crypto_provider=CCXTCrypto(CCXTCryptoConfig(exchange_id=exchange))
    )


def fetch_real_bars(
    symbols: Sequence[str], start: datetime, end: datetime, service
) -> dict[str, pd.DataFrame]:
    """
    Historico diario real por simbolo. Los que el exchange no sirve se OMITEN y se
    declaran; no se sustituyen por nada (un activo deslistado, como MATIC/USDT en
    Binance, simplemente no tiene contraparte real que comparar).
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = service.get_daily_bars(symbol, start, end)
        except Exception as exc:  # noqa: BLE001 - un simbolo caido no tumba el estudio
            logger.warning("Sin datos reales para %s: %s", symbol, exc)
            continue
        if df is None or df.empty:
            logger.warning("Sin datos reales para %s (respuesta vacia)", symbol)
            continue
        out[symbol] = df
        logger.info(
            "  %-12s %d barras reales | %s -> %s",
            symbol, len(df), df.index.min().date(), df.index.max().date(),
        )
    return out
