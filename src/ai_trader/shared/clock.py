from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

import pandas as pd


class Clock(Protocol):
    """
    Fuente de tiempo del sistema.

    Existe para que el motor pueda correr contra el reloj real o contra un reloj
    simulado que avanza sobre datos historicos, sin que las estrategias ni el
    orquestador sepan cual de los dos tienen delante.
    """

    def now(self) -> datetime:
        ...


class LiveClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class HistoricalClock:
    """
    Reloj simulado. El motor de backtest lo posiciona en cada dia y el resto del
    sistema cree que ese es el 'ahora'. Es la costura que permite correr el mismo
    orquestador contra historico sin que estrategias ni runner se enteren.
    """

    def __init__(self, current: datetime) -> None:
        self._current = _as_utc(current)

    def now(self) -> datetime:
        return self._current

    def set(self, moment: datetime) -> None:
        self._current = _as_utc(moment)

    def advance(self, days: int = 1) -> None:
        self._current = self._current + timedelta(days=days)


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def visible_cutoff(now: datetime) -> pd.Timestamp:
    """
    Frontera anti look-ahead del sistema: MEDIANOCHE UTC del dia del reloj. Todo lo que
    sea estrictamente ANTERIOR a este instante es visible; el dia de hoy, no.

    Es la misma regla que aplica `HistoricalDataSource.get_daily_bars` a las barras: la
    barra de hoy aun no ha cerrado desde el punto de vista de la decision, asi que ningun
    otro canal de informacion —regimen, senales externas— puede verla tampoco. Si cada
    consumidor la reimplementa, basta con que uno escriba `<=` en vez de `<` para que la
    evidencia de ese consumidor deje de ser comparable con la de los demas.

    Vive aqui, junto al reloj, porque es una propiedad del TIEMPO del sistema y no de las
    barras: `observation/regime.py` la tenia copiada y el paquete `signals/` la
    necesitaba, que habria sido la tercera copia.

    LIMITE DECLARADO: corta por el dia de la OBSERVACION, no por el de su publicacion.
    Para una fuente que revisa hacia atras (`pit='archive_revisable'` en el catalogo de
    senales) esto no basta por si solo: lo que hoy dice el proveedor sobre el martes
    pasado puede no ser lo que decia el martes pasado. Lo que protege ahi es el archivo
    crudo con su `fetched_at`, no esta funcion.
    """
    ts = pd.Timestamp(now)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.normalize()
