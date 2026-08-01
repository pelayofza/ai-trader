from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


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
