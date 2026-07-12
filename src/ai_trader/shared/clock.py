from __future__ import annotations

from datetime import datetime, timezone
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
