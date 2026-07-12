from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    """
    Canal de salida hacia el humano.

    El nucleo no sabe si detras hay un Telegram, un log o un correo. Esto es lo que
    permite que Telegram sea un canal y no el runtime: el orquestador se limita a
    emitir eventos.
    """

    def info(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...


class NullNotifier:
    """Notificador por defecto: solo registra. Util en backtest y en tests."""

    def info(self, message: str) -> None:
        logger.info(message)

    def warning(self, message: str) -> None:
        logger.warning(message)

    def error(self, message: str) -> None:
        logger.error(message)
