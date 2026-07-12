from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_TELEGRAM_MESSAGE_CHARS = 4000


class TelegramNotifier:
    """
    Canal de Telegram para el nucleo.

    El orquestador corre en un hilo aparte (no en el event loop), asi que los envios
    se programan sobre el loop de la aplicacion de forma thread-safe. Nunca lanza:
    un fallo del canal de alertas no puede tumbar un ciclo de trading.
    """

    def __init__(self, bot: Any, chat_ids: list[int], loop: asyncio.AbstractEventLoop) -> None:
        self.bot = bot
        self.chat_ids = chat_ids
        self.loop = loop

    def info(self, message: str) -> None:
        logger.info(message)
        self._send(message)

    def warning(self, message: str) -> None:
        logger.warning(message)
        self._send(f"WARNING\n{message}")

    def error(self, message: str) -> None:
        logger.error(message)
        self._send(f"ERROR\n{message}")

    def _send(self, message: str) -> None:
        if not self.chat_ids:
            return

        text = message[:MAX_TELEGRAM_MESSAGE_CHARS]

        for chat_id in self.chat_ids:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.bot.send_message(chat_id=chat_id, text=text),
                    self.loop,
                )
            except Exception:
                logger.exception("Could not dispatch Telegram message to chat_id=%s", chat_id)
