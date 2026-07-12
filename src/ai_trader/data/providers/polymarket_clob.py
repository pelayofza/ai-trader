from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ai_trader.data.providers.http import JsonHttpClient, JsonHttpConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PolymarketClobConfig:
    base_url: str = "https://clob.polymarket.com"
    timeout_seconds: float = 10.0


class PolymarketClobProvider:
    def __init__(self, config: PolymarketClobConfig | None = None) -> None:
        self.config = config or PolymarketClobConfig()
        self._http = JsonHttpClient(
            self.config.base_url,
            JsonHttpConfig(timeout_seconds=self.config.timeout_seconds),
        )

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        normalized_token_id = token_id.strip()
        if not normalized_token_id:
            raise ValueError("token_id cannot be empty")

        return self._http.get_json("/book", params={"token_id": normalized_token_id})

    def get_midpoint(self, token_id: str) -> float | None:
        normalized_token_id = token_id.strip()
        if not normalized_token_id:
            raise ValueError("token_id cannot be empty")

        try:
            payload = self._http.get_json("/midpoint", params={"token_id": normalized_token_id})
        except Exception:
            logger.exception("Could not fetch Polymarket midpoint | token_id=%s", normalized_token_id)
            return None

        if not isinstance(payload, dict):
            return None

        try:
            return float(payload.get("mid"))
        except (TypeError, ValueError):
            return None
