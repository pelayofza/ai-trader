from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(slots=True)
class PolymarketClobConfig:
    base_url: str = "https://clob.polymarket.com"
    timeout_seconds: float = 10.0


class PolymarketClobProvider:
    def __init__(self, config: PolymarketClobConfig | None = None) -> None:
        self.config = config or PolymarketClobConfig()

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        normalized_token_id = token_id.strip()
        if not normalized_token_id:
            raise ValueError("token_id cannot be empty")

        return self._get_json("/book", params={"token_id": normalized_token_id})

    def get_midpoint(self, token_id: str) -> float | None:
        payload = self._get_json("/midpoint", params={"token_id": token_id})
        if isinstance(payload, dict):
            value = payload.get("mid")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        query = urlencode(params or {}, doseq=True)
        url = f"{self.config.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ai-trader/0.1.0",
            },
            method="GET",
        )

        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)