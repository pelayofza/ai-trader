from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JsonHttpConfig:
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_seconds: float = 0.5
    user_agent: str = "ai-trader/0.1.0"


class JsonHttpClient:
    """Cliente GET/JSON con reintentos y backoff. Compartido por los proveedores REST."""

    def __init__(self, base_url: str, config: JsonHttpConfig | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.config = config or JsonHttpConfig()

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        query = urlencode(params or {}, doseq=True)
        if query:
            url = f"{url}?{query}"

        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": self.config.user_agent},
            method="GET",
        )

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                # 4xx que no sea rate limit es un error del cliente: reintentar no ayuda.
                if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                    raise

                if attempt < self.config.max_retries - 1:
                    delay = self.config.backoff_seconds * (2**attempt)
                    logger.warning(
                        "HTTP GET failed (attempt %s/%s) | url=%s | error=%s | retrying in %.1fs",
                        attempt + 1,
                        self.config.max_retries,
                        url,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        assert last_error is not None
        raise last_error
