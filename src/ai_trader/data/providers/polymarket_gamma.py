from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ai_trader.shared.instruments import OutcomeToken, PredictionMarket


@dataclass(slots=True)
class PolymarketGammaConfig:
    base_url: str = "https://gamma-api.polymarket.com"
    timeout_seconds: float = 10.0
    default_limit: int = 50


class PolymarketGammaProvider:
    def __init__(self, config: PolymarketGammaConfig | None = None) -> None:
        self.config = config or PolymarketGammaConfig()

    def list_markets(
        self,
        *,
        limit: int | None = None,
        active: bool = True,
        closed: bool = False,
        archived: bool = False,
        tag: str | None = None,
    ) -> list[PredictionMarket]:
        params: dict[str, Any] = {
            "limit": limit or self.config.default_limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "archived": str(archived).lower(),
        }
        if tag:
            params["tag"] = tag

        payload = self._get_json("/markets", params=params)
        if not isinstance(payload, list):
            return []

        return [self._parse_market(item) for item in payload if isinstance(item, dict)]

    def search_markets(
        self,
        query: str,
        *,
        limit: int | None = None,
        active_only: bool = True,
    ) -> list[PredictionMarket]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        payload = self._get_json(
            "/search",
            params={
                "q": normalized_query,
                "limit": limit or self.config.default_limit,
            },
        )

        if not isinstance(payload, list):
            return []

        results: list[PredictionMarket] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            try:
                market = self._parse_market(item)
            except Exception:
                continue

            if active_only and (not market.active or market.closed or market.archived):
                continue

            results.append(market)

        return results

    def get_market_by_slug(self, slug: str) -> PredictionMarket | None:
        normalized_slug = slug.strip().lower()
        if not normalized_slug:
            return None

        try:
            payload = self._get_json(f"/markets/slug/{normalized_slug}")
            if isinstance(payload, dict):
                return self._parse_market(payload)
        except Exception:
            pass

        markets = self.list_markets(limit=500, active=True, closed=False, archived=False)
        for market in markets:
            if market.slug.lower() == normalized_slug:
                return market
            if market.market_slug and market.market_slug.lower() == normalized_slug:
                return market
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

    def _parse_market(self, payload: dict[str, Any]) -> PredictionMarket:
        outcomes_raw = payload.get("outcomes")
        prices_raw = payload.get("outcomePrices")
        token_ids_raw = payload.get("clobTokenIds")

        outcomes = self._safe_json_list(outcomes_raw)
        prices = self._safe_json_list(prices_raw)
        token_ids = self._safe_json_list(token_ids_raw)

        parsed_outcomes: list[OutcomeToken] = []
        for idx, outcome in enumerate(outcomes):
            token_id = str(token_ids[idx]).strip() if idx < len(token_ids) and token_ids[idx] is not None else ""
            if not token_id:
                continue

            price: float | None = None
            if idx < len(prices):
                try:
                    price = float(prices[idx])
                except (TypeError, ValueError):
                    price = None

            parsed_outcomes.append(
                OutcomeToken(
                    outcome=str(outcome),
                    token_id=token_id,
                    price=price,
                )
            )

        tags = []
        raw_tags = payload.get("tags")
        if isinstance(raw_tags, list):
            tags = [str(item) for item in raw_tags]

        market_id = payload.get("id")
        if market_id is None:
            market_id = payload.get("conditionId") or payload.get("questionID") or payload.get("slug")

        question = payload.get("question") or payload.get("title") or payload.get("slug") or "unknown-market"
        slug = payload.get("slug") or payload.get("market_slug") or str(market_id)

        if not parsed_outcomes:
            raise ValueError("market has no parseable outcome tokens")

        return PredictionMarket(
            market_id=str(market_id),
            question=str(question),
            slug=str(slug),
            active=bool(payload.get("active", False)),
            closed=bool(payload.get("closed", False)),
            archived=bool(payload.get("archived", False)),
            enable_order_book=bool(payload.get("enableOrderBook", False)),
            outcomes=parsed_outcomes,
            condition_id=self._as_optional_str(payload.get("conditionId")),
            market_slug=self._as_optional_str(payload.get("market_slug")),
            end_date_iso=self._as_optional_str(payload.get("endDate")),
            tags=tags,
            raw=payload,
        )

    @staticmethod
    def _safe_json_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None