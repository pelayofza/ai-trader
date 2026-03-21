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
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        page_size = min(max(limit or self.config.default_limit, 1), 100)
        target_results = limit or self.config.default_limit

        results: list[PredictionMarket] = []
        seen_slugs: set[str] = set()

        offset = 0
        max_pages = 10

        for _ in range(max_pages):
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
                "archived": "false",
            }

            if active_only:
                params["active"] = "true"
                params["closed"] = "false"

            payload = self._get_json("/markets", params=params)
            if not isinstance(payload, list) or not payload:
                break

            for item in payload:
                if not isinstance(item, dict):
                    continue

                try:
                    market = self._parse_market(item)
                except Exception:
                    continue

                haystacks = [
                    market.question.lower(),
                    market.slug.lower(),
                    (market.market_slug or "").lower(),
                    " ".join(market.tags).lower(),
                ]

                if not any(normalized_query in value for value in haystacks):
                    continue

                if market.slug in seen_slugs:
                    continue

                seen_slugs.add(market.slug)
                results.append(market)

                if len(results) >= target_results:
                    return results

            if len(payload) < page_size:
                break

            offset += page_size

        return results

    def get_market_by_slug(self, slug: str) -> PredictionMarket | None:
        normalized_slug = slug.strip().lower()
        if not normalized_slug:
            return None

        page_size = 100
        offset = 0
        max_pages = 20

        for _ in range(max_pages):
            payload = self._get_json(
                "/markets",
                params={
                    "limit": page_size,
                    "offset": offset,
                    "archived": "false",
                },
            )

            if not isinstance(payload, list) or not payload:
                break

            for item in payload:
                if not isinstance(item, dict):
                    continue

                try:
                    market = self._parse_market(item)
                except Exception:
                    continue

                if market.slug.lower() == normalized_slug:
                    return market
                if market.market_slug and market.market_slug.lower() == normalized_slug:
                    return market

            if len(payload) < page_size:
                break

            offset += page_size

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