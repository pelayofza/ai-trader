from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ai_trader.data.providers.http import JsonHttpClient, JsonHttpConfig
from ai_trader.shared.instruments import OutcomeToken, PredictionMarket

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PolymarketGammaConfig:
    base_url: str = "https://gamma-api.polymarket.com"
    timeout_seconds: float = 10.0
    default_limit: int = 50
    max_search_pages: int = 10


class PolymarketGammaProvider:
    def __init__(self, config: PolymarketGammaConfig | None = None) -> None:
        self.config = config or PolymarketGammaConfig()
        self._http = JsonHttpClient(
            self.config.base_url,
            JsonHttpConfig(timeout_seconds=self.config.timeout_seconds),
        )

    def get_market_by_slug(self, slug: str) -> PredictionMarket | None:
        normalized_slug = slug.strip().lower()
        if not normalized_slug:
            return None

        payload = self._http.get_json("/markets", params={"slug": normalized_slug})

        if not isinstance(payload, list) or not payload:
            logger.info("No Polymarket market found for slug=%s", normalized_slug)
            return None

        return self._parse_market(payload[0])

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

        for _ in range(self.config.max_search_pages):
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
                "archived": "false",
            }
            if active_only:
                params["active"] = "true"
                params["closed"] = "false"

            payload = self._http.get_json("/markets", params=params)
            if not isinstance(payload, list) or not payload:
                break

            for item in payload:
                if not isinstance(item, dict):
                    continue

                market = self._try_parse_market(item)
                if market is None or market.slug in seen_slugs:
                    continue

                haystacks = (
                    market.question.lower(),
                    market.slug.lower(),
                    (market.market_slug or "").lower(),
                    " ".join(market.tags).lower(),
                )
                if not any(normalized_query in value for value in haystacks):
                    continue

                seen_slugs.add(market.slug)
                results.append(market)

                if len(results) >= target_results:
                    return results

            if len(payload) < page_size:
                break

            offset += page_size

        return results

    def _try_parse_market(self, payload: dict[str, Any]) -> PredictionMarket | None:
        try:
            return self._parse_market(payload)
        except (ValueError, KeyError, TypeError) as exc:
            logger.debug("Skipping unparseable market | slug=%s | error=%s", payload.get("slug"), exc)
            return None

    def _parse_market(self, payload: dict[str, Any]) -> PredictionMarket:
        outcomes = self._safe_json_list(payload.get("outcomes"))
        prices = self._safe_json_list(payload.get("outcomePrices"))
        token_ids = self._safe_json_list(payload.get("clobTokenIds"))

        parsed_outcomes: list[OutcomeToken] = []
        for idx, outcome in enumerate(outcomes):
            raw_token_id = token_ids[idx] if idx < len(token_ids) else None
            token_id = str(raw_token_id).strip() if raw_token_id is not None else ""
            if not token_id:
                continue

            price: float | None = None
            if idx < len(prices):
                try:
                    price = float(prices[idx])
                except (TypeError, ValueError):
                    price = None

            parsed_outcomes.append(
                OutcomeToken(outcome=str(outcome), token_id=token_id, price=price)
            )

        if not parsed_outcomes:
            raise ValueError("market has no parseable outcome tokens")

        raw_tags = payload.get("tags")
        tags = [str(item) for item in raw_tags] if isinstance(raw_tags, list) else []

        market_id = payload.get("id") or payload.get("conditionId") or payload.get("slug")
        question = payload.get("question") or payload.get("title") or payload.get("slug")
        slug = payload.get("slug") or payload.get("market_slug") or str(market_id)

        return PredictionMarket(
            market_id=str(market_id),
            question=str(question or "unknown-market"),
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
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return parsed
        return []

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
