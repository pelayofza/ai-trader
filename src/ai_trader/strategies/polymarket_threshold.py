from __future__ import annotations

from dataclasses import dataclass

from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import OrderRequest, OrderType, Side


@dataclass(slots=True)
class PolymarketThresholdConfig:
    slug: str
    outcome: str = "yes"
    buy_below_price: float = 0.45
    size: float = 10.0
    strategy_id: str = "polymarket_threshold"

    def __post_init__(self) -> None:
        self.slug = self.slug.strip().lower()
        self.outcome = self.outcome.strip().lower()

        if not self.slug:
            raise ValueError("slug cannot be empty")
        if self.outcome not in {"yes", "no"}:
            raise ValueError("outcome must be 'yes' or 'no'")
        if not 0.0 < self.buy_below_price < 1.0:
            raise ValueError("buy_below_price must be between 0 and 1")
        if self.size <= 0:
            raise ValueError("size must be greater than 0")


class PolymarketThresholdStrategy:
    def __init__(self, config: PolymarketThresholdConfig) -> None:
        self.config = config
        self.strategy_id = config.strategy_id

    def supports_symbol(self, symbol: str) -> bool:
        return symbol.strip().lower() == f"pm::{self.config.slug}"

    def build_order_request(self, market, midpoint: float) -> OrderRequest | None:
        if midpoint >= self.config.buy_below_price:
            return None

        token = market.yes_token if self.config.outcome == "yes" else market.no_token
        if token is None:
            return None

        return OrderRequest(
            symbol=f"PM::{market.slug}",
            side=Side.BUY,
            size=self.config.size,
            order_type=OrderType.MARKET,
            strategy_id=self.strategy_id,
            venue=Venue.POLYMARKET,
            asset_class=AssetClass.PREDICTION,
            instrument_id=token.token_id,
            outcome=token.outcome,
            metadata={
                "question": market.question,
                "slug": market.slug,
                "rule": f"buy_when_midpoint_below_{self.config.buy_below_price}",
                "midpoint_at_signal": str(midpoint),
            },
        )