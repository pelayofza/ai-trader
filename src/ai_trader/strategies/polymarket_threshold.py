from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from ai_trader.shared.clock import utc_now
from ai_trader.shared.instruments import AssetClass, PredictionMarket, Venue
from ai_trader.shared.schemas import Side, Signal


@dataclass(slots=True)
class PolymarketThresholdConfig:
    slug: str
    outcome: str = "yes"
    buy_below_price: float = 0.45
    confidence: float = 0.70
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
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class PolymarketThresholdStrategy:
    """
    Compra un outcome cuando su midpoint cae por debajo de un umbral.

    Emite una Signal, no una OrderRequest. Antes construia la orden directamente y
    el orquestador la ejecutaba sin pasar por el motor de riesgo, de modo que estas
    operaciones no tenian ni limite de tamano, ni de exposicion, ni stop loss.
    """

    def __init__(self, config: PolymarketThresholdConfig) -> None:
        self.config = config
        self.strategy_id = config.strategy_id

    def supports_symbol(self, symbol: str) -> bool:
        return symbol.strip().lower() == f"pm::{self.config.slug}"

    def generate_signal(self, symbol: str, context: PredictionMarket) -> Signal | None:
        market = context

        token = market.yes_token if self.config.outcome == "yes" else market.no_token
        if token is None or token.price is None:
            return None

        midpoint = token.price
        if midpoint >= self.config.buy_below_price:
            return None

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            timeframe="1d",
            timestamp=utc_now(),
            side=Side.BUY,
            confidence=self.config.confidence,
            entry_price=midpoint,
            # Un outcome no puede valer menos de 0 ni mas de 1. El stop y el take
            # profit definitivos los fija el motor de riesgo.
            stop_loss=None,
            take_profit=None,
            reason=f"Midpoint {midpoint:.3f} below threshold {self.config.buy_below_price:.3f}",
            features={"midpoint": midpoint, "question": market.question},
            signal_id=f"{self.strategy_id}:{market.slug}:{uuid4().hex[:12]}",
            venue=Venue.POLYMARKET,
            asset_class=AssetClass.PREDICTION,
            instrument_id=token.token_id,
            outcome=token.outcome,
        )
