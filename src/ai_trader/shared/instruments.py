from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Venue(str, Enum):
    ALPACA = "alpaca"
    CCXT = "ccxt"
    POLYMARKET = "polymarket"


class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    PREDICTION = "prediction"


@dataclass(slots=True)
class OutcomeToken:
    outcome: str
    token_id: str
    price: float | None = None

    def __post_init__(self) -> None:
        if not self.outcome:
            raise ValueError("outcome cannot be empty")
        if not self.token_id:
            raise ValueError("token_id cannot be empty")
        if self.price is not None and not 0.0 <= self.price <= 1.0:
            raise ValueError("price must be between 0 and 1")


@dataclass(slots=True)
class PredictionMarket:
    market_id: str
    question: str
    slug: str
    active: bool
    closed: bool
    archived: bool
    enable_order_book: bool
    outcomes: list[OutcomeToken]
    condition_id: str | None = None
    market_slug: str | None = None
    end_date_iso: str | None = None
    tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError("market_id cannot be empty")
        if not self.question:
            raise ValueError("question cannot be empty")
        if not self.slug:
            raise ValueError("slug cannot be empty")
        if not self.outcomes:
            raise ValueError("outcomes cannot be empty")

    @property
    def yes_token(self) -> OutcomeToken | None:
        for item in self.outcomes:
            if item.outcome.strip().lower() == "yes":
                return item
        return None

    @property
    def no_token(self) -> OutcomeToken | None:
        for item in self.outcomes:
            if item.outcome.strip().lower() == "no":
                return item
        return None