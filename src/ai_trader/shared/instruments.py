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


# Prefijo de los mercados de prediccion en el universo ("PM::<slug>"). Vive aqui, junto
# a AssetClass, porque es parte de la convencion de nombres de simbolo, no de un
# proveedor concreto: los tres modulos que enrutan por simbolo lo importan de aqui.
PREDICTION_PREFIX = "PM::"


def detect_asset_class(symbol: str) -> AssetClass:
    """
    Clase de activo DEDUCIDA DEL NOMBRE del simbolo, sin tocar red ni proveedores.

    Es la regla canonica del proyecto y la unica fuente de verdad de la convencion:
      "PM::<slug>" -> prediccion, "BTC/USDT" (con barra) -> cripto, "AAPL" -> accion.

    Los proveedores pueden AFINARLA con lo que saben (MarketDataService pregunta ademas
    a CCXT si sabe servir un simbolo sin barra), pero nadie debe reimplementarla.
    """
    normalized = symbol.strip().upper()
    if normalized.startswith(PREDICTION_PREFIX):
        return AssetClass.PREDICTION
    if "/" in normalized:
        return AssetClass.CRYPTO
    return AssetClass.STOCK


# Monedas de cotizacion con las que el sistema sabe partir un par PEGADO ("BTCUSDT" ->
# BTC + USDT). Vive aqui, junto a `detect_asset_class`, por el mismo motivo que
# PREDICTION_PREFIX: es parte de la convencion de nombres, no de un proveedor. La tenia
# copiada `providers/ccxt_crypto.py::normalize_symbol` y la necesita la resolucion de
# entidad de `shared/entities.py`, que no puede depender de un proveedor concreto.
#
# El ORDEN es significativo y va de mas larga a mas corta dentro de cada familia: se
# prueba en secuencia y gana la primera que encaja, asi que "USDT" tiene que ir antes que
# "USD" para que "XUSDT" no se parta como "XUSD" + "T".
QUOTE_CURRENCIES: tuple[str, ...] = ("USDT", "USD", "USDC", "BTC", "ETH", "EUR")


def split_pair(symbol: str) -> tuple[str, str] | None:
    """
    (base, cotizacion) de un par cripto, o None si el nombre no lo es.

    Acepta las tres formas que llegan de los proveedores: "BTC/USDT", "BTC-USDT" y
    "BTCUSDT". Con separador explicito NO se exige que la cotizacion este en
    `QUOTE_CURRENCIES` —un par nuevo contra una moneda que no conocemos sigue siendo un
    par, y el separador ya nos dice donde corta—; sin separador si, porque es la unica
    informacion con la que adivinar donde termina la base.
    """
    cleaned = (symbol or "").strip().upper().replace("-", "/")
    if not cleaned or cleaned.startswith(PREDICTION_PREFIX):
        return None

    if "/" in cleaned:
        base, _, quote = cleaned.partition("/")
        return (base, quote) if base and quote else None

    for quote in QUOTE_CURRENCIES:
        if cleaned.endswith(quote) and len(cleaned) > len(quote):
            return cleaned[: -len(quote)], quote

    return None


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