from __future__ import annotations

from dataclasses import dataclass

from ai_trader.shared.instruments import AssetClass

# --- factores macro ---------------------------------------------------------
#
# El modelo de correlacion es un modelo de factores: cada activo carga (beta) sobre
# un conjunto pequeno de factores macro comunes. Las correlaciones entre activos
# emergen de compartir exposicion a los mismos factores, no de una matriz explicita.
# Ventaja: la matriz de covarianzas resultante es SIEMPRE definida positiva, y la IA
# solo tiene que razonar sobre "que hacen los factores" en cada escenario, no sobre
# NxN correlaciones sueltas.

EQUITY = "EQUITY"        # apetito de riesgo bursatil (risk-on / risk-off)
RATES = "RATES"          # tipos de interes / duracion (sube -> bonos y growth bajan)
USD = "USD"              # fortaleza del dolar
COMMODITY = "COMMODITY"  # materias primas / activos reales (petroleo, oro)
CRYPTO = "CRYPTO"        # beta especifica del ecosistema cripto

DEFAULT_FACTORS: tuple[str, ...] = (EQUITY, RATES, USD, COMMODITY, CRYPTO)

# Descripcion legible por la IA (se inyecta en el prompt del disenador de escenarios).
FACTOR_DESCRIPTIONS: dict[str, str] = {
    EQUITY: "Broad equity risk appetite. Positive on risk-on days, negative on risk-off.",
    RATES: "Interest rates / bond yields. Positive = yields rising (hawkish, bonds fall).",
    USD: "US dollar strength. Positive = stronger dollar (headwind for crypto/commodities).",
    COMMODITY: "Real assets / commodities (oil, gold). Positive = commodities rally.",
    CRYPTO: "Crypto-specific beta (adoption, on-chain flows, regulation shocks).",
}


@dataclass(slots=True, frozen=True)
class SyntheticAsset:
    """
    Un activo del universo sintetico y su estructura ESTABLE (no cambia entre
    escenarios): cargas sobre cada factor, volatilidad idiosincratica y precio inicial.

    La respuesta del activo a un escenario sale de: loadings x movimientos-de-factor
    (comun a todos) + ruido idiosincratico + tilts que la IA anade por escenario.
    """

    symbol: str
    asset_class: AssetClass
    start_price: float
    # beta a cada factor, en unidades de retorno diario del activo por retorno del factor.
    loadings: dict[str, float]
    # vol diaria idiosincratica (log-retorno): lo que no explican los factores.
    idio_vol: float

    def loading(self, factor: str) -> float:
        return float(self.loadings.get(factor, 0.0))


@dataclass(slots=True, frozen=True)
class SyntheticUniverse:
    """Conjunto de activos + factores sobre los que se construye todo el generador."""

    assets: tuple[SyntheticAsset, ...]
    factors: tuple[str, ...] = DEFAULT_FACTORS

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.assets]

    def asset(self, symbol: str) -> SyntheticAsset | None:
        target = symbol.strip().upper()
        for a in self.assets:
            if a.symbol.strip().upper() == target:
                return a
        return None


# --- universo por defecto ---------------------------------------------------
#
# Mezcla deliberada de clases de activo para que las correlaciones cruzadas tengan
# sentido y no se malinterpreten (requisito del diseno: cada escenario cubre TODOS
# los tipos). Las cargas son cualitativas pero plausibles; la IA no las toca, solo
# mueve los factores y aplica tilts por escenario.

DEFAULT_UNIVERSE = SyntheticUniverse(
    factors=DEFAULT_FACTORS,
    assets=(
        # --- Cripto ---
        SyntheticAsset(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            start_price=40_000.0,
            loadings={EQUITY: 0.30, RATES: -0.10, USD: -0.25, COMMODITY: 0.10, CRYPTO: 1.00},
            idio_vol=0.020,
        ),
        SyntheticAsset(
            symbol="ETH/USDT",
            asset_class=AssetClass.CRYPTO,
            start_price=2_500.0,
            loadings={EQUITY: 0.35, RATES: -0.10, USD: -0.25, COMMODITY: 0.10, CRYPTO: 1.15},
            idio_vol=0.026,
        ),
        SyntheticAsset(
            symbol="SOL/USDT",
            asset_class=AssetClass.CRYPTO,
            start_price=100.0,
            loadings={EQUITY: 0.40, RATES: -0.10, USD: -0.25, COMMODITY: 0.10, CRYPTO: 1.35},
            idio_vol=0.038,
        ),
        # --- Renta variable ---
        SyntheticAsset(
            symbol="SPY",
            asset_class=AssetClass.STOCK,
            start_price=450.0,
            loadings={EQUITY: 1.00, RATES: -0.20, USD: -0.05, COMMODITY: 0.05, CRYPTO: 0.05},
            idio_vol=0.004,
        ),
        SyntheticAsset(
            symbol="QQQ",
            asset_class=AssetClass.STOCK,
            start_price=380.0,
            loadings={EQUITY: 1.10, RATES: -0.40, USD: -0.05, COMMODITY: 0.00, CRYPTO: 0.10},
            idio_vol=0.006,
        ),
        SyntheticAsset(
            symbol="NVDA",
            asset_class=AssetClass.STOCK,
            start_price=500.0,
            loadings={EQUITY: 1.20, RATES: -0.50, USD: -0.05, COMMODITY: 0.00, CRYPTO: 0.15},
            idio_vol=0.016,
        ),
        SyntheticAsset(
            symbol="XOM",
            asset_class=AssetClass.STOCK,
            start_price=110.0,
            loadings={EQUITY: 0.60, RATES: 0.10, USD: 0.00, COMMODITY: 0.90, CRYPTO: 0.00},
            idio_vol=0.011,
        ),
        SyntheticAsset(
            symbol="JPM",
            asset_class=AssetClass.STOCK,
            start_price=150.0,
            loadings={EQUITY: 0.90, RATES: 0.40, USD: 0.00, COMMODITY: 0.05, CRYPTO: 0.00},
            idio_vol=0.010,
        ),
        # --- Activos reales / refugio ---
        SyntheticAsset(
            symbol="GLD",
            asset_class=AssetClass.STOCK,
            start_price=180.0,
            loadings={EQUITY: -0.10, RATES: -0.30, USD: -0.50, COMMODITY: 0.60, CRYPTO: 0.00},
            idio_vol=0.006,
        ),
        SyntheticAsset(
            symbol="TLT",
            asset_class=AssetClass.STOCK,
            start_price=95.0,
            loadings={EQUITY: -0.10, RATES: -1.00, USD: 0.10, COMMODITY: -0.05, CRYPTO: 0.00},
            idio_vol=0.006,
        ),
    ),
)


def universe_summary(universe: SyntheticUniverse = DEFAULT_UNIVERSE) -> list[dict]:
    """Vista compacta del universo para inyectar en el prompt de la IA o inspeccionar."""
    return [
        {
            "symbol": a.symbol,
            "asset_class": a.asset_class.value,
            "loadings": {f: a.loading(f) for f in universe.factors},
            "idio_vol": a.idio_vol,
        }
        for a in universe.assets
    ]
