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

def _crypto(symbol, price, equity, usd, crypto, idio):
    return SyntheticAsset(
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        start_price=price,
        loadings={EQUITY: equity, RATES: -0.10, USD: usd, COMMODITY: 0.10, CRYPTO: crypto},
        idio_vol=idio,
    )


def _stock(symbol, price, equity, rates, usd, commodity, crypto, idio):
    return SyntheticAsset(
        symbol=symbol,
        asset_class=AssetClass.STOCK,
        start_price=price,
        loadings={EQUITY: equity, RATES: rates, USD: usd, COMMODITY: commodity, CRYPTO: crypto},
        idio_vol=idio,
    )


DEFAULT_UNIVERSE = SyntheticUniverse(
    factors=DEFAULT_FACTORS,
    assets=(
        # --- Cripto (12): beta CRYPTO alta, sensibles a EQUITY y al dolar (USD<0) ---
        _crypto("BTC/USDT", 40_000.0, equity=0.30, usd=-0.25, crypto=1.00, idio=0.020),
        _crypto("ETH/USDT", 2_500.0, equity=0.35, usd=-0.25, crypto=1.15, idio=0.026),
        _crypto("SOL/USDT", 100.0, equity=0.40, usd=-0.25, crypto=1.35, idio=0.038),
        _crypto("BNB/USDT", 400.0, equity=0.32, usd=-0.22, crypto=1.05, idio=0.030),
        _crypto("XRP/USDT", 0.60, equity=0.25, usd=-0.20, crypto=0.95, idio=0.036),
        _crypto("ADA/USDT", 0.45, equity=0.32, usd=-0.22, crypto=1.20, idio=0.034),
        _crypto("AVAX/USDT", 35.0, equity=0.40, usd=-0.24, crypto=1.40, idio=0.042),
        _crypto("LINK/USDT", 15.0, equity=0.38, usd=-0.24, crypto=1.30, idio=0.040),
        _crypto("DOGE/USDT", 0.12, equity=0.30, usd=-0.18, crypto=1.25, idio=0.055),
        _crypto("DOT/USDT", 6.50, equity=0.36, usd=-0.22, crypto=1.25, idio=0.038),
        _crypto("MATIC/USDT", 0.80, equity=0.38, usd=-0.22, crypto=1.30, idio=0.044),
        _crypto("LTC/USDT", 90.0, equity=0.28, usd=-0.20, crypto=1.00, idio=0.032),
        # --- Renta variable (20): indices amplios + nombres por sector ---
        # Indices
        _stock("SPY", 450.0, equity=1.00, rates=-0.20, usd=-0.05, commodity=0.05, crypto=0.05, idio=0.004),
        _stock("QQQ", 380.0, equity=1.10, rates=-0.40, usd=-0.05, commodity=0.00, crypto=0.10, idio=0.006),
        # Tech / crecimiento
        _stock("AAPL", 180.0, equity=1.05, rates=-0.30, usd=-0.05, commodity=0.00, crypto=0.08, idio=0.012),
        _stock("MSFT", 400.0, equity=1.05, rates=-0.30, usd=-0.05, commodity=0.00, crypto=0.06, idio=0.011),
        _stock("GOOGL", 140.0, equity=1.10, rates=-0.30, usd=-0.05, commodity=0.00, crypto=0.06, idio=0.014),
        _stock("AMZN", 150.0, equity=1.15, rates=-0.35, usd=-0.05, commodity=0.00, crypto=0.07, idio=0.016),
        _stock("META", 350.0, equity=1.15, rates=-0.30, usd=-0.05, commodity=0.00, crypto=0.08, idio=0.018),
        # Semiconductores (beta alta, muy sensibles a tipos y con correlacion cripto)
        _stock("NVDA", 500.0, equity=1.20, rates=-0.50, usd=-0.05, commodity=0.00, crypto=0.15, idio=0.022),
        _stock("AMD", 140.0, equity=1.25, rates=-0.50, usd=-0.05, commodity=0.00, crypto=0.15, idio=0.026),
        _stock("TSLA", 240.0, equity=1.30, rates=-0.45, usd=-0.05, commodity=0.05, crypto=0.20, idio=0.030),
        # Financieras (cargan RATES en positivo: se benefician de tipos altos)
        _stock("JPM", 150.0, equity=0.90, rates=0.40, usd=0.00, commodity=0.05, crypto=0.00, idio=0.012),
        _stock("BAC", 35.0, equity=0.95, rates=0.45, usd=0.00, commodity=0.05, crypto=0.00, idio=0.014),
        _stock("GS", 380.0, equity=1.00, rates=0.35, usd=0.00, commodity=0.05, crypto=0.02, idio=0.015),
        # Energia (cargan COMMODITY fuerte)
        _stock("XOM", 110.0, equity=0.60, rates=0.10, usd=0.00, commodity=0.90, crypto=0.00, idio=0.012),
        _stock("CVX", 155.0, equity=0.60, rates=0.10, usd=0.00, commodity=0.85, crypto=0.00, idio=0.012),
        # Defensivas / consumo / salud (beta baja)
        _stock("KO", 60.0, equity=0.45, rates=-0.05, usd=-0.05, commodity=0.05, crypto=0.00, idio=0.008),
        _stock("PG", 155.0, equity=0.40, rates=-0.05, usd=-0.05, commodity=0.00, crypto=0.00, idio=0.008),
        _stock("JNJ", 155.0, equity=0.45, rates=-0.05, usd=-0.05, commodity=0.00, crypto=0.00, idio=0.008),
        _stock("UNH", 520.0, equity=0.55, rates=-0.05, usd=-0.05, commodity=0.00, crypto=0.00, idio=0.012),
        # Industrial (algo de exposicion a materias primas)
        _stock("CAT", 280.0, equity=0.95, rates=0.05, usd=-0.05, commodity=0.30, crypto=0.00, idio=0.014),
        # --- Macro / refugio (3): oro, bonos largos, dolar ---
        _stock("GLD", 180.0, equity=-0.10, rates=-0.30, usd=-0.50, commodity=0.60, crypto=0.00, idio=0.006),
        _stock("TLT", 95.0, equity=-0.10, rates=-1.00, usd=0.10, commodity=-0.05, crypto=0.00, idio=0.006),
        _stock("UUP", 28.0, equity=-0.15, rates=0.20, usd=1.00, commodity=-0.10, crypto=-0.05, idio=0.004),
    ),
)


def universe_summary(universe: SyntheticUniverse = DEFAULT_UNIVERSE) -> list[dict]:
    """
    Vista compacta y COMPLETA del universo. Se guarda en el manifiesto para que una
    libreria sea autocontenida: incluye start_price, de modo que se pueda reconstruir
    el universo exacto (y regenerar paths identicos) aunque el codigo cambie despues.
    """
    return [
        {
            "symbol": a.symbol,
            "asset_class": a.asset_class.value,
            "start_price": a.start_price,
            "loadings": {f: a.loading(f) for f in universe.factors},
            "idio_vol": a.idio_vol,
        }
        for a in universe.assets
    ]


def universe_from_summary(
    summary: list[dict], factors: tuple[str, ...] = DEFAULT_FACTORS
) -> SyntheticUniverse:
    """
    Reconstruye un SyntheticUniverse desde el resumen guardado en un manifiesto.
    Requiere start_price (manifiestos nuevos lo incluyen). Lanza KeyError si falta,
    para que el llamante pueda recurrir al universo del codigo en librerias antiguas.
    """
    assets = tuple(
        SyntheticAsset(
            symbol=item["symbol"],
            asset_class=AssetClass(item["asset_class"]),
            start_price=float(item["start_price"]),
            loadings={f: float(item.get("loadings", {}).get(f, 0.0)) for f in factors},
            idio_vol=float(item["idio_vol"]),
        )
        for item in summary
    )
    return SyntheticUniverse(assets=assets, factors=tuple(factors))
