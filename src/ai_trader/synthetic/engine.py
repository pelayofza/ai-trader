from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ai_trader.shared.bars import CLOSE, HIGH, LOW, OPEN, VOLUME
from ai_trader.synthetic.scenarios import ScenarioSpec
from ai_trader.synthetic.universe import SyntheticUniverse

# Anclaje temporal por defecto de las series sinteticas. Las fechas son solo
# etiquetas (los datos no son reales); lo que importa es la longitud y el orden.
DEFAULT_ANCHOR = datetime(2015, 1, 1, tzinfo=timezone.utc)

# Microestructura intradia: como se abren las velas y cuanto sobresalen las mechas.
# Son fracciones de la volatilidad diaria del activo, no numeros absolutos.
OVERNIGHT_GAP_FACTOR = 0.4  # dispersion del hueco de apertura vs cierre anterior
WICK_FACTOR = 0.5           # tamano tipico de las mechas por encima/debajo del cuerpo
BASE_VOLUME = 1_000_000.0


class PathEngine:
    """
    Motor numerico determinista: convierte un ScenarioSpec (la "fisica" que disena la
    IA) en velas OHLCV multi-activo mediante un modelo de factores.

    Para cada dia t y activo i:
        r_i(t) = tilt_i + sum_k beta_ik * f_k(t) + idio_i * eps_i(t)
        f_k(t) = drift_k(fase) + vol_k(fase) * w_k(t) + shocks

    Las correlaciones entre activos surgen de compartir factores f_k; por construccion
    la covarianza resultante es definida positiva (no hay matriz que pueda degenerar).

    Determinismo total: misma (spec, universe, seed) -> mismas velas, byte a byte.
    """

    def __init__(
        self,
        universe: SyntheticUniverse,
        *,
        anchor: datetime = DEFAULT_ANCHOR,
        overnight_gap_factor: float = OVERNIGHT_GAP_FACTOR,
        wick_factor: float = WICK_FACTOR,
    ) -> None:
        self.universe = universe
        self.anchor = anchor
        self.overnight_gap_factor = overnight_gap_factor
        self.wick_factor = wick_factor

    def generate(self, spec: ScenarioSpec, seed: int) -> dict[str, pd.DataFrame]:
        """Un unico path del escenario. dict[symbol -> DataFrame OHLCV]."""
        horizon = spec.horizon_days
        if horizon < 2:
            raise ValueError(f"Scenario '{spec.id}' has horizon {horizon}; need at least 2 days")

        factors = self.universe.factors
        rng = np.random.default_rng(seed)

        drift, vol = self._factor_timelines(spec, horizon, factors)  # (horizon, n_factors)
        self._apply_shocks(spec, drift, factors)

        # Retornos de los factores: media de fase + ruido gaussiano escalado por su vol.
        w = rng.standard_normal((horizon, len(factors)))
        factor_returns = drift + vol * w  # (horizon, n_factors)

        index = pd.DatetimeIndex(
            [self.anchor + timedelta(days=i) for i in range(horizon)], name="timestamp"
        )
        mean_factor_vol = vol.mean(axis=0)  # para dimensionar huecos y mechas por activo

        bars: dict[str, pd.DataFrame] = {}
        for asset in self.universe.assets:
            beta = np.array([asset.loading(f) for f in factors])
            tilt = float(spec.asset_tilts.get(asset.symbol, 0.0))

            eps = rng.standard_normal(horizon)
            asset_returns = tilt + factor_returns @ beta + asset.idio_vol * eps

            # Vol diaria efectiva del activo: sirve de escala para huecos y mechas.
            factor_var = float(np.sum((beta * mean_factor_vol) ** 2))
            daily_vol = float(np.sqrt(factor_var + asset.idio_vol**2))

            bars[asset.symbol] = self._build_ohlcv(
                asset_returns, asset.start_price, daily_vol, index, rng
            )

        return bars

    def _factor_timelines(
        self, spec: ScenarioSpec, horizon: int, factors: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expande las fases a arrays por dia (horizon, n_factors) de drift y vol."""
        n = len(factors)
        drift = np.zeros((horizon, n))
        vol = np.zeros((horizon, n))

        cursor = 0
        for phase in spec.phases:
            end = min(cursor + phase.length_days, horizon)
            for j, factor in enumerate(factors):
                drift[cursor:end, j] = phase.drift.get(factor, 0.0)
                vol[cursor:end, j] = phase.vol.get(factor, 0.0)
            cursor = end
            if cursor >= horizon:
                break

        # Piso de vol: evita factores exactamente deterministas (columnas degeneradas).
        vol = np.maximum(vol, 1e-6)
        return drift, vol

    def _apply_shocks(
        self, spec: ScenarioSpec, drift: np.ndarray, factors: tuple[str, ...]
    ) -> None:
        """Suma los shocks discretos como deriva puntual en su dia y factor."""
        index_of = {f: j for j, f in enumerate(factors)}
        horizon = drift.shape[0]
        for shock in spec.shocks:
            if shock.factor not in index_of:
                continue
            if not 0 <= shock.day < horizon:
                continue
            drift[shock.day, index_of[shock.factor]] += shock.magnitude

    def _build_ohlcv(
        self,
        returns: np.ndarray,
        start_price: float,
        daily_vol: float,
        index: pd.DatetimeIndex,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """De log-retornos diarios a velas OHLCV validas (high>=cuerpo>=low, >0)."""
        horizon = len(returns)

        close = start_price * np.exp(np.cumsum(returns))
        prev_close = np.empty(horizon)
        prev_close[0] = start_price
        prev_close[1:] = close[:-1]

        # Apertura: cierre anterior con un hueco nocturno pequeno.
        gap = rng.standard_normal(horizon) * (self.overnight_gap_factor * daily_vol)
        open_ = prev_close * np.exp(gap)

        body_high = np.maximum(open_, close)
        body_low = np.minimum(open_, close)

        # Mechas: extension no negativa (half-normal) por encima y por debajo del cuerpo.
        wick_scale = self.wick_factor * daily_vol
        up = np.abs(rng.standard_normal(horizon)) * wick_scale
        down = np.abs(rng.standard_normal(horizon)) * wick_scale
        high = body_high * (1.0 + up)
        low = body_low * (1.0 - down)

        # Volumen sintetico: base con ruido, mayor en dias de movimiento fuerte.
        vol_noise = np.exp(rng.standard_normal(horizon) * 0.3)
        volume = BASE_VOLUME * vol_noise * (1.0 + 5.0 * np.abs(returns))

        return pd.DataFrame(
            {
                OPEN: open_,
                HIGH: high,
                LOW: low,
                CLOSE: close,
                VOLUME: volume,
            },
            index=index,
        )


def generate_paths(
    spec: ScenarioSpec,
    universe: SyntheticUniverse,
    *,
    n_paths: int,
    seed_base: int,
    anchor: datetime = DEFAULT_ANCHOR,
) -> list[dict[str, pd.DataFrame]]:
    """
    Ensemble Monte Carlo: N realizaciones del mismo escenario, misma fisica y distinta
    semilla (seed_base + indice). Cada path es un dict[symbol -> DataFrame] listo para
    BacktestEngine.from_bars.
    """
    engine = PathEngine(universe, anchor=anchor)
    return [engine.generate(spec, seed=seed_base + i) for i in range(n_paths)]
