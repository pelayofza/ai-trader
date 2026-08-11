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
        self._apply_shocks(spec, drift, factors, seed)

        # Timelines por dia de la microestructura estadistica.
        idio_ar_t = self._phase_scalar_timeline(spec, horizon, "idio_ar")          # B5
        tail_dof_t = self._phase_scalar_timeline(spec, horizon, "tail_dof")        # B2
        persistence_t = self._phase_scalar_timeline(spec, horizon, "vol_persistence")  # B3
        news_t = self._phase_scalar_timeline(spec, horizon, "vol_news")                # B3
        jump_intensity_t = self._phase_scalar_timeline(spec, horizon, "jump_intensity")  # B4
        jump_scale_t = self._phase_scalar_timeline(spec, horizon, "jump_scale")         # B4
        beta_stress_t = self._phase_scalar_timeline(spec, horizon, "beta_stress")       # B6
        # Si ninguna fase pide colas, se usa la via gaussiana EXACTA de siempre (mismo
        # consumo de RNG), de modo que ai_v1 y los tests no cambian byte a byte. La via
        # gaussiana sigue siendo la RUTA POR DEFECTO cuando la microestructura esta
        # desactivada; ai_v3 la pide en todas las fases, pero eso es una eleccion del
        # spec, no del motor.
        dof_t = tail_dof_t if np.any(tail_dof_t > 2.0) else None
        # Igual con las cargas: si ninguna fase declara estres, ni siquiera se construye
        # la beta por dia y la suma sobre factores es la misma operacion (mismo BLAS,
        # mismo redondeo) que antes de que existiera este campo.
        stress_t = 1.0 + beta_stress_t if np.any(beta_stress_t != 0.0) else None

        # Innovaciones de factor: colas (t-Student) + clustering (GARCH), luego escala vol.
        w = _draw_innovations(rng, (horizon, len(factors)), dof_t)
        w = _apply_garch(w, persistence_t, news_t)
        factor_returns = drift + vol * w  # (horizon, n_factors)

        index = pd.DatetimeIndex(
            [self.anchor + timedelta(days=i) for i in range(horizon)], name="timestamp"
        )

        bars: dict[str, pd.DataFrame] = {}
        for asset in self.universe.assets:
            beta = np.array([asset.loading(f) for f in factors])
            tilt = float(spec.asset_tilts.get(asset.symbol, 0.0))

            eps = _draw_innovations(rng, (horizon,), dof_t)
            eps = _apply_garch(eps, persistence_t, news_t)
            # Componente idiosincratico con AR(1) por fase: >0 tiende, <0 revierte. Es lo
            # que rompe el mundo iid y da edge real a momentum vs mean-reversion segun el
            # regimen. Variance-matched: con idio_ar=0 se reduce al ruido de siempre.
            idio = _ar1_idio(eps, asset.idio_vol, idio_ar_t)

            # Cargas EFECTIVAS del dia (B6): en las fases de estres las betas suben, que es
            # el unico mecanismo por el que la correlacion entre activos puede acercarse a
            # 1 en las caidas. La covarianza sigue siendo B_t Sigma B_t' + D con D diagonal
            # positiva, asi que sigue siendo definida positiva por construccion.
            if stress_t is None:
                factor_component = factor_returns @ beta
                beta_t = beta  # (n_factors,) -> broadcast en la vol diaria
            else:
                beta_t = beta * stress_t[:, None]  # (horizon, n_factors)
                factor_component = np.einsum("tk,tk->t", factor_returns, beta_t)
            asset_returns = tilt + factor_component + idio

            # Vol diaria efectiva del activo, POR DIA (no promediada al horizonte): la
            # varianza de factor usa la vol de la FASE de cada dia Y la beta efectiva de
            # ese dia, asi los huecos y las mechas se ensanchan en las fases de panico.
            # Antes se promediaba y el rango intradia no reaccionaba al regimen,
            # corrompiendo el ATR.
            daily_vol_t = np.sqrt(np.sum((vol * beta_t) ** 2, axis=1) + asset.idio_vol**2)

            bars[asset.symbol] = self._build_ohlcv(
                asset_returns, asset.start_price, daily_vol_t,
                jump_intensity_t, jump_scale_t, index, rng, asset.adv_units,
            )

        return bars

    def _phase_scalar_timeline(self, spec: ScenarioSpec, horizon: int, attr: str) -> np.ndarray:
        """Expande un campo escalar de fase (idio_ar, tail_dof, ...) a un array por dia."""
        out = np.zeros(horizon)
        cursor = 0
        for phase in spec.phases:
            end = min(cursor + phase.length_days, horizon)
            out[cursor:end] = float(getattr(phase, attr))
            cursor = end
            if cursor >= horizon:
                break
        return out

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
        self, spec: ScenarioSpec, drift: np.ndarray, factors: tuple[str, ...], seed: int
    ) -> None:
        """Suma los shocks discretos como deriva puntual en su dia y factor.

        Si un shock declara jitter_days>0, su dia se recoloca por path de forma
        determinista con un RNG DERIVADO (seed+salt), sin tocar la secuencia principal
        del path: asi las velas sin shock quedan identicas y solo se mueve el evento."""
        index_of = {f: j for j, f in enumerate(factors)}
        horizon = drift.shape[0]
        jitter_rng: np.random.Generator | None = None
        for shock in spec.shocks:
            if shock.factor not in index_of:
                continue
            day = shock.day
            if shock.jitter_days > 0:
                if jitter_rng is None:
                    jitter_rng = np.random.default_rng(seed + _JITTER_SALT)
                day += int(jitter_rng.integers(-shock.jitter_days, shock.jitter_days + 1))
            if not 0 <= day < horizon:
                continue
            drift[day, index_of[shock.factor]] += shock.magnitude

    def _build_ohlcv(
        self,
        returns: np.ndarray,
        start_price: float,
        daily_vol: np.ndarray,
        jump_intensity: np.ndarray,
        jump_scale: np.ndarray,
        index: pd.DatetimeIndex,
        rng: np.random.Generator,
        base_volume: float,
    ) -> pd.DataFrame:
        """De log-retornos diarios a velas OHLCV validas (high>=cuerpo>=low, >0).

        `daily_vol` es un array POR DIA (misma longitud que returns): la escala de
        huecos y mechas de cada vela usa la volatilidad de su propia fase.

        `base_volume` es el volumen tipico del activo en unidades por barra: es lo que
        convierte la columna 'volume' en un eje de liquidez real (BTC no se negocia como
        un altcoin) en vez de en el mismo numero para los 35 activos."""
        horizon = len(returns)

        close = start_price * np.exp(np.cumsum(returns))
        prev_close = np.empty(horizon)
        prev_close[0] = start_price
        prev_close[1:] = close[:-1]

        # Apertura: cierre anterior con un hueco nocturno pequeno.
        gap = rng.standard_normal(horizon) * (self.overnight_gap_factor * daily_vol)

        # Saltos en el hueco (B4): con prob jump_intensity el dia abre con un salto de
        # tamano jump_scale*vol. Rompe la hipotesis de gaps gaussianos que hacia que los
        # stops llenaran SIEMPRE cerca del nivel: ahora un gap puede saltarse el stop y la
        # perdida de cola deja de estar subestimada. Sin saltos no se consume RNG extra.
        if np.any(jump_intensity > 0.0):
            occur = rng.random(horizon) < jump_intensity
            jump = rng.standard_normal(horizon) * (jump_scale * daily_vol)
            gap = gap + occur * jump

        open_ = prev_close * np.exp(gap)

        body_high = np.maximum(open_, close)
        body_low = np.minimum(open_, close)

        # Mechas: extension no negativa (half-normal) por encima y por debajo del cuerpo.
        wick_scale = self.wick_factor * daily_vol
        up = np.abs(rng.standard_normal(horizon)) * wick_scale
        down = np.abs(rng.standard_normal(horizon)) * wick_scale
        high = body_high * (1.0 + up)
        low = body_low * (1.0 - down)

        # Volumen sintetico: la escala es la del activo (su ADV en unidades), con ruido
        # log-normal y mas actividad en los dias de movimiento fuerte. El consumo de RNG
        # es identico al de antes -solo cambia el factor de escala-, asi que ningun path
        # ya generado cambia de OHLC ni el volumen relativo (volumen/media) se altera.
        vol_noise = np.exp(rng.standard_normal(horizon) * 0.3)
        volume = base_volume * vol_noise * (1.0 + 5.0 * np.abs(returns))

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


# Desplazamiento de semilla para el RNG del jitter de shocks: separado del RNG principal
# del path para que recolocar el dia del shock no altere el resto de las velas.
_JITTER_SALT = 987_654_321

# Tope de |phi| para que el AR(1) sea estacionario y sigma_innov real.
_MAX_AR = 0.98

# dof que aproxima la normal en los dias sin colas de un path que SI usa la via t.
_GAUSSIAN_DOF = 1e6

# Reparto news/inercia del GARCH cuando la fase no declara `vol_news`. Es el valor con el
# que se genero ai_v2 y por eso NO se toca: cambiarlo reescribiria una libreria publicada.
# El valor calibrado contra el informe de fidelidad vive en retrofit.py (es una decision
# de la fisica de la fase, no del motor).
DEFAULT_NEWS_SHARE = 0.15


def _draw_innovations(
    rng: np.random.Generator, shape: tuple[int, ...], dof_t: np.ndarray | None
) -> np.ndarray:
    """
    Innovaciones de varianza unidad (B2).

    dof_t None => gaussianas EXACTAS (mismo consumo de RNG que antes: es la via neutra
    que mantiene ai_v1 identico). Si alguna fase pide colas, t-Student POR DIA escalada
    a varianza 1 con sqrt((dof-2)/dof); los dias neutros de ese path usan dof enorme
    (~normal).
    """
    if dof_t is None:
        return rng.standard_normal(shape)
    dof = np.where(dof_t > 2.0, dof_t, _GAUSSIAN_DOF)
    if dof.ndim == 1 and len(shape) == 2:
        dof = dof[:, None]  # (horizon,1) -> broadcast a (horizon, n_factors)
    scale = np.sqrt((dof - 2.0) / dof)
    return rng.standard_t(dof, size=shape) * scale


def _apply_garch(
    innov: np.ndarray, persistence_t: np.ndarray, news_t: np.ndarray | None = None
) -> np.ndarray:
    """
    Clustering de volatilidad tipo GARCH(1,1) sobre innovaciones YA dibujadas (B3), sin
    consumir RNG extra:

        sigma2_t = (1-p) + a*p*e_{t-1}^2 + (1-a)*p*sigma2_{t-1}

    con p = `vol_persistence` de la fase (persistencia TOTAL) y a = `vol_news` (que parte
    de esa persistencia reacciona a la noticia de ayer en vez de arrastrar la inercia).
    Como los dos pesos suman p, la varianza incondicional es 1 para cualquier a
    (variance-matched): subir la reaccion a la noticia mueve el CLUSTERING medible, no el
    nivel de volatilidad.

    `news_t` None -o 0 en un dia- usa DEFAULT_NEWS_SHARE: 0 es "sin override", igual que
    tail_dof=0 es "gaussiano". Ese default es el reparto con el que se genero ai_v2, asi
    que esa libreria se sigue regenerando byte a byte; ai_v3 declara su `vol_news` en el
    propio spec, de modo que las velas las sigue determinando (spec, semilla) y no una
    constante del codigo.

    p=0 => multiplica por 1 (no-op exacto). Actua a lo largo del tiempo (eje 0).
    """
    if not np.any(persistence_t > 0.0):
        return innov

    news = (
        np.full(len(persistence_t), DEFAULT_NEWS_SHARE)
        if news_t is None
        else np.where(news_t > 0.0, news_t, DEFAULT_NEWS_SHARE)
    )

    out = np.empty_like(innov)
    is_2d = innov.ndim == 2
    ncols = innov.shape[1] if is_2d else 1
    for c in range(ncols):
        col = innov[:, c] if is_2d else innov
        res = out[:, c] if is_2d else out
        sigma2 = 1.0
        eps_prev = 0.0
        for t in range(len(col)):
            p = min(0.999, max(0.0, float(persistence_t[t])))
            a = min(1.0, max(0.0, float(news[t])))
            if t > 0:
                sigma2 = (1.0 - p) + a * p * eps_prev * eps_prev + (1.0 - a) * p * sigma2
            e = col[t] * np.sqrt(sigma2)
            res[t] = e
            eps_prev = e
    return out


def _ar1_idio(eps: np.ndarray, idio_vol: float, phi_t: np.ndarray) -> np.ndarray:
    """
    Componente idiosincratico como AR(1) con coeficiente phi por dia:
        g_t = phi_t * g_{t-1} + idio_vol * sqrt(1 - phi_t^2) * eps_t

    El factor sqrt(1 - phi^2) es VARIANCE-MATCHING: la varianza estacionaria de g es
    idio_vol^2 para cualquier phi, asi que anadir autocorrelacion NO cambia la vol total
    del activo, solo su estructura serial. phi=0 -> g_t = idio_vol * eps_t (iid de siempre).
    """
    horizon = len(eps)
    g = np.empty(horizon)
    prev = 0.0
    for t in range(horizon):
        phi = min(_MAX_AR, max(-_MAX_AR, float(phi_t[t])))
        sigma = idio_vol * np.sqrt(1.0 - phi * phi)
        prev = phi * prev + sigma * eps[t]
        g[t] = prev
    return g


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
