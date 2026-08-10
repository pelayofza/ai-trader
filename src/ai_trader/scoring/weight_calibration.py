"""
Calibracion CON EVIDENCIA de los pesos del headline score.

El headline es `Sharpe_OOS - lambda*turnover - kappa*maxDD` (backtest.metrics). Los
pesos no son una preferencia estetica: son una REGLA DE SELECCION, y una regla de
seleccion se juzga por lo que elige. Este modulo mide exactamente eso sobre la libreria
sintetica, en dos ejes que no se deben confundir:

- Eje TEMPORAL (dentro de cada muestra): ventana train vs ventana test. Da la
  ESTABILIDAD DE LA ELECCION: si rankeo un conjunto de configuraciones con los datos
  que vi, ¿sobrevive ese ranking fuera de muestra? -> `rank_ic`, `top1_oos_pct`.
- Eje de ESCENARIOS (hold-out de arquetipos, el de scenario_split): escenarios train vs
  validation, siempre con la ventana OOS. Da el GAP TRAIN-VALIDATION: cuanto se
  desinfla en arquetipos nunca vistos la configuracion que elegi -> `selection_gap`.

La pieza que hace barato el barrido: los COMPONENTES (Sharpe, turnover, maxDD) no
dependen de los pesos. Se corre el backtest UNA vez por (configuracion, muestra), se
guardan los componentes crudos de ambas ventanas y despues la rejilla (lambda, kappa)
se evalua en memoria. 1.000 backtests, no 1.000 x |rejilla|.

Ademas de la rejilla, `turnover_cost_audit` responde la pregunta de doble cobro: la
curva de equity YA paga fee_rate + slippage cada vez que rota, asi que parte del coste
de rotar esta dentro del Sharpe. La auditoria mide ese coste implicito EN UNIDADES DE
SHARPE (que es la unidad de lambda) para saber si la penalizacion explicita es un
regularizador modesto o una segunda factura del mismo tamano.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY, BacktestEngine, WindowResult
from ai_trader.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    HeadlineWeights,
    PerformanceMetrics,
    headline_score,
    window_days,
)
from ai_trader.config import AppConfig, StrategySpec
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, aggregate_reward
from ai_trader.scoring.scenario_split import ScenarioSplit
from ai_trader.scoring.search_space import get_space

logger = logging.getLogger(__name__)

# Rejilla por defecto. Cubre dos ordenes de magnitud alrededor del valor razonado
# original (0.5 / 1.0) e incluye el 0 en ambos ejes: sin el 0 no se puede afirmar que
# penalizar sea mejor que no penalizar.
DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_KAPPA_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)

# Evidencia publicada del estudio. La generan `weight_study` y la consumen los
# generadores de docs y dashboard, y el test que congela DEFAULT_HEADLINE_WEIGHTS: los
# pesos por defecto y las cifras que los justifican no pueden separarse.
CALIBRATION_REPORT = Path("data") / "calibration" / "report_ai_v2.json"


def load_calibration_report(path: Path | str = CALIBRATION_REPORT) -> dict | None:
    """Lee el informe del estudio. Devuelve None si no esta, para que los generadores de
    documentacion degraden a prosa sin cifras en vez de romperse."""
    report = Path(path)
    if not report.exists():
        return None
    return json.loads(report.read_text(encoding="utf-8"))


def grid_point(report: dict, lambda_turnover: float, kappa_maxdd: float) -> dict | None:
    """Punto de la rejilla publicada correspondiente a unos pesos concretos."""
    for point in report.get("grid", {}).get("points", []):
        if (
            point["lambda_turnover"] == lambda_turnover
            and point["kappa_maxdd"] == kappa_maxdd
        ):
            return point
    return None


# ------------------------------------------------------------ componentes crudos -----


@dataclass(frozen=True, slots=True)
class WindowComponents:
    """
    Piezas crudas de UNA ventana de UNA muestra, INDEPENDIENTES de los pesos.

    Guardar las metricas enteras (no solo los tres sumandos) es deliberado: el audit de
    costes necesita volatilidad y comisiones, y el score se recalcula llamando a
    `headline_score`, nunca reimplementando la formula (que podria derivar).
    """

    metrics: PerformanceMetrics
    days: int
    starting_equity: float

    def score(self, weights: HeadlineWeights) -> float:
        return headline_score(self.metrics, weights)

    @property
    def traded_notional_usd(self) -> float:
        """Notional rotado en la ventana, reconstruido desde el turnover (que es
        notional/dia en unidades del equity inicial). Cuenta las dos patas."""
        return self.metrics.turnover * self.starting_equity * self.days

    def as_dict(self) -> dict:
        return {
            "metrics": dataclasses.asdict(self.metrics),
            "days": self.days,
            "starting_equity": self.starting_equity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WindowComponents:
        return cls(
            metrics=PerformanceMetrics(**data["metrics"]),
            days=int(data["days"]),
            starting_equity=float(data["starting_equity"]),
        )


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """Un backtest completo: una configuracion sobre una muestra (escenario, path), con
    los componentes de la ventana in-sample (train) y out-of-sample (test)."""

    config_id: str
    strategy_type: str
    scenario_id: str
    path_index: int
    in_sample: WindowComponents
    out_of_sample: WindowComponents
    failed: bool = False

    @property
    def sample_key(self) -> tuple[str, int]:
        return (self.scenario_id, self.path_index)

    def as_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "strategy_type": self.strategy_type,
            "scenario_id": self.scenario_id,
            "path_index": self.path_index,
            "in_sample": self.in_sample.as_dict(),
            "out_of_sample": self.out_of_sample.as_dict(),
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationSample:
        return cls(
            config_id=data["config_id"],
            strategy_type=data["strategy_type"],
            scenario_id=data["scenario_id"],
            path_index=int(data["path_index"]),
            in_sample=WindowComponents.from_dict(data["in_sample"]),
            out_of_sample=WindowComponents.from_dict(data["out_of_sample"]),
            failed=bool(data.get("failed", False)),
        )


def _components(window: WindowResult, starting_equity: float) -> WindowComponents:
    return WindowComponents(
        metrics=window.metrics,
        days=window_days(window.equity_curve),
        starting_equity=starting_equity,
    )


def evaluate_calibration_sample(
    base_config: AppConfig,
    spec: StrategySpec,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    config_id: str,
    scenario_id: str,
    path_index: int,
    split_ratio: float = 0.7,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
) -> CalibrationSample:
    """
    Corre el backtest REAL de una configuracion sobre una muestra y devuelve los
    componentes de AMBAS ventanas.

    Es el gemelo de `sample_eval.evaluate_sample_detailed` con una diferencia unica y
    necesaria: aquel devuelve solo out-of-sample (es lo que el optimizador puntua), y la
    calibracion necesita tambien la ventana in-sample para medir si el ranking sobrevive
    al salir de muestra. Todo lo demas (universo, riesgo, ejecucion, fees, slippage) es
    el config del sistema, sin tocar.
    """
    config = dataclasses.replace(base_config, strategies=[spec])
    try:
        result = BacktestEngine.from_bars(
            config, bars, starting_equity=starting_equity
        ).run(start, end, split_ratio=split_ratio)
    except Exception as exc:  # noqa: BLE001 - una muestra mala no debe tumbar el estudio
        logger.warning(
            "Calibration backtest failed (%s | %s p%d): %s",
            config_id, scenario_id, path_index, exc,
        )
        empty = WindowComponents(_zero_metrics(starting_equity), 0, starting_equity)
        return CalibrationSample(
            config_id=config_id,
            strategy_type=spec.type,
            scenario_id=scenario_id,
            path_index=path_index,
            in_sample=empty,
            out_of_sample=empty,
            failed=True,
        )

    return CalibrationSample(
        config_id=config_id,
        strategy_type=spec.type,
        scenario_id=scenario_id,
        path_index=path_index,
        in_sample=_components(result.train, starting_equity),
        out_of_sample=_components(result.test, starting_equity),
        failed=False,
    )


def _zero_metrics(starting_equity: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        starting_equity=starting_equity,
        ending_equity=starting_equity,
        total_return_pct=0.0,
        cagr_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe=0.0,
        sortino=0.0,
        calmar=0.0,
        volatility_pct=0.0,
        turnover=0.0,
        num_trades=0,
        win_rate_pct=0.0,
        profit_factor=None,
        avg_win_usd=0.0,
        avg_loss_usd=0.0,
        avg_holding_days=0.0,
        total_fees_usd=0.0,
    )


# ------------------------------------------------- conjunto de configuraciones -------


def candidate_specs(strategy_type: str, n: int, *, seed: int = 0) -> list[StrategySpec]:
    """
    Conjunto de configuraciones que se van a rankear, por HIPERCUBO LATINO sobre el
    espacio de busqueda real de la primitiva.

    Latino y no rejilla regular: con 7-8 dimensiones una rejilla es inviable, y un
    muestreo uniforme deja huecos. El hipercubo garantiza que cada dimension quede
    cubierta en toda su extension, que es justo lo que hace falta para que el conjunto
    contenga configuraciones de rotacion alta Y baja (sin ese contraste, barrer lambda
    no informa de nada).

    Determinista: mismo (strategy_type, n, seed) -> mismas configuraciones.
    """
    if n < 2:
        raise ValueError("need at least 2 candidate configs to rank")

    space = get_space(strategy_type)
    rng = np.random.default_rng(seed)

    # Hipercubo latino: un estrato por muestra en cada dimension, permutado por dimension.
    strata = (np.arange(n, dtype=float)[:, None] + rng.random((n, space.dim))) / n
    for j in range(space.dim):
        strata[:, j] = strata[rng.permutation(n), j]
    vectors = space.lows + strata * (space.highs - space.lows)

    specs: list[StrategySpec] = []
    seen: set[tuple] = set()
    for i, vector in enumerate(vectors):
        params = space.to_params(vector)
        fingerprint = tuple(sorted(params.items()))
        if fingerprint in seen:
            logger.warning("Duplicate candidate config dropped (%s #%d)", strategy_type, i)
            continue
        seen.add(fingerprint)
        specs.append(
            StrategySpec(type=strategy_type, id=f"{strategy_type}#{i:02d}", params=params)
        )
    return specs


# ----------------------------------------------------------- correlacion de rangos ---


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rangos con empates promediados (lo que exige Spearman para no sesgar)."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Correlacion de rangos de Spearman. Devuelve NaN si alguna serie es constante (no
    hay ranking que correlacionar) o si hay menos de 2 elementos: NaN se propaga y se
    filtra explicitamente, en vez de fingir un 0 que se leeria como "sin relacion".
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size != y.size:
        raise ValueError("spearman needs two series of the same length")
    if x.size < 2:
        return float("nan")

    rx = _average_ranks(x)
    ry = _average_ranks(y)
    sx = rx.std()
    sy = ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _percentile_of(value: float, population: Sequence[float]) -> float:
    """Percentil (0-100) que ocupa `value` dentro de `population`. 100 = es el mejor."""
    arr = np.asarray(list(population), dtype=float)
    if arr.size < 2:
        return float("nan")
    below = float((arr < value).sum())
    equal = float((arr == value).sum())
    return (below + 0.5 * equal) / arr.size * 100.0


# ------------------------------------------------------------------- la rejilla ------


@dataclass(frozen=True, slots=True)
class WeightGridPoint:
    """
    Una combinacion (lambda, kappa) evaluada sobre el conjunto de configuraciones.

    Las dos cifras de cabecera son adimensionales A PROPOSITO: subir lambda o kappa
    encoge y desplaza la escala del score, asi que comparar gaps crudos entre puntos de
    la rejilla premiaria a los pesos grandes por deflactar la unidad, no por elegir
    mejor. `rank_ic` es invariante a la escala por construccion y `selection_gap_norm`
    esta normalizado por la dispersion entre configuraciones de ese mismo punto.
    """

    lambda_turnover: float
    kappa_maxdd: float
    # --- estabilidad de la eleccion (eje temporal: ventana train -> ventana test) ---
    rank_ic_mean: float  # media, por muestra, de Spearman(ranking IS, ranking OOS)
    rank_ic_se: float  # error estandar de esa media (mide si la diferencia es real)
    rank_ic_pooled: float  # Spearman de los rankings agregados (CVaR) IS vs OOS
    # Comparacion PAREADA contra no penalizar nada (lambda=kappa=0), muestra a muestra.
    # Es la cifra que decide si penalizar aporta algo: los dos puntos de la rejilla se
    # miden sobre las MISMAS muestras, asi que la varianza comun se cancela y el
    # contraste es mucho mas sensible que restar dos medias con sus errores sueltos.
    rank_ic_gain: float
    rank_ic_gain_se: float
    top1_oos_pct: float  # percentil OOS de la config que gana in-sample (100 = mejor)
    # --- gap train-validation (eje de escenarios: hold-out de arquetipos) ---
    selected_config: str
    train_reward: float
    validation_reward: float
    selection_gap: float
    selection_gap_norm: float  # gap / std entre configs del reward de train
    # --- que se ha elegido, en magnitudes fisicas (no dependen de los pesos) ---
    val_sharpe: float
    val_turnover: float
    val_max_drawdown_pct: float

    def as_dict(self) -> dict:
        return {
            "lambda_turnover": self.lambda_turnover,
            "kappa_maxdd": self.kappa_maxdd,
            "rank_ic_mean": round(self.rank_ic_mean, 4),
            "rank_ic_se": round(self.rank_ic_se, 4),
            "rank_ic_pooled": round(self.rank_ic_pooled, 4),
            "rank_ic_gain": round(self.rank_ic_gain, 4),
            "rank_ic_gain_se": round(self.rank_ic_gain_se, 4),
            "top1_oos_pct": round(self.top1_oos_pct, 2),
            "selected_config": self.selected_config,
            "train_reward": round(self.train_reward, 4),
            "validation_reward": round(self.validation_reward, 4),
            "selection_gap": round(self.selection_gap, 4),
            "selection_gap_norm": round(self.selection_gap_norm, 4),
            "val_sharpe": round(self.val_sharpe, 4),
            "val_turnover": round(self.val_turnover, 4),
            "val_max_drawdown_pct": round(self.val_max_drawdown_pct, 3),
        }


class CalibrationMatrix:
    """
    Matriz (configuracion x muestra) de componentes crudos, ya limpia.

    Una configuracion con algun backtest fallido se DESCARTA entera y se declara en
    `dropped`: rankear una configuracion sobre un subconjunto distinto de muestras que
    las demas no es rankear, y rellenarla con una penalizacion inventada contaminaria
    justo la cifra que se quiere medir.
    """

    def __init__(self, samples: Sequence[CalibrationSample]) -> None:
        by_config: dict[str, dict[tuple[str, int], CalibrationSample]] = {}
        for sample in samples:
            by_config.setdefault(sample.config_id, {})[sample.sample_key] = sample

        sample_keys = sorted({s.sample_key for s in samples})
        dropped: list[str] = []
        kept: dict[str, list[CalibrationSample]] = {}
        for config_id in sorted(by_config):
            rows = by_config[config_id]
            if len(rows) != len(sample_keys) or any(r.failed for r in rows.values()):
                dropped.append(config_id)
                continue
            kept[config_id] = [rows[key] for key in sample_keys]

        if dropped:
            logger.warning(
                "Dropping %d/%d configs with failed or incomplete backtests: %s",
                len(dropped), len(by_config), ", ".join(dropped),
            )

        self.sample_keys: list[tuple[str, int]] = sample_keys
        self.config_ids: list[str] = list(kept)
        self.dropped: list[str] = dropped
        self._rows = kept

    @property
    def n_configs(self) -> int:
        return len(self.config_ids)

    @property
    def n_samples(self) -> int:
        return len(self.sample_keys)

    def rows(self, config_id: str) -> list[CalibrationSample]:
        return self._rows[config_id]

    def all_samples(self) -> list[CalibrationSample]:
        return [s for config_id in self.config_ids for s in self._rows[config_id]]

    def scores(self, weights: HeadlineWeights, *, window: str) -> np.ndarray:
        """Matriz [config][muestra] de headline scores para unos pesos dados."""
        if window not in ("in_sample", "out_of_sample"):
            raise ValueError("window must be 'in_sample' or 'out_of_sample'")
        return np.array(
            [
                [getattr(s, window).score(weights) for s in self._rows[config_id]]
                for config_id in self.config_ids
            ],
            dtype=float,
        )

    def sample_mask(self, scenario_ids: Sequence[str]) -> np.ndarray:
        wanted = set(scenario_ids)
        return np.array([key[0] in wanted for key in self.sample_keys], dtype=bool)


def filter_active_configs(
    samples: Sequence[CalibrationSample], *, min_median_trades: int
) -> list[CalibrationSample]:
    """
    Deja solo las configuraciones que OPERAN de verdad (mediana de trades OOS por encima
    del umbral).

    Sirve para comprobar la robustez de la rejilla: en un conjunto donde media docena de
    configuraciones apenas abre posiciones, el ranking lo domina el contraste
    "viva vs muerta" y no el arbitraje entre Sharpe, rotacion y caida que lambda y kappa
    existen para resolver. Se reporta el barrido sobre ambos conjuntos; si la conclusion
    cambia entre uno y otro, la conclusion no es solida.
    """
    trades: dict[str, list[int]] = {}
    for sample in samples:
        if not sample.failed:
            trades.setdefault(sample.config_id, []).append(sample.out_of_sample.metrics.num_trades)
    keep = {
        config_id
        for config_id, values in trades.items()
        if values and float(np.median(values)) >= min_median_trades
    }
    return [s for s in samples if s.config_id in keep]


def per_sample_rank_ic(matrix: CalibrationMatrix, weights: HeadlineWeights) -> np.ndarray:
    """
    Spearman(ranking in-sample, ranking out-of-sample) MUESTRA A MUESTRA.

    Devolver el vector entero (con NaN donde la muestra es degenerada) y no solo su media
    es lo que permite comparar dos puntos de la rejilla de forma pareada.
    """
    is_scores = matrix.scores(weights, window="in_sample")
    oos_scores = matrix.scores(weights, window="out_of_sample")
    return np.array(
        [spearman(is_scores[:, j], oos_scores[:, j]) for j in range(matrix.n_samples)],
        dtype=float,
    )


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    usable = values[~np.isnan(values)]
    if usable.size == 0:
        return float("nan"), float("nan")
    if usable.size == 1:
        return float(usable[0]), float("nan")
    return float(usable.mean()), float(usable.std(ddof=1) / math.sqrt(usable.size))


def evaluate_weight_point(
    matrix: CalibrationMatrix,
    weights: HeadlineWeights,
    split: ScenarioSplit,
    *,
    alpha: float = DEFAULT_CVAR_ALPHA,
    baseline_ic: np.ndarray | None = None,
) -> WeightGridPoint:
    """
    Mide UN punto (lambda, kappa) de la rejilla sobre la matriz ya calculada.

    `baseline_ic` es el rank IC por muestra del punto de referencia (por defecto, no
    penalizar nada). Si se pasa, se reporta ademas la ganancia PAREADA contra el.
    """
    is_scores = matrix.scores(weights, window="in_sample")
    oos_scores = matrix.scores(weights, window="out_of_sample")

    # --- eje temporal: ¿el ranking que veo dentro de muestra sobrevive fuera? ---
    ic = per_sample_rank_ic(matrix, weights)
    ic_mean, ic_se = _mean_and_se(ic)
    if baseline_ic is None:
        gain, gain_se = 0.0, 0.0
    else:
        gain, gain_se = _mean_and_se(ic - baseline_ic)

    is_reward = np.array([aggregate_reward(row, alpha=alpha).reward for row in is_scores])
    oos_reward = np.array([aggregate_reward(row, alpha=alpha).reward for row in oos_scores])
    ic_pooled = spearman(is_reward, oos_reward)
    top1_oos = _percentile_of(float(oos_reward[int(np.argmax(is_reward))]), oos_reward)

    # --- eje de escenarios: hold-out de arquetipos, siempre con la ventana OOS ---
    train_mask = matrix.sample_mask(split.train)
    val_mask = matrix.sample_mask(split.validation)
    train_reward = np.array(
        [aggregate_reward(row[train_mask], alpha=alpha).reward for row in oos_scores]
    )
    val_reward = np.array(
        [aggregate_reward(row[val_mask], alpha=alpha).reward for row in oos_scores]
    )

    winner = int(np.argmax(train_reward))
    spread = float(train_reward.std(ddof=1)) if train_reward.size > 1 else 0.0
    gap = float(train_reward[winner] - val_reward[winner])

    winner_rows = matrix.rows(matrix.config_ids[winner])
    val_rows = [row for row, keep in zip(winner_rows, val_mask) if keep]

    return WeightGridPoint(
        lambda_turnover=weights.lambda_turnover,
        kappa_maxdd=weights.kappa_maxdd,
        rank_ic_mean=ic_mean,
        rank_ic_se=ic_se,
        rank_ic_pooled=ic_pooled,
        rank_ic_gain=gain,
        rank_ic_gain_se=gain_se,
        top1_oos_pct=top1_oos,
        selected_config=matrix.config_ids[winner],
        train_reward=float(train_reward[winner]),
        validation_reward=float(val_reward[winner]),
        selection_gap=gap,
        selection_gap_norm=(gap / spread) if spread > 0 else float("nan"),
        val_sharpe=float(np.mean([r.out_of_sample.metrics.sharpe for r in val_rows])),
        val_turnover=float(np.mean([r.out_of_sample.metrics.turnover for r in val_rows])),
        val_max_drawdown_pct=float(
            np.mean([r.out_of_sample.metrics.max_drawdown_pct for r in val_rows])
        ),
    )


def sweep_weights(
    matrix: CalibrationMatrix,
    split: ScenarioSplit,
    *,
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    kappas: Sequence[float] = DEFAULT_KAPPA_GRID,
    alpha: float = DEFAULT_CVAR_ALPHA,
) -> list[WeightGridPoint]:
    """
    Barre la rejilla completa. Coste ~0: no hay backtests aqui, solo aritmetica sobre los
    componentes ya medidos.

    La referencia de la comparacion pareada es NO PENALIZAR NADA (lambda = kappa = 0):
    la pregunta que el estudio tiene que responder antes que ninguna otra es si penalizar
    aporta algo, no cual de dos penalizaciones es mejor.
    """
    baseline_ic = per_sample_rank_ic(matrix, HeadlineWeights(0.0, 0.0))
    return [
        evaluate_weight_point(
            matrix,
            HeadlineWeights(lambda_turnover=lam, kappa_maxdd=kap),
            split,
            alpha=alpha,
            baseline_ic=baseline_ic,
        )
        for lam in lambdas
        for kap in kappas
    ]


# --------------------------------------------------- ¿se cobra dos veces la rotacion? -


@dataclass(frozen=True, slots=True)
class TurnoverCostAudit:
    """
    Coste de rotar que YA paga la curva de equity, expresado EN UNIDADES DE SHARPE, que
    son las unidades de lambda.

    Cadena de conversion, toda medida sobre las mismas ventanas OOS del estudio:
      coste diario (fraccion de equity) = cost_rate * turnover
      arrastre anualizado               = cost_rate * turnover * 365
      arrastre en Sharpe                = arrastre anualizado / vol anualizada
      => lambda IMPLICITO               = cost_rate * 365 / vol   (por unidad de turnover)

    `measured_fee_rate` es el control de que la cadena es correcta y no una hipotesis:
    comisiones realmente cobradas / notional rotado reconstruido desde el turnover. Si
    no reproduce el fee_rate del config, la aritmetica de arriba no se sostiene.
    """

    fee_rate: float
    slippage_rate: float
    cost_rate: float
    measured_fee_rate: float
    min_trades: int
    n_windows: int
    median_turnover: float
    p90_turnover: float
    median_volatility: float  # anualizada, en fraccion
    implied_lambda_median: float
    implied_lambda_p25: float
    implied_lambda_p75: float
    median_sharpe_drag: float  # puntos de Sharpe que la friccion ya se come

    def as_dict(self) -> dict:
        return {
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "cost_rate": self.cost_rate,
            "measured_fee_rate": round(self.measured_fee_rate, 6),
            "min_trades": self.min_trades,
            "n_windows": self.n_windows,
            "median_turnover": round(self.median_turnover, 4),
            "p90_turnover": round(self.p90_turnover, 4),
            "median_volatility": round(self.median_volatility, 4),
            "implied_lambda_median": round(self.implied_lambda_median, 4),
            "implied_lambda_p25": round(self.implied_lambda_p25, 4),
            "implied_lambda_p75": round(self.implied_lambda_p75, 4),
            "median_sharpe_drag": round(self.median_sharpe_drag, 4),
        }


def turnover_cost_audit(
    samples: Sequence[CalibrationSample],
    *,
    fee_rate: float,
    slippage_bps: float,
    window: str = "out_of_sample",
    min_trades: int = 1,
) -> TurnoverCostAudit:
    """
    Mide el lambda implicito en la friccion ya pagada por la curva de equity.

    `min_trades` filtra ventanas casi inactivas: con 5 operaciones y un 1% de
    volatilidad el cociente coste/vol se dispara y contamina la mediana con un regimen
    en el que la rotacion no es el problema de nadie.

    LIMITACION conocida: `slippage_bps` es el coste PLANO de referencia del config, no
    lo que cobra hoy el motor de ejecucion (spread por simbolo + volatilidad + impacto,
    ver execution/microstructure.py). El lambda implicito que sale de aqui es, por tanto,
    una COTA INFERIOR de la friccion real: el modelo cobra mas, no menos, asi que la
    conclusion del estudio -que la rotacion ya se paga dentro del Sharpe- se refuerza.
    Re-medirlo con el slippage realmente cobrado por operacion esta pendiente.
    """
    slippage_rate = slippage_bps / 10_000.0
    cost_rate = fee_rate + slippage_rate

    turnovers: list[float] = []
    vols: list[float] = []
    implied: list[float] = []
    drags: list[float] = []
    fees_usd = 0.0
    notional_usd = 0.0

    for sample in samples:
        if sample.failed:
            continue
        components: WindowComponents = getattr(sample, window)
        metrics = components.metrics
        if metrics.num_trades < max(1, min_trades) or components.days <= 0:
            continue

        fees_usd += metrics.total_fees_usd
        notional_usd += components.traded_notional_usd
        turnovers.append(metrics.turnover)

        vol = metrics.volatility_pct / 100.0
        if vol <= 0:
            continue
        vols.append(vol)
        # lambda implicito = puntos de Sharpe por unidad de turnover que ya paga la curva
        implied.append(cost_rate * TRADING_DAYS_PER_YEAR / vol)
        drags.append(cost_rate * metrics.turnover * TRADING_DAYS_PER_YEAR / vol)

    if not implied:
        raise ValueError("no usable windows for the turnover cost audit")

    return TurnoverCostAudit(
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        cost_rate=cost_rate,
        measured_fee_rate=(fees_usd / notional_usd) if notional_usd > 0 else float("nan"),
        min_trades=max(1, min_trades),
        n_windows=len(implied),
        median_turnover=float(np.median(turnovers)),
        p90_turnover=float(np.percentile(turnovers, 90)),
        median_volatility=float(np.median(vols)),
        implied_lambda_median=float(np.median(implied)),
        implied_lambda_p25=float(np.percentile(implied, 25)),
        implied_lambda_p75=float(np.percentile(implied, 75)),
        median_sharpe_drag=float(np.median(drags)),
    )
