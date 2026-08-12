"""
Validacion multiventana: de UN numero out-of-sample a una DISTRIBUCION de numeros.

El corte unico 70/30 producia exactamente un headline score por muestra. Un numero no
tiene cola, y la cola es lo que este proyecto puntua: el CVaR@25% de una lista de un
elemento es ese mismo elemento. Es decir, el estadistico robusto existia pero no tenia
sobre que ser robusto — solo agregaba entre muestras sinteticas, no entre tramos de
historia. Aqui se cierra ese hueco: cada muestra se evalua en VARIAS ventanas OOS
disjuntas (walk-forward o CPCV, con purga y embargo) y sus headline scores se agregan
con el MISMO estadistico que ya usa el resto del sistema, `aggregate_reward`.

Que aporta, en concreto:

- **Dispersion visible.** `stats.std` y `stats.worst` dicen cuanto depende el resultado
  del tramo de historia que te toco. Con un corte unico eso era invisible por
  construccion.
- **Comparacion con el corte antiguo.** `single_split_score` lo corre ademas sobre la
  misma muestra, asi que `optimism` cuantifica la diferencia. Ojo con el nombre de esa
  propiedad: medido sobre ai_v2 la diferencia frente a la MEDIANA de las ventanas es
  ~0 —el corte unico no esta sesgado al alza, esta arbitrario— y donde si hay brecha
  sistematica es frente a la COLA, por la razon del parrafo anterior. Ver
  `scoring.validation_study` y `data/validation/`.
- **El gate sobrevive.** Los baselines pasivos se evaluan en LAS MISMAS ventanas (mismo
  encadenado de tramos, mismos costes), asi que "batir a no hacer nada" se sigue
  pudiendo decidir, ahora fold a fold.

Que NO aporta, y conviene tenerlo escrito: dentro de un backtest no se ajusta ningun
parametro, la configuracion entra fija. La ventana de train de un fold no influye en su
test —el motor construye reloj, estado y estrategias nuevos por tramo—, de modo que la
purga y el embargo no estan tapando una fuga de ajuste que aqui no existe. Protegen de
otra cosa, igual de real: el solape de HORIZONTE (una posicion abierta al final del
train sigue viva dentro del test, hasta `max_holding_days`) y el eco serial de los
retornos. Y el valor principal del esquema es tener varias ventanas OOS independientes
frente al sobreajuste del bucle EXTERIOR, que es donde este sistema elige de verdad
(el CEM barriendo configuraciones).
"""
from __future__ import annotations

import dataclasses
import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY, BacktestEngine, FoldResult
from ai_trader.backtest.metrics import (
    DEFAULT_HEADLINE_WEIGHTS,
    HeadlineWeights,
    chain_equity_curves,
    compute_metrics,
    daily_returns,
    headline_score,
    kurtosis,
    periods_per_year_for_symbols,
    skewness,
)
from ai_trader.backtest.validation import (
    DEFAULT_N_FOLDS,
    DEFAULT_N_GROUPS,
    DEFAULT_N_TEST_GROUPS,
    SCHEME_CPCV,
    SCHEME_SINGLE,
    SCHEME_WALK_FORWARD,
    Block,
    Fold,
    build_folds,
    coverage,
)
from ai_trader.config import AppConfig, StrategySpec
from ai_trader.scoring.activity import DEFAULT_ACTIVITY_FLOOR, ActivityStats
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, RewardStats, aggregate_reward
from ai_trader.scoring.baselines import (
    BASELINE_LABELS,
    BaselineGate,
    compute_baselines,
    gate,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FoldScore:
    """El headline score de UN fold, con las piezas que lo componen."""

    label: str
    test_start: datetime
    test_end: datetime
    test_days: int
    n_test_blocks: int
    score: float
    sharpe: float
    turnover: float
    max_drawdown_pct: float
    num_trades: int
    oos_observations: int
    returns_skew: float
    returns_kurtosis: float
    train_score: float | None = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "test_days": self.test_days,
            "n_test_blocks": self.n_test_blocks,
            "score": round(self.score, 4),
            "sharpe": round(self.sharpe, 3),
            "turnover": round(self.turnover, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "num_trades": self.num_trades,
            "oos_observations": self.oos_observations,
            "train_score": None if self.train_score is None else round(self.train_score, 4),
        }


@dataclass(slots=True, frozen=True)
class MultiWindowValidation:
    """
    Resultado de validar UNA muestra en varias ventanas.

    `stats` es la agregacion robusta (CVaR@alpha por defecto, el mismo estadistico que
    rankea el resto del sistema) de los headline scores de todos los folds. `reward` es
    la cifra con la que comparar; `mean`, `std`, `worst` y `best` describen la forma que
    el corte unico no podia mostrar.

    Junto a la recompensa viaja SIEMPRE la actividad (`stats.activity`, y el atajo
    `activity`): operaciones por ventana OOS y cuantas de esas ventanas estuvieron vacias.
    Sin ella un reward de 0 es ambiguo —"no perdio" y "no jugo" se escriben igual— y el
    ranking se puede ordenar por inactividad sin que nadie lo vea. Ver `scoring.activity`.
    """

    scheme: str
    folds: tuple[FoldScore, ...]
    stats: RewardStats
    purge_days: int
    embargo_days: int
    coverage: dict
    leakage_ok: bool
    headline_weights: HeadlineWeights
    baselines: dict[str, RewardStats] = dataclasses.field(default_factory=dict)
    # Series CRUDAS de los baselines, un valor por fold y en el mismo orden que `folds`.
    # `baselines` ya trae su agregado, pero agregar dos veces (CVaR de CVaR) compone dos
    # conservadurismos: quien junte varias validaciones en una sola distribucion necesita
    # los scores fold a fold, no su cola. Ver `scoring.transfer_study`.
    baseline_scores: dict[str, tuple[float, ...]] = dataclasses.field(default_factory=dict)
    baseline_gate: BaselineGate | None = None
    single_split_score: float | None = None

    @property
    def scores(self) -> list[float]:
        return [f.score for f in self.folds]

    @property
    def trades(self) -> list[int]:
        """Operaciones de cada fold, en el mismo orden que `scores`."""
        return [f.num_trades for f in self.folds]

    @property
    def activity(self) -> ActivityStats | None:
        """Atajo a `stats.activity`: la actividad medida sobre estas mismas ventanas."""
        return self.stats.activity

    @property
    def rankable(self) -> bool:
        """¿Supera el suelo de actividad declarado? Una configuracion que no lo supera se
        sigue midiendo y publicando; lo que no hace es competir. Ver `scoring.activity`."""
        return DEFAULT_ACTIVITY_FLOOR.eligible(self.activity)

    @property
    def median(self) -> float:
        return statistics.median(self.scores) if self.folds else 0.0

    @property
    def optimism(self) -> float | None:
        """Diferencia entre el corte unico y la MEDIANA de las ventanas. Positivo = el
        70/30 puntuo mejor en esta muestra.

        No es un sesgo: medido sobre 32 unidades de ai_v2 esta diferencia se centra en
        cero y va de -3.4 a +4.7. El corte unico es arbitrario, no optimista; la brecha
        sistematica esta contra la cola (`stats.reward`), no contra la mediana."""
        if self.single_split_score is None or not self.folds:
            return None
        return self.single_split_score - self.median

    @property
    def approved(self) -> bool:
        return bool(self.baseline_gate and self.baseline_gate.approved)

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "n_folds": len(self.folds),
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
            "leakage_ok": self.leakage_ok,
            "coverage": self.coverage,
            "headline_weights": self.headline_weights.as_dict(),
            "stats": self.stats.as_dict(),
            "activity": None if self.activity is None else self.activity.as_dict(),
            "rankable": self.rankable,
            "activity_floor": DEFAULT_ACTIVITY_FLOOR.as_dict(),
            "median": round(self.median, 4),
            "single_split_score": (
                None if self.single_split_score is None else round(self.single_split_score, 4)
            ),
            "optimism": None if self.optimism is None else round(self.optimism, 4),
            "approved": self.approved,
            "gate": None if self.baseline_gate is None else self.baseline_gate.as_dict(),
            "baselines": {n: s.as_dict() for n, s in self.baselines.items()},
            "folds": [f.as_dict() for f in self.folds],
        }


def resolve_purge_days(config: AppConfig) -> int:
    """
    Purga por defecto = `max_holding_days` del runner.

    No es una constante elegida a ojo: es exactamente cuantos dias puede seguir viva una
    posicion abierta el ultimo dia de train. Purgar menos dejaria que el desenlace de esa
    operacion cayera dentro del test; purgar mas seria tirar historia sin motivo.
    """
    return max(1, int(config.runner.max_holding_days))


def validate_multiwindow(
    base_config: AppConfig,
    spec: StrategySpec | None,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    scheme: str = SCHEME_WALK_FORWARD,
    n_folds: int = DEFAULT_N_FOLDS,
    n_groups: int = DEFAULT_N_GROUPS,
    n_test_groups: int = DEFAULT_N_TEST_GROUPS,
    purge_days: int | None = None,
    embargo_days: int | None = None,
    anchored: bool = True,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    run_train: bool = False,
    with_baselines: bool = True,
    compare_single_split: bool = True,
    block_cache: dict | None = None,
    baseline_cache: dict | None = None,
    signal_provider_factory: Callable[..., object] | None = None,
) -> MultiWindowValidation:
    """
    Corre UNA muestra por el plan de validacion pedido y agrega sus ventanas.

    `spec` sustituye a la estrategia del config base (como en `evaluate_sample_detailed`);
    con `None` se usa la del propio config. Todo lo demas —universo, riesgo, ejecucion,
    fees, microestructura— es el del sistema en vivo: lo que se puntua es lo que operaria.

    Determinista de punta a punta: la geometria de los folds solo depende de las fechas
    y de los parametros, y cada tramo se corre con el motor real, que ya lo es.

    `block_cache` y `baseline_cache` sirven para evaluar VARIOS esquemas sobre la misma
    muestra sin repetir tramos (walk-forward y CPCV comparten particion si n_folds+1 ==
    n_groups). Pasalos solo entre llamadas con la MISMA muestra, config y spec.

    `signal_provider_factory` se pasa tal cual al motor: es la costura por la que el canal
    de observacion sintetico entrega sus senales a las estrategias (ver `BacktestEngine`).
    None —el caso normal— deja el radar de siempre. Los BASELINES no la reciben y no es un
    olvido: son carteras pasivas que no consultan ninguna senal, y darles una las
    convertiria en otra cosa que el liston.
    """
    config = base_config if spec is None else dataclasses.replace(base_config, strategies=[spec])
    purge = resolve_purge_days(base_config) if purge_days is None else purge_days

    folds = build_folds(
        start, end,
        scheme=scheme, n_folds=n_folds, n_groups=n_groups, n_test_groups=n_test_groups,
        purge_days=purge, embargo_days=embargo_days, anchored=anchored,
    )
    engine = BacktestEngine.from_bars(
        config, bars,
        starting_equity=starting_equity,
        headline_weights=headline_weights,
        signal_provider_factory=signal_provider_factory,
    )
    results = engine.run_folds(folds, run_train=run_train, block_cache=block_cache)

    scores = tuple(_fold_score(r) for r in results)
    # La actividad se mide sobre EXACTAMENTE las mismas ventanas que la recompensa, y por
    # eso entra aqui y no en un paso posterior que pudiera usar otro conjunto de folds.
    fold_trades = [s.num_trades for s in scores]
    stats = aggregate_reward([s.score for s in scores], alpha=cvar_alpha, trades=fold_trades)

    baseline_series: dict[str, list[float]] = {}
    baseline_stats: dict[str, RewardStats] = {}
    baseline_gate: BaselineGate | None = None
    if with_baselines:
        baseline_series = _baseline_fold_scores(
            base_config, bars, folds,
            starting_equity=starting_equity, headline_weights=headline_weights,
            cache=baseline_cache,
        )
        baseline_stats = {
            name: aggregate_reward(values, alpha=cvar_alpha)
            for name, values in baseline_series.items()
        }
        baseline_gate = gate(
            [s.score for s in scores], baseline_series, alpha=cvar_alpha,
            # El gate exige ademas actividad: batir a los pasivos con una curva plana es
            # batirlos por no jugar. Ver `baselines.BaselineGate`.
            trades=fold_trades,
            # Un baseline que no se pudo construir (p.ej. SPY en un universo solo cripto)
            # se DECLARA; el gate no puede aprobar contra una lista silenciosamente corta.
            missing=tuple(n for n in sorted(BASELINE_LABELS) if n not in baseline_series),
        )

    single = None
    if compare_single_split and scheme != SCHEME_SINGLE:
        single = _single_split_score(
            config, bars, start, end,
            starting_equity=starting_equity, headline_weights=headline_weights,
            signal_provider_factory=signal_provider_factory,
        )

    validation = MultiWindowValidation(
        scheme=scheme,
        folds=scores,
        stats=stats,
        purge_days=purge,
        embargo_days=folds[0].embargo_days,
        coverage=coverage(folds),
        leakage_ok=all(f.leakage().ok for f in folds),
        headline_weights=headline_weights,
        baselines=baseline_stats,
        baseline_scores={n: tuple(v) for n, v in baseline_series.items()},
        baseline_gate=baseline_gate,
        single_split_score=single,
    )
    activity = validation.activity
    logger.info(
        "Validacion %s | %d folds | reward(CVaR)=%+.4f | mediana=%+.4f | std=%.4f "
        "| peor=%+.4f | ops/ventana=%.1f (mediana %.0f, %.0f%% vacias)%s | corte unico=%s "
        "| optimismo=%s",
        scheme, len(scores), stats.reward, validation.median, stats.std, stats.worst,
        activity.trades_per_window if activity else float("nan"),
        activity.median_trades_per_window if activity else float("nan"),
        activity.zero_window_pct if activity else float("nan"),
        "" if validation.rankable else " NO RANKEABLE",
        "n/a" if single is None else f"{single:+.4f}",
        "n/a" if validation.optimism is None else f"{validation.optimism:+.4f}",
    )
    return validation


# ------------------------------------------------------------------ internals -------


def _fold_score(result: FoldResult) -> FoldScore:
    metrics = result.test.metrics
    returns = daily_returns(result.test.equity_curve)
    return FoldScore(
        label=result.fold.label,
        test_start=result.fold.test_start,
        test_end=result.fold.test[-1].last_day,
        test_days=result.fold.test_days,
        n_test_blocks=len(result.fold.test),
        score=result.headline_score,
        sharpe=metrics.sharpe,
        turnover=metrics.turnover,
        max_drawdown_pct=metrics.max_drawdown_pct,
        num_trades=metrics.num_trades,
        oos_observations=len(returns),
        returns_skew=skewness(returns),
        returns_kurtosis=kurtosis(returns),
        train_score=result.train_headline_score,
    )


def _single_split_score(
    config: AppConfig,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    starting_equity: float,
    headline_weights: HeadlineWeights,
    signal_provider_factory: Callable[..., object] | None = None,
) -> float | None:
    """El headline del corte unico 70/30, para poder medir cuanto optimismo aportaba.
    Si falla no se propaga: la comparacion es informativa, no el resultado."""
    try:
        result = BacktestEngine.from_bars(
            config, bars,
            starting_equity=starting_equity,
            headline_weights=headline_weights,
            signal_provider_factory=signal_provider_factory,
        ).run(start, end)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Corte unico de referencia no evaluable (%s)", exc)
        return None
    return result.headline_score


def _baseline_fold_scores(
    base_config: AppConfig,
    bars: dict[str, pd.DataFrame],
    folds: Sequence[Fold],
    *,
    starting_equity: float,
    headline_weights: HeadlineWeights,
    cache: dict | None = None,
) -> dict[str, list[float]]:
    """
    Puntua los baselines pasivos en LAS MISMAS ventanas que la estrategia.

    Un baseline se compra al principio de cada TRAMO y se liquida al final, igual que la
    estrategia liquida al cierre de ventana; los tramos de un fold se encadenan con el
    mismo `chain_equity_curves`. Nadie compara gratis: los baselines pagan sus dos patas
    de costes y sufren el mismo troceado del calendario.

    Solo se devuelven los baselines disponibles en TODOS los folds: una serie con huecos
    no seria comparable fold a fold.
    """
    periods = periods_per_year_for_symbols(base_config.runner.symbols)
    if cache is None:
        cache = {}
    per_fold: list[dict[str, float]] = []

    for fold in folds:
        blocks: dict[str, list] = {}
        for block in fold.test:
            for name, baseline in _baselines_on(
                bars, block, base_config, starting_equity, headline_weights, periods, cache
            ).items():
                blocks.setdefault(name, []).append(baseline)

        scores: dict[str, float] = {}
        for name, parts in blocks.items():
            if len(parts) != len(fold.test):  # el baseline falta en algun tramo
                continue
            chained = chain_equity_curves([p.curve for p in parts])
            trades = [t for p in parts for t in p.trades]
            metrics = compute_metrics(
                chained.points, trades,
                periods_per_year=periods, active_days=chained.active_days,
            )
            scores[name] = headline_score(metrics, headline_weights)
        per_fold.append(scores)

    if not per_fold:
        return {}
    common = set.intersection(*(set(d) for d in per_fold)) if per_fold else set()
    return {name: [d[name] for d in per_fold] for name in sorted(common)}


def _baselines_on(
    bars: dict[str, pd.DataFrame],
    block: Block,
    base_config: AppConfig,
    starting_equity: float,
    headline_weights: HeadlineWeights,
    periods: int,
    cache: dict,
) -> dict:
    """Baselines de UN tramo, cacheados: CPCV reevalua los mismos tramos muchas veces."""
    key = ("__block__", block)
    cached = cache.get(key)
    if cached is None:
        cached = compute_baselines(
            bars, block.start, block.last_day,
            starting_equity=starting_equity,
            fee_rate=base_config.execution.fee_rate,
            slippage_bps=base_config.execution.slippage_bps,
            weights=headline_weights,
            periods_per_year=periods,
        )
        cache[key] = cached
    return cached


__all__ = [
    "SCHEME_CPCV",
    "SCHEME_SINGLE",
    "SCHEME_WALK_FORWARD",
    "FoldScore",
    "MultiWindowValidation",
    "resolve_purge_days",
    "validate_multiwindow",
]
