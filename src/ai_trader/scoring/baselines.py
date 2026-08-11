"""
Baselines pasivos y el GATE que deben superar las estrategias.

Una estrategia no "funciona" porque su score sea positivo: funciona si bate a lo que
consigue cualquiera sin hacer nada. Aqui se construyen tres alternativas pasivas sobre
LA MISMA ventana out-of-sample y con LOS MISMOS costes que paga la estrategia:

- `btc_hold`      : comprar y mantener BTC.
- `equal_weight`  : cartera equiponderada del universo (comprada el primer dia, sin
                    rebalanceo: rebalancear seria rotacion, y la rotacion se paga).
- `spy_hold`      : comprar y mantener SPY.

Cada baseline se puntua con el MISMO headline score (Sharpe - lambda*turnover -
kappa*maxDD), asi que la comparacion es homogenea: los baselines tambien pagan sus dos
patas de comisiones y su drawdown. Un baseline cuyo simbolo no esta en las barras de la
muestra no se inventa: simplemente no aparece, y quien reporta dice cual falto.

El gate tiene DOS condiciones desde que se midio que una sola no bastaba: batir al mejor
baseline y superar el suelo de actividad (`scoring.activity`). Una curva plana bate a los
pasivos en un mercado que cae sin haber operado nunca, y eso no es batirlos. Ver
`BaselineGate`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ai_trader.backtest.metrics import (
    DEFAULT_HEADLINE_WEIGHTS,
    EquityPoint,
    HeadlineWeights,
    PerformanceMetrics,
    compute_metrics,
    headline_score,
    periods_per_year_for_symbols,
)
from ai_trader.scoring.activity import (
    DEFAULT_ACTIVITY_FLOOR,
    ActivityFloor,
    ActivityStats,
)
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, RewardStats, aggregate_reward
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.schemas import Position, PositionStatus, Side

BTC_SYMBOL = "BTC/USDT"
SPY_SYMBOL = "SPY"

BASELINE_BTC = "btc_hold"
BASELINE_EQUAL_WEIGHT = "equal_weight"
BASELINE_SPY = "spy_hold"

BASELINE_LABELS = {
    BASELINE_BTC: "Comprar y mantener BTC",
    BASELINE_EQUAL_WEIGHT: "Cartera equiponderada del universo",
    BASELINE_SPY: "Comprar y mantener SPY",
}


@dataclass(slots=True, frozen=True)
class Baseline:
    """Un baseline evaluado sobre UNA muestra.

    `curve` y `trades` son la materia prima con la que se construyo `metrics`. Se
    exponen porque la validacion multiventana necesita ENCADENAR varios tramos de un
    mismo baseline antes de puntuarlo (un fold de CPCV puede tener dos tramos de test
    separados por meses), y encadenar Sharpes ya calculados no significa nada."""

    name: str
    label: str
    symbols: tuple[str, ...]
    score: float
    metrics: PerformanceMetrics
    curve: tuple[EquityPoint, ...] = ()
    trades: tuple[Position, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "symbols": list(self.symbols),
            "score": round(self.score, 4),
            "metrics": self.metrics.as_dict(),
        }


@dataclass(slots=True, frozen=True)
class BaselineGate:
    """
    Veredicto del gate sobre la DISTRIBUCION de muestras.

    Aprobar exige DOS cosas, y por eso el veredicto se publica descompuesto:

    - `beats_baselines`: la recompensa agregada (CVaR@25%, no una media ni un path suelto)
      es mejor que la del MEJOR baseline sobre exactamente las mismas ventanas.
    - `eligible`: la estrategia supera el suelo de actividad (`scoring.activity`), es
      decir, es rankeable.

    La segunda condicion se anadio despues de medirla, no por prudencia abstracta. Una
    configuracion que no abre posiciones deja la curva plana y puntua 0 EXACTO en cada
    ventana; su CVaR es 0, y 0 le gana a los baselines pasivos en cualquier periodo en el
    que el mercado caiga —en la evidencia publicada (`data/transfer/`) los pasivos estaban
    en -1.42 y -1.39, asi que NO HACER NADA aprobaba con margen—. Batir a los baselines por
    no jugar no es batirlos: el gate pregunta si la estrategia aporta algo sobre comprar y
    esperar, y una curva plana no ha respondido a esa pregunta, la ha esquivado. Medido
    sobre esa misma rejilla: de 16 configuraciones aprobaban 7 y aprueban 1 al exigir
    actividad; las 6 que caen abren entre 0 y 2 operaciones por ventana OOS.

    Lo que el suelo NO hace: cambiar ninguna recompensa. Una configuracion inelegible
    conserva su cifra y sigue publicandose; lo que no puede es aprobar. Ver `scoring.activity`.

    Y no se le aplica a los baselines: un `btc_hold` hace dos operaciones por ventana por
    definicion. El baseline es el LISTON, no un candidato al ranking, asi que no tiene que
    demostrar actividad -tiene que demostrar exposicion, y la tiene toda-.

    `win_rate_pct` es informativo (en cuantas muestras gana), no decide.
    """

    strategy_reward: float
    baselines: dict[str, RewardStats]
    best_name: str | None
    best_reward: float
    approved: bool
    win_rate_pct: float
    margin: float
    missing: tuple[str, ...]
    beats_baselines: bool = False
    activity: ActivityStats | None = None
    activity_floor: ActivityFloor = DEFAULT_ACTIVITY_FLOOR
    ineligible_reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        """¿Supera el suelo de actividad? Sin actividad medida es False y se declara en
        `activity_checked`: un requisito que no se ha comprobado no se da por bueno."""
        return not self.ineligible_reasons

    @property
    def activity_checked(self) -> bool:
        return self.activity is not None

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "beats_baselines": self.beats_baselines,
            "eligible": self.eligible,
            "activity_checked": self.activity_checked,
            "ineligible_reasons": list(self.ineligible_reasons),
            "activity": None if self.activity is None else self.activity.as_dict(),
            "activity_floor": self.activity_floor.as_dict(),
            "strategy_reward": round(self.strategy_reward, 4),
            "best_baseline": self.best_name,
            "best_baseline_reward": round(self.best_reward, 4),
            "margin": round(self.margin, 4),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "missing": list(self.missing),
            "baselines": {name: stats.as_dict() for name, stats in self.baselines.items()},
        }


def compute_baselines(
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    starting_equity: float,
    fee_rate: float = 0.0,
    slippage_bps: float = 0.0,
    weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
    periods_per_year: int | None = None,
) -> dict[str, Baseline]:
    """Puntua los tres baselines pasivos sobre la ventana [start, end] de una muestra.

    Los baselines que no tengan barras (p.ej. SPY en un universo solo cripto) se omiten
    del resultado; no se sustituyen ni se rellenan con ceros.

    `periods_per_year` se resuelve UNA VEZ para todo el universo de la muestra y se aplica
    a los tres baselines por igual, aunque `spy_hold` sea puramente bursatil: los tres
    recorren el mismo calendario union que la estrategia, y un Sharpe anualizado con un
    factor distinto no seria comparable con el que tiene que batir. Pasalo desde el
    llamante si la estrategia usa uno propio."""
    cost_rate = fee_rate + slippage_bps / 10_000.0
    calendar = _calendar(bars, start, end)
    if len(calendar) < 2:
        return {}

    out: dict[str, Baseline] = {}
    universe = tuple(sorted(bars))
    if periods_per_year is None:
        periods_per_year = periods_per_year_for_symbols(universe)

    for name, symbols in (
        (BASELINE_BTC, (BTC_SYMBOL,)),
        (BASELINE_SPY, (SPY_SYMBOL,)),
        (BASELINE_EQUAL_WEIGHT, universe),
    ):
        available = tuple(s for s in symbols if s in bars)
        if not available:
            continue
        baseline = _hold_portfolio(
            name, bars, available, calendar,
            starting_equity=starting_equity, cost_rate=cost_rate, weights=weights,
            periods_per_year=periods_per_year,
        )
        if baseline is not None:
            out[name] = baseline

    return out


def gate(
    strategy_scores: Sequence[float],
    baseline_scores: dict[str, Sequence[float]],
    *,
    alpha: float = DEFAULT_CVAR_ALPHA,
    missing: Sequence[str] = (),
    trades: Sequence[int] | None = None,
    activity_floor: ActivityFloor = DEFAULT_ACTIVITY_FLOOR,
) -> BaselineGate:
    """
    Decide si la estrategia APRUEBA: su recompensa agregada debe superar la del mejor
    baseline sobre exactamente las mismas muestras Y superar el suelo de actividad.

    `trades` son las operaciones de cada una de esas mismas ventanas (mismo orden que
    `strategy_scores`). Todos los llamantes del repositorio lo pasan; si no se pasa, la
    elegibilidad NO se da por buena: el gate queda en `approved=False` con
    `activity_checked=False`, para que un requisito sin comprobar no pueda colarse como
    comprobado.

    Sin ningun baseline disponible tampoco se puede aprobar nada: el veredicto es `False` y
    `best_name` es None. Un gate que no puede evaluarse no es un gate superado.
    """
    strategy = aggregate_reward(strategy_scores, alpha=alpha, trades=trades)
    stats = {
        name: aggregate_reward(scores, alpha=alpha)
        for name, scores in baseline_scores.items()
        if len(scores) > 0
    }
    reasons = activity_floor.reasons(strategy.activity)

    if not stats:
        return BaselineGate(
            strategy_reward=strategy.reward,
            baselines={},
            best_name=None,
            best_reward=float("-inf"),
            approved=False,
            win_rate_pct=0.0,
            margin=0.0,
            missing=tuple(missing),
            beats_baselines=False,
            activity=strategy.activity,
            activity_floor=activity_floor,
            ineligible_reasons=reasons,
        )

    # Desempate por nombre para que el mejor baseline sea determinista.
    best_name = min(stats, key=lambda n: (-stats[n].reward, n))
    best_reward = stats[best_name].reward

    per_sample_best = _elementwise_best(baseline_scores, len(strategy_scores))
    wins = sum(1 for s, b in zip(strategy_scores, per_sample_best) if s > b)
    win_rate = (wins / len(per_sample_best) * 100.0) if per_sample_best else 0.0

    beats = strategy.reward > best_reward
    return BaselineGate(
        strategy_reward=strategy.reward,
        baselines=stats,
        best_name=best_name,
        best_reward=best_reward,
        approved=beats and not reasons,
        win_rate_pct=win_rate,
        margin=strategy.reward - best_reward,
        missing=tuple(missing),
        beats_baselines=beats,
        activity=strategy.activity,
        activity_floor=activity_floor,
        ineligible_reasons=reasons,
    )


# ------------------------------------------------------------------ internals --------


def _calendar(
    bars: dict[str, pd.DataFrame], start: datetime, end: datetime
) -> list[pd.Timestamp]:
    """Union ordenada de dias con barra en [start, end]. Es el mismo calendario que
    recorre el backtest, para que la ventana del baseline sea la de la estrategia."""
    start_ts = _to_utc(start).normalize()
    end_ts = _to_utc(end).normalize()

    days: set[pd.Timestamp] = set()
    for df in bars.values():
        if df is None or df.empty:
            continue
        normalized = df.index.normalize()
        mask = (normalized >= start_ts) & (normalized <= end_ts)
        days.update(normalized[mask])
    return sorted(days)


def _to_utc(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _closes_on(df: pd.DataFrame, calendar: list[pd.Timestamp]) -> pd.Series | None:
    """Cierres alineados al calendario. Un dia sin barra hereda el ultimo cierre (el
    activo no se revaloriza, que es exactamente lo que pasa si no cotiza)."""
    closes = bar_schema.series(df, bar_schema.CLOSE)
    closes.index = df.index.normalize()
    closes = closes[~closes.index.duplicated(keep="last")]
    aligned = closes.reindex(calendar).ffill()
    if aligned.isna().any() or (aligned <= 0).any():
        return None
    return aligned


def _hold_portfolio(
    name: str,
    bars: dict[str, pd.DataFrame],
    symbols: tuple[str, ...],
    calendar: list[pd.Timestamp],
    *,
    starting_equity: float,
    cost_rate: float,
    weights: HeadlineWeights,
    periods_per_year: int,
) -> Baseline | None:
    """Compra equiponderada el primer dia y liquida el ultimo, pagando coste en ambas
    patas. Es el mismo trato que recibe la estrategia (que tambien liquida al cierre de
    la ventana), de modo que la comparacion no regala friccion a nadie."""
    tracked: dict[str, pd.Series] = {}
    for symbol in symbols:
        closes = _closes_on(bars[symbol], calendar)
        if closes is not None:
            tracked[symbol] = closes
    if not tracked:
        return None

    allocation = starting_equity / len(tracked)
    positions: list[Position] = []
    units: dict[str, float] = {}

    for symbol, closes in tracked.items():
        entry_price = float(closes.iloc[0])
        entry_fees = allocation * cost_rate
        units[symbol] = (allocation - entry_fees) / entry_price

    curve: list[EquityPoint] = []
    for i, day in enumerate(calendar):
        equity = sum(units[s] * float(tracked[s].iloc[i]) for s in tracked)
        if i == len(calendar) - 1:
            equity *= 1.0 - cost_rate  # liquidacion final: se paga la segunda pata
        curve.append(EquityPoint(day=day.to_pydatetime(), equity=equity))

    opened_at = calendar[0].to_pydatetime()
    closed_at = calendar[-1].to_pydatetime()
    for symbol, closes in tracked.items():
        entry_price = float(closes.iloc[0])
        exit_price = float(closes.iloc[-1])
        size = units[symbol]
        entry_fees = allocation * cost_rate
        exit_fees = size * exit_price * cost_rate
        positions.append(
            Position(
                symbol=symbol,
                side=Side.BUY,
                size=size,
                entry_price=entry_price,
                opened_at=opened_at,
                strategy_id=name,
                status=PositionStatus.CLOSED,
                closed_at=closed_at,
                exit_price=exit_price,
                realized_pnl=size * (exit_price - entry_price) - entry_fees - exit_fees,
                close_reason="baseline_window_end",
                entry_fees_usd=entry_fees,
                exit_fees_usd=exit_fees,
            )
        )

    metrics = compute_metrics(curve, positions, periods_per_year=periods_per_year)
    return Baseline(
        name=name,
        label=BASELINE_LABELS[name],
        symbols=tuple(sorted(tracked)),
        score=headline_score(metrics, weights),
        metrics=metrics,
        curve=tuple(curve),
        trades=tuple(positions),
    )


def _elementwise_best(
    baseline_scores: dict[str, Sequence[float]], n: int
) -> list[float]:
    """Mejor baseline MUESTRA A MUESTRA (no el mejor en agregado): asi el win-rate mide
    'gano al mejor rival de ese mundo', que es la comparacion exigente."""
    usable = [list(v) for v in baseline_scores.values() if len(v) == n]
    if not usable or n == 0:
        return []
    return [max(col) for col in zip(*usable)]
