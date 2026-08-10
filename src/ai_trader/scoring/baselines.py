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

    `approved` compara recompensas agregadas (CVaR@25%), no medias ni un path suelto:
    la estrategia aprueba si su cola mala es mejor que la cola mala del MEJOR baseline.
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

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
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
) -> BaselineGate:
    """
    Decide si la estrategia APRUEBA: su recompensa agregada debe superar la del mejor
    baseline sobre exactamente las mismas muestras.

    Sin ningun baseline disponible no se puede aprobar nada: el veredicto es `False` y
    `best_name` es None. Un gate que no puede evaluarse no es un gate superado.
    """
    strategy = aggregate_reward(strategy_scores, alpha=alpha)
    stats = {
        name: aggregate_reward(scores, alpha=alpha)
        for name, scores in baseline_scores.items()
        if len(scores) > 0
    }

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
        )

    # Desempate por nombre para que el mejor baseline sea determinista.
    best_name = min(stats, key=lambda n: (-stats[n].reward, n))
    best_reward = stats[best_name].reward

    per_sample_best = _elementwise_best(baseline_scores, len(strategy_scores))
    wins = sum(1 for s, b in zip(strategy_scores, per_sample_best) if s > b)
    win_rate = (wins / len(per_sample_best) * 100.0) if per_sample_best else 0.0

    return BaselineGate(
        strategy_reward=strategy.reward,
        baselines=stats,
        best_name=best_name,
        best_reward=best_reward,
        approved=strategy.reward > best_reward,
        win_rate_pct=win_rate,
        margin=strategy.reward - best_reward,
        missing=tuple(missing),
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
