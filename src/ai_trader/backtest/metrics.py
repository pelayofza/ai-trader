from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ai_trader.shared.schemas import Position

TRADING_DAYS_PER_YEAR = 365  # cripto opera 24/7; para renta variable se ajustaria a 252


@dataclass(slots=True)
class EquityPoint:
    day: datetime
    equity: float


@dataclass(slots=True)
class PerformanceMetrics:
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float  # metrica cabecera: CAGR / max drawdown
    volatility_pct: float
    num_trades: int
    win_rate_pct: float
    profit_factor: float | None
    avg_win_usd: float
    avg_loss_usd: float
    avg_holding_days: float
    total_fees_usd: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(self.ending_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "calmar": round(self.calmar, 3),
            "volatility_pct": round(self.volatility_pct, 2),
            "num_trades": self.num_trades,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": (
                round(self.profit_factor, 3) if self.profit_factor is not None else None
            ),
            "avg_win_usd": round(self.avg_win_usd, 2),
            "avg_loss_usd": round(self.avg_loss_usd, 2),
            "avg_holding_days": round(self.avg_holding_days, 2),
            "total_fees_usd": round(self.total_fees_usd, 2),
        }


def daily_returns(curve: Sequence[EquityPoint]) -> list[float]:
    returns: list[float] = []
    for prev, curr in zip(curve, curve[1:]):
        if prev.equity > 0:
            returns.append(curr.equity / prev.equity - 1.0)
        else:
            returns.append(0.0)
    return returns


def max_drawdown_pct(curve: Sequence[EquityPoint]) -> float:
    """Maxima caida desde un pico, en %. Positivo (p.ej. 20.0 = -20%)."""
    peak = -math.inf
    worst = 0.0
    for point in curve:
        peak = max(peak, point.equity)
        if peak > 0:
            drawdown = (peak - point.equity) / peak
            worst = max(worst, drawdown)
    return worst * 100.0


def _annualized(returns: Sequence[float], downside_only: bool) -> float:
    if len(returns) < 2:
        return 0.0

    mean = sum(returns) / len(returns)

    if downside_only:
        negatives = [r for r in returns if r < 0]
        if not negatives:
            return 0.0
        variance = sum(r * r for r in negatives) / len(returns)
    else:
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)

    std = math.sqrt(variance)
    if std == 0:
        return 0.0

    # Ratio diario anualizado por sqrt(dias); se asume tasa libre de riesgo 0.
    return (mean / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: Sequence[float]) -> float:
    return _annualized(returns, downside_only=False)


def sortino_ratio(returns: Sequence[float]) -> float:
    return _annualized(returns, downside_only=True)


def cagr_pct(curve: Sequence[EquityPoint]) -> float:
    if len(curve) < 2:
        return 0.0

    start_equity = curve[0].equity
    end_equity = curve[-1].equity
    if start_equity <= 0 or end_equity <= 0:
        return 0.0

    days = (curve[-1].day - curve[0].day).days
    if days <= 0:
        return 0.0

    years = days / TRADING_DAYS_PER_YEAR
    return ((end_equity / start_equity) ** (1.0 / years) - 1.0) * 100.0


def volatility_pct(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0


def compute_metrics(
    curve: Sequence[EquityPoint],
    closed_positions: Sequence[Position],
) -> PerformanceMetrics:
    if not curve:
        raise ValueError("equity curve cannot be empty")

    start_equity = curve[0].equity
    end_equity = curve[-1].equity
    returns = daily_returns(curve)

    total_return = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0 else 0.0
    cagr = cagr_pct(curve)
    max_dd = max_drawdown_pct(curve)

    # Calmar: retorno anualizado por unidad de caida maxima. Es la cabecera de scoring:
    # premia crecimiento y castiga el drawdown. Sin drawdown, se degrada a 0 (indefinido).
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    winners = [p for p in closed_positions if (p.realized_pnl or 0.0) > 0]
    losers = [p for p in closed_positions if (p.realized_pnl or 0.0) < 0]

    gross_profit = sum(p.realized_pnl or 0.0 for p in winners)
    gross_loss = abs(sum(p.realized_pnl or 0.0 for p in losers))

    win_rate = (len(winners) / len(closed_positions) * 100.0) if closed_positions else 0.0
    avg_win = (gross_profit / len(winners)) if winners else 0.0
    avg_loss = (-gross_loss / len(losers)) if losers else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    holding = [
        (p.closed_at - p.opened_at).days
        for p in closed_positions
        if p.closed_at is not None
    ]
    avg_holding = sum(holding) / len(holding) if holding else 0.0

    return PerformanceMetrics(
        starting_equity=start_equity,
        ending_equity=end_equity,
        total_return_pct=total_return,
        cagr_pct=cagr,
        max_drawdown_pct=max_dd,
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        calmar=calmar,
        volatility_pct=volatility_pct(returns),
        num_trades=len(closed_positions),
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        avg_holding_days=avg_holding,
        total_fees_usd=sum(p.total_fees_usd for p in closed_positions),
    )
