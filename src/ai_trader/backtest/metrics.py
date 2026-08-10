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


@dataclass(frozen=True, slots=True)
class HeadlineWeights:
    """
    Pesos del headline score. Estan en "unidades de Sharpe": el score parte del Sharpe
    OOS y le RESTA el coste de rotar y el de sufrir caidas, de modo que cada peso dice
    cuanto Sharpe vale la pena pagar por una unidad de esa magnitud.

    CALIBRACION MEDIDA (no razonada): 480 backtests reales sobre la libreria ai_v2
    (16 configuraciones de un hipercubo latino x 30 escenarios), barriendo
    lambda ∈ {0, .25, .5, 1, 2, 4} x kappa ∈ {0, .5, 1, 2, 4}. Evidencia completa y
    reproducible en `data/calibration/` (ver ai_trader.scoring.weight_study). Tres
    resultados, y los tres empujan hacia pesos PEQUENOS:

    1. Los pesos NO cambian la decision. La misma configuracion gana en los 30 puntos de
       la rejilla, tanto en el conjunto completo como en el de configuraciones activas.
       En el rango medido, lambda y kappa no arbitran nada: quien decide es el Sharpe.
    2. Penalizar NO estabiliza el ranking; lo degrada un poco. La correlacion de rangos
       entre el ranking in-sample y el out-of-sample es maxima sin penalizar
       (0.130 +- 0.064) y baja de forma monotona con ambos pesos. Contra los antiguos
       (0.5, 1.0) la perdida pareada era -0.022 +- 0.008, un 17% del nivel de la senal.
       El gap train-validation normalizado se mueve en la misma direccion (1.94 -> 2.06).
    3. La rotacion YA se paga dentro del Sharpe. fee_rate + slippage = 0.15% de cada
       notional rotado equivale a lambda ~6.3 puntos de Sharpe por unidad de turnover
       (IQR 4.6-9.7); la friccion se come 0.24 puntos de Sharpe al turnover mediano.

    De ahi los valores fijados:
    - `lambda_turnover` = 0.25: el menor valor NO NULO de la rejilla. No puede ser 0
      porque el headline tiene una propiedad comprometida —misma curva de equity y mas
      rotacion tiene que puntuar peor— y porque rotar cuesta dinero de verdad. Su coste
      medido es -0.004 +- 0.002 (indistinguible de cero en las configuraciones activas)
      y supone un 4% del coste implicito ya cobrado: es un margen de seguridad explicito
      sobre el modelo de costes, no una segunda factura por la misma rotacion.
    - `kappa_maxdd` = 0.0: ninguna propiedad del diseno lo exige, es el termino que mas
      estabilidad cuesta por unidad, y el maxDD es el estadistico mas ruidoso de una
      curva de equity —la misma objecion que retiro al Calmar—. El mecanismo sigue
      disponible: quien quiera aversion explicita al drawdown pasa `kappa_maxdd`, pero
      ya no se cobra por defecto sin que nadie lo haya pedido.

    Limite declarado: se midio con un unico corte temporal 70/30, un camino por escenario
    y 16 configuraciones. Basta para descartar que los pesos grandes ayuden; no para
    afinar decimales. Mover estos valores exige re-correr el estudio (hay un test que los
    ata a la evidencia publicada).
    """

    lambda_turnover: float = 0.25
    kappa_maxdd: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "lambda_turnover": self.lambda_turnover,
            "kappa_maxdd": self.kappa_maxdd,
        }


DEFAULT_HEADLINE_WEIGHTS = HeadlineWeights()


@dataclass(slots=True)
class PerformanceMetrics:
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float  # reportado, NO cabecera: CAGR / max drawdown (ver headline_score)
    volatility_pct: float
    turnover: float  # notional rotado por dia, en unidades del equity inicial
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
            "turnover": round(self.turnover, 4),
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


def headline_score(
    metrics: PerformanceMetrics,
    weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
) -> float:
    """
    Puntuacion de cabecera de una ventana: `Sharpe - lambda*turnover - kappa*maxDD`.

    Sustituye al Calmar, que era toxico como objetivo de optimizacion:
    - el maxDD es un EXTREMO (el estadistico mas ruidoso de una curva de equity) y en
      el denominador dispara la varianza del estimador;
    - premiaba la inactividad: casi no operar -> maxDD minusculo -> Calmar altisimo;
    - degeneraba a 0 cuando no habia drawdown.

    Aqui el maxDD entra como penalizacion SUAVE y aditiva (en fraccion, no en %), el
    Sharpe no se puede inflar dejando de operar (una curva plana puntua 0) y la rotacion
    tiene un precio explicito, asi que el optimo degenerado desaparece por construccion.

    Con los pesos por defecto CALIBRADOS (ver HeadlineWeights) el termino de maxDD queda
    en 0: la medicion no encontro ningun beneficio en cobrarlo y si un coste en
    estabilidad del ranking. El sumando sigue en la formula porque el peso es un
    parametro, no una constante: pasar `kappa_maxdd` lo reactiva.
    """
    return (
        metrics.sharpe
        - weights.lambda_turnover * metrics.turnover
        - weights.kappa_maxdd * (metrics.max_drawdown_pct / 100.0)
    )


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


def window_days(curve: Sequence[EquityPoint]) -> int:
    """Dias calendario cubiertos por la curva. Cae al numero de puntos si las marcas de
    tiempo no avanzan (curvas de test construidas sin fechas reales)."""
    if len(curve) < 2:
        return 0
    spanned = (curve[-1].day - curve[0].day).days
    return spanned if spanned > 0 else len(curve) - 1


def turnover_ratio(
    curve: Sequence[EquityPoint],
    closed_positions: Sequence[Position],
) -> float:
    """
    Rotacion: notional negociado por dia, en unidades del equity INICIAL.

    Cuenta las dos patas de cada operacion (entrada y salida) a su precio real, asi que
    0.2 significa "cada dia rota el 20% de la cartera". Es la magnitud que penaliza el
    headline score: mide churn de verdad (tamano x frecuencia), no solo nº de trades.
    """
    days = window_days(curve)
    if days <= 0 or not curve:
        return 0.0

    equity = curve[0].equity
    if equity <= 0:
        return 0.0

    traded = 0.0
    for position in closed_positions:
        exit_price = position.exit_price if position.exit_price is not None else position.entry_price
        traded += position.size * (position.entry_price + exit_price)

    return traded / (equity * days)


def skewness(returns: Sequence[float]) -> float:
    """Asimetria muestral (momento poblacional). 0 en una normal."""
    return _standardized_moment(returns, order=3, fallback=0.0)


def kurtosis(returns: Sequence[float]) -> float:
    """Curtosis muestral NO en exceso (momento poblacional). 3 en una normal."""
    return _standardized_moment(returns, order=4, fallback=3.0)


def _standardized_moment(returns: Sequence[float], *, order: int, fallback: float) -> float:
    if len(returns) < 3:
        return fallback
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    if variance <= 0:
        return fallback
    std = math.sqrt(variance)
    return sum(((r - mean) / std) ** order for r in returns) / len(returns)


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

    # Calmar: retorno anualizado por unidad de caida maxima. Se REPORTA por costumbre,
    # pero ya no es la cabecera de scoring (ver headline_score): sin drawdown degenera a
    # 0 y con drawdown minusculo se dispara, premiando la inactividad.
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
        turnover=turnover_ratio(curve, closed_positions),
        num_trades=len(closed_positions),
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        avg_holding_days=avg_holding,
        total_fees_usd=sum(p.total_fees_usd for p in closed_positions),
    )
