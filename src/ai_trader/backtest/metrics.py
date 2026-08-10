from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ai_trader.shared.instruments import AssetClass, detect_asset_class
from ai_trader.shared.schemas import Position

# --- factores de anualizacion ------------------------------------------------
#
# Hay DOS unidades distintas y confundirlas es el error clasico:
#
# 1. TIEMPO DE CALENDARIO. El CAGR mide cuanto renta un ano NATURAL, y el ano natural
#    tiene 365 dias para todo el mundo: una accion que sube un 10% entre enero y julio
#    ha subido un 10% en medio ano de calendario, aunque solo haya cotizado 126 sesiones.
#    Por eso `cagr_pct` divide SIEMPRE por CALENDAR_DAYS_PER_YEAR, sea cual sea el activo.
#
# 2. NUMERO DE OBSERVACIONES. Sharpe, Sortino y volatilidad se anualizan por sqrt(N),
#    donde N es cuantos retornos entran en un ano. Y ahi si depende del mercado: cripto
#    cotiza 24/7 (una barra por dia natural, N=365) y la renta variable solo en sesion
#    (N=252). Anualizar una curva bursatil por sqrt(365) infla el Sharpe un 20%
#    (sqrt(365/252) = 1.204) y la volatilidad otro tanto. Ese era el bug: una constante
#    global de 365 aplicada tambien a las acciones.
CALENDAR_DAYS_PER_YEAR = 365  # dias naturales por ano: la unidad del CAGR
CRYPTO_PERIODS_PER_YEAR = 365  # cripto cotiza 24/7 -> una observacion por dia natural
STOCK_PERIODS_PER_YEAR = 252  # renta variable -> solo sesiones bursatiles

# Default historico y unidad de las magnitudes 24/7. Se conserva con este nombre porque
# lo consumen los modulos de observacion (features, regime) y de scoring (overfit,
# weight_calibration), todos calibrados sobre librerias sinteticas 24/7. Para elegir el
# factor de una cartera concreta usa `periods_per_year_for` / `periods_per_year_for_symbols`.
TRADING_DAYS_PER_YEAR = CRYPTO_PERIODS_PER_YEAR

_PERIODS_BY_ASSET_CLASS: dict[AssetClass, int] = {
    AssetClass.CRYPTO: CRYPTO_PERIODS_PER_YEAR,
    AssetClass.STOCK: STOCK_PERIODS_PER_YEAR,
    # Los mercados de prediccion tampoco cierran: mismo calendario que cripto.
    AssetClass.PREDICTION: CRYPTO_PERIODS_PER_YEAR,
}


def periods_per_year_for(asset_classes: Iterable[AssetClass]) -> int:
    """
    Observaciones por ano de una curva de equity construida sobre esas clases de activo.

    Es una propiedad del CALENDARIO de la curva, no de cada activo por separado: el
    backtest (y los baselines) recorren la UNION de dias con barra, asi que basta un
    activo 24/7 para que haya un punto cada dia natural y el factor sea 365 aunque la
    cartera sea mayoritariamente bursatil. Solo una cartera exclusivamente de renta
    variable vive en el calendario de sesiones, y esa es la que usa 252.

    Sin clases de activo (curva sintetica de test, universo vacio) se devuelve el default
    24/7: es el que reproduce el comportamiento historico y no cambia ninguna cifra ya
    publicada.
    """
    classes = {c for c in asset_classes}
    if not classes:
        return TRADING_DAYS_PER_YEAR
    return max(_PERIODS_BY_ASSET_CLASS.get(c, CRYPTO_PERIODS_PER_YEAR) for c in classes)


def periods_per_year_for_symbols(symbols: Iterable[str]) -> int:
    """`periods_per_year_for` a partir de los simbolos, deduciendo la clase por nombre."""
    return periods_per_year_for(detect_asset_class(s) for s in symbols)


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
    # Observaciones/ano con las que se anualizaron sharpe, sortino y volatility_pct.
    # Se reporta porque sin el esas tres cifras no son interpretables ni comparables
    # entre ventanas de universos distintos (365 cripto vs 252 renta variable).
    periods_per_year: int = TRADING_DAYS_PER_YEAR

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
            "periods_per_year": self.periods_per_year,
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


@dataclass(slots=True, frozen=True)
class ChainedCurve:
    """Curva compuesta a partir de varios tramos DISJUNTOS, mas los dias realmente
    operados (`active_days`), que es lo que hay que pasarle a `compute_metrics`."""

    points: list[EquityPoint]
    active_days: int


def chain_equity_curves(curves: Sequence[Sequence[EquityPoint]]) -> ChainedCurve:
    """
    Encadena curvas de bloques disjuntos COMPONIENDO sus retornos, no pegandolas.

    Es la operacion que convierte los varios tramos de test de un fold (CPCV puede
    darte dos tramos separados por meses) en una sola serie puntuable. Cada tramo se
    corrio por separado partiendo del mismo capital inicial, asi que lo unico
    comparable entre tramos son sus RETORNOS: se multiplican en orden y el nivel se
    arrastra. El salto entre el final de un bloque y el principio del siguiente no
    genera retorno —no hubo posicion viva ahi—, de modo que el hueco no aporta ni
    ganancia ni perdida. Los dias del hueco tampoco cuentan como tiempo operado.
    """
    usable = [list(c) for c in curves if c]
    if not usable:
        raise ValueError("no equity curves to chain")

    level = usable[0][0].equity
    points = [EquityPoint(day=usable[0][0].day, equity=level)]
    active = 0
    for curve in usable:
        active += window_days(curve)
        for prev, curr in zip(curve, curve[1:]):
            if prev.equity > 0:
                level *= curr.equity / prev.equity
            points.append(EquityPoint(day=curr.day, equity=level))
    return ChainedCurve(points=points, active_days=active)


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


def window_days(curve: Sequence[EquityPoint], active_days: int | None = None) -> int:
    """Dias calendario cubiertos por la curva. Cae al numero de puntos si las marcas de
    tiempo no avanzan (curvas de test construidas sin fechas reales).

    `active_days` lo sobreescribe. Hace falta para las curvas ENCADENADAS de la
    validacion multiventana: sus puntos abarcan bloques disjuntos, asi que el hueco
    entre bloques esta dentro del primer y ultimo dia pero NO es tiempo en el que la
    estrategia estuviera viva. Contarlo diluiria la rotacion y falsearia el CAGR."""
    if active_days is not None:
        return max(0, active_days)
    if len(curve) < 2:
        return 0
    spanned = (curve[-1].day - curve[0].day).days
    return spanned if spanned > 0 else len(curve) - 1


def turnover_ratio(
    curve: Sequence[EquityPoint],
    closed_positions: Sequence[Position],
    active_days: int | None = None,
) -> float:
    """
    Rotacion: notional negociado por dia, en unidades del equity INICIAL.

    Cuenta las dos patas de cada operacion (entrada y salida) a su precio real, asi que
    0.2 significa "cada dia rota el 20% de la cartera". Es la magnitud que penaliza el
    headline score: mide churn de verdad (tamano x frecuencia), no solo nº de trades.
    """
    days = window_days(curve, active_days)
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


def _annualized(
    returns: Sequence[float],
    downside_only: bool,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
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

    # Ratio por observacion anualizado por sqrt(observaciones/ano); tasa libre de riesgo 0.
    return (mean / std) * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: Sequence[float], periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    return _annualized(returns, downside_only=False, periods_per_year=periods_per_year)


def sortino_ratio(
    returns: Sequence[float], periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    return _annualized(returns, downside_only=True, periods_per_year=periods_per_year)


def cagr_pct(curve: Sequence[EquityPoint], active_days: int | None = None) -> float:
    """Retorno anualizado en TIEMPO DE CALENDARIO. No lleva `periods_per_year`: el ano
    natural tiene 365 dias para cripto y para renta variable por igual (ver la nota de
    los factores de anualizacion arriba). `active_days` sustituye al span de la curva
    para las curvas encadenadas (ver `window_days`)."""
    if len(curve) < 2:
        return 0.0

    start_equity = curve[0].equity
    end_equity = curve[-1].equity
    if start_equity <= 0 or end_equity <= 0:
        return 0.0

    days = active_days if active_days is not None else (curve[-1].day - curve[0].day).days
    if days <= 0:
        return 0.0

    years = days / CALENDAR_DAYS_PER_YEAR
    return ((end_equity / start_equity) ** (1.0 / years) - 1.0) * 100.0


def volatility_pct(
    returns: Sequence[float], periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year) * 100.0


def compute_metrics(
    curve: Sequence[EquityPoint],
    closed_positions: Sequence[Position],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    active_days: int | None = None,
) -> PerformanceMetrics:
    """
    Metricas de una ventana. `periods_per_year` es el factor de anualizacion de las
    magnitudes por observacion (Sharpe, Sortino, volatilidad): 365 para carteras 24/7 y
    252 para carteras exclusivamente de renta variable. Resuelvelo con
    `periods_per_year_for_symbols(universo)` y pasalo IGUAL a la estrategia y a sus
    baselines: comparar dos Sharpe anualizados con factores distintos no significa nada.

    `active_days` son los dias en los que la estrategia estuvo REALMENTE viva. Solo hace
    falta para curvas encadenadas de bloques disjuntos (validacion multiventana), donde
    el span de la curva incluye huecos que no se operaron: ahi corrige la rotacion y el
    CAGR. En una ventana contigua se deja en None y no cambia nada.
    """
    if not curve:
        raise ValueError("equity curve cannot be empty")

    start_equity = curve[0].equity
    end_equity = curve[-1].equity
    returns = daily_returns(curve)

    total_return = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0 else 0.0
    cagr = cagr_pct(curve, active_days)
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
        sharpe=sharpe_ratio(returns, periods_per_year),
        sortino=sortino_ratio(returns, periods_per_year),
        calmar=calmar,
        volatility_pct=volatility_pct(returns, periods_per_year),
        turnover=turnover_ratio(curve, closed_positions, active_days),
        num_trades=len(closed_positions),
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        avg_holding_days=avg_holding,
        total_fees_usd=sum(p.total_fees_usd for p in closed_positions),
        periods_per_year=periods_per_year,
    )
