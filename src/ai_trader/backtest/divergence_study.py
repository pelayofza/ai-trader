r"""
DIVERGENCIA live-vs-backtest: cuanto se aparta lo ejecutado de lo que el motor predecia.

Es la medicion que justifica el capitulo 3 entero. Todo lo demas de este repo se puede
acelerar con computo: generar mas mundos sinteticos, correr mas folds, barrer mas pesos.
Esto no. Esto consume CALENDARIO, porque la materia prima es el diario de ciclos que el
paper trading en vivo escribe una linea por ciclo (`app/journal.py`).

    .venv\Scripts\python.exe -m ai_trader.backtest.divergence_study
    .venv\Scripts\python.exe -m ai_trader.backtest.divergence_study --offline
    .venv\Scripts\python.exe -m ai_trader.backtest.divergence_study --verify-determinism

Salida: data/live/divergence.json

COMO SE PAREA, Y POR QUE NO ES "UN SHARPE CONTRA OTRO"
-----------------------------------------------------
Comparar el agregado no dice donde esta la diferencia: dos curvas distintas pueden dar
el mismo Sharpe y un Sharpe distinto puede venir de un solo dia. El pareo se hace por
DECISION, con la clave `(dia UTC, simbolo, estrategia)`.

Esa clave y no otra, porque un "ciclo" no significa lo mismo en los dos mundos: en vivo
el runner despierta cada `interval_seconds` (96 veces al dia con el defecto de 900 s) y
en backtest una vez por dia de mercado. Lo que SI es identico es la informacion sobre la
que se decide —las barras diarias ya cerradas—, asi que un dia de calendario es la unidad
mas fina en la que las dos ejecuciones son comparables. Por eso los recuentos se publican
DOS veces: en bruto (que es lo que paso) y deduplicados por dia (que es lo que se puede
comparar). Contar 96 senales en vivo contra 1 en backtest no seria una divergencia, seria
una diferencia de cadencia.

La re-simulacion no reimplementa nada: se le engancha un `MemoryJournal` al mismo
`BacktestEngine`, de modo que la ventana re-simulada emite EXACTAMENTE el mismo esquema de
linea que el vivo y el pareo es una funcion sobre dos listas de lo mismo.

LAS TRES PIERNAS, Y QUE SUMEN
-----------------------------
Para cada orden de ENTRADA archivada se reconstruye lo que el motor habria hecho: el mismo
`SlippageModel`, con la liquidez que la re-simulacion vio (`BarLiquidityProvider` sobre las
barras diarias cerradas) y la referencia que el backtest usa (el OPEN del dia,
`IntrabarMarketModel`). Con `s = +1` en compra y `-1` en venta, y midiendo en pb sobre la
referencia del modelo, la diferencia se parte en tres sumandos EXACTOS:

    total = referencia + coste + cruzado

    referencia = s * 1e4 * (ref_live - ref_modelo) / ref_modelo
    coste      = bps_live - bps_modelo
    cruzado    = bps_live * (ref_live - ref_modelo) / ref_modelo

`referencia` es lo que vale decidir con un precio distinto —en vivo, el ultimo cierre
diario, que en el instante del fill puede tener hasta 24 h—; `coste` es lo que el modelo
de microestructura cobro de mas o de menos; `cruzado` es el termino de segundo orden, que
se publica en vez de repartirse para que la descomposicion cierre y se pueda comprobar.
Que sumen no es cosmetico: es lo que impide que una pierna absorba en silencio el error de
otra, y el informe publica el residuo maximo para que se pueda auditar.

El residuo no sale exactamente cero, y conviene saber por que: los precios se archivan
REDONDEADOS a 8 decimales (`execution/paper.py::fill_price`), asi que la igualdad
algebraica se cumple sobre numeros que ya perdieron los ultimos digitos. Con precios de
tres cifras eso vale del orden de 1e-6 pb —siete ordenes de magnitud por debajo de
cualquier divergencia con significado—, y por eso la tolerancia declarada es
`DECOMPOSITION_TOLERANCE_BPS` y no cero. Si el residuo la superara, el problema no seria
de redondeo: seria que las piernas no miden lo que dicen.

La LATENCIA se publica aparte y en sus propias unidades, porque es una pregunta de tiempo
antes que de precio: `decided_at -> executed_at` (el hueco real de cada orden, que el
diario ahora sella) y la ANTIGUEDAD del precio de referencia en el instante del fill,
tasada contra barras 1H reales y comparada con `session_study.reference_cost_bps`.

Y una prediccion, escrita antes de tener la cifra para que no se pueda ajustar despues:
**se espera que esta regla FALLE**. El estudio de sesiones ya midio la escalera —una hora
de retraso vale 3,86x el coste de referencia, dos horas 5,11x, ocho horas 8,56x—, y en
vivo el sistema decide con el ultimo cierre DIARIO, que a la hora del fill lleva puesto
lo que lleve el reloj de ese ciclo. El umbral esta en 1,0x igualmente: una regla calibrada
para aprobar no mide nada, y lo que interesa es EL FACTOR con el que falla, que es el que
dira si hay que modelar la latencia o cambiar cuando se decide.

EL TECHO DE LO QUE ESTO PUEDE MEDIR HOY, DECLARADO
--------------------------------------------------
La ejecucion en vivo es DE PAPEL: el `filled_price` del diario lo produjo el mismo
`PaperExecutionEngine` que usa el backtest. Por tanto la pierna de coste NO mide
"modelo contra mercado" —eso exige un broker real—, mide **contexto en vivo contra
contexto re-simulado**: la instantanea de liquidez que el proveedor vio a las 19:08 con
barras servidas por la cache frente a la que ve la re-simulacion sobre barras ya cerradas.
Es una cifra real y accionable (si diverge, el backtest esta puntuando con una liquidez
que en vivo no existe), pero no es la que sera cuando haya broker. El codigo no cambia
cuando lo haya: cambia lo que significa, y por eso se dice aqui y en el informe.

La que NO tiene ese techo, y por eso es la cifra con dientes desde el primer mes, es la
pierna de referencia/latencia: en vivo se decide y se llena con un cierre diario que ya
es viejo, y contra un mercado real eso se paga entero.

SIN POTENCIA NO SE PUBLICA CIFRA
--------------------------------
Si el diario no cubre `MIN_JOURNAL_DAYS` de calendario, el estudio NO re-simula ni publica
divergencia: escribe el informe con `status = "sin_potencia"`, cuanto falta y por que, y
sale. Publicar una divergencia medida sobre cuatro dias seria peor que no publicarla,
porque tendria el mismo aspecto que la buena.

REGLAS DE DECISION, declaradas antes de mirar el resultado:
  cobertura de decisiones < DECISION_COVERAGE_MIN -> los dos mundos NO ven lo mismo, y
      entonces el problema son los DATOS y no el coste: cualquier cifra de PnL comparado
      esta explicando la diferencia equivocada.
  |coste realizado - modelado| mediano > COST_DIVERGENCE_MAX_BPS -> el modelo de
      microestructura del backtest no reproduce lo que se cobra en vivo.
  desplazamiento por latencia mediano > LATENCY_MAX_COST_SHARE x coste de referencia ->
      la latencia vale mas que todo el coste que el motor modela, y hay que modelarla.

Determinismo: no hay muestreo ni semillas. El diario es un fichero, la ventana sale de el,
el backtest es determinista dado (config, barras, ventana) y toda la aritmetica es una
reduccion. `--verify-determinism` recalcula el informe entero y exige igualdad campo a
campo salvo el sello de generacion.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.app.journal import (
    DEFAULT_JOURNAL_PATH,
    CycleJournal,
    MemoryJournal,
    journal_summary,
)
from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY, BacktestEngine
from ai_trader.backtest.session_study import reference_cost_bps
from ai_trader.backtest.validation import Block, Fold
from ai_trader.config import AppConfig, load_config
from ai_trader.data.backtest_source import HistoricalDataSource
from ai_trader.data.cache import cache_symbol, load_bars
from ai_trader.data.intraday import get_hourly_bars
from ai_trader.execution.microstructure import BarLiquidityProvider, SlippageModel
from ai_trader.execution.paper import fill_price
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.instruments import PREDICTION_PREFIX, AssetClass, detect_asset_class
from ai_trader.shared.reports import load_report, write_report

logger = logging.getLogger("divergence_study")

OUT_DIR = Path("data") / "live"
DIVERGENCE_REPORT = OUT_DIR / "divergence.json"

DEFAULT_CONFIG = Path("config") / "default.toml"

BPS = 10_000.0

# --- potencia -----------------------------------------------------------------------
# Un mes de calendario. No es un numero redondo por comodidad: por debajo de ahi el
# diario no ha visto ni un ciclo completo de las cosas que hacen divergir a los dos
# mundos (un fin de semana largo, un dia de volatilidad, una posicion que vive varios
# dias), y la mediana de una decena de fills la fija cualquiera de ellos.
MIN_JOURNAL_DAYS = 30
# Por debajo de esto las distribuciones de coste son anecdota, aunque el calendario de,
# porque el sistema puede pasarse semanas sin operar.
MIN_PAIRED_FILLS = 20

# --- reglas de decision (declaradas antes de mirar el resultado) ---------------------
# Cuanta cobertura de decisiones hace falta para que comparar PnL signifique algo.
# 0,80 = de cada cinco decisiones, cuatro existen en los dos mundos.
DECISION_COVERAGE_MIN = 0.80
# Cuanto puede desviarse el coste realmente cobrado del modelado, en pb medianos. 5 pb es
# el deslizamiento plano de referencia del config: si el error tipico del modelo vale
# tanto como el coste plano que sustituye, el modelo no esta aportando precision.
COST_DIVERGENCE_MAX_BPS = 5.0
# Cuanto puede valer el desplazamiento por latencia frente al coste de referencia que el
# motor YA cobra. 1,0 = si la latencia vale mas que todo el coste modelado, el backtest
# es sistematicamente optimista por una via que no esta en el modelo.
LATENCY_MAX_COST_SHARE = 1.0
# Cuanto puede desviarse de cero la suma de las tres piernas. No es cero porque los
# precios se archivan redondeados a 8 decimales (ver docstring); 1e-4 pb es cuatro
# ordenes por encima de ese ruido y cuatro por debajo de cualquier cifra con significado.
DECOMPOSITION_TOLERANCE_BPS = 1e-4

STATUS_MEASURED = "medido"
STATUS_NO_POWER = "sin_potencia"

KIND_ENTRY = "entry"
KIND_EXIT = "exit"

# Cierres que provoca el propio motor al terminar la ventana re-simulada. No son
# decisiones de la estrategia y no entran en ninguna comparacion.
WINDOW_END_REASON = "window_end"

# Cuantos ejemplos de decision descuadrada se publican. Es diagnostico, no censo: el
# recuento entero va en las cifras, la lista es para poder ir a mirar una.
MAX_EXAMPLES = 12


# ------------------------------------------------------------------ ejecuciones ------


@dataclass(frozen=True, slots=True)
class Execution:
    """
    Una orden ya ejecutada, tal y como quedo archivada. Es la unidad de pareo.

    `reference_price` es el precio con el que se DECIDIO y `filled_price` al que se
    lleno; `decided_at` y `executed_at` son los dos instantes que definen la latencia.
    Los dos ultimos pueden faltar en lineas antiguas —el diario no sellaba la decision
    antes de que existiera este estudio— y el informe publica su cobertura en vez de
    rellenarlos.
    """

    day: str
    symbol: str
    strategy_id: str
    side: str
    kind: str
    size: float
    filled_size: float
    reference_price: float | None
    filled_price: float | None
    slippage_bps: float
    fees_usd: float
    decided_at: datetime | None
    executed_at: datetime | None
    close_reason: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.day, self.symbol, self.strategy_id)

    @property
    def direction(self) -> float:
        """+1 si comprar, -1 si vender. Convierte "precio mas alto" en "peor para mi"."""
        return 1.0 if self.side == "buy" else -1.0

    @property
    def latency_seconds(self) -> float | None:
        if self.decided_at is None or self.executed_at is None:
            return None
        return (self.executed_at - self.decided_at).total_seconds()

    @property
    def is_priceable(self) -> bool:
        return (
            self.filled_price is not None
            and self.reference_price is not None
            and self.reference_price > 0
            and self.filled_size > 0
        )


def _parse_stamp(value) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def _day_of(record: Mapping) -> str:
    return str(record.get("timestamp") or "")[:10]


def executions(records: Iterable[Mapping]) -> list[Execution]:
    """
    Todas las ordenes ejecutadas del diario, entradas y salidas.

    Las entradas salen de emparejar `orders` con `fills` POR POSICION. Es valido porque
    el runner los anade en el mismo paso y en el mismo orden (`_open_position` anota la
    orden y, acto seguido, su resultado), de modo que el i-esimo fill es el de la
    i-esima orden. Si alguna vez dejaran de estar alineados, el desajuste de longitudes
    se cuenta y se declara en vez de aparearse mal en silencio.
    """
    out: list[Execution] = []
    for record in records:
        day = _day_of(record)
        for symbol_block in record.get("symbols", []):
            symbol = str(symbol_block.get("symbol") or "")
            orders = list(symbol_block.get("orders", []))
            fills = list(symbol_block.get("fills", []))
            for order, fill in zip(orders, fills):
                out.append(
                    Execution(
                        day=day,
                        symbol=symbol,
                        strategy_id=str(order.get("strategy_id") or ""),
                        side=str(order.get("side") or ""),
                        kind=KIND_ENTRY,
                        size=float(order.get("size") or 0.0),
                        filled_size=float(fill.get("filled_size") or 0.0),
                        reference_price=_as_float(order.get("reference_price")),
                        filled_price=_as_float(fill.get("filled_price")),
                        slippage_bps=float(fill.get("slippage_bps") or 0.0),
                        fees_usd=float(fill.get("fees_usd") or 0.0),
                        decided_at=_parse_stamp(order.get("decided_at")),
                        executed_at=_parse_stamp(fill.get("executed_at")),
                    )
                )
        for exit_fill in record.get("exits", []):
            out.append(
                Execution(
                    day=day,
                    symbol=str(exit_fill.get("symbol") or ""),
                    strategy_id=str(exit_fill.get("strategy_id") or ""),
                    # El diario guarda el fill de salida, no un `side` propio: la orden
                    # de cierre es la contraria a la posicion y el signo se deriva del
                    # motivo, no de un campo que no existe. Se marca como salida y se
                    # excluye de la descomposicion de precio (ver `price_divergence`).
                    side="",
                    kind=KIND_EXIT,
                    size=float(exit_fill.get("filled_size") or 0.0),
                    filled_size=float(exit_fill.get("filled_size") or 0.0),
                    reference_price=_as_float(exit_fill.get("reference_price")),
                    filled_price=_as_float(exit_fill.get("filled_price")),
                    slippage_bps=float(exit_fill.get("slippage_bps") or 0.0),
                    fees_usd=float(exit_fill.get("fees_usd") or 0.0),
                    decided_at=_parse_stamp(exit_fill.get("decided_at")),
                    executed_at=_parse_stamp(exit_fill.get("executed_at")),
                    close_reason=str(exit_fill.get("close_reason") or "") or None,
                )
            )
    return out


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def order_fill_mismatches(records: Iterable[Mapping]) -> int:
    """Ordenes sin su fill (o al reves). Deberia ser 0 siempre; se cuenta para poder
    afirmarlo en vez de suponerlo."""
    total = 0
    for record in records:
        for symbol_block in record.get("symbols", []):
            total += abs(
                len(symbol_block.get("orders", [])) - len(symbol_block.get("fills", []))
            )
    return total


# ------------------------------------------------------------------- potencia --------


def journal_span(records: Sequence[Mapping]) -> dict:
    """Que trozo de calendario cubre el diario. Los dias son de CALENDARIO, no de
    ciclos: un proceso que despierta 96 veces al dia sigue teniendo un dia de historia."""
    stamps = [_parse_stamp(r.get("timestamp")) for r in records]
    stamps = [s for s in stamps if s is not None]
    if not stamps:
        return {"n_cycles": len(records), "first": None, "last": None, "span_days": 0, "n_days": 0}

    first, last = min(stamps), max(stamps)
    return {
        "n_cycles": len(records),
        "first": first.isoformat(),
        "last": last.isoformat(),
        # Span de calendario, redondeado hacia abajo a dias enteros.
        "span_days": int((last - first).days),
        # Dias UTC DISTINTOS con al menos un ciclo. Es lo que de verdad hay: un proceso
        # apagado dos semanas tiene span grande y pocos dias.
        "n_days": len({s.date() for s in stamps}),
    }


def check_power(span: Mapping, *, min_days: int = MIN_JOURNAL_DAYS) -> dict:
    """
    ¿Hay diario suficiente para que la cifra signifique algo?

    Exige las dos cosas a la vez, y no es redundante: el SPAN dice que ha pasado un mes
    de mercado (que es lo que trae fines de semana, huecos y un dia malo) y los DIAS CON
    CICLOS dicen que el sistema estuvo mirando. Un proceso que corrio dos dias, se apago
    cinco semanas y volvio tiene span de sobra y no ha observado nada.
    """
    span_days = int(span.get("span_days") or 0)
    n_days = int(span.get("n_days") or 0)
    reasons: list[str] = []
    if span_days < min_days:
        reasons.append(f"el diario cubre {span_days} dias de calendario y hacen falta {min_days}")
    if n_days < min_days:
        reasons.append(f"solo hay ciclos en {n_days} dias distintos y hacen falta {min_days}")

    return {
        "sufficient": not reasons,
        "required_days": min_days,
        "span_days": span_days,
        "n_days_with_cycles": n_days,
        "missing_days": max(0, min_days - min(span_days, n_days)),
        "reasons": reasons,
    }


# ---------------------------------------------------------------- re-simulacion ------


def universe(config: AppConfig) -> list[str]:
    """Simbolos con OHLCV del config. Los de prediccion quedan fuera: no tienen barras,
    asi que no hay nada que re-simular ni con que tasar un fill."""
    return [
        s.strip().upper()
        for s in config.runner.symbols
        if not s.strip().upper().startswith(PREDICTION_PREFIX)
    ]


def load_daily_bars(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    offline: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """
    Las barras diarias con las que se re-simula, y los simbolos que se quedaron fuera.

    `offline=True` lee SOLO la cache parquet, con la misma clave que escribe el servicio
    de datos (`data/cache.py::cache_symbol`). Es el modo reproducible: el estudio se
    re-corre dentro de un ano sin depender de que el exchange siga sirviendo el mismo
    historico. El modo con red se apoya en el servicio, que cachea igualmente.
    """
    bars: dict[str, pd.DataFrame] = {}
    omitted: list[dict] = []

    service = None
    if not offline:
        from ai_trader.data.market_data import MarketDataService

        service = MarketDataService()

    for symbol in symbols:
        asset_class = detect_asset_class(symbol)
        if offline:
            frame = load_bars(cache_symbol(symbol, asset_class), timeframe="1D")
        else:
            frame = service.get_daily_bars(symbol, start, end)
        if frame is None or frame.empty:
            omitted.append({"symbol": symbol, "reason": "sin barras diarias disponibles"})
            continue
        bars[symbol] = frame

    return bars, omitted


def resimulate(
    config: AppConfig,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
) -> list[dict]:
    """
    Corre el MISMO periodo del diario con el motor de backtest, y devuelve sus ciclos.

    Una sola ventana, sin corte train/test: aqui no se ajusta nada ni se puntua una
    estrategia, se reproduce un tramo de calendario. Se expresa como un fold de train
    VACIO —`validation.Fold` lo admite explicitamente— en vez de anadirle un metodo al
    motor, porque la geometria temporal ya vive en un sitio y este estudio no es una
    excepcion a esa regla.

    `end` es INCLUSIVO como dia (el ultimo dia del diario se re-simula); el bloque se
    construye semiabierto, que es el convenio de `validation`.
    """
    journal = MemoryJournal()
    engine = BacktestEngine.from_bars(
        config, bars, starting_equity=starting_equity, journal=journal
    )
    fold = Fold(
        label="live",
        scheme="divergence",
        train=(),
        test=(Block(start, end + timedelta(days=1)),),
        purge_days=0,
        embargo_days=0,
    )
    engine.run_folds([fold])
    return journal.records


# ------------------------------------------------------ (3) decisiones no tomadas ----


def decision_keys(records: Iterable[Mapping]) -> dict[str, set[tuple[str, str, str]]]:
    """
    Las decisiones de un diario, deduplicadas por `(dia, simbolo, estrategia)`.

    Deduplicar no es perder informacion: es la unica forma de comparar dos ejecuciones
    con cadencias distintas. En vivo la misma senal sobre la misma barra diaria puede
    repetirse en varios ciclos del mismo dia (o no repetirse, porque abrir posicion corta
    el paso en los siguientes); en backtest hay exactamente una oportunidad por dia. El
    recuento en BRUTO se publica igual, al lado, para que la cadencia no quede escondida.
    """
    signals: set[tuple[str, str, str]] = set()
    approved: set[tuple[str, str, str]] = set()
    rejected: set[tuple[str, str, str]] = set()
    filled: set[tuple[str, str, str]] = set()

    for record in records:
        day = _day_of(record)
        for symbol_block in record.get("symbols", []):
            symbol = str(symbol_block.get("symbol") or "")
            for signal in symbol_block.get("signals", []):
                signals.add((day, symbol, str(signal.get("strategy_id") or "")))
            for decision in symbol_block.get("risk", []):
                key = (day, symbol, str(decision.get("strategy_id") or ""))
                (approved if decision.get("approved") else rejected).add(key)
            for order, fill in zip(
                symbol_block.get("orders", []), symbol_block.get("fills", [])
            ):
                if fill.get("success"):
                    filled.add((day, symbol, str(order.get("strategy_id") or "")))

    return {"signals": signals, "approved": approved, "rejected": rejected, "filled": filled}


def decision_divergence(live: Sequence[Mapping], resim: Sequence[Mapping]) -> dict:
    """
    El embudo de decisiones de los dos mundos, y donde se separan.

    Es la pierna que detecta lo que el PnL esconde: si en vivo se generan la mitad de las
    senales, el problema no es el coste de ejecucion sino los DATOS, y comparar dinero
    estaria explicando la diferencia equivocada. Por eso la cobertura tiene umbral propio
    y se evalua ANTES que cualquier cifra de precio.
    """
    live_keys = decision_keys(live)
    resim_keys = decision_keys(resim)
    live_raw = journal_summary(list(live))
    resim_raw = journal_summary(list(resim))

    stages: dict[str, dict] = {}
    for stage in ("signals", "approved", "rejected", "filled"):
        a, b = live_keys[stage], resim_keys[stage]
        union = a | b
        stages[stage] = {
            "live": len(a),
            "resim": len(b),
            "both": len(a & b),
            "only_live": len(a - b),
            "only_resim": len(b - a),
            # Jaccard: acuerdo sobre el conjunto de decisiones, no sobre su numero. Dos
            # mundos que toman 10 decisiones cada uno pero ninguna en comun dan 0.
            "coverage": round(len(a & b) / len(union), 4) if union else None,
            "examples_only_live": _examples(a - b),
            "examples_only_resim": _examples(b - a),
        }

    signals = stages["signals"]
    return {
        "unit": "(dia UTC, simbolo, estrategia)",
        "stages": stages,
        # Cobertura de la primera etapa: si las SENALES ya no coinciden, lo de abajo es
        # consecuencia y no causa.
        "coverage": signals["coverage"],
        "threshold": DECISION_COVERAGE_MIN,
        "raw_counts": {
            "live": _funnel(live_raw),
            "resim": _funnel(resim_raw),
            "note": (
                "Recuento en bruto, sin deduplicar: en vivo hay un ciclo cada "
                "interval_seconds y en la re-simulacion uno por dia de mercado, asi que "
                "estas dos columnas NO son comparables entre si. Lo comparable es "
                "`stages`."
            ),
        },
    }


def _funnel(summary: Mapping) -> dict:
    return {
        key: summary.get(key)
        for key in (
            "n_cycles",
            "n_signals",
            "n_approved",
            "n_rejected",
            "n_fills",
            "n_exit_fills",
        )
    }


def _examples(keys: set[tuple[str, str, str]]) -> list[dict]:
    return [
        {"day": day, "symbol": symbol, "strategy_id": strategy}
        for day, symbol, strategy in sorted(keys)[:MAX_EXAMPLES]
    ]


# --------------------------------------------- (1)(2) precio, coste y su reparto -----


@dataclass(frozen=True, slots=True)
class Repriced:
    """Una entrada archivada, re-tasada con el modelo y la liquidez del backtest."""

    execution: Execution
    modeled_reference: float
    modeled_slippage_bps: float
    modeled_price: float
    modeled_fees_usd: float

    @property
    def total_bps(self) -> float:
        """Cuanto peor se lleno en vivo que lo que el modelo predecia, en pb."""
        gap = self.execution.filled_price - self.modeled_price
        return self.execution.direction * BPS * gap / self.modeled_reference

    @property
    def reference_bps(self) -> float:
        drift = (self.execution.reference_price - self.modeled_reference) / self.modeled_reference
        return self.execution.direction * BPS * drift

    @property
    def cost_bps(self) -> float:
        return self.execution.slippage_bps - self.modeled_slippage_bps

    @property
    def cross_bps(self) -> float:
        drift = (self.execution.reference_price - self.modeled_reference) / self.modeled_reference
        return self.execution.slippage_bps * drift

    def as_dict(self) -> dict:
        return {
            "day": self.execution.day,
            "symbol": self.execution.symbol,
            "side": self.execution.side,
            "size": self.execution.filled_size,
            "live_reference": self.execution.reference_price,
            "modeled_reference": round(self.modeled_reference, 8),
            "live_price": self.execution.filled_price,
            "modeled_price": round(self.modeled_price, 8),
            "live_slippage_bps": self.execution.slippage_bps,
            "modeled_slippage_bps": round(self.modeled_slippage_bps, 6),
            "total_bps": round(self.total_bps, 4),
            "reference_bps": round(self.reference_bps, 4),
            "cost_bps": round(self.cost_bps, 4),
            "cross_bps": round(self.cross_bps, 6),
        }


class Repricer:
    """
    Vuelve a poner precio a una orden archivada, con las piezas del BACKTEST.

    No reimplementa el modelo: usa el mismo `SlippageModel` del config, la misma
    `BarLiquidityProvider` sobre el mismo `HistoricalDataSource` y la misma referencia
    (el OPEN del dia) que `IntrabarMarketModel`. Lo unico que cambia respecto de lo que
    paso en vivo es el CONTEXTO, que es justo lo que se quiere medir.
    """

    def __init__(self, config: AppConfig, bars: dict[str, pd.DataFrame]) -> None:
        self.clock = HistoricalClock(datetime(1970, 1, 1, tzinfo=timezone.utc))
        self.source = HistoricalDataSource(bars, self.clock)
        self.liquidity = BarLiquidityProvider(self.source, self.clock)
        self.slippage: SlippageModel = config.execution.slippage
        self.fee_rate = float(config.execution.fee_rate)

    def reprice(self, execution: Execution) -> Repriced | None:
        """None si el dia no tiene barra para ese simbolo: sin barra, el backtest no
        habria podido operarlo y no hay nada con que comparar."""
        day = pd.Timestamp(execution.day, tz="UTC").to_pydatetime()
        self.clock.set(day)

        bar = self.source.bar_on(execution.symbol, day)
        if bar is None or not bar.open > 0:
            return None

        snapshot = self.liquidity.snapshot(execution.symbol)
        modeled_bps = self.slippage.slippage_bps(
            symbol=execution.symbol,
            size=execution.filled_size,
            snapshot=snapshot,
            asset_class=detect_asset_class(execution.symbol),
        )
        modeled_price = fill_price(bar.open, execution.side, modeled_bps)
        return Repriced(
            execution=execution,
            modeled_reference=float(bar.open),
            modeled_slippage_bps=modeled_bps,
            modeled_price=modeled_price,
            modeled_fees_usd=round(modeled_price * execution.filled_size * self.fee_rate, 8),
        )


def price_divergence(execs: Sequence[Execution], repricer: Repricer) -> dict:
    """
    (1) y (2): el precio de llenado y su reparto en referencia + coste + cruzado.

    Solo ENTRADAS. Una salida no tiene un precio de referencia que el backtest fije de
    la misma forma —sale al stop, al objetivo o a la marca, cada uno con su convencion—,
    asi que meterla aqui exigiria inventarle al modelo una referencia que no usa. Las
    salidas se miden en la pierna de coste (`cost_divergence`), donde el deslizamiento
    realizado no necesita referencia, y en la de latencia.
    """
    entries = [e for e in execs if e.kind == KIND_ENTRY]
    priceable = [e for e in entries if e.is_priceable]
    repriced = [r for r in (repricer.reprice(e) for e in priceable) if r is not None]

    # Las tres bajas se cuentan por separado porque significan cosas distintas: una orden
    # rechazada (sin precio de llenado) es una decision que no llego a ejecutarse, y una
    # sin barra del dia es un simbolo que el backtest no habria podido operar.
    census = {
        "n_entries": len(entries),
        "n_unfilled": len(entries) - len(priceable),
        "n_without_bar": len(priceable) - len(repriced),
        "n_repriced": len(repriced),
    }
    if not repriced:
        return {
            **census,
            "reason": "ninguna entrada tenia barra diaria con la que re-tasarla",
        }

    total = [r.total_bps for r in repriced]
    residual = round(
        max(
            abs(r.total_bps - (r.reference_bps + r.cost_bps + r.cross_bps)) for r in repriced
        ),
        12,
    )
    return {
        **census,
        "unit": "pb sobre la referencia del modelo; positivo = en vivo se pago MAS",
        "total_bps": _stats(total),
        "components": {
            "reference_bps": _stats([r.reference_bps for r in repriced]),
            "cost_bps": _stats([r.cost_bps for r in repriced]),
            "cross_bps": _stats([r.cross_bps for r in repriced]),
        },
        # La comprobacion de que la descomposicion CIERRA. Si el residuo pasara de la
        # tolerancia, alguna pierna estaria absorbiendo el error de otra sin decirlo.
        "decomposition_residual_max": residual,
        "decomposition_tolerance_bps": DECOMPOSITION_TOLERANCE_BPS,
        "decomposition_ok": bool(residual <= DECOMPOSITION_TOLERANCE_BPS),
        "worst": [
            r.as_dict()
            for r in sorted(repriced, key=lambda x: -abs(x.total_bps))[:MAX_EXAMPLES]
        ],
    }


def cost_divergence(execs: Sequence[Execution], repricer: Repricer) -> dict:
    """
    (2) aislado: deslizamiento y comision REALMENTE cobrados contra los modelados.

    En el DESLIZAMIENTO entran tambien las salidas: comparar pb contra pb no necesita
    saber a que precio de referencia se decidio, solo el simbolo, el tamano y el dia.

    En la COMISION entran solo las entradas, y no por descuido: la comision modelada es
    `fee_rate x precio x tamano`, y el precio que el backtest habria usado para una
    SALIDA depende de por donde salio —stop, objetivo o marca, cada uno con su
    convencion—, que no es reconstruible desde el diario. Meterlas con el open del dia
    daria una cifra con pinta de medida que en realidad seria una aproximacion; se
    prefiere medir menos y que lo medido sea exacto.

    La comision es hoy un control de consistencia: mientras la ejecucion sea de papel la
    cobra el mismo motor en los dos mundos y tiene que cuadrar al centimo. El dia que
    haya broker, el mismo campo pasa a ser una medicion sin tocar una linea.
    """
    rows: list[tuple[Execution, Repriced]] = []
    for execution in execs:
        if execution.filled_size <= 0 or execution.filled_price is None:
            continue
        repriced = repricer.reprice(execution)
        if repriced is not None:
            rows.append((execution, repriced))

    if not rows:
        return {"n_fills": 0, "reason": "ningun fill tenia barra diaria con la que modelar el coste"}

    slippage_gap = [e.slippage_bps - r.modeled_slippage_bps for e, r in rows]
    entries = [(e, r) for e, r in rows if e.kind == KIND_ENTRY]
    fee_gap = [e.fees_usd - r.modeled_fees_usd for e, r in entries]
    return {
        "n_fills": len(rows),
        "n_entries": sum(1 for e, _ in rows if e.kind == KIND_ENTRY),
        "n_exits": sum(1 for e, _ in rows if e.kind == KIND_EXIT),
        "slippage_bps": {
            "live": _stats([e.slippage_bps for e, _ in rows]),
            "modeled": _stats([r.modeled_slippage_bps for _, r in rows]),
            "gap": _stats(slippage_gap),
            "abs_gap_median": _median([abs(v) for v in slippage_gap]),
        },
        "fees_usd": {
            "n_entries": len(entries),
            "live_total": round(sum(e.fees_usd for e, _ in entries), 6),
            "modeled_total": round(sum(r.modeled_fees_usd for _, r in entries), 6),
            "abs_gap_max": round(max((abs(v) for v in fee_gap), default=0.0), 8),
            "note": (
                "Solo ENTRADAS: el precio con el que el backtest habria cerrado una "
                "posicion depende de por donde salio y no se puede reconstruir. Es un "
                "control de consistencia mientras la ejecucion sea de papel, porque la "
                "comision la cobra el mismo motor en los dos mundos."
            ),
        },
        "threshold_bps": COST_DIVERGENCE_MAX_BPS,
    }


# ------------------------------------------------------------------ (2c) latencia ----


def latency_divergence(
    execs: Sequence[Execution],
    config: AppConfig,
    *,
    hourly: Mapping[str, pd.DataFrame] | None = None,
) -> dict:
    """
    (2c): el hueco entre decidir y llenar, en tiempo y en dinero.

    Dos huecos distintos, y se publican separados porque no son la misma pregunta:

    - `decision_to_fill`: `executed_at - decided_at`. Es la latencia del propio sistema
      (construir la orden, enrutarla, que el motor responda). En papel son milisegundos;
      con broker sera la cifra que diga si hace falta modelarla.
    - `reference_age`: cuanto tiempo llevaba cerrado el precio con el que se decidio, en
      el instante del fill. En vivo el sistema decide con el ultimo cierre DIARIO ya
      cerrado, asi que esa antiguedad va de 0 a 24 h segun la hora a la que despierte el
      ciclo. Es el hueco que un mercado real cobra entero, y el que el backtest no ve
      porque llena al open de la misma barra.

    Y el precio de lo segundo: con barras 1H reales, cuanto se habia desplazado ya el
    mercado entre ese cierre y el fill, en pb y con signo (positivo = en contra). Se
    compara con `reference_cost_bps`, que es el coste que el motor SI cobra: si el
    desplazamiento vale mas que todo ese coste, la latencia no es un detalle.
    """
    reference = reference_cost_bps(config)
    latencies = [e.latency_seconds for e in execs]
    measured = [v for v in latencies if v is not None]

    ages: list[float] = []
    drifts: list[float] = []
    for execution in execs:
        if execution.executed_at is None:
            continue
        bar_close = execution.executed_at.replace(hour=0, minute=0, second=0, microsecond=0)
        ages.append((execution.executed_at - bar_close).total_seconds() / 3600.0)
        # El desplazamiento se tasa CON SIGNO ("a favor" o "en contra"), y el signo lo da
        # el lado de la orden. Una salida no archiva su lado —es el contrario al de la
        # posicion, que vive en otro bloque de la linea—, asi que se queda fuera de la
        # tasacion en pb y dentro de la de tiempo, donde el signo no hace falta.
        if execution.kind != KIND_ENTRY:
            continue
        drift = _drift_bps(execution, bar_close, hourly)
        if drift is not None:
            drifts.append(drift)

    out = {
        "n_executions": len(execs),
        "decision_to_fill_seconds": {
            "n_measured": len(measured),
            "coverage": round(len(measured) / len(latencies), 4) if latencies else None,
            "note": (
                "Las lineas archivadas antes de que el diario sellara `decided_at` no "
                "tienen este hueco y NO se rellenan: se declara su cobertura."
            ),
            **_stats(measured),
        },
        "reference_age_hours": _stats(ages),
        "reference_cost_bps": round(reference, 4),
        "threshold_share": LATENCY_MAX_COST_SHARE,
    }

    if not drifts:
        out["drift_bps"] = {
            "n": 0,
            "reason": (
                "sin barras 1H en cache para los simbolos operados: la latencia se "
                "publica en tiempo, no en pb"
            ),
        }
        return out

    stats = _stats(drifts)
    median = stats["median"]
    out["drift_bps"] = {
        **stats,
        "unit": "pb; positivo = el mercado ya se habia movido EN CONTRA al llenar",
        "share_of_reference_cost": (
            None if median is None or reference <= 0 else round(median / reference, 4)
        ),
    }
    return out


def _drift_bps(
    execution: Execution,
    bar_close: datetime,
    hourly: Mapping[str, pd.DataFrame] | None,
) -> float | None:
    """Desplazamiento entre el precio de referencia y el mercado en el instante del fill,
    en pb y con el signo de la operacion. None si no hay barra horaria que lo diga."""
    if hourly is None or execution.reference_price is None or execution.reference_price <= 0:
        return None
    frame = hourly.get(execution.symbol)
    if frame is None or frame.empty or execution.executed_at is None:
        return None

    window = frame[(frame.index >= bar_close) & (frame.index <= execution.executed_at)]
    if window.empty:
        return None

    last = float(bar_schema.series(window, bar_schema.CLOSE).iloc[-1])
    gap = (last - execution.reference_price) / execution.reference_price
    return execution.direction * BPS * gap


def load_hourly(
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    """Barras 1H de la cache, sin red (`provider=None`). Solo cripto: el modulo horario
    es de cripto por construccion, y los simbolos sin cache se quedan fuera en silencio
    porque su ausencia ya se refleja en el recuento de `drift_bps`."""
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        if detect_asset_class(symbol) != AssetClass.CRYPTO:
            continue
        try:
            frame = get_hourly_bars(symbol, start, end + timedelta(days=1), provider=None)
        except ValueError:
            continue
        if frame is not None and not frame.empty:
            out[symbol] = frame
    return out


# ------------------------------------------------------------------- veredicto -------


def verdict(decisions: Mapping, cost: Mapping, latency: Mapping, *, n_fills: int) -> dict:
    """
    Las tres reglas declaradas, evaluadas. Cada una puede FALLAR, y una regla que no
    puede fallar no es evidencia.

    `ok` global es la conjuncion. Una regla sin datos suficientes no se da por buena: se
    marca `null` y arrastra el global a `null`, que es distinto de aprobar.
    """
    rules: dict[str, dict] = {}

    coverage = decisions.get("coverage")
    rules["decisions"] = {
        "rule": f"cobertura de senales >= {DECISION_COVERAGE_MIN}",
        "value": coverage,
        "ok": None if coverage is None else bool(coverage >= DECISION_COVERAGE_MIN),
        "text": (
            "sin decisiones que comparar"
            if coverage is None
            else (
                f"los dos mundos comparten el {coverage:.0%} de las senales"
                if coverage >= DECISION_COVERAGE_MIN
                else (
                    f"solo el {coverage:.0%} de las senales existe en los dos mundos: la "
                    "divergencia es de DATOS, no de coste"
                )
            )
        ),
    }

    gap = cost.get("slippage_bps", {}).get("abs_gap_median") if cost.get("n_fills") else None
    enough = n_fills >= MIN_PAIRED_FILLS
    rules["cost"] = {
        "rule": f"|coste realizado - modelado| mediano <= {COST_DIVERGENCE_MAX_BPS} pb",
        "value": gap,
        "min_fills": MIN_PAIRED_FILLS,
        "n_fills": n_fills,
        "ok": None if (gap is None or not enough) else bool(gap <= COST_DIVERGENCE_MAX_BPS),
        "text": (
            f"solo {n_fills} fills pareados (hacen falta {MIN_PAIRED_FILLS}): sin potencia"
            if not enough
            else "sin fills que modelar"
            if gap is None
            else f"el coste realizado se aparta {gap:.2f} pb del modelado (mediana absoluta)"
        ),
    }

    share = latency.get("drift_bps", {}).get("share_of_reference_cost")
    rules["latency"] = {
        "rule": (
            f"desplazamiento por latencia mediano <= {LATENCY_MAX_COST_SHARE} x coste de "
            "referencia"
        ),
        "value": share,
        "ok": None if share is None else bool(share <= LATENCY_MAX_COST_SHARE),
        "text": (
            "sin barras 1H con las que tasar la latencia"
            if share is None
            else f"la latencia vale {share:.2f} veces el coste que el motor ya cobra"
        ),
    }

    flags = [r["ok"] for r in rules.values()]
    return {
        "rules": rules,
        "ok": None if any(f is None for f in flags) else all(flags),
    }


# --------------------------------------------------------------------- informe -------


def cycle_interval_seconds() -> int | None:
    """Cada cuanto despierta el ciclo automatico. Es CONTEXTO —lo que explica por que en
    vivo hay 96 lineas al dia y en la re-simulacion una—, no un dato del que dependa
    ninguna cifra: si el paquete de Telegram no esta instalado, el informe se publica
    igual con None."""
    try:
        from ai_trader.bots.telegram_bot import AUTO_CYCLE_INTERVAL_SECONDS
    except Exception:  # noqa: BLE001 - dependencia opcional
        return None
    return int(AUTO_CYCLE_INTERVAL_SECONDS)


def build_plan(args: argparse.Namespace, config: AppConfig, symbols: Sequence[str]) -> dict:
    return {
        "config_path": str(args.config),
        "journal_path": str(args.journal).replace("\\", "/"),
        "offline": bool(args.offline),
        "starting_equity": float(args.starting_equity),
        "symbols": list(symbols),
        "cycle_interval_seconds": cycle_interval_seconds(),
        "reference_cost_bps": round(reference_cost_bps(config), 4),
        "thresholds": {
            "min_journal_days": int(args.min_days),
            "min_paired_fills": MIN_PAIRED_FILLS,
            "decision_coverage_min": DECISION_COVERAGE_MIN,
            "cost_divergence_max_bps": COST_DIVERGENCE_MAX_BPS,
            "latency_max_cost_share": LATENCY_MAX_COST_SHARE,
        },
    }


def analyze(
    live_records: Sequence[Mapping],
    config: AppConfig,
    plan: dict,
    *,
    min_days: int = MIN_JOURNAL_DAYS,
    offline: bool = False,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    bars: dict[str, pd.DataFrame] | None = None,
    hourly: Mapping[str, pd.DataFrame] | None = None,
) -> dict:
    """
    El informe entero. Con potencia insuficiente devuelve el diagnostico y NADA de cifra.

    El orden de las piernas no es casual: primero se comprueba que los dos mundos vean
    las mismas decisiones y solo despues se comparan precios. Publicar una divergencia de
    coste cuando la mitad de las senales no existe en un lado seria atribuir al coste una
    diferencia que es de datos.

    `bars` y `hourly` permiten pasar las series ya cargadas. Es la misma costura que
    `from_bars` en el motor —quien las tenga en memoria no vuelve a leerlas de disco— y
    es lo que hace que el camino MEDIDO sea testeable hoy, meses antes de que el diario
    real tenga calendario suficiente para correrlo. Que `hourly` sea inyectable ademas
    evita un fallo sutil: sin esa costura, un test sobre precios inventados acabaria
    tasando su latencia contra las barras 1H REALES que haya en la cache del que lo
    corra, y mediria la distancia entre dos mundos que no tienen nada que ver.
    """
    span = journal_span(live_records)
    power = check_power(span, min_days=min_days)
    header = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "journal": {**span, "order_fill_mismatches": order_fill_mismatches(live_records)},
        "power": power,
    }

    if not power["sufficient"]:
        logger.warning(
            "Sin potencia: %s. No se re-simula ni se publica divergencia.",
            "; ".join(power["reasons"]),
        )
        return {
            **header,
            "status": STATUS_NO_POWER,
            "decisions": None,
            "fill_price": None,
            "cost": None,
            "latency": None,
            "verdict": None,
        }

    start = _utc_midnight(span["first"])
    end = _utc_midnight(span["last"])
    symbols = list(plan["symbols"])

    omitted: list[dict] = []
    if bars is None:
        bars, omitted = load_daily_bars(
            symbols, start - timedelta(days=400), end, offline=offline
        )
    if not bars:
        raise ValueError(
            "Ningun simbolo del universo tiene barras diarias: no hay nada que re-simular"
        )

    logger.info(
        "Re-simulando [%s, %s] sobre %d simbolos%s",
        start.date(), end.date(), len(bars), " [offline: solo cache]" if offline else "",
    )
    resim_records = resimulate(config, bars, start, end, starting_equity=starting_equity)

    live_execs = executions(live_records)
    resim_execs = [
        e
        for e in executions(resim_records)
        # Los cierres que el motor provoca al terminar la ventana no son decisiones de la
        # estrategia: existirian aunque el sistema no hubiera hecho nada.
        if e.close_reason != WINDOW_END_REASON
    ]
    repricer = Repricer(config, bars)

    decisions = decision_divergence(live_records, resim_records)
    fill_price_block = price_divergence(live_execs, repricer)
    cost = cost_divergence(live_execs, repricer)
    if hourly is None:
        hourly = load_hourly((e.symbol for e in live_execs), start, end)
    latency = latency_divergence(live_execs, config, hourly=hourly)

    return {
        **header,
        "status": STATUS_MEASURED,
        "resimulation": {
            "n_cycles": len(resim_records),
            "n_executions": len(resim_execs),
            "n_symbols": len(bars),
            "omitted_symbols": omitted,
            # Lo que separa a los dos mundos ADEMAS de la ejecucion. Van declaradas
            # porque son la primera explicacion que hay que descartar al leer una
            # divergencia de recuentos: sin esta lista, "sobran senales en la
            # re-simulacion" parece un fallo de datos y puede ser solo el arranque.
            "asymmetries": [
                "La re-simulacion arranca SIN posiciones abiertas. Si el diario empieza "
                "con posiciones vivas, en vivo esos simbolos estaban bloqueados y en la "
                "re-simulacion no: aparecen como senales 'solo resim' al principio.",
                "La re-simulacion liquida todo el ultimo dia (convencion del motor). Esos "
                "cierres llevan close_reason='window_end' y quedan fuera de toda "
                "comparacion.",
                "El enfriamiento por simbolo se cuenta en horas y en vivo el runner "
                "despierta decenas de veces al dia: bloquea distinto en cada mundo.",
            ],
        },
        "decisions": decisions,
        "fill_price": fill_price_block,
        "cost": cost,
        "latency": latency,
        "verdict": verdict(decisions, cost, latency, n_fills=cost.get("n_fills", 0)),
        "ceiling": (
            "La ejecucion en vivo es de PAPEL: el precio de llenado lo produce el mismo "
            "motor que el backtest, asi que la pierna de coste mide contexto en vivo "
            "contra contexto re-simulado, no modelo contra mercado. La pierna de "
            "referencia/latencia no tiene ese techo."
        ),
    }


def _utc_midnight(value: str | None) -> datetime:
    stamp = _parse_stamp(value)
    if stamp is None:
        raise ValueError("El diario no tiene marcas de tiempo utilizables")
    return stamp.replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------- estadistica --------


def _stats(values: Sequence[float]) -> dict:
    """Resumen de una distribucion. `n` primero a proposito: una mediana sin su n no
    dice nada, y aqui casi todas las muestras seran pequenas durante meses."""
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"n": 0, "median": None, "mean": None, "p90": None, "max_abs": None}
    array = np.asarray(clean, dtype=float)
    return {
        "n": int(array.size),
        "median": round(float(np.median(array)), 6),
        "mean": round(float(array.mean()), 6),
        "p90": round(float(np.percentile(array, 90)), 6),
        "max_abs": round(float(np.abs(array).max()), 6),
    }


def _median(values: Sequence[float]) -> float | None:
    return _stats(values)["median"]


# ------------------------------------------------------------------- lectura ---------


def load_divergence_report(path: Path | str = DIVERGENCE_REPORT) -> dict | None:
    """Lee el informe publicado. None si nunca se ha corrido, para que el dashboard y la
    documentacion degraden a prosa sin cifras en vez de romperse."""
    return load_report(path)


# ----------------------------------------------------------------------- CLI ---------


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    journal = CycleJournal(args.journal)
    records = journal.read()
    plan = build_plan(args, config, universe(config))
    plan["n_shards"] = len(journal.shards())
    return analyze(
        records,
        config,
        plan,
        min_days=int(args.min_days),
        offline=bool(args.offline),
        starting_equity=float(args.starting_equity),
    )


def _strip_volatile(report: dict) -> dict:
    return {k: v for k, v in report.items() if k != "generated_at"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_PATH))
    parser.add_argument("--out", default=str(DIVERGENCE_REPORT))
    parser.add_argument("--min-days", type=int, default=MIN_JOURNAL_DAYS)
    parser.add_argument("--starting-equity", type=float, default=DEFAULT_STARTING_EQUITY)
    parser.add_argument(
        "--offline", action="store_true", help="No tocar la red: usar solo la cache."
    )
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.WARNING)

    report = run(args)

    if args.verify_determinism:
        replay = run(args)
        identical = _strip_volatile(report) == _strip_volatile(replay)
        report["determinism"] = {"checked": True, "identical": identical}
        logger.info("Determinismo: informe %s", "identico" if identical else "DISTINTO")

    path = write_report(report, args.out)
    logger.info("Informe -> %s", path)
    _print_report(report)
    return 0


def _print_report(report: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    journal, power = report["journal"], report["power"]
    print("\n=== DIVERGENCIA LIVE-VS-BACKTEST ===")
    print(
        f"  diario: {journal['n_cycles']:,} ciclos | {journal['first']} -> {journal['last']} "
        f"| {journal['span_days']} dias de calendario, {journal['n_days']} con ciclos"
    )

    if report["status"] == STATUS_NO_POWER:
        print(f"\n  SIN POTENCIA (hacen falta {power['required_days']} dias):")
        for reason in power["reasons"]:
            print(f"    - {reason}")
        print(
            "\n  No se publica cifra de divergencia. Es la unica parte del proyecto que "
            "consume calendario y no computo."
        )
        return

    decisions = report["decisions"]
    print("\n--- (3) Decisiones: lo que el PnL esconde ---")
    print(f"{'etapa':<12}{'vivo':>8}{'resim':>8}{'ambos':>8}{'solo vivo':>12}{'solo resim':>12}")
    for stage, row in decisions["stages"].items():
        print(
            f"{stage:<12}{row['live']:>8}{row['resim']:>8}{row['both']:>8}"
            f"{row['only_live']:>12}{row['only_resim']:>12}"
        )
    print(f"  {report['verdict']['rules']['decisions']['text']}")

    price = report["fill_price"]
    print("\n--- (1) Precio de llenado, y su reparto ---")
    if price.get("n_repriced"):
        total, comps = price["total_bps"], price["components"]
        print(f"  total          mediana {total['median']:+.2f} pb  (n={total['n']})")
        print(f"    referencia   mediana {comps['reference_bps']['median']:+.2f} pb")
        print(f"    coste        mediana {comps['cost_bps']['median']:+.2f} pb")
        print(f"    cruzado      mediana {comps['cross_bps']['median']:+.4f} pb")
        print(f"  residuo de la descomposicion: {price['decomposition_residual_max']:.2e}")
    else:
        print(f"  {price.get('reason', 'sin entradas re-tasables')}")

    print("\n--- (2) Coste: cobrado contra modelado ---")
    print(f"  {report['verdict']['rules']['cost']['text']}")

    latency = report["latency"]
    print("\n--- (2c) Latencia ---")
    age = latency["reference_age_hours"]
    print(f"  antiguedad del precio de referencia al llenar: mediana {age['median']} h "
          f"(p90 {age['p90']} h)")
    print(f"  {report['verdict']['rules']['latency']['text']}")

    print(f"\n  veredicto global: {report['verdict']['ok']}")
    print(f"  techo declarado: {report['ceiling']}")


if __name__ == "__main__":
    raise SystemExit(main())
