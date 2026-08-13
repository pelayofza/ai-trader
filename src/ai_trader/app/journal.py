"""
EL DIARIO DE CICLOS: la pelicula, no la foto.

    data/live/cycles.jsonl          <- el ciclo en curso, siempre este nombre
    data/live/cycles-2026-08.001.jsonl   <- shards ya cerrados (mes / tamano)

`app/state_store.py` guarda la FOTO: que posiciones hay ahora y cuanto PnL llevan. Es
todo lo que el sistema necesita para seguir operando tras un reinicio, y es exactamente
lo que NO sirve para auditar. La foto no dice que se decidio, ni con que precio de
referencia, ni por que el riesgo rechazo una senal, ni cuanto deslizamiento se cobro de
verdad; y esa es la materia prima de la unica medicion que este proyecto no puede
acelerar con computo: cuanto se aparta lo ejecutado de lo que el backtest predecia.

POR QUE APPEND-ONLY, Y POR QUE fsync
------------------------------------
El mismo principio que el archivo crudo de senales (`signals/store.py:22`): lo que no se
escriba en el momento no vuelve. Un midpoint de Polymarket de las 15:00 de hoy no se
puede reconstruir manana, y el `slippage_bps` que el modelo de microestructura cobro en
un fill depende del estado de liquidez de ese instante. De ahi las dos propiedades:

- NUNCA se reescribe el fichero entero. Se abre en modo 'a' y se escribe una linea. Un
  fallo a mitad de ciclo deja una linea rota al final —que el lector cuenta y salta— y
  no un fichero de meses truncado.
- `fsync` por linea. El proceso corre durante meses en una maquina domestica: el corte
  de luz no es un caso hipotetico, es el caso. Sin fsync, el buffer del sistema
  operativo puede llevar minutos de ciclos sin tocar el disco. Son ~96 fsync al dia con
  el intervalo por defecto de 900 s: irrelevante frente a una sola llamada de red del
  ciclo.

ROTACION
--------
Dos disparadores, y el primero que salte manda: el mes del ciclo cambia, o el fichero
supera `MAX_SHARD_BYTES`. Rotar es RENOMBRAR el fichero actual a un shard con secuencia
(`cycles-2026-08.001.jsonl`) y empezar uno nuevo; nunca se reescribe ni se recorta nada.
La secuencia va en el nombre para que el orden alfabetico de los shards sea el orden
cronologico, que es lo que permite leerlos con un `sorted()` y sin metadatos.

El fichero en curso conserva SIEMPRE el mismo nombre. Es lo que permite que un `tail -f`,
el dashboard o una tarea programada apunten a una ruta fija.

QUE NO VA AQUI
--------------
Nada derivado. El diario no calcula metricas al escribir: guarda lo que paso y las
funciones de lectura (`pnl_curve`, `journal_summary`) derivan lo que haga falta. Si
manana se quiere medir la divergencia de otra forma, se recalcula sobre lo archivado; si
se hubiera guardado ya agregado, no.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ai_trader.backtest.metrics import EquityPoint, max_drawdown_pct
from ai_trader.shared.jsonl import read_records, to_line
from ai_trader.shared.schemas import ExecutionResult, Position, RiskDecision, Signal

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_PATH = Path("data") / "live" / "cycles.jsonl"

# Tamano al que se cierra un shard. 8 MB son del orden de un mes de ciclos con el
# universo de 24 pares y el intervalo de 900 s, asi que en la practica manda el mes y
# esto es solo la red que impide que un fichero crezca sin limite si algo se dispara.
MAX_SHARD_BYTES = 8 * 1024 * 1024

# Estados posibles de un ciclo. Un ciclo pausado o deshabilitado TAMBIEN deja linea: es
# lo que distingue "el proceso sigue vivo y no opera" de "el proceso se cayo".
STATUS_RAN = "ran"
STATUS_PAUSED = "paused"
STATUS_DISABLED = "disabled"


# --- serializacion de los eventos de un ciclo ---------------------------------------
#
# Vive aqui y no en el runner para que el ESQUEMA del diario este en un solo fichero: el
# runner emite eventos, no JSON.


def signal_record(signal: Signal) -> dict[str, Any]:
    return {
        "strategy_id": signal.strategy_id,
        "side": signal.side.value,
        "confidence": round(signal.confidence, 4),
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "reason": signal.reason,
        "signal_id": signal.signal_id,
    }


def risk_record(signal: Signal, decision: RiskDecision) -> dict[str, Any]:
    return {
        "strategy_id": signal.strategy_id,
        "approved": decision.approved,
        # El motivo COMPLETO, con sus cifras dentro ("Signal confidence below minimum
        # threshold: 0.42"). Agrupar por familia es cosa de quien lee, no de quien
        # escribe: recortarlo aqui perderia el numero que explica el rechazo.
        "reason": decision.reason,
        "size_usd": decision.size_usd,
        "stop_loss": decision.stop_loss,
        "take_profit": decision.take_profit,
        "warnings": list(decision.warnings),
    }


def order_record(
    strategy_id: str,
    side: str,
    size: float,
    reference_price: float,
    *,
    decided_at: datetime,
) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "side": side,
        "size": size,
        # El precio con el que se DECIDIO. Comparado con `filled_price` da el
        # deslizamiento realizado, que es la mitad de la divergencia live-vs-backtest.
        "reference_price": reference_price,
        # CUANDO se decidio, y por que no vale el sello del ciclo: la linea del diario
        # se escribe al TERMINAR el ciclo, asi que su `timestamp` es posterior a todos
        # los fills y restarselos daria un hueco negativo. La otra mitad de la
        # divergencia -la latencia- es `executed_at` menos ESTE instante, y no hay
        # ningun otro campo del que se pueda derivar.
        "decided_at": decided_at.isoformat(),
    }


def fill_record(result: ExecutionResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "status": result.status.value,
        "order_id": result.order_id,
        "filled_price": result.filled_price,
        "filled_size": result.filled_size,
        "fees_usd": result.fees,
        # El deslizamiento REALMENTE cobrado por el modelo de microestructura, no el
        # plano de referencia de la configuracion.
        "slippage_bps": result.slippage_bps,
        "message": result.message,
        "executed_at": result.executed_at.isoformat(),
    }


def exit_record(
    position: Position,
    result: ExecutionResult,
    reason: str,
    *,
    reference_price: float,
    decided_at: datetime,
) -> dict[str, Any]:
    """
    La EJECUCION de la orden que cierra una posicion: el mismo fill, mas de que
    posicion salia, por que, y con que precio se decidio salir. Su gemelo
    `closed_record` es la CONTABILIDAD del cierre.

    Lleva las mismas dos cosas que la orden de entrada, y por el mismo motivo: salir
    tambien paga deslizamiento y tambien tiene latencia. Sin `reference_price` no se
    puede saber cuanto se deslizo la salida, y sin `decided_at` no se puede fechar; y
    ninguna de las dos se puede reconstruir despues.
    """
    return {
        **fill_record(result),
        "symbol": position.symbol,
        "strategy_id": position.strategy_id,
        "close_reason": reason,
        "reference_price": reference_price,
        "decided_at": decided_at.isoformat(),
    }


def closed_record(position: Position) -> dict[str, Any]:
    return {
        "symbol": position.symbol,
        "side": position.side.value,
        "strategy_id": position.strategy_id,
        "size": position.size,
        "entry_price": position.entry_price,
        "exit_price": position.exit_price,
        "opened_at": position.opened_at.isoformat(),
        "closed_at": position.closed_at.isoformat() if position.closed_at else None,
        "close_reason": position.close_reason,
        # Las dos patas por separado: la de entrada ya quedo en el fill que abrio la
        # posicion, asi que sumar `fees_usd` sobre todo el diario la contaria dos veces.
        "entry_fees_usd": position.entry_fees_usd,
        "exit_fees_usd": position.exit_fees_usd,
        "fees_usd": position.total_fees_usd,
        "realized_pnl_usd": position.realized_pnl,
    }


def open_record(position: Position, mark_price: float | None) -> dict[str, Any]:
    return {
        "symbol": position.symbol,
        "side": position.side.value,
        "strategy_id": position.strategy_id,
        "size": position.size,
        "entry_price": position.entry_price,
        "mark_price": mark_price,
        "opened_at": position.opened_at.isoformat(),
        "stop_loss": position.stop_loss,
        "take_profit": position.take_profit,
        "fees_usd": position.entry_fees_usd,
        "unrealized_pnl_usd": (
            None if mark_price is None else round(position.net_pnl_at(mark_price), 6)
        ),
    }


def cycle_record(
    *,
    timestamp: datetime,
    status: str,
    symbols: Sequence[Mapping[str, Any]],
    opened: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
    exits: Sequence[Mapping[str, Any]],
    equity_usd: float | None,
    realized_pnl_usd: float,
    unrealized_pnl_usd: float | None,
    exposure_usd: float,
    open_positions: int,
    max_open_positions: int,
    daily_realized_pnl_usd: float,
    watchlist: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """
    La linea del diario. Un ciclo, entero.

    `equity_usd` es None en vivo y NO es un olvido: sin capital declarado el motor de
    riesgo dimensiona en absoluto (`runner.py::_build_portfolio_state`), y declarar un
    capital para poder dibujar una curva cambiaria el dimensionado a fraccion de equity.
    Lo que siempre hay es el PnL realizado acumulado y el no realizado marcado a mercado,
    que es la misma curva salvo una constante aditiva.

    `unrealized_pnl_usd` es None cuando el ciclo NO se marco a mercado —un ciclo pausado
    no toca la red— y entonces `net_pnl_usd` tambien lo es. Poner un cero ahi seria
    afirmar que las posiciones abiertas no se han movido, que es una afirmacion, no un
    dato que falta; y aplanaria la curva justo en los tramos en que el sistema estuvo
    parado.
    """
    marked = unrealized_pnl_usd is not None
    return {
        "timestamp": timestamp.isoformat(),
        "status": status,
        "symbols_evaluated": len(symbols),
        "symbols": list(symbols),
        "opened": list(opened),
        # `closed` es la CONTABILIDAD del cierre (precios, PnL neto, comisiones de las
        # dos patas) y `exits` es la EJECUCION de la orden que lo cerro (estado, precio
        # de llenado y deslizamiento realmente cobrado). Son dos cosas distintas, igual
        # que en la entrada lo son `opened` y los `fills` de cada simbolo: salir tambien
        # paga deslizamiento, y sin este bloque la mitad del coste no quedaria medida.
        "closed": list(closed),
        "exits": list(exits),
        "marked_to_market": marked,
        "equity_usd": equity_usd,
        "realized_pnl_usd": round(realized_pnl_usd, 6),
        "unrealized_pnl_usd": round(unrealized_pnl_usd, 6) if marked else None,
        "net_pnl_usd": round(realized_pnl_usd + unrealized_pnl_usd, 6) if marked else None,
        "exposure_usd": round(exposure_usd, 6),
        "open_positions": open_positions,
        "max_open_positions": max_open_positions,
        "daily_realized_pnl_usd": round(daily_realized_pnl_usd, 6),
        "watchlist": list(watchlist),
    }


# --- escritura -----------------------------------------------------------------------


class Journal(Protocol):
    """Lo unico que el runner le pide a un diario: poder anadir una linea, y decir si
    hay algo detras. Lo segundo existe para que el backtest no pague por construir una
    linea que nadie va a leer."""

    is_null: bool

    def append(self, record: Mapping[str, Any]) -> Any:
        ...


class NullJournal:
    """
    Diario que no escribe. Es el DEFECTO del runner, y a proposito.

    El backtest conduce el mismo runner miles de ciclos por ventana: un diario ahi no
    seria auditoria, seria ruido y E/S. Mismo papel que `InMemoryStateStore` frente a
    `JsonStateStore` y que `NullNotifier` frente a `TelegramNotifier`: el nucleo emite
    siempre, y quien lo construye decide si hay algo detras.
    """

    is_null = True

    def append(self, record: Mapping[str, Any]) -> None:
        return None

    def read(self) -> list[dict]:
        return []


class MemoryJournal:
    """
    Diario que guarda en una lista y no toca el disco.

    Existe para UNA cosa, y conviene que se lea porque es lo que hace medible la
    divergencia: cuando el motor de backtest re-simula el mismo periodo que corrio en
    vivo, se le engancha este diario y entonces las dos ejecuciones —la real y la
    simulada— emiten EXACTAMENTE el mismo esquema de linea. Comparar deja de ser
    traducir dos formatos y pasa a ser una funcion sobre dos listas de lo mismo.

    No sustituye a `NullJournal`, que sigue siendo el defecto del runner: el backtest
    normal corre miles de ciclos por ventana y acumularlos en memoria seria pagar por
    algo que nadie va a leer. Este se pide explicitamente.
    """

    is_null = False

    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(self, record: Mapping[str, Any]) -> dict:
        stored = dict(record)
        self.records.append(stored)
        return stored

    def read(self) -> list[dict]:
        return list(self.records)


class CycleJournal:
    """Registro append-only de ciclos, con rotacion por mes o por tamano."""

    is_null = False

    def __init__(
        self,
        path: str | Path = DEFAULT_JOURNAL_PATH,
        *,
        max_bytes: int = MAX_SHARD_BYTES,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        # Mes del fichero en curso, cacheado. Se resuelve leyendo su PRIMERA linea, no
        # el fichero entero, y se olvida al rotar. Tras un reinicio se vuelve a leer.
        self._month: str | None = None

    # --- escritura ---

    def append(self, record: Mapping[str, Any]) -> Path:
        """Anade una linea y la lleva a disco. Devuelve el fichero en el que quedo."""
        month = _month_of(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_if_needed(month)

        # Si la ultima escritura se corto a mitad, el fichero acaba SIN salto de linea.
        # Escribir detras pegaria el ciclo nuevo a la cola rota y se perderian los dos
        # registros en vez de uno. Cerrar la linea huerfana es la unica reparacion
        # compatible con append-only: no reescribe nada, solo termina lo que quedo abierto.
        prefix = "" if self._ends_clean() else "\n"

        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(prefix + to_line(record))
            handle.flush()
            os.fsync(handle.fileno())

        self._month = month if self._month is None else self._month
        return self.path

    def _ends_clean(self) -> bool:
        """True si el fichero no existe, esta vacio o termina en salto de linea."""
        if not self.path.exists():
            return True
        size = self.path.stat().st_size
        if size == 0:
            return True
        with open(self.path, "rb") as handle:
            handle.seek(size - 1)
            return handle.read(1) == b"\n"

    def _rotate_if_needed(self, month: str) -> None:
        if not self.path.exists():
            self._month = None
            return

        current = self._current_month()
        too_big = self.path.stat().st_size >= self.max_bytes
        new_month = current is not None and current != month
        if not (too_big or new_month):
            return

        target = self._next_shard(current or month)
        self.path.replace(target)
        self._month = None
        logger.info("Diario de ciclos rotado a %s", target.name)

    def _next_shard(self, month: str) -> Path:
        """Siguiente shard del mes. La secuencia va en el nombre para que el orden
        alfabetico sea el cronologico tambien cuando un mes se parte por tamano."""
        stem, suffix = self.path.stem, self.path.suffix
        seq = 1
        while True:
            candidate = self.path.parent / f"{stem}-{month}.{seq:03d}{suffix}"
            if not candidate.exists():
                return candidate
            seq += 1

    def _current_month(self) -> str | None:
        """Mes al que pertenece el fichero en curso, por su primera linea. None si aun
        no se puede saber: entonces solo rota por tamano, que es lo conservador."""
        if self._month is not None:
            return self._month
        try:
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    self._month = _month_of(json.loads(line))
                    return self._month
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("No se pudo leer el mes de %s; solo rotara por tamano", self.path)
        return None

    # --- lectura ---

    def shards(self) -> list[Path]:
        """Todos los ficheros del diario, del mas antiguo al mas reciente."""
        directory = self.path.parent
        if not directory.exists():
            return []
        stem, suffix = self.path.stem, self.path.suffix
        archived = sorted(directory.glob(f"{stem}-*{suffix}"))
        return [*archived, self.path] if self.path.exists() else archived

    def read(self) -> list[dict]:
        """Los ciclos archivados, en orden. Una linea rota se cuenta y se salta."""
        out: list[dict] = []
        for shard in self.shards():
            with open(shard, encoding="utf-8") as handle:
                records, broken = read_records(handle)
            if broken:
                logger.warning("%s lineas ilegibles en %s (se ignoran)", broken, shard)
            out.extend(records)
        return out


def _month_of(record: Mapping[str, Any]) -> str:
    """`YYYY-MM` del ciclo. El diario escribe su propia marca de tiempo con
    `datetime.isoformat()`, asi que aqui basta un corte: no hay formatos de proveedor
    que adivinar (eso es lo que hace `signals/store.py::month_of`, y por eso no se
    reutiliza: esta parametrizada sobre las claves del registro crudo de senales)."""
    stamp = str(record.get("timestamp") or "")
    if len(stamp) >= 7 and stamp[4] == "-":
        return stamp[:7]
    raise ValueError(f"Ciclo sin marca de tiempo utilizable: {dict(record)!r}")


# --- derivadas (las calcula quien lee, nunca quien escribe) --------------------------


def pnl_curve(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """
    Curva marcada a mercado, un punto por ciclo: PnL neto, equity (si la hay) y
    exposicion desplegada.

    Los ciclos SIN marcar (pausados) se quedan fuera: son puntos de liveness, no de
    valoracion, y meterlos con un cero aplanaria la curva mientras el sistema no opera.
    """
    return [
        {
            "timestamp": r.get("timestamp"),
            "net_pnl_usd": float(r["net_pnl_usd"]),
            "realized_pnl_usd": float(r.get("realized_pnl_usd") or 0.0),
            "equity_usd": r.get("equity_usd"),
            "exposure_usd": float(r.get("exposure_usd") or 0.0),
            "open_positions": int(r.get("open_positions") or 0),
        }
        for r in records
        if r.get("net_pnl_usd") is not None
    ]


def max_drawdown_usd(curve: Sequence[Mapping[str, Any]]) -> float:
    """
    Maxima caida desde un pico, en USD. Positivo.

    NO es `backtest/metrics.py::max_drawdown_pct` con otras unidades, y no puede serlo:
    aquel divide por el pico, y en vivo la curva es de PnL —arranca en cero y puede ser
    negativa—, asi que el cociente no esta definido. El porcentaje se publica solo
    cuando hay `equity_usd`, y entonces SI se calcula con la funcion de metrics.
    """
    peak = -float("inf")
    worst = 0.0
    for point in curve:
        value = float(point.get("net_pnl_usd") or 0.0)
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def journal_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Que ha hecho el sistema desde que corre. Todo derivado de lo archivado.

    Devuelve `{"n_cycles": 0, ...}` con un diario vacio en vez de reventar: la vista del
    dashboard tiene que poder decir "sin ciclos registrados" y no romperse.
    """
    curve = pnl_curve(records)
    rejections: dict[str, int] = {}
    n_signals = n_approved = n_rejected = n_fills = n_exit_fills = 0
    fees = 0.0
    slippages: list[float] = []

    for record in records:
        for symbol in record.get("symbols", []):
            n_signals += len(symbol.get("signals", []))
            for decision in symbol.get("risk", []):
                if decision.get("approved"):
                    n_approved += 1
                else:
                    n_rejected += 1
                    # La familia del rechazo es lo que va antes de los dos puntos; la
                    # cifra concreta queda en la linea archivada.
                    family = str(decision.get("reason", "")).split(":")[0].strip()
                    rejections[family] = rejections.get(family, 0) + 1
            for fill in symbol.get("fills", []):
                n_fills += 1
                fees += float(fill.get("fees_usd") or 0.0)
                if fill.get("success"):
                    slippages.append(float(fill.get("slippage_bps") or 0.0))
        for closed in record.get("closed", []):
            # Solo la pata de salida: la de entrada ya se conto en el fill que abrio.
            fees += float(closed.get("exit_fees_usd") or 0.0)
        for exit_fill in record.get("exits", []):
            n_exit_fills += 1
            if exit_fill.get("success"):
                slippages.append(float(exit_fill.get("slippage_bps") or 0.0))

    equities = [p["equity_usd"] for p in curve if p["equity_usd"] is not None]
    drawdown_pct = None
    if equities and len(equities) == len(curve):
        drawdown_pct = max_drawdown_pct(
            [
                EquityPoint(day=datetime.fromisoformat(p["timestamp"]), equity=float(p["equity_usd"]))
                for p in curve
            ]
        )

    last = curve[-1] if curve else {}
    return {
        # Todos los ciclos archivados, incluidos los pausados: es el recuento que dice
        # si el proceso ha estado vivo. La curva solo lleva los marcados a mercado.
        "n_cycles": len(records),
        "n_marked_cycles": len(curve),
        "n_paused_cycles": sum(1 for r in records if r.get("status") == STATUS_PAUSED),
        "first_cycle": records[0].get("timestamp") if records else None,
        "last_cycle": records[-1].get("timestamp") if records else None,
        "n_signals": n_signals,
        "n_approved": n_approved,
        "n_rejected": n_rejected,
        # Ordenes de ENTRADA y de SALIDA por separado: el embudo de decisiones acaba en
        # las de entrada, pero el coste de ejecucion lo pagan las dos.
        "n_fills": n_fills,
        "n_exit_fills": n_exit_fills,
        "rejections": dict(sorted(rejections.items(), key=lambda kv: -kv[1])),
        "fees_usd": round(fees, 6),
        "slippage_bps_mean": (
            round(sum(slippages) / len(slippages), 3) if slippages else None
        ),
        "slippage_bps_max": round(max(slippages), 3) if slippages else None,
        "net_pnl_usd": last.get("net_pnl_usd", 0.0),
        "realized_pnl_usd": last.get("realized_pnl_usd", 0.0),
        "exposure_usd": last.get("exposure_usd", 0.0),
        "open_positions": last.get("open_positions", 0),
        "equity_usd": last.get("equity_usd"),
        "max_drawdown_usd": round(max_drawdown_usd(curve), 6),
        "max_drawdown_pct": None if drawdown_pct is None else round(drawdown_pct, 4),
        "curve": curve,
    }


__all__ = [
    "DEFAULT_JOURNAL_PATH",
    "MAX_SHARD_BYTES",
    "STATUS_DISABLED",
    "STATUS_PAUSED",
    "STATUS_RAN",
    "CycleJournal",
    "Journal",
    "MemoryJournal",
    "NullJournal",
    "closed_record",
    "cycle_record",
    "exit_record",
    "fill_record",
    "journal_summary",
    "max_drawdown_usd",
    "open_record",
    "order_record",
    "pnl_curve",
    "risk_record",
    "signal_record",
]
