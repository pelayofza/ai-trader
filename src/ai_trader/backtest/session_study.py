r"""
Descomposicion por sesion horaria: donde se forma el precio DENTRO de la barra diaria.

El sistema entero decide con barras 1D ya cerradas y llena al OPEN del dia siguiente
(`execution/market_model.py::IntrabarMarketModel.entry_reference_price`), y comprueba
stops contra el HIGH/LOW de esa barra. Toda esa convencion trata las 24 horas de la vela
como un bloque opaco. Nadie habia medido nunca que fraccion de la formacion de precio
ocurre en cada tramo de ese bloque, ni -sobre todo- cuanto vale el hueco entre el cierre
que la estrategia VE y el open al que se LLENA.

    .venv\Scripts\python.exe -m ai_trader.backtest.session_study
    .venv\Scripts\python.exe -m ai_trader.backtest.session_study --offline
    .venv\Scripts\python.exe -m ai_trader.backtest.session_study --verify-determinism

Salida: data/sessions/report.json

Cinco decisiones que son las que hacen que las cifras signifiquen algo:

1. LOS CORTES DE SESION SE RAZONAN, NO SE ELIGEN A OJO. Cada frontera es la hora UTC de
   una apertura de mercado real, tomada en su version MAS TEMPRANA a lo largo del ano
   (ver `SESSIONS`). Y no se desplazan con el horario de verano: la rejilla de velas de
   Binance es UTC fija, y mover los cortes dos veces al ano haria que la serie de cuotas
   midiera el calendario civil ademas del mercado.

2. LAS SESIONES NO DURAN LO MISMO, Y SE DICE. Cortar el dia por aperturas reales da 7 / 6
   / 11 horas, no tres bloques de 8. Por eso toda cuota se publica junto a su INTENSIDAD
   (cuota dividida por la fraccion de reloj que ocupa la sesion): 1,0 es neutro. Comparar
   cuotas brutas de tramos de distinta longitud seria comparar duraciones.

3. SOLO DIAS COMPLETOS Y ENCADENADOS. Un dia entra si tiene sus 24 barras horarias (horas
   0..23) y ademas existe la barra de las 23:00 del dia anterior. Lo segundo no es
   cosmetico: sin ella no hay retorno para la hora 0 y la sesion asiatica saldria
   sistematicamente infravalorada en la descomposicion. Los dias caidos se declaran.

4. LA TENDENCIA SE MIDE SOBRE UN PANEL EQUILIBRADO. La mitad del universo operable no
   cotizaba en 2020. Si la cuota anual se promediara sobre "los simbolos que haya", la
   serie temporal mediria la COMPOSICION del universo, no el desplazamiento de la
   actividad. La tendencia se mide sobre la cohorte de pares presentes en TODOS los anos
   de la ventana; el resto se publica igual, pero fuera del contraste.

5. VENTANA HISTORICA CERRADA. `[2020-01-01, 2026-01-01)`: seis anos naturales completos,
   fijos. Si el final se moviera con la fecha de ejecucion, dos regeneraciones del informe
   no serian comparables y la palabra "determinista" no significaria nada.

REGLAS DE DECISION, declaradas en el codigo antes de mirar el resultado:
  hueco  >= GAP_MATERIAL_SHARE del rango diario  -> el llenado al open NO es inocuo y hay
           que declararlo como limitacion del motor (y considerar modelarlo).
  hueco  <  GAP_MATERIAL_SHARE                   -> la convencion es exacta en un mercado
           24/7; lo que queda sin modelar es OTRA cosa (la latencia de ejecucion), y se
           mide aparte.
  cuota US post-2024 - pre-2024 >= US_SHARE_GROWTH_MIN y test de signos significativo
           -> la hipotesis de desplazamiento hacia la sesion estadounidense se sostiene.

Determinismo: no hay muestreo ni semillas en ninguna parte. La ventana es cerrada, el
universo sale del config, los cortes son constantes y toda la aritmetica es una reduccion
sobre las mismas barras. `--verify-determinism` recalcula el informe entero una segunda
vez y exige igualdad campo a campo.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.config import AppConfig, load_config
from ai_trader.data.intraday import get_hourly_bars, slice_window, to_utc
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.instruments import AssetClass, detect_asset_class

logger = logging.getLogger("session_study")

OUT_DIR = Path("data") / "sessions"
SESSIONS_REPORT = OUT_DIR / "report.json"

DEFAULT_CONFIG = Path("config") / "default.toml"
DEFAULT_EXCHANGE = "binance"

# Ventana CERRADA (ver docstring): seis anos naturales completos. Arranca en 2020 y no
# antes porque la mitad del universo operable no existia, y termina en el mismo corte
# fijo que ya usan los estudios de fidelidad y transferencia.
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-01-01"

HOURS_PER_DAY = 24

# Un simbolo-ano entra en la tabla anual si tiene al menos esto de dias utilizables. Por
# debajo, la cuota anual la fijarian cuatro semanas sueltas del ano.
MIN_DAYS_PER_YEAR = 240

# El corte de la hipotesis. No es una fecha comoda: los ETF al contado de bitcoin se
# aprobaron en EE.UU. el 10 de enero de 2024 y empezaron a cotizar el 11. La hipotesis
# declarada es que a partir de ahi el peso de la sesion estadounidense crece. El corte se
# pone en el 1 de enero para que coincida con la frontera de ano de la tabla y no haya que
# partir 2024 por la mitad.
TREND_SPLIT = "2024-01-01"
TREND_MECHANISM = (
    "ETF al contado de bitcoin aprobados en EE.UU. el 10-01-2024 (cotizando desde el 11)"
)

# --- reglas de decision (declaradas antes de mirar el resultado) --------------------
# Cuanto tiene que valer el hueco cierre-visto -> open-llenado, en fraccion del rango
# diario, para que la convencion de llenado deje de ser inocua. 5% del rango de un dia es
# del orden del coste total de ejecucion que el motor ya cobra: por debajo de eso, el
# hueco esta dentro del ruido que el modelo de microestructura ya absorbe.
GAP_MATERIAL_SHARE = 0.05
# Cuanto tiene que desplazarse el precio con UNA hora de retraso, en fraccion del rango
# diario, para que "se llena al open" deje de poder tratarse como instantaneo.
LATENCY_MATERIAL_SHARE = 0.10
# Crecimiento minimo de la cuota estadounidense (en fraccion, no en puntos porcentuales
# relativos) para dar por sostenida la hipotesis.
US_SHARE_GROWTH_MIN = 0.02
# Correlacion de rangos ano-cuota sobre los anos ANTERIORES al corte por encima de la cual
# se considera que la cuota YA venia subiendo. Importa porque un contraste pre/post no
# distingue "sube por el mecanismo" de "sube desde antes y el corte pilla la pendiente": si
# la deriva previa ya esta ahi, la direccion de la hipotesis puede sostenerse y su
# mecanismo no.
PRE_TREND_RHO = 0.60

# Retrasos de ejecucion a medir, en horas despues del open al que el backtest llena.
LATENCY_HOURS = (1, 2, 4, 8)

VERDICT_GAP_MATERIAL = "hueco_material"
VERDICT_GAP_NEGLIGIBLE = "hueco_inmaterial"
VERDICT_TREND_UP = "us_crece"
VERDICT_TREND_FLAT = "us_estable"
VERDICT_TREND_DOWN = "us_decrece"

# Forma del cambio, que es una pregunta distinta de si lo hay.
SHAPE_JUMP = "salto_en_el_corte"
SHAPE_DRIFT = "deriva_previa"
SHAPE_MIXED = "mixta"


# ------------------------------------------------------------------ sesiones ---------


@dataclass(frozen=True, slots=True)
class Session:
    """Un tramo del dia UTC. `end_hour` es EXCLUSIVO."""

    key: str
    label: str
    start_hour: int
    end_hour: int
    rationale: str

    @property
    def hours(self) -> int:
        return self.end_hour - self.start_hour

    @property
    def clock_share(self) -> float:
        """Fraccion del dia que ocupa. Es el liston contra el que se lee su cuota."""
        return self.hours / HOURS_PER_DAY

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "hours": self.hours,
            "clock_share": round(self.clock_share, 4),
            "rationale": self.rationale,
        }


# Los cortes. Cada frontera es la hora UTC de una apertura real, tomada en su version MAS
# TEMPRANA del ano (los mercados de referencia cambian de hora UTC con el horario de
# verano; la rejilla de velas de Binance no). Se elige la mas temprana y no la media para
# que una sesion nunca contenga el arranque de la siguiente: es preferible que el tramo
# europeo se coma una hora tranquila de Asia a que se coma la apertura de Wall Street.
SESSIONS: tuple[Session, ...] = (
    Session(
        key="asia",
        label="Asiática",
        start_hour=0,
        end_hour=7,
        rationale=(
            "00:00 UTC es a la vez la frontera de la vela diaria de Binance y la apertura "
            "del contado de Tokio (09:00 JST). Termina a las 07:00 UTC, la primera hora en "
            "que ya puede haber abierto un mercado europeo: es el tramo del dia sin "
            "presencia occidental."
        ),
    ),
    Session(
        key="europe",
        label="Europea",
        start_hour=7,
        end_hour=13,
        rationale=(
            "07:00 UTC es la apertura mas temprana del contado europeo a lo largo del ano "
            "(Londres y Frankfurt abren a las 07:00 UTC en horario de verano y a las 08:00 "
            "en invierno). Termina a las 13:00 UTC, la ultima hora entera libre de flujo "
            "estadounidense: la sesion de Nueva York abre a las 13:30 UTC en horario de "
            "verano."
        ),
    ),
    Session(
        key="us",
        label="Estadounidense",
        start_hour=13,
        end_hour=24,
        rationale=(
            "Desde la apertura mas temprana de EE.UU. (13:30 UTC en horario de verano; el "
            "premercado y los futuros de CME ya estan activos desde las 13:00) hasta la "
            "frontera de la vela diaria. Incluye el cierre del contado (20:00 / 21:00 UTC "
            "segun estacion) y la liquidacion diaria de CME de las 22:00 UTC. Es el tramo "
            "mas largo porque es el unico que ademas contiene la noche americana, que no "
            "tiene mercado de referencia propio."
        ),
    ),
)

SESSION_KEYS: tuple[str, ...] = tuple(s.key for s in SESSIONS)
SESSION_BY_KEY: dict[str, Session] = {s.key: s for s in SESSIONS}
US_KEY = "us"


def _hour_to_session() -> np.ndarray:
    """Vector de 24 posiciones: hora UTC -> indice de sesion. Valida la particion."""
    if SESSIONS[0].start_hour != 0 or SESSIONS[-1].end_hour != HOURS_PER_DAY:
        raise ValueError("Las sesiones deben cubrir el dia UTC entero")
    mapping = np.full(HOURS_PER_DAY, -1, dtype=int)
    for i, session in enumerate(SESSIONS):
        if session.hours <= 0:
            raise ValueError(f"Sesion vacia o invertida: {session.key}")
        if i and session.start_hour != SESSIONS[i - 1].end_hour:
            raise ValueError(f"Sesiones no contiguas en {session.key}")
        mapping[session.start_hour : session.end_hour] = i
    if (mapping < 0).any():
        raise ValueError("Hay horas del dia sin sesion asignada")
    return mapping


HOUR_TO_SESSION: np.ndarray = _hour_to_session()
SESSION_MASKS: tuple[np.ndarray, ...] = tuple(
    HOUR_TO_SESSION == i for i in range(len(SESSIONS))
)


def session_of_hour(hour: int) -> str:
    """La sesion a la que pertenece una hora UTC."""
    if not 0 <= hour < HOURS_PER_DAY:
        raise ValueError(f"Hora fuera de rango: {hour}")
    return SESSION_KEYS[int(HOUR_TO_SESSION[hour])]


# -------------------------------------------------------------- matriz diaria --------


@dataclass(frozen=True, slots=True)
class DayMatrix:
    """
    Las barras 1H de un simbolo reorganizadas en dias UTC COMPLETOS y encadenados.

    Todo lo demas del estudio es una reduccion sobre esto. `open/high/low/close` son
    matrices (n_dias x 24) y `prev_close` el vector con el cierre de las 23:00 del dia
    anterior, que es literalmente el precio que la estrategia VE cuando decide.
    """

    days: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ret: np.ndarray  # log-retorno hora a hora, encadenado con el dia anterior
    prev_close: np.ndarray
    n_bars: int
    n_days_dropped: int

    @property
    def n_days(self) -> int:
        return len(self.days)


def _empty_matrix() -> DayMatrix:
    empty = np.empty((0, HOURS_PER_DAY), dtype=float)
    return DayMatrix(
        days=pd.DatetimeIndex([], tz="UTC", name="day"),
        open=empty, high=empty, low=empty, close=empty, ret=empty,
        prev_close=np.empty(0, dtype=float),
        n_bars=0,
        n_days_dropped=0,
    )


def build_day_matrix(hourly: pd.DataFrame) -> DayMatrix:
    """
    Reorganiza barras 1H en dias UTC utilizables.

    Un dia es utilizable si tiene sus 24 barras (horas 0..23) Y los 24 log-retornos son
    calculables, lo que exige que exista la barra de las 23:00 del dia anterior. La segunda
    condicion no es cosmetica: sin ella el retorno de la hora 0 seria NaN y la sesion
    asiatica -que empieza justo ahi- saldria infravalorada de forma sistematica en la
    descomposicion de |retorno| y de varianza.
    """
    frame = bar_schema.normalize_bars(hourly)
    if frame.empty:
        return _empty_matrix()

    # Fuera las barras que no caen en la rejilla horaria exacta: una vela con minutos
    # sueltos no es una hora del dia y contarla como tal desalinearia el reshape.
    on_grid = frame.index == frame.index.floor("h")
    frame = frame[on_grid]

    closes = bar_schema.series(frame, bar_schema.CLOSE)
    prev_close = closes.shift(1)
    contiguous = frame.index.to_series().diff() == pd.Timedelta(hours=1)
    log_ret = np.log(closes / prev_close).where(contiguous)

    work = pd.DataFrame(
        {
            "day": frame.index.normalize(),
            "open": bar_schema.series(frame, bar_schema.OPEN),
            "high": bar_schema.series(frame, bar_schema.HIGH),
            "low": bar_schema.series(frame, bar_schema.LOW),
            "close": closes,
            "ret": log_ret,
            "prev_close": prev_close,
        },
        index=frame.index,
    )
    work["ok"] = work[["open", "high", "low", "close", "ret", "prev_close"]].notna().all(axis=1)

    grouped = work.groupby("day", sort=True)
    counts = grouped.size()
    all_ok = grouped["ok"].all()
    # 24 barras distintas dentro del dia implica horas 0..23 exactas: el indice viene ya
    # deduplicado por `normalize_bars`, asi que no puede haber dos veces la misma hora.
    usable = counts.index[(counts == HOURS_PER_DAY) & all_ok.reindex(counts.index, fill_value=False)]

    selected = work[work["day"].isin(usable)].sort_index()
    n = len(usable)

    def matrix(column: str) -> np.ndarray:
        return selected[column].to_numpy(dtype=float).reshape(n, HOURS_PER_DAY)

    return DayMatrix(
        days=pd.DatetimeIndex(usable, name="day"),
        open=matrix("open"),
        high=matrix("high"),
        low=matrix("low"),
        close=matrix("close"),
        ret=matrix("ret"),
        prev_close=matrix("prev_close")[:, 0],
        n_bars=int(len(frame)),
        n_days_dropped=int(len(counts) - n),
    )


# ------------------------------------------------------------ tabla por dia ----------


def daily_table(matrix: DayMatrix) -> pd.DataFrame:
    """
    Una fila por dia utilizable con TODO lo que el estudio necesita medir.

    Las columnas por sesion son sumas y rangos crudos, no cuotas: las cuotas se calculan
    despues, al agregar, para que se pueda elegir el denominador (ano, cohorte, ventana
    entera) sin volver a recorrer las barras.
    """
    if matrix.n_days == 0:
        return pd.DataFrame()

    high, low, open_, close, ret = matrix.high, matrix.low, matrix.open, matrix.close, matrix.ret
    day_high = high.max(axis=1)
    day_low = low.min(axis=1)
    day_open = open_[:, 0]
    day_close = close[:, -1]
    day_range = day_high - day_low

    data: dict[str, np.ndarray] = {
        "year": matrix.days.year.to_numpy(),
        "open": day_open,
        "high": day_high,
        "low": day_low,
        "close": day_close,
        "prev_close": matrix.prev_close,
        "range": day_range,
    }

    abs_ret = np.abs(ret)
    sq_ret = ret**2
    for key, mask in zip(SESSION_KEYS, SESSION_MASKS):
        data[f"abs_{key}"] = abs_ret[:, mask].sum(axis=1)
        data[f"var_{key}"] = sq_ret[:, mask].sum(axis=1)
        data[f"rng_{key}"] = high[:, mask].max(axis=1) - low[:, mask].min(axis=1)

    # Que sesion FIJA el extremo del dia. Es lo que decide si la convencion pesimista de
    # `_intrabar_exit` (si se tocan stop y objetivo en la misma barra, gana el stop) muerde
    # de forma simetrica o concentrada en un tramo.
    data["hi_session"] = HOUR_TO_SESSION[high.argmax(axis=1)]
    data["lo_session"] = HOUR_TO_SESSION[low.argmin(axis=1)]

    # LA CIFRA QUE DECIDE: el hueco entre el cierre que la estrategia ve (cierre de las
    # 23:00 de ayer, que es el cierre de la vela 1D con la que decide) y el open al que el
    # motor llena (open de las 00:00 de hoy, que es el open de la vela 1D de hoy).
    data["gap"] = day_open - matrix.prev_close
    data["daily_move"] = np.abs(day_close - matrix.prev_close)

    # Latencia de ejecucion: si el llenado no ocurre exactamente en el open sino k horas
    # despues, a que precio se llena y cuanto camino ha recorrido ya el precio.
    for k in LATENCY_HOURS:
        if k >= HOURS_PER_DAY:
            continue
        data[f"slip_{k}h"] = np.abs(open_[:, k] - day_open)
        data[f"path_{k}h"] = high[:, :k].max(axis=1) - low[:, :k].min(axis=1)

    return pd.DataFrame(data, index=matrix.days)


# ------------------------------------------------------------- agregaciones ----------


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Ratio elemento a elemento con NaN donde el denominador es cero (un dia plano)."""
    out = np.full(len(numerator), np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _median(values: np.ndarray) -> float | None:
    clean = values[np.isfinite(values)]
    return None if clean.size == 0 else float(np.median(clean))


def _pct(values: np.ndarray, q: float) -> float | None:
    clean = values[np.isfinite(values)]
    return None if clean.size == 0 else float(np.percentile(clean, q))


def _mean(values: np.ndarray) -> float | None:
    clean = values[np.isfinite(values)]
    return None if clean.size == 0 else float(np.mean(clean))


def session_shares(table: pd.DataFrame) -> dict:
    """
    La descomposicion por sesion de un bloque de dias (un ano, una cohorte, la ventana).

    Tres cuotas, y cada una responde a una pregunta distinta:

    - `abs_return`: cuota de |retorno| acumulado. Cuanto CAMINO recorre el precio en el
      tramo. Es un ratio de sumas porque la magnitud a repartir es la suma.
    - `variance`: cuota de varianza realizada (suma de r^2). Es la descomposicion natural
      de la volatilidad: las varianzas suman, las desviaciones tipicas no. Pondera mas las
      horas violentas, y por eso puede separarse mucho de la anterior.
    - `range`: cuota del rango. Aqui SI es media de ratios diarios y no ratio de sumas,
      porque el rango va en unidades de precio y un ano de BTC a 20.000 y a 100.000 mezcla
      escalas; el ratio diario es adimensional por construccion.

    Ademas se publica `range_vs_daily`: cuanto del rango del DIA cubre el tramo. Esa no
    suma 1 (suma mas), y es a proposito: mide solape, no reparto.
    """
    if table.empty:
        return {}

    abs_total = float(sum(table[f"abs_{k}"].sum() for k in SESSION_KEYS))
    var_total = float(sum(table[f"var_{k}"].sum() for k in SESSION_KEYS))
    rng_sum = sum(table[f"rng_{k}"].to_numpy() for k in SESSION_KEYS)
    day_range = table["range"].to_numpy()

    out: dict[str, dict] = {}
    for key in SESSION_KEYS:
        session = SESSION_BY_KEY[key]
        abs_share = float(table[f"abs_{key}"].sum() / abs_total) if abs_total > 0 else None
        var_share = float(table[f"var_{key}"].sum() / var_total) if var_total > 0 else None
        rng = table[f"rng_{key}"].to_numpy()
        rng_share = _mean(_safe_ratio(rng, rng_sum))
        out[key] = {
            "abs_return": _round(abs_share),
            "variance": _round(var_share),
            "range": _round(rng_share),
            "range_vs_daily": _round(_mean(_safe_ratio(rng, day_range))),
            # Intensidad: cuota dividida por la fraccion de reloj. 1,0 = el tramo aporta
            # justo lo que le toca por durar lo que dura. Es la unica lectura honesta
            # cuando las sesiones no miden lo mismo.
            "abs_intensity": _round(
                None if abs_share is None else abs_share / session.clock_share, 3
            ),
            "variance_intensity": _round(
                None if var_share is None else var_share / session.clock_share, 3
            ),
            "sets_high": _round(float((table["hi_session"] == SESSION_KEYS.index(key)).mean())),
            "sets_low": _round(float((table["lo_session"] == SESSION_KEYS.index(key)).mean())),
        }
    return out


def _round(value, decimals: int = 4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), decimals)


def gap_stats(table: pd.DataFrame) -> dict:
    """
    LA CIFRA QUE DECIDE: el hueco entre el cierre visto y el open llenado.

    Tres denominadores porque tres preguntas:
      - contra el RANGO del dia: cuanto del recorrido del dia se pierde el motor.
      - contra el MOVIMIENTO del dia (cierre a cierre): cuanto del resultado se decide ahi.
      - en puntos basicos: cuanto cuesta en dinero, comparable con el coste de ejecucion.
    """
    if table.empty:
        return {"n_days": 0}

    gap = np.abs(table["gap"].to_numpy())
    share_range = _safe_ratio(gap, table["range"].to_numpy())
    share_move = _safe_ratio(gap, table["daily_move"].to_numpy())
    bps = _safe_ratio(gap * 1e4, table["prev_close"].to_numpy())
    measurable = share_range[np.isfinite(share_range)]

    return {
        "n_days": int(len(table)),
        "share_of_range": {
            "median": _round(_median(share_range), 6),
            "mean": _round(_mean(share_range), 6),
            "p90": _round(_pct(share_range, 90), 6),
            "p99": _round(_pct(share_range, 99), 6),
        },
        "share_of_daily_move": {
            "median": _round(_median(share_move), 6),
            "mean": _round(_mean(share_move), 6),
            "p90": _round(_pct(share_move, 90), 6),
        },
        "bps": {
            "median": _round(_median(bps), 4),
            "mean": _round(_mean(bps), 4),
            "p90": _round(_pct(bps, 90), 4),
            "p99": _round(_pct(bps, 99), 4),
        },
        # Dias en los que el hueco llega a ser material. Es la cola, no la mediana, la que
        # diria que el motor tiene un problema puntual aunque de media no lo tenga.
        "days_above_threshold_pct": _round(
            100.0 * float((measurable >= GAP_MATERIAL_SHARE).mean()) if measurable.size else None,
            3,
        ),
        "n_days_measurable": int(measurable.size),
        "threshold": GAP_MATERIAL_SHARE,
    }


def latency_stats(table: pd.DataFrame) -> dict:
    """
    Lo que el hueco de arriba NO mide: que pasa si el llenado no es instantaneo.

    El backtest llena al open de las 00:00 UTC. Eso solo es exacto si la orden se manda en
    ese instante. Cada fila responde: si se llena k horas tarde, cuanto se ha desplazado ya
    el precio (`slip`) y cuanto del rango del dia se ha gastado antes de entrar (`path`).
    """
    if table.empty:
        return {"n_days": 0, "rows": []}

    day_range = table["range"].to_numpy()
    rows = []
    for k in LATENCY_HOURS:
        column = f"slip_{k}h"
        if column not in table.columns:
            continue
        slip = _safe_ratio(table[column].to_numpy(), day_range)
        path = _safe_ratio(table[f"path_{k}h"].to_numpy(), day_range)
        bps = _safe_ratio(table[column].to_numpy() * 1e4, table["open"].to_numpy())
        rows.append(
            {
                "hours": k,
                "session": session_of_hour(k % HOURS_PER_DAY),
                "slip_share_of_range_median": _round(_median(slip), 4),
                "slip_share_of_range_p90": _round(_pct(slip, 90), 4),
                "slip_bps_median": _round(_median(bps), 2),
                "path_share_of_range_median": _round(_median(path), 4),
            }
        )
    return {"n_days": int(len(table)), "threshold": LATENCY_MATERIAL_SHARE, "rows": rows}


# ---------------------------------------------------------------- tendencia ----------


def _binomial_two_sided(successes: int, n: int) -> float:
    """
    Test de signos exacto (p = 0,5), a dos colas. Sin scipy y sin aproximacion normal:
    la cohorte tiene una decena larga de simbolos y ahi la normal ya miente.
    """
    if n == 0:
        return 1.0
    tail = min(successes, n - successes)
    cumulative = sum(math.comb(n, i) for i in range(tail + 1)) / (2.0**n)
    return min(1.0, 2.0 * cumulative)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Correlacion de rangos. Devuelve None si no hay varianza que correlacionar."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.size < 3:
        return None
    rx = pd.Series(xa).rank().to_numpy()
    ry = pd.Series(ya).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def trend_analysis(
    tables: dict[str, pd.DataFrame],
    cohort: Sequence[str],
    years: Sequence[int],
) -> dict:
    """
    La hipotesis declarada: el peso de la sesion estadounidense crece tras enero de 2024.

    Se contrasta de dos formas independientes y las dos se publican, aunque no coincidan:

    - PAREADA PRE/POST. Por simbolo de la cohorte, cuota estadounidense de varianza antes
      y desde el corte. La comparacion es pareada (cada simbolo consigo mismo), asi que el
      nivel de cada par -que varia mucho- se cancela. El contraste es un test de SIGNOS
      exacto, no una t: con una decena de simbolos correlacionados entre si, una t
      fingiria una precision que la muestra no tiene. El estadistico t se publica como
      descriptivo, no como prueba.

    - SERIE ANUAL. Cuota estadounidense agregada ano a ano sobre la MISMA cohorte, y su
      correlacion de rangos con el ano. Responde a algo distinto: si el desplazamiento es
      un salto en 2024 o una deriva que ya venia de antes. Un salto sin deriva y una deriva
      sin salto tienen implicaciones opuestas para el mecanismo.
    """
    split = to_utc(TREND_SPLIT)
    per_symbol = []
    for symbol in cohort:
        table = tables[symbol]
        pre, post = table[table.index < split], table[table.index >= split]
        if pre.empty or post.empty:
            continue
        pre_share = session_shares(pre).get(US_KEY, {})
        post_share = session_shares(post).get(US_KEY, {})
        if pre_share.get("variance") is None or post_share.get("variance") is None:
            continue
        per_symbol.append(
            {
                "symbol": symbol,
                "n_days_pre": int(len(pre)),
                "n_days_post": int(len(post)),
                "us_variance_pre": pre_share["variance"],
                "us_variance_post": post_share["variance"],
                "delta_variance": _round(post_share["variance"] - pre_share["variance"]),
                "us_abs_pre": pre_share["abs_return"],
                "us_abs_post": post_share["abs_return"],
                "delta_abs": _round(post_share["abs_return"] - pre_share["abs_return"]),
            }
        )

    deltas = np.array([row["delta_variance"] for row in per_symbol], dtype=float)
    n = int(deltas.size)
    positives = int((deltas > 0).sum())
    mean_delta = float(deltas.mean()) if n else float("nan")
    sd = float(deltas.std(ddof=1)) if n > 1 else float("nan")
    t_stat = mean_delta / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    p_sign = _binomial_two_sided(positives, n)

    yearly = []
    for year in years:
        blocks = [t[t["year"] == year] for t in (tables[s] for s in cohort)]
        blocks = [b for b in blocks if not b.empty]
        if not blocks:
            continue
        merged = pd.concat(blocks)
        shares = session_shares(merged)
        yearly.append(
            {
                "year": int(year),
                "n_days": int(len(merged)),
                "n_symbols": len(blocks),
                **{
                    key: {
                        "abs_return": shares[key]["abs_return"],
                        "variance": shares[key]["variance"],
                        "range": shares[key]["range"],
                    }
                    for key in SESSION_KEYS
                },
            }
        )

    rho = _spearman(
        [row["year"] for row in yearly], [row[US_KEY]["variance"] for row in yearly]
    )

    # FORMA del cambio. Un contraste pre/post no distingue "sube por el mecanismo" de
    # "lleva anos subiendo y el corte pilla la pendiente". Se separan aqui: la pendiente
    # sobre los anos ANTERIORES al corte, y el escalon del ultimo ano previo al primero
    # posterior. Si la pendiente previa ya esta, el mecanismo no puede reclamar el credito
    # aunque la direccion de la hipotesis se sostenga.
    split_year = int(pd.Timestamp(TREND_SPLIT).year)
    before = [row for row in yearly if row["year"] < split_year]
    after = [row for row in yearly if row["year"] >= split_year]
    pre_rho = _spearman(
        [row["year"] for row in before], [row[US_KEY]["variance"] for row in before]
    )
    step = (
        after[0][US_KEY]["variance"] - before[-1][US_KEY]["variance"]
        if before and after
        else None
    )
    pre_drift = pre_rho is not None and pre_rho >= PRE_TREND_RHO
    step_up = step is not None and step >= 0
    if pre_drift and not step_up:
        shape = SHAPE_DRIFT
    elif not pre_drift and step_up:
        shape = SHAPE_JUMP
    else:
        shape = SHAPE_MIXED

    if n and mean_delta >= US_SHARE_GROWTH_MIN and p_sign < 0.05:
        verdict = VERDICT_TREND_UP
    elif n and mean_delta <= -US_SHARE_GROWTH_MIN and p_sign < 0.05:
        verdict = VERDICT_TREND_DOWN
    else:
        verdict = VERDICT_TREND_FLAT

    return {
        "hypothesis": (
            "el peso de la sesion estadounidense crece a partir de enero de 2024"
        ),
        "mechanism": TREND_MECHANISM,
        "split": TREND_SPLIT,
        "threshold": US_SHARE_GROWTH_MIN,
        "cohort": list(cohort),
        "n_symbols": n,
        "mean_delta_variance": _round(mean_delta),
        "sd_delta_variance": _round(sd),
        "t_stat": _round(t_stat, 3),
        "n_positive": positives,
        "sign_test_p": _round(p_sign, 4),
        "spearman_year_vs_us_variance": _round(rho, 3),
        "shape": shape,
        "pre_split_spearman": _round(pre_rho, 3),
        "pre_split_years": [row["year"] for row in before],
        "step_at_split": _round(step),
        "pre_trend_rho_threshold": PRE_TREND_RHO,
        "yearly": yearly,
        "per_symbol": sorted(per_symbol, key=lambda r: -(r["delta_variance"] or 0.0)),
        "verdict": verdict,
    }


# ------------------------------------------------------------------ datos ------------


def crypto_universe(config: AppConfig) -> list[str]:
    """Los pares CRIPTO del universo operable. La renta variable no cotiza 24/7 y la
    descomposicion por sesion no significaria lo mismo (ni CCXT la sirve)."""
    return sorted(
        s for s in config.runner.symbols if detect_asset_class(s) is AssetClass.CRYPTO
    )


def build_provider(exchange: str, *, offline: bool):
    """None en modo offline: `get_hourly_bars` entiende eso como 'solo cache'."""
    if offline:
        return None
    # Import tardio: construir el proveedor llama a load_markets(), que es red.
    from ai_trader.data.providers.ccxt_crypto import CCXTCrypto, CCXTCryptoConfig

    return CCXTCrypto(CCXTCryptoConfig(exchange_id=exchange))


def fetch_hourly(
    symbols: Sequence[str], start, end, provider
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Barras 1H por simbolo. Los que el exchange no sirve se OMITEN y se declaran."""
    loaded: dict[str, pd.DataFrame] = {}
    omitted: list[dict] = []
    for symbol in symbols:
        df = get_hourly_bars(symbol, start, end, provider=provider)
        if df is None or df.empty:
            omitted.append({"symbol": symbol, "reason": "sin barras 1H disponibles"})
            continue
        loaded[symbol] = slice_window(df, start, end)
        logger.info(
            "  %-12s %6d barras 1H  %s -> %s",
            symbol, len(loaded[symbol]),
            loaded[symbol].index.min().date(), loaded[symbol].index.max().date(),
        )
    return loaded, omitted


# ------------------------------------------------------------------ informe ----------


def analyze(
    tables: dict[str, pd.DataFrame],
    matrices: dict[str, DayMatrix],
    plan: dict,
    omitted: Sequence[dict],
) -> dict:
    """Convierte las tablas diarias por simbolo en el informe publicable."""
    years = list(range(plan["start_year"], plan["end_year"] + 1))
    symbols = sorted(tables)

    per_symbol_year = []
    for symbol in symbols:
        table = tables[symbol]
        for year in years:
            block = table[table["year"] == year]
            if len(block) < MIN_DAYS_PER_YEAR:
                continue
            per_symbol_year.append(
                {
                    "symbol": symbol,
                    "year": int(year),
                    "n_days": int(len(block)),
                    "sessions": session_shares(block),
                }
            )

    # La cohorte equilibrada: los pares que llegan al minimo de dias en TODOS los anos.
    # Es la unica base sobre la que una serie temporal de cuotas mide desplazamiento de
    # actividad y no crecimiento del universo.
    covered: dict[str, set[int]] = {}
    for row in per_symbol_year:
        covered.setdefault(row["symbol"], set()).add(row["year"])
    cohort = sorted(s for s in symbols if covered.get(s, set()) >= set(years))
    if not cohort:
        raise ValueError(
            "Ningun par cubre todos los anos de la ventana: no hay cohorte equilibrada "
            "sobre la que medir la tendencia"
        )

    all_days = pd.concat([tables[s] for s in symbols])
    cohort_days = pd.concat([tables[s] for s in cohort])

    overall = session_shares(all_days)
    gap = gap_stats(all_days)
    latency = latency_stats(all_days)
    trend = trend_analysis(tables, cohort, years)

    gap_median = (gap.get("share_of_range") or {}).get("median")
    gap_verdict = (
        VERDICT_GAP_MATERIAL
        if gap_median is not None and gap_median >= GAP_MATERIAL_SHARE
        else VERDICT_GAP_NEGLIGIBLE
    )
    latency_1h = next((r for r in latency["rows"] if r["hours"] == 1), None)
    latency_material = bool(
        latency_1h
        and latency_1h["slip_share_of_range_median"] is not None
        and latency_1h["slip_share_of_range_median"] >= LATENCY_MATERIAL_SHARE
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "sessions": [s.as_dict() for s in SESSIONS],
        "symbols": [
            {
                "symbol": s,
                "n_days": int(len(tables[s])),
                "n_bars": matrices[s].n_bars,
                "n_days_dropped": matrices[s].n_days_dropped,
                "first_day": str(tables[s].index.min().date()),
                "last_day": str(tables[s].index.max().date()),
                "in_cohort": s in cohort,
            }
            for s in symbols
        ],
        "omitted": list(omitted),
        "cohort": cohort,
        "overall": {
            "n_days": int(len(all_days)),
            "n_symbols": len(symbols),
            "sessions": overall,
        },
        "cohort_overall": {
            "n_days": int(len(cohort_days)),
            "n_symbols": len(cohort),
            "sessions": session_shares(cohort_days),
        },
        "by_symbol_year": per_symbol_year,
        "gap": gap,
        "latency": latency,
        "trend": trend,
        "verdicts": _verdicts(
            gap, gap_verdict, latency, latency_material, trend,
            plan.get("reference_cost_bps", 0.0),
        ),
        "caveats": caveats(plan, symbols, cohort),
    }


def _verdicts(
    gap: dict,
    gap_verdict: str,
    latency: dict,
    latency_material: bool,
    trend: dict,
    reference_cost_bps: float,
) -> dict:
    """Las conclusiones ya leidas, en el informe y no en la prosa del dashboard: lo que se
    audita es el JSON, y tiene que llevar la conclusion, no la instruccion de sacarla."""
    gap_median = (gap.get("share_of_range") or {}).get("median")
    bps = (gap.get("bps") or {}).get("median")
    latency_1h = next((r for r in latency["rows"] if r["hours"] == 1), None)

    if gap_verdict == VERDICT_GAP_MATERIAL:
        gap_text = (
            f"El hueco entre el cierre que la estrategia ve y el open al que se llena vale "
            f"{_pctfmt(gap_median)} del rango del dia (mediana), por encima del umbral "
            f"declarado del {_pctfmt(GAP_MATERIAL_SHARE)}. El motor esta ignorando un tramo "
            "de formacion de precio que importa: hay que declararlo como limitacion y "
            "considerar modelarlo."
        )
    else:
        gap_text = (
            f"El hueco vale {_pctfmt(gap_median)} del rango del dia (mediana, "
            f"{_num(bps)} pb), por debajo del umbral declarado del "
            f"{_pctfmt(GAP_MATERIAL_SHARE)}. En un mercado 24/7 la vela diaria de las 00:00 "
            "UTC empieza donde termino la de ayer, asi que la ventana ciega del backtest no "
            "tiene ancho: la convencion de llenar al open NO introduce sesgo. Es un "
            "resultado, no una ausencia de resultado, y cambia donde hay que mirar."
        )

    if latency_1h is None:
        latency_text = "No hay dias suficientes para medir la latencia de ejecucion."
    else:
        share = latency_1h["slip_share_of_range_median"]
        slip_bps = latency_1h["slip_bps_median"]
        # La comparacion que de verdad importa no es contra el rango del dia sino contra lo
        # que el motor SI cobra por operar: un desplazamiento pequeno en fraccion de rango
        # puede ser varias veces el coste de ejecucion modelado, y entonces no es de
        # segundo orden por mucho que la fraccion parezca modesta.
        multiple = None if not reference_cost_bps else (slip_bps or 0.0) / reference_cost_bps
        umbral = (
            f"por encima del umbral del {_pctfmt(LATENCY_MATERIAL_SHARE)}"
            if latency_material
            else f"por debajo del umbral del {_pctfmt(LATENCY_MATERIAL_SHARE)}"
        )
        sesgo = (
            "esa suposicion no sesga el precio -el hueco es cero-"
            if gap_verdict == VERDICT_GAP_NEGLIGIBLE
            else "esa suposicion se suma al hueco, que ya era material por si solo,"
        )
        latency_text = (
            f"Lo que si queda sin modelar es la LATENCIA. Con una hora de retraso sobre el "
            f"open, el precio de llenado se ha desplazado {_pctfmt(share)} del rango del dia "
            f"({_num(slip_bps)} pb), {umbral}. Pero la fraccion de rango es el denominador "
            "equivocado para decidir si importa: el que importa es el coste de ejecucion que "
            f"el motor ya cobra por entrar ({_num(reference_cost_bps)} pb de referencia, "
            f"comision mas deslizamiento plano). Frente a ese, llegar una hora tarde cuesta "
            f"{_num(multiple)}x. El backtest supone ejecucion instantanea a las 00:00 UTC; "
            f"{sesgo} pero pone un techo a la puntualidad con la que el ciclo real tiene que "
            "ejecutar para que el backtest siga describiendolo."
        )

    if trend["verdict"] == VERDICT_TREND_UP:
        trend_text = (
            f"La DIRECCION de la hipotesis se sostiene: la cuota estadounidense de varianza "
            f"sube {_pctfmt(trend['mean_delta_variance'])} de media entre el antes y el "
            f"despues de {trend['split']}, en {trend['n_positive']} de {trend['n_symbols']} "
            f"pares (test de signos p = {trend['sign_test_p']}). " + _shape_reading(trend)
        )
    elif trend["verdict"] == VERDICT_TREND_DOWN:
        trend_text = (
            f"La hipotesis sale AL REVES: la cuota estadounidense CAE "
            f"{_pctfmt(abs(trend['mean_delta_variance'] or 0.0))} entre el antes y el "
            f"despues de {trend['split']} (test de signos p = {trend['sign_test_p']})."
        )
    else:
        trend_text = (
            f"La hipotesis NO se sostiene con esta evidencia: el cambio medio de cuota "
            f"estadounidense es {_pctfmt(trend['mean_delta_variance'])} "
            f"({trend['n_positive']}/{trend['n_symbols']} pares al alza, test de signos "
            f"p = {trend['sign_test_p']}), y no supera el umbral declarado del "
            f"{_pctfmt(US_SHARE_GROWTH_MIN)}."
        )

    return {
        "gap": {"key": gap_verdict, "material": gap_verdict == VERDICT_GAP_MATERIAL,
                "text": gap_text},
        "latency": {"material": latency_material, "text": latency_text},
        "trend": {"key": trend["verdict"], "text": trend_text},
    }


def _shape_reading(trend: dict) -> str:
    """El MECANISMO, que es una pregunta distinta de la direccion.

    Se escribe aparte porque el error de lectura mas facil aqui es cobrarle a un evento de
    enero de 2024 una pendiente que ya llevaba anos. La cifra que lo separa es la
    correlacion ano-cuota RESTRINGIDA a los anos anteriores al corte."""
    pre_rho, step = trend["pre_split_spearman"], trend["step_at_split"]
    if pre_rho is None:
        return (
            "No hay anos suficientes antes del corte para separar un salto de una deriva, "
            "asi que el mecanismo queda sin contrastar."
        )
    if trend["shape"] == SHAPE_DRIFT:
        return (
            f"Su MECANISMO, en cambio, NO: la cuota ya venia subiendo antes del corte "
            f"(Spearman ano-cuota = {pre_rho:+.2f} sobre "
            f"{trend['pre_split_years'][0]}-{trend['pre_split_years'][-1]}, por encima del "
            f"umbral declarado de {PRE_TREND_RHO:.2f}) y el escalon del ultimo ano previo "
            f"al primero posterior es {_pctfmt(step)}. Es una DERIVA de varios anos que el "
            "corte parte por la mitad, no un salto atribuible a lo que paso en enero de "
            "2024. El contraste pre/post mide la pendiente acumulada; leerlo como efecto "
            "del mecanismo seria atribuirle credito ajeno."
        )
    if trend["shape"] == SHAPE_JUMP:
        return (
            f"Y su MECANISMO tambien encaja: antes del corte no habia pendiente "
            f"(Spearman = {pre_rho:+.2f}, por debajo de {PRE_TREND_RHO:.2f}) y el escalon "
            f"en el corte es {_pctfmt(step)}. La forma es la de un salto, que es lo que "
            "predice el mecanismo declarado."
        )
    return (
        f"La FORMA es ambigua: la pendiente previa vale {pre_rho:+.2f} y el escalon en el "
        f"corte {_pctfmt(step)}. Con esta evidencia no se puede separar el salto de la "
        "deriva, asi que la direccion se sostiene pero el mecanismo queda sin contrastar."
    )


def _pctfmt(value) -> str:
    return "n/d" if value is None else f"{100.0 * float(value):.2f}%"


def _num(value, decimals: int = 1) -> str:
    return "n/d" if value is None else f"{float(value):.{decimals}f}"


def caveats(plan: dict, symbols: Sequence[str], cohort: Sequence[str]) -> list[dict]:
    """Los limites del estudio, EN el informe. Quien lee estas cuotas es quien decide si el
    motor se toca, y esa decision necesita saber en que direccion empuja cada sesgo."""
    return [
        {
            "key": "sesiones_desiguales",
            "title": "Las tres sesiones no duran lo mismo",
            "text": (
                "Cortar el dia por aperturas reales da tramos de "
                + ", ".join(f"{s.hours} h ({s.label.lower()})" for s in SESSIONS)
                + ". Una cuota bruta mayor puede significar solo 'dura mas'. Por eso toda "
                "cuota va acompanada de su intensidad (cuota / fraccion de reloj), que es "
                "lo unico comparable entre tramos; 1,0 es neutro."
            ),
        },
        {
            "key": "solape_de_sesiones",
            "title": "Las sesiones reales se solapan; la particion no puede",
            "text": (
                "Londres y Nueva York coinciden varias horas al dia, y ese solape es "
                "precisamente el tramo mas activo del calendario financiero. Una particion "
                "exhaustiva y disjunta -que es lo que hace falta para que las cuotas sumen "
                "1- tiene que asignar esas horas a UNA sesion. Aqui van a la "
                "estadounidense, porque el corte se pone en la apertura de EE.UU. La "
                "consecuencia es que la cuota estadounidense incorpora el solape y la "
                "europea no: es una decision, no una medida."
            ),
        },
        {
            "key": "sin_horario_de_verano",
            "title": "Los cortes no se mueven con el horario de verano",
            "text": (
                "Los mercados de referencia cambian de hora UTC dos veces al ano; los "
                "cortes de este estudio no. Desplazarlos haria que la serie de cuotas "
                "midiera tambien el calendario civil, y ademas la rejilla de velas del "
                "exchange es UTC fija. El coste es que durante el invierno del hemisferio "
                "norte la primera hora de cada sesion occidental cae en el tramo anterior; "
                "se ha elegido la frontera MAS TEMPRANA justo para que el error vaya en esa "
                "direccion y nunca en la contraria (una sesion no contiene la apertura de "
                "la siguiente)."
            ),
        },
        {
            "key": "un_solo_exchange",
            "title": f"Un solo exchange ({plan['exchange']}) y un solo quote (USDT)",
            "text": (
                "Todo esto es el libro de un exchange concreto en pares contra USDT. La "
                "distribucion horaria de la actividad puede diferir en plataformas con otra "
                "base de usuarios (una regulada en EE.UU. cargaria mas la sesion "
                "estadounidense por construccion). La cifra es del sustrato sobre el que el "
                "sistema opera y se backtestea, que es lo que se queria medir, pero no es "
                "'el mercado' en abstracto."
            ),
        },
        {
            "key": "cohorte_equilibrada",
            "title": "La tendencia se mide sobre una cohorte, no sobre el universo",
            "text": (
                f"{len(cohort)} de {len(symbols)} pares cubren todos los anos de la ventana "
                f"con al menos {MIN_DAYS_PER_YEAR} dias utilizables. La serie anual y el "
                "contraste pre/post se calculan SOLO sobre ellos. Con el universo completo, "
                "la cuota de 2024 incorporaria pares que no existian en 2020 y la 'tendencia' "
                "seria en parte la historia de que se listo cuando. Los pares fuera de la "
                "cohorte se publican igual en la tabla por simbolo y ano."
            ),
        },
        {
            "key": "sesgo_supervivencia",
            "title": "Sesgo de supervivencia",
            "text": (
                "Los pares son los que cotizan HOY en el universo operable. Los deslistados "
                "entre 2020 y 2025 no estan. Para una descomposicion horaria el sesgo es "
                "menos grave que para un backtest -no se esta midiendo rentabilidad- pero "
                "sigue siendo un universo seleccionado por haber sobrevivido."
            ),
        },
        {
            "key": "dias_incompletos",
            "title": "Los dias incompletos se caen enteros",
            "text": (
                "Un dia entra solo si tiene sus 24 barras y ademas existe la de las 23:00 "
                "del dia anterior. Las paradas de mantenimiento del exchange tienden a caer "
                "en horas de poca actividad, asi que descartar el dia entero es mas "
                "conservador que rellenarlo: rellenar inventaria justo el dato que se "
                "quiere medir. Cada simbolo declara cuantos dias ha perdido."
            ),
        },
        {
            "key": "no_es_causal",
            "title": "Esto describe cuando se mueve el precio, no por que",
            "text": (
                "Una cuota horaria no identifica quien opera. Que la actividad se concentre "
                "en horario estadounidense es compatible con flujo institucional local, con "
                "el solape de Londres, con la publicacion de datos macro de EE.UU. y con que "
                "los tres coincidan. El estudio mide el CUANDO, que es lo que el motor "
                "necesita saber; el porque exigiria datos que no estan en una vela."
            ),
        },
    ]


# --------------------------------------------------------------------- main ----------


def reference_cost_bps(config: AppConfig) -> float:
    """Coste de REFERENCIA de una entrada, en puntos basicos: comision mas deslizamiento
    plano del config. No es lo que cobra el motor -que usa el modelo de microestructura,
    dependiente del simbolo y del tamano- sino el numero unico y comparable que ya usan los
    baselines y la auditoria de costes. Sirve de vara de medir para la latencia: sin el, un
    desplazamiento en fraccion de rango no dice si importa o no."""
    execution = config.execution
    return float(execution.fee_rate) * 1e4 + float(execution.slippage_bps)


def build_plan(args: argparse.Namespace, symbols: Sequence[str], config: AppConfig) -> dict:
    start, end = to_utc(args.start), to_utc(args.end)
    return {
        "config_path": str(args.config),
        "exchange": args.exchange,
        "offline": bool(args.offline),
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "start_year": int(start.year),
        # `end` es exclusivo y siempre cae en frontera de ano, asi que el ultimo ano
        # COMPLETO de la ventana es el anterior.
        "end_year": int((end - pd.Timedelta(days=1)).year),
        "timeframe": "1H",
        "requested_symbols": list(symbols),
        "min_days_per_year": MIN_DAYS_PER_YEAR,
        "hours_per_day": HOURS_PER_DAY,
        "latency_hours": list(LATENCY_HOURS),
        "reference_cost_bps": round(reference_cost_bps(config), 4),
        "thresholds": {
            "gap_material_share": GAP_MATERIAL_SHARE,
            "latency_material_share": LATENCY_MATERIAL_SHARE,
            "us_share_growth_min": US_SHARE_GROWTH_MIN,
        },
    }


def load_sessions_report(path: Path | str = SESSIONS_REPORT) -> dict | None:
    """Lee el informe publicado. None si no esta, para que el dashboard y la documentacion
    degraden a prosa sin cifras en vez de romperse."""
    report = Path(path)
    if not report.exists():
        return None
    return json.loads(report.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    symbols = crypto_universe(config)
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols}
        symbols = [s for s in symbols if s in wanted]
    if not symbols:
        raise ValueError("El universo cripto pedido esta vacio")

    plan = build_plan(args, symbols, config)
    logger.info(
        "Barras 1H (%s, %s -> %s) para %d pares cripto%s",
        args.exchange, plan["window"]["start"], plan["window"]["end"], len(symbols),
        " [offline: solo cache]" if args.offline else "",
    )
    provider = build_provider(args.exchange, offline=args.offline)
    hourly, omitted = fetch_hourly(symbols, args.start, args.end, provider)
    if not hourly:
        raise ValueError("Ningun par ha devuelto barras 1H")

    matrices: dict[str, DayMatrix] = {}
    tables: dict[str, pd.DataFrame] = {}
    for symbol, frame in hourly.items():
        matrix = build_day_matrix(frame)
        if matrix.n_days == 0:
            omitted.append({"symbol": symbol, "reason": "ningun dia UTC completo y encadenado"})
            continue
        matrices[symbol] = matrix
        tables[symbol] = daily_table(matrix)
        logger.info(
            "  %-12s %5d dias utilizables (%d descartados)",
            symbol, matrix.n_days, matrix.n_days_dropped,
        )
    if not tables:
        raise ValueError("Ningun par tiene dias utilizables")

    return analyze(tables, matrices, plan, omitted)


def _strip_volatile(report: dict) -> dict:
    return {k: v for k, v in report.items() if k != "generated_at"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END, help="EXCLUSIVO")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument(
        "--offline", action="store_true", help="No llamar al exchange: usar solo la cache."
    )
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.WARNING)

    report = run(args)

    if args.verify_determinism:
        replay = run(args)
        identical = _strip_volatile(report) == _strip_volatile(replay)
        report["determinism"] = {"checked": True, "identical": identical}
        logger.info("Determinismo: informe %s", "identico" if identical else "DISTINTO")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SESSIONS_REPORT.name
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Informe -> %s", path)
    _print_report(report)
    return 0


def _print_report(report: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    plan, overall, trend = report["plan"], report["overall"], report["trend"]
    print("\n=== DESCOMPOSICION POR SESION HORARIA ===")
    print(
        f"  {plan['exchange']} 1H | {plan['window']['start']} -> {plan['window']['end']} "
        f"(exclusivo) | {overall['n_symbols']} pares | "
        f"{overall['n_days']:,} dias-simbolo utilizables"
    )
    print(f"\n{'sesion':<16}{'UTC':>9}{'|ret|':>9}{'var':>9}{'rango':>9}{'int.var':>9}"
          f"{'fija max':>10}{'fija min':>10}")
    for session in SESSIONS:
        row = overall["sessions"][session.key]
        print(
            f"{session.label:<16}{f'{session.start_hour:02d}-{session.end_hour:02d}':>9}"
            f"{_pctfmt(row['abs_return']):>9}{_pctfmt(row['variance']):>9}"
            f"{_pctfmt(row['range']):>9}{_num(row['variance_intensity'], 2):>9}"
            f"{_pctfmt(row['sets_high']):>10}{_pctfmt(row['sets_low']):>10}"
        )

    print("\n--- LA CIFRA QUE DECIDE: cierre visto -> open llenado ---")
    gap = report["gap"]
    print(
        f"  |hueco| / rango diario: mediana {_pctfmt(gap['share_of_range']['median'])} · "
        f"p90 {_pctfmt(gap['share_of_range']['p90'])} · "
        f"p99 {_pctfmt(gap['share_of_range']['p99'])}"
    )
    print(
        f"  |hueco| en pb: mediana {_num(gap['bps']['median'], 2)} · "
        f"p99 {_num(gap['bps']['p99'], 2)} | dias por encima del umbral "
        f"({_pctfmt(gap['threshold'])} del rango): {_num(gap['days_above_threshold_pct'], 2)}%"
    )
    print(f"  {report['verdicts']['gap']['text']}")

    print("\n--- Lo que si queda sin modelar: latencia de ejecucion ---")
    print(f"{'retraso':<10}{'sesion':<16}{'desplaz./rango':>16}{'pb':>9}{'rango gastado':>16}")
    for row in report["latency"]["rows"]:
        delay = f"+{row['hours']} h"
        print(
            f"{delay:<10}{row['session']:<16}"
            f"{_pctfmt(row['slip_share_of_range_median']):>16}"
            f"{_num(row['slip_bps_median'], 1):>9}"
            f"{_pctfmt(row['path_share_of_range_median']):>16}"
        )
    print(f"  {report['verdicts']['latency']['text']}")

    print(f"\n--- Tendencia: cuota estadounidense ({len(trend['cohort'])} pares en cohorte) ---")
    print(f"{'ano':<8}{'asia':>10}{'europa':>10}{'EE.UU.':>10}{'dias':>9}")
    for row in trend["yearly"]:
        print(
            f"{row['year']:<8}{_pctfmt(row['asia']['variance']):>10}"
            f"{_pctfmt(row['europe']['variance']):>10}{_pctfmt(row['us']['variance']):>10}"
            f"{row['n_days']:>9,}"
        )
    print(
        f"  forma: {trend['shape']} | pendiente previa (Spearman "
        f"{trend['pre_split_years'][0] if trend['pre_split_years'] else '-'}-"
        f"{trend['pre_split_years'][-1] if trend['pre_split_years'] else '-'}) = "
        f"{_num(trend['pre_split_spearman'], 2)} | escalon en el corte = "
        f"{_pctfmt(trend['step_at_split'])}"
    )
    print(f"  {report['verdicts']['trend']['text']}")


if __name__ == "__main__":
    raise SystemExit(main())
