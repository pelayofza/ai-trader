"""
REPORTE DIARIO EXPERTO: la primera primitiva del proyecto SIN nucleo de precio.

QUE LA HACE DISTINTA DE LAS NUEVE ANTERIORES
---------------------------------------------
Las nueve que ya existen tienen la misma anatomia: un nucleo de precio que decide y una capa
de senal que modula. Incluso `signal_composite`, cuya tesis vive entera en la capa, necesita
que el precio le diga CUANDO —un giro de la media corta— antes de mirar nada. Aqui no hay
nucleo: la decision entera sale de las 37 respuestas categoricas que el agente externo
escribe cada manana en `data/signals_raw/ai_reports/`, y el precio solo aporta el numero al
que se entra.

Es deliberado y es el punto: si la segunda via de captura vale algo, tiene que poder sostener
una decision por si sola. Mezclarla con un filtro de medias moviles haria imposible saber
cual de las dos partes decidio.

LO QUE ESTA ESTRATEGIA NO PROMETE, Y HAY QUE LEERLO ANTES QUE NADA
-------------------------------------------------------------------
**No esta medida.** No hay backtest, no hay ranking, no hay comparacion con las otras ocho, y
no puede haberla: el archivo del reporte diario empezo el 2026-08-20 y hoy tiene tres dias.
Un backtest sobre tres dias no es un backtest debil, es un numero sin contenido. Sus pesos
(`observation/daily_report_scores.py`) son juicio experto AFIRMADO, no estimado, y sus
umbrales se fijaron mirando la unica seccion cruzada disponible —lo cual es sobreajuste de un
dia, dicho con todas las letras—.

Se pone a operar igualmente, y con prioridad forzada, por la razon que fija la Regla 2 del
proyecto desde el 2026-08-20: es mejor tener la herramienta corriendo con sobreajuste y
atacarlo despues con evidencia de calendario, que seguir refinando el juez de un backtest
que no decide nada. La unica evidencia que esto va a producir es el diario del paper
trading, y esa se compra con tiempo. **Es paper trading, no hay dinero real, y la medida es
temporal.**

COMO DECIDE, EN CUATRO PASOS
-----------------------------
1. FRESCURA. El reporte tiene hora de corte (06:00Z). Pasadas `max_report_age_hours` no se
   opera: una lectura de anteayer repetida cada quince minutos seria la peor version de
   esto. Sin hora de corte legible se trata como caducado.
2. LADO Y ELEGIBILIDAD. Hace falta conviccion ABSOLUTA (|score| >= `min_abs_score`) Y estar
   entre los `top_n` mejores del corte transversal por ese lado. Lo segundo no es una
   floritura: el runner recorre el universo en el orden del config y para al llegar a
   `max_trades_per_cycle`, asi que sin ordenacion el que opera es el que estaba antes en la
   lista. El 2026-08-22 las 24 lecturas salieron positivas; con umbral absoluto a secas se
   habrian abierto los cinco primeros del fichero, no los cinco mejores.
3. LA HORQUILLA. El stop se mide en sigmas DIARIAS del propio activo, y la sigma sale del
   reporte (P32 realizada, P33 implicita), no de las barras. El objetivo es un multiplo del
   stop que sube con la conviccion y BAJA con el riesgo de evento (P28, P29) y con la
   aglomeracion en contra (P16, P25, P31). Esa es la parte "variable en funcion de las
   senales": dos activos con el mismo lado y distinto reporte salen con horquillas distintas.
4. CONFIANZA. Conviccion, cobertura y profundidad del libro. Es el mando de tamano del motor
   de riesgo, asi que un reporte flojo entra con menos dinero.

DOS COMPORTAMIENTOS CONOCIDOS QUE NO SON BUGS
----------------------------------------------
* **La senal es constante dentro del dia.** El reporte cambia una vez cada 24 h y el ciclo
  corre cada 15 minutos, asi que la misma senal se repite. Quien impide que eso sea una
  posicion nueva cada cuarto de hora es el runner (posicion abierta + `symbol_cooldown_hours`),
  que es su trabajo y no el de la estrategia. La consecuencia a vigilar: si una posicion se
  cierra pronto, pasado el cooldown se puede reabrir contra el MISMO reporte.
* **Sin proveedor no emite nada.** En backtest no se engancha a proposito: el archivo tiene
  tres dias y aplicar el reporte de hoy a una barra de 2023 seria look-ahead con otro nombre.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from ai_trader.observation.daily_report_scores import MIN_COVERAGE, AssetScore, ticker_for
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import utc_now
from ai_trader.shared.schemas import Side, Signal
from ai_trader.strategies.signal_layer import CONF_CEILING, CONF_FLOOR

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DailyReportExpertConfig:
    timeframe: str = "1d"

    # --- elegibilidad -----------------------------------------------------------------
    # |score| minimo. El score es una media ponderada en [-1, +1] y en la unica seccion
    # cruzada disponible fue de +0,17 a +0,52, asi que 0,20 deja fuera el cuartil mas flojo
    # sin pretender ser un umbral medido. No lo es.
    min_abs_score: float = 0.20
    # Cuantos activos del dia son elegibles por lado. Cinco = el mismo techo que
    # `max_open_positions` en `config/default.toml`: pedir mas seria proponer operaciones que
    # el riesgo va a rechazar por cupo, y el rechazo se lo comeria el orden del universo.
    top_n: int = 5
    # Desde la HORA DE CORTE del activo. 30 h cubre el dia entero mas el hueco hasta que
    # entra el reporte de la manana siguiente (08:00 Europe/Madrid), y no mas.
    max_report_age_hours: float = 30.0
    allow_short: bool = True

    # --- horquilla --------------------------------------------------------------------
    # Stop = k sigmas diarias. Con k=2 y la sigma mediana del 2026-08-22 (4,3%/dia) el stop
    # sale al 8,6%, del orden del `default_stop_loss_pct` del motor de riesgo (5%).
    stop_sigma_mult: float = 2.0
    # El suelo evita que un activo tranquilo entre con un stop que el ruido de un dia se
    # lleva. El techo se queda POR DEBAJO del `max_stop_distance_pct` del riesgo (15%): si lo
    # cruzara, el motor lo apretaria y la horquilla real dejaria de ser la que dice el log.
    min_stop_pct: float = 2.0
    max_stop_pct: float = 12.0
    # Objetivo = multiplo del stop. Base + conviccion - riesgo de evento - aglomeracion.
    base_reward: float = 1.4
    conviction_reward: float = 1.6
    event_reward_drag: float = 0.5
    crowding_reward_drag: float = 0.5
    min_reward: float = 1.0
    max_reward: float = 3.5

    def __post_init__(self) -> None:
        if not 0.0 < self.min_abs_score <= 1.0:
            raise ValueError("min_abs_score must be in (0, 1]")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.max_report_age_hours <= 0:
            raise ValueError("max_report_age_hours must be > 0")
        if self.stop_sigma_mult <= 0:
            raise ValueError("stop_sigma_mult must be > 0")
        if not 0.0 < self.min_stop_pct <= self.max_stop_pct:
            raise ValueError("min_stop_pct must be > 0 and <= max_stop_pct")
        # El riesgo aprieta cualquier stop que pase de `max_stop_distance_pct`. Un techo por
        # encima no seria "mas permisivo": seria una horquilla que el motor reescribe sin que
        # el log de la estrategia se entere.
        if self.max_stop_pct > 15.0:
            raise ValueError("max_stop_pct must be <= 15 (max_stop_distance_pct del riesgo)")
        if self.base_reward <= 0 or self.conviction_reward < 0:
            raise ValueError("base_reward must be > 0 and conviction_reward >= 0")
        if self.event_reward_drag < 0 or self.crowding_reward_drag < 0:
            raise ValueError("the reward drags cannot be negative")
        if not 0.0 < self.min_reward <= self.max_reward:
            raise ValueError("min_reward must be > 0 and <= max_reward")


class DailyReportExpertStrategy:
    strategy_id = "daily_report_expert_v1"
    theme = "daily_report"

    def __init__(self, config: DailyReportExpertConfig | None = None) -> None:
        self.config = config or DailyReportExpertConfig()
        self._reports = None
        self._warned_no_provider = False

    def attach_daily_report_provider(self, provider) -> None:
        """La misma costura duck-typed que el regimen y el radar. Ver `main.py`."""
        self._reports = provider

    def supports_symbol(self, symbol: str) -> bool:
        # El universo del agente externo es `config/assets.json`: 24 criptomonedas al
        # contado. Un mercado de prediccion o una accion no tienen reporte que leer.
        return ticker_for(symbol) is not None

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        cfg = self.config

        if self._reports is None:
            if not self._warned_no_provider:
                logger.warning(
                    "Reporte diario | sin proveedor enganchado: %s no emitira ninguna senal. "
                    "En backtest es lo esperado; en vivo significa que falta la captura.",
                    self.strategy_id,
                )
                self._warned_no_provider = True
            return None

        reading = self._reports.reading(symbol)
        if reading is None:
            return None

        age = self._reports.age_hours(reading.ticker)
        if age is None or age > cfg.max_report_age_hours:
            logger.info(
                "Reporte diario | %s caducado (%s h > %.1f h): no se opera",
                symbol, "sin corte" if age is None else f"{age:.1f}", cfg.max_report_age_hours,
            )
            return None

        side = self._side(reading)
        if side is None:
            return None

        entry = bar_schema.last_close(bars) if bars is not None and not bars.empty else None
        if entry is None or entry <= 0:
            logger.info("Reporte diario | %s sin cierre para fijar la entrada", symbol)
            return None
        entry = float(entry)

        stop_pct = min(max(cfg.stop_sigma_mult * reading.sigma_daily_pct,
                           cfg.min_stop_pct), cfg.max_stop_pct)
        reward = self._reward(reading, side)
        direction = 1.0 if side is Side.BUY else -1.0
        stop_loss = entry * (1.0 - direction * stop_pct / 100.0)
        take_profit = entry * (1.0 + direction * reward * stop_pct / 100.0)
        if stop_loss <= 0 or take_profit <= 0:
            logger.info("Reporte diario | horquilla degenerada en %s; se salta", symbol)
            return None

        confidence = self._confidence(reading)

        logger.info(
            "Reporte diario | %s | score=%+.3f (rank L%d/S%d de %d) | cobertura=%.2f | "
            "sigma=%.2f%%/dia | stop=%.2f%% | R=%.2f | conf=%.2f",
            symbol, reading.score, reading.rank_long, reading.rank_short, reading.n_scored,
            reading.coverage, reading.sigma_daily_pct, stop_pct, reward, confidence,
        )

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            timeframe=cfg.timeframe,
            timestamp=utc_now(),
            side=side,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=(
                f"Reporte del {self._reports.date}: score {reading.score:+.2f} sobre "
                f"{reading.n_answered} respuestas ({reading.coverage:.0%} de cobertura "
                f"ponderada), puesto {reading.rank_long if side is Side.BUY else reading.rank_short}"
                f" de {reading.n_scored}. Pesan {', '.join(reading.drivers)}."
            ),
            features={
                "report_date": self._reports.date,
                "report_age_hours": round(age, 2),
                "expert_score": round(reading.score, 4),
                "coverage": round(reading.coverage, 4),
                "n_answered": reading.n_answered,
                "rank_long": reading.rank_long,
                "rank_short": reading.rank_short,
                "n_scored": reading.n_scored,
                "sigma_daily_pct": round(reading.sigma_daily_pct, 4),
                "sigma_source": reading.sigma_source,
                "event_risk": round(reading.event_risk, 4),
                "crowding": round(reading.crowding, 4),
                "crowding_coverage": round(reading.crowding_coverage, 4),
                "depth_factor": reading.depth_factor,
                "beta_scale": reading.beta_scale,
                "stop_pct": round(stop_pct, 4),
                "reward_multiple": round(reward, 4),
                **{f"block_{k}": v for k, v in reading.blocks.items()},
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )

    # --- las tres decisiones, cada una en su sitio ------------------------------------

    def _side(self, reading: AssetScore) -> Side | None:
        """Conviccion absoluta Y puesto en el corte transversal. Las dos, no una."""
        cfg = self.config
        if reading.score >= cfg.min_abs_score and reading.rank_long <= cfg.top_n:
            return Side.BUY
        if (
            cfg.allow_short
            and reading.score <= -cfg.min_abs_score
            and reading.rank_short <= cfg.top_n
        ):
            return Side.SELL
        return None

    def _reward(self, reading: AssetScore, side: Side) -> float:
        """El multiplo del stop al que se pone el objetivo.

        Sube con la conviccion —una lectura fuerte merece que la dejen correr— y baja por dos
        motivos distintos que no se confunden: el riesgo de EVENTO (banderas activas y macro
        en el calendario), que dice que el camino tiene obstaculos, y la AGLOMERACION en
        contra, que dice que el sitio al que vas ya esta lleno. La aglomeracion se escala por
        su propia cobertura: con solo el RSI medido, una lectura de +1,00 pesa 0,41 y no 1,00.
        """
        cfg = self.config
        direction = 1.0 if side is Side.BUY else -1.0
        against = max(0.0, reading.crowding * direction) * reading.crowding_coverage
        reward = (
            cfg.base_reward
            + cfg.conviction_reward * reading.strength
            - cfg.event_reward_drag * reading.event_risk
            - cfg.crowding_reward_drag * against
        )
        return min(max(reward, cfg.min_reward), cfg.max_reward)

    def _confidence(self, reading: AssetScore) -> float:
        """Conviccion, cobertura y liquidez -> el mando de tamano del motor de riesgo.

        El rango [0,55, 0,90] se importa de `signal_layer` en vez de reescribirse: es el mismo
        intervalo que producen las otras estrategias por construccion, y dos definiciones de
        lo mismo son una divergencia esperando a pasar. La cobertura se mide POR ENCIMA del
        piso, no desde cero: por debajo del piso no hay lectura, asi que un activo que apenas
        lo pasa tiene que entrar con la confianza minima y no con media.
        """
        headroom = max(1.0 - MIN_COVERAGE, 1e-9)
        coverage_factor = min(max((reading.coverage - MIN_COVERAGE) / headroom, 0.0), 1.0)
        raw = 0.65 * reading.strength + 0.25 * coverage_factor + 0.10 * reading.depth_factor
        confidence = CONF_FLOOR + (CONF_CEILING - CONF_FLOOR) * raw
        return round(min(max(confidence, CONF_FLOOR), CONF_CEILING), 2)
