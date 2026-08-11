"""
Actividad: cuanto OPERA una configuracion, medida sobre las MISMAS ventanas que fijan su
recompensa, y el suelo de elegibilidad que hay que superar para ser rankeable.

EL PROBLEMA QUE RESUELVE (medido, no supuesto). El headline score de una ventana es
`Sharpe - lambda*turnover - kappa*maxDD`. Si una configuracion no abre ninguna posicion en
esa ventana, la curva de equity es una recta: Sharpe 0, rotacion 0, caida 0 -> headline 0
EXACTO. No es una puntuacion mala ni buena: es la ausencia de puntuacion, y sin embargo
entra en la distribucion como un numero mas. Como la recompensa del sistema es el CVaR@25%
(la media del peor cuartil), y un cero le gana a cualquier cosa que arriesgue y pierda, en
un periodo en el que casi todo lo que se juega pierde el ranking se ordena por INACTIVIDAD.

Eso paso de verdad y esta publicado: en `data/transfer/report_ai_v3.json`, el lado real da
Spearman(recompensa, operaciones por ventana OOS) = -0.84 —cuanto menos opera una
configuracion, mejor puntua— y la ganadora del ranking (mean_reversion#07) tiene el 93% de
sus ventanas planas y una recompensa de 0.0000 exacto.

LO QUE ESTE MODULO **NO** ES. No es una penalizacion por rotar poco. Esa ya existe
(`lambda_turnover`) y el estudio de pesos midio que penalizar no estabiliza nada
(`data/calibration/`, ver `HeadlineWeights`). Aqui no se le resta nada a nadie: la
recompensa de una configuracion inactiva se calcula, se publica y se deja intacta. Lo que
se le niega es la ELEGIBILIDAD: estar en la lista de la que se elige, y poder aprobar el
gate. Son dos cosas distintas —una cambia el valor, la otra cambia quien juega— y
confundirlas seria cobrar dos veces por lo mismo.

EL SUELO, Y DE DONDE SALE. `ActivityFloor` tiene dos condiciones y las dos son necesarias:

1. `max_zero_window_pct = 25`. Como mucho una de cada cuatro ventanas puede estar vacia.
   El 25 no es un numero nuevo ni elegido: es `DEFAULT_CVAR_ALPHA` en porcentaje, la
   fraccion de cola que ES la recompensa. Si mas de un cuarto de las ventanas esta vacio,
   el cuartil que fija el CVaR puede estar hecho de ceros estructurales, y entonces la
   cifra no es una medicion. Esta condicion esta DERIVADA.
2. `min_median_trades_per_window = 3`. La ventana mediana tiene que contener al menos 3
   operaciones. Es el proxy ROBUSTO de la anterior: una fraccion de ventanas vacias
   estimada sobre pocas ventanas es ruidosa, y se puede operar poquisimo (una vez cada dos
   meses) sin dejar ni una ventana vacia, con lo que el headline mide una o dos
   operaciones. Esta condicion esta MEDIDA, y se calibra contra la primera en vez de
   inventarse.

El 3 sale de `scoring.activity_study` (informe en `data/activity/`), sobre los 32 pares
(configuracion, mundo) de la rejilla de transferencia:

- Regla declarada antes de mirar, con la misma disciplina que `transfer_study.RHO_ACCEPT`:
  el umbral es el valor de la rejilla {1, 2, 3, 5, 8, 13, 21} que reproduce con MENOS
  desacuerdos la clasificacion de la condicion derivada; a igualdad, el mayor.
- Medido: 3 gana con UN desacuerdo (frente a 6 del 1, 3 del 2, 3 del 5, 4 del 8, 8 del 13).
  El unico desacuerdo es `mean_reversion#00` en el mundo sintetico, que se queda fuera con
  exactamente el 25.0% de ventanas vacias —el filo del cuchillo de la condicion derivada—.
  El 5 que estaba publicado antes a ojo (`transfer_study.ACTIVE_MIN_TRADES_PER_FOLD`)
  ademas tira dos configuraciones que ahi si median algo (mr#01 y mr#05, con 5.8% y 2.5%
  de ventanas vacias).
- Y la eleccion del numero casi no importa, lo cual es la mejor noticia posible: en el lado
  real el conjunto rankeable es IDENTICO para todo umbral de la rejilla hasta 5 (9 de 16
  configuraciones), porque a esa escala quien excluye es la condicion derivada. En esta
  rejilla la condicion 2 solo muerde una vez —el mr#00 sintetico de arriba—; se mantiene
  porque cubre un modo de fallo que estas 16 configuraciones no contienen, no porque aqui
  haga el trabajo.
- Lo que NO se uso para elegirlo: la reproducibilidad del ranking entre mitades del
  historico. Se midio y sale MAYOR con las inactivas dentro (0.39) que sin ellas (0.28;
  subconjuntos aleatorios del mismo tamano, 0.36), porque una configuracion que no opera
  puntua 0 en todos los bloques y su puesto no se mueve nunca. Es estabilidad de
  cementerio: ese criterio habria premiado justo lo que el suelo existe para quitar. Se
  publica como control, con ese nombre.

Limite declarado: medido con 16 configuraciones de dos familias sobre un universo cripto.
Es suficiente para separar "opera" de "no opera" —lo que separa a las dos mitades es un
orden de magnitud, no un decimal— y no lo es para afinar el umbral a un decimal. Moverlo
exige re-correr el estudio (hay un test que lo ata a la evidencia publicada).
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA

# Operaciones que tiene que contener la ventana MEDIANA para que la configuracion sea
# rankeable. Ver el docstring del modulo: medido, no elegido a ojo.
MIN_MEDIAN_TRADES_PER_WINDOW = 3.0

# Maximo de ventanas VACIAS (cero operaciones -> headline 0 exacto) que se toleran, en %.
# Es `DEFAULT_CVAR_ALPHA` expresado en porcentaje: la cola que define la recompensa no
# puede estar hecha de ceros estructurales.
MAX_ZERO_WINDOW_PCT = DEFAULT_CVAR_ALPHA * 100.0


@dataclass(slots=True, frozen=True)
class ActivityStats:
    """
    Cuanto opero una configuracion en las ventanas OOS que produjeron su recompensa.

    Se construye SIEMPRE emparejada con los scores (mismo numero de ventanas y en el mismo
    orden): la actividad de otras ventanas no dice nada sobre esta recompensa.
    """

    n_windows: int
    trades: int
    trades_per_window: float  # media
    median_trades_per_window: float  # el estadistico de elegibilidad
    zero_windows: int  # ventanas sin ninguna operacion (headline 0 exacto)

    @property
    def zero_window_pct(self) -> float:
        return 100.0 * self.zero_windows / self.n_windows if self.n_windows else 0.0

    @property
    def measured(self) -> bool:
        """False si no hay ninguna ventana: no es "no opera", es "no se midio"."""
        return self.n_windows > 0

    def as_dict(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "trades": self.trades,
            "trades_per_window": round(self.trades_per_window, 3),
            "median_trades_per_window": round(self.median_trades_per_window, 3),
            "zero_windows": self.zero_windows,
            "zero_window_pct": round(self.zero_window_pct, 2),
        }


def measure_activity(trades_per_window: Sequence[int]) -> ActivityStats:
    """Resume las operaciones ventana a ventana. La lista es una por ventana OOS, en el
    mismo orden que los scores que se agregaron en la recompensa."""
    counts = [int(t) for t in trades_per_window]
    if not counts:
        return ActivityStats(0, 0, 0.0, 0.0, 0)
    return ActivityStats(
        n_windows=len(counts),
        trades=sum(counts),
        trades_per_window=sum(counts) / len(counts),
        median_trades_per_window=float(statistics.median(counts)),
        zero_windows=sum(1 for c in counts if c == 0),
    )


@dataclass(slots=True, frozen=True)
class ActivityFloor:
    """
    El requisito de ELEGIBILIDAD: que hace falta para que una configuracion sea rankeable.

    No modifica ninguna recompensa. Decide quien entra en la lista y quien puede aprobar el
    gate; el resto se sigue midiendo, publicando y comparando, solo que fuera de la
    competicion. Ver el docstring del modulo para el origen de los dos numeros.
    """

    min_median_trades_per_window: float = MIN_MEDIAN_TRADES_PER_WINDOW
    max_zero_window_pct: float = MAX_ZERO_WINDOW_PCT

    def eligible(self, activity: ActivityStats | None) -> bool:
        """¿Es rankeable? Sin medicion (None o cero ventanas) la respuesta es NO: una
        elegibilidad que no se ha comprobado no es una elegibilidad concedida."""
        return not self.reasons(activity)

    def reasons(self, activity: ActivityStats | None) -> tuple[str, ...]:
        """Por que NO es rankeable, en claro y en el informe. Vacio = si lo es."""
        if activity is None or not activity.measured:
            return ("actividad no medida",)
        out: list[str] = []
        if activity.median_trades_per_window < self.min_median_trades_per_window:
            out.append(
                f"la ventana mediana tiene {activity.median_trades_per_window:.2f} "
                f"operaciones (< {self.min_median_trades_per_window:.0f})"
            )
        if activity.zero_window_pct > self.max_zero_window_pct:
            out.append(
                f"{activity.zero_window_pct:.1f}% de ventanas vacias "
                f"(> {self.max_zero_window_pct:.0f}%)"
            )
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "min_median_trades_per_window": self.min_median_trades_per_window,
            "max_zero_window_pct": self.max_zero_window_pct,
        }


DEFAULT_ACTIVITY_FLOOR = ActivityFloor()


def eligibility_dict(
    activity: ActivityStats | None, floor: ActivityFloor = DEFAULT_ACTIVITY_FLOOR
) -> dict:
    """El veredicto de elegibilidad tal y como se publica: si entra, con que suelo y, si no
    entra, por que. Lo comparten todos los informes que sacan un ranking."""
    reasons = floor.reasons(activity)
    return {
        "rankable": not reasons,
        "floor": floor.as_dict(),
        "reasons": list(reasons),
        "activity": None if activity is None else activity.as_dict(),
    }


__all__ = [
    "DEFAULT_ACTIVITY_FLOOR",
    "MAX_ZERO_WINDOW_PCT",
    "MIN_MEDIAN_TRADES_PER_WINDOW",
    "ActivityFloor",
    "ActivityStats",
    "eligibility_dict",
    "measure_activity",
]
