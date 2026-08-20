"""
LA PUERTA DE SENAL: con que parametro se arma cada familia y con que valor se abre.

Son las constantes que convierten "la capa de senal esta encendida" en un cambio concreto
de `StrategySpec`. Las estreno el barrido de rho sobre el canal sintetico, pero no tienen
nada de sintetico: el estudio de la capa tematica arma exactamente la misma puerta sobre
archivo de senales REAL, y necesita el mismo mapa para que las dos mediciones hablen de lo
mismo. Por eso viven fuera de los dos estudios.
"""
from __future__ import annotations

# Umbral con el que se abre la puerta: opera solo si el tono es positivo. CERO, y no un
# numero sorteado: es la mediana de una z, o sea "la senal apunta arriba", el corte que no
# tiene ningun grado de libertad que ajustar. Sortearlo convertiria el barrido en una
# optimizacion de la puerta y el break-even dejaria de ser una propiedad del diseno.
GATE_MIN_TONE = 0.0
# El unico parametro que se inyecta. Existe con este nombre y el mismo significado —un
# PISO de tono— en las dos familias publicadas y en cinco de las seis tematicas.
GATE_PARAM = "min_signal_tone"

# La excepcion, y no es un caso especial arbitrario: `event_calendar_drift` NO DECLARA
# `min_signal_tone`, porque de las seis fuentes de su tema solo `ofac_sdn` tiene polaridad y
# el tono sale ~0 por construccion. Un piso de tono ahi seria un mando que parece hacer algo
# y no puede. Su equivalente —"solo opero si hay catalizador cerca"— es un piso de INTENSIDAD.
GATE_PARAM_BY_FAMILY: dict[str, str] = {
    "event_calendar_drift": "min_signal_intensity",
}

# Con que valor se abre cada puerta. El de intensidad no puede ser 0,0: ese es su valor
# INERTE (el borde inferior del rango), asi que inyectarlo no encenderia nada. Se usa la
# mediana del rango util, que es el analogo de "la senal apunta arriba" para un eje sin signo:
# "esta pasando algo por encima de lo normal".
GATE_VALUE_BY_PARAM: dict[str, float] = {
    "min_signal_tone": GATE_MIN_TONE,
    "min_signal_intensity": 0.5,
}


def gate_param_for(strategy_type: str) -> str:
    """El parametro con el que se arma la puerta de una familia."""
    return GATE_PARAM_BY_FAMILY.get(strategy_type, GATE_PARAM)
