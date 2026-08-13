"""
LA CAPA DE SENAL: lo que una primitiva tematica hace con su tema, en un solo sitio.

POR QUE EXISTE
--------------
Seis estrategias que consultan un tema necesitan las mismas cuatro decisiones —¿se puede
leer?, ¿cancela el lado?, ¿lo decide?, ¿mueve el tamano?— y el precedente esta escrito en
`shared/indicators.py`: el ATR se extrajo el dia que estuvo duplicado DOS veces, porque dos
copias de la misma regla son una divergencia esperando a pasar. Seis copias de cuarenta
lineas cada una es la misma apuesta multiplicada por tres.

QUE SIGNIFICA "INERTE" AQUI, Y POR QUE ES MAS FUERTE QUE EN MOMENTUM
--------------------------------------------------------------------
En `momentum_crypto` inerte significa "ningun umbral alcanzable bloquea". Aqui significa
algo mas: **con la configuracion por defecto la capa no puede cambiar la elegibilidad, ni el
lado, ni el tamano, con NINGUN radar** —vacio o lleno, con cobertura o sin ella—. Se sostiene
sobre cuatro defaults y cada uno es inerte por su RANGO, no por ser permisivo:

- `min_signal_tone = INERT_MIN_TONE = -Z_CLIP`, y el tono direccional vive en [-Z_CLIP, Z_CLIP]
  (ver `directional_tone`), asi que el piso esta en el borde exacto.
- `min_signal_intensity = 0.0` y `max_signal_intensity = Z_CLIP`, los dos extremos del rango
  de la intensidad.
- `signal_side_mode = SIDE_CORE`: vocabulario CERRADO, y el default es el unico valor que no
  consulta el tono.
- `signal_weight = 0.0`, con rango [0, 1]. El cero es el borde exacto: `resolve_confidence`
  devuelve la confianza base sin tocarla.

Y cada estrategia expone `_signals_active()`, que es False con los defaults: la puerta ni
siquiera se consulta, igual que hoy en momentum.

EL TONO DIRECCIONAL, QUE ES LA UNICA IDEA NUEVA
------------------------------------------------
`signal_gate_reason` pone un PISO al tono, y eso funciona mientras la primitiva solo compre.
Aplicado tal cual a un corto significaria lo contrario de lo que dice: exigir tono alto para
vender es exigir que el mundo este a favor de lo que no vas a hacer. La generalizacion es
mirar el tono A FAVOR DEL LADO —`+tono` para un largo, `-tono` para un corto—, que sigue
viviendo en el mismo intervalo y por tanto conserva la inercia del borde.

UN AVISO QUE HAY QUE TENER ESCRITO
-----------------------------------
La confianza ES el mando de tamano (`risk/engine.py`: `size = equity * risk_fraction *
confidence`) y el riesgo rechaza por debajo de `min_confidence_per_trade = 0.65`. Con
`signal_weight > 0`, una senal en contra puede empujar la confianza por debajo de ese minimo
y hacer que el riesgo rechace: **la modulacion de tamano es una puerta implicita**. No rompe
el invariante —solo actua cuando el tema es legible— pero conviene no descubrirlo leyendo un
informe de rechazos.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from ai_trader.observation.signal_radar import (
    INERT_MAX_INTENSITY,
    INERT_MIN_INTENSITY,
    INERT_MIN_TONE,
    MIN_SIGNAL_COVERAGE,
)
from ai_trader.observation.signal_themes import (
    THEME_NAMES,
    ThemeReading,
    theme_features,
    theme_reading,
    themed_gate_reason,
)
from ai_trader.shared.schemas import Side

# --- que hace el tema con el lado. Vocabulario CERRADO -------------------------------
#
# El default es el unico que no consulta el tono, y eso no es casualidad: es la misma regla
# que gobierna `SignalChannel` en el generador sintetico —"0 = MENOS edge, nunca mas"—.
# Un default olvidado tiene que degradar a "la senal no hace nada", no a "la senal manda".
SIDE_CORE = "core"  # el nucleo decide el lado; la senal no lo toca
SIDE_VETO = "veto"  # la senal puede CANCELAR el lado del nucleo, nunca invertirlo
SIDE_TONE = "tone"  # el tono decide el lado y el nucleo solo decide CUANDO
SIDE_MODES: tuple[str, ...] = (SIDE_CORE, SIDE_VETO, SIDE_TONE)

# Peso de la senal en la confianza. 0 = la capa no puede mover el tamano.
INERT_SIGNAL_WEIGHT = 0.0

# Inclinacion MAXIMA de la confianza, con peso 1 y tono direccional saturado. Es deliberado
# que sea pequena: la confianza ya vive en [0,55, 0,90] y el riesgo corta en 0,65, asi que
# una inclinacion mayor convertiria el peso en un interruptor de elegibilidad disfrazado.
CONF_TILT = 0.20

# Confianza minima y maxima que publica una primitiva, el mismo rango que momentum y
# reversion producen por construccion. Se recorta aqui para que la capa no pueda sacar la
# confianza del intervalo que el resto del sistema espera.
CONF_FLOOR = 0.55
CONF_CEILING = 0.90


def validate_signal_fields(
    *,
    min_tone: float | None = None,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
    side_mode: str | None = None,
    weight: float | None = None,
) -> None:
    """
    Rechaza en el constructor lo que no es un umbral exigente sino un error.

    Un umbral fuera del rango alcanzable no es "muy estricto": es una puerta cerrada a cal y
    canto que nadie escribio queriendo. Misma leccion que `min_relative_strength = -1.0`,
    que se documento como "sin filtro" y si podia bloquear.
    """
    if min_tone is not None and not INERT_MIN_TONE <= min_tone <= -INERT_MIN_TONE:
        raise ValueError(f"min_signal_tone must be between {INERT_MIN_TONE} and {-INERT_MIN_TONE}")
    if min_intensity is not None and not (
        INERT_MIN_INTENSITY <= min_intensity <= INERT_MAX_INTENSITY
    ):
        raise ValueError(
            f"min_signal_intensity must be between {INERT_MIN_INTENSITY} and {INERT_MAX_INTENSITY}"
        )
    if max_intensity is not None and not (
        INERT_MIN_INTENSITY <= max_intensity <= INERT_MAX_INTENSITY
    ):
        raise ValueError(
            f"max_signal_intensity must be between {INERT_MIN_INTENSITY} and {INERT_MAX_INTENSITY}"
        )
    if min_intensity is not None and max_intensity is not None and min_intensity > max_intensity:
        raise ValueError("min_signal_intensity must be <= max_signal_intensity")
    if side_mode is not None and side_mode not in SIDE_MODES:
        raise ValueError(f"signal_side_mode must be one of {SIDE_MODES} (got '{side_mode}')")
    if weight is not None and not 0.0 <= weight <= 1.0:
        raise ValueError("signal_weight must be between 0 and 1")


def directional_tone(reading: ThemeReading | None, side: Side) -> float:
    """
    El tono A FAVOR del lado: `+tono` para un largo, `-tono` para un corto.

    Vive en el mismo [-Z_CLIP, Z_CLIP] que el tono crudo, que es lo que hace que un piso
    puesto en el borde siga siendo inerte para las dos direcciones.
    """
    if reading is None:
        return 0.0
    return reading.tone if side is Side.BUY else -reading.tone


def resolve_side(
    core_side: Side | None,
    reading: ThemeReading | None,
    *,
    side_mode: str,
    threshold: float,
) -> Side | None:
    """
    El lado final: el del nucleo, cancelado por la senal, o el que dicta el tono.

    Devuelve None = no se opera. Con `SIDE_CORE` (el default) devuelve `core_side` TAL CUAL,
    sin mirar la lectura, asi que la capa es transparente por construccion y no por que la
    lectura resulte estar vacia.

    Un tema que no se puede leer NUNCA cambia el lado, en ninguno de los tres modos. En
    `SIDE_TONE` eso significa caer al nucleo: la primitiva sigue operando lo que el precio
    dice, que es su variante ciega, y no deja de operar por falta de datos.
    """
    if side_mode not in SIDE_MODES:
        raise ValueError(f"unknown side_mode '{side_mode}'")
    if side_mode == SIDE_CORE or reading is None or not reading.readable:
        return core_side
    if side_mode == SIDE_VETO:
        # El tono en contra por encima del umbral cancela; nunca invierte. Un veto que
        # invirtiera seria otro modo, y confundirlos es como se cuela una estrategia que
        # opera justo lo contrario de lo que su nombre dice.
        if core_side is not None and directional_tone(reading, core_side) <= -abs(threshold):
            return None
        return core_side
    # SIDE_TONE: el tono manda, y el nucleo solo ha dicho que HOY se puede operar.
    if core_side is None:
        return None
    if reading.tone >= abs(threshold):
        return Side.BUY
    if reading.tone <= -abs(threshold):
        return Side.SELL
    return None


def resolve_confidence(
    base: float,
    reading: ThemeReading | None,
    side: Side,
    *,
    weight: float,
) -> float:
    """
    Inclina la confianza con el tono direccional, ponderado por `weight`.

    Con `weight = 0` devuelve `base` EXACTO —el mismo objeto flotante, sin aritmetica—, que
    es lo que hace demostrable que el default no mueve el tamano ni un centimo. Sin lectura
    legible tambien devuelve `base`: la ausencia de datos no inclina nada.
    """
    if weight <= 0.0 or reading is None or not reading.readable:
        return base
    tilt = CONF_TILT * weight * (directional_tone(reading, side) / -INERT_MIN_TONE)
    return round(min(max(base + tilt, CONF_FLOOR), CONF_CEILING), 2)


def side_gate_reason(
    features: Mapping[str, float],
    theme: str,
    side: Side,
    *,
    min_tone: float,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
) -> str | None:
    """
    La puerta de un tema para un lado concreto. Falla ABIERTA sin cobertura, como todas.

    Para un largo es literalmente `themed_gate_reason`. Para un corto se le pasa el tono ya
    volteado, de modo que el mismo `min_tone` significa lo mismo —"la senal no puede apuntar
    en contra de lo que voy a hacer"— en las dos direcciones.
    """
    if side is Side.BUY:
        return themed_gate_reason(
            features,
            theme,
            min_tone=min_tone,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
        )
    reading = theme_reading(features, theme)
    flipped = dict(features)
    tone_key, _, _ = theme_features(theme)
    flipped[tone_key] = -reading.tone
    return themed_gate_reason(
        flipped,
        theme,
        min_tone=min_tone,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
    )


def composite_reading(
    features: Mapping[str, float], *, themes: Sequence[str] = THEME_NAMES
) -> ThemeReading:
    """
    Los cinco temas en UNA lectura, que es donde la breadth paga.

    Por la ley fundamental del gestor activo, K apuestas poco correlacionadas de capacidad
    predictiva pequena valen raiz de K veces una sola. Los cinco temas son observadores
    disjuntos —un skew de Deribit y un informe COT de la CFTC no comparten fuente, proveedor
    ni sesgo—, asi que agregarlos es la unica construccion del sistema que puede cobrar esa
    raiz. Cinco estrategias que leen un tema cada una la reparten.

    Solo entran los temas LEGIBLES, y por eso el promedio es sobre ellos y no sobre los cinco:
    un tema sin cobertura no es un tema neutro, es un tema del que no se sabe nada, y meterlo
    como cero diluiria a los que si dicen algo. Es la misma distincion que `signal_coverage`
    existe para hacer.

    La cobertura del compuesto es `temas_legibles / len(themes)`, y de ahi sale gratis una
    propiedad que conviene no perder de vista: con `MIN_SIGNAL_COVERAGE = 0,25` y cinco temas
    hacen falta DOS legibles (1/5 = 0,20 < 0,25; 2/5 = 0,40). La regla del minimo de dos
    aparece sola, sin una constante nueva.
    """
    names = tuple(themes)
    if not names:
        raise ValueError("composite_reading needs at least one theme")
    readings = [theme_reading(features, theme) for theme in names]
    legible = [r for r in readings if r.readable]
    coverage = len(legible) / float(len(names))
    if not legible:
        return ThemeReading(theme="composite", tone=0.0, intensity=0.0, coverage=coverage)
    return ThemeReading(
        theme="composite",
        tone=sum(r.tone for r in legible) / len(legible),
        intensity=sum(r.intensity for r in legible) / len(legible),
        coverage=coverage,
    )


def composite_gate_reason(
    reading: ThemeReading,
    side: Side,
    *,
    min_tone: float,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
) -> str | None:
    """
    La puerta del compuesto, sobre una lectura ya agregada.

    No pasa por `themed_gate_reason` porque el compuesto no publica claves en el diccionario
    de features: es una agregacion que hace el lector, no un bloque del radar. La regla, en
    cambio, es exactamente la misma y esta escrita una sola vez aqui: **por debajo de
    `MIN_SIGNAL_COVERAGE` no se evalua nada**.
    """
    if reading.coverage < MIN_SIGNAL_COVERAGE:
        return None
    tone = directional_tone(reading, side)
    if tone < min_tone:
        return f"tono compuesto {tone:+.2f} < {min_tone:+.2f}"
    if min_intensity is not None and reading.intensity < min_intensity:
        return f"intensidad compuesta {reading.intensity:.2f} < {min_intensity:.2f}"
    if max_intensity is not None and reading.intensity > max_intensity:
        return f"intensidad compuesta {reading.intensity:.2f} > {max_intensity:.2f}"
    return None


def signal_features(reading: ThemeReading | None) -> dict[str, float]:
    """
    Lo que dijo el tema, para meterlo DENTRO de `Signal.features`.

    Ni momentum ni reversion lo hacen, y por eso hoy el diario no registra nunca que dijo el
    radar cuando dejo pasar una operacion. Registrarlo es lo que hace distinguible una capa
    de senal que resulto inutil de una que nunca llego a consultarse: sin el rastro, las dos
    se leen igual en el informe.

    Devuelve un diccionario VACIO cuando no hubo lectura, en vez de tres ceros: un cero en
    `Signal.features` diria "el tema estaba neutro", que es justo lo contrario de "no se
    miro". Es la misma distincion que `signal_coverage` existe para hacer.
    """
    if reading is None:
        return {}
    return {
        "signal_theme_tone": reading.tone,
        "signal_theme_intensity": reading.intensity,
        "signal_theme_coverage": reading.coverage,
    }


def atr_bracket(
    entry: float,
    atr_value: float,
    side: Side,
    *,
    stop_mult: float,
    target_mult: float,
    stop_anchor: float | None = None,
) -> tuple[float, float] | None:
    """
    Stop y objetivo en multiplos de ATR, con el signo del lado. None si degeneran.

    Vive aqui —y no seis veces— por el mismo motivo que el ATR vive en
    `shared/indicators.py`. La comprobacion de degeneracion no es defensiva: un stop <= 0 es
    alcanzable en un activo muy volatil con un multiplo alto, y `mean_reversion` ya descarta
    ese caso en vez de emitir una senal que el riesgo tendria que rechazar despues.

    `stop_anchor` permite colgar el stop del extremo de la barra en vez de del cierre, que es
    lo que quiere una primitiva de capitulacion: el stop tiene que quedar al otro lado de la
    mecha, no al otro lado del cierre.
    """
    if not (entry > 0 and atr_value > 0 and stop_mult > 0 and target_mult > 0):
        return None
    anchor = entry if stop_anchor is None else stop_anchor
    if side is Side.BUY:
        stop_loss = anchor - atr_value * stop_mult
        take_profit = entry + atr_value * target_mult
        if stop_loss <= 0 or stop_loss >= entry or take_profit <= entry:
            return None
    else:
        stop_loss = anchor + atr_value * stop_mult
        take_profit = entry - atr_value * target_mult
        if take_profit <= 0 or stop_loss <= entry or take_profit >= entry:
            return None
    return float(stop_loss), float(take_profit)


__all__ = [
    "CONF_CEILING",
    "CONF_FLOOR",
    "CONF_TILT",
    "INERT_SIGNAL_WEIGHT",
    "SIDE_CORE",
    "SIDE_MODES",
    "SIDE_TONE",
    "SIDE_VETO",
    "atr_bracket",
    "composite_gate_reason",
    "composite_reading",
    "directional_tone",
    "resolve_confidence",
    "resolve_side",
    "side_gate_reason",
    "signal_features",
    "validate_signal_fields",
]
