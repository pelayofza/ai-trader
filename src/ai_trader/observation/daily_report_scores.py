"""
PUNTUACION EXPERTA DEL REPORTE DIARIO: 37 variables categoricas -> una decision.

QUE ES ESTO Y POR QUE NO ES EL RADAR
-------------------------------------
El radar (`signal_radar.py`) reduce treinta fuentes NUMERICAS a seis ejes, y para eso
necesita historia: normaliza contra su propio pasado. Aqui no hay pasado. La segunda via de
captura —el agente externo que escribe `data/signals_raw/ai_reports/` cada manana— empezo el
2026-08-20, asi que a dia de hoy hay TRES capturas y ningun horizonte contra el que medir
nada. Con tres dias no se estima una media, no se estandariza, no se ajusta un peso y desde
luego no se backtestea.

Lo que si se puede hacer con cero profundidad historica es AFIRMAR. Cada pregunta del
cuestionario ya viene con su escala firmada de -2 a +2 (`caida_fuerte = -2`, `entradas_fuertes
= +2`), asi que lo unico que falta para convertir 37 categorias en un numero es decir cuanto
pesa cada una y en que sentido. Eso es lo que hay en la tabla `DIRECTIONAL` de este modulo:
juicio experto escrito a mano, sin una sola cifra estimada de los datos.

    ESTOS PESOS NO ESTAN MEDIDOS. Son una hipotesis, no un resultado. Se escriben aqui,
    juntos y con su motivo, para que el dia que haya calendario suficiente se puedan
    SUSTITUIR por unos aprendidos y el diff diga exactamente que cambio.

Y por eso mismo no entran en `scoring/search_space.py`: optimizarlos contra un dia de datos
seria ajustar 32 grados de libertad con 24 observaciones perfectamente correlacionadas entre
si. La linea del proyecto para la capa de senal —"la capa se afirma, no se optimiza"— vale
aqui con mas motivo todavia.

LAS TRES COSAS QUE HACE UNA RESPUESTA, Y POR QUE NO SON LA MISMA
----------------------------------------------------------------
El propio cuestionario ya separa lo que suma de lo que no, y la separacion no es cosmetica:
las preguntas excluidas de la suma lo estan porque interpretarlas ES modelado, y el modelado
no le toca a quien captura. Aqui es donde toca. Cada una de las 37 tiene UN papel declarado:

  * DIRECCION (`DIRECTIONAL`, 32 preguntas). Suman al score con peso y polaridad. Veintinueve
    van con polaridad +1 —la escala del cuestionario ya apunta hacia donde debe—, y tres van
    con polaridad -1 porque miden AGLOMERACION y no fuerza: P16 (interes abierto), P25 (RSI)
    y P31 (funding). Un RSI de 81 no es una razon para comprar mas, es una razon para
    desconfiar del largo que ibas a abrir.
  * ANCHURA DE LA HORQUILLA (P32, P33). La volatilidad no dice hacia donde, dice cuanto se
    mueve: es la unidad en la que se miden el stop y el objetivo, no un voto.
  * MODULADORES (P34 profundidad, P35 beta). La profundidad recorta la confianza cuando el
    libro es fino; la beta escala el bloque de mercado, porque una lectura de "BTC sube
    fuerte" no significa lo mismo para un activo de beta 0,6 que para uno de beta 1,4.

Y una que no hace NADA: **P30 no se lee**. Es el sesgo global que le pone al dia el mismo
redactor que respondio P01-P29, asi que usarla seria preguntarle dos veces a la misma fuente
y contar la respuesta como dos evidencias. El lector la deja fuera de la estructura
(`ai_reports.load_day_answers`), asi que este modulo no podria usarla ni queriendo.

POR QUE LA POLARIDAD CONTRARIAN SE PARA EN TRES PREGUNTAS
----------------------------------------------------------
Es tentador seguir: la codicia extrema del Fear & Greed (P06) tambien es un clasico
contrarian, y la euforia de la comunidad (P26) tambien. El criterio para no hacerlo es que
esas dos lecturas se sostienen sobre una U —bien mientras suben, mal en el extremo— y una U
no se afirma, se mide, porque todo el contenido esta en DONDE esta el codo. Las tres que si
llevan -1 no son una U: son un coste o una posicion abarrotada, monotono en todo el rango.
La distincion se escribe aqui para que el dia que haya datos se pueda contrastar, y no para
que parezca que hubo una razon.

EL SESGO DE COBERTURA, QUE AQUI ES EL PELIGRO REAL
----------------------------------------------------
El resumen del dia publica `diagnostico_cobertura_vs_media` justamente para vigilar esto: si
el score correlaciona con CUANTAS preguntas se pudieron responder, el ranking ordena por
disponibilidad de datos y no por criterio. Dos defensas, las dos en `score_asset`:

  1. El score es una MEDIA PONDERADA sobre lo disponible, no una suma. Un activo con 15
     respuestas y otro con 25 se comparan en la misma escala [-1, +1].
  2. Hay un piso de cobertura PONDERADA (`MIN_COVERAGE`), y es una constante, no un
     parametro. Sorteable, el optimizador podria subirlo hasta convertir la estrategia en un
     filtro de "que activos tuvieron buen dia de captura".
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

# La direccion de la dependencia que declara `signals/__init__.py`: el paquete de ingesta no
# sabe que existe la capa de decision, y esta si sabe leerlo. Aqui se importa el LECTOR, no
# el archivo: `ai_reports` no toca red ni depende de nada de `observation`.
from ai_trader.shared.clock import Clock
from ai_trader.signals.ai_reports import QUESTIONNAIRE_ID as EXPECTED_QUESTIONNAIRE

logger = logging.getLogger(__name__)

# --- bloques tematicos ---------------------------------------------------------------
#
# Existen para dos cosas: escalar el de mercado por la beta del activo, y poder explicar una
# entrada por bloques en vez de por 32 numeros sueltos.
BLOCK_PRICE = "precio"
BLOCK_MARKET = "mercado"
BLOCK_WHALES = "whales"
BLOCK_INSTITUTIONAL = "institucional"
BLOCK_DERIVATIVES = "derivados"
BLOCK_NEWS = "noticias"
BLOCK_TECHNICAL = "tecnico"
BLOCK_SENTIMENT = "sentimiento"
BLOCK_RISK = "riesgo"
BLOCK_SUPPLY = "oferta"
BLOCK_CROWDING = "aglomeracion"

BLOCKS: tuple[str, ...] = (
    BLOCK_PRICE,
    BLOCK_MARKET,
    BLOCK_WHALES,
    BLOCK_INSTITUTIONAL,
    BLOCK_DERIVATIVES,
    BLOCK_NEWS,
    BLOCK_TECHNICAL,
    BLOCK_SENTIMENT,
    BLOCK_RISK,
    BLOCK_SUPPLY,
    BLOCK_CROWDING,
)

# Las claves de arriba son IDENTIFICADORES --viajan en `features` de la senal y en el JSON del
# dashboard, asi que se quedan en ASCII y estables--. Esto es lo que se ENSENA, y vive aqui y no
# en los dos generadores para que no haya dos listas que se desincronicen.
BLOCK_LABELS: dict[str, str] = {
    BLOCK_PRICE: "Precio",
    BLOCK_MARKET: "Mercado",
    BLOCK_WHALES: "Whales",
    BLOCK_INSTITUTIONAL: "Institucional",
    BLOCK_DERIVATIVES: "Derivados",
    BLOCK_NEWS: "Noticias",
    BLOCK_TECHNICAL: "Técnico",
    BLOCK_SENTIMENT: "Sentimiento",
    BLOCK_RISK: "Riesgo",
    BLOCK_SUPPLY: "Oferta",
    BLOCK_CROWDING: "Aglomeración",
}


@dataclass(frozen=True, slots=True)
class Question:
    """Lo que este modulo AFIRMA sobre una pregunta: cuanto pesa, en que sentido y de que
    bloque es. `polarity = -1` significa contrarian: la escala del cuestionario apunta al
    reves de lo que conviene a la posicion."""

    weight: float
    polarity: int
    block: str
    note: str = ""


# La escala de cada valor. El cuestionario responde en enteros de -2 a +2, asi que dividir
# entre 2 deja cada voto en [-1, +1] y hace que el score agregado viva en el mismo intervalo
# sin depender de cuantas preguntas se pudieron responder.
VALUE_SCALE = 2.0

# --- LA TABLA. Un juicio por linea, y todos revisables de un vistazo -------------------
#
# Criterio de los pesos, en tres niveles y sin mas finura porque afinar decimales sobre cero
# datos es teatro:
#   1,0  la pregunta que, sola, te haria cambiar de opinion (catalizador, flujo ETF, medias)
#   0,5-0,8  evidencia de verdad pero que necesita compania
#   0,3-0,4  contexto: mueve el margen, nunca decide
DIRECTIONAL: dict[str, Question] = {
    # --- precio del propio activo ---
    "P01": Question(0.5, +1, BLOCK_PRICE, "24h: la más ruidosa de las cuatro"),
    "P02": Question(1.0, +1, BLOCK_PRICE, "fuerza relativa a 7d: cross-sectional puro"),
    "P03": Question(0.8, +1, BLOCK_PRICE, "distancia al máximo de 90 sesiones"),
    "P04": Question(0.6, +1, BLOCK_PRICE, "volumen vs media 30d: confirma participación"),
    # --- contexto de mercado. Este bloque se escala por la beta del activo (P35) ---
    "P05": Question(0.7, +1, BLOCK_MARKET, "BTC 24h: el factor común de la clase"),
    "P06": Question(0.35, +1, BLOCK_MARKET, "nivel de Fear & Greed, leído como apetito"),
    "P07": Question(0.5, +1, BLOCK_MARKET, "cambio del F&G a 7d: la derivada, no el nivel"),
    "P08": Question(0.7, +1, BLOCK_MARKET, "risk-on / risk-off del día"),
    # --- whales y cadena ---
    "P09": Question(0.6, +1, BLOCK_WHALES, "ventas masivas de whales"),
    "P10": Question(0.8, +1, BLOCK_WHALES, "intensidad de acumulación 48-72h"),
    "P11": Question(0.6, +1, BLOCK_WHALES, "tenencias agregadas de grandes wallets"),
    "P12": Question(0.9, +1, BLOCK_WHALES, "flujo neto a exchanges: salidas = oferta que se retira"),
    # --- institucional ---
    "P13": Question(1.0, +1, BLOCK_INSTITUTIONAL, "flujos de ETF/ETP: dinero con fecha y cifra"),
    "P14": Question(0.5, +1, BLOCK_INSTITUTIONAL, "divergencia whale vs institucional"),
    "P15": Question(0.5, +1, BLOCK_INSTITUTIONAL, "actividad corporativa (tesorerías, mineras)"),
    # --- derivados (el nivel de OI va abajo, en aglomeracion) ---
    "P17": Question(0.5, +1, BLOCK_DERIVATIVES, "posicionamiento de grandes traders"),
    "P18": Question(0.6, +1, BLOCK_DERIVATIVES, "liquidaciones / squeeze en 24h"),
    # --- noticias y regulacion ---
    "P19": Question(0.7, +1, BLOCK_NEWS, "tono agregado de noticias del activo a 48h"),
    "P20": Question(1.0, +1, BLOCK_NEWS, "catalizador de alto impacto: listado, hack, upgrade"),
    "P21": Question(0.6, +1, BLOCK_NEWS, "saldo regulatorio del día"),
    # --- tecnico (el RSI va abajo, en aglomeracion) ---
    "P22": Question(0.7, +1, BLOCK_TECHNICAL, "posición frente a soporte/resistencia"),
    "P23": Question(1.0, +1, BLOCK_TECHNICAL, "precio vs medias de 50/100/200"),
    "P24": Question(0.6, +1, BLOCK_TECHNICAL, "patrón técnico reportado"),
    # --- sentimiento ---
    "P26": Question(0.4, +1, BLOCK_SENTIMENT, "sentimiento de comunidad y redes"),
    "P27": Question(0.3, +1, BLOCK_SENTIMENT, "estacionalidad del mes"),
    # --- riesgo. Ademas de votar, estas dos ESTRECHAN el objetivo (ver `event_risk`) ---
    "P28": Question(0.8, +1, BLOCK_RISK, "banderas de riesgo tipificadas activas"),
    "P29": Question(0.5, +1, BLOCK_RISK, "eventos macro de alto impacto a 7 días"),
    # --- oferta ---
    "P36": Question(0.7, +1, BLOCK_SUPPLY, "desbloqueo de tokens a 14 días"),
    "P37": Question(0.4, +1, BLOCK_SUPPLY, "emisión neta anualizada"),
    # --- AGLOMERACION: las tres de polaridad invertida -------------------------------
    #
    # El cuestionario marca P16 y P25 como `estado_crudo` y las saca de la suma diciendo,
    # con todas las letras, que la lectura contrarian es feature engineering y no captura.
    # Este es el sitio donde ese feature engineering se hace, y se hace explicito.
    "P16": Question(0.5, -1, BLOCK_CROWDING, "interés abierto extremo = posición abarrotada"),
    "P25": Question(0.7, -1, BLOCK_CROWDING, "RSI alto = recorrido ya consumido"),
    "P31": Question(0.5, -1, BLOCK_CROWDING, "funding positivo = el largo PAGA por estar"),
}

# Las tres de arriba, aparte, porque tambien deciden cuanto se acorta el objetivo cuando la
# aglomeracion apunta contra la posicion que se va a abrir.
CROWDING_QUESTIONS: tuple[str, ...] = ("P16", "P25", "P31")

# El bloque que la beta escala. Separado de `BLOCK_MARKET` como constante propia para que la
# relacion "esto es lo que la beta multiplica" no dependa de leer un bucle.
BETA_SCALED_BLOCK = BLOCK_MARKET

# P30 no aparece en la tabla y no puede aparecer: es el benchmark del redactor. La constante
# existe para que el test que lo comprueba tenga algo que nombrar.
BENCHMARK_QUESTION = "P30"

TOTAL_WEIGHT = sum(q.weight for q in DIRECTIONAL.values())
CROWDING_WEIGHT = sum(DIRECTIONAL[q].weight for q in CROWDING_QUESTIONS)

# --- las cinco que no votan ----------------------------------------------------------

# P32/P33: la volatilidad da la UNIDAD de la horquilla. Con `valor_crudo` se usa el numero;
# sin el, la categoria se traduce por el centro aproximado de su tramo. Los tramos salen del
# propio cuestionario (`derivacion`), no de una estimacion sobre datos.
VOL_FALLBACK_ANNUAL_PCT: dict[str, float] = {
    "muy_baja": 25.0,
    "baja": 45.0,
    "media": 65.0,
    "alta": 95.0,
    "muy_alta": 140.0,
}
# Cripto cotiza los siete dias, asi que la anualizada se convierte a diaria con 365 y no con
# 252. Usar 252 aqui inflaria la sigma diaria un 20% y con ella todos los stops.
CRYPTO_DAYS_PER_YEAR = 365.0
# El suelo evita que una lectura rara ("muy_baja" con crudo 3%) genere un stop de 0,2% que el
# ruido de un dia se lleva por delante. El techo evita el simetrico.
SIGMA_DAILY_FLOOR_PCT = 0.8
SIGMA_DAILY_CEILING_PCT = 12.0

# P35: cuanto del bloque de mercado hereda el activo. Un activo de beta baja no se merece la
# lectura entera de "BTC sube fuerte", y uno de beta alta se merece mas que la entera.
BETA_SCALE: dict[str, float] = {
    "muy_baja": 0.6,
    "baja": 0.8,
    "en_linea": 1.0,
    "alta": 1.2,
    "muy_alta": 1.4,
}

# P34: profundidad ejecutable. Solo puede RESTAR confianza, nunca sumarla, y sin dato vale 1
# —o sea, no penaliza—. Es la regla de todo el proyecto: una puerta no bloquea por falta de
# datos, porque "no lo se" no es "esta mal".
DEPTH_FACTOR: dict[str, float] = {
    "muy_baja": 0.35,
    "baja": 0.70,
    "media": 1.0,
    "alta": 1.0,
    "muy_alta": 1.0,
}

# P28 y P29 otra vez, ahora como RIESGO DE EVENTO. No es doble conteo: alli deciden hacia
# donde apunta el dia y aqui deciden cuanto se acorta el objetivo. Es el mismo dato haciendo
# dos trabajos distintos, y los dos estan escritos.
RISK_FLAG_LEVEL: dict[str, float] = {
    "ninguno": 0.0,
    "pocos": 0.33,
    "varios": 0.67,
    "muchos": 1.0,
}
MACRO_EVENT_LEVEL: dict[str, float] = {
    "ninguno": 0.0,
    "uno": 0.5,
    "varios": 1.0,
}

# --- constantes de la puntuacion (NO son parametros: no se optimizan) -----------------
#
# Piso de cobertura PONDERADA. Por debajo, el activo no se puntua: no es que puntue mal, es
# que no hay con que. Constante y no parametro por el motivo del docstring de arriba.
MIN_COVERAGE = 0.35

# El |score| que se considera conviccion plena a efectos de confianza y de objetivo. No es
# 1,0 porque 1,0 exige que las 32 preguntas contesten con el extremo de su escala a la vez, y
# eso no pasa: en la unica captura disponible el maximo observado es 0,53.
FULL_CONVICTION_SCORE = 0.60


@dataclass(frozen=True, slots=True)
class AssetScore:
    """La lectura de un activo en un dia. Todo lo que la estrategia necesita, ya digerido."""

    ticker: str
    score: float
    coverage: float
    n_answered: int
    sigma_daily_pct: float
    sigma_source: str
    event_risk: float
    crowding: float
    crowding_coverage: float
    depth_factor: float
    beta_scale: float
    blocks: dict[str, float] = field(default_factory=dict)
    drivers: tuple[str, ...] = ()
    # Puestos en el corte transversal del dia, 1 = el mejor por ese lado. Los rellena
    # `score_day`, que es quien ve a los 24 a la vez.
    rank_long: int = 0
    rank_short: int = 0
    n_scored: int = 0

    @property
    def strength(self) -> float:
        """|score| normalizado a [0, 1] contra la conviccion plena."""
        return min(abs(self.score) / FULL_CONVICTION_SCORE, 1.0)


def _answer(answers: dict, qid: str) -> dict | None:
    value = answers.get(qid)
    return value if isinstance(value, dict) else None


def _value(answers: dict, qid: str) -> float | None:
    """El valor entero de una respuesta, o None si no se pudo medir.

    `valor = null` cubre los dos estados vacios del cuestionario a la vez, y eso es
    exactamente lo que se quiere aqui: `sin_datos` ("no lo encontre") y `no_aplica` ("este
    activo no tiene ETF") se parecen en lo unico que importa para puntuar, que es que no hay
    voto. La diferencia entre ambos SI importa para el dataset, y por eso el fichero la
    guarda en `estado`; para una media ponderada no cambia nada.
    """
    answer = _answer(answers, qid)
    if answer is None:
        return None
    raw = answer.get("valor")
    return float(raw) if isinstance(raw, (int, float)) else None


def _option(answers: dict, qid: str) -> str | None:
    answer = _answer(answers, qid)
    option = answer.get("id_opcion") if answer else None
    return option if isinstance(option, str) else None


def _raw(answers: dict, qid: str) -> float | None:
    answer = _answer(answers, qid)
    raw = answer.get("valor_crudo") if answer else None
    return float(raw) if isinstance(raw, (int, float)) else None


def daily_sigma_pct(answers: dict) -> tuple[float, str]:
    """La sigma DIARIA en % que fija la anchura de la horquilla, y de donde salio.

    Prefiere el numero crudo a la categoria, siempre: `valor_crudo = 55.0` es un 55%
    anualizado y `baja` es un tramo entero. Es justamente el error que la v2 del
    cuestionario vino a arreglar ("v1 bucketizaba de forma destructiva"), y seria absurdo
    reintroducirlo aqui teniendo el numero al lado.

    Con realizada (P32) e implicita (P33) disponibles se promedian: la primera dice lo que ha
    pasado y la segunda lo que el mercado de opciones cobra por lo que viene, y con un dia de
    historia no hay forma de justificar preferir una. Sin ninguna de las dos devuelve la
    mediana del mapa de respaldo, que es el unico valor que no inventa una direccion.
    """
    readings: list[float] = []
    sources: list[str] = []
    for qid, label in (("P32", "realizada"), ("P33", "implicita")):
        annual = _raw(answers, qid)
        if annual is None:
            option = _option(answers, qid)
            annual = VOL_FALLBACK_ANNUAL_PCT.get(option or "")
            if annual is not None:
                label = f"{label}_categoria"
        if annual is not None and annual > 0:
            readings.append(annual)
            sources.append(label)

    if not readings:
        annual = VOL_FALLBACK_ANNUAL_PCT["media"]
        source = "respaldo"
    else:
        annual = sum(readings) / len(readings)
        source = "+".join(sources)

    daily = annual / math.sqrt(CRYPTO_DAYS_PER_YEAR)
    return min(max(daily, SIGMA_DAILY_FLOOR_PCT), SIGMA_DAILY_CEILING_PCT), source


def event_risk(answers: dict) -> float:
    """Cuanto riesgo de EVENTO trae el dia, en [0, 1]. Sin dato, 0: no penaliza."""
    levels = [
        RISK_FLAG_LEVEL.get(_option(answers, "P28") or ""),
        MACRO_EVENT_LEVEL.get(_option(answers, "P29") or ""),
    ]
    present = [x for x in levels if x is not None]
    return sum(present) / len(present) if present else 0.0


def crowding(answers: dict) -> tuple[float, float]:
    """Aglomeracion en [-1, +1] y que fraccion de ella se pudo medir.

    Positivo = el LARGO esta abarrotado y pagando. Se calcula con la escala del cuestionario
    SIN invertir (a diferencia del score, donde estas tres entran con polaridad -1): aqui lo
    que interesa es hacia que lado esta la multitud, no que le conviene a la posicion.

    Devuelve tambien la COBERTURA del bloque, y no es un extra: el 2026-08-22 solo P25 tiene
    dato en 22 de los 24 activos, asi que la media ponderada de una sola pregunta con RSI
    `muy_alto` da exactamente +1,00 —aglomeracion maxima— cuando lo unico que se sabe es que
    el RSI esta alto. Quien use esto para acortar un objetivo tiene que poder escalar por
    cuanto sabe; si no, una pregunta de tres manda como si fueran tres.
    """
    total = 0.0
    weight = 0.0
    for qid in CROWDING_QUESTIONS:
        value = _value(answers, qid)
        if value is None:
            continue
        w = DIRECTIONAL[qid].weight
        total += w * (value / VALUE_SCALE)
        weight += w
    if not weight:
        return 0.0, 0.0
    return total / weight, weight / CROWDING_WEIGHT


def score_asset(ticker: str, answers: dict) -> AssetScore | None:
    """Puntua un activo. `None` si no llega al piso de cobertura ponderada.

    El score es la media ponderada de los votos DISPONIBLES, no una suma: asi un activo con
    18 respuestas y otro con 25 se leen en la misma escala [-1, +1] y la cobertura no se cuela
    como una prima de "hoy se pudo medir mucho de este".
    """
    beta = BETA_SCALE.get(_option(answers, "P35") or "", 1.0)

    total = 0.0
    weight_available = 0.0
    weight_total = 0.0
    blocks: dict[str, float] = dict.fromkeys(BLOCKS, 0.0)
    contributions: list[tuple[float, str]] = []
    n_answered = 0

    for qid, question in DIRECTIONAL.items():
        w = question.weight * (beta if question.block == BETA_SCALED_BLOCK else 1.0)
        weight_total += w
        value = _value(answers, qid)
        if value is None:
            continue
        contribution = w * question.polarity * (value / VALUE_SCALE)
        total += contribution
        weight_available += w
        blocks[question.block] += contribution
        contributions.append((contribution, qid))
        n_answered += 1

    coverage = weight_available / weight_total if weight_total else 0.0
    if weight_available <= 0 or coverage < MIN_COVERAGE:
        logger.info(
            "Reporte diario | %s sin cobertura suficiente (%.2f < %.2f): no se puntua",
            ticker, coverage, MIN_COVERAGE,
        )
        return None

    score = total / weight_available
    sigma, sigma_source = daily_sigma_pct(answers)
    crowd, crowd_coverage = crowding(answers)
    drivers = tuple(
        qid for _, qid in sorted(contributions, key=lambda c: -abs(c[0]))[:3]
    )

    return AssetScore(
        ticker=ticker,
        score=score,
        coverage=coverage,
        n_answered=n_answered,
        sigma_daily_pct=sigma,
        sigma_source=sigma_source,
        event_risk=event_risk(answers),
        crowding=crowd,
        crowding_coverage=crowd_coverage,
        depth_factor=DEPTH_FACTOR.get(_option(answers, "P34") or "", 1.0),
        beta_scale=beta,
        blocks={k: round(v / weight_available, 4) for k, v in blocks.items()},
        drivers=drivers,
    )


def score_day(day: dict) -> dict[str, AssetScore]:
    """Puntua los activos de un dia y les pone su puesto en el corte transversal.

    EL PUESTO ES LA MITAD DEL TRABAJO, Y NO SE PUEDE CALCULAR ACTIVO A ACTIVO
    ------------------------------------------------------------------------
    El runner recorre el universo EN EL ORDEN DEL CONFIG y para cuando llega a
    `max_trades_per_cycle`. Con un umbral absoluto y un dia en que todo apunta al mismo lado
    —que es exactamente lo que paso el 2026-08-22: 24 de 24 con media positiva— el que acaba
    operando no es el mejor, es el que estaba antes en la lista. Ordenar aqui, viendo a los 24
    a la vez, es lo que convierte eso en una eleccion.

    Devuelve solo los que superan el piso de cobertura. Un dia entero por debajo del piso
    devuelve un diccionario vacio, y la estrategia no opera: es el comportamiento correcto.
    """
    questionnaire = day.get("questionnaire")
    if questionnaire != EXPECTED_QUESTIONNAIRE:
        logger.warning(
            "Reporte diario | cuestionario %r; las tablas de este modulo estan escritas "
            "contra %r. No se puntua nada: los ids podrian no significar lo mismo.",
            questionnaire, EXPECTED_QUESTIONNAIRE,
        )
        return {}

    scores: dict[str, AssetScore] = {}
    for ticker, payload in (day.get("assets") or {}).items():
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            continue
        scored = score_asset(ticker, answers)
        if scored is not None:
            scores[ticker] = scored

    if not scores:
        return {}

    n = len(scores)
    longs = sorted(scores, key=lambda t: -scores[t].score)
    shorts = sorted(scores, key=lambda t: scores[t].score)
    rank_long = {t: i + 1 for i, t in enumerate(longs)}
    rank_short = {t: i + 1 for i, t in enumerate(shorts)}

    return {
        ticker: replace(
            scored,
            rank_long=rank_long[ticker],
            rank_short=rank_short[ticker],
            n_scored=n,
        )
        for ticker, scored in scores.items()
    }


def ticker_for(symbol: str) -> str | None:
    """`BTC/USDT` -> `BTC`. `None` para lo que no es un par de cripto al contado.

    Los mercados de prediccion (`PM::...`) y la renta variable no tienen reporte diario: el
    universo del agente externo es `config/assets.json`, y son 24 criptomonedas.
    """
    clean = symbol.strip().upper()
    if not clean or clean.startswith("PM::") or "/" not in clean:
        return None
    base = clean.split("/", 1)[0].strip()
    return base or None


def _parse_utc(stamp: str | None) -> datetime | None:
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class DailyReportProvider:
    """El colaborador que sirve la lectura del dia, con la misma costura que el radar.

    Misma forma que `MarketRegimeProvider` y `ThemedSignalRadarProvider` —se inyecta con un
    `attach_*` duck-typed y lleva su propio reloj—, asi que engancharlo es anadir una linea al
    diccionario de ensambladores y no un caso especial.

    Con `day = None` (sin captura, clon recien hecho, backtest) todo devuelve None y la
    estrategia no emite nada. Eso NO es un fallo silencioso: es la unica respuesta correcta
    cuando lo que la estrategia lee no existe.
    """

    def __init__(self, day: dict | None, clock: Clock) -> None:
        self._day = day if isinstance(day, dict) else None
        self._clock = clock
        self._scores = score_day(self._day) if self._day else {}
        if self._day and not self._scores:
            logger.warning(
                "Reporte diario | captura del %s cargada pero ningun activo puntuable",
                self._day.get("date"),
            )

    @property
    def date(self) -> str | None:
        return self._day.get("date") if self._day else None

    @property
    def cutoff_utc(self) -> str | None:
        return self._day.get("cutoff_utc") if self._day else None

    @property
    def n_scored(self) -> int:
        return len(self._scores)

    def tickers(self) -> frozenset[str]:
        return frozenset(self._scores)

    def age_hours(self, ticker: str) -> float | None:
        """Horas transcurridas desde la HORA DE CORTE de ese activo, no desde la del dia.

        Por activo y no por dia porque el corte es un atributo del fichero: el agente puede
        tardar tres horas en recorrer los 24 y lo que fecha la respuesta de SUI es cuando se
        midio SUI. `None` si no hay corte legible, y quien pregunta trata eso como caducado.
        """
        payload = (self._day or {}).get("assets", {}).get(ticker)
        cutoff = _parse_utc((payload or {}).get("cutoff_utc")) or _parse_utc(self.cutoff_utc)
        if cutoff is None:
            return None
        return (self._clock.now() - cutoff).total_seconds() / 3600.0

    def reading(self, symbol: str) -> AssetScore | None:
        """La lectura del simbolo, o `None` si no la hay. No comprueba frescura: eso es
        politica de la estrategia y viaja en su config."""
        ticker = ticker_for(symbol)
        return self._scores.get(ticker) if ticker else None
