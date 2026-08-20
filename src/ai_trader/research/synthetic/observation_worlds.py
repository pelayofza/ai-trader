"""
MUNDOS DE OBSERVACION: que ve un observador imperfecto, tema a tema.

QUE ES ESTO
-----------
`retrofit.py` enriquece la FISICA del mundo —colas, clustering, saltos, carga serial—. Este
modulo enriquece lo que se puede OBSERVAR de el: declara los canales
(`scenarios.SignalChannel`) que un escenario publica, uno por tema del radar.

Vive aparte de `retrofit.py` por tres motivos, y ninguno es de gusto:

1. Las constantes de `retrofit` estan calibradas contra `data/fidelity/report_ai_v3.json`.
   Las de aqui se calibran contra OTROS dos ficheros —`data/signals/history_depth.json` y
   `data/signal_channel/report_ai_v3.json`—. Dos bucles de calibracion en un fichero
   convierten el fichero en un cajon.
2. `retrofit.enrich_spec` es la funcion de la que ai_v2 y ai_v3 se regeneran. La forma mas
   barata de garantizar que no se mueve es no abrir el fichero.
3. Un canal no es microestructura del proceso de precios. El motor NI SIQUIERA LO LEE: la
   emision es un pase aparte sobre barras ya cerradas (ver `signal_channel.py`), y por eso
   declarar canales no puede mover una sola vela.

LA REGLA DE CALIBRACION, DECLARADA ANTES DE LOS NUMEROS
--------------------------------------------------------
    coverage          ~ profundidad medida del tema, CON SUELO (ver mas abajo)
    informative_share ~ fraccion de features del tema con POLARIDAD declarada en
                        `observation/signal_radar.py::POLARITY`
    lead_days         = el horizonte del MECANISMO del tema, no un parametro libre
    noise_ar          = la persistencia de la serie observable del tema
    rho               = fuerza del mecanismo CUANDO dispara
    corr_group        = uno por tema (ver la nota de breadth)

EL SUELO DE `coverage`, QUE ES UNA CONCESION AL ESTIMADOR Y NO AL MUNDO
------------------------------------------------------------------------
La cobertura REAL de dos de los cinco temas es 0 y 1 fuentes backtesteables. Traducirla
literalmente hace que la libreria NO PUEDA PASAR SU PROPIO ESTUDIO DE FIDELIDAD, y no porque
el mundo este mal simulado sino porque el estimador se queda sin muestra. Medido sobre 36
series de ai_v3 (rho=0,10, h=3, phi=0,30, informative=0,5):

    coverage 0,20 ->   0 de 36 series certificables (`fidelity.MIN_OBSERVATIONS` = 200)
    coverage 0,25 ->   3 de 36
    coverage 0,30 ->  34 de 36, pero `past_leak` mediano 0,097 contra una tolerancia de 0,10
    coverage 0,40 ->  36 de 36, `past_leak` 0,079
    coverage 0,50 ->  36 de 36, `past_leak` 0,077

`past_leak` es el maximo de cinco correlaciones que DEBERIAN ser ruido, y su mediana sigue
casi exactamente 1,5 / sqrt(n): con poca muestra suspende sin que se fugue nada. Asi que el
suelo es 0,50, y el sesgo va en la direccion segura —declarar mas cobertura de la que hay
hace el mundo MAS favorable a la senal, asi que el break-even que se publique es una cota
optimista, que es como esta construida toda esta pieza—.

DEUDA DECLARADA: la correccion de fondo es que `CHANNEL_LEAK_TOLERANCE` sea consciente de la
muestra (k / sqrt(n) con k declarado) en vez de una constante. Es barata —`channel_checks`
no ha corrido nunca sobre ninguna libreria publicada, asi que cambiarla no invalida ni una
cifra— pero exige que `measure_channels` propague `n`, que hoy no lo hace.

CINCO GRUPOS DE CORRELACION, Y QUE MIDE ESA ELECCION
-----------------------------------------------------
Cada tema es su propio grupo. Tres razones:

- Con grupo compartido no habria cinco canales: la semilla de flujo es
  `_stream_seed(seed, group, symbol)`, asi que dos canales del mismo grupo comparten el ruido
  y las DOS mascaras, y como las mascaras salen de comparar los mismos uniformes contra
  umbrales distintos quedan anidadas. Serian la misma apuesta escalada.
- Es LA PREGUNTA. La ley fundamental dice que K apuestas independientes de IC pequeno valen
  raiz de K. Con estos cinco canales el IC agregado es sqrt(suma de cuadrados) = 0,074
  contra 0,048 del mejor solo: la breadth vale x1,55, y eso es medible sin un solo backtest.
- En el mundo real los observadores SON disjuntos: un skew de Deribit y un informe COT de la
  CFTC no comparten fuente, ni proveedor, ni sesgo.

SESGO DECLARADO: los temas reales SI estan correlacionados —en un crash disparan liquidacion,
volatilidad y flujo a la vez— y `corr_group` es binario, no expresa correlacion parcial.
Declarar cinco grupos independientes coloca a esta libreria en el EXTREMO OPTIMISTA del eje
de breadth, que es la direccion correcta para una cota: si ni con la estructura de
correlacion mas favorable se alcanza el frente, no se alcanza.

UN JUEGO UNICO PARA LOS TREINTA ESCENARIOS
-------------------------------------------
Ni por escenario ni por fase, y no por comodidad: `channel_values` sortea UNA mascara de
probabilidad constante sobre los 730 dias y el emisor solo ve `closes`, sin la linea temporal
de fases. Ademas `fidelity_study.measure_channels` colapsa los canales por NOMBRE (gana el
ultimo escenario), asi que declaraciones distintas por escenario certificarian la mediana
agrupada contra la declaracion de uno solo. Con declaraciones identicas es inofensivo, y hay
un test que lo fija. Y por ultimo confundiria el hold-out: el escenario es la unidad del
split, asi que "regimen mas dificil" y "senal mejor" quedarian confundidos en la particion.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from ai_trader.research.synthetic.retrofit import enrich_spec
from ai_trader.research.synthetic.scenarios import ScenarioSpec, SignalChannel

# Suelo de cobertura declarada. Ver el docstring: es una concesion al ESTIMADOR (la muestra
# minima que `fidelity.channel_facts` necesita para certificar un canal), no una afirmacion
# sobre el mundo. Un test lo congela para que nadie lo baje sin releer por que esta.
MIN_DECLARED_COVERAGE = 0.50


# --- los cinco canales de ai_v4 -------------------------------------------------------
#
# Los nombres son los de los temas de `observation/signal_themes.py` a proposito: el informe
# del barrido no tiene que traducir entre dos vocabularios. Que la tabla de temas del panel
# sintetico apunte TODOS los temas a TODOS los canales (ver `SignalPanel.theme_table`) no lo
# contradice: ahi lo que se mide es el break-even de un diseno, y las seis primitivas tienen
# que enfrentarse al mismo canal para ser comparables entre si.

V4_CHANNELS: tuple[SignalChannel, ...] = (
    # El unico tema cuyo contenido es un MECANISMO ("hay 200 M$ que revientan un 3% mas
    # abajo") y no una correlacion, asi que se lleva el rho mas alto. `informative_share`
    # baja: el mapa solo dice algo cuando el precio se acerca al cumulo. Dos dias de
    # horizonte porque una cascada se resuelve en horas, y `noise_ar` alto porque el
    # apalancamiento es una variable de estado: el mapa de hoy se parece al de ayer.
    # COBERTURA REAL 0 de 4 fuentes backtesteables -> la fila donde el mundo declarado es
    # mas optimista que el real.
    SignalChannel(
        name="liquidation",
        rho=0.16,
        lead_days=2,
        noise_ar=0.55,
        informative_share=0.25,
        coverage=MIN_DECLARED_COVERAGE,
        corr_group="liquidation",
    ),
    # De las cuatro features de `deribit_volatility` solo `skew_25d` tiene polaridad
    # declarada —dvol, atm_iv y el slope estan explicitamente fuera de POLARITY—, y de ahi
    # el informative_share bajo. Diez dias porque la superficie precia a treinta. `noise_ar`
    # el mas alto de los cinco: una serie de volatilidad implicita es lo mas persistente del
    # catalogo. Cobertura 1 de 2 fuentes, medida diaria.
    SignalChannel(
        name="vol_surface",
        rho=0.12,
        lead_days=10,
        noise_ar=0.60,
        informative_share=0.25,
        coverage=0.75,
        corr_group="vol_surface",
    ),
    # El tema con MAS profundidad medida y MENOS direccion: POLARITY no incluye ninguna de
    # sus fuentes salvo `ofac_sdn`, y el catalogo lo razona una a una ("un recuento de
    # dockets no distingue una aprobacion de una demanda"). Por eso rho y sobre todo
    # informative_share son los mas bajos: un calendario aporta CUANDO, no hacia donde.
    # Cobertura alta y `noise_ar` casi nulo: las fuentes de evento siempre cubren (tienen
    # calendario) y su contenido es espasmodico.
    SignalChannel(
        name="macro",
        rho=0.10,
        lead_days=5,
        noise_ar=0.10,
        informative_share=0.10,
        coverage=0.85,
        corr_group="macro",
    ),
    # rho bajo por la razon que el propio catalogo escribio: "la atencion sube con el miedo
    # y con la euforia", y solo dos de sus siete fuentes tienen polaridad. `noise_ar` alto
    # porque las series de atencion son de las mas autocorreladas que existen. Cobertura
    # real 2 de 7 = 0,29 -> suelo.
    SignalChannel(
        name="attention",
        rho=0.10,
        lead_days=2,
        noise_ar=0.50,
        informative_share=0.25,
        coverage=MIN_DECLARED_COVERAGE,
        corr_group="attention",
    ),
    # El tema con mas profundidad Y mas direccion: once de sus doce fuentes tienen polaridad
    # razonada y ocho tienen historia medida. De ahi el informative_share mas alto y la
    # cobertura mas alta. Diez dias porque un flujo se despliega en semanas. Es el canal con
    # el expected_ic mayor de los cinco, y eso ES el resultado que la tabla transmite: si
    # algo va a funcionar, es flujo.
    SignalChannel(
        name="flow",
        rho=0.12,
        lead_days=10,
        noise_ar=0.45,
        informative_share=0.40,
        coverage=0.85,
        corr_group="flow",
    ),
)


def aggregate_expected_ic(channels: Sequence[SignalChannel] = V4_CHANNELS) -> float:
    """
    El IC que la ley fundamental predice para el AGREGADO de canales independientes.

    Raiz de la suma de cuadrados, que es lo que sale de promediar K senales incorreladas de
    varianza igual. Es la cifra contra la que se contrasta la certificacion de breadth: si el
    tono agregado del radar no se le acerca, el radar no agrega y la primitiva compuesta no
    aporta nada sobre las otras cinco.
    """
    return float(sum(c.expected_ic**2 for c in channels) ** 0.5)


def with_channels(
    channels: Sequence[SignalChannel],
) -> Callable[[ScenarioSpec], ScenarioSpec]:
    """
    Enriquecedor que DECLARA `channels`. Reemplaza, no acumula.

    Que reemplace es lo que lo hace idempotente, y la idempotencia es lo que hace que
    derivar desde ai_v1 y desde ai_v3 produzcan la MISMA libreria en vez de dos cosas con el
    mismo nombre segun de donde se corriera.
    """
    frozen = tuple(channels)

    def enrich(spec: ScenarioSpec) -> ScenarioSpec:
        return replace(spec, signals=frozen)

    return enrich


def enrich_spec_v4(spec: ScenarioSpec) -> ScenarioSpec:
    """
    ai_v4 = ai_v3 + los cinco canales tematicos.

    Aplica `enrich_spec` ADEMAS de los canales, no en lugar de. Como `enrich_spec` es
    idempotente (verificado sobre los treinta escenarios), derivar desde ai_v1 y desde ai_v3
    son la misma libreria POR CONSTRUCCION. Sin eso, "ai_v4" designaria dos mundos distintos
    segun la libreria de origen, y el nombre dejaria de significar algo.
    """
    return with_channels(V4_CHANNELS)(enrich_spec(spec))


# Enriquecedores con nombre, para que la CLI pueda ofrecerlos sin importar codigo arbitrario
# y para que `--help` liste los mundos derivables. Es documentacion que no se puede
# desincronizar del codigo porque es el codigo.
ENRICHERS: dict[str, Callable[[ScenarioSpec], ScenarioSpec]] = {
    "v2": enrich_spec,  # microestructura; produjo ai_v2 y ai_v3
    "v4": enrich_spec_v4,  # microestructura + los cinco canales tematicos
}


__all__ = [
    "ENRICHERS",
    "MIN_DECLARED_COVERAGE",
    "V4_CHANNELS",
    "aggregate_expected_ic",
    "enrich_spec_v4",
    "with_channels",
]
