"""
EL CANAL DE OBSERVACION SINTETICO: como se ve, desde fuera, algo que el mundo ya decidio.

QUE SE SIMULA AQUI, Y QUE NO
----------------------------
No se simula una senal. Simular "el sentimiento de Twitter" —su nivel, su estacionalidad,
su distribucion, su reaccion a un hack— pide un generador aprendido de datos, y con el
llega la circularidad que este proyecto lleva evitando desde el principio: el mundo contra
el que se mide lo habriamos ajustado nosotros.

Lo que se simula es el CANAL DE OBSERVACION: dado que el mundo ya ha decidido el retorno
de los proximos h dias (lo genero `PathEngine`, y esta escrito en las velas), ¿que ve un
observador imperfecto? La respuesta cuesta cinco numeros interpretables:

    senal_t = rho * z(retorno_t->t+h) + sqrt(1 - rho^2) * ruido_t

y el producto del barrido no son "los mejores parametros" sino el rho de BREAK-EVEN, que
es una propiedad del DISENO —de la estrategia, sus costes y su puerta— y no de ningun
historico. No se puede sobreajustar a un historico que nunca entro.

TRES PROPIEDADES QUE NO SON DE ADORNO
-------------------------------------
1. LA CAUSALIDAD VA DEL MUNDO A LA SENAL. La senal se deriva de un retorno futuro YA
   CALCULADO (`shift(-h)` sobre las barras cerradas). El mundo no cambia porque se le
   observe; el unico que sufre el adelanto es quien lee la senal.

2. EL PASE ES APARTE, y por eso este modulo existe en vez de un campo mas en
   `PathEngine.generate`. El emisor recibe barras ya cerradas y trae su propio generador
   aleatorio, asi que "no interfiere con la secuencia RNG del motor" no es una promesa que
   haya que auditar leyendo codigo: es imposible por construccion. Las dos SHA que congela
   `tests/test_synthetic.py::TestEngineByteIdentity` no pueden moverse desde aqui.

3. LAS SENALES ENTRAN POR EL CONTRATO DE PRODUCCION. No hay un canal paralelo hacia la
   estrategia: se construye un `SignalRadarProvider` —la MISMA clase que corre en vivo—
   con sus mismas dos z, su misma cobertura, su mismo recorte anti look-ahead y su misma
   puerta (`signal_gate_reason`). Lo unico distinto es el CATALOGO de fuentes que se le
   inyecta: las simuladas nunca aparecen en `signals/catalog.py::CATALOG`, porque un
   catalogo que mezcla fuentes reales y simuladas deja de poder auditarse. Si el canal
   entrara por otra puerta, el barrido mediria una estrategia que no es la que opera.

LO QUE ESTE MODULO NO SABE
--------------------------
Ni que es un backtest, ni que es una estrategia, ni que es rho "bueno". Produce senales y
las envuelve en un proveedor. Quien barre y quien decide es `scoring/signal_study.py`.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_trader.observation.signal_radar import SignalRadarProvider
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import Clock
from ai_trader.shared.entities import EntityKind, resolve_entity
from ai_trader.shared.signals import AGG_LAST, DAY, ENTITY, normalize_signals
from ai_trader.signals.catalog import (
    PIT_DERIVED_FROM_PRICE,
    SCOPE_ASSET,
    TIER_STATISTICAL,
    SignalFeature,
    SignalSource,
)
from ai_trader.synthetic.engine import ar1_series
from ai_trader.synthetic.scenarios import SignalChannel

logger = logging.getLogger(__name__)

# Prefijo de las fuentes simuladas. Sirve para dos cosas: que un informe distinga de un
# vistazo una clave sintetica de una real, y que un test pueda exigir que ninguna clave
# con este prefijo aparezca nunca en el catalogo de fuentes de verdad.
SOURCE_PREFIX = "synthetic_"

# Semilla del pase de emision. Es SUYA, distinta de la del motor de paths y sin relacion
# con ella: el emisor no puede tocar la secuencia del generador ni por accidente.
DEFAULT_EMISSION_SEED = 7_654_321

# El canal es una z: positivo = el futuro que anticipa es al alza. La polaridad no es una
# hipotesis sobre el mundo (que es lo que la tabla POLARITY del radar declara fuente a
# fuente, una por una y razonada) sino una propiedad de COMO se emite, asi que se fija
# aqui, en +1, y se inyecta en el radar en vez de escribirse en su tabla.
CHANNEL_POLARITY = 1.0


def source_key(channel: SignalChannel) -> str:
    """Clave de la fuente simulada de un canal."""
    return f"{SOURCE_PREFIX}{channel.name}"


def feature_name(channel: SignalChannel) -> str:
    """Nombre de la columna que publica un canal. Coincide con su clave de fuente: una
    fuente, una feature, y asi el informe no tiene que traducir entre dos vocabularios."""
    return source_key(channel)


def channel_source(channel: SignalChannel) -> SignalSource:
    """
    La declaracion de fuente de un canal, en el mismo vocabulario que el catalogo real.

    Se construye al vuelo y NO se registra en `CATALOG`: `history_from=None` porque una
    fuente simulada no tiene historia medida que declarar, y el radar no lee ese campo
    (solo lo leen las auditorias del catalogo, que hablan del mundo real).

    `entity_kind=TOKEN` es una limitacion declarada: una fuente declara UNA clase de
    entidad, asi que un universo mixto (cripto + renta variable) publicaria acciones bajo
    una fuente declarada de token. El radar no lo usa para nada —cruza por clave de
    entidad— y el barrido corre sobre universo cripto, pero conviene saberlo antes de
    emitir sobre los 35 activos del universo sintetico completo.
    """
    return SignalSource(
        key=source_key(channel),
        title=f"Canal sintetico '{channel.name}'",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature(
                feature_name(channel),
                AGG_LAST,
                "z",
                "Observacion ruidosa y adelantada del retorno futuro.",
            ),
        ),
        pit=PIT_DERIVED_FROM_PRICE,
        license="Simulado: no procede.",
        notes=(
            f"rho={channel.rho}, h={channel.lead_days}, phi={channel.noise_ar}, "
            f"informative_share={channel.informative_share}, coverage={channel.coverage}, "
            f"corr_group='{channel.corr_group}'. NO es una fuente del catalogo real: se "
            "construye al vuelo para el barrido y nunca entra en CATALOG."
        ),
    )


@dataclass(frozen=True, slots=True)
class SignalPanel:
    """
    Lo que produce el pase de emision: las senales de una muestra, listas para el radar.

    `frames` esta en la forma canonica de la ingesta —`(entity, day) -> feature`, la misma
    que devuelve `signals/feed.py::load_frames`— a proposito: es lo que permite que el
    proveedor sea el de produccion y no uno de mentira.
    """

    channels: tuple[SignalChannel, ...]
    frames: dict[str, pd.DataFrame]
    sources: tuple[SignalSource, ...]
    polarity: dict[str, float]
    seed: int
    # Simbolos que no llegaron a emitir, con el motivo. Un hueco declarado es un dato; un
    # hueco silencioso convierte el barrido en una medicion sobre un universo desconocido.
    omitted: tuple[tuple[str, str], ...] = ()
    # Valores emitidos por (canal, simbolo), con NaN en los dias sin observacion. Es la
    # materia prima con la que se CERTIFICA el canal (IC medido, autocorrelacion,
    # lead-lag): sin ella habria que reconstruirla desde el frame ya agregado.
    values: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.frames

    def provider(self, clock: Clock) -> SignalRadarProvider:
        """El radar de produccion, con el catalogo y la polaridad de los canales."""
        return SignalRadarProvider(
            self.frames, clock, sources=self.sources, polarity=self.polarity
        )

    def provider_factory(self) -> Callable[[Clock], SignalRadarProvider]:
        """La costura que consume `BacktestEngine`: reloj -> proveedor."""
        return self.provider

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "channels": [c.to_dict() for c in self.channels],
            "sources": [s.key for s in self.sources],
            "n_entities": {
                key: int(frame.index.get_level_values(ENTITY).nunique())
                for key, frame in sorted(self.frames.items())
            },
            "rows": {key: int(len(frame)) for key, frame in sorted(self.frames.items())},
            "omitted": [{"symbol": s, "reason": r} for s, r in self.omitted],
        }


EMPTY_PANEL = SignalPanel(channels=(), frames={}, sources=(), polarity={}, seed=0)


def emit_signals(
    bars: Mapping[str, pd.DataFrame],
    channels: Sequence[SignalChannel],
    *,
    seed: int = DEFAULT_EMISSION_SEED,
) -> SignalPanel:
    """
    EL PASE DE EMISION. Barras ya cerradas + canales -> senales.

    Determinista: mismas (barras, canales, semilla) -> mismos numeros, en cualquier proceso
    y en cualquier orden. La semilla de cada flujo sale de un SHA-256 de (grupo, simbolo) y
    no del `hash()` de Python, que esta salado por proceso y romperia el barrido en cuanto
    corre con varios workers.

    Sin canales devuelve el panel vacio, y un panel vacio construye un radar vacio: el
    mundo sin senales es el caso por defecto, no un caso degradado.
    """
    channels = tuple(channels)
    if not channels:
        return EMPTY_PANEL

    entities, omitted = _entities_for(bars)
    frames: dict[str, pd.DataFrame] = {}
    values: dict[tuple[str, str], np.ndarray] = {}

    for channel in channels:
        column = feature_name(channel)
        blocks: list[pd.DataFrame] = []
        for symbol, entity in entities.items():
            closes = bar_schema.series(bars[symbol], bar_schema.CLOSE).to_numpy(dtype=float)
            emitted = channel_values(closes, channel, seed=seed, symbol=symbol)
            values[(channel.name, symbol)] = emitted
            observed = np.isfinite(emitted)
            if not observed.any():
                continue
            blocks.append(
                pd.DataFrame(
                    {
                        ENTITY: entity,
                        DAY: bars[symbol].index[observed],
                        column: emitted[observed],
                    }
                )
            )
        if blocks:
            # A la forma canonica por la puerta de siempre: la agregacion a dia, el huso y
            # la columna `observed` los pone `shared/signals.py`, no este modulo.
            frames[source_key(channel)] = normalize_signals(
                pd.concat(blocks, ignore_index=True), features=[column], agg=AGG_LAST
            )

    return SignalPanel(
        channels=channels,
        frames=frames,
        sources=tuple(channel_source(c) for c in channels),
        polarity={feature_name(c): CHANNEL_POLARITY for c in channels},
        seed=seed,
        omitted=omitted,
        values=values,
    )


def channel_values(
    closes: np.ndarray, channel: SignalChannel, *, seed: int, symbol: str
) -> np.ndarray:
    """
    La serie que emite UN canal para UN simbolo. NaN = ese dia no hubo observacion.

        senal_t = a_t * rho * z(r_t->t+h) + sqrt(1 - (a_t * rho)^2) * ruido_t

    con `a_t` en {0, 1} segun el dia sea informativo o no. Tres decisiones que son las que
    hacen que el barrido mida lo que dice medir:

    - VARIANZAS IGUALADAS. El dia no informativo emite ruido de la MISMA varianza que el
      informativo (por eso el sqrt de arriba). Un falso positivo tiene que ser
      indistinguible de un acierto mirando solo la senal; si fuera mas pequeno, la
      estrategia podria filtrarlos por tamano y el canal seria mejor de lo declarado.
    - EL RUIDO ES UNA SERIE, NO CONFETI. `noise_ar` entra por el MISMO AR(1) con
      variance-matching que usa el motor para el componente idiosincratico
      (`engine.ar1_series`), no por una copia.
    - LOS ULTIMOS h DIAS NO TIENEN FUTURO que observar, asi que emiten ruido puro. Es la
      unica forma honesta de terminar la serie: inventarles informacion seria mirar mas
      alla del final del mundo.

    El orden de los sorteos (ruido, informativos, cobertura) es fijo y no depende del
    canal: es lo que hace que dos canales del MISMO grupo de correlacion compartan
    exactamente el mismo ruido y las mismas mascaras, es decir, que sean la misma apuesta
    repetida en vez de dos apuestas.
    """
    n = int(len(closes))
    if n == 0:
        return np.empty(0, dtype=float)

    rng = np.random.default_rng(_stream_seed(seed, channel.group, symbol))
    noise = ar1_series(rng.standard_normal(n), 1.0, np.full(n, channel.noise_ar))
    informative = rng.random(n) < channel.informative_share
    observed = rng.random(n) < channel.coverage

    z = forward_z(closes, channel.lead_days)
    load = np.where(informative & np.isfinite(z), channel.rho, 0.0)
    value = load * np.nan_to_num(z, nan=0.0) + np.sqrt(1.0 - load * load) * noise
    return np.where(observed, value, np.nan)


def forward_z(closes: np.ndarray, lead_days: int) -> np.ndarray:
    """
    z del retorno FUTURO t -> t+h. NaN en los ultimos h dias, que no tienen futuro.

    La tipificacion usa la media y la desviacion de la serie ENTERA del camino, y eso no es
    un look-ahead camuflado: es una propiedad del mundo simulado —la escala del canal— y no
    una estimacion que ningun participante haga. Lo que la estrategia ve pasa despues por
    el radar, que re-normaliza con una z EXPANSIVA y causal (`signals/normalize.py`), asi
    que esa escala global se cancela casi entera. Fijarla aqui es lo que hace que `rho` sea
    literalmente un coeficiente de correlacion y no un numero sin unidades.
    """
    if lead_days < 1:
        raise ValueError("lead_days must be >= 1")
    n = int(len(closes))
    out = np.full(n, np.nan)
    if n <= lead_days:
        return out

    prices = np.asarray(closes, dtype=float)
    valid = np.isfinite(prices) & (prices > 0)
    logp = np.where(valid, np.log(np.where(valid, prices, 1.0)), np.nan)
    forward = logp[lead_days:] - logp[:-lead_days]

    finite = forward[np.isfinite(forward)]
    if finite.size < 2:
        return out
    scale = float(finite.std())
    if scale <= 0:
        return out
    out[:-lead_days] = (forward - float(finite.mean())) / scale
    return out


def _entities_for(
    bars: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    """
    Simbolo -> entidad, con los descartes DECLARADOS.

    Dos motivos para quedarse fuera y los dos se publican: el simbolo no resuelve a
    ninguna entidad (`shared/entities.py`), o su entidad ya la ocupa otro simbolo. El
    segundo caso no puede pasar en el universo de hoy pero pasaria con "BTC/USDT" y
    "BTC/USD" a la vez: como el frame canonico se indexa por entidad, el segundo pisaria al
    primero EN SILENCIO y el canal quedaria emitido sobre un universo distinto del que se
    cree.
    """
    entities: dict[str, str] = {}
    taken: dict[str, str] = {}
    omitted: list[tuple[str, str]] = []
    for symbol in sorted(bars):
        frame = bars[symbol]
        if frame is None or frame.empty:
            omitted.append((symbol, "sin barras"))
            continue
        entity = resolve_entity(symbol)
        if not entity.resolved:
            omitted.append((symbol, "simbolo sin entidad resoluble"))
            continue
        if entity.key in taken:
            omitted.append((symbol, f"entidad '{entity.key}' ya ocupada por {taken[entity.key]}"))
            logger.warning(
                "Canal sintetico: %s comparte entidad '%s' con %s; se omite",
                symbol, entity.key, taken[entity.key],
            )
            continue
        taken[entity.key] = symbol
        entities[symbol] = entity.key
    return entities, tuple(omitted)


def _stream_seed(seed: int, group: str, symbol: str) -> int:
    """
    Semilla del flujo aleatorio de (grupo de correlacion, simbolo).

    SHA-256 y no `hash()`: el hash de Python esta salado por proceso, asi que con varios
    workers cada uno emitiria un canal distinto y el barrido no seria reproducible. Es el
    mismo cuidado que ya tiene `scoring/scenario_split.py` al permutar con numpy en vez de
    ordenar por hash.
    """
    digest = hashlib.sha256(f"{group}|{symbol}".encode()).digest()[:8]
    return (int(seed) + int.from_bytes(digest, "big")) % (2**63)


__all__ = [
    "CHANNEL_POLARITY",
    "DEFAULT_EMISSION_SEED",
    "EMPTY_PANEL",
    "SOURCE_PREFIX",
    "SignalPanel",
    "channel_source",
    "channel_values",
    "emit_signals",
    "feature_name",
    "forward_z",
    "source_key",
]
