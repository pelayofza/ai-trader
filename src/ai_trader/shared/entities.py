"""
Resolucion SIMBOLO -> ENTIDAD, en dos capas, y auditable.

EL PROBLEMA. El sistema habla en simbolos de mercado ("BTC/USDT", "SPY", "PM::<slug>")
porque eso es lo que cotiza. Las fuentes externas no: DefiLlama habla de protocolos,
Wikipedia de articulos, la CFTC de contratos, un explorador de cadena de tokens. Para
cruzar unas con otros hace falta una clave comun —la ENTIDAD— y una funcion honesta que
la derive del simbolo.

DOS CAPAS, Y EL ORDEN IMPORTA.

1. Una REGLA DERIVADA (`_from_rule`). "XYZ/USDT" -> "XYZ" reutilizando `split_pair` de
   `instruments.py`. Es la capa que hace el trabajo y la que tiene la propiedad que
   importa: **funciona el dia que se anade un listado nuevo**, sin que nadie edite una
   tabla. Una tabla curada a mano de 24 pares empieza completa y esta rota en cuanto el
   universo cambia; el fallo, ademas, es silencioso.
2. Una tabla de OVERRIDES (`ENTITY_OVERRIDES`) que **arranca vacia** y solo crece con
   fallos OBSERVADOS. El precedente exacto es `SPREAD_BPS_BY_SYMBOL`
   (`execution/microstructure.py`): una tabla explicita, auditable, para los hechos que
   ninguna formula produce. La diferencia es que aquella nacio poblada porque el spread
   de cada simbolo es un hecho de mercado que no se deduce; aqui la clave SI se deduce
   casi siempre, asi que cada linea de la tabla es la confesion de un caso en el que la
   regla fallo. Que arranque vacia es el punto: el descubrimiento es MEDICION (ver
   `signals/audit.py`), no una sesion de curacion previa.

POR QUE `EntityRef` LLEVA `source`. Sin ese campo, "BTC" y "BTC" son indistinguibles
aunque uno venga de la regla y el otro de un parche. Con el, la auditoria puede contar
cuantos simbolos dependen de la tabla y cuantos no se resuelven en absoluto, y esa cifra
—no una impresion— es la que dice si el mapeo aguanta. `unmapped` es un estado de primera
clase a proposito: es preferible una entidad declarada como no resuelta a una entidad
inventada que se cruza en silencio con los datos de otro.

LIMITE DECLARADO, y es el motivo por el que existe la capa 2. La regla hereda la
convencion canonica de `detect_asset_class`: un nombre sin barra es renta variable. Asi
que `resolve_entity("BTC")` (sin par) devuelve una entidad de tipo `equity`, no `token`.
Es coherente con el resto del sistema y es exactamente el tipo de caso que la tabla de
overrides existe para arreglar el dia que se observe, no antes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ai_trader.shared.instruments import PREDICTION_PREFIX, split_pair


class EntityKind(str, Enum):
    """Que clase de cosa es la entidad. Las fuentes declaran cual consumen (`entity_kind`
    en el catalogo de senales) para que no se le pida a una fuente de cadena la entidad de
    una accion."""

    TOKEN = "token"
    EQUITY = "equity"
    PREDICTION = "prediction"
    CHAIN = "chain"
    MACRO = "macro"
    VENUE = "venue"
    # Una sola entidad para todo el mercado (flujos agregados, calendario macro). Su clave
    # es MARKET_ENTITY: no depende de ningun simbolo.
    MARKET = "market"
    UNKNOWN = "unknown"


# Clave de la entidad unica de las fuentes de alcance mercado. Un asterisco y no "GLOBAL"
# para que sea imposible confundirla con un ticker.
MARKET_ENTITY = "*"

# Procedencia del mapeo. Va dentro de cada EntityRef y es lo que hace auditable el cruce.
SOURCE_RULE = "rule"
SOURCE_OVERRIDE = "override"
SOURCE_UNMAPPED = "unmapped"

# --- overrides: ARRANCA VACIA -------------------------------------------------------
#
# simbolo normalizado (MAYUSCULAS, sin espacios) -> clave de entidad.
#
# NO se rellena por adelantado. Cada entrada tiene que venir de un fallo observado por
# `signals.audit`, y lo suyo es dejar escrito en el comentario que lo destapo. Casos que
# se ESPERA que aparezcan cuando haya cobertura real —no se anticipan aqui porque
# anticiparlos seria curar en vez de medir—: pares con multiplicador en el ticker
# ("1000PEPE/USDT", cuya entidad es PEPE), tokens renombrados por el exchange
# (MATIC -> POL) y envoltorios (WBTC, cuya entidad economica es BTC).
ENTITY_OVERRIDES: dict[str, str] = {}


@dataclass(slots=True, frozen=True)
class EntityRef:
    """
    La entidad de un simbolo, CON la procedencia del mapeo.

    `key` es la clave con la que se archivan y se cruzan las senales. `symbol` es de donde
    salio (vacio si la entidad no viene de un simbolo, como la de alcance mercado).
    """

    key: str
    kind: EntityKind
    symbol: str = ""
    source: str = SOURCE_RULE

    @property
    def resolved(self) -> bool:
        """False = no hay entidad. Un consumidor que ignore esto se cruzara con basura."""
        return self.source != SOURCE_UNMAPPED and bool(self.key)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind.value,
            "symbol": self.symbol,
            "source": self.source,
        }


MARKET_REF = EntityRef(key=MARKET_ENTITY, kind=EntityKind.MARKET, source=SOURCE_RULE)


def normalize_symbol_key(symbol: str) -> str:
    """La forma en la que se busca un simbolo en la tabla de overrides."""
    return (symbol or "").strip().upper()


def resolve_entity(symbol: str) -> EntityRef:
    """
    Entidad de `symbol`. Override primero, regla despues, `unmapped` si ninguna aplica.

    El override va primero porque la tabla existe justo para los casos en los que la regla
    da una respuesta PLAUSIBLE Y EQUIVOCADA: si la regla ganara, un parche solo podria
    arreglar los simbolos que ya fallan de forma visible.
    """
    normalized = normalize_symbol_key(symbol)
    if not normalized:
        return EntityRef(key="", kind=EntityKind.UNKNOWN, symbol="", source=SOURCE_UNMAPPED)

    override = ENTITY_OVERRIDES.get(normalized)
    if override:
        return EntityRef(
            key=override.strip(),
            kind=_kind_from_rule(normalized),
            symbol=normalized,
            source=SOURCE_OVERRIDE,
        )

    return _from_rule(normalized)


def resolve_entities(symbols) -> dict[str, EntityRef]:
    """Mapeo simbolo -> entidad para un universo entero, conservando el orden de entrada."""
    return {normalize_symbol_key(s): resolve_entity(s) for s in symbols if normalize_symbol_key(s)}


def entity_keys(symbols) -> tuple[str, ...]:
    """Claves de entidad RESUELTAS y sin repetir. Lo que se le pide a una fuente por activo.

    Sin repetir importa: "BTC/USDT" y "BTC/USD" son la misma entidad, y pedirla dos veces
    seria descargar y archivar dos veces lo mismo.
    """
    out: list[str] = []
    for ref in resolve_entities(symbols).values():
        if ref.resolved and ref.key not in out:
            out.append(ref.key)
    return tuple(out)


def _from_rule(normalized: str) -> EntityRef:
    kind = _kind_from_rule(normalized)

    if kind is EntityKind.PREDICTION:
        slug = normalized[len(PREDICTION_PREFIX) :].strip()
        if not slug:
            return EntityRef("", EntityKind.UNKNOWN, normalized, SOURCE_UNMAPPED)
        return EntityRef(slug, kind, normalized, SOURCE_RULE)

    pair = split_pair(normalized)
    if pair is not None:
        return EntityRef(pair[0], EntityKind.TOKEN, normalized, SOURCE_RULE)

    if "/" in normalized or not _is_plain_ticker(normalized):
        # Tiene forma de par pero no se pudo partir (p.ej. "/USDT"), o lleva caracteres
        # que ningun ticker lleva. Se declara sin resolver en vez de adivinar.
        return EntityRef("", EntityKind.UNKNOWN, normalized, SOURCE_UNMAPPED)

    return EntityRef(normalized, EntityKind.EQUITY, normalized, SOURCE_RULE)


def _kind_from_rule(normalized: str) -> EntityKind:
    if normalized.startswith(PREDICTION_PREFIX):
        return EntityKind.PREDICTION
    if split_pair(normalized) is not None:
        return EntityKind.TOKEN
    return EntityKind.EQUITY


def _is_plain_ticker(value: str) -> bool:
    return bool(value) and all(ch.isalnum() or ch in {".", "-", "_"} for ch in value)


__all__ = [
    "ENTITY_OVERRIDES",
    "MARKET_ENTITY",
    "MARKET_REF",
    "SOURCE_OVERRIDE",
    "SOURCE_RULE",
    "SOURCE_UNMAPPED",
    "EntityKind",
    "EntityRef",
    "entity_keys",
    "normalize_symbol_key",
    "resolve_entities",
    "resolve_entity",
]
