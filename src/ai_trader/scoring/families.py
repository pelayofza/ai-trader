"""
LA REJILLA DE CONFIGURACIONES QUE SE RANKEAN: las familias, la semilla y el conjunto.

Estas cinco constantes y la funcion que las combina definen la IDENTIDAD del conjunto de
configuraciones sobre el que se ha medido todo lo publicado. Vivian en `weight_study`, que
es el estudio que las estreno; desde que hay mas de un estudio que las usa -y desde que uno
de ellos (la capa tematica) corre sobre archivo REAL- tenerlas dentro de un estudio concreto
hacia que cualquiera que quisiera rankear tuviera que importar de ese estudio.

Lo que NO se puede tocar sin invalidar lo publicado, y por que:

- `STUDY_SEED` siembra el hipercubo latino. Cambiarlo cambia las 64 configuraciones.
- El ORDEN de `FAMILIES` importa: `build_specs` siembra con `STUDY_SEED + indice`, asi que
  anadir AL FINAL preserva byte a byte las configuraciones ya publicadas e insertar en medio
  las SUSTITUYE en silencio por otras. Hay un test que lo congela.
- `CONFIGS_PER_FAMILY` es el n del hipercubo: con otro n, los puntos no son un subconjunto
  de los anteriores, son otros puntos.
"""
from __future__ import annotations

from collections.abc import Sequence

from ai_trader.config import StrategySpec
from ai_trader.scoring.weight_calibration import candidate_specs

STUDY_SEED = 20260809  # seed del hipercubo latino; fija la identidad del conjunto

# LAS DOS PRIMITIVAS DE PRECIO SOBRE LAS QUE SE MIDIO TODO LO PUBLICADO. No es un duplicado
# de `FAMILIES`: es el nombre de la HUELLA congelada, y es contra lo que asertan los tests de
# evidencia publicada (`tests/test_transfer.py::TestPublishedFingerprint`).
FAMILIES_PUBLISHED = ("crypto_momentum", "mean_reversion")

# Las seis tematicas, EN ESTE ORDEN Y AL FINAL. El orden no es cosmetico: `build_specs` siembra
# el hipercubo con `STUDY_SEED + indice_de_familia`, asi que anadir al final preserva las 16
# configuraciones publicadas byte a byte, e insertar en medio las SUSTITUYE en silencio por
# otras 16. Hay un test que lo congela.
NEW_FAMILIES = (
    "liquidation_cascade",
    "vol_term_structure",
    "event_calendar_drift",
    "attention_ignition",
    "flow_persistence",
    "signal_composite",
)

FAMILIES = FAMILIES_PUBLISHED + NEW_FAMILIES  # 8 familias x 8 = 64 configuraciones

CONFIGS_PER_FAMILY = 8  # x 2 familias = 16 configuraciones rankeadas (huella publicada)


def build_specs(
    families: Sequence[str] = FAMILIES, per_family: int = CONFIGS_PER_FAMILY
) -> list[StrategySpec]:
    """Las configuraciones a rankear.

    Misma funcion (`candidate_specs`), misma semilla base y mismo desplazamiento por
    familia para todos los estudios. Cambiar cualquiera de las tres cosas romperia la
    comparabilidad entre estudios sin avisar, asi que las tres viven aqui y los estudios
    solo las importan.
    """
    specs: list[StrategySpec] = []
    for i, family in enumerate(families):
        specs.extend(candidate_specs(family, per_family, seed=STUDY_SEED + i))
    return specs
