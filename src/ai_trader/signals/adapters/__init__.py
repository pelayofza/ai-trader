"""
PRIMER LOTE CONECTADO: las once fuentes CONTINUAS (Tier B) del catalogo.

Tier B son series con efectos pequenos y decadentes, y por eso entran como FEATURES y no
como vetos: sentimiento, flujos de ETF, macro, oferta de stablecoins, ingresos de
protocolo, cuota DEX/CEX, atencion por idioma, prima P2P, dispersion de funding, actividad
de desarrollo y posicionamiento en futuros. Ninguna decide nada por si sola, y ninguna esta
cableada a una estrategia: esta tarea llena el archivo, no el vector de observacion.

    guavy.py       Sentimiento por token. SOLO conteos crudos, nunca trend/signal.
    etf_flows.py   Flujos de ETF spot por EMISOR (TFTC, CC BY 4.0). BTC medido; ETH no hay.
    fred.py        Macro mapeada UNO A UNO sobre los factores del generador sintetico.
    defillama.py   Stablecoins por cadena, fees/revenue/TVL por protocolo, DEX vs CEX.
    wikipedia.py   Atencion DESGLOSADA POR IDIOMA (la descomposicion geografica barata).
    p2p.py         Prima P2P en TRY/ARS/NGN: demanda por crisis monetaria.
    funding.py     DISPERSION del funding entre venues, no el nivel (que ya esta arbitrado).
    github.py      Commits diarios y contribuidores unicos.
    cftc.py        COT/TFF semanal, fechado el dia en que se PUBLICA.

LO QUE COMPARTEN, QUE ES LO QUE HACE BARATA LA FUENTE N+1
---------------------------------------------------------
Los nueve modulos tienen la misma forma: un `fetch_raw` que toca red y devuelve el payload
INTACTO, y un `daily_from_raw` PURO que lo traduce. Ninguno decide donde se guarda, ni
cuando se llama, ni que entidades le tocan; ninguno normaliza nada —eso es
`signals/normalize.py`, y es comun— y ninguno declara su propia profundidad: `history_from`
sale de `signals/depth.py`, que lo MIDE.

REGISTRAR ES UN ACTO EXPLICITO
------------------------------
`register_all()` es lo que pasa el registro de adaptadores de vacio a once. Se llama desde
`signals/capture.py` al arrancar una captura, no al importar el catalogo: importar una
lista de declaraciones no puede tener como efecto que exista un cliente HTTP. Y es
idempotente, porque `register_adapter` rechaza el duplicado y aqui se ignora ese rechazo a
proposito: llamarlo dos veces en el mismo proceso es normal (dos capturas, un test tras
otro) y no deberia reventar.

LO QUE SE MIDIO ANTES DE ESCRIBIR NADA
--------------------------------------
Cada tabla de este paquete —slugs de DefiLlama, repos de GitHub, titulos de Wikipedia,
contratos de la CFTC, venues de funding— se comprobo contra su proveedor, y lo que devolvia
404/400/500 NO se escribio. Un hueco declarado sale en la auditoria como cobertura cero; una
entrada inventada sale como una peticion que falla en silencio cada dia.
"""
from __future__ import annotations

import logging

from ai_trader.signals.adapters import (
    cftc,
    defillama,
    etf_flows,
    fred,
    funding,
    github,
    guavy,
    p2p,
    wikipedia,
)

logger = logging.getLogger(__name__)

# Modulos con adaptadores, en el orden en que se registran. El orden no importa para nada
# salvo para el log, y por eso es el del catalogo: se lee al lado.
MODULES = (guavy, etf_flows, fred, defillama, wikipedia, p2p, funding, github, cftc)


def register_all() -> tuple[str, ...]:
    """Registra todos los adaptadores del lote. Idempotente. Devuelve las claves conectadas."""
    from ai_trader.signals.source import REGISTRY, connected_keys

    for module in MODULES:
        try:
            module.register()
        except ValueError:
            # Ya estaba registrado: llamar dos veces es normal y no es un error.
            logger.debug("%s ya estaba registrado", module.__name__)
    logger.debug("Adaptadores conectados: %s", len(REGISTRY))
    return connected_keys()


__all__ = [
    "MODULES",
    "cftc",
    "defillama",
    "etf_flows",
    "fred",
    "funding",
    "github",
    "guavy",
    "p2p",
    "register_all",
    "wikipedia",
]
