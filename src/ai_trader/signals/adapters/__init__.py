"""
EL CATALOGO ENTERO, CONECTADO: once fuentes continuas y seis de evento.

Las once continuas (tier `B`) son series con efectos pequenos y decadentes: sentimiento,
flujos de ETF, macro, oferta de stablecoins, ingresos de protocolo, cuota DEX/CEX, atencion
por idioma, prima P2P, dispersion de funding, actividad de desarrollo y posicionamiento en
futuros.

    guavy.py       Sentimiento por token. SOLO conteos crudos, nunca trend/signal.
    etf_flows.py   Flujos de ETF spot por EMISOR (TFTC, CC BY 4.0). BTC medido; ETH no hay.
    fred.py        Macro mapeada UNO A UNO sobre los factores del generador sintetico.
    defillama.py   Stablecoins por cadena, fees/revenue/TVL por protocolo, DEX vs CEX.
    wikipedia.py   Atencion DESGLOSADA POR IDIOMA (la descomposicion geografica barata).
    p2p.py         Prima P2P en TRY/ARS/NGN: demanda por crisis monetaria.
    funding.py     DISPERSION del funding entre venues, no el nivel (que ya esta arbitrado).
    github.py      Commits diarios y contribuidores unicos.
    cftc.py        COT/TFF semanal, fechado el dia en que se PUBLICA.

Las seis de evento (tier `A`) son oferta calendarizada y determinista, y viven juntas en un
solo modulo porque comparten forma —una fila el dia que pasa algo, con su magnitud— y
porque lo que las separaba del resto ya no existe:

    events.py      Ajuste de dificultad, hacks, calendario macro, OFAC, unlocks, staking.

TODAS VAN AL MISMO SITIO. El tier ya no enruta: describe la CODIFICACION que le toca a cada
una (`signals/normalize.py` para las continuas, `signals/events.py` para las de evento) y
las dos acaban en el mismo espacio de observacion, en backtest y en vivo.

LO QUE COMPARTEN, QUE ES LO QUE HACE BARATA LA FUENTE N+1
---------------------------------------------------------
Todos los modulos tienen la misma forma: un `fetch_raw` que toca red y devuelve el payload
INTACTO, y un `daily_from_raw` PURO que lo traduce. Ninguno decide donde se guarda, ni
cuando se llama, ni que entidades le tocan; ninguno normaliza nada —eso es
`signals/normalize.py` y `signals/events.py`, y es comun— y ninguno declara su propia
profundidad: `history_from` sale de `signals/depth.py`, que lo MIDE.

REGISTRAR ES UN ACTO EXPLICITO
------------------------------
`register_all()` es lo que pasa el registro de adaptadores de vacio a diecisiete. Se llama
desde `signals/capture.py` al arrancar una captura, no al importar el catalogo: importar
una lista de declaraciones no puede tener como efecto que exista un cliente HTTP. Y es
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
    events,
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
MODULES = (guavy, etf_flows, fred, defillama, wikipedia, p2p, funding, github, cftc, events)


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
    "events",
    "fred",
    "funding",
    "github",
    "guavy",
    "p2p",
    "register_all",
    "wikipedia",
]
