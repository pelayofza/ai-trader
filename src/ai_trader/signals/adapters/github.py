"""
GitHub: velocidad de desarrollo. El dato por el que Santiment cobra y que es publico.

DOS ENDPOINTS Y DOS GRANULARIDADES
----------------------------------
    /stats/commit_activity   52 semanas, con el reparto DIARIO dentro de cada una
    /stats/contributors      por autor, con un cubo semanal cada uno

De ahi que `commits` sea diario y `contributors` semanal: el segundo se ancla al lunes de
su semana y los demas dias no tienen valor. Es una asimetria real de la fuente, y taparla
repitiendo el valor siete veces convertiria una observacion en siete y romperia `observed`,
que es justo la columna que existe para saber cuanto hay detras de cada celda.

EL 202 NO ES UN ERROR
---------------------
Las estadisticas de GitHub se calculan en diferido: la primera peticion a un repo frio
devuelve 202 con el cuerpo VACIO y hay que volver. El cliente HTTP devuelve None ante un
cuerpo vacio y aqui se trata como "hoy no hay dato": la captura de manana lo recoge. Tres
reintentos en caliente no lo arreglarian —el calculo tarda mas que el backoff— y en cambio
gastarian la cuota.

LA VENTANA ES DE 52 SEMANAS Y NO SE ELIGE
-----------------------------------------
`commit_activity` devuelve un ano y no acepta rango. Asi que la profundidad de esta fuente
es un ano movil: cada captura refresca el ano anterior, y la historia mas larga se acumula
en el archivo crudo, captura a captura. Es un caso claro de por que el archivo es
append-only: la unica forma de tener tres anos es haber guardado tres.

REVISABLE POR NATURALEZA
------------------------
Un `force-push` reescribe el historico de commits, asi que la respuesta de hoy sobre marzo
puede no ser la de ayer sobre marzo. `archive_revisable`, y las dos lineas con su
`fetched_at` en el archivo son lo que permite verlo.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import pandas as pd

from ai_trader.data.providers.http import JsonHttpConfig
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import UTC, env_secret, iso_day, numeric, safe_call
from ai_trader.signals.source import (
    RAW_FETCHED_AT,
    RAW_REQUEST,
    BaseJsonAdapter,
    RawRecord,
)

logger = logging.getLogger(__name__)

GITHUB_BASE = "https://api.github.com"

# entidad -> repositorio principal. COMPROBADO: los 24 existen y estaban vivos el dia que
# se escribio la tabla (INJ es `InjectiveFoundation/injective-core`; `InjectiveLabs/...`
# devuelve 404). "Principal" es una eleccion: un ecosistema es mas que un repo, y esta
# columna mide el nucleo, no la comunidad.
REPOS: dict[str, str] = {
    "BTC": "bitcoin/bitcoin",
    "ETH": "ethereum/go-ethereum",
    "SOL": "anza-xyz/agave",
    "BNB": "bnb-chain/bsc",
    "XRP": "XRPLF/rippled",
    "ADA": "IntersectMBO/cardano-node",
    "DOGE": "dogecoin/dogecoin",
    "AVAX": "ava-labs/avalanchego",
    "DOT": "paritytech/polkadot-sdk",
    "LINK": "smartcontractkit/chainlink",
    "LTC": "litecoin-project/litecoin",
    "UNI": "Uniswap/v4-core",
    "ATOM": "cosmos/gaia",
    "NEAR": "near/nearcore",
    "APT": "aptos-labs/aptos-core",
    "ARB": "OffchainLabs/nitro",
    "OP": "ethereum-optimism/optimism",
    "INJ": "InjectiveFoundation/injective-core",
    "FIL": "filecoin-project/lotus",
    "ETC": "etclabscore/core-geth",
    "AAVE": "aave-dao/aave-v3-origin",
    "SUI": "MystenLabs/sui",
    "SEI": "sei-protocol/sei-chain",
    "TIA": "celestiaorg/celestia-node",
}

SERIES_COMMITS = "commit_activity"
SERIES_CONTRIBUTORS = "contributors"


class GithubActivity(BaseJsonAdapter):
    """Un registro crudo por semana y serie. El reparto a dia se hace en la capa 2."""

    def __init__(self, source, **kwargs) -> None:
        kwargs.setdefault("http_config", JsonHttpConfig(timeout_seconds=30.0))
        super().__init__(source, base_url=GITHUB_BASE, **kwargs)

    def _headers(self) -> dict[str, str]:
        token = env_secret(self.source.auth_env)
        # Sin token la API da 60 peticiones/hora, con token 5.000. Con 24 repos y dos
        # endpoints cada uno, sin token se agota en la segunda captura del dia: por eso el
        # catalogo declara la variable aunque la fuente funcione sin ella.
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            logger.info("· github: sin %s, la cuota es de 60 peticiones/hora", self.source.auth_env)
        return headers

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        headers = self._headers()
        out: list[RawRecord] = []

        for entity in entities:
            repo = REPOS.get(entity)
            if repo is None:
                continue

            weeks = safe_call(
                lambda repo=repo: self.client.get_json(
                    f"/repos/{repo}/stats/commit_activity", headers=headers
                ),
                what=f"github commits {repo}",
                logger=logger,
            )
            if weeks is None:
                logger.info("· github: %s aun calculando (202), se recoge en la siguiente", repo)
            for week in weeks or []:
                day = _week_day(week.get("week"))
                if day:
                    out.append(
                        self.record(
                            entity,
                            week,
                            day=day,
                            request={"repo": repo, "series": SERIES_COMMITS},
                        )
                    )

            authors = safe_call(
                lambda repo=repo: self.client.get_json(
                    f"/repos/{repo}/stats/contributors", headers=headers
                ),
                what=f"github contributors {repo}",
                logger=logger,
            )
            for day, active in _weekly_contributors(authors).items():
                out.append(
                    self.record(
                        entity,
                        {"week_start": day, "contributors": active},
                        day=day,
                        request={"repo": repo, "series": SERIES_CONTRIBUTORS},
                    )
                )

        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        rows: list[dict] = []
        for record in records or ():
            series = (record.get(RAW_REQUEST) or {}).get("series")
            entity = str(record.get("entity") or "")
            payload = record.get("payload") or {}
            if not entity:
                continue

            if series == SERIES_COMMITS:
                # El horizonte sale del PROPIO REGISTRO, no del reloj: la capa 2 es pura y
                # tiene que dar lo mismo hoy que dentro de un ano sobre el mismo archivo.
                rows.extend(_commit_rows(entity, payload, iso_day(record.get(RAW_FETCHED_AT))))
            elif series == SERIES_CONTRIBUTORS:
                day = payload.get("week_start")
                active = numeric(payload.get("contributors"))
                if day and active is not None:
                    rows.append(
                        {ENTITY: entity, DAY: day, "commits": None, "contributors": active}
                    )
        return self.to_daily(rows)


def _week_day(stamp) -> str | None:
    value = numeric(stamp)
    if value is None or value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=UTC).date().isoformat()


def _commit_rows(entity: str, week: Mapping, horizon: str | None = None) -> list[dict]:
    """Una semana de `commit_activity` -> hasta siete filas diarias.

    `days` viene con el domingo primero, que es la convencion de GitHub y no la de nadie
    mas: el indice 0 ES el dia del sello `week`, asi que sumarle el indice basta y no hay
    que ajustar por dia de la semana.

    HASTA siete, y no siete: la semana en curso llega con ceros en los dias que TODAVIA NO
    HAN PASADO. Sin el corte, la sonda medio una serie que terminaba cuatro dias en el
    FUTURO —observaciones fechadas en dias que no existen— y esos ceros no son "no hubo
    commits", son "aun no ha ocurrido". El horizonte es el `fetched_at` del registro y no el
    reloj, porque esta funcion es pura: sobre el mismo archivo tiene que dar lo mismo hoy
    que dentro de un ano.
    """
    start = _week_day(week.get("week"))
    days = week.get("days") or []
    if not start or len(days) != 7:
        return []
    first = datetime.fromisoformat(start).replace(tzinfo=UTC)
    rows = []
    for offset, commits in enumerate(days):
        value = numeric(commits)
        if value is None:
            continue
        day = (first + timedelta(days=offset)).date().isoformat()
        if horizon is not None and day > horizon:
            break
        rows.append(
            {
                ENTITY: entity,
                DAY: day,
                "commits": value,
                "contributors": None,
            }
        )
    return rows


def _weekly_contributors(authors) -> dict[str, int]:
    """Autores DISTINTOS con al menos un commit en cada semana."""
    out: dict[str, int] = {}
    for author in authors or []:
        for week in (author or {}).get("weeks") or []:
            commits = numeric(week.get("c"))
            if not commits:
                continue
            day = _week_day(week.get("w"))
            if day:
                out[day] = out.get(day, 0) + 1
    return out


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("github_activity", GithubActivity)


__all__ = ["GITHUB_BASE", "REPOS", "GithubActivity", "register"]
