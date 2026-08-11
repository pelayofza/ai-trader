"""
Guavy: sentimiento social por token. SOLO los conteos crudos.

LA REGLA QUE DEFINE ESTE ADAPTADOR
----------------------------------
Guavy publica tres familias de endpoints: el historico de conteos
(`get-sentiment-history`), y los de `trend` y `signal`. Este adaptador consume el primero y
tiene PROHIBIDOS los otros dos, con una comprobacion en el codigo (`_check_path`) y un test
que la vigila. No es purismo:

  - `trend` y `signal` son la salida del MODELO de Guavy, no una observacion. Si su modelo
    cambia, la serie historica cambia con el y ya no es la que se vio aquel dia.
  - Nada garantiza que ese output no este recalculado con informacion POSTERIOR al dia que
    lleva escrito. Un backtest sobre una serie asi no mide una estrategia: mide cuanto
    futuro se colo dentro de la etiqueta.
  - Los conteos, en cambio, son un hecho: cuantos mensajes positivos, negativos y neutros
    hubo ese dia. Lo que se haga con ellos es decision nuestra y es auditable.

Es la misma frontera que el resto del sistema ya aplica: se archiva el hecho y se deriva
aqui, en vez de consumir la conclusion de otro.

PROFUNDIDAD: NO MEDIDA
----------------------
La API exige Bearer token y no hay uno en este entorno (el host responde 401, que es la
prueba de que el host es el correcto y la credencial la que falta). Asi que `history_from`
sigue en None y esta fuente solo existe HACIA ADELANTE hasta que alguien con token corra
`signals depth`. El parametro `limit` se pide en dias y se manda al maximo declarado por el
proveedor; cuanto devuelva de verdad es justo lo que esa medicion tiene que averiguar.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    env_secret,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

# Host medido: `https://guavy.com/api/v1/...` responde 401 sin token (y 404 en `api.guavy.com`),
# que es la forma de comprobar un host sin tener credencial. Se puede sobreescribir por
# entorno para no tener que tocar codigo si el proveedor mueve la puerta.
GUAVY_BASE = os.environ.get("GUAVY_API_URL", "https://guavy.com").rstrip("/")
HISTORY_PATH = "/api/v1/sentiment/get-sentiment-history"

# Los endpoints que este adaptador NO va a llamar nunca. Ver el docstring.
FORBIDDEN_PATHS: tuple[str, ...] = ("trend", "signal")

# Dias que se piden por peticion. Alto a proposito: lo que devuelva de menos es la medicion
# de la profundidad real, y pedir poco la escondería.
DEFAULT_LIMIT = 3650


def _check_path(path: str) -> str:
    """Cierra la puerta a los endpoints derivados del modelo del proveedor."""
    lowered = path.lower()
    for banned in FORBIDDEN_PATHS:
        if banned in lowered:
            raise ValueError(
                f"Guavy: '{path}' es un endpoint derivado ({banned}). Solo se consumen los "
                "conteos crudos: el output de su modelo puede estar recalculado con "
                "informacion posterior y no es auditable."
            )
    return path


class GuavySentiment(BaseJsonAdapter):
    """Serie diaria de conteos por token. Un registro crudo por dia."""

    def __init__(self, source, *, limit: int = DEFAULT_LIMIT, **kwargs) -> None:
        super().__init__(source, base_url=GUAVY_BASE, **kwargs)
        self.limit = limit

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        token = env_secret(self.source.auth_env)
        if not token:
            raise RuntimeError(
                f"Guavy necesita la variable de entorno {self.source.auth_env} (Bearer token)."
            )
        headers = {"Authorization": f"Bearer {token}"}

        out: list[RawRecord] = []
        for entity in entities:
            path = _check_path(f"{HISTORY_PATH}/{entity}")
            payload = safe_call(
                lambda path=path: self.client.get_json(
                    path, params={"limit": self.limit}, headers=headers
                ),
                what=f"guavy {entity}",
                logger=logger,
            )
            for point in _series_of(payload):
                day = iso_day(point.get("date"))
                if day:
                    # `request` lleva la ruta pero no la cabecera: el token no entra en el
                    # archivo ni por asomo.
                    out.append(self.record(entity, point, day=day, request={"path": path}))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_sentiment_row))


def _series_of(payload) -> list[dict]:
    """La lista de dias, este donde este.

    Una API con plan gratuito envuelve su respuesta hoy de una forma y en tres meses de
    otra (`{data: [...]}`, `{result: {history: [...]}}`, o la lista pelada). Buscar la
    primera lista de diccionarios con `date` cuesta cuatro lineas y evita que un cambio de
    envoltorio pare la captura de una fuente cuyo pasado no se puede re-descargar.
    """
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and any(isinstance(p, dict) and "date" in p for p in value):
                return [p for p in value if isinstance(p, dict)]
            if isinstance(value, dict):
                found = _series_of(value)
                if found:
                    return found
    return []


def _sentiment_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("date"))
    if not day:
        return None
    positive = numeric(payload.get("positive"))
    negative = numeric(payload.get("negative"))
    neutral = numeric(payload.get("neutral"))
    total = numeric(payload.get("total"))
    if total is None:
        parts = [v for v in (positive, negative, neutral) if v is not None]
        total = sum(parts) if parts else None
    if all(v is None for v in (positive, negative, neutral, total)):
        return None
    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        "sentiment_positive": positive,
        "sentiment_negative": negative,
        "sentiment_neutral": neutral,
        "sentiment_total": total,
    }


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("guavy_sentiment", GuavySentiment)


__all__ = [
    "FORBIDDEN_PATHS",
    "GUAVY_BASE",
    "HISTORY_PATH",
    "GuavySentiment",
    "register",
]
