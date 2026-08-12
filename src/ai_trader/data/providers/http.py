from __future__ import annotations

import json
import logging
import ssl
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JsonHttpConfig:
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_seconds: float = 0.5
    user_agent: str = "ai-trader/0.1.0"
    # Cabeceras fijas del cliente: es donde va un `Authorization: Bearer ...` o el
    # User-Agent descriptivo que exige la politica de Wikimedia. NUNCA se registran en el
    # log —el log imprime la url y nada mas— porque una credencial en un fichero de log es
    # una credencial filtrada.
    headers: Mapping[str, str] = field(default_factory=dict)
    # Bundle de certificados alternativo. Vacio = el del sistema, que es lo normal. Existe
    # por un fallo MEDIDO: el almacen de Windows no valida la cadena de
    # sanctionslistservice.ofac.treas.gov y urlopen aborta con CERTIFICATE_VERIFY_FAILED
    # antes de llegar al servidor. Se pone por fuente, no de forma global, para que
    # relajar la validacion nunca sea un efecto colateral invisible.
    ca_bundle: str | None = None


class JsonHttpClient:
    """Cliente GET/POST con reintentos y backoff. Compartido por los proveedores REST.

    Habla JSON (`get_json`/`post_json`) y tambien TEXTO PLANO (`get_text`), que hacia falta
    el dia que una fuente publico su lista en XML: escribir un segundo cliente para eso
    habria significado mantener la regla de reintentos —un 4xx que no es 429 no se
    reintenta— en dos sitios.
    """

    def __init__(self, base_url: str, config: JsonHttpConfig | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.config = config or JsonHttpConfig()

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        return self._request(path, params=params, headers=headers)

    def get_text(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        """El cuerpo tal cual, sin interpretarlo. Para XML, HTML o CSV."""
        return self._request(path, params=params, headers=headers, parse=False) or ""

    def post_json(
        self,
        path: str,
        *,
        body: Any = None,
        params: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """POST con cuerpo JSON. Existe por una sola fuente —el libro P2P de Binance, que
        solo se consulta por POST— y comparte los reintentos con el GET a proposito: la
        busqueda P2P es idempotente (es una consulta), asi que reintentarla no duplica nada.
        """
        payload = json.dumps(body if body is not None else {}).encode("utf-8")
        return self._request(
            path,
            params=params,
            headers={"Content-Type": "application/json", **(headers or {})},
            data=payload,
        )

    def _request(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        data: bytes | None = None,
        parse: bool = True,
    ) -> Any:
        url = f"{self.base_url}{path}"
        query = urlencode(params or {}, doseq=True)
        if query:
            url = f"{url}?{query}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.config.user_agent,
                **dict(self.config.headers),
                **dict(headers or {}),
            },
            method="POST" if data is not None else "GET",
            data=data,
        )

        last_error: Exception | None = None
        context = ssl.create_default_context(cafile=self.config.ca_bundle) if self.config.ca_bundle else None

        for attempt in range(self.config.max_retries):
            try:
                with urlopen(
                    request, timeout=self.config.timeout_seconds, context=context
                ) as response:
                    # utf-8-sig y no utf-8: identico salvo que se come el BOM que algunos
                    # servidores (el calendario de la Fed) ponen delante de su JSON y que
                    # hace fallar a json.loads con un error que no se parece a la causa.
                    body = response.read().decode("utf-8-sig", errors="replace").strip()
                    if not parse:
                        return body
                    # Un 2xx con cuerpo vacio no es un JSON roto: es la respuesta que da
                    # GitHub (202) mientras calcula sus estadisticas. Devolver None deja que
                    # el adaptador lo trate como "hoy no hay dato" en vez de reintentar tres
                    # veces algo que no va a cambiar en dos segundos.
                    return json.loads(body) if body else None
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                # 4xx que no sea rate limit es un error del cliente: reintentar no ayuda.
                if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                    raise

                if attempt < self.config.max_retries - 1:
                    delay = self.config.backoff_seconds * (2**attempt)
                    logger.warning(
                        "HTTP %s failed (attempt %s/%s) | url=%s | error=%s | retrying in %.1fs",
                        request.get_method(),
                        attempt + 1,
                        self.config.max_retries,
                        url,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        assert last_error is not None
        raise last_error
