"""
DERIBIT: lo mas informativo que se publica gratis sobre expectativas de precio.

TRES NUMEROS Y UN CALENDARIO
----------------------------
    DVOL            la volatilidad implicita del indice, con historia REAL.
    skew de 25d     cuanto mas cara esta la proteccion que la apuesta. Es la asimetria
                    que el precio no ensena: dos activos con la misma volatilidad y
                    distinto skew estan diciendo cosas opuestas.
    term structure  la pendiente entre el vencimiento corto y el largo. Invertida =
                    el mercado paga por cubrirse AHORA.
    vencimientos    fechas fijas con OI por strike. Ver mas abajo.

EL DELTA SE CALCULA AQUI, Y ES LA DECISION TECNICA DE ESTE MODULO
------------------------------------------------------------------
El "skew de 25 delta" es la diferencia de volatilidad implicita entre el put cuyo delta es
-0,25 y el call cuyo delta es +0,25. Hace falta el delta de cada opcion, y MEDIDO
2026-08-13: `get_book_summary_by_currency` NO publica griegas. Publica `mark_iv`, el strike,
el vencimiento y el precio del subyacente, que es todo lo que hace falta para calcularlo.

Se calcula con Black-Scholes de tipo cero sobre el precio del subyacente, y se declaran las
dos aproximaciones que eso lleva dentro:

  1. las opciones de Deribit son INVERSAS (se liquidan en la moneda, no en dolares), asi
     que su delta "verdadero" lleva una correccion respecto del Black-Scholes estandar;
  2. el forward se aproxima por el precio del subyacente, es decir tipo y base a cero.

Las dos mueven el delta unas centesimas, que en la practica cambia como mucho el strike
elegido cuando hay dos candidatos casi empatados; MEDIDO ese dia, los strikes que salen
tienen delta entre 0,229 y 0,295 en la rejilla, y por eso se INTERPOLA en delta en vez de
quedarse con el mas cercano: la rejilla de strikes es gruesa en los vencimientos cortos y
"el strike mas parecido a 25 delta" puede estar en 0,19.

Y se hace en la capa PURA, sobre el libro archivado, no en la de red. Es lo que permite
testear la cuenta sin proveedor y, sobre todo, RE-DERIVARLA: el dia que se decida corregir
por inversion, el libro de todos los dias anteriores sigue en disco.

DOS PROFUNDIDADES EN LA MISMA FUENTE
------------------------------------
El DVOL tiene historia descargable y MEDIDA: 2021-03-24 en BTC y en ETH. El skew y la
pendiente salen del libro de HOY y no tienen ninguna: nadie publica el libro de opciones de
una fecha pasada. Es el mismo caso que `macro_calendar` (calendario de la Fed hasta 2017,
ventana rodante del BLS) y se declara igual, porque la alternativa —presentar la fuente
como si tuviera cinco anos de skew— seria falsa en la mitad de sus columnas.

LA PAGINACION DEL DVOL NO ES UN DETALLE
---------------------------------------
MEDIDO: la API devuelve como mucho 1.000 puntos por peticion y entrega LOS MAS RECIENTES.
Pedir doce anos devuelve los ultimos 1.000 dias y ni un aviso. Sin paginar hacia atras, la
sonda habria declarado que el DVOL empieza en 2023-11-16, que es donde termina la primera
pagina. La profundidad real —2021-03-24— sale de seguir pidiendo hacia atras hasta que una
ventana vuelve vacia.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone

import pandas as pd

from ai_trader.data.providers.http import JsonHttpConfig
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    day_or_none,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
    unix_day,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

DERIBIT_BASE = "https://www.deribit.com"
DVOL_PATH = "/api/v2/public/get_volatility_index_data"
BOOK_SUMMARY_PATH = "/api/v2/public/get_book_summary_by_currency"

# Las monedas con libro de opciones vivo. MEDIDO 2026-08-13: BTC (820 instrumentos) y ETH
# (688). SOL tuvo DVOL entre 2022-05 y 2022-11 y dejo de publicarse; XRP y PAXG aparecen en
# `get_currencies` y devuelven cero instrumentos. Escribir aqui una moneda que no tiene
# libro seria una peticion que falla en silencio todos los dias.
CURRENCIES: tuple[str, ...] = ("BTC", "ETH")

# Resolucion del DVOL, en segundos. Un dia: el sistema entero es diario y pedir una hora
# multiplicaria por 24 las peticiones para agregar despues a lo mismo.
DVOL_RESOLUTION = "86400"

# Tope de puntos por peticion que impone la API. Es el numero que obliga a paginar.
DVOL_PAGE_POINTS = 1000

# Cuantas paginas hacia atras como mucho. Doce anos de dias caben en cinco paginas de mil;
# con seis hay margen y sigue habiendo un final garantizado si el proveedor cambiara el tope.
DVOL_MAX_PAGES = 6

# Vencimientos de referencia para la estructura temporal, en dias. El corto es el semanal
# tipico y el largo es el trimestral: son los dos que existen SIEMPRE, y por eso la
# pendiente es comparable entre dias. Elegir "el primero" y "el ultimo" del libro haria que
# la feature cambiara de significado cada vez que se lista un vencimiento nuevo.
TERM_SHORT_DAYS = 7.0
TERM_LONG_DAYS = 90.0

# El vencimiento sobre el que se miden el skew y la IV at-the-money. Treinta dias es la
# convencion del sector (es el horizonte del VIX y el del propio DVOL), asi que la feature
# se puede comparar con cualquier serie publicada.
SKEW_TENOR_DAYS = 30.0

# El delta al que se mide el skew, en valor absoluto.
SKEW_DELTA = 0.25

_INSTRUMENT_RE = re.compile(r"^([A-Z0-9]+)-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([CP])$")


# =====================================================================================
# 1. Superficie de volatilidad (serie continua)
# =====================================================================================


class DeribitVolatility(BaseJsonAdapter):
    """
    DVOL con backfill paginado + una foto diaria del libro para skew y pendiente.

    Archiva DOS clases de registro y las distingue por `request.series`, igual que hace
    `MacroCalendar` con la Fed y el BLS: `dvol` (un punto por dia, re-descargable) y `book`
    (la foto de hoy, irrecuperable manana).
    """

    def __init__(self, source, *, currencies: Sequence[str] = CURRENCIES, **kwargs) -> None:
        config = kwargs.pop("http_config", None) or JsonHttpConfig(timeout_seconds=60.0)
        super().__init__(source, base_url=DERIBIT_BASE, http_config=config, **kwargs)
        self._currencies = tuple(currencies)

    def _targets(self, entities: Sequence[str]) -> tuple[str, ...]:
        wanted = {str(e).upper() for e in entities}
        return tuple(c for c in self._currencies if not wanted or c in wanted)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        out: list[RawRecord] = []
        since = day_or_none(start)
        for currency in self._targets(entities):
            for day, point in self.dvol_points(currency, start, end):
                if since is not None and day < since:
                    continue
                out.append(
                    self.record(currency, point, day=day, request={"series": "dvol"})
                )

            book = safe_call(
                lambda c=currency: self.client.get_json(
                    BOOK_SUMMARY_PATH, params={"currency": c, "kind": "option"}
                ),
                what=f"deribit book summary {currency}",
                logger=logger,
            )
            rows = (book or {}).get("result") or []
            if rows:
                day = datetime.now(timezone.utc).date().isoformat()
                out.append(
                    self.record(
                        currency,
                        {"currency": currency, "book": rows},
                        day=day,
                        request={"series": "book"},
                    )
                )
        return out

    def dvol_points(
        self, currency: str, start: datetime | None, end: datetime | None
    ) -> list[tuple[str, list]]:
        """La serie de DVOL, paginando hacia atras. Ver el docstring del modulo."""
        stop = end or datetime.now(timezone.utc)
        floor = start or (stop - timedelta(days=365 * 12))
        cursor = int(stop.timestamp() * 1000)
        floor_ms = int(floor.timestamp() * 1000)

        seen: dict[str, list] = {}
        for _ in range(DVOL_MAX_PAGES):
            payload = safe_call(
                lambda c=cursor: self.client.get_json(
                    DVOL_PATH,
                    params={
                        "currency": currency,
                        "start_timestamp": floor_ms,
                        "end_timestamp": c,
                        "resolution": DVOL_RESOLUTION,
                    },
                ),
                what=f"deribit dvol {currency}",
                logger=logger,
            )
            data = ((payload or {}).get("result") or {}).get("data") or []
            if not data:
                break
            for point in data:
                day = unix_day(point[0]) if point else None
                if day:
                    seen[day] = list(point)
            oldest = min(int(p[0]) for p in data if p)
            if len(data) < DVOL_PAGE_POINTS or oldest <= floor_ms:
                break
            cursor = oldest - 1  # justo antes del punto mas viejo: sin solape y sin hueco
        return sorted(seen.items())

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_volatility_row))


# Las cuatro columnas de la fuente. Las DOS clases de registro tienen que emitirlas todas
# —con None en las que no saben— porque el frame canonico exige las features declaradas y
# porque el DVOL y el libro caen en la misma fila del mismo dia: el que no sabe nada de una
# columna tiene que decir "no se", no omitirla.
_VOLATILITY_COLUMNS = ("dvol_index", "skew_25d", "atm_iv_30d", "iv_term_slope")


def _volatility_row(record: Mapping) -> dict | None:
    series = str(((record.get("request") or {}).get("series")) or "")
    entity = str(record.get("entity") or "")
    row = None
    if series == "dvol":
        row = _dvol_row(record, entity)
    elif series == "book":
        row = _book_row(record, entity)
    if row is None:
        return None
    return {**dict.fromkeys(_VOLATILITY_COLUMNS, None), **row}


def _dvol_row(record: Mapping, entity: str) -> dict | None:
    payload = record.get("payload")
    if not isinstance(payload, (list, tuple)) or len(payload) < 5:
        return None
    day = unix_day(payload[0])
    close = numeric(payload[4])  # [timestamp, open, high, low, close]
    if not day or close is None:
        return None
    return {ENTITY: entity, DAY: day, "dvol_index": close}


def _book_row(record: Mapping, entity: str) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(record.get("day")) or iso_day(record.get("fetched_at"))
    book = payload.get("book") or []
    if not day or not book:
        return None
    surface = volatility_surface(book, as_of=day)
    if not surface:
        return None
    return {ENTITY: entity, DAY: day, **surface}


def parse_instrument(name: str) -> tuple[str, datetime, float, str] | None:
    """`BTC-25SEP26-84000-P` -> `(BTC, vencimiento UTC, 84000.0, 'P')`. None si no encaja.

    El vencimiento de Deribit es a las 08:00 UTC y no a medianoche. Poner medianoche
    inflaria el tiempo hasta vencimiento en ocho horas, que en un semanal es un 5% del
    plazo y se nota en el delta.
    """
    match = _INSTRUMENT_RE.match(str(name or "").upper())
    if not match:
        return None
    currency, expiry, strike, kind = match.groups()
    try:
        moment = datetime.strptime(expiry, "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    except ValueError:
        return None
    return currency, moment, float(strike), kind


def norm_cdf(x: float) -> float:
    """N(x). Con `math.erf` y no con SciPy: es una linea y evita una dependencia."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(forward: float, strike: float, iv: float, years: float, kind: str) -> float | None:
    """Delta Black-Scholes con tipo cero. Ver las aproximaciones en el docstring."""
    if forward <= 0 or strike <= 0 or iv <= 0 or years <= 0:
        return None
    d1 = (math.log(forward / strike) + 0.5 * iv * iv * years) / (iv * math.sqrt(years))
    return norm_cdf(d1) if kind == "C" else norm_cdf(d1) - 1.0


def volatility_surface(book: Sequence[Mapping], *, as_of: str) -> dict[str, float]:
    """
    El libro de opciones -> skew de 25 delta, IV at-the-money y pendiente temporal.

    Devuelve solo lo que se pueda calcular: un libro sin vencimientos a los dos lados de
    los treinta dias no produce skew, y publicar el del vencimiento mas cercano "porque es
    lo que hay" cambiaria el significado de la columna sin avisar.
    """
    reference = pd.Timestamp(as_of, tz="UTC")
    by_expiry: dict[datetime, list[dict]] = {}
    for row in book:
        parsed = parse_instrument((row or {}).get("instrument_name"))
        iv = numeric(row.get("mark_iv"))
        forward = numeric(row.get("underlying_price"))
        if not parsed or not iv or iv <= 0 or not forward:
            continue
        _, expiry, strike, kind = parsed
        years = (pd.Timestamp(expiry) - reference).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            continue
        delta = bs_delta(forward, strike, iv / 100.0, years, kind)
        if delta is None:
            continue
        by_expiry.setdefault(expiry, []).append(
            {"delta": delta, "iv": iv, "kind": kind, "days": years * 365.25}
        )

    if not by_expiry:
        return {}

    out: dict[str, float] = {}
    tenors = sorted((rows[0]["days"], expiry) for expiry, rows in by_expiry.items())

    near = _closest_expiry(tenors, SKEW_TENOR_DAYS)
    if near is not None:
        rows = by_expiry[near]
        call = _iv_at_delta(rows, SKEW_DELTA, "C")
        put = _iv_at_delta(rows, -SKEW_DELTA, "P")
        atm = _iv_at_delta(rows, 0.5, "C")
        if call is not None and put is not None:
            # PUT MENOS CALL: positivo = la proteccion cuesta mas que la apuesta, que es el
            # estado normal en cripto. El signo esta escrito aqui para que nadie tenga que
            # deducirlo de la polaridad del radar.
            out["skew_25d"] = float(put - call)
        if atm is not None:
            out["atm_iv_30d"] = float(atm)

    short = _closest_expiry(tenors, TERM_SHORT_DAYS)
    long = _closest_expiry(tenors, TERM_LONG_DAYS)
    if short is not None and long is not None and short != long:
        short_atm = _iv_at_delta(by_expiry[short], 0.5, "C")
        long_atm = _iv_at_delta(by_expiry[long], 0.5, "C")
        if short_atm is not None and long_atm is not None:
            out["iv_term_slope"] = float(long_atm - short_atm)
    return out


def _closest_expiry(tenors, target_days: float):
    """El vencimiento con plazo mas parecido al pedido. None si el libro esta vacio."""
    if not tenors:
        return None
    return min(tenors, key=lambda pair: abs(pair[0] - target_days))[1]


def _iv_at_delta(rows: Sequence[Mapping], target: float, kind: str) -> float | None:
    """
    IV en el delta pedido, INTERPOLADA entre los dos strikes que lo rodean.

    Quedarse con el strike de delta mas parecido seria mas corto y peor: MEDIDO 2026-08-13,
    en los vencimientos de menos de una semana la rejilla es tan gruesa que el candidato mas
    cercano a 0,25 estaba en 0,187, y llamar "skew de 25 delta" a eso es llamarlo mal.
    """
    points = sorted(
        ((float(r["delta"]), float(r["iv"])) for r in rows if r.get("kind") == kind),
        key=lambda p: p[0],
    )
    if not points:
        return None
    if len(points) == 1:
        return points[0][1]

    for (d0, iv0), (d1, iv1) in zip(points, points[1:]):
        if d0 <= target <= d1:
            if d1 == d0:
                return iv0
            weight = (target - d0) / (d1 - d0)
            return iv0 + weight * (iv1 - iv0)
    # Fuera del rango: el extremo mas cercano. Extrapolar una sonrisa de volatilidad hacia
    # un delta que el libro no cotiza es inventar el dato que falta.
    return min(points, key=lambda p: abs(p[0] - target))[1]


# =====================================================================================
# 2. Calendario de vencimientos con OI por strike (evento fechado)
# =====================================================================================


class DeribitExpiries(BaseJsonAdapter):
    """
    Una linea por vencimiento vivo, con el OI que se le acumula.

    La fecha es la mejor que hay en el catalogo: el ultimo viernes del mes a las 08:00 UTC,
    fijada al listar el instrumento y no revisada nunca. La MAGNITUD, en cambio, es la foto
    de hoy: cuanto OI hay colgado de ese vencimiento cambia cada dia y nadie publica la foto
    de ayer. Por eso el `pit` es forward_capture aunque las fechas ya esten.
    """

    def __init__(self, source, *, currencies: Sequence[str] = CURRENCIES, **kwargs) -> None:
        config = kwargs.pop("http_config", None) or JsonHttpConfig(timeout_seconds=60.0)
        super().__init__(source, base_url=DERIBIT_BASE, http_config=config, **kwargs)
        self._currencies = tuple(currencies)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        wanted = {str(e).upper() for e in entities}
        out: list[RawRecord] = []
        for currency in self._currencies:
            if wanted and currency not in wanted:
                continue
            book = safe_call(
                lambda c=currency: self.client.get_json(
                    BOOK_SUMMARY_PATH, params={"currency": c, "kind": "option"}
                ),
                what=f"deribit expiries {currency}",
                logger=logger,
            )
            rows = (book or {}).get("result") or []
            for day, payload in expiry_payloads(rows, currency):
                out.append(
                    self.record(currency, payload, day=day, request={"series": "expiry"})
                )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_expiry_row))


def expiry_payloads(book: Sequence[Mapping], currency: str) -> list[tuple[str, dict]]:
    """El libro -> un payload por VENCIMIENTO, con su OI por strike.

    El OI se pasa a dolares con `underlying_price` porque Deribit lo publica en contratos, y
    un contrato de BTC y uno de ETH no son la misma cantidad de dinero: sin la conversion,
    la fraccion de OI que vence seria comparable dentro de una moneda y no entre las dos.
    """
    by_expiry: dict[str, dict] = {}
    total = 0.0
    for row in book:
        parsed = parse_instrument((row or {}).get("instrument_name"))
        open_interest = numeric(row.get("open_interest"))
        price = numeric(row.get("underlying_price"))
        if not parsed or not open_interest or not price:
            continue
        _, expiry, strike, kind = parsed
        day = expiry.date().isoformat()
        notional = open_interest * price
        total += notional
        block = by_expiry.setdefault(
            day, {"currency": currency, "expiry": day, "oi_usd": 0.0, "strikes": {}}
        )
        block["oi_usd"] += notional
        key = f"{strike:.0f}{kind}"
        block["strikes"][key] = block["strikes"].get(key, 0.0) + notional

    for block in by_expiry.values():
        # La fraccion se calcula CONTRA EL TOTAL VIVO DEL MISMO DIA y se archiva ya hecha:
        # re-derivarla manana exigiria tener delante todos los vencimientos de esta captura,
        # y el archivo se lee vencimiento a vencimiento.
        block["oi_share"] = block["oi_usd"] / total if total > 0 else None
        block["total_oi_usd"] = total
    return sorted(by_expiry.items())


def _expiry_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("expiry")) or iso_day(record.get("day"))
    oi = numeric(payload.get("oi_usd"))
    if not day or oi is None:
        return None
    return {
        ENTITY: record.get("entity") or payload.get("currency") or "",
        DAY: day,
        "expiry_oi_usd": oi,
        "expiry_oi_share": numeric(payload.get("oi_share")),
    }


# =====================================================================================

ADAPTERS = {
    "deribit_volatility": DeribitVolatility,
    "deribit_expiries": DeribitExpiries,
}


def register() -> None:
    from ai_trader.signals.source import register_adapter

    for key, factory in ADAPTERS.items():
        try:
            register_adapter(key, factory)
        except ValueError:
            logger.debug("%s ya estaba registrado", key)


__all__ = [
    "ADAPTERS",
    "CURRENCIES",
    "DVOL_MAX_PAGES",
    "DVOL_PAGE_POINTS",
    "SKEW_DELTA",
    "SKEW_TENOR_DAYS",
    "TERM_LONG_DAYS",
    "TERM_SHORT_DAYS",
    "DeribitExpiries",
    "DeribitVolatility",
    "bs_delta",
    "expiry_payloads",
    "norm_cdf",
    "parse_instrument",
    "register",
    "volatility_surface",
]
