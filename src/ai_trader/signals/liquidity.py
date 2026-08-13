"""
LA SONDA DE LIQUIDEZ: convierte "esta senal es buena" en "esta senal es buena Y cabe".

EL FALLO QUE ESTE MODULO EXISTE PARA HACER VISIBLE
--------------------------------------------------
Una senal puede ser genuina, tener historia, pasar el break-even de IC y no servir para
nada, y la razon casi nunca es estadistica: es que vive en activos donde no cabe tamano.
Ese fallo no lo detecta ninguna metrica de las que ya hay —el Sharpe de un backtest sin
impacto de mercado no sabe cuanto volumen tenia el activo— y se descubre tarde, cuando ya
se ha escalado.

Las dos cifras que hicieron falta para verlo, MEDIDAS el 2026-08-13:

    Hyperliquid   232 perpetuos. El MEDIANO mueve 173.976 $ al dia. La media diria 12,7 M$
                  porque BTC mueve 1.671 M$; la mediana dice donde vive la fuente.
    Upbit KRW     283 mercados. Mediana 253.029 $, decil inferior 40.193 $, maximo 102,6 M$.

Las dos fuentes que cuelgan de esos venues son buenas. Y con un tamano de posicion de cien
mil dolares, la mitad de sus entidades no existen.

LA MISMA DISCIPLINA QUE `depth.py`, Y POR EL MISMO MOTIVO
---------------------------------------------------------
El catalogo DECLARA `typical_adv_usd`; este modulo lo MIDE y lo apunta en
`data/signals/entity_adv.json`; y un test compara las dos cosas. Sin eso, el campo seria un
numero escrito de memoria que nadie puede auditar dentro de seis meses, que es exactamente
lo que `history_from` habria sido sin la sonda de profundidad.

LA TOLERANCIA ES DE UN ORDEN DE MAGNITUD, Y ES DELIBERADO
----------------------------------------------------------
`ADV_TOLERANCE_FACTOR = 10`. Un volumen diario se mueve tres o cuatro veces entre un
domingo tranquilo y un dia de panico, asi que exigir igualdad convertiria el test en una
alarma que salta por el mercado y no por un error. Y sobre todo: la pregunta que contesta
el campo —¿cabe tamano aqui?— es de orden de magnitud. Entre 174.000 y 253.000 no hay
ninguna decision distinta; entre 174.000 y 174 millones estan todas.

QUE SE PUEDE MEDIR Y QUE NO
---------------------------
Solo las fuentes cuyo eje es un ACTIVO NEGOCIABLE en un venue que publique volumen. Las de
alcance mercado (calendario macro, EDGAR, App Store) no tienen ADV que medir: su eje no es
un activo, y el tamano que las limita es el del universo que se opere. Eso se declara con
`None` y una razon, no con un cero.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_trader.data.providers.http import JsonHttpClient, JsonHttpConfig
from ai_trader.shared.clock import utc_now
from ai_trader.shared.reports import load_report, write_report
from ai_trader.signals.adapters.common import certifi_bundle
from ai_trader.signals.catalog import CATALOG, get_source

logger = logging.getLogger(__name__)

# El registro de mediciones. En data/ y no en .cache/ por lo mismo que el de profundidad:
# es la EVIDENCIA de lo que se comprobo y cuando.
ADV_LEDGER = Path("data") / "signals" / "entity_adv.json"

# Cuanto puede alejarse lo declarado de lo medido antes de que el test falle. Ver arriba.
ADV_TOLERANCE_FACTOR = 10.0


@dataclass(frozen=True, slots=True)
class VenueLiquidity:
    """Lo que se midio de UN venue: el ADV en dolares de cada entidad donde publica."""

    venue: str
    measured_at: str
    entities: dict[str, float]
    error: str | None = None

    @property
    def traded(self) -> list[float]:
        """Las entidades con volumen. Ver `n_traded` en `as_dict`."""
        return sorted(v for v in self.entities.values() if v and v > 0)

    @property
    def median_usd(self) -> float | None:
        values = self.traded
        return float(np.median(values)) if values else None

    def as_dict(self) -> dict:
        values = self.traded
        percentile = (
            (lambda q: float(np.percentile(values, q))) if values else (lambda q: None)
        )
        return {
            "venue": self.venue,
            "measured_at": self.measured_at,
            "n_entities": len(self.entities),
            # LAS DOS CIFRAS SON DISTINTAS Y LAS DOS IMPORTAN. Hyperliquid lista 232 perps y
            # una parte de ellos no negocia NADA en 24 horas. Los ceros se conservan en
            # `entities` —son una observacion, y ademas la mas dura— pero la mediana se
            # calcula sobre los que negocian: incluir los ceros daria un numero que describe
            # cuantos mercados muertos hay listados, no donde vive la senal.
            "n_traded": len(values),
            "median_usd": self.median_usd,
            "p10_usd": percentile(10),
            "p90_usd": percentile(90),
            "max_usd": values[-1] if values else None,
            "total_usd": float(sum(values)) if values else None,
            "entities": dict(sorted(self.entities.items())),
            "error": self.error,
        }


# --- los venues que se pueden medir ---------------------------------------------------
#
# clave de fuente -> venue. Varias fuentes comparten venue (las dos de Hyperliquid, las dos
# de Deribit) y se mide UNA vez: el ADV es una propiedad del mercado, no de la senal.
SOURCE_VENUE: dict[str, str] = {
    "hyperliquid_leverage": "hyperliquid",
    "hyperliquid_liqmap": "hyperliquid",
    "deribit_volatility": "deribit",
    "deribit_expiries": "deribit",
    "cex_listings": "upbit",
}


def measure_hyperliquid(client: JsonHttpClient | None = None) -> dict[str, float]:
    """Volumen 24h en dolares de cada perpetuo. Lo publica el propio contexto de activo."""
    api = client or JsonHttpClient(
        "https://api.hyperliquid.xyz",
        JsonHttpConfig(timeout_seconds=30.0, ca_bundle=certifi_bundle()),
    )
    payload = api.post_json("/info", body={"type": "metaAndAssetCtxs"})
    if not isinstance(payload, (list, tuple)) or len(payload) < 2:
        return {}
    universe = (payload[0] or {}).get("universe") or []
    out: dict[str, float] = {}
    for meta, ctx in zip(universe, payload[1] or []):
        name = str((meta or {}).get("name") or "").upper()
        volume = (ctx or {}).get("dayNtlVlm")
        if name and volume is not None:
            out[name] = float(volume)
    return out


def measure_deribit(client: JsonHttpClient | None = None) -> dict[str, float]:
    """Volumen 24h del PERPETUO de cada moneda con libro de opciones.

    Se mide el perpetuo y no las opciones a proposito: la pregunta es donde se pone el
    tamano si la senal dice algo, y el tamano se pone en el subyacente. El volumen de primas
    de opciones seria un numero mucho menor que responde a otra pregunta.
    """
    from ai_trader.signals.adapters.deribit import CURRENCIES

    api = client or JsonHttpClient("https://www.deribit.com", JsonHttpConfig(timeout_seconds=60.0))
    out: dict[str, float] = {}
    for currency in CURRENCIES:
        payload = api.get_json(
            "/api/v2/public/get_book_summary_by_currency",
            params={"currency": currency, "kind": "future"},
        )
        total = sum(
            float(row.get("volume_usd") or 0.0)
            for row in (payload or {}).get("result") or []
            if str(row.get("instrument_name") or "").endswith("PERPETUAL")
        )
        if total > 0:
            out[currency] = total
    return out


def measure_upbit(client: JsonHttpClient | None = None) -> dict[str, float]:
    """Volumen 24h de cada mercado KRW, en dolares.

    EL TIPO DE CAMBIO SE MIDE, NO SE ESCRIBE. Upbit cotiza BTC en KRW y en USDT, asi que el
    cociente de los dos precios ES el cambio implicito del propio venue. Escribir 1.350 aqui
    seria meter una constante que envejece sin que nadie se entere, y ademas una que no sale
    de ninguna medicion nuestra.
    """
    api = client or JsonHttpClient(
        "https://api.upbit.com",
        JsonHttpConfig(timeout_seconds=30.0, ca_bundle=certifi_bundle()),
    )
    markets = [
        str(m.get("market"))
        for m in api.get_json("/v1/market/all") or []
        if str(m.get("market", "")).startswith("KRW-")
    ]
    if not markets:
        return {}

    krw_per_usd = _upbit_fx(api)
    if not krw_per_usd:
        return {}

    out: dict[str, float] = {}
    for start in range(0, len(markets), 100):
        chunk = markets[start : start + 100]
        for row in api.get_json("/v1/ticker", params={"markets": ",".join(chunk)}) or []:
            symbol = str(row.get("market") or "").split("-")[-1]
            volume = row.get("acc_trade_price_24h")
            if symbol and volume is not None:
                out[symbol] = float(volume) / krw_per_usd
    return out


def _upbit_fx(api: JsonHttpClient) -> float | None:
    """KRW por dolar, del propio venue: precio de BTC en KRW dividido por el de BTC en USDT."""
    rows = api.get_json("/v1/ticker", params={"markets": "KRW-BTC,USDT-BTC"}) or []
    prices = {str(r.get("market")): float(r.get("trade_price") or 0) for r in rows}
    krw, usdt = prices.get("KRW-BTC"), prices.get("USDT-BTC")
    if not krw or not usdt:
        return None
    return krw / usdt


VENUE_PROBES = {
    "hyperliquid": measure_hyperliquid,
    "deribit": measure_deribit,
    "upbit": measure_upbit,
}


def measure_liquidity(
    *,
    venues: Sequence[str] | None = None,
    ledger_path: Path | str | None = ADV_LEDGER,
) -> dict[str, VenueLiquidity]:
    """Mide el ADV de cada venue medible y escribe el registro. Toca red."""
    now = utc_now().isoformat()
    targets = tuple(venues) if venues is not None else tuple(VENUE_PROBES)

    out: dict[str, VenueLiquidity] = {}
    for venue in targets:
        probe = VENUE_PROBES.get(venue)
        if probe is None:
            continue
        try:
            entities = probe()
            error = None
        except Exception as exc:  # noqa: BLE001 - un venue caido no puede tumbar los demas
            logger.warning("· %s: no se pudo medir el ADV (%s)", venue, exc)
            entities, error = {}, str(exc)
        out[venue] = VenueLiquidity(venue=venue, measured_at=now, entities=entities, error=error)
        logger.info("· %s: %s entidades con volumen", venue, len(entities))

    if ledger_path is not None:
        write_adv_ledger(out, ledger_path)
    return out


def write_adv_ledger(
    measured: Mapping[str, VenueLiquidity],
    path: Path | str = ADV_LEDGER,
    *,
    merge: bool = True,
) -> Path:
    """Escribe el registro de ADV. Fusiona por defecto, como el de profundidad: medir un
    solo venue es la operacion normal y no puede borrar lo medido antes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    venues = {venue: value.as_dict() for venue, value in measured.items()}
    if merge:
        previous = load_adv_ledger(target) or {}
        merged = dict(previous.get("venues") or {})
        merged.update(venues)
        venues = merged

    payload = {
        "generated_at": utc_now().isoformat(),
        "tolerance_factor": ADV_TOLERANCE_FACTOR,
        "source_venue": dict(sorted(SOURCE_VENUE.items())),
        "venues": dict(sorted(venues.items())),
    }
    # El escritor comun (`shared/reports.py`), con el formato de los otros dos registros de
    # `signals/`: indent 2 y sin acentos. Escribirlo a mano habria sido la septima copia de
    # un cuerpo que la auditoria de deuda ya unifico una vez.
    return write_report(payload, target, indent=2, ensure_ascii=True)


def load_adv_ledger(path: Path | str = ADV_LEDGER) -> dict | None:
    """El ultimo registro de ADV, o None si nunca se ha corrido la sonda.

    Envuelve al cargador comun en vez de reimplementarlo, y la envoltura tiene una razon:
    `shared/reports.py` devuelve None si el fichero NO ESTA y deja escapar el error si esta
    corrupto, mientras que los registros de `signals/` (profundidad, recuento de eventos)
    tratan las dos cosas igual —no hay dato— porque los escribe una sonda que puede quedarse
    a medias. Este comparte politica con sus hermanos, no con los informes de estudio.
    """
    try:
        return load_report(path)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - fichero corrupto
        return None


def measured_adv(path: Path | str = ADV_LEDGER) -> dict[str, float]:
    """clave de fuente -> ADV mediano MEDIDO de su venue. Lo que el catalogo puede declarar."""
    ledger = load_adv_ledger(path) or {}
    venues = ledger.get("venues") or {}
    out: dict[str, float] = {}
    for key, venue in (ledger.get("source_venue") or SOURCE_VENUE).items():
        median = (venues.get(venue) or {}).get("median_usd")
        if median:
            out[key] = float(median)
    return out


def declared_vs_measured_adv(path: Path | str = ADV_LEDGER) -> dict[str, dict]:
    """
    Compara el ADV que declara el catalogo con el que dice el registro.

    Tres desajustes, y solo uno es un error del catalogo:
      - declarado sin medir   -> alguien escribio una cifra de memoria
      - medido y no declarado -> hay una medicion que el catalogo no esta usando
      - fuera de tolerancia   -> o el venue cambio de tamano, o la cifra esta mal
    """
    measured = measured_adv(path)
    out: dict[str, dict] = {}
    for key in {*measured, *(s.key for s in CATALOG if s.typical_adv_usd)}:
        try:
            declared = get_source(key).typical_adv_usd
        except KeyError:  # pragma: no cover - fuente retirada del catalogo
            declared = None
        value = measured.get(key)
        within = None
        if declared and value:
            ratio = max(declared / value, value / declared)
            within = ratio <= ADV_TOLERANCE_FACTOR
        out[key] = {
            "declared_usd": declared,
            "measured_usd": value,
            "venue": SOURCE_VENUE.get(key),
            "within_tolerance": within,
        }
    return dict(sorted(out.items()))


def liquidity_summary(path: Path | str = ADV_LEDGER) -> dict:
    """El resumen publicable: que fuentes tienen ADV y cual es la mas estrecha."""
    ledger = load_adv_ledger(path) or {}
    venues = ledger.get("venues") or {}
    declared = [(s.key, s.typical_adv_usd) for s in CATALOG if s.typical_adv_usd]
    thinnest = min(declared, key=lambda pair: pair[1]) if declared else (None, None)
    return {
        "measured_at": ledger.get("generated_at"),
        "n_venues": len(venues),
        "n_declared": len(declared),
        "tolerance_factor": ADV_TOLERANCE_FACTOR,
        "thinnest_source": thinnest[0],
        "thinnest_adv_usd": thinnest[1],
        "venues": {
            name: {
                "n_entities": row.get("n_entities"),
                "n_traded": row.get("n_traded"),
                "median_usd": row.get("median_usd"),
                "p10_usd": row.get("p10_usd"),
                "max_usd": row.get("max_usd"),
            }
            for name, row in sorted(venues.items())
        },
    }


__all__ = [
    "ADV_LEDGER",
    "ADV_TOLERANCE_FACTOR",
    "SOURCE_VENUE",
    "VENUE_PROBES",
    "VenueLiquidity",
    "declared_vs_measured_adv",
    "liquidity_summary",
    "load_adv_ledger",
    "measure_deribit",
    "measure_hyperliquid",
    "measure_liquidity",
    "measure_upbit",
    "measured_adv",
    "write_adv_ledger",
]
