"""
Estudio de fidelidad: los stylized-facts de una libreria sintetica contra el mercado REAL.

Descarga el historico diario de cripto via CCXT (cacheado en disco por
`ai_trader.data.cache`, la misma cache que usa el sistema en vivo), mide los mismos
hechos estilizados sobre el mundo real y sobre la libreria sintetica, y publica el
informe que consumen el dashboard y la documentacion.

    .venv\\Scripts\\python.exe -m ai_trader.synthetic.fidelity_study
    .venv\\Scripts\\python.exe -m ai_trader.synthetic.fidelity_study --offline
    .venv\\Scripts\\python.exe -m ai_trader.synthetic.fidelity_study --verify-determinism

Salida: data/fidelity/report_<lib>.json

Dos decisiones que hacen que la comparacion signifique algo:

1. MISMA LONGITUD DE MUESTRA. El historico real se trocea en ventanas del mismo tamano
   que un path sintetico (el horizonte de la libreria, 730 dias). La autocorrelacion y
   sobre todo la curtosis son estimadores sesgados en muestras cortas: medir el real
   sobre ocho anos seguidos y el sintetico sobre dos anos compararia el sesgo, no el
   mundo. Las ventanas solapan al 50% para no quedarse en cuatro estimaciones; el solape
   se declara porque implica que NO son independientes (sirven para la tendencia
   central, no para un intervalo de confianza).

2. VENTANA REAL CERRADA Y FIJA. El periodo descargado es una constante del estudio, no
   "hasta hoy": si el final se moviera con la fecha de ejecucion, el informe no seria
   reproducible y dos regeneraciones no serian comparables.

Determinismo: no hay aleatoriedad en ninguna parte (ni muestreo ni semillas); las
muestras sinteticas son los primeros `--paths` caminos de cada escenario y los datos
reales, un rango historico cerrado. `--verify-determinism` recalcula todo una segunda
vez y exige un informe identico campo a campo.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.data.cache import load_bars as load_cached_bars
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.instruments import AssetClass
from ai_trader.synthetic.fidelity import (
    DEFAULT_MAX_LAG,
    DEFAULT_SIGMA,
    MIN_OBSERVATIONS,
    MIN_PAIR_OVERLAP,
    TARGET_METRIC_KEYS,
    Estimate,
    SeriesFacts,
    aggregate_facts,
    aggregate_pairs,
    compare_cross_correlations,
    compare_metrics,
    pair_correlations,
    series_facts,
)
from ai_trader.synthetic.store import SyntheticStore

logger = logging.getLogger("fidelity_study")

OUT_DIR = Path("data") / "fidelity"
DEFAULT_LIBRARY_ID = "ai_v2"
DEFAULT_EXCHANGE = "binance"
# Ventana historica CERRADA (ver docstring): arranca cuando Binance ya tiene profundidad
# en la mayoria de los pares y termina en un corte fijo, no en "hoy".
DEFAULT_REAL_START = "2017-09-01"
DEFAULT_REAL_END = "2026-01-01"
# Caminos por escenario. 5 x 30 escenarios = 150 estimaciones sinteticas por activo:
# suficiente para un [p10, p90] estable sin leer la libreria entera.
DEFAULT_PATHS = 5


# ------------------------------------------------------------- datos reales ---------


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class CachedBarsProvider:
    """
    Fuente de barras que SOLO lee la cache en disco (`--offline`).

    Permite reproducir el estudio sin red -y sin depender de que el exchange siga
    sirviendo el mismo historico- una vez los datos se han descargado alguna vez.
    """

    def get_daily_bars(self, symbol: str, start: datetime, end: datetime):
        cached = load_cached_bars(f"crypto::{symbol.strip().upper()}", timeframe="1D")
        if cached is None or cached.empty:
            return None
        # Mismos limites que MarketDataService: hasta la ultima vela diaria CERRADA. Si
        # no, el modo offline devolveria una barra mas que el online y el estudio no
        # seria el mismo estudio.
        first = _utc(start).normalize()
        last = _utc(end).normalize() - pd.Timedelta(days=1)
        return cached.loc[first : max(first, last)]


def build_service(exchange: str, *, offline: bool):
    if offline:
        return CachedBarsProvider()
    # Import tardio: construir el servicio arrastra los proveedores de stocks y de
    # mercados de prediccion, que este estudio no usa.
    from ai_trader.data.market_data import MarketDataService
    from ai_trader.data.providers.ccxt_crypto import CCXTCrypto, CCXTCryptoConfig

    return MarketDataService(
        crypto_provider=CCXTCrypto(CCXTCryptoConfig(exchange_id=exchange))
    )


def fetch_real_bars(
    symbols: Sequence[str], start: datetime, end: datetime, service
) -> dict[str, pd.DataFrame]:
    """
    Historico diario real por simbolo. Los que el exchange no sirve se OMITEN y se
    declaran; no se sustituyen por nada (un activo deslistado, como MATIC/USDT en
    Binance, simplemente no tiene contraparte real que comparar).
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = service.get_daily_bars(symbol, start, end)
        except Exception as exc:  # noqa: BLE001 - un simbolo caido no tumba el estudio
            logger.warning("Sin datos reales para %s: %s", symbol, exc)
            continue
        if df is None or df.empty:
            logger.warning("Sin datos reales para %s (respuesta vacia)", symbol)
            continue
        out[symbol] = df
        logger.info(
            "  %-12s %d barras reales | %s -> %s",
            symbol, len(df), df.index.min().date(), df.index.max().date(),
        )
    return out


def returns_frame(bars: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Retornos log diarios, un simbolo por columna, alineados por fecha."""
    columns: dict[str, pd.Series] = {}
    for symbol, df in bars.items():
        close = bar_schema.series(df, bar_schema.CLOSE)
        close = close[close > 0]
        if len(close) < 2:
            continue
        columns[symbol] = np.log(close).diff().dropna()
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index()


def calendar_windows(
    start: pd.Timestamp, end: pd.Timestamp, window_days: int, step_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventanas de calendario [t0, t1) de `window_days`, avanzando `step_days`."""
    if window_days <= 0 or step_days <= 0:
        raise ValueError("window_days and step_days must be positive")
    span = pd.Timedelta(days=window_days)
    stride = pd.Timedelta(days=step_days)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    t0 = start
    while t0 + span <= end:
        out.append((t0, t0 + span))
        t0 = t0 + stride
    return out


# --------------------------------------------------------------- medicion -----------


@dataclass(slots=True)
class FactsCollector:
    """Acumula mediciones (una por ventana real o por path sintetico) y las agrega."""

    max_lag: int = DEFAULT_MAX_LAG
    sigma: float = DEFAULT_SIGMA
    min_observations: int = MIN_OBSERVATIONS
    min_pair_overlap: int = MIN_PAIR_OVERLAP
    by_symbol: dict[str, list[SeriesFacts]] = field(default_factory=dict, init=False)
    pair_snapshots: list[dict[str, float]] = field(default_factory=list, init=False)
    n_windows: int = field(default=0, init=False)

    def add(self, returns: pd.DataFrame) -> None:
        """Mide UNA ventana: los facts de cada columna y su matriz de correlacion."""
        if returns.empty:
            return
        self.n_windows += 1
        for symbol in returns.columns:
            values = returns[symbol].dropna().to_numpy(dtype=float)
            facts = series_facts(
                values,
                max_lag=self.max_lag,
                sigma=self.sigma,
                min_observations=self.min_observations,
            )
            if facts is not None:
                self.by_symbol.setdefault(str(symbol), []).append(facts)
        self.pair_snapshots.append(
            pair_correlations(returns, min_overlap=self.min_pair_overlap)
        )

    def symbols(self) -> dict[str, dict[str, Estimate]]:
        return {
            symbol: aggregate_facts(facts)
            for symbol, facts in sorted(self.by_symbol.items())
            if facts
        }

    def pairs(self) -> dict[str, Estimate]:
        return aggregate_pairs(self.pair_snapshots)

    def as_dict(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "symbols": {
                symbol: {
                    "n_estimates": len(self.by_symbol[symbol]),
                    "n_observations_median": int(
                        np.median([f.n for f in self.by_symbol[symbol]])
                    ),
                    "facts": {key: est.as_dict() for key, est in facts.items()},
                }
                for symbol, facts in self.symbols().items()
            },
            "pairs": {key: est.as_dict() for key, est in sorted(self.pairs().items())},
        }


def measure_real(
    bars: Mapping[str, pd.DataFrame],
    windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    **options,
) -> FactsCollector:
    frame = returns_frame(bars)
    collector = FactsCollector(**options)
    for t0, t1 in windows:
        collector.add(frame.loc[t0 : t1 - pd.Timedelta(days=1)])
    return collector


def measure_synthetic(
    store: SyntheticStore,
    library_id: str,
    scenario_ids: Sequence[str],
    symbols: Sequence[str],
    n_paths: int,
    **options,
) -> FactsCollector:
    """
    Mide los primeros `n_paths` caminos de cada escenario.

    Lee el parquet UNA vez por escenario y solo la columna de cierre: el resto del OHLCV
    no interviene en ningun stylized-fact y la libreria pesa cerca de un giga.
    """
    collector = FactsCollector(**options)
    wanted = list(symbols)
    for index, scenario_id in enumerate(scenario_ids, start=1):
        try:
            paths = store.load_scenario_paths(
                library_id,
                scenario_id,
                path_indices=range(n_paths),
                columns=[bar_schema.CLOSE],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sin caminos para %s: %s", scenario_id, exc)
            continue
        for path_index in sorted(paths):
            bars = {s: df for s, df in paths[path_index].items() if s in wanted}
            collector.add(returns_frame(bars))
        logger.info(
            "  [%2d/%d] %-28s %d caminos", index, len(scenario_ids), scenario_id, len(paths)
        )
    return collector


# ------------------------------------------------------------------ informe ---------


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Todo lo que define el estudio. Se serializa con los resultados: sin esto las
    cifras no son ni reproducibles ni auditables."""

    library_id: str
    exchange: str
    timeframe: str
    real_start: str
    real_end: str
    window_days: int
    step_days: int
    n_scenarios: int
    n_paths: int
    symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    max_lag: int
    sigma: float
    min_observations: int
    min_pair_overlap: int
    offline: bool

    def as_dict(self) -> dict:
        return {
            "library_id": self.library_id,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "real_window": {"start": self.real_start, "end": self.real_end},
            "window_days": self.window_days,
            "step_days": self.step_days,
            "windows_overlap": self.step_days < self.window_days,
            "n_scenarios": self.n_scenarios,
            "n_paths": self.n_paths,
            "symbols": list(self.symbols),
            "missing_symbols": list(self.missing_symbols),
            "max_lag": self.max_lag,
            "sigma": self.sigma,
            "min_observations": self.min_observations,
            "min_pair_overlap": self.min_pair_overlap,
            "offline": self.offline,
        }


def crypto_symbols(manifest_universe: Iterable[Mapping]) -> list[str]:
    """Los simbolos cripto del universo de la libreria: los unicos con contraparte real
    accesible por CCXT (la renta variable va por otro proveedor y otra sesion de mercado)."""
    return sorted(
        str(item["symbol"])
        for item in manifest_universe
        if str(item.get("asset_class", "")).lower() == AssetClass.CRYPTO.value
    )


def build_report(plan: StudyPlan, real: FactsCollector, synth: FactsCollector) -> dict:
    """Ensambla el informe: plan, mediciones crudas de ambos mundos y comparaciones."""
    comparisons = compare_metrics(real.symbols(), synth.symbols())
    cross = compare_cross_correlations(real.pairs(), synth.pairs())

    targets = [c for c in comparisons if c.key in TARGET_METRIC_KEYS]
    covered = [c.coverage_pct for c in targets if c.n]
    ranks = [c.rank_corr for c in targets if c.n and not np.isnan(c.rank_corr)]

    return {
        "plan": plan.as_dict(),
        "real": real.as_dict(),
        "synthetic": synth.as_dict(),
        "metrics": [c.as_dict() for c in comparisons],
        "cross_correlation": cross.as_dict(),
        "summary": {
            "n_symbols": len(set(real.symbols()) & set(synth.symbols())),
            "n_pairs": cross.n,
            "n_real_windows": real.n_windows,
            "n_synthetic_samples": synth.n_windows,
            "coverage_mean_pct": round(float(np.mean(covered)), 1) if covered else None,
            "rank_corr_mean": round(float(np.mean(ranks)), 4) if ranks else None,
            "rank_corr_cross": cross.as_dict()["rank_corr"],
        },
    }


def run_study(args: argparse.Namespace) -> dict:
    """Corre el estudio completo y devuelve el informe (sin la marca de tiempo)."""
    store = SyntheticStore(args.synthetic_root) if args.synthetic_root else SyntheticStore()
    manifest = store.load_manifest(args.library)

    scenario_ids = [s["id"] for s in manifest.scenarios]
    if args.scenarios:
        scenario_ids = scenario_ids[: args.scenarios]
    symbols = crypto_symbols(manifest.universe)

    window_days = args.window_days or manifest.horizon_days
    step_days = args.step_days or max(1, window_days // 2)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    logger.info(
        "Datos reales (%s, %s -> %s) para %d simbolos cripto%s",
        args.exchange, args.start, args.end, len(symbols),
        " [offline: solo cache]" if args.offline else "",
    )
    service = build_service(args.exchange, offline=args.offline)
    real_bars = fetch_real_bars(symbols, start.to_pydatetime(), end.to_pydatetime(), service)
    missing = tuple(s for s in symbols if s not in real_bars)
    if missing:
        logger.warning("Sin contraparte real: %s", ", ".join(missing))

    options = {
        "max_lag": args.max_lag,
        "sigma": args.sigma,
        "min_observations": args.min_observations,
        "min_pair_overlap": args.min_pair_overlap,
    }
    windows = calendar_windows(start, end, window_days, step_days)
    logger.info(
        "Midiendo %d ventanas reales de %d dias (paso %d)", len(windows), window_days, step_days
    )
    real = measure_real(real_bars, windows, **options)

    logger.info(
        "Midiendo %s: %d escenarios x %d caminos", args.library, len(scenario_ids), args.paths
    )
    synth = measure_synthetic(
        store, args.library, scenario_ids, sorted(real_bars), args.paths, **options
    )

    plan = StudyPlan(
        library_id=args.library,
        exchange=args.exchange,
        timeframe="1d",
        real_start=args.start,
        real_end=args.end,
        window_days=window_days,
        step_days=step_days,
        n_scenarios=len(scenario_ids),
        n_paths=args.paths,
        symbols=tuple(sorted(real_bars)),
        missing_symbols=missing,
        max_lag=args.max_lag,
        sigma=args.sigma,
        min_observations=args.min_observations,
        min_pair_overlap=args.min_pair_overlap,
        offline=args.offline,
    )
    return build_report(plan, real, synth)


def _print_report(report: dict) -> None:
    # Las etiquetas llevan sigma y acentos; la consola de Windows es cp1252 por defecto.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(f"\n{'metrica':<52}{'real':>10}{'sintetico':>12}{'ratio':>8}"
          f"{'rank corr':>11}{'cobertura':>11}")
    rows = [*report["metrics"], report["cross_correlation"]]
    for row in rows:
        def num(key, decimals=3):
            value = row[key]
            return "—" if value is None else f"{value:.{decimals}f}"
        print(
            f"{row['label'][:50]:<52}{num('real_median'):>10}{num('synth_median'):>12}"
            f"{num('ratio', 2):>8}{num('rank_corr'):>11}"
            f"{row['coverage_pct']:>10.0f}%"
        )
    s = report["summary"]
    print(
        f"\n{s['n_symbols']} simbolos | {s['n_pairs']} pares | "
        f"{s['n_real_windows']} ventanas reales vs {s['n_synthetic_samples']} muestras sinteticas"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=DEFAULT_LIBRARY_ID)
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    parser.add_argument("--start", default=DEFAULT_REAL_START)
    parser.add_argument("--end", default=DEFAULT_REAL_END)
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--scenarios", type=int, default=0, help="0 = todos")
    parser.add_argument("--window-days", type=int, default=0, help="0 = horizonte de la libreria")
    parser.add_argument("--step-days", type=int, default=0, help="0 = media ventana")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--min-observations", type=int, default=MIN_OBSERVATIONS)
    parser.add_argument("--min-pair-overlap", type=int, default=MIN_PAIR_OVERLAP)
    parser.add_argument(
        "--offline", action="store_true", help="No llamar al exchange: usar solo la cache."
    )
    parser.add_argument(
        "--verify-determinism", action="store_true",
        help="Recalcula el estudio y exige un informe identico campo a campo.",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.ERROR)

    report = run_study(args)

    if args.verify_determinism:
        # Segunda pasada offline: la primera ya dejo los datos reales en cache, asi que
        # esto compara el CALCULO, no la estabilidad del exchange.
        replay_args = argparse.Namespace(**{**vars(args), "offline": True})
        replay = run_study(replay_args)
        replay["plan"]["offline"] = report["plan"]["offline"]  # unica diferencia esperada
        identical = replay == report
        report["determinism"] = {"verified": True, "identical": identical}
        logger.info("Determinismo: informe %s", "identico" if identical else "DISTINTO")

    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"report_{args.library}.json"
    target.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Informe -> %s", target)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
