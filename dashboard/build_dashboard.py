"""
Generador del dashboard de AI-Trader (v1).

Extrae datos REALES del repo (librerias sinteticas, estrategias, ranking) y escribe un
`dashboard/index.html` AUTOCONTENIDO (datos embebidos, charts en SVG, sin CDN) que se
abre con doble clic en el navegador. Se re-ejecuta cuando la herramienta evoluciona:

    .venv\\Scripts\\python.exe -m dashboard.build_dashboard

El ranking se corre sobre una MUESTRA REDUCIDA (pocos escenarios/paths, universo y
ventana recortados) para que el build sea tratable; el propio dashboard indica el scope
y el comando para regenerar el ranking completo.
"""
from __future__ import annotations

import dataclasses
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ai_trader.backtest.metrics import DEFAULT_HEADLINE_WEIGHTS
from ai_trader.config import StrategySpec, load_config
from ai_trader.data.backtest_source import HistoricalDataSource
from ai_trader.execution.microstructure import BarLiquidityProvider
from ai_trader.observation.features import OWN_ASSET_FEATURES
from ai_trader.observation.regime import REGIME_FEATURES
from ai_trader.scoring.aggregate import aggregate_reward
from ai_trader.scoring.baselines import BASELINE_LABELS, gate
from ai_trader.scoring.overfit import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from ai_trader.scoring.sample_eval import evaluate_baselines, evaluate_sample_detailed
from ai_trader.scoring.validation_study import (
    VALIDATION_REPORT,
    load_validation_report,
)
from ai_trader.scoring.weight_calibration import (
    CALIBRATION_REPORT,
    grid_point,
    load_calibration_report,
)
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.instruments import AssetClass
from ai_trader.strategies import build_strategy
from ai_trader.strategies.mean_reversion import MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy
from ai_trader.synthetic.fidelity import (
    FIDELITY_BASELINE_LIBRARY,
    FIDELITY_LIBRARY,
    TARGET_METRIC_KEYS,
    fidelity_report_path,
    load_fidelity_report,
    metric,
)
from ai_trader.synthetic.store import SyntheticStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("ai_trader").setLevel(logging.WARNING)  # silencia el chatter de estrategias
logger = logging.getLogger("dashboard")

ROOT = Path(__file__).resolve().parent.parent
OUT_HTML = Path(__file__).resolve().parent / "index.html"
PRIMARY_LIB = "ai_v2"
COMPARE_LIB = "ai_v1"
# Las tres generaciones del mundo sintetico, en orden: iid -> microestructura -> calibrado
# contra el mercado. Se ensenan juntas porque cada una solo significa algo contra la
# anterior. PRIMARY_LIB sigue siendo ai_v2: es la libreria sobre la que se midieron la
# calibracion de pesos y la validacion multiventana, y cambiarla obligaria a recorrerlas.
LIBRARY_LINEAGE = (COMPARE_LIB, PRIMARY_LIB, FIDELITY_LIBRARY)

# --- scope del ranking de muestra (reducido para que el build sea rapido) -----------
RANK_LIB = "ai_v2"
RANK_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SPY", "QQQ", "GLD", "TLT"]
RANK_N_PATHS = 2
RANK_N_SCENARIOS = 4
RANK_WINDOW_DAYS = 300
RANK_CONFIGS = [
    ("Momentum (default)", "crypto_momentum", {}),
    ("Mean-reversion (default)", "mean_reversion", {}),
    ("Momentum (rapido)", "crypto_momentum",
     {"fast_sma_window": 5, "slow_sma_window": 20, "breakout_lookback": 3}),
    ("Mean-reversion (estricto)", "mean_reversion",
     {"entry_z": 1.5, "exit_z": 0.2, "lookback": 15}),
]

CHART_SYMBOLS = ["BTC/USDT", "SPY", "GLD"]
CHART_POINTS = 160  # downsample de las series de precio para el JSON

# --- panel de costes de ejecucion ---------------------------------------------------
# Muestra transversal del universo: dos cripto de primer nivel, tres altcoins, dos
# indices y dos macro. Los tamanos van de "orden de andar por casa" a institucional,
# que es donde el impacto y el techo de capacidad dejan de ser teoria.
COST_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "MATIC/USDT",
    "SPY", "QQQ", "GLD", "UUP",
]
COST_ORDER_USD = [1_000.0, 250_000.0, 25_000_000.0]


# ------------------------------------------------------------------ util ------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _downsample(values: np.ndarray, n: int) -> list[float]:
    if len(values) <= n:
        idx = np.arange(len(values))
    else:
        idx = np.linspace(0, len(values) - 1, n).astype(int)
    return [round(float(v), 2) for v in values[idx]]


def _phase_market_vol(phase) -> float:
    return phase.vol.get("EQUITY", 0.0)


def _scenario_regime(spec) -> tuple[str, float]:
    """Etiqueta de regimen por la autocorrelacion idio media (ponderada por longitud)."""
    total = sum(p.length_days for p in spec.phases) or 1
    ar = sum(p.idio_ar * p.length_days for p in spec.phases) / total
    if ar > 0.05:
        return "tendencia", ar
    if ar < -0.05:
        return "reversion", ar
    return "mixto", ar


# ------------------------------------------------------------ recoleccion -----------


def collect_synthetic(store: SyntheticStore) -> dict:
    """Escenarios de ai_v2: narrativa, fases con microestructura, y series de precio."""
    out: dict = {"library": PRIMARY_LIB, "scenarios": []}
    try:
        manifest = store.load_manifest(PRIMARY_LIB)
        specs = {s.id: s for s in store.load_specs(PRIMARY_LIB)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("No pude cargar %s: %s", PRIMARY_LIB, exc)
        return out

    out["created_at"] = manifest.created_at
    out["horizon_days"] = manifest.horizon_days
    out["n_paths"] = manifest.n_paths
    out["designer"] = manifest.designer

    for meta in manifest.scenarios:
        sid = meta["id"]
        spec = specs.get(sid)
        if spec is None:
            continue
        regime, ar = _scenario_regime(spec)
        phases = [
            {
                "length_days": p.length_days,
                "top_drift": _top_factor(p.drift),
                "top_vol": _top_factor(p.vol),
                "idio_ar": round(p.idio_ar, 3),
                "tail_dof": p.tail_dof,
                "vol_persistence": p.vol_persistence,
                "jump_intensity": p.jump_intensity,
                "crisis": _phase_market_vol(p) >= 0.025,
            }
            for p in spec.phases
        ]
        series: dict = {}
        try:
            bars = store.load_bars(PRIMARY_LIB, sid, 0)
            for sym in CHART_SYMBOLS:
                if sym in bars:
                    close = bar_schema.series(bars[sym], bar_schema.CLOSE).to_numpy()
                    base = close[0] if close[0] else 1.0
                    series[sym] = _downsample(close / base * 100.0, CHART_POINTS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sin barras para %s: %s", sid, exc)

        out["scenarios"].append(
            {
                "id": sid,
                "name": meta.get("name", sid),
                "narrative": meta.get("narrative", ""),
                "regime": regime,
                "idio_ar_avg": round(ar, 3),
                "n_shocks": len(spec.shocks),
                "phases": phases,
                "series": series,
            }
        )
    return out


def _top_factor(d: dict) -> str:
    if not d:
        return "-"
    f = max(d, key=lambda k: abs(d[k]))
    return f"{f} {d[f]:+.4f}"


def stylized_facts(store: SyntheticStore, n_paths: int = 2, n_scen: int = 12) -> dict:
    """Compara ai_v1 (iid) vs ai_v2 (retrofit) en autocorr, clustering y colas."""
    def survey(lib: str) -> dict | None:
        try:
            manifest = store.load_manifest(lib)
        except Exception:  # noqa: BLE001
            return None
        scen = manifest.scenarios[:n_scen]
        per_ac, absac, exc = [], [], []
        for meta in scen:
            acs, aacs, es = [], [], []
            for p in range(min(n_paths, manifest.n_paths)):
                try:
                    bars = store.load_bars(lib, meta["id"], p)
                except Exception:  # noqa: BLE001
                    continue
                for sym in ("BTC/USDT", "DOGE/USDT"):
                    if sym not in bars:
                        continue
                    c = bar_schema.series(bars[sym], bar_schema.CLOSE).to_numpy()
                    r = np.diff(np.log(c))
                    if len(r) < 10 or r.std() == 0:
                        continue
                    acs.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))
                    a = np.abs(r)
                    aacs.append(float(np.corrcoef(a[:-1], a[1:])[0, 1]))
                    z = (r - r.mean()) / r.std()
                    es.append(float(np.mean(np.abs(z) > 3.0)))
            if acs:
                per_ac.append(float(np.mean(acs)))
                absac.append(float(np.mean(aacs)))
                exc.append(float(np.mean(es)))
        if not per_ac:
            return None
        per = np.array(per_ac)
        return {
            "ac_min": round(float(per.min()), 3),
            "ac_max": round(float(per.max()), 3),
            "ac_spread": round(float(per.max() - per.min()), 3),
            "n_revert": int(np.sum(per < -0.05)),
            "n_trend": int(np.sum(per > 0.05)),
            "n_total": int(len(per)),
            "clustering": round(float(np.median(absac)), 3),
            "exceed_pct": round(float(np.median(exc)) * 100.0, 2),
        }

    return {lib: survey(lib) for lib in LIBRARY_LINEAGE}


def collect_strategies() -> dict:
    """Catalogo de estrategias + espacio de observacion, con logica explicable."""
    mom = CryptoMomentumStrategy().config
    mr = MeanReversionStrategy().config
    feat_desc = _feature_descriptions()
    return {
        "observation": {
            "own_asset": [{"name": n, "desc": feat_desc.get(n, "")} for n in OWN_ASSET_FEATURES],
            "regime": [{"name": n, "desc": feat_desc.get(n, "")} for n in REGIME_FEATURES],
        },
        "strategies": [
            {
                "id": "crypto_momentum",
                "name": "Momentum / seguimiento de tendencia",
                "regime": "tendencia",
                "idea": "Compra fuerza: cruce de medias al alza + ruptura de maximos, "
                        "con filtro de volatilidad. Gana cuando el precio tiene inercia.",
                "rules": [
                    "SMA rapida por encima de la SMA lenta (tendencia alcista).",
                    "Cierre supera el maximo de los ultimos N dias (ruptura Donchian).",
                    "ATR% por encima de un minimo (hay movimiento que capturar).",
                    "Stop = cierre - k*ATR; take-profit = cierre + m*ATR.",
                ],
                "params": _params_dict(mom),
            },
            {
                "id": "mean_reversion",
                "name": "Reversion a la media",
                "regime": "reversion",
                "idea": "Compra debilidad estirada: precio k*sigma por debajo de su media, "
                        "apostando a que revierte. Gana en rangos sin tendencia.",
                "rules": [
                    "z-score = (cierre - media) / sigma <= -entry_z (sobreventa).",
                    "sigma/precio por encima de un minimo (hay reversion que capturar).",
                    "Take-profit en la media (+ exit_z*sigma); stop = cierre - k*ATR.",
                    "Filtro de regimen opcional: solo rezagados, no en selloff amplio.",
                ],
                "params": _params_dict(mr),
            },
        ],
    }


def _params_dict(cfg) -> list[dict]:
    return [
        {"name": f.name, "value": getattr(cfg, f.name)}
        for f in dataclasses.fields(cfg)
        if f.name != "timeframe"
    ]


def _feature_descriptions() -> dict:
    return {
        "ret_1d": "Retorno log a 1 dia",
        "ret_5d": "Retorno log a 5 dias",
        "ret_20d": "Retorno log a 20 dias",
        "ret_60d": "Retorno log a 60 dias",
        "sma_ratio": "Ratio SMA rapida/lenta (tendencia)",
        "price_sma_z": "Distancia precio-SMA en z-score",
        "rsi": "RSI 14 (momentum)",
        "macd_norm": "Histograma MACD normalizado",
        "atr_pct": "ATR como % del precio (volatilidad)",
        "realized_vol_20": "Vol realizada 20d anualizada",
        "realized_vol_60": "Vol realizada 60d anualizada",
        "donchian_pos": "Posicion en el canal Donchian (0-1)",
        "dist_to_high": "Distancia al maximo de N dias (%)",
        "dist_to_low": "Distancia al minimo de N dias (%)",
        "volume_ratio": "Volumen vs su media (OJO: proxy sintetico)",
        "drawdown_from_peak": "Caida desde el pico reciente (%)",
        "relative_strength": "Fuerza relativa vs el mercado",
        "corr_to_market": "Correlacion movil al mercado",
        "breadth": "Amplitud: fraccion del universo > SMA50",
        "agg_vol": "Volatilidad agregada del universo",
    }


def strategy_signals_demo(store: SyntheticStore, synthetic: dict) -> dict:
    """Series reales anotadas con las entradas de cada primitiva (ilustrativo).

    Para cada primitiva escanea unos pocos (escenario, simbolo) y se queda con el que
    MAS senales dispara, de modo que el chart sea representativo (p.ej. mean-reversion
    no queda vacio si el activo elegido resulto estar en tendencia)."""
    demo: dict = {}
    scen = synthetic.get("scenarios", [])
    cand_syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "GLD", "SPY"]

    for strat_type, want in (("crypto_momentum", "tendencia"), ("mean_reversion", "reversion")):
        preferred = [s["id"] for s in scen if s["regime"] == want]
        scan = (preferred or [s["id"] for s in scen])[:4]
        strat = build_strategy(strat_type, {})
        best: dict | None = None
        for sid in scan:
            try:
                allbars = store.load_bars(PRIMARY_LIB, sid, 0)
            except Exception:  # noqa: BLE001
                continue
            for sym in cand_syms:
                if sym not in allbars:
                    continue
                bars = allbars[sym]
                signals = [
                    t
                    for t in range(60, len(bars), 2)
                    if _safe_signal(strat, sym, bars.iloc[: t + 1])
                ]
                if best is None or len(signals) > best["_n"]:
                    close = bar_schema.series(bars, bar_schema.CLOSE).to_numpy()
                    base = close[0] if close[0] else 1.0
                    norm = close / base * 100.0
                    idx = np.linspace(0, len(norm) - 1, min(CHART_POINTS, len(norm))).astype(int)
                    marks = sorted({int(np.searchsorted(idx, t)) for t in signals})
                    best = {
                        "scenario": sid,
                        "symbol": sym,
                        "series": _downsample(norm, CHART_POINTS),
                        "signals": [m for m in marks if 0 <= m < len(idx)],
                        "_n": len(signals),
                    }
        if best is not None:
            best.pop("_n", None)
            demo[strat_type] = best
    return demo


def _safe_signal(strat, sym, window) -> bool:
    try:
        return strat.generate_signal(sym, window) is not None
    except Exception:  # noqa: BLE001
        return False


def run_ranking(store: SyntheticStore) -> dict:
    """
    Ranking real sobre una muestra reducida de ai_v2.

    Rankea por CVaR@25% del HEADLINE score out-of-sample (Sharpe - lambda*turnover -
    kappa*maxDD). Ademas de las estrategias corre los BASELINES pasivos sobre las mismas
    muestras (el gate que hay que batir para 'aprobar') y descuenta el sobreajuste por
    multiples pruebas con PBO y DSR sobre la distribucion de scores del propio ranking.
    """
    result: dict = {
        "scope": {
            "library": RANK_LIB,
            "universe": RANK_UNIVERSE,
            "n_scenarios": RANK_N_SCENARIOS,
            "n_paths": RANK_N_PATHS,
            "window_days": RANK_WINDOW_DAYS,
            "weights": DEFAULT_HEADLINE_WEIGHTS.as_dict(),
        },
        "rows": [],
        "baselines": [],
        "distributions": {},
        "overfit": {},
    }
    try:
        base_config = load_config(ROOT / "config" / "synthetic.toml")
        manifest = store.load_manifest(RANK_LIB)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ranking no disponible: %s", exc)
        return result

    base_config = dataclasses.replace(
        base_config, runner=dataclasses.replace(base_config.runner, symbols=list(RANK_UNIVERSE))
    )
    anchor = datetime.fromisoformat(manifest.anchor)
    warmup = base_config.runner.lookback_days + 5
    start = anchor + timedelta(days=warmup)
    end = start + timedelta(days=RANK_WINDOW_DAYS)

    # Escenarios elegidos por diversidad de regimen (por su idio_ar medio, si disponible).
    specs = {s.id: s for s in store.load_specs(RANK_LIB)}
    scen_ids = [m["id"] for m in manifest.scenarios]
    scen_ids = sorted(scen_ids, key=lambda i: _scenario_regime(specs[i])[1] if i in specs else 0)
    chosen = _spread_pick(scen_ids, RANK_N_SCENARIOS)
    result["scope"]["scenarios"] = chosen

    specs = [(label, stype, StrategySpec(type=stype, id=label, params=params))
             for label, stype, params in RANK_CONFIGS]

    # Una sola pasada por muestra: el parquet se lee una vez y sobre esas mismas barras
    # se puntuan todas las configuraciones Y los baselines. Asi la comparacion es
    # pareada (mismo mundo para todos), que es lo que el gate necesita.
    scores: dict[str, list[float]] = {label: [] for label, _, _ in specs}
    sharpes: dict[str, list[float]] = {label: [] for label, _, _ in specs}
    baseline_scores: dict[str, list[float]] = {}
    baseline_stats: dict[str, list] = {}
    oos_obs: list[int] = []

    for sid in chosen:
        for p in range(RANK_N_PATHS):
            try:
                allbars = store.load_bars(RANK_LIB, sid, p)
                bars = {s: allbars[s] for s in RANK_UNIVERSE if s in allbars}
            except Exception as exc:  # noqa: BLE001
                logger.warning("barras no disponibles %s/%s: %s", sid, p, exc)
                continue

            for label, _, spec in specs:
                try:
                    ev = evaluate_sample_detailed(
                        base_config, spec, bars, start, end, split_ratio=0.7
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("eval fallo %s/%s: %s", label, sid, exc)
                    continue
                scores[label].append(ev.score)
                sharpes[label].append(ev.sharpe)
                oos_obs.append(ev.oos_observations)

            try:
                for name, baseline in evaluate_baselines(
                    base_config, bars, start, end, split_ratio=0.7
                ).items():
                    baseline_scores.setdefault(name, []).append(baseline.score)
                    baseline_stats.setdefault(name, []).append(baseline)
            except Exception as exc:  # noqa: BLE001
                logger.warning("baselines fallaron %s/%s: %s", sid, p, exc)

    n_samples = max((len(v) for v in scores.values()), default=0)
    usable_baselines = {k: v for k, v in baseline_scores.items() if len(v) == n_samples}

    for name, values in sorted(usable_baselines.items()):
        stats = aggregate_reward(values)
        result["baselines"].append(
            {
                "name": name,
                "label": BASELINE_LABELS.get(name, name),
                "symbols": len(baseline_stats[name][0].symbols),
                **_stats_row(stats),
            }
        )
        result["distributions"][name] = [round(s, 3) for s in values]

    for label, stype, _ in specs:
        if not scores[label]:
            continue
        stats = aggregate_reward(scores[label])
        verdict = gate(scores[label], usable_baselines)
        result["rows"].append(
            {
                "label": label,
                "type": stype,
                "approved": verdict.approved,
                "margin": round(verdict.margin, 3),
                "win_rate_pct": round(verdict.win_rate_pct, 1),
                **_stats_row(stats),
            }
        )
        result["distributions"][label] = [round(s, 3) for s in scores[label]]

    result["rows"].sort(key=lambda r: r["cvar25"], reverse=True)
    result["gate"] = {
        "best_baseline": (
            max(usable_baselines, key=lambda n: aggregate_reward(usable_baselines[n]).reward)
            if usable_baselines else None
        ),
        "missing": [n for n in sorted(BASELINE_LABELS) if n not in usable_baselines],
    }
    result["overfit"] = _overfit_report(scores, sharpes, oos_obs)
    return result


def collect_costs(store: SyntheticStore) -> dict:
    """Lo que cuesta EJECUTAR en cada mercado del universo.

    No corre backtests: toma un escenario real de la libreria, resuelve la liquidez de
    cada simbolo con la misma costura que usa el motor (mediana de volumen y volatilidad
    de las ultimas barras cerradas) y evalua el modelo para ordenes de tamano creciente.
    Es la evidencia de que la friccion dejo de ser una constante."""
    out: dict = {"library": PRIMARY_LIB, "sizes_usd": COST_ORDER_USD, "rows": []}
    try:
        config = load_config(ROOT / "config" / "synthetic.toml")
        manifest = store.load_manifest(PRIMARY_LIB)
        bars = store.load_bars(PRIMARY_LIB, manifest.scenarios[0]["id"], 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Panel de costes no disponible: %s", exc)
        return out

    symbols = [s for s in COST_SYMBOLS if s in bars and len(bars[s]) > 2]
    if not symbols:
        logger.warning("Panel de costes: ningun simbolo de la muestra esta en %s", PRIMARY_LIB)
        return out

    model = config.execution.slippage
    out["max_participation"] = config.execution.max_participation
    out["fee_rate"] = config.execution.fee_rate

    # Reloj al final de la serie: la liquidez se mide con barras ya cerradas.
    clock = HistoricalClock(bars[symbols[0]].index[-1].to_pydatetime())
    provider = BarLiquidityProvider(HistoricalDataSource(bars, clock), clock)

    for symbol in symbols:
        price = float(bar_schema.series(bars[symbol], bar_schema.CLOSE).iloc[-2])
        asset_class = AssetClass.CRYPTO if "/" in symbol else AssetClass.STOCK
        snapshot = provider.snapshot(symbol)

        out["rows"].append(
            {
                "symbol": symbol,
                "spread_bps": model.base_spread_bps(symbol, asset_class),
                "vol_pct": round(100.0 * (snapshot.recent_volatility or 0.0), 2),
                "capacity_usd": round((snapshot.bar_volume or 0.0)
                                      * config.execution.max_participation * price),
                "slippage_bps": [
                    round(model.slippage_bps(symbol, usd / price, snapshot, asset_class), 1)
                    for usd in COST_ORDER_USD
                ],
            }
        )

    out["rows"].sort(key=lambda r: r["slippage_bps"][-1], reverse=True)
    return out


def _stats_row(stats) -> dict:
    return {
        "cvar25": round(stats.cvar25, 3),
        "mean": round(stats.mean, 3),
        "p25": round(stats.p25, 3),
        "std": round(stats.std, 3),
        "worst": round(stats.worst, 3),
        "best": round(stats.best, 3),
        "reward": round(stats.reward, 3),
        "n": stats.n,
    }


def _overfit_report(
    scores: dict[str, list[float]],
    sharpes: dict[str, list[float]],
    oos_obs: list[int],
) -> dict:
    """PBO y DSR sobre la distribucion de scores del propio ranking: cuantas
    configuraciones se probaron y si el ganador sobrevive al descuento."""
    columns = [v for v in scores.values() if v]
    if not columns:
        return {}

    width = min(len(c) for c in columns)
    matrix = [[c[i] for c in columns] for i in range(width)]
    pbo = probability_of_backtest_overfitting(matrix)

    trial_sharpes = [sum(v) / len(v) for v in sharpes.values() if v]
    winner = max(scores, key=lambda k: aggregate_reward(scores[k]).reward)
    observed = sum(sharpes[winner]) / len(sharpes[winner])
    n_obs = sorted(oos_obs)[len(oos_obs) // 2] if oos_obs else 0
    dsr = deflated_sharpe_ratio(observed, trial_sharpes, n_obs)

    return {"pbo": pbo.as_dict(), "dsr": dsr.as_dict(), "winner": winner}


def _spread_pick(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).astype(int)
    return [items[i] for i in idx]


def collect_kpis(store: SyntheticStore, synthetic: dict) -> dict:
    def lib_stats(lib):
        try:
            m = store.load_manifest(lib)
            return {"scenarios": m.num_scenarios, "paths": m.n_paths, "samples": m.num_samples}
        except Exception:  # noqa: BLE001
            return None

    return {
        **{lib: lib_stats(lib) for lib in LIBRARY_LINEAGE},
        "lineage": list(LIBRARY_LINEAGE),
        "n_strategies": 2,
        "n_own_features": len(OWN_ASSET_FEATURES),
        "n_regime_features": len(REGIME_FEATURES),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "commit_count": _git("rev-list", "--count", "HEAD"),
        "generated_at": _git("log", "-1", "--format=%cd", "--date=short"),
    }


def collect_calibration() -> dict | None:
    """Evidencia del estudio que fija lambda y kappa (data/calibration).

    Se LEE del informe publicado; no se recalcula. El estudio son cientos de backtests
    reales y el dashboard debe seguir siendo regenerable en minutos."""
    report = load_calibration_report(ROOT / CALIBRATION_REPORT)
    if not report:
        logger.warning("Sin informe de calibracion: el panel de pesos saldra vacio")
        return None

    weights = DEFAULT_HEADLINE_WEIGHTS
    chosen = grid_point(report, weights.lambda_turnover, weights.kappa_maxdd)
    if chosen is None:
        return None

    audit = report["cost_audit_active"]
    return {
        "weights": weights.as_dict(),
        "library": report["plan"]["library_id"],
        "n_configs": len(report["configs"]["kept"]),
        "n_samples": report["configs"]["n_samples"],
        "n_backtests": len(report["configs"]["kept"]) * report["configs"]["n_samples"],
        "lambdas": report["grid"]["lambdas"],
        "kappas": report["grid"]["kappas"],
        "points": report["grid"]["points"],
        "chosen": chosen,
        "neutral": grid_point(report, 0.0, 0.0),
        "prev": grid_point(report, 0.5, 1.0),  # los pesos razonados "a ojo" que esto sustituye
        "best": report["ranked_by_stability"][0],
        # 1 ganadora en toda la rejilla = los pesos no cambian la decision (hallazgo central).
        "n_winners": len({p["selected_config"] for p in report["grid"]["points"]}),
        "cost": audit,
        "share_pct": 100.0 * weights.lambda_turnover / audit["implied_lambda_median"],
        "generated_at": report["generated_at"][:10],
    }


def _fidelity_rows(report: dict) -> tuple[list[dict], dict]:
    """Metricas y correlacion cruzada de un informe, con lo que necesita la vista."""
    metrics = [
        {
            **m,
            "is_target": m["key"] in TARGET_METRIC_KEYS,
            "decimals": metric(m["key"]).decimals,
        }
        for m in report["metrics"]
    ]
    return metrics, {**report["cross_correlation"], "is_target": True, "decimals": 3}


def collect_fidelity() -> dict | None:
    """Los stylized-facts de la libreria realista contra el historico real, y contra la
    libreria ANTERIOR (data/fidelity).

    Se publican los dos informes juntos a proposito: ai_v2 es el generador cuyo hueco se
    midio y ai_v3 el que lo cierra, asi que la vista no ensena "el sintetico se parece al
    real" sino "esto es lo que se arreglo y cuanto". Sin el antes, la correccion seria una
    afirmacion sin control.

    Se LEE de los informes publicados; no se recalcula. Medirlo exige descargar ocho anos
    de historico y recorrer la libreria entera, y el dashboard tiene que seguir siendo
    regenerable en minutos."""
    report = load_fidelity_report(ROOT / fidelity_report_path(FIDELITY_LIBRARY))
    if not report:
        logger.warning("Sin informe de fidelidad: el panel sintetico-vs-real saldra vacio")
        return None

    plan = report["plan"]
    metrics, cross = _fidelity_rows(report)

    baseline = load_fidelity_report(ROOT / fidelity_report_path(FIDELITY_BASELINE_LIBRARY))
    before = None
    if baseline:
        prev_metrics, prev_cross = _fidelity_rows(baseline)
        before = {
            "library": baseline["plan"]["library_id"],
            "by_key": {
                row["key"]: {
                    "synth_median": row["synth_median"],
                    "coverage_pct": row["coverage_pct"],
                    "ratio": row["ratio"],
                    "rank_corr": row["rank_corr"],
                }
                for row in (*prev_metrics, prev_cross)
            },
            "summary": baseline["summary"],
            "generated_at": baseline["generated_at"][:10],
        }
    else:
        logger.warning("Sin informe de %s: la vista no podra comparar", FIDELITY_BASELINE_LIBRARY)

    return {
        "library": plan["library_id"],
        "exchange": plan["exchange"],
        "real_start": plan["real_window"]["start"],
        "real_end": plan["real_window"]["end"],
        "window_days": plan["window_days"],
        "step_days": plan["step_days"],
        "overlap": plan["windows_overlap"],
        "n_paths": plan["n_paths"],
        "n_scenarios": plan["n_scenarios"],
        "missing": plan["missing_symbols"],
        "metrics": metrics,
        "cross": cross,
        "summary": report["summary"],
        "acceptance": report["acceptance"],
        "before": before,
        "generated_at": report["generated_at"][:10],
    }


def _fold_geometry(plan: dict) -> dict:
    """La geometria REAL de los folds del estudio, en fracciones del rango, para dibujar
    el diagrama de bandas. No corre backtests: reconstruye los mismos folds que se
    corrieron, asi que lo que se dibuja es lo que se ejecuto, no una ilustracion."""
    from ai_trader.backtest.validation import build_folds

    start = datetime.fromisoformat(plan["start"])
    end = datetime.fromisoformat(plan["end"])
    span = (end - start).days or 1

    def bands(folds) -> list[dict]:
        return [
            {
                "label": f.label,
                "train": [
                    {"a": (b.start - start).days / span, "b": (b.end - start).days / span}
                    for b in f.train
                ],
                "test": [
                    {"a": (b.start - start).days / span, "b": (b.end - start).days / span}
                    for b in f.test
                ],
            }
            for f in folds
        ]

    common = dict(purge_days=plan["purge_days"])
    wf = build_folds(start, end, scheme="walk_forward", n_folds=plan["n_folds"], **common)
    cpcv = build_folds(
        start, end, scheme="cpcv",
        n_groups=plan["n_groups"], n_test_groups=plan["n_test_groups"], **common,
    )
    single = build_folds(start, end, scheme="single_split", **common)
    return {
        "span_days": span,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "single_split": bands(single),
        "walk_forward": bands(wf),
        "cpcv": bands(cpcv),
    }


def collect_validation() -> dict | None:
    """Comparacion medida entre el corte unico 70/30 y la validacion multiventana
    (data/validation).

    Se LEE del informe publicado; no se recalcula. Cada unidad del estudio son ~20
    ventanas de backtest real y el dashboard tiene que seguir siendo regenerable en
    minutos."""
    report = load_validation_report(ROOT / VALIDATION_REPORT)
    if not report:
        logger.warning("Sin informe de validacion: el panel multiventana saldra vacio")
        return None

    plan = report["plan"]
    rows = report["rows"]

    # Una muestra representativa para dibujar la distribucion de ventanas: la que tenga
    # el optimismo mas cercano a la mediana, para no elegir el caso mas favorable.
    pairs = [r for r in rows if r["single_score"] is not None]
    example = None
    if pairs:
        target = report["optimism"]["walk_forward"]["median"]
        pick = min(pairs, key=lambda r: abs((r["single_score"] - r["walk_forward"]["median"]) - target))
        example = {
            "config_id": pick["config_id"],
            "scenario_id": pick["scenario_id"],
            "single": round(pick["single_score"], 3),
            "walk_forward": [round(s, 3) for s in pick["walk_forward"]["scores"]],
            "cpcv": [round(s, 3) for s in pick["cpcv"]["scores"]],
        }

    return {
        "geometry": _fold_geometry(plan),
        "library": plan["library_id"],
        "n_samples": len(plan["scenario_ids"]) * plan["n_paths"],
        "n_configs": len(plan["config_ids"]),
        "n_units": len(rows),
        "n_folds_wf": plan["n_folds"],
        "n_folds_cpcv": rows[0]["cpcv"]["n_folds"] if rows else 0,
        "n_groups": plan["n_groups"],
        "n_test_groups": plan["n_test_groups"],
        "purge_days": plan["purge_days"],
        "embargo_days": rows[0]["walk_forward"]["embargo_days"] if rows else None,
        "optimism": report["optimism"],
        "dispersion": report["dispersion"],
        "svn": report["signal_vs_noise"],
        "rank_agreement": report["rank_agreement"],
        "flips": report["decision_flips"],
        "leakage": report["leakage"],
        "gate": report["gate"],
        # Las tres series pareadas sobre las MISMAS unidades: es la comparacion que
        # justifica la palabra "optimismo".
        "paired": {
            "single": [round(r["single_score"], 3) for r in pairs],
            "walk_forward": [round(r["walk_forward"]["median"], 3) for r in pairs],
            "cpcv": [round(r["cpcv"]["median"], 3) for r in pairs],
        },
        "example": example,
        "generated_at": report["generated_at"][:10],
    }


def collect_roadmap() -> list[dict]:
    """Evoluciones pendientes, ordenadas por criticidad, con prompt para Claude Code."""
    from ai_trader.synthetic import retrofit  # noqa: F401  (asegura que el modulo existe)
    ranks = [r["rank"] for r in ROADMAP]
    if sorted(ranks) != list(range(1, len(ROADMAP) + 1)):
        raise ValueError(f"Los rangos del roadmap deben ser 1..N sin huecos: {sorted(ranks)}")
    groups = {r["group"] for r in ROADMAP}
    unknown = groups - {g["key"] for g in ROADMAP_GROUPS}
    if unknown:
        raise ValueError(f"Grupos de roadmap desconocidos: {sorted(unknown)}")
    return sorted(ROADMAP, key=lambda r: r["rank"])


def build() -> None:
    store = SyntheticStore(ROOT / "data" / "synthetic")
    logger.info("Recolectando datos sinteticos...")
    synthetic = collect_synthetic(store)
    logger.info("Stylized facts ai_v1 vs ai_v2...")
    facts = stylized_facts(store)
    logger.info("Fidelidad sintetico vs real (informe publicado)...")
    fidelity = collect_fidelity()
    logger.info("Catalogo de estrategias...")
    strategies = collect_strategies()
    logger.info("Demo de señales...")
    signals = strategy_signals_demo(store, synthetic)
    logger.info("Costes de ejecucion por simbolo...")
    costs = collect_costs(store)
    logger.info("Ranking (muestra reducida, puede tardar unos minutos)...")
    ranking = run_ranking(store)
    kpis = collect_kpis(store, synthetic)

    data = {
        "kpis": kpis,
        "synthetic": synthetic,
        "facts": facts,
        "fidelity": fidelity,
        "strategies": strategies,
        "signals": signals,
        "costs": costs,
        "ranking": ranking,
        "calibration": collect_calibration(),
        "validation": collect_validation(),
        "roadmap": collect_roadmap(),
        "roadmap_groups": ROADMAP_GROUPS,
    }

    from dashboard.template import render_html  # import tardio: template en modulo aparte

    OUT_HTML.write_text(render_html(data), encoding="utf-8")
    logger.info("Dashboard escrito en %s", OUT_HTML)


# --- catalogo de evoluciones pendientes (con prompts detallados para Claude Code) ---
#
# ORDEN: `rank` 1..N de mayor a menor criticidad. No es una lista de deseos ordenada por
# gusto: el criterio es el de la revision externa de 2026-08-10, y su asimetria de coste
# es la que manda -- una estrategia anadida hoy se re-evalua gratis cuando el juez mejore,
# pero un juez malo contamina TODO lo que puntue mientras siga malo. Por eso el sustrato
# (fidelidad) y el juez (validacion multiventana) van delante de la cosecha (estrategias),
# y por eso el paper trading en vivo se lanza en paralelo: es lo unico que compra tiempo
# de calendario, que no se puede comprimir despues.
#
# FOCO: cripto. Renta variable y mercados de prediccion quedan en segundo plano de forma
# EXPLICITA (grupo 'segundo-plano'), no por olvido: toda la evidencia empirica del repo
# -fidelidad contra Binance, calibracion de pesos, estudio de validacion- es cripto, y la
# pata de renta variable no tiene ni un solo dato real detras.

ROADMAP_GROUPS = [
    {
        "key": "ahora",
        "title": "Ahora",
        "subtitle": "El bucle abierto: un sustrato fiel, un juez honesto y el reloj corriendo. "
                    "Nada de lo que se puntue mientras tanto vale mas que el juez que lo puntua.",
    },
    {
        "key": "despues",
        "title": "Despues",
        "subtitle": "Rigor del juez: potencia estadistica, fugas menores y transparencia de los "
                    "descuentos. Barato, y necesario antes de creerse un ranking.",
    },
    {
        "key": "no-prioritario",
        "title": "No prioritario",
        "subtitle": "Trabajo legitimo que NO se aborda todavia. Anadir candidatos a un juez en el "
                    "que aun no se confia solo multiplica el problema de multiples pruebas.",
    },
    {
        "key": "segundo-plano",
        "title": "Segundo plano (no cripto)",
        "subtitle": "Renta variable y mercados de prediccion. Aparcados a proposito hasta que el "
                    "bucle cripto -sintetico, real, paper- este cerrado y medido.",
    },
]

ROADMAP = [
    {
        "id": "rank-transfer-real-vs-synthetic",
        "rank": 1,
        "group": "ahora",
        "priority": "critica",
        "title": "Estudio de transferencia: ¿ordena el mundo sintetico las estrategias como el real?",
        "line": "B/D", "status": "pendiente", "impact": "alto", "effort": "alto",
        "evidence": "Sin medir. La fidelidad mide hechos estilizados (curtosis, clustering, "
                    "correlacion); NO mide transferencia de ranking, que es lo unico que el "
                    "producto necesita de verdad del generador.",
        "why": "Es el unico bucle que sigue completamente abierto, y es barato porque las dos "
               "mitades ya existen: fidelity_study ya descarga y cachea 8 anos de Binance, y "
               "multiwindow ya sabe correr CPCV purgado sobre cualquier bars_dict. La pregunta "
               "que responde es la que decide la arquitectura del producto: si el Spearman entre "
               "el ranking real y el sintetico sale 0,3-0,5, el generador es una maquina de "
               "pre-cribado legitima y el flujo definitivo queda 'real como sustrato primario del "
               "ranking, sintetico como capa de estres y veto'. Si sale ~0, es mucho mejor "
               "saberlo ahora que despues de disenar diez estrategias contra el.",
        "prompt": (
            "Proyecto ai-trader (Python). Hay un bucle sin cerrar: se ha medido la FIDELIDAD del "
            "generador sintetico (hechos estilizados: colas, clustering, correlacion, en "
            "src/ai_trader/synthetic/fidelity_study.py) pero NO se ha medido lo unico que el "
            "producto necesita de el: si el mundo sintetico ORDENA las estrategias igual que el "
            "mundo real. Un generador puede tener colas perfectas y ordenar al reves.\n"
            "\n"
            "TAREA: crear src/ai_trader/scoring/transfer_study.py, un estudio de transferencia de "
            "ranking real-vs-sintetico, y publicar su informe en data/transfer/.\n"
            "\n"
            "(a) REJILLA DE CONFIGURACIONES. Reusa exactamente la que ya existe: "
            "`candidate_specs` de src/ai_trader/scoring/weight_calibration.py con el hipercubo "
            "latino de semilla STUDY_SEED = 20260809 y 8 configuraciones por familia sobre "
            "('crypto_momentum', 'mean_reversion') -> 16 specs. Usar la MISMA rejilla que el "
            "estudio de pesos hace los dos estudios comparables sin coste extra.\n"
            "(b) LADO REAL. Barras diarias reales de los simbolos cripto del universo operable "
            "(config/default.toml, 24 pares USDT), reusando `fetch_real_bars` y "
            "`CachedBarsProvider` de synthetic/fidelity_study.py con --offline (ya hay cache de "
            "2017-09-01 a 2026-01-01). Evalua cada config con "
            "`validate_multiwindow` (src/ai_trader/scoring/multiwindow.py) en esquema CPCV, purga "
            "= max_holding_days del runner y embargo por defecto, con los baselines y el gate "
            "activados. Cuida dos cosas: el config base tiene que ser el de universo CRIPTO (para "
            "que `periods_per_year_for_symbols` de 365 y no 252), y los simbolos sin historico "
            "suficiente se declaran y se omiten, no se rellenan.\n"
            "(c) LADO SINTETICO. La MISMA rejilla sobre ai_v3 (la libreria corregida; si aun no "
            "existe, sobre ai_v2 dejandolo escrito en el informe), con el mismo esquema CPCV y "
            "los mismos parametros de purga/embargo. Agrega por configuracion poniendo en comun "
            "TODOS los scores fold x muestra en una sola distribucion y tomando su CVaR -no CVaR "
            "de CVaR, que compone dos conservadurismos y dispara la varianza del estimador.\n"
            "(d) TRANSFERENCIA. Spearman entre los dos rankings de 16 configuraciones, con "
            "intervalo de confianza por bootstrap. Anade dos lecturas mas: solape del top-k (¿las "
            "4 mejores del sintetico caen en la mitad buena del real?) y el signo de las "
            "discrepancias grandes. REGLA DE DECISION, escrita en el informe y en el dashboard: "
            "rho >= 0,3 -> el generador es un pre-cribado legitimo y el flujo definitivo es 'real "
            "como sustrato primario del ranking, sintetico como capa de estres y veto'; rho ~ 0 "
            "-> no se disenan estrategias contra el sintetico, y hay que decirlo en la "
            "metodologia.\n"
            "\n"
            "HONESTIDAD ESTADISTICA (declarala en el informe, no la escondas): el historico real "
            "es UN solo camino, asi que ese lado no tiene ensemble y su unica fuente de "
            "dispersion son los folds del CPCV -usa bootstrap por bloques sobre folds para el "
            "intervalo, no bootstrap iid-. Ademas los 24 simbolos son los que cotizan HOY: hay "
            "sesgo de supervivencia en el lado real, y conviene dejarlo por escrito porque juega "
            "a favor del real y por tanto en contra de la hipotesis que se quiere validar.\n"
            "\n"
            "Salida: data/transfer/report_<lib>.json con el plan completo (rejilla, ventana, "
            "esquema, purga/embargo, simbolos usados y omitidos), los dos rankings, el rho con su "
            "IC y el veredicto. Vista nueva en el dashboard (o ampliacion de 'Fidelidad') que "
            "muestre el scatter de rangos real-vs-sintetico y el veredicto. Determinismo + tests "
            "+ .venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "line-d-cpcv-two-stage-cem",
        "rank": 2,
        "group": "ahora",
        "priority": "critica",
        "title": "CPCV en dos etapas dentro del optimizador (que el CEM deje de puntuar con el corte unico)",
        "line": "D", "status": "pendiente", "impact": "alto", "effort": "medio",
        "evidence": "mom_default en crypto_winter: +2,63 con el corte unico y -2,20 con CPCV. En "
                    "crypto_bull_supercycle: -0,18 vs -0,60, con dispersion entre folds de "
                    "sigma ~ 2-3 unidades de Sharpe.",
        "why": "El optimizador sigue puntuando con el corte que su propio estudio desacredita: "
               "run_optimization -> evaluate_sample_detailed -> BacktestEngine.run(split_ratio=0.7). "
               "El corte unico no esta SESGADO, esta ARBITRARIO -y arbitrario es letal para un "
               "optimizador: el CEM escala un paisaje cuyo relieve depende de que tramo de "
               "historia cayo en el 30% de test. La objecion obvia es el coste (CPCV multiplica "
               "x15), y la solucion no es elegir sino separar cribado de decision: dos etapas.",
        "prompt": (
            "Proyecto ai-trader (Python). La validacion multiventana esta implementada, testeada "
            "y MEDIDA: src/ai_trader/backtest/validation.py (geometria de folds con purga y "
            "embargo), BacktestEngine.run_folds, src/ai_trader/scoring/multiwindow.py "
            "(validate_multiwindow -> distribucion robusta) y el estudio publicado en "
            "data/validation/report_ai_v2.json. Pero el camino que usa el OPTIMIZADOR sigue "
            "siendo el corte unico: scoring/sample_eval.py::evaluate_sample_detailed llama a "
            "BacktestEngine.run(split_ratio=0.7), y scoring/optimize.py::run_optimization agrega "
            "un score por muestra. El objetivo del CEM es, literalmente, el score del split "
            "70/30. El propio informe muestra por que eso es un problema: mom_default puntua "
            "+2,63 con el corte unico y -2,20 con CPCV en crypto_winter, y -0,18 vs -0,60 en "
            "crypto_bull_supercycle, con dispersion entre folds de sigma ~ 2-3 unidades de "
            "Sharpe.\n"
            "\n"
            "TAREA: arquitectura en DOS ETAPAS dentro de run_optimization.\n"
            "(1) CRIBADO (barato): el CEM sigue explorando con el corte unico o con un "
            "walk-forward corto (2-3 folds). Es el paisaje que se escala, y basta con que sea "
            "informativo, no con que sea la verdad.\n"
            "(2) FINALISTAS (caro y decisivo): los top-k del CEM (5-10 configuraciones, "
            "parametrizable) se RE-EVALUAN con CPCV completo, y es esa evaluacion la que decide "
            "el ranking final y la que alimenta el gate de baselines, el DSR y el PBO. El "
            "resultado de la etapa 1 no debe aparecer como si fuera el veredicto en ningun sitio.\n"
            "\n"
            "AGREGACION -- este matiz es el que hay que hacer bien: agrega poniendo en comun "
            "TODOS los scores fold x muestra en UNA sola distribucion y toma el CVaR de eso. NO "
            "hagas CVaR-de-CVaR (cola de la cola): compone dos conservadurismos y deja un "
            "estimador con varianza enorme sobre 5 folds. Documenta la eleccion en el docstring "
            "de la funcion, con esa razon.\n"
            "\n"
            "CONSERVAR PARA EL DESCUENTO: la evaluacion multiventana tiene que seguir devolviendo "
            "lo que DSR y PBO necesitan (oos_observations y los momentos de los retornos), "
            "sumando/encadenando las ventanas en vez de reportar las de un unico corte; hoy esos "
            "campos salen de SampleEvaluation en scoring/sample_eval.py.\n"
            "\n"
            "COSTE: CPCV multiplica por ~15 el numero de backtests por muestra, y por eso solo lo "
            "pagan los finalistas. Aun asi, mide y reporta el coste real, y ofrece subsampleo de "
            "paths (ya existe `n_paths` con aviso por log) y paralelizacion por muestra "
            "(multiprocessing, como hacen weight_study y validation_study).\n"
            "\n"
            "EVIDENCIA DEL CAMBIO: re-corre el ranking del dashboard con el esquema nuevo, y "
            "reporta explicitamente cuantos finalistas CAMBIAN de posicion entre la etapa 1 y la "
            "etapa 2, y cuantos aprueban el gate en cada una. Ese numero es el valor del cambio; "
            "si es cero, tambien hay que publicarlo. Tests + determinismo + "
            ".venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "paper-trading-live",
        "rank": 3,
        "group": "ahora",
        "priority": "alta",
        "title": "Poner el paper trading a correr en vivo (y la vista del dashboard que lo lea)",
        "line": "Live", "status": "pendiente", "impact": "alto", "effort": "bajo",
        "evidence": "Cero desarrollo pendiente para arrancar: runner, motor de riesgo, ejecucion "
                    "en papel con microestructura y bot de Telegram existen y funcionan (ciclo "
                    "automatico cada 900 s, estado persistido en data/runtime_state.json).",
        "why": "No compite con lo demas: se lanza YA, en paralelo, porque compra la unica cosa "
               "que no se puede comprimir despues -tiempo de calendario. La divergencia "
               "live-vs-backtest necesita meses para ser medible, asi que cada semana que el bot "
               "no corre es una semana perdida al final del proyecto. Ademas, corriendo en vivo "
               "es cuando Polymarket empieza a generar el historico de midpoints que hoy no "
               "existe, que es la unica via barata hacia un backtest de mercados de prediccion. "
               "Con el universo actual y sin tocar tickers.",
        "prompt": (
            "Proyecto ai-trader (Python). El sistema ya opera en paper y no le falta desarrollo "
            "para arrancar: TradingRunner (src/ai_trader/app/runner.py), RiskEngine, "
            "PaperExecutionEngine con el modelo de microestructura "
            "(src/ai_trader/execution/microstructure.py), ExecutionRouter, estado persistido en "
            "JsonStateStore (data/runtime_state.json) y bot de Telegram con ciclo automatico cada "
            "AUTO_CYCLE_INTERVAL_SECONDS = 900 (src/ai_trader/bots/telegram_bot.py). El arranque "
            "es src/ai_trader/main.py con TELEGRAM_BOT_TOKEN y TELEGRAM_ALLOWED_CHAT_IDS.\n"
            "\n"
            "OBJETIVO: dejarlo corriendo de forma continua y que cada ciclo deje HUELLA "
            "auditable, porque hoy el estado guarda la foto (posiciones, PnL) pero no la pelicula "
            "(que se decidio, con que precio, cuanto deslizamiento se cobro), y la pelicula es "
            "exactamente el material con el que dentro de unos meses se medira la divergencia "
            "live-vs-backtest.\n"
            "\n"
            "(1) DIARIO DE CICLOS. Anade un registro append-only en JSONL (p.ej. "
            "data/live/cycles.jsonl) que escriba, por ciclo: marca de tiempo, simbolos "
            "evaluados, senales generadas con su confianza, decisiones del motor de riesgo "
            "(aprobada/rechazada y motivo), ordenes enviadas, fills con precio, comision y "
            "slippage_bps REALMENTE cobrado (viene en ExecutionResult), equity marcada a mercado "
            "y exposicion desplegada. Rotacion por tamano o por mes, y escritura resistente a "
            "reinicio (append + fsync, nunca reescribir el fichero entero). Este fichero es un "
            "activo del proyecto: no lo metas en data/ sin decidir si va a git (recomendado: "
            "ignorado en git, pero con su ruta documentada).\n"
            "(2) DURABILIDAD DEL ESTADO. Copia de seguridad rotatoria de runtime_state.json antes "
            "de cada escritura y arranque tolerante a fichero corrupto (si no parsea, avisa por "
            "Telegram y arranca del backup, no de cero silenciosamente).\n"
            "(3) PROCEDIMIENTO DE ARRANQUE documentado en el README: variables de entorno, "
            "AI_TRADER_CONFIG=config/default.toml (universo de 24 pares cripto), como se lanza "
            "para que sobreviva a un reinicio de la maquina (servicio/tarea programada en "
            "Windows), como se pausa desde Telegram y como se comprueba que sigue vivo.\n"
            "(4) VISTA DE PAPER TRADING EN EL DASHBOARD. La seccion 'Paper trading' hoy es un "
            "placeholder. Conectala a las dos fuentes reales (JsonStateStore + el JSONL de "
            "ciclos): curva de equity marcada a mercado, tabla de posiciones abiertas y cerradas "
            "con PnL neto de comisiones, y metricas de riesgo (exposicion desplegada, nº de "
            "posiciones frente al maximo, drawdown de cuenta). Reusa los helpers SVG del template "
            "(dashboard/template.py) y manten el HTML autocontenido. Si no hay datos todavia, la "
            "vista debe decir 'sin ciclos registrados' en vez de romperse.\n"
            "(5) OPCIONAL, y explicitamente de segunda prioridad: registrar en el mismo diario el "
            "midpoint de una watchlist de mercados de Polymarket aunque no se opere ninguno. Es "
            "solo lectura y coste casi nulo, y es lo unico que puede construir el historico que "
            "hoy hace imposible backtestear mercados de prediccion.\n"
            "\n"
            "NO amplies el universo ni anadas tickers: el valor de esta tarea es empezar a contar "
            "el tiempo, no cambiar el sistema. Tests de lo que anadas (serializacion del diario, "
            "rotacion, arranque con estado corrupto) + .venv\\Scripts\\python.exe (poetry run esta "
            "roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "validation-study-full-ensemble",
        "rank": 4,
        "group": "despues",
        "priority": "media",
        "title": "Re-correr el estudio de validacion con el ensemble completo",
        "line": "D", "status": "pendiente", "impact": "medio", "effort": "bajo",
        "evidence": "El informe publicado corrio con n_paths=1: 8 escenarios x 4 configuraciones "
                    "= 32 muestras.",
        "why": "Las conclusiones sobre el corte unico -que es arbitrario y no optimista, y que la "
               "brecha real esta contra la cola- son direccionales pero descansan en poca "
               "muestra. Al integrar CPCV en el pipeline (evolucion #3) conviene re-correrlo con "
               "varios caminos por escenario para que la dispersion medida tenga detras "
               "observaciones suficientes.",
        "prompt": (
            "Proyecto ai-trader (Python). El estudio de validacion "
            "(src/ai_trader/scoring/validation_study.py, informe en "
            "data/validation/report_ai_v2.json, vista 'Validacion' del dashboard) se corrio con "
            "--paths 1: 8 escenarios x 4 configuraciones = 32 muestras. Sus conclusiones "
            "(el corte unico es arbitrario y no optimista; la brecha sistematica esta contra la "
            "COLA, no contra la mediana; la dispersion entre ventanas cambia que configuracion "
            "gana) son direccionales pero tienen poca muestra detras.\n"
            "TAREA: (1) Re-corre el estudio con el ensemble completo -mas escenarios y varios "
            "caminos por escenario- sobre la libreria vigente (ai_v3 si ya existe; si no, ai_v2, "
            "dejandolo dicho en el informe), paralelizando con --workers. Publica el informe "
            "nuevo SIN borrar el anterior, para poder comparar. (2) Anade intervalos de confianza "
            "por bootstrap sobre ESCENARIOS (no sobre muestras: los caminos de un mismo escenario "
            "no son independientes) a las cifras de optimismo, dispersion, acuerdo de rangos y "
            "cambios de decision. (3) Deja escrito en el informe y en el dashboard si alguna "
            "conclusion cambia de signo con la muestra grande. Determinismo + "
            ".venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "pbo-blocks-scenario-aligned",
        "rank": 5,
        "group": "despues",
        "priority": "media",
        "title": "Alinear los bloques del PBO con las fronteras de escenario",
        "line": "A", "status": "pendiente", "impact": "medio", "effort": "bajo",
        "evidence": "La matriz muestras x configuraciones se trocea en bloques CONTIGUOS y las "
                    "muestras van en orden escenario-mayor: los 30 caminos de un escenario pueden "
                    "quedar mitad en train y mitad en test de la CSCV.",
        "why": "Es la misma fuga de arquetipo que `scenario_split` evita cuidadosamente en el "
               "hold-out, colandose dentro del PBO: si el mismo escenario esta a los dos lados, "
               "el PBO sale mas benigno de lo que deberia, porque elegir por train ya sabe algo "
               "del test. Arreglo pequeno, y afecta a una cifra que se publica como garantia.",
        "prompt": (
            "Proyecto ai-trader (Python). `probability_of_backtest_overfitting` "
            "(src/ai_trader/scoring/overfit.py) implementa PBO por CSCV: recibe una matriz "
            "muestras x configuraciones y la parte en `n_blocks` bloques CONTIGUOS "
            "(partition = [range(b*block_size, (b+1)*block_size) ...]). El problema: quien la "
            "llama es run_optimization (scoring/optimize.py), y sus muestras vienen en orden "
            "ESCENARIO-MAYOR (por cada escenario, sus paths). Con 30 caminos por escenario, los "
            "caminos de un mismo escenario pueden quedar mitad en train y mitad en test de la "
            "CSCV: fuga de ARQUETIPO dentro del PBO, exactamente la que "
            "scoring/scenario_split.py evita en el hold-out.\n"
            "TAREA: (1) Anade a `probability_of_backtest_overfitting` un parametro opcional "
            "`groups` (una etiqueta por FILA de la matriz, p.ej. el scenario_id) y construye los "
            "bloques respetando fronteras de grupo -reparto codicioso equilibrado de grupos "
            "enteros entre bloques-, en lugar de trocear por indice. Con groups=None, el "
            "comportamiento actual se conserva bit a bit. (2) Haz que run_optimization pase el "
            "scenario_id de cada muestra (ya lo conoce: _SampleEvaluator._samples genera los "
            "pares (scenario_id, path_index)). (3) Declara en el resultado cuantos grupos y "
            "cuantas filas se usaron y cuantas se descartaron por no encajar en bloques iguales "
            "-nada de recortes silenciosos. (4) Tests: una matriz con estructura de grupo "
            "conocida donde se compruebe que ningun grupo se parte, y el caso degenerado "
            "(menos grupos que bloques). (5) Reporta el PBO antes y despues del cambio sobre la "
            "misma corrida, para que se vea cuanto benigno era. .venv\\Scripts\\python.exe "
            "(poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "report-n-failed-with-reward",
        "rank": 6,
        "group": "despues",
        "priority": "media",
        "title": "Reportar n_failed junto al reward (la penalizacion domina la cola)",
        "line": "A", "status": "pendiente", "impact": "medio", "effort": "bajo",
        "evidence": "FAILURE_PENALTY = -5 frente a un headline que vive en el rango -2..2: una "
                    "configuracion que falla en 2-3 muestras de 100 tiene el CVaR@25% "
                    "practicamente capturado por los -5.",
        "why": "Que fallar duela es correcto y no se discute; lo que no puede ser es que el "
               "numero final no distinga 'esta config tiene mala cola' de 'esta config es "
               "fragil'. Son dos diagnosticos distintos con dos acciones distintas, y hoy "
               "colapsan en el mismo -5. No hay que cambiar la penalizacion, hay que publicar el "
               "recuento al lado.",
        "prompt": (
            "Proyecto ai-trader (Python). `FAILURE_PENALTY = -5.0` "
            "(src/ai_trader/scoring/sample_eval.py) puntua las muestras cuyo backtest falla. El "
            "headline score vive en unidades de Sharpe (tipicamente -2..2), asi que en el "
            "CVaR@25% que agrega las muestras (scoring/aggregate.py::aggregate_reward) unas pocas "
            "muestras fallidas COPAN la cola: una config que falla en 2-3 de 100 tiene su reward "
            "determinado por los -5, no por su comportamiento. La eleccion es defendible -fallar "
            "debe doler- pero el numero resultante no distingue 'cola mala' de 'config fragil'.\n"
            "TAREA: (1) Anade a RewardStats (scoring/aggregate.py) el recuento y la fraccion de "
            "muestras fallidas, propagando el flag `failed` que SampleEvaluation ya lleva; "
            "aggregate_reward debe poder recibir esa informacion sin cambiar su firma escalar "
            "actual para quien no la tenga. (2) Reporta n_failed / failure_rate junto al reward "
            "en run_optimization, en el ranking del dashboard y en el estudio de validacion. "
            "(3) Emite un aviso explicito cuando failure_rate >= alpha del CVaR: en ese caso la "
            "cola esta hecha SOLO de fallos y el reward no esta midiendo rendimiento. (4) NO "
            "cambies el valor de la penalizacion ni el estadistico de agregacion. Tests + "
            ".venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "dsr-independent-trials-caveat",
        "rank": 7,
        "group": "despues",
        "priority": "baja",
        "title": "Declarar que el DSR asume intentos independientes y el CEM no los produce",
        "line": "A", "status": "pendiente", "impact": "bajo", "effort": "bajo",
        "evidence": "n_trials = 192 configuraciones evaluadas por el CEM, tratadas como intentos "
                    "independientes en el maximo esperado bajo la nula.",
        "why": "Las generaciones tardias del CEM se concentran alrededor de la elite, asi que los "
               "intentos efectivos son menos que los contados y el DSR SOBRE-deflacta. El sentido "
               "del error es el prudente, pero mientras no este escrito alguien puede leer el DSR "
               "como una probabilidad calibrada, que no lo es. Una linea de docstring y una nota "
               "en la metodologia.",
        "prompt": (
            "Proyecto ai-trader (Python). `deflated_sharpe_ratio` "
            "(src/ai_trader/scoring/overfit.py) deflacta el Sharpe del ganador usando "
            "`_expected_max_sharpe(std_trials, n_trials)`, que es el maximo esperado bajo la nula "
            "asumiendo n_trials intentos INDEPENDIENTES. Quien lo llama es run_optimization con "
            "n_trials = numero de configuraciones que el CEM evaluo (del orden de 192). Pero el "
            "CEM no produce intentos independientes: las generaciones tardias se concentran "
            "alrededor de la elite, asi que los intentos EFECTIVOS son bastantes menos y el DSR "
            "sobre-deflacta.\n"
            "TAREA: (1) Documenta el supuesto y su direccion en el docstring de "
            "deflated_sharpe_ratio: el error va del lado prudente (el DSR sale mas bajo de lo que "
            "corresponderia), pero por eso mismo el DSR NO debe leerse como una probabilidad "
            "calibrada. (2) Refleja la misma advertencia en la vista del dashboard donde se "
            "publica el DSR y en docs/ (metodologia). (3) OPCIONAL y solo si sale limpio: reporta "
            "junto al DSR una estimacion INFORMATIVA de intentos efectivos (p.ej. a partir de la "
            "correlacion media entre las series de scores de las configuraciones probadas, "
            "n_eff = n / (1 + (n-1)*rho_medio)), etiquetada claramente como informativa y sin "
            "usarla para deflactar. .venv\\Scripts\\python.exe (poetry run esta roto) + ruff. "
            "Regenera dashboard y docs."
        ),
    },
    {
        "id": "fidelity-rank-corr-ordering",
        "rank": 8,
        "group": "despues",
        "priority": "media",
        "title": "Ordenacion de colas y clustering entre activos: el eje que ai_v3 no arreglo",
        "line": "B", "status": "pendiente", "impact": "medio", "effort": "medio",
        "evidence": "ai_v3 cumple los umbrales de NIVEL y COBERTURA (98,3% de cobertura media, "
                    "curtosis 3,40 vs 4,19 real, clustering 0,196 vs 0,190) pero su rank_corr "
                    "medio es -0,23: clustering -0,74 y curtosis -0,18. En ai_v2 el clustering "
                    "ordenaba +0,32. El nivel se arreglo y la ordenacion empeoro.",
        "why": "El generador ya produce la magnitud correcta de cola y de agrupamiento, pero no "
               "sabe QUE activo tiene mas: en el mercado real son los mas ruidosos (DOGE 17,8 de "
               "curtosis y 0,31 de clustering; XRP 13,9) y en el sintetico salen de los que menos "
               "ruido propio tienen. La causa es identificable en el motor: el componente "
               "idiosincratico pasa por `_ar1_idio`, un AR(1) con phi negativo en las fases de "
               "rango, que BLANQUEA la estructura de |r| justo en los activos donde ese "
               "componente pesa mas (idio_vol alto). Importa para la seleccion cross-sectional: "
               "una politica que elija activos por su regimen de volatilidad esta aprendiendo un "
               "ordenamiento invertido respecto al mercado. No es critico -las tres lecturas se "
               "publican por separado y esta se declara- pero es el ultimo hueco medido del "
               "sustrato.",
        "prompt": (
            "Proyecto ai-trader (Python). El estudio de fidelidad "
            "(src/ai_trader/synthetic/fidelity_study.py, informes en data/fidelity/) ya acepta la "
            "libreria ai_v3: cobertura media 98,3% y las medianas reales de curtosis, clustering "
            "y exceedances dentro de la banda sintetica. Queda UN eje medido y no arreglado: la "
            "ORDENACION entre activos. rank_corr (Spearman de la seccion cruzada real contra la "
            "sintetica) sale -0,74 en clustering (ac_abs1) y -0,18 en curtosis; el medio de las "
            "metricas objetivo es -0,23. Esta INVERTIDO, no solo flojo.\n"
            "\n"
            "HIPOTESIS DE CAUSA (verificala antes de tocar nada, no la des por buena): en el "
            "mercado real los activos con mas ruido propio son los que mas cola y mas "
            "agrupamiento tienen (DOGE: idio_vol 0,055 en el universo, curtosis real 17,8, "
            "clustering 0,308; XRP: 13,9 y 0,245; frente a BTC: 3,9 y 0,152). En el motor "
            "(src/ai_trader/synthetic/engine.py) el componente idiosincratico se dibuja con colas "
            "y GARCH pero DESPUES pasa por `_ar1_idio`, un AR(1) con phi por fase que en las "
            "fases de rango es negativo (-0,30, ver `_idio_ar_for` en synthetic/retrofit.py). Un "
            "AR(1) mezcla dias adyacentes: baja la curtosis y BLANQUEA la autocorrelacion de |r| "
            "de ese componente, y lo hace MAS en los activos donde el idio pesa mas. Eso "
            "invertiria la ordenacion. Compruebalo midiendo, con el mismo `series_facts` del "
            "harness, curtosis y ac_abs1 de un solo activo variando idio_vol con y sin idio_ar.\n"
            "\n"
            "TAREA: subir el rank_corr de ac_abs1 y excess_kurtosis a >= +0,3 SIN romper lo que ya "
            "cumple. Ideas, en orden de preferencia (elige con la medicion, no a priori): (a) que "
            "el AR(1) idiosincratico actue sobre el SIGNO/direccion pero no destruya el "
            "agrupamiento -p.ej. aplicando el GARCH DESPUES del AR(1) en vez de antes, que "
            "reordena las dos operaciones sin cambiar la varianza-; (b) dar al idio su propia "
            "persistencia/cola por activo en vez de compartir el timeline de fase; (c) escalar "
            "tail_dof o vol_persistence del idio con el idio_vol del activo. Mide cada una con el "
            "harness antes de quedarte con ninguna.\n"
            "\n"
            "REGLAS DEL SITIO, no negociables: (1) La aceptacion actual NO puede empeorar: "
            "'.venv\\Scripts\\python.exe -m ai_trader.synthetic.fidelity_study --library ai_v4 "
            "--offline' tiene que seguir devolviendo 0 y la cobertura media quedarse en >= 95%, "
            "con el nivel de volatilidad en ratio 0,9-1,1. (2) Neutralidad EXACTA de los defaults: "
            "tests/test_synthetic.py::TestEngineByteIdentity congela con un hash que ai_v1 y ai_v2 "
            "se regeneran byte a byte desde sus spec.json; si se pone rojo, el cambio esta mal "
            "hecho, no el test. (3) Libreria nueva (ai_v4) derivada de ai_v1 con "
            "`SyntheticDataService.derive_library`, sin tocar las anteriores, y su informe "
            "publicado junto a los otros dos. (4) Variance-matching: cualquier reordenacion de "
            "AR(1)/GARCH tiene que dejar la volatilidad total donde estaba. Tests + "
            ".venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "rl-full-run",
        "rank": 9,
        "group": "despues",
        "priority": "alta",
        "title": "Optimizacion CEM completa, ya con el juez validado",
        "line": "RL", "status": "bloqueada", "impact": "alto", "effort": "medio",
        "depends": 2,
        "evidence": "El harness CEM esta listo, pero cada backtest cuesta ~60 s con 35 activos y "
                    "hoy apuntaria al juez del corte unico.",
        "why": "Correr la optimizacion a escala solo tiene sentido cuando el objetivo que escala "
               "el CEM es el bueno: primero el sustrato (#1) y el juez en dos etapas (#3), "
               "despues esto. Hacerlo antes es gastar computo caro produciendo un ganador que "
               "habria que tirar.",
        "prompt": (
            "Proyecto ai-trader (Python). El harness de optimizacion por Cross-Entropy Method "
            "(src/ai_trader/scoring/optimize.py, run_optimization) esta listo, pero cada backtest "
            "cuesta ~60 s con los 35 activos, asi que una corrida completa sobre 900 muestras "
            "(30 escenarios x 30 paths) necesita subsampleo o paralelizacion.\n"
            "REQUISITO PREVIO: esta tarea solo debe ejecutarse cuando esten hechas (a) la "
            "correccion de fidelidad del generador con su libreria ai_v3 y (b) el juez en dos "
            "etapas con CPCV dentro de run_optimization. Si alguna falta, paralo y dilo: "
            "optimizar contra el juez viejo produce un ganador que habria que descartar.\n"
            "TAREA: ejecuta y consolida la optimizacion CEM de las primitivas disponibles "
            "(crypto_momentum y mean_reversion) sobre la libreria vigente (ai_v3). La metrica de "
            "cabecera honesta (Sharpe - lambda*turnover - kappa*maxDD, agregada por CVaR@25%, con "
            "gate de baselines y descuento DSR/PBO) ya esta implementada, asi que "
            "run_optimization devuelve tambien el veredicto del gate y el sobreajuste por "
            "multiples pruebas: reportalos junto al recuento de muestras fallidas. Optimiza el "
            "rendimiento del backtest o paraleliza la evaluacion de muestras para que la corrida "
            "sea tratable, y declara por log cualquier subsampleo. Guarda los mejores parametros "
            "por primitiva con su distribucion train/validation y vuelca los resultados al "
            "dashboard (seccion Ranking). Determinismo + tests + .venv\\Scripts\\python.exe "
            "(poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "new-crypto-strategies",
        "rank": 10,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Nuevas estrategias cripto (deliberadamente NO priorizada)",
        "line": "Estrategias", "status": "pendiente", "impact": "medio", "effort": "alto",
        "depends": 2,
        "evidence": "Solo 6 de 32 filas aprueban el gate bajo CPCV. Eso admite dos lecturas -las "
                    "estrategias son flojas, o el juez es ruidoso- y hasta saber cual, anadir "
                    "candidatos multiplica el problema de multiples pruebas.",
        "why": "Aparece en la lista porque es trabajo real y acordado, pero NO se aborda todavia, "
               "y la razon es una asimetria de coste, no una preferencia: una estrategia anadida "
               "hoy se re-evalua GRATIS cuando el juez mejore; un juez malo contamina todo lo que "
               "puntue hoy. Las estrategias son la cosecha; el juez es el suelo. Ademas cada "
               "candidato nuevo sube el n_trials del DSR, o sea que anadir sin ganar edge "
               "empeora activamente el veredicto de todo lo demas.",
        "prompt": (
            "Proyecto ai-trader (Python). ANTES DE EMPEZAR: comprueba que el juez esta arreglado "
            "-CPCV en dos etapas dentro de run_optimization y libreria ai_v3 con la fidelidad "
            "aceptada. Si no lo esta, no anadas estrategias: solo 6 de 32 filas aprueban el gate "
            "bajo CPCV, y hasta saber si eso es culpa de las estrategias o del juez, cada "
            "candidato nuevo multiplica el problema de multiples pruebas sobre un juez en el que "
            "aun no se confia (y sube el n_trials que deflacta el DSR de todos los demas).\n"
            "\n"
            "TAREA (cuando toque): amplia el catalogo de primitivas CRIPTO. Hoy hay dos "
            "operables (src/ai_trader/strategies/: crypto_momentum y mean_reversion; "
            "polymarket_threshold no entra en el backtest). Candidatas con edge plausible en el "
            "timescale diario de pares cripto de segunda fila, que es donde el sistema tiene "
            "alguna probabilidad: (a) momentum TRANSVERSAL (rankear el universo y operar los "
            "extremos, no cada simbolo contra si mismo -es la unica que aprovecha de verdad las "
            "features cross-sectional de observation/); (b) breakout con filtro de regimen, "
            "usando las features de regimen ya existentes; (c) volatility targeting sobre una "
            "senal existente, que cambia el perfil de riesgo sin cambiar la senal; (d) pares "
            "cointegrados dentro del universo.\n"
            "\n"
            "REQUISITOS por estrategia: clase con config tipada en src/ai_trader/strategies/, "
            "registro en strategies/registry.py::STRATEGY_REGISTRY, ParamSpace en el modulo de "
            "espacios del scoring para que el CEM pueda optimizarla, tests unitarios de la senal "
            "sobre casos construidos a mano (no solo 'no revienta'), y coste de ejecucion "
            "pagado por el modelo de microestructura real -nada de suponer fills gratis. "
            "Evaluacion OBLIGATORIA con el juez validado: CPCV en dos etapas, gate contra los "
            "tres baselines pasivos con los mismos costes, y DSR/PBO reportados con el n_trials "
            "actualizado. Una estrategia que no bate el gate no entra en el catalogo operable: "
            "se publica su resultado negativo y se archiva. Determinismo + "
            ".venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "weights-recalibrate-power",
        "rank": 11,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Re-medir lambda y kappa con los costes nuevos y mas potencia estadistica",
        "line": "A/C", "status": "pendiente", "impact": "bajo", "effort": "medio",
        "evidence": "480 backtests ya medidos: la superficie (lambda, kappa) sale PLANA y la "
                    "eleccion de configuracion no cambia. Pero se midio con un unico corte 70/30, "
                    "un camino por escenario y el slippage PLANO de 5 bps que ya no existe.",
        "why": "La conclusion publicada -penalizar no estabiliza, y los costes ya dentro del "
               "Sharpe equivalen a un lambda ~ 6,3, muy por encima del 0,25 elegido- se refuerza "
               "con el modelo de costes nuevo, que cobra mas friccion. Es decir: re-medirlo casi "
               "seguro no mueve nada. Por eso es de impacto bajo y va aqui, no arriba.",
        "prompt": (
            "Proyecto ai-trader (Python). Los pesos del headline score "
            "(src/ai_trader/backtest/metrics.py::DEFAULT_HEADLINE_WEIGHTS, hoy lambda=0.25, "
            "kappa=0.0) estan calibrados con evidencia: estudio en "
            "src/ai_trader/scoring/weight_study.py, informe en "
            "data/calibration/report_ai_v2.json, 480 backtests. Resultado: la superficie "
            "(lambda, kappa) es PLANA en rank IC y en gap train-validation, y la auditoria de "
            "costes dice que el slippage ya cobrado dentro del Sharpe equivale a un lambda "
            "implicito ~ 6,3. Dos motivos para repetirlo, ninguno urgente:\n"
            "  (a) se midio con UN corte temporal 70/30, un camino por escenario y 16 "
            "configuraciones: sirve para descartar que los pesos importen mucho, no para afinar "
            "decimales;\n"
            "  (b) se midio cuando el motor cobraba un slippage PLANO de 5 bps, y hoy la "
            "ejecucion usa el modelo de microestructura (medio spread por simbolo + volatilidad "
            "reciente + impacto por raiz cuadrada de la participacion, con techo de capacidad).\n"
            "TAREA: (1) Repite el barrido con los costes actuales y con validacion multiventana "
            "(scoring/multiwindow.py, walk-forward o CPCV con purga y embargo) y varios caminos "
            "por escenario, reutilizando el cacheo de componentes crudos que ya existe (los "
            "componentes no dependen de los pesos, asi que la rejilla sale gratis). (2) Anade "
            "intervalos de confianza por bootstrap sobre ESCENARIOS al rank IC y al gap, para "
            "poder afirmar o descartar diferencias entre puntos de la rejilla en vez de leer "
            "ruido. (3) Corrige turnover_cost_audit en scoring/weight_calibration.py: hoy deriva "
            "el lambda implicito de un cost_rate PLANO (fee_rate + slippage_bps); hazlo con el "
            "slippage REALMENTE cobrado por operacion (ExecutionResult.slippage_bps), que ya no "
            "es una constante. (4) Publica el informe nuevo sin borrar el anterior. (5) Solo si "
            "la evidencia nueva mueve el optimo FUERA del error, actualiza "
            "DEFAULT_HEADLINE_WEIGHTS y el test que los congela "
            "(tests/test_backtest_metrics.py::TestDefaultWeightsAreCalibrated). Determinismo + "
            ".venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "designer-model-in-manifest",
        "rank": 12,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Anotar el modelo de IA en el manifiesto de cada libreria",
        "line": "E", "status": "pendiente", "impact": "bajo", "effort": "bajo",
        "evidence": "El manifiesto guarda la CLASE del disenador ('ClaudeScenarioDesigner'), no "
                    "el identificador del modelo ni la fecha del diseno.",
        "why": "El diseno con IA no es reproducible y ya no puede serlo (los modelos actuales "
               "retiraron los parametros de muestreo), asi que la unica trazabilidad posible es "
               "registrar CON QUE se genero. Hoy dos librerias disenadas con modelos distintos "
               "son indistinguibles en disco.",
        "prompt": (
            "Proyecto ai-trader (Python). El disenador con IA "
            "(src/ai_trader/synthetic/designer.py, ClaudeScenarioDesigner) produce escenarios NO "
            "reproducibles por diseno: los modelos actuales retiraron temperature/top_p/top_k, "
            "asi que no hay ninguna palanca de determinismo. La mitigacion vigente es guardar el "
            "spec.json. Falta la trazabilidad: SyntheticDataService "
            "(src/ai_trader/synthetic/service.py) escribe en el manifiesto "
            "`designer=type(self.designer).__name__`, es decir solo la clase.\n"
            "TAREA: haz que el manifiesto registre tambien el identificador del modelo y la fecha "
            "de generacion del diseno, sin romper las librerias ya publicadas (ai_v1, ai_v2, "
            "ai_v3): los manifiestos antiguos sin ese campo deben seguir cargando. Sugerencia: un "
            "metodo/propiedad opcional en el protocolo ScenarioDesigner (p.ej. `describe()`) que "
            "el servicio consulte con getattr y que TemplateScenarioDesigner tambien implemente, "
            "y un campo nuevo en SyntheticManifest con valor por defecto. Expon el dato en la "
            "vista 'Datos sinteticos' del dashboard. Tests de compatibilidad hacia atras (cargar "
            "un manifiesto sin el campo) + .venv\\Scripts\\python.exe (poetry run esta roto) + "
            "ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "equities-parked",
        "rank": 13,
        "group": "segundo-plano",
        "priority": "aparcada",
        "title": "Renta variable: aparcada a proposito (no se activa la clase de activo)",
        "line": "Universo", "status": "aparcada", "impact": "bajo", "effort": "alto",
        "evidence": "Cero estrategias de renta variable en el repo (hay proveedor, no estrategia). "
                    "Cero datos reales detras de la pata de equity del generador. Y el universo "
                    "de 20 megacaps esta elegido en 2024-26 entre las que sobrevivieron.",
        "why": "La decision es SOLO CRIPTO, y no por poco. Toda la evidencia empirica del repo es "
               "cripto: la fidelidad se midio contra Binance y la calibracion y la validacion "
               "corrieron sobre universos con calendario 365. Activar renta variable seria "
               "construir estrategia + validar el proveedor Alpaca (¿ajusta splits y "
               "dividendos?) + verificar el calendario 252 de punta a punta + resolver un sesgo "
               "de supervivencia estructural (constituyentes point-in-time), todo antes de la "
               "primera senal. Y el edge plausible no esta ahi: momentum diario sobre AAPL o SPY "
               "compite contra el segmento mas eficiente del planeta. Lo que NO se hace es quitar "
               "los stocks del universo SINTETICO: alli GLD, TLT y UUP son lo que hace que los "
               "escenarios de tipos y de dolar signifiquen algo para cripto via los factores "
               "compartidos. Se genera con los 35, se puntua y se opera solo cripto.",
        "prompt": (
            "Proyecto ai-trader (Python). NO EJECUTAR TODAVIA: esta tarea esta aparcada a "
            "proposito hasta que el bucle cripto (sintetico fiel -> transferencia contra el real "
            "-> paper trading con meses de historico) este cerrado y medido. Se deja escrita para "
            "que el dia que se retome no haya que redescubrir el alcance.\n"
            "CONTEXTO: el repo tiene proveedor de renta variable (Alpaca) pero NINGUNA estrategia "
            "de renta variable; el universo operable (config/default.toml) es de 24 pares cripto; "
            "el universo SINTETICO si incluye acciones y ETFs (GLD, TLT, UUP y 20 megacaps) y eso "
            "se mantiene, porque son la estructura de correlacion que da sentido a los escenarios "
            "de tipos y de dolar. La regla vigente: se genera con los 35 activos, se puntua y se "
            "opera solo cripto.\n"
            "ALCANCE cuando se retome, en este orden: (1) validar el proveedor Alpaca con datos "
            "reales -¿los precios vienen ajustados por splits y dividendos?, ¿que pasa con los "
            "dias sin sesion?- porque un backtest sobre precios no ajustados es basura silenciosa; "
            "(2) verificar el calendario 252 de punta a punta (periods_per_year_for_symbols, "
            "anualizacion del Sharpe, ventanas de los folds y de los baselines) en un universo "
            "MIXTO, que es el caso que hoy no se ejercita; (3) resolver el sesgo de supervivencia "
            "de la lista de 20 megacaps -elegidas en 2024-26 precisamente porque sobrevivieron y "
            "ganaron- con constituyentes point-in-time, que es un proyecto en si mismo y sin el "
            "cual cualquier backtest nace inflado; (4) solo entonces, una estrategia de renta "
            "variable y su evaluacion con el mismo juez que las cripto. Antes de nada de esto, "
            "re-lee la evidencia de fidelidad: la pata de equity del generador no tiene ni un "
            "dato real detras, asi que sus betas, su idio_vol y sus spreads son plausibles pero "
            "NO contrastados, y contrastarlos es parte del trabajo."
        ),
    },
    {
        "id": "polymarket-parked",
        "rank": 14,
        "group": "segundo-plano",
        "priority": "aparcada",
        "title": "Polymarket en el backtest: aparcado hasta tener historico propio",
        "line": "Universo", "status": "aparcada", "impact": "bajo", "effort": "alto",
        "evidence": "Sin OHLCV historico de mercados de prediccion. La estrategia "
                    "polymarket_threshold existe y opera en papel, pero no entra en ningun "
                    "backtest.",
        "why": "No es un olvido ni una limitacion del codigo: no hay historico que backtestear, y "
               "comprarlo no es barato. La via realista es la que abre la evolucion #4: con el "
               "paper trading corriendo, el sistema empieza a guardar midpoints por ciclo, y en "
               "unos meses habra una serie propia. Retomar esto antes de tener esa serie es "
               "construir sobre nada.",
        "prompt": (
            "Proyecto ai-trader (Python). NO EJECUTAR TODAVIA: depende de tener meses de "
            "midpoints registrados por el paper trading en vivo (evolucion 'Poner el paper "
            "trading a correr en vivo'). Se deja escrita para no perder el alcance.\n"
            "CONTEXTO: existe la estrategia polymarket_threshold "
            "(src/ai_trader/strategies/polymarket_threshold.py) y su ejecucion en papel "
            "(execution/polymarket_paper.py), pero los mercados de prediccion NO entran en el "
            "backtest porque no hay OHLCV historico: un mercado de Polymarket no publica velas "
            "diarias comparables, y su vida util es corta y con final absorbente (resuelve a 0 o "
            "a 1).\n"
            "ALCANCE cuando se retome: (1) construir un almacen de series de midpoint por mercado "
            "a partir del diario de ciclos del paper trading, con su resolucion final cuando la "
            "haya; (2) decidir la semantica de 'barra' para un mercado que resuelve -no es un "
            "precio que sigue, es una probabilidad que colapsa- y como se anualiza cualquier "
            "metrica sobre eso (el Sharpe anualizado no significa lo mismo en un instrumento con "
            "vencimiento); (3) definir sus costes reales (spread del libro, no el fee plano) y su "
            "capacidad; (4) solo entonces, un motor de backtest para esta clase de activo y su "
            "evaluacion con el mismo juez. Mientras tanto, lo unico que hay que hacer es "
            "REGISTRAR: sin serie propia, no hay nada que medir."
        ),
    },
]


if __name__ == "__main__":
    build()
