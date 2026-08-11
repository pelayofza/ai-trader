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
from ai_trader.backtest.session_study import (
    SESSIONS_REPORT,
    US_KEY,
    load_sessions_report,
)
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
from ai_trader.scoring.activity_study import (
    activity_report_path,
    load_activity_report,
)
from ai_trader.scoring.sample_eval import evaluate_baselines, evaluate_sample_detailed
from ai_trader.scoring.transfer_study import (
    DEFAULT_LIBRARY_ID as TRANSFER_LIBRARY,
    load_transfer_report,
    transfer_report_path,
)
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


def collect_transfer() -> dict | None:
    """¿Ordena el mundo sintetico las estrategias como el real? (data/transfer).

    Es la pregunta que la fidelidad NO responde: un generador puede clavar las colas y
    ordenar al reves. Se LEE del informe publicado; no se recalcula. Son 208 unidades de
    15 ventanas de backtest real cada una y el dashboard tiene que seguir siendo
    regenerable en minutos."""
    report = load_transfer_report(ROOT / transfer_report_path(TRANSFER_LIBRARY))
    if not report:
        logger.warning("Sin informe de transferencia: el panel de ranking saldra vacio")
        return None

    plan = report["plan"]
    configs = report["configs"]
    return {
        "library": plan["library_id"],
        "is_fallback": plan["library_is_fallback"],
        "symbols": plan["symbols"],
        "omitted": plan["real"]["symbols_omitted"],
        "library_omitted": plan["synthetic"]["library_symbols_omitted"],
        "real_window": plan["real"]["window"],
        "sub_windows": plan["real"]["sub_windows"],
        "head_discarded_days": plan["real"]["head_discarded_days"],
        "min_history_days": plan["real"]["min_history_days"],
        "n_samples": plan["synthetic"]["n_samples"],
        "validation": plan["validation"],
        "grid": plan["grid"],
        "n_configs": len(configs),
        "dropped": report["configs_dropped"],
        # El scatter de rangos: un punto por configuracion. Es la vista que hace visible
        # de un golpe si los dos mundos ordenan igual (puntos en la diagonal) o no.
        "points": [
            {
                "config_id": c["config_id"],
                "family": c["family"],
                "rank_real": c["rank_real"],
                "rank_synthetic": c["rank_synthetic"],
                "delta": c["rank_delta"],
                "reward_real": c["reward_real"],
                "reward_synthetic": c["reward_synthetic"],
                "trades_real": c["trades_per_fold"]["real"],
                "trades_synthetic": c["trades_per_fold"]["synthetic"],
                "active": c["active"],
                "rankable_real": c.get("rankable_real"),
                "approved_real": c["real"]["approved_pooled"],
                "approved_synthetic": c["synthetic"]["approved_pooled"],
                "n_real": c["real"]["n"],
                "n_synthetic": c["synthetic"]["n"],
            }
            for c in sorted(configs, key=lambda c: c["rank_real"])
        ],
        "transfer": report["transfer"],
        "eligibility": report.get("eligibility"),
        "verdict": report["verdict"],
        "caveats": report["caveats"],
        "baselines": report["baselines"],
        "leakage": report["leakage"],
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


def collect_sessions() -> dict | None:
    """Descomposicion por sesion horaria de la formacion de precio (data/sessions).

    Se LEE del informe publicado; no se recalcula. El estudio descarga seis anos de barras
    1H de 24 pares (algo mas de un millon de velas) y el dashboard tiene que seguir siendo
    regenerable en minutos. El informe ya trae los veredictos leidos, asi que aqui no se
    interpreta nada: se reempaqueta para el render."""
    report = load_sessions_report(ROOT / SESSIONS_REPORT)
    if not report:
        logger.warning("Sin informe de sesiones: el panel de la ventana ciega saldra vacio")
        return None

    plan = report["plan"]
    trend = report["trend"]
    sessions = report["sessions"]
    overall = report["overall"]["sessions"]

    # El pico de cada reparto, ya resuelto: el render no deberia estar buscando maximos.
    def leader(field: str) -> dict:
        key = max(overall, key=lambda k: overall[k][field] or 0.0)
        return {"key": key, "label": _session_label(sessions, key), "value": overall[key][field]}

    return {
        "window": plan["window"],
        "exchange": plan["exchange"],
        "timeframe": plan["timeframe"],
        "thresholds": plan["thresholds"],
        "latency_hours": plan["latency_hours"],
        "min_days_per_year": plan["min_days_per_year"],
        "sessions": sessions,
        "overall": report["overall"],
        "cohort_overall": report["cohort_overall"],
        "cohort": report["cohort"],
        "symbols": report["symbols"],
        "omitted": report["omitted"],
        "by_symbol_year": report["by_symbol_year"],
        "gap": report["gap"],
        "latency": report["latency"],
        "trend": trend,
        "verdicts": report["verdicts"],
        "caveats": report["caveats"],
        "leader_variance": leader("variance"),
        "leader_sets_low": leader("sets_low"),
        "us": overall[US_KEY],
        "us_key": US_KEY,
        "generated_at": report["generated_at"][:10],
    }


def _session_label(sessions: list[dict], key: str) -> str:
    return next((s["label"] for s in sessions if s["key"] == key), key)


def collect_activity() -> dict | None:
    """El suelo de actividad y la evidencia con la que se eligio (data/activity).

    Se LEE del informe publicado; no se recalcula. El estudio se apoya en las 208 unidades
    del estudio de transferencia (15 ventanas de backtest real cada una) y el dashboard
    tiene que seguir siendo regenerable en minutos. El informe ya trae la regla aplicada y
    el umbral elegido: aqui no se decide nada, se reempaqueta para el render."""
    report = load_activity_report(ROOT / activity_report_path(TRANSFER_LIBRARY))
    if not report:
        logger.warning("Sin informe de actividad: el panel del suelo saldra vacio")
        return None

    real = report["gate"]["real"]
    return {
        "library": report["source"]["library_id"],
        "source": report["source"],
        "floor": report["floor"],
        "mechanism": report["mechanism"],
        "decision": report["decision"],
        "band": report["band"],
        "sweep": report["sweep"],
        "reproducibility": report["reproducibility"],
        "gate": report["gate"],
        "n_lost": real["n_lost"],
        "lost": real["lost_detail"],
        # Una fila por configuracion con las dos cifras que la hacen (o no) rankeable.
        "rows": [
            {
                "config_id": c["config_id"],
                "trades_per_window": c["real"]["trades_per_window"],
                "median_trades_per_window": c["real"]["median_trades_per_window"],
                "zero_window_pct": c["real"]["zero_window_pct"],
                "reward": c["real"]["reward"],
                "rankable": c["real"]["rankable"],
                "trades_per_window_synthetic": c["synthetic"]["trades_per_window"],
                "zero_window_pct_synthetic": c["synthetic"]["zero_window_pct"],
                "rankable_synthetic": c["synthetic"]["rankable"],
            }
            for c in sorted(report["configs"], key=lambda c: -c["real"]["reward"])
        ],
        "generated_at": report["generated_at"][:10],
    }


def _etf_dispersion(store) -> dict:
    """La distribucion de `etf_issuer_dispersion`, que es lo que justifica bajar al emisor.

    Un solo numero no lo demuestra: la MEDIANA cerca de 1 dice que casi todos los dias son
    flujo neto, y la cola alta dice que existe un pun~ado de dias de rotacion pura en los
    que el agregado marca cero y el mercado se movio entero. Se mide aqui porque es barato
    (662 registros) y porque una afirmacion asi no puede ir escrita a mano en la plantilla.
    """
    from ai_trader.signals.adapters.etf_flows import TftcEtfFlows
    from ai_trader.signals.catalog import get_source

    records = store.read("etf_flows")
    if not records:
        return {}
    frame = TftcEtfFlows(get_source("etf_flows")).daily_from_raw(records)
    column = frame["etf_issuer_dispersion"].dropna()
    if column.empty:
        return {}
    return {
        "days": int(len(frame)),
        "median": round(float(column.median()), 2),
        "p95": round(float(column.quantile(0.95)), 1),
        "rotation_days": int((column > 3.0).sum()),
    }


def collect_signals() -> dict:
    """
    La plataforma de ingesta de senales: catalogo, mapeo de entidades, archivo crudo y
    —lo que cambia la lectura de todo lo demas— la PROFUNDIDAD MEDIDA de cada fuente.

    Todo se lee de disco y del registro de mediciones (`data/signals/history_depth.json`);
    nada de esto toca red al generar el dashboard. La cifra que hay que mirar no es cuantas
    fuentes hay declaradas, sino cuantas tienen `history_from` MEDIDO: esas son las unicas
    que pueden entrar en un backtest, y el resto solo existen hacia adelante.
    """
    from ai_trader.signals.audit import audit_archive, audit_entities
    from ai_trader.signals.capture import (
        CAPTURE_REPORT,
        connect_adapters,
        entities_for,
        load_capture_report,
    )
    from ai_trader.signals.catalog import CATALOG, catalog_summary
    from ai_trader.signals.depth import DEPTH_LEDGER, load_ledger
    from ai_trader.signals.normalize import normalization_spec
    from ai_trader.signals.source import connected_keys
    from ai_trader.signals.store import SignalStore

    config = load_config(ROOT / "config" / "default.toml")
    universe = list(config.runner.symbols)

    connect_adapters()  # sin esto, "conectadas" seria 0 por no haber importado nada
    entities = audit_entities(universe)
    archive = audit_archive(SignalStore(ROOT / "data" / "signals_raw"))
    capture_report = load_capture_report(ROOT / CAPTURE_REPORT)
    connected = set(connected_keys())

    ledger = load_ledger(ROOT / DEPTH_LEDGER) or {}
    depth_by_key = {row["source_key"]: row for row in ledger.get("sources") or []}
    archive_by_key = {row.source_key: row for row in archive.sources}
    etf = _etf_dispersion(SignalStore(ROOT / "data" / "signals_raw"))

    return {
        "summary": {
            **catalog_summary(),
            "n_connected": len(connected),
            "n_measured": sum(1 for r in depth_by_key.values() if r.get("first_day")),
        },
        "universe": universe,
        "normalization": normalization_spec(),
        "etf_dispersion": etf,
        "depth_measured_at": (ledger.get("generated_at") or "")[:10] or None,
        "sources": [
            {
                "key": s.key,
                "title": s.title,
                "tier": s.tier,
                "scope": s.scope,
                "cadence": s.cadence,
                "entity_kind": s.entity_kind.value,
                "pit": s.pit,
                "history_from": s.history_from.isoformat() if s.history_from else None,
                "backtestable": s.backtestable,
                "license": s.license,
                "auth_env": s.auth_env,
                "features": list(s.feature_names),
                "n_entities": len(entities_for(s, universe)),
                "connected": s.key in connected,
                "notes": s.notes,
                # Lo MEDIDO, al lado de lo declarado: sin las dos cosas juntas no se ve la
                # diferencia entre "no hay historia" y "nadie la ha comprobado".
                "measured_from": (depth_by_key.get(s.key) or {}).get("first_day"),
                "measured_to": (depth_by_key.get(s.key) or {}).get("last_day"),
                "measured_days": (depth_by_key.get(s.key) or {}).get("days", 0),
                "measured_entities": (depth_by_key.get(s.key) or {}).get("n_entities", 0),
                "measure_method": (depth_by_key.get(s.key) or {}).get("method"),
                "measure_error": (depth_by_key.get(s.key) or {}).get("error"),
                "archived_records": (
                    archive_by_key[s.key].records if s.key in archive_by_key else 0
                ),
            }
            for s in CATALOG
        ],
        "entities": {
            "n_symbols": entities.n_symbols,
            "coverage_pct": round(entities.coverage_pct, 2),
            "by_source": entities.by_source,
            "by_kind": entities.by_kind,
            "unmapped": list(entities.unmapped),
            "collisions": {k: list(v) for k, v in entities.collisions.items()},
            "n_entities": len({r.key for r in entities.refs if r.resolved}),
        },
        "archive": {
            "root": archive.root,
            "records": archive.records,
            "n_with_archive": archive.n_with_archive,
        },
        "capture": None if capture_report is None else {
            "finished_at": capture_report["finished_at"][:10],
            "n_connected": capture_report["n_connected"],
            "n_pending": capture_report["n_pending"],
            "records": capture_report["records"],
        },
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
    logger.info("Transferencia de ranking real vs sintetico (informe publicado)...")
    transfer = collect_transfer()
    logger.info("Descomposicion por sesion horaria (informe publicado)...")
    logger.info("Suelo de actividad del ranking (informe publicado)...")
    logger.info("Catalogo de senales externas y auditoria de cobertura...")
    signals_platform = collect_signals()
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
        "transfer": transfer,
        "strategies": strategies,
        "signals": signals,
        "costs": costs,
        "ranking": ranking,
        "calibration": collect_calibration(),
        "validation": collect_validation(),
        "sessions": collect_sessions(),
        "activity": collect_activity(),
        "signals_platform": signals_platform,
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
# ACTUALIZACION 2026-08-11: el estudio de transferencia cerro el bucle que ocupaba el
# puesto 1 y devolvio el resultado malo -- la libreria sintetica pasa todos los umbrales de
# fidelidad y aun asi NO ordena las estrategias como el mercado (rho = -0,04; -0,67 entre
# las que operan de verdad).
#
# Lo que ese resultado NO dice, y por eso la lista se reordena hacia otro lado: se midio
# con estrategias que solo ven PRECIO Y VOLUMEN, y el unico edge del mundo sintetico es un
# AR(1) idiosincratico colocado a mano por regimen (retrofit._idio_ar_for). Rankear
# configuraciones de momentum sobre eso mide que configuracion ajusta mejor ese AR(1); no
# hay motivo para que transfiera. En el mercado real el momentum viene de flujo, atencion y
# narrativa -- el propio informe de fidelidad ya lo decia: "el edge sintetico es mas limpio
# que el real".
#
# Asi que antes de sacar el sintetico del nucleo se amplia el ESPACIO DE INPUTS y se vuelve
# a medir con el mismo instrumento. La hipotesis es testeable y la lista de ahora la sigue:
# medir primero lo que ya se puede medir gratis (sesion horaria, suelo de actividad),
# construir la plataforma de ingesta de senales, y re-correr la transferencia de forma
# pareada. Sacar el sintetico del criterio de seleccion pasa a ser la CONTINGENCIA, no la
# conclusion automatica.
#
# ACTUALIZACION 2026-08-11 (4): el primer lote CONTINUO (Tier B) esta conectado: 11 de las
# 17 fuentes tienen adaptador (`src/ai_trader/signals/adapters/`), 9 tienen profundidad MEDIDA
# y 7 son backtesteables. Las tres cifras se separan a proposito: la profundidad se mide en
# vez de creerse el folleto (`signals/depth.py` -> `data/signals/history_depth.json`), el
# catalogo solo declara `history_from` donde hay medicion Y al menos un ano de ventana (en una
# fuente forward_capture el primer dia es el dia que arranco la captura), y un test lo exige. Toda feature se publica normalizada con dos varas —z contra
# la propia historia (expansiva, causal) y z contra la seccion cruzada del dia, mediana/IQR,
# recorte declarado a +-4, huecos a NaN— para que sirva igual a BTC que a un listado nuevo.
# Lo que la medicion destapo y no estaba en ningun sitio: el COT se conoce TRES dias despues
# de su fecha (se archiva por dia de publicacion), el sello de funding de CCXT es el proximo
# cobro y habria fechado observaciones en el futuro, TFTC publica los ETF de BTC pero no los
# de ETH, FRED ya no sirve oro, y un solo slug con 400 se llevaba por delante las otras 23
# series de su fuente. Sigue sin cablearse nada a estrategias.
#
# ACTUALIZACION 2026-08-11 (3): el ESQUELETO de ingesta de senales esta construido
# (`src/ai_trader/signals/`: catalogo, puerto de dos capas, archivo crudo append-only,
# captura y auditoria; esquema y entidades en `shared/`). No conecta ninguna fuente a
# proposito, y lo que publica es una medicion incomoda y util: 17 fuentes declaradas, 0 con
# adaptador y 0 backtesteables, porque ninguna tiene `history_from` MEDIDO. Lo que si
# arranca hoy es la captura, y ese es el motivo de haberlo hecho antes que los adaptadores:
# 6 de las 17 son forward_capture y su profundidad solo la compra el calendario.
#
# ACTUALIZACION 2026-08-11 (2): las dos mediciones baratas estan hechas y las dos cambiaron
# algo. La ventana ciega resulto no tener ancho (el hueco cierre->open es 0,55 pb) pero la
# LATENCIA si cuesta. Y el suelo de actividad destapo que el ranking real premiaba no
# operar: Spearman(recompensa, operaciones) = -0,84, la ganadora no abria posiciones y aun
# asi aprobaba el gate. Ya no: rankear exige operar (scoring.activity, evidencia en
# data/activity/), el gate exige las dos cosas y el ranking se publica con y sin suelo.
# Ninguna de las dos toca el generador, que sigue siendo el problema de fondo.
#
# DOS PUERTAS, y esta separacion es la que impide sobreajustar: las senales MECANICAS
# (unlocks, colas de staking, mNAV<1, mapas de liquidacion) tienen muestras de decenas, asi
# que entran como ELEGIBILIDAD en el runner y NUNCA en search_space -- el CEM no puede
# alcanzar ni [risk] ni las guardas del runner, luego la imposibilidad de construir una
# estrategia sobre catorce observaciones es una propiedad estructural, no una convencion.
# Las senales ESTADISTICAS (dispersion de funding, atencion geografica, ratios de fees,
# flujos de ETF) si entran como features continuas, esperando efectos pequenos y decadentes.
#
# FOCO: cripto. Renta variable y mercados de prediccion quedan en segundo plano de forma
# EXPLICITA (grupo 'segundo-plano'), no por olvido: toda la evidencia empirica del repo
# -fidelidad contra Binance, calibracion de pesos, estudio de validacion- es cripto, y la
# pata de renta variable no tiene ni un solo dato real detras.

ROADMAP_GROUPS = [
    {
        "key": "ahora",
        "title": "Ahora",
        "subtitle": "El sustrato es fiel pero NO ordena (medido el 2026-08-11). La hipotesis "
                    "que se persigue antes de dar el generador por perdido es que el cuello de "
                    "botella no es el generador sino el ESPACIO DE INPUTS: hoy las estrategias "
                    "solo ven precio y volumen. Las dos mediciones baratas que condicionaban la "
                    "lectura ya estan hechas -la ventana ciega (vista Sesiones) y el suelo de "
                    "actividad del ranking (vista Actividad)-, y la plataforma de ingesta tambien "
                    "(vista Senales): esqueleto, ONCE fuentes continuas conectadas y su "
                    "profundidad MEDIDA fuente a fuente. Lo que queda aqui es la puerta de "
                    "ELEGIBILIDAD para las mecanicas y el cableado al espacio de observacion, con "
                    "el reloj de la captura corriendo en paralelo. Nada de lo que se puntue "
                    "mientras tanto vale mas que el juez que lo puntua.",
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
        "id": "signals-tier-a-eligibility",
        "rank": 1,
        "group": "ahora",
        "priority": "critica",
        "title": "Senales mecanicas (Tier A) como ELEGIBILIDAD, nunca como alfa",
        "line": "Inputs/Riesgo", "status": "pendiente", "impact": "alto", "effort": "medio",
        "evidence": "Los eventos mas informativos (unlocks, colas de staking, deslistados) tienen "
                    "muestras de DECENAS. Y el sistema no tiene hoy ningun concepto de "
                    "elegibilidad: grep de eligib|veto|blacklist|blackout|halt en src/ no "
                    "devuelve nada. Las unicas guardas por simbolo son la posicion abierta y el "
                    "cooldown (app/runner.py:229-235).",
        "why": "Un motor RL sobre features de N pequeno sobreajusta de forma espectacular: si el "
               "optimizador las descubre libre, construye una estrategia preciosa sobre catorce "
               "observaciones. La defensa no es una advertencia en un docstring, es "
               "ARQUITECTONICA -- estas senales entran como guarda de elegibilidad en el runner y "
               "no como features. Y la garantia se hereda gratis: el CEM no puede alcanzar ni "
               "[risk] ni las guardas del runner (scoring/optimize.py solo reconstruye "
               "`strategies`, y search_space.SPACES contiene exclusivamente parametros de "
               "estrategia), asi que la imposibilidad de sobreajustarlas es una propiedad del "
               "sistema, no una convencion que haya que defender.",
        "prompt": (
            "Proyecto ai-trader (Python). Con el esqueleto de src/ai_trader/signals/ construido, "
            "conecta el lote de senales MECANICAS y creales una puerta propia.\n"
            "\n"
            "FUENTES (oferta calendarizada y determinista, todas gratis):\n"
            "1. Unlocks/vesting via la API de emisiones de DefiLlama. Lo valioso NO es 'hay "
            "unlock' (algo explotado) sino el desbloqueo como PORCENTAJE DEL FLOAT CIRCULANTE Y "
            "DEL ADV: un 3% sobre un token con 40 dias de volumen en circulacion es un evento; el "
            "mismo 3% sobre uno liquido no lo es. Casi nadie lo normaliza asi.\n"
            "2. Cola de salida del staking de Ethereum (beaconcha.in, gratis): oferta futura con "
            "FECHA CONOCIDA dias antes. Equivalentes en Solana (activacion/desactivacion por "
            "epoca) y Cosmos (unbonding 21 dias).\n"
            "3. Ajuste de dificultad y hashprice de Bitcoin (mempool.space, gratis sin auth): "
            "cada 2016 bloques con fecha estimable. Hashprice comprimido implica venta forzada de "
            "mineros.\n"
            "4. Hacks y exploits fechados (DefiLlama), lista OFAC SDN legible por maquina, y un "
            "calendario macro (FOMC, CPI, vencimientos).\n"
            "\n"
            "LA PUERTA (signals/eligibility.py): un TradabilityProvider consultado como UNA GUARDA "
            "MAS en TradingRunner._process_symbol, junto al cooldown por simbolo, ANTES de pedir "
            "senal. NO dentro de RiskEngine, y las tres razones importan: (i) RiskEngine no tiene "
            "reloj ni colaboradores -- es una funcion pura de (limites, senal, cartera) y esa "
            "pureza vale; (ii) en modo equity-aware, que es SIEMPRE el del backtest, tres de sus "
            "guardas no se ejecutan (risk/engine.py:108-119), asi que un veto ahi correria el "
            "riesgo de estar silenciosamente inactivo en toda la evidencia; (iii) "
            "_symbol_in_cooldown (app/runner.py:489-502) es el precedente estructural exacto: la "
            "unica guarda que consulta reloj e historial por simbolo y descarta antes de pedir "
            "senal, y ademas es mas barato porque evita cargar barras y correr estrategias.\n"
            "\n"
            "CUATRO PROPIEDADES NO NEGOCIABLES:\n"
            "1. NUNCA en search_space.py. Anadir una dimension al hipercubo latino no anade un "
            "campo a las 16 configuraciones publicadas: candidate_specs hace "
            "rng.random((n, space.dim)), asi que las SUSTITUYE por 16 objetos distintos. Congela "
            "la huella con un test contra data/transfer/report_ai_v3.json ANTES de tocar nada.\n"
            "2. Umbrales DECLARADOS y razonados en codigo (p.ej. desbloqueo > X% del ADV en los "
            "proximos N dias), nunca optimizados.\n"
            "3. Falla abierta: sin datos no hay veto. Un fallo del proveedor no puede parar el "
            "trading.\n"
            "4. EL VETO SE MIDE. Hoy un rechazo es una cadena de texto libre en "
            "RiskDecision.reason (risk/engine.py:239-241): no hay taxonomia ni recuento, asi que "
            "un veto que nunca dispara y uno que dispara siempre son indistinguibles. Cada veto "
            "lleva source_key estructurado y se acumula en SymbolCycleDiagnostics (que ya existe) "
            "para reportar cuantas veces veto cada fuente y sobre que simbolos. Sin esa cifra, "
            "una regla mecanica es una creencia.\n"
            "\n"
            "El orden de las guardas de _process_symbol no esta testeado hoy: congelalo con un "
            "test ANTES de insertar la nueva. Tests + .venv\\Scripts\\python.exe (poetry run esta "
            "roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "signal-radar-wiring",
        "rank": 2,
        "group": "ahora",
        "priority": "alta",
        "title": "Radar de features y cableado en backtest Y en vivo (cierra el hueco del regimen)",
        "line": "Inputs", "status": "pendiente", "impact": "alto", "effort": "medio",
        "evidence": "attach_regime_provider NO se llama en produccion: main.py::build_runner "
                    "construye las estrategias y no adjunta nada, asi que las puertas "
                    "cross-sectional (min_breadth, min/max_relative_strength) solo existen en "
                    "backtest. Cualquier configuracion que el CEM elija con esos filtros activos "
                    "se comporta distinto en paper que en backtest.",
        "why": "Es donde las senales llegan de verdad a la decision. Y hay un regalo: "
               "MarketRegimeProvider solo usa .get(symbol) e iteracion sobre el dict de barras, "
               "asi que un Mapping perezoso sobre MarketDataService lo hace funcionar en vivo SIN "
               "TOCAR observation/regime.py ni una linea -- el mismo adaptador cierra el hueco "
               "que ya arrastra el regimen y sirve al radar nuevo. Cerrar los dos a la vez sale "
               "casi gratis y evita que el sistema tenga dos comportamientos segun donde corra.",
        "prompt": (
            "Proyecto ai-trader (Python). Con las fuentes Tier B ya ingiriendo, cablea el radar a "
            "las estrategias, en backtest y en vivo.\n"
            "\n"
            "(a) PROVIDER (observation/signal_radar.py) con la forma EXACTA de "
            "observation/regime.py::MarketRegimeProvider: features(symbol) -> dict[str, float], "
            "memo por el 'ahora' del reloj, y anti-look-ahead propio importando visible_cutoff de "
            "shared/clock.py. Separa en codigo, no en un comentario, las features de MERCADO "
            "(iguales para todos los simbolos ese dia) de las de ACTIVO: es un invariante "
            "testeable y el generador sintetico lo necesitara.\n"
            "(b) LA TRAMPA DE LOS TRES ESTADOS. La convencion del repo es 'feature no disponible "
            "= 0.0 neutro' (observation/features.py:58-63), y aqui es PELIGROSA: tono 0 es "
            "neutral, pero 'no tengo datos' NO es 'no hay senal'. Anade una feature de COBERTURA "
            "en [0,1] que los distinga, y con ella el invariante central: una puerta de senales "
            "NUNCA bloquea por falta de datos -- sin cobertura suficiente no se evalua (falla "
            "abierta). Implementa el umbral como constante NO configurable, para que ningun "
            "sorteo del CEM convierta el radar en un filtro de disponibilidad de datos.\n"
            "(c) PUERTAS en crypto_momentum y mean_reversion siguiendo el patron de cinco piezas "
            "del regimen: attach_*_provider duck-typed, params con DEFAULT INERTE validados en "
            "__post_init__, puerta _X_active() consultada DESPUES de la de regimen. Fija los "
            "neutros en el borde exacto del recorte de las z, de modo que la inercia del default "
            "quede DEMOSTRADA y no confiada (mejora sobre min_relative_strength=-1.0, que no esta "
            "fuera del rango alcanzable). Ojo con la polaridad: momentum y mean-reversion no "
            "quieren lo opuesto de TODAS las features -- el tono es un piso en ambas (en momentum "
            "como confirmacion, en mean-reversion como filtro de catastrofe: su modo de fallo "
            "caracteristico es comprar una caida de -3 sigma que es el primer dia de un reprecio "
            "permanente), y solo la intensidad es el eje genuinamente opuesto.\n"
            "(d) CABLEADO EN LOS DOS SITIOS. En backtest, backtest/engine.py: __init__ y "
            "from_bars aceptan las senales, y se adjuntan en _build_runner:411-425 con el mismo "
            "bucle duck-typed. En vivo, main.py:58-89: crea un LiveUniverseBars(Mapping) sobre "
            "MarketDataService y adjunta el proveedor de REGIMEN (que hoy no se adjunta, es un "
            "hueco conocido) y el de senales. regime.py no deberia necesitar ni una linea de "
            "cambio.\n"
            "(e) CONFIG: seccion [signals] nueva con enabled=false por defecto. El splat de "
            "config.py es estricto, asi que hace falta dataclass + campo en AppConfig + linea en "
            "load_config. Sin credenciales el sistema tiene que ARRANCAR IGUAL: radar vacio, "
            "cobertura 0, todas las puertas se saltan, warning explicito.\n"
            "\n"
            "COMPUERTA: con las puertas neutras, validate_multiwindow tiene que devolver scores "
            "IDENTICOS a los publicados en data/transfer/units_ai_v3.json. Si no coinciden, para: "
            "algo no es lo que creemos. Tests + .venv\\Scripts\\python.exe (poetry run esta roto) "
            "+ ruff. Regenera dashboard y docs."
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
        "id": "signals-expensive-batch",
        "rank": 5,
        "group": "despues",
        "priority": "media",
        "title": "Lote caro de senales: apalancamiento observable, opciones, atencion geografica y legal",
        "line": "Inputs", "status": "pendiente", "impact": "alto", "effort": "alto",
        "evidence": "Hyperliquid publica ON-CHAIN el libro completo y las posiciones de todos los "
                    "traders, con API gratuita y sin KYC (POST api.hyperliquid.xyz/info). Ningun "
                    "CEX da eso: en Binance ves funding y OI agregado, aqui ves la DISTRIBUCION.",
        "why": "Son fuentes con senal genuina cuya barrera no es el precio sino la friccion de "
               "ingenieria: parsear filings, reconstruir estado on-chain, normalizar APIs mal "
               "documentadas. Nadie lo hace porque es trabajo sucio, no porque sea secreto -- que "
               "es exactamente lo que las mantiene sin arbitrar. Hyperliquid es la mas valiosa: "
               "convierte las cascadas de liquidacion de fenomeno impredecible en algo con "
               "estructura conocida por adelantado.",
        "prompt": (
            "Proyecto ai-trader (Python). Sobre el puerto de src/ai_trader/signals/ ya "
            "construido, conecta el lote de fuentes de alta friccion.\n"
            "\n"
            "1. HYPERLIQUID (POST api.hyperliquid.xyz/info, gratis, sin auth ni KYC). Tres cosas "
            "que ningun CEX permite calcular: distribucion REAL del apalancamiento por activo (no "
            "la media), MAPA DE PRECIOS DE LIQUIDACION (donde estan los clusters y cuanto "
            "notional hay en cada nivel), y concentracion (que fraccion del OI esta en 5 "
            "cuentas: un perp con OI concentrado es un perp con riesgo de gap). El mapa de "
            "liquidacion es Tier A (elegibilidad); la distribucion y la concentracion son Tier B.\n"
            "2. DERIBIT: skew de 25 delta, DVOL y term structure -- de lo mas informativo y "
            "gratuito que existe. Y el calendario de vencimientos mensuales/trimestrales con OI "
            "por strike, que es Tier A por ser fechas fijas.\n"
            "3. LIQUIDACIONES ON-CHAIN de prestamos (Aave/Compound via subgraphs): distribucion "
            "de health factors -> mapa de liquidacion del colateral spot. Es el OTRO LADO del "
            "apalancamiento y complementa a Hyperliquid.\n"
            "4. ATENCION GEOGRAFICA: ranking en App Store de Upbit/Coinbase/Binance/Bitget, pero "
            "la version buena no es el ranking de Coinbase en EE.UU. sino el DIFERENCIAL entre "
            "Upbit en Corea y Coinbase en EE.UU., que dice que retail esta entrando. Mas Naver "
            "DataLab (Corea) y Yandex Wordstat (Rusia), gratuitos y que practicamente nadie usa "
            "fuera de esos paises -- y Corea es desproporcionadamente importante para altcoins.\n"
            "5. LEGAL E INSTITUCIONAL, todo gratis y llega ANTES que la noticia: SEC EDGAR "
            "full-text search (API EFTS) para 13F/13G/S-1/8-K, Federal Register API, "
            "CourtListener/RECAP para dockets. Tier A.\n"
            "6. LISTADOS Y DESLISTADOS DE CEX: el 'efecto Upbit' es de los eventos mas limpios "
            "que existen en cripto, y un deslistado es oferta forzada mas riesgo de liquidez. "
            "Tier A.\n"
            "\n"
            "Respeta la separacion de las dos puertas: lo mecanico y de N pequeno va a "
            "elegibilidad con umbral declarado; lo continuo va a features normalizadas. Registra "
            "en el catalogo el ADV tipico de las entidades donde cada senal existe: varias son "
            "genuinas pero viven en activos donde no cabe tamano, y eso hay que saberlo antes de "
            "escalar. Tests + .venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera "
            "dashboard y docs."
        ),
    },
    {
        "id": "synthetic-signal-emission",
        "rank": 4,
        "group": "despues",
        "priority": "critica",
        "title": "Que el generador emita las senales, y re-medir la transferencia de forma pareada",
        "line": "B/D", "status": "pendiente", "impact": "alto", "effort": "alto",
        "evidence": "Los FactorShock del generador YA son eventos con dia, factor y magnitud "
                    "(synthetic/scenarios.py:105-134), descritos como 'un anuncio de la Fed, un "
                    "default, un ataque'. El generador ya sabe QUE pasa y CUANDO: las senales "
                    "sinteticas serian la emision observable de un estado latente que ya existe.",
        "why": "Es la ficha que VALIDA O REFUTA la tesis entera: si ampliar el espacio de inputs "
               "hace que el ranking transfiera, el sintetico se queda en el nucleo; si no, la "
               "contingencia (rank 17) se activa. Y es la mas delicada del roadmap por un riesgo "
               "concreto: si las senales sinteticas se emiten del mismo estado latente que los "
               "precios con un acoplamiento limpio, funcionaran demasiado bien en el sintetico; y "
               "si ajustamos ese acoplamiento hasta que rho suba, habremos calibrado el generador "
               "contra nuestro propio instrumento de medida.",
        "prompt": (
            "Proyecto ai-trader (Python). Con las senales reales ya ingiriendo y medidas, haz que "
            "el generador sintetico las emita y re-mide la transferencia.\n"
            "\n"
            "(a) EMISION EN UN PASE APARTE, no dentro de PathEngine.generate. Motivo decisivo: "
            "asi la no interferencia con la secuencia RNG no es una promesa que haya que auditar "
            "leyendo el codigo, es una IMPOSIBILIDAD ESTRUCTURAL -- el emisor recibe las barras ya "
            "cerradas. tests/test_synthetic.py::TestEngineByteIdentity congela dos SHA de "
            "librerias publicadas y su docstring avisa de que ponerlos en rojo significa que una "
            "libreria ha dejado de ser reproducible.\n"
            "(b) CAMPOS DE SPEC via MICROSTRUCTURE_FIELDS (synthetic/scenarios.py:12-20, punto de "
            "extension ya declarado: solo se serializa lo NO neutro, asi que los spec.json "
            "existentes no cambian ni un byte). Regla dura: 0 = MENOS edge, nunca mas. Parametriza "
            "en positivo (informative_share, coverage) y no en negativo (false_positive_rate), "
            "para que un default olvidado degrade a 'sin senal' y no a 'senal perfecta'.\n"
            "(c) EL MODELO necesita cuatro piezas y ninguna es opcional: llegada AGRUPADA "
            "(sin autoexcitacion, 'pico de atencion' no existe como concepto y el experimento "
            "queda sesgado hacia el nulo); FALSOS POSITIVOS (son lo unico que da COSTE a la "
            "puerta: sin ellos, cerrar ante una senal nunca renuncia a nada bueno y radar-on "
            "domina por construccion); FALSOS NEGATIVOS (sin ellos, la ausencia de senal es un "
            "certificado perfecto de calma, y encima refuerza que el CVaR ya premia no operar); y "
            "LEAD/LAG (sin ese eje el radar colapsa a un filtro de volatilidad).\n"
            "(d) OJO CON EL JITTER: _apply_shocks recoloca el dia del shock por path con un RNG "
            "salado, y ai_v3 pone jitter_days=15 en TODOS los shocks. Si el emisor usa shock.day "
            "del spec, la senal y el evento se desacoplan en todos los paths. Extrae una funcion "
            "pura effective_shocks(...) que replique exactamente la logica actual.\n"
            "(e) ANTICIRCULARIDAD, y esto es el corazon: cada mando se calibra contra UNA cifra "
            "medida en datos REALES (el acoplamiento contra el IC medido, informative_share "
            "contra la precision, coverage contra el recall), con la misma definicion de 'evento' "
            "en los dos mundos. Cuatro cerrojos: constantes atadas por test a un informe medido; "
            "BANDA DE ACEPTACION DE DOS COLAS en el estudio de fidelidad (ser DEMASIADO predictivo "
            "es un FALLO y devuelve 1 -- es lo que convierte la circularidad en un test rojo en "
            "vez de en un logro); huella de calibracion que el estudio de transferencia exige "
            "para correr; y un libro de intentos que haga visible la multiplicidad.\n"
            "(f) EL EXPERIMENTO: cuatro brazos sobre LAS MISMAS 16 configuraciones (inyecta los "
            "params de puerta con dataclasses.replace, sin tocar search_space, para que los "
            "config_id sean literalmente los publicados). off (control, verificando que reproduce "
            "data/transfer/units_ai_v3.json), on, PLACEBO (senales desplazadas CIRCULARMENTE, no "
            "barajadas: el desplazamiento preserva el agrupamiento y la distribucion y destruye "
            "SOLO la alineacion con los precios, asi que controla a la vez la informacion y la "
            "frecuencia de operacion) y ORACULO (acoplamiento absurdo, etiquetado como "
            "diagnostico no publicable: si ni haciendo trampa sube rho, el cuello de botella no "
            "son los inputs sino el instrumento). Bootstrap PAREADO: remuestrea los indices de "
            "bloque UNA vez por replica y calcula los dos brazos sobre los mismos bloques.\n"
            "(g) Criterio de exito declarado en el codigo ANTES de correr, y la lectura de cada "
            "rama posible escrita de antemano para que ninguna se pueda reinterpretar despues.\n"
            "\n"
            "Tests + determinismo + .venv\\Scripts\\python.exe (poetry run esta roto) + ruff. "
            "Regenera dashboard y docs."
        ),
    },
    {
        "id": "dat-mnav-index",
        "rank": 6,
        "group": "despues",
        "priority": "media",
        "title": "Indice de estres de vendedores forzados (mNAV de tesorerias cotizadas)",
        "line": "Inputs", "status": "pendiente", "impact": "medio", "effort": "alto",
        "evidence": "El canal se ha extendido a mas de 200 companias con mas de 100.000 millones "
                    "en cripto en 2026, y la maquina funciona en reversa de forma observable: "
                    "Strategy vendio 3.588 BTC por unos 216 millones entre el 29 de junio y el 5 "
                    "de julio de 2026, por debajo de su coste medio y sin ventas de equity via "
                    "ATM -- la prima se habia comprimido lo bastante como para que vender saliera "
                    "mas barato que emitir acciones.",
        "why": "Es la novedad estructural del ciclo y la construccion que casi nadie hace: no el "
               "mNAV de una compania, sino la DISTRIBUCION de mNAV a traves de los 200+ DATs, "
               "desagregada por activo subyacente. Cuando la cola inferior engorda hay oferta "
               "futura estructural sobre ese activo. Y como las tesorerias de SOL o ETH pueden "
               "crecer organicamente via staking, sus mNAV corren mas altos que los puros de BTC: "
               "la compresion RELATIVA entre ellos dice donde se cierra el grifo primero. Va al "
               "final por coste de ingenieria, no por falta de valor -- y ese coste es "
               "precisamente lo que lo deja sin arbitrar.",
        "prompt": (
            "Proyecto ai-trader (Python). Sobre el puerto de src/ai_trader/signals/, construye un "
            "indice de estres de vendedores forzados a partir de las tesorerias cotizadas (DATs).\n"
            "\n"
            "No hay API libre: bitcointreasuries.net, mnav.io, bitcoinquant y Artemis son "
            "dashboards. Hay que COMPONER la serie a mano, y esa friccion es exactamente lo que "
            "mantiene la senal sin arbitrar:\n"
            "(a) Holdings por compania y por activo subyacente (comunicados y trackers publicos).\n"
            "(b) Share count desde SEC EDGAR (la API EFTS ya conectada en el lote caro).\n"
            "(c) Precio de mercado de la accion.\n"
            "-> mNAV = capitalizacion / valor del tesoro.\n"
            "\n"
            "LO QUE HAY QUE PUBLICAR no es el mNAV de cada compania sino la DISTRIBUCION: "
            "percentiles por activo subyacente (BTC, ETH, SOL), fraccion de companias por debajo "
            "de 1, y la compresion relativa entre grupos. La cola inferior engordando es oferta "
            "futura estructural. Los descuentos extremos existen y no son teoricos: Hyperion DeFi "
            "cotizaba a finales de julio de 2026 con un mNAV entre 0,24x y 0,31x, con un tesoro "
            "de 120,6 millones contra una capitalizacion de unos 37.\n"
            "\n"
            "ENTRA COMO TIER A (elegibilidad), no como feature: es un evento estructural de N "
            "pequeno. Umbral declarado y razonado, falla abierta, y veto contado por fuente y "
            "simbolo. Cuidado con la latencia: los holdings se publican con retraso y de forma "
            "irregular, asi que el descriptor tiene que declarar el lag real y el archivo guardar "
            "fetched_at -- sin eso, el backtest usaria informacion que no existia ese dia.\n"
            "\n"
            "Tests + .venv\\Scripts\\python.exe (poetry run esta roto) + ruff. Regenera dashboard "
            "y docs."
        ),
    },
    {
        "id": "line-d-cpcv-two-stage-cem",
        "rank": 7,
        "group": "despues",
        "priority": "alta",
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
        "id": "validation-study-full-ensemble",
        "rank": 8,
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
        "rank": 9,
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
        "rank": 10,
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
        "rank": 11,
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
        "rank": 12,
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
        "id": "real-substrate-primary-ranking",
        "rank": 13,
        "group": "despues",
        "priority": "critica",
        "title": "CONTINGENCIA: mover el sustrato primario del ranking al historico REAL",
        "line": "B/D", "status": "bloqueada", "impact": "alto", "effort": "alto",
        "evidence": "MEDIDO: Spearman entre el ranking real y el sintetico = -0,04 sobre 16 "
                    "configuraciones (IC95% por bloques [-0,44, +0,49], p = 0,89). El top-4 del "
                    "sintetico acierta 1 de 4 en la mitad buena del real, peor que el azar (2,0). "
                    "Y sobre las 9 que operan de verdad en los dos mundos el acuerdo es NEGATIVO "
                    "(-0,67). Informe: data/transfer/report_ai_v3.json.",
        "why": "La evidencia de arriba sigue en pie y no se toca. Lo que cambia es la LECTURA: se "
               "midio con estrategias que solo ven precio y volumen, y el unico edge del mundo "
               "sintetico es un AR(1) colocado a mano por regimen -- rankear momentum sobre eso "
               "mide que configuracion ajusta mejor ese AR(1), y no hay motivo para que "
               "transfiera. La hipotesis alternativa (el cuello de botella es el ESPACIO DE "
               "INPUTS, no el generador) es testeable con el mismo instrumento, y es lo que "
               "persiguen los ranks 1-9. Por eso esta ficha pasa de conclusion automatica a "
               "CONTINGENCIA: se ejecuta si `synthetic-signal-emission` (rank 9) refuta esa "
               "hipotesis -- es decir, si con el espacio de inputs ampliado el ranking sigue sin "
               "transferir, y en particular si tampoco transfiere en el brazo ORACULO, que hace "
               "trampa a proposito. Si ni haciendo trampa transfiere, el problema no son los "
               "inputs y hay que sacar el sintetico del criterio de seleccion. Bloqueada, no "
               "descartada: el codigo sigue eligiendo hoy con un juez del que se sabe que no "
               "transfiere, y eso no deja de ser cierto mientras se prueba la alternativa.",
        "prompt": (
            "Proyecto ai-trader (Python). El estudio de transferencia "
            "(src/ai_trader/scoring/transfer_study.py, informe en "
            "data/transfer/report_ai_v3.json, vista 'Transferencia' del dashboard) ya respondio la "
            "pregunta que decidia la arquitectura, y la respuesta fue NO: el Spearman entre el "
            "ranking real y el sintetico es -0,04 (IC95% por bloques [-0,44, +0,49], p = 0,89), el "
            "top-4 del sintetico acierta 1 de 4 en la mitad buena del real (azar = 2) y sobre las 9 "
            "configuraciones que operan de verdad en los dos mundos el acuerdo es NEGATIVO (-0,67, "
            "IC [-0,88, +0,23]). La regla de decision estaba escrita en el codigo ANTES de mirar "
            "(transfer_study.RHO_ACCEPT = 0.30), asi que el flujo queda fijado: 'real como sustrato "
            "primario del ranking, sintetico como capa de estres y veto'.\n"
            "\n"
            "PROBLEMA: el codigo no hace eso. scoring/optimize.py::run_optimization, "
            "scoring/sample_eval.py::evaluate_sample_detailed y el CEM entero puntuan SOLO sobre "
            "librerias sinteticas (SyntheticStore -> load_bars). El unico sitio donde una estrategia "
            "se puntua hoy contra el mercado real es el estudio de transferencia y el comando "
            "`backtest` a mano.\n"
            "\n"
            "TAREA: que el ranking que DECIDE salga del historico real.\n"
            "(a) Un evaluador real reutilizable: la maquinaria de transfer_study (universo comun, "
            "sub-ventanas del mismo tamano, CPCV purgado, agregacion en comun de todos los folds) "
            "extraida a algo que el optimizador pueda llamar, no duplicada. Cuida el punto que ya "
            "resolvio ese estudio: los simbolos sin historico suficiente se declaran y se omiten.\n"
            "(b) Dos etapas explicitas: el sintetico CRIBA (es barato, tiene ensemble y cubre "
            "regimenes que la historia no dio) y el real DECIDE. El resultado de la criba no puede "
            "aparecer como veredicto en ningun sitio.\n"
            "(c) El sintetico como VETO, que es para lo que si esta validado: una configuracion que "
            "gana en el real pero se hunde en los escenarios de crisis sinteticos no asciende. "
            "Declara el criterio de veto y mide cuantas veta.\n"
            "(d) HONESTIDAD: el historico real es UN camino con pocos bloques independientes (5 "
            "sub-ventanas de 544 dias en el estudio actual). Rankear ahi tiene su propio problema de "
            "sobreajuste, y es el problema que el sintetico venia a resolver. Hay que medirlo, no "
            "taparlo: PBO y DSR sobre el lado real, y el numero de bloques efectivos declarado en "
            "cada ranking publicado.\n"
            "\n"
            "Tests + determinismo + .venv\\Scripts\\python.exe (poetry run esta roto) + ruff. "
            "Regenera dashboard y docs."
        ),
    },
    {
        "id": "rl-full-run",
        "rank": 14,
        "group": "despues",
        "priority": "alta",
        "title": "Optimizacion CEM completa, ya con el juez validado",
        "line": "RL", "status": "bloqueada", "impact": "alto", "effort": "medio",
        "depends": 1,
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
        "id": "execution-latency-budget",
        "rank": 15,
        "group": "despues",
        "priority": "media",
        "title": "Presupuesto de latencia: el backtest supone que se llena a las 00:00 UTC en punto",
        "line": "Medicion", "status": "pendiente", "impact": "medio", "effort": "bajo",
        "evidence": "MEDIDO (data/sessions/report.json): el hueco entre el cierre que la estrategia "
                    "ve y el open al que se llena es CERO a efectos practicos -0,07% del rango "
                    "diario, 0,55 pb-, asi que la convencion de llenado no sesga nada. Pero eso "
                    "solo vale si la orden sale en el instante del open: con UNA hora de retraso el "
                    "precio de llenado ya se ha desplazado 57,9 pb (9,2% del rango del dia), que "
                    "son 3,9x el coste de entrada de referencia que el motor si cobra (15 pb).",
        "why": "El estudio de sesiones cerro la pregunta que tenia abierta el backtest (la ventana "
               "ciega no tiene ancho) y abrio otra que hoy no esta ni medida ni presupuestada: el "
               "backtest describe un sistema PUNTUAL, y nada en el repo obliga al ciclo real a "
               "serlo. No es un bug del motor -por eso no se toco- sino un requisito no escrito "
               "que hay que convertir en presupuesto explicito: cuanto puede tardar el ciclo en "
               "ejecutar antes de que el backtest deje de describirlo. Va en 'despues' y no en "
               "'ahora' porque solo muerde cuando el paper trading corra en vivo (#6), que es "
               "donde la latencia deja de ser hipotetica.",
        "depends": 6,
        "prompt": (
            "Proyecto ai-trader (Python). El estudio de sesiones (data/sessions/report.json, "
            "ai_trader/backtest/session_study.py) ya midio que el hueco cierre-visto -> "
            "open-llenado es cero, pero que llegar UNA hora tarde desplaza el llenado 57,9 pb "
            "(3,9x el coste de entrada de referencia). El backtest supone ejecucion instantanea al "
            "open de las 00:00 UTC y nada obliga al ciclo real a cumplirlo.\n"
            "\n"
            "TAREA: convertir eso en un presupuesto de latencia explicito y comprobable.\n"
            "(a) Instrumenta el runner para registrar el retraso real entre la frontera de la vela "
            "diaria y el instante del fill, y persistelo con el resto del estado de ejecucion.\n"
            "(b) Declara el presupuesto como constante razonada (cuanto retraso se acepta antes de "
            "que el coste no modelado supere una fraccion declarada del coste de ejecucion que el "
            "motor ya cobra) y derivalo de las cifras del informe de sesiones, no a ojo.\n"
            "(c) Alerta cuando se incumpla, por el mismo canal que el resto de avisos operativos.\n"
            "(d) NO cambies el modelo de mercado del backtest sin volver a medir: si se decide "
            "cobrar la latencia, tiene que salir del informe de sesiones re-corrido, no de una "
            "estimacion.\n"
            "\n"
            "Determinismo + tests + .venv\\Scripts\\python.exe (poetry run esta roto) + ruff. "
            "Amplia la vista Sesiones del dashboard con lo medido en vivo. Regenera dashboard y "
            "docs."
        ),
    },
    {
        "id": "new-crypto-strategies",
        "rank": 16,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Nuevas estrategias cripto (deliberadamente NO priorizada)",
        "line": "Estrategias", "status": "pendiente", "impact": "medio", "effort": "alto",
        "depends": 1,
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
        "rank": 17,
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
        "rank": 18,
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
        "rank": 19,
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
        "rank": 20,
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
