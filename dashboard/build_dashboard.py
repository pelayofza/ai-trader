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
    FIDELITY_REPORT,
    TARGET_METRIC_KEYS,
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

    return {"ai_v1": survey(COMPARE_LIB), "ai_v2": survey(PRIMARY_LIB)}


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
        "ai_v1": lib_stats(COMPARE_LIB),
        "ai_v2": lib_stats(PRIMARY_LIB),
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


def collect_fidelity() -> dict | None:
    """Comparacion de los stylized-facts de ai_v2 contra el historico real (data/fidelity).

    Se LEE del informe publicado; no se recalcula. Medirlo exige descargar ocho anos de
    historico y recorrer la libreria entera, y el dashboard tiene que seguir siendo
    regenerable en minutos."""
    report = load_fidelity_report(ROOT / FIDELITY_REPORT)
    if not report:
        logger.warning("Sin informe de fidelidad: el panel sintetico-vs-real saldra vacio")
        return None

    plan = report["plan"]
    metrics = [
        {
            **m,
            "is_target": m["key"] in TARGET_METRIC_KEYS,
            "decimals": metric(m["key"]).decimals,
        }
        for m in report["metrics"]
    ]
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
        "cross": {**report["cross_correlation"], "is_target": True, "decimals": 3},
        "summary": report["summary"],
        "generated_at": report["generated_at"][:10],
    }


def collect_roadmap() -> list[dict]:
    """Evoluciones pendientes, cada una con un prompt copiable para Claude Code."""
    from ai_trader.synthetic import retrofit  # noqa: F401  (asegura que el modulo existe)
    return ROADMAP


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
        "roadmap": collect_roadmap(),
    }

    from dashboard.template import render_html  # import tardio: template en modulo aparte

    OUT_HTML.write_text(render_html(data), encoding="utf-8")
    logger.info("Dashboard escrito en %s", OUT_HTML)


# --- catalogo de evoluciones pendientes (con prompts detallados para Claude Code) ---

ROADMAP = [
    {
        "id": "line-a-calibration-multiwindow",
        "title": "Calibracion de pesos con validacion multiventana",
        "line": "A", "status": "pendiente", "impact": "bajo", "effort": "medio",
        "why": "El barrido de lambda/kappa ya esta hecho y publicado (data/calibration): la "
               "superficie resulto PLANA y lambda no duplica los costes ya pagados. Pero se midio "
               "con un unico split temporal 70/30, un camino por escenario y 16 configuraciones: "
               "sirve para descartar que los pesos importen mucho, no para afinar decimales.",
        "prompt": (
            "Proyecto ai-trader (Python). Los pesos del headline score "
            "(src/ai_trader/backtest/metrics.py: DEFAULT_HEADLINE_WEIGHTS) ya estan calibrados "
            "con evidencia: el estudio vive en src/ai_trader/scoring/weight_study.py y su "
            "informe publicado en data/calibration/report_ai_v2.json. El resultado fue que la "
            "superficie (lambda, kappa) es PLANA en rank IC y en gap train-validation.\n"
            "TAREA: subir la potencia estadistica del estudio para saber si la planitud es real "
            "o falta de resolucion. (1) Repite el barrido con varios splits temporales por "
            "muestra (walk-forward, no un unico 70/30) y con mas de un camino por escenario, "
            "reutilizando el cacheo de componentes crudos que ya existe (los componentes no "
            "dependen de los pesos). (2) Anade intervalos de confianza por bootstrap sobre "
            "escenarios al rank IC y al gap, para poder afirmar o descartar diferencias entre "
            "puntos de la rejilla. (3) Si y solo si la evidencia nueva mueve el optimo fuera del "
            "error, actualiza DEFAULT_HEADLINE_WEIGHTS y el test que los congela "
            "(tests/test_backtest_metrics.py::TestDefaultWeightsAreCalibrated). Determinismo + "
            ".venv\\Scripts\\python.exe (poetry run roto) + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "line-c-recalibrate",
        "title": "Re-medir lambda y kappa con los costes nuevos",
        "line": "A/C", "status": "pendiente", "impact": "bajo", "effort": "medio",
        "why": "La calibracion publicada (data/calibration) se midio con el slippage PLANO "
               "que ya no existe. El modelo actual cobra mas friccion y la reparte por "
               "simbolo y tamano: refuerza la conclusion (lambda es un margen pequeno sobre "
               "costes ya pagados), pero los decimales no estan re-medidos.",
        "prompt": (
            "Proyecto ai-trader (Python). El estudio que fija los pesos del headline score "
            "(src/ai_trader/scoring/weight_study.py, informe en data/calibration/"
            "report_ai_v2.json) se midio cuando el motor cobraba un slippage PLANO de 5 bps. "
            "Hoy la ejecucion usa un modelo de microestructura "
            "(src/ai_trader/execution/microstructure.py: medio spread por simbolo + "
            "volatilidad reciente + impacto por raiz cuadrada de la participacion) y un techo "
            "de capacidad por barra.\n"
            "TAREA: (1) Repite el barrido de (lambda, kappa) con los costes actuales sobre "
            "ai_v2 y publica el informe nuevo sin borrar el anterior, para poder comparar. "
            "(2) Revisa turnover_cost_audit en src/ai_trader/scoring/weight_calibration.py: "
            "hoy deriva el lambda implicito de un cost_rate PLANO (fee_rate + slippage_bps); "
            "hazlo a partir del slippage REALMENTE cobrado por operacion (ExecutionResult."
            "slippage_bps), que ya no es una constante. (3) Decide con la evidencia nueva si "
            "DEFAULT_HEADLINE_WEIGHTS se mueve, y actualiza el test que los congela. "
            "Determinismo + .venv\\Scripts\\python.exe (poetry run roto) + ruff. Regenera "
            "dashboard y docs."
        ),
    },
    {
        "id": "line-d-validation",
        "title": "Validacion CPCV / walk-forward",
        "line": "D", "status": "pendiente", "impact": "medio", "effort": "medio",
        "why": "El split actual es un unico 70/30 (_resolve_cutoff). Sobre-estima la robustez "
               "y no purga ni embarga entre train y test.",
        "prompt": (
            "Proyecto ai-trader (Python). El backtest (src/ai_trader/backtest/engine.py, metodo "
            "_resolve_cutoff) parte cada muestra en train/test con un unico split temporal "
            "70/30, sin purga ni embargo; sobre-estima la robustez.\n"
            "TAREA: implementa validacion multiventana — sustituye o complementa ese split con "
            "walk-forward multiventana y, si es viable, CPCV (Combinatorial Purged "
            "Cross-Validation) con purga y embargo entre train y test. Agrega los headline "
            "scores de todas las ventanas en una distribucion robusta. Anade tests de que no "
            "hay fuga temporal entre folds. Regenera dashboard y docs. Determinismo + "
            ".venv\\Scripts\\python.exe (poetry run roto) + ruff."
        ),
    },
    {
        "id": "line-b8-tail-calibration",
        "title": "Subir colas y clustering de ai_v2 al nivel medido",
        "line": "B", "status": "pendiente", "impact": "alto", "effort": "medio",
        "why": "El harness de fidelidad ya midio el hueco contra el mercado real "
               "(data/fidelity): la curtosis sintetica es una fraccion de la real y el "
               "clustering, la mitad. Mientras siga asi, los backtests subestiman la perdida "
               "de cola y las estrategias parecen mas seguras de lo que serian.",
        "prompt": (
            "Proyecto ai-trader (Python). El estudio de fidelidad "
            "(src/ai_trader/synthetic/fidelity_study.py, informe en data/fidelity/"
            "report_ai_v2.json, vista 'Fidelidad' del dashboard) ya midio los stylized-facts de "
            "la libreria sintetica ai_v2 contra el historico real de cripto de Binance. "
            "Resultado: el NIVEL de volatilidad y la ORDENACION de las correlaciones cruzadas "
            "son razonables, pero las COLAS (curtosis en exceso y exceedances mas alla de 3 "
            "sigma) y el CLUSTERING de volatilidad se quedan muy por debajo del mercado real, "
            "con cobertura casi nula: el ensemble sintetico no llega a producir el valor real "
            "ni en su percentil 90.\n"
            "TAREA: cierra ese hueco en el generador. (1) En src/ai_trader/synthetic/"
            "retrofit.py y synthetic/engine.py, recalibra los parametros de microestructura "
            "(tail_dof, vol_persistence, jump_intensity/jump_scale y su asignacion por fase) "
            "para que las metricas medidas se acerquen a las reales; hoy las colas solo se "
            "activan en fases de crisis y el clustering base es 0,85 plano. (2) Usa el propio "
            "harness como funcion objetivo: itera regenerando y re-midiendo con "
            "'python -m ai_trader.synthetic.fidelity_study --offline' (los datos reales ya "
            "estan cacheados) hasta que la cobertura de curtosis y clustering deje de ser "
            "cero. (3) Conserva los invariantes que ya estan testeados: ajuste en varianza "
            "(anadir cola no cambia la volatilidad total), velas validas y neutralidad de los "
            "valores por defecto. (4) Regenera la libreria a una nueva version (ai_v3) sin "
            "borrar ai_v2, publica el informe de fidelidad de ambas y compara. Determinismo + "
            "tests + .venv\\Scripts\\python.exe (poetry run roto) + ruff. Regenera dashboard y "
            "docs."
        ),
    },
    {
        "id": "line-e-cleanup",
        "title": "Limpieza de consistencia",
        "line": "E", "status": "pendiente", "impact": "bajo", "effort": "bajo",
        "why": "MATIC/USDT sigue en el universo pese a estar deslistado en Binance; "
               "TRADING_DAYS_PER_YEAR=365 desanualiza mal la renta variable; el designer usa "
               "temperature=1.0 (diseño no reproducible, mitigado guardando spec.json).",
        "prompt": (
            "Proyecto ai-trader (Python). Tres inconsistencias de higiene a resolver:\n"
            "1) MATIC/USDT sigue en el universo sintetico (src/ai_trader/synthetic/universe.py) "
            "y en config/synthetic.toml pese a estar deslistado en Binance: o retiralo de ambos "
            "sitios, o documenta explicitamente por que se mantiene; deja consistentes "
            "config/default.toml y config/synthetic.toml.\n"
            "2) TRADING_DAYS_PER_YEAR en src/ai_trader/backtest/metrics.py es 365 global, lo que "
            "desanualiza mal la renta variable: usa 252 para acciones y 365 para cripto al "
            "anualizar metricas por clase de activo.\n"
            "3) Verifica que DEFAULT_MODEL del disenador (src/ai_trader/synthetic/designer.py) "
            "resuelve contra la API actual y documenta que con temperature=1.0 el diseño no es "
            "reproducible (se mitiga guardando el spec.json). Tests donde aplique. Regenera "
            "dashboard y docs. .venv\\Scripts\\python.exe (poetry run roto) + ruff."
        ),
    },
    {
        "id": "paper-trading-view",
        "title": "Vista de paper trading en el dashboard",
        "line": "Dashboard", "status": "placeholder", "impact": "medio", "effort": "bajo",
        "why": "El runner ya opera en paper con estado persistido (JsonStateStore). Falta "
               "exponer equity, posiciones abiertas y PnL en el dashboard.",
        "prompt": (
            "Proyecto ai-trader (Python). El orquestador (TradingRunner, "
            "src/ai_trader/app/runner.py) ya opera en paper con estado persistido via "
            "JsonStateStore (posiciones abiertas/cerradas, PnL realizado, pausa). El dashboard "
            "(dashboard/) tiene una seccion 'Paper trading' que hoy es solo placeholder.\n"
            "TAREA: anade al generador del dashboard (dashboard/build_dashboard.py + "
            "dashboard/template.py) una vista de paper trading que lea el estado del runner "
            "(JsonStateStore): curva de equity marcada a mercado, tabla de posiciones abiertas "
            "y cerradas con PnL neto de comisiones, y metricas de riesgo (exposicion desplegada, "
            "nº de posiciones vs maximo, drawdown de cuenta). Renderiza la equity como line "
            "chart (reusa los helpers SVG del template) y las posiciones como tabla; manten el "
            "HTML autocontenido. Regenera con python -m dashboard.build_dashboard."
        ),
    },
    {
        "id": "rl-full-run",
        "title": "Optimizacion CEM completa sobre ai_v2",
        "line": "RL", "status": "pendiente", "impact": "alto", "effort": "medio",
        "why": "El harness CEM (scoring/optimize.py) esta listo pero cada backtest cuesta ~60s "
               "con 35 activos; una corrida completa necesita subsampleo o paralelizacion.",
        "prompt": (
            "Proyecto ai-trader (Python). El harness de optimizacion por Cross-Entropy Method "
            "(src/ai_trader/scoring/optimize.py, funcion run_optimization) esta listo, pero cada "
            "backtest cuesta ~60s con los 35 activos, asi que una corrida completa sobre las 900 "
            "muestras (30 escenarios x 30 paths) necesita subsampleo o paralelizacion.\n"
            "TAREA: ejecuta y consolida la optimizacion CEM de las dos primitivas "
            "(crypto_momentum y mean_reversion) sobre la libreria ai_v2 (ya es el sustrato por "
            "defecto: DEFAULT_LIBRARY_ID en scoring/optimize.py). La metrica de cabecera honesta "
            "(Sharpe - lambda*turnover - kappa*maxDD, agregada por CVaR@25%, con gate de "
            "baselines y descuento DSR/PBO) YA esta implementada, asi que run_optimization "
            "devuelve tambien el veredicto del gate y el sobreajuste por multiples pruebas: "
            "reportalos. Optimiza el rendimiento del backtest o paraleliza la evaluacion de "
            "muestras para que una corrida con subsampleo razonable sea tratable. Guarda los "
            "mejores params por primitiva y su distribucion train/validation, y vuelca los "
            "resultados al dashboard (seccion ranking, dashboard/build_dashboard.py). Regenera "
            "dashboard y docs. Determinismo + tests + .venv\\Scripts\\python.exe (poetry run roto)."
        ),
    },
]


if __name__ == "__main__":
    build()
