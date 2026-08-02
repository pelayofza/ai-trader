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

from ai_trader.config import StrategySpec, load_config
from ai_trader.observation.features import OWN_ASSET_FEATURES
from ai_trader.observation.regime import REGIME_FEATURES
from ai_trader.scoring.aggregate import aggregate_reward
from ai_trader.scoring.sample_eval import evaluate_sample
from ai_trader.shared import bars as bar_schema
from ai_trader.strategies import build_strategy
from ai_trader.strategies.mean_reversion import MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy
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
    """Ranking real sobre una muestra reducida de ai_v2. Rankea por CVaR@25% del Calmar OOS."""
    result: dict = {
        "scope": {
            "library": RANK_LIB,
            "universe": RANK_UNIVERSE,
            "n_scenarios": RANK_N_SCENARIOS,
            "n_paths": RANK_N_PATHS,
            "window_days": RANK_WINDOW_DAYS,
        },
        "rows": [],
        "distributions": {},
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

    for label, stype, params in RANK_CONFIGS:
        spec = StrategySpec(type=stype, id=label, params=params)
        scores: list[float] = []
        for sid in chosen:
            for p in range(RANK_N_PATHS):
                try:
                    allbars = store.load_bars(RANK_LIB, sid, p)  # una sola lectura del parquet
                    bars = {s: allbars[s] for s in RANK_UNIVERSE if s in allbars}
                    val = evaluate_sample(base_config, spec, bars, start, end, split_ratio=0.7)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("eval fallo %s/%s: %s", label, sid, exc)
                    continue
                scores.append(val)
        if not scores:
            continue
        stats = aggregate_reward(scores, lam=0.5)
        result["rows"].append(
            {
                "label": label,
                "type": stype,
                "cvar25": round(stats.cvar25, 3),
                "mean": round(stats.mean, 3),
                "p25": round(stats.p25, 3),
                "std": round(stats.std, 3),
                "worst": round(stats.worst, 3),
                "best": round(stats.best, 3),
                "reward": round(stats.reward, 3),
                "n": stats.n,
            }
        )
        result["distributions"][label] = [round(s, 3) for s in scores]

    result["rows"].sort(key=lambda r: r["cvar25"], reverse=True)
    return result


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
    logger.info("Catalogo de estrategias...")
    strategies = collect_strategies()
    logger.info("Demo de señales...")
    signals = strategy_signals_demo(store, synthetic)
    logger.info("Ranking (muestra reducida, puede tardar unos minutos)...")
    ranking = run_ranking(store)
    kpis = collect_kpis(store, synthetic)

    data = {
        "kpis": kpis,
        "synthetic": synthetic,
        "facts": facts,
        "strategies": strategies,
        "signals": signals,
        "ranking": ranking,
        "roadmap": collect_roadmap(),
    }

    from dashboard.template import render_html  # import tardio: template en modulo aparte

    OUT_HTML.write_text(render_html(data), encoding="utf-8")
    logger.info("Dashboard escrito en %s", OUT_HTML)


# --- catalogo de evoluciones pendientes (con prompts detallados para Claude Code) ---

ROADMAP = [
    {
        "id": "wiring-ai-v2",
        "title": "Cablear ai_v2 como sustrato por defecto del RL",
        "line": "Wiring", "status": "pendiente", "impact": "alto", "effort": "bajo",
        "why": "ai_v2 corrige los sesgos optimistas del generador (colas, clustering, "
               "estructura serial). El harness de scoring aun optimiza sobre ai_v1 iid.",
        "prompt": (
            "En scoring/optimize.py cambia el library_id por defecto de 'ai_v1' a 'ai_v2' "
            "en run_optimization. Revisa que scoring/sample_eval y cualquier CLI/documentacion "
            "que referencie 'ai_v1' se actualicen o queden explicitos. Anade un test que "
            "verifique que run_optimization usa ai_v2 por defecto. Corre la suite completa "
            "con .venv\\Scripts\\python.exe -m pytest -q."
        ),
    },
    {
        "id": "line-a-metric",
        "title": "Linea A - Metrica y ranking honestos",
        "line": "A", "status": "pendiente", "impact": "alto", "effort": "medio",
        "why": "El headline actual es Calmar OOS, que es toxico: el maxDD en el denominador "
               "es ruidoso (un extremo de un solo path), premia la inactividad y degenera a 0 "
               "sin drawdown. Un optimizador convergeria a 'no operar casi nunca'.",
        "prompt": (
            "Implementa la Linea A del roadmap (metrica honesta) en ai-trader:\n"
            "1) Sustituye el headline per-sample: en backtest/metrics.py y BacktestResult, deja "
            "de usar Calmar como headline_score. Nuevo headline = Sharpe_OOS - lambda_turnover*"
            "turnover - kappa*maxDD (maxDD como PENALIZACION SUAVE, no denominador ni descarte). "
            "Anade turnover (nº trades / dia o rotacion de notional) a PerformanceMetrics.\n"
            "2) En scoring/aggregate.py fija la recompensa/ranking como CVaR@25% (ya se computa) "
            "en lugar de media-lambda*std. Manten mean/std/p25 como reportados.\n"
            "3) Anade baselines como gate obligatorio en el informe: buy&hold BTC, equiponderado "
            "del universo, y SPY. Una estrategia debe batir el mejor baseline para 'aprobar'.\n"
            "4) Anade DSR (Deflated Sharpe Ratio) / PBO sobre la distribucion de scores del "
            "ranking para descontar el overfitting por multiples pruebas.\n"
            "Respeta los principios: determinismo, honestidad estadistica (distribucion, no un "
            "path), tests para cada pieza. Usa .venv\\Scripts\\python.exe para pytest/ruff."
        ),
    },
    {
        "id": "line-c-costs",
        "title": "Linea C - Costes que muerden",
        "line": "C", "status": "pendiente", "impact": "alto", "effort": "medio",
        "why": "El slippage es plano (5 bps) y allow_partial_fills=False llena cualquier "
               "tamano entero: no hay techo de capacidad. 5 bps en un altcoin es ficcion.",
        "prompt": (
            "Implementa la Linea C (costes realistas) en ai-trader:\n"
            "1) En execution/paper.py, sustituye el slippage plano por un modelo funcion de "
            "(spread base del simbolo, volatilidad reciente del activo, tamano/volumen de la "
            "barra). El campo volume de las velas sinteticas hoy es decorativo: usalo como "
            "proxy de liquidez para el impacto de mercado.\n"
            "2) Activa fills parciales (allow_partial_fills) con un techo de capacidad por barra "
            "(p.ej. una fraccion del volumen). Ordenes grandes se llenan parcialmente.\n"
            "3) Anade tests que verifiquen que un altcoin iliquido paga mas slippage que BTC y "
            "que una orden mayor que el techo se llena parcial. Corre la suite completa."
        ),
    },
    {
        "id": "line-d-validation",
        "title": "Linea D - Validacion (CPCV / walk-forward)",
        "line": "D", "status": "pendiente", "impact": "medio", "effort": "medio",
        "why": "El split actual es un unico 70/30 (_resolve_cutoff). Sobre-estima la robustez "
               "y no purga ni embarga entre train y test.",
        "prompt": (
            "Implementa validacion multiventana en ai-trader/backtest: sustituye o complementa "
            "el split unico 70/30 con walk-forward multiventana y, si es viable, CPCV "
            "(Combinatorial Purged Cross-Validation) con purga y embargo entre train y test. "
            "Agrega los headline scores de todas las ventanas en una distribucion robusta. "
            "Anade tests de que no hay fuga temporal entre folds. Determinismo + .venv python."
        ),
    },
    {
        "id": "line-b7-rankcorr",
        "title": "Linea B7 - Validacion rank-corr sintetico vs real",
        "line": "B", "status": "pendiente", "impact": "medio", "effort": "medio",
        "why": "Falta medir que ai_v2 se parece al mercado real. Se difirio a sub-linea; los "
               "stylized-facts objetivo ya estan definidos (autocorr, clustering, colas).",
        "prompt": (
            "Construye un harness de validacion que compare los stylized-facts de la libreria "
            "sintetica ai_v2 contra el historico REAL de cripto (via CCXT, ya integrado en el "
            "proyecto): autocorrelacion de retornos, clustering de volatilidad (autocorr de "
            "|retorno|), indice de cola (exceedances / kurtosis) y correlaciones cruzadas. "
            "Reporta la correlacion de rangos sintetico-vs-real por metrica. Cachea los datos "
            "reales. Anade al dashboard una vista con la comparacion. Determinismo + tests."
        ),
    },
    {
        "id": "line-e-cleanup",
        "title": "Linea E - Limpieza de consistencia",
        "line": "E", "status": "pendiente", "impact": "bajo", "effort": "bajo",
        "why": "MATIC/USDT sigue en el universo pese a estar deslistado en Binance; "
               "TRADING_DAYS_PER_YEAR=365 desanualiza mal la renta variable; el designer usa "
               "temperature=1.0 (diseño no reproducible, mitigado guardando spec.json).",
        "prompt": (
            "Limpieza de consistencia en ai-trader (Linea E):\n"
            "1) Resuelve MATIC/USDT: o se retira del universo sintetico y de config, o se "
            "documenta explicitamente por que se mantiene. Consistencia entre default.toml y "
            "synthetic.toml.\n"
            "2) TRADING_DAYS_PER_YEAR: usa 252 para renta variable y 365 para cripto al "
            "anualizar metricas por clase de activo (hoy es 365 global).\n"
            "3) Verifica que DEFAULT_MODEL del designer resuelve contra la API actual y "
            "documenta que con temperature=1.0 el diseño no es reproducible (se mitiga guardando "
            "spec.json). Tests donde aplique."
        ),
    },
    {
        "id": "paper-trading-view",
        "title": "Vista de paper trading en el dashboard",
        "line": "Dashboard", "status": "placeholder", "impact": "medio", "effort": "bajo",
        "why": "El runner ya opera en paper con estado persistido (JsonStateStore). Falta "
               "exponer equity, posiciones abiertas y PnL en el dashboard.",
        "prompt": (
            "Anade al dashboard (dashboard/build_dashboard.py) una vista de paper trading que "
            "lea el estado del TradingRunner (JsonStateStore): curva de equity, posiciones "
            "abiertas y cerradas, PnL realizado/no-realizado y metricas de riesgo. Renderiza "
            "equity como line chart y posiciones como tabla. Manten el HTML autocontenido."
        ),
    },
    {
        "id": "rl-full-run",
        "title": "Optimizacion CEM completa sobre ai_v2",
        "line": "RL", "status": "pendiente", "impact": "alto", "effort": "medio",
        "why": "El harness CEM (scoring/optimize.py) esta listo pero cada backtest cuesta ~60s "
               "con 35 activos; una corrida completa necesita subsampleo o paralelizacion.",
        "prompt": (
            "Ejecuta y consolida la optimizacion CEM de las dos primitivas sobre ai_v2 "
            "(scoring/optimize.py). Antes: (a) cablea ai_v2 como default, (b) aplica la metrica "
            "de la Linea A si ya esta. Optimiza el rendimiento del backtest o paraleliza la "
            "evaluacion de muestras para que una corrida con subsampleo razonable sea tratable. "
            "Guarda los mejores params por primitiva y su distribucion train/validation, y "
            "vuelca los resultados al dashboard (seccion ranking). Determinismo + tests."
        ),
    },
]


if __name__ == "__main__":
    build()
