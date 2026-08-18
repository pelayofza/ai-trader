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

from ai_trader.backtest.divergence_study import (
    DIVERGENCE_REPORT,
    STATUS_MEASURED,
    load_divergence_report,
)
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
from ai_trader.scoring.signal_study import (
    DEFAULT_LIBRARY_ID as SIGNAL_LIBRARY,
    load_signal_report,
    report_path as signal_report_path,
)
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
from ai_trader.shared.reports import load_report
from ai_trader.shared.instruments import AssetClass
from ai_trader.scoring.weight_study import FAMILIES, NEW_FAMILIES
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
# El linaje gana una cuarta generacion: ai_v4 = ai_v3 + cinco canales de observacion, con
# las MISMAS velas (verificado por SHA). PRIMARY_LIB y RANK_LIB se quedan en ai_v2 por el
# motivo del comentario de arriba, que sobrevive al cambio.
CHANNELS_LIB = "ai_v4"
LIBRARY_LINEAGE = (COMPARE_LIB, PRIMARY_LIB, FIDELITY_LIBRARY, CHANNELS_LIB)


def _blind_themes() -> frozenset[str]:
    """
    Que temas NO alcanzan cobertura en un backtest historico, DERIVADO y no escrito a mano.

    Un tema se puede evaluar hacia atras si sus fuentes con `history_from` medido llegan al
    minimo de cobertura. Calcularlo aqui —en vez de listar dos nombres— hace que el dia que
    una fuente gane profundidad medida, esta vista deje de mentir sola.
    """
    from ai_trader.observation.signal_radar import MIN_SIGNAL_COVERAGE
    from ai_trader.observation.signal_themes import THEMES, effective_denominator
    from ai_trader.signals.catalog import CATALOG

    backtestable = {s.key for s in CATALOG if s.backtestable}
    blind = set()
    for name, spec in THEMES.items():
        denominator = effective_denominator(len(spec.sources), spec.min_sources)
        if len(set(spec.sources) & backtestable) / denominator < MIN_SIGNAL_COVERAGE:
            blind.add(name)
    return frozenset(blind)


BLIND_THEMES = _blind_themes()

# --- scope del ranking de muestra (reducido para que el build sea rapido) -----------
RANK_LIB = "ai_v2"
RANK_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SPY", "QQQ", "GLD", "TLT"]
# Un path por escenario, y no dos, desde que la vista tiene OCHO familias en vez de dos: este
# ranking es una MUESTRA ilustrativa —la evidencia vive en data/transfer/— y el build se paga
# en cada `verify.ps1`, que es el bucle de desarrollo diario. Con los valores anteriores el
# artefacto tardaba ~25 min en regenerarse y la verificacion completa pasaba de ~12 a ~40.
RANK_N_PATHS = 1
RANK_N_SCENARIOS = 4
RANK_WINDOW_DAYS = 300
# Una entrada por familia y no dos: cada una anade ~40 s al build, y el build se paga DOS
# veces en cada `verify.ps1` (una la caracterizacion y otra la regeneracion). Las dos de
# precio conservan ademas su variante, que es lo que hace legible el panel de sobreajuste.
RANK_CONFIGS = [
    ("Momentum (default)", "crypto_momentum", {}),
    ("Mean-reversion (default)", "mean_reversion", {}),
    ("Momentum (rapido)", "crypto_momentum",
     {"fast_sma_window": 5, "slow_sma_window": 20, "breakout_lookback": 3}),
    ("Mean-reversion (estricto)", "mean_reversion",
     {"entry_z": 1.5, "exit_z": 0.2, "lookback": 15}),
    ("Liquidacion (default)", "liquidation_cascade", {}),
    ("Volatilidad (default)", "vol_term_structure", {}),
    ("Calendario (default)", "event_calendar_drift", {}),
    ("Atencion (default)", "attention_ignition", {}),
    ("Flujo (default)", "flow_persistence", {}),
    ("Compuesta (default)", "signal_composite", {}),
]

CHART_SYMBOLS = ["BTC/USDT", "SPY", "GLD"]
CHART_POINTS = 160  # downsample de las series de precio para el JSON

# Puntos de la curva de paper trading que se embeben en el HTML. El diario crece sin
# limite (un ciclo cada 15 minutos) y el dashboard es un fichero autocontenido.
CURVE_POINTS = 400

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


def _pick_indices(total: int, n: int) -> np.ndarray:
    """Hasta `n` indices repartidos por igual, incluyendo SIEMPRE el primero y el ultimo.

    Estaba dentro de `_downsample`, que solo sabe recortar series de numeros. La curva de
    paper trading necesita el mismo recorte pero sobre FILAS (marca de tiempo, PnL,
    exposicion), asi que lo que se comparte es la eleccion de indices, no el formateo."""
    if total <= n:
        return np.arange(total)
    return np.linspace(0, total - 1, n).astype(int)


def _downsample(values: np.ndarray, n: int) -> list[float]:
    return [round(float(v), 2) for v in values[_pick_indices(len(values), n)]]


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
            *_themed_strategies(),
        ],
    }


# Prosa editorial de las seis tematicas. No es derivable del codigo —que mira cada una y con
# que fuentes es una decision, no un atributo—, pero SI lo son los parametros y el tema, asi
# que se derivan. El `raise` de abajo es lo que impide que anadir una familia a la rejilla de
# scoring y olvidarse del dashboard publique una vista que dice "dos" mientras se miden ocho.
THEMED_PROSE: dict[str, dict] = {
    "liquidation_cascade": {
        "name": "Cascada de liquidaciones",
        "regime": "capitulacion",
        "idea": "Compra la capitulacion, salvo que el mapa diga que queda combustible debajo. "
                "El precio ve el agotamiento; la senal ve cuanto notional revienta y a que "
                "distancia.",
        "rules": [
            "Precio estirado |cierre - media| >= k*ATR respecto de su media.",
            "Rango verdadero del dia >= m*ATR: la barra de capitulacion.",
            "Cierre en el extremo del rango (25% inferior para el largo).",
            "Capa: veto si el tono del tema apunta en contra del lado.",
        ],
    },
    "vol_term_structure": {
        "name": "Estructura temporal de volatilidad",
        "regime": "compresion",
        "idea": "Rompe la compresion en la direccion que se esta pagando. La vol realizada se "
                "comprime antes de expandirse; el skew dice hacia donde.",
        "rules": [
            "rv_corta / rv_larga <= umbral, medido en la barra ANTERIOR a la rotura.",
            "Cierre fuera del canal de Donchian de N dias (arriba o abajo).",
            "Stop y objetivo en multiplos de ATR.",
            "Capa: techo de intensidad = trampa de gamma en vencimiento.",
        ],
    },
    "event_calendar_drift": {
        "name": "Deriva de calendario",
        "regime": "deriva",
        "idea": "Sigue el movimiento entre hitos, dosificando por lo que hay en la agenda. El "
                "tema NO dice hacia donde —su tono es ~0 por construccion— sino CUANDO.",
        "rules": [
            "Deriva de N dias dentro de una banda [minimo, maximo].",
            "Ventana corta confirmando el mismo sentido.",
            "Lado = signo de la deriva; la senal no lo toca nunca.",
            "Capa: SOLO intensidad, como piso y como techo.",
        ],
    },
    "attention_ignition": {
        "name": "Ignicion de atencion",
        "regime": "atencion",
        "idea": "Compra el dia en que el minorista se entera. La atencion llega tarde, lenta e "
                "insensible al precio, asi que produce continuacion. Solo largo, por tesis.",
        "rules": [
            "Volumen del dia >= k veces su MEDIANA movil (mediana, no media: colas).",
            "Cierre en el 70% superior del rango del dia.",
            "Precio por encima de su media larga.",
            "Capa: veto por tono (deslistado) y techo de intensidad (atencion saturada).",
        ],
    },
    "flow_persistence": {
        "name": "Persistencia de flujo",
        "regime": "tendencia",
        "idea": "Compra la pausa mientras el dinero sigue entrando. Es el unico tema con tono "
                "de calidad: once de sus doce fuentes tienen polaridad razonada.",
        "rules": [
            "Pendiente de la media larga en un sentido.",
            "Fraccion de dias a favor por encima de un minimo (persistencia).",
            "Precio retrocedido hasta la media sin perderla (<= k ATR).",
            "Capa: el tono puede decidir el lado; piso de intensidad.",
        ],
    },
    "signal_composite": {
        "name": "Compuesto de senales",
        "regime": "senal",
        "idea": "La unica que ve los cinco temas a la vez, y por tanto la unica que cobra la "
                "raiz de la ley fundamental. Ciega es un seguidor de tendencia corriente: toda "
                "su tesis esta en la capa.",
        "rules": [
            "Piso de ATR: que el activo sea operable.",
            "Giro reciente de la media corta: decide CUANDO, no hacia donde.",
            "No perseguir: estiramiento respecto de esa media acotado en ATRs.",
            "Capa: tono = media de los temas LEGIBLES; hacen falta dos de cinco.",
        ],
    },
}


def _themed_strategies() -> list[dict]:
    missing = set(NEW_FAMILIES) - set(THEMED_PROSE)
    if missing:
        raise ValueError(
            f"Familias tematicas sin prosa en el dashboard: {sorted(missing)}. Anadir una "
            "familia a la rejilla y olvidar esta vista tiene que ROMPER el build, no publicar "
            "una vista que dice dos mientras los estudios miden ocho."
        )
    out = []
    for family in NEW_FAMILIES:
        prose = THEMED_PROSE[family]
        strategy = build_strategy(family)
        out.append(
            {
                "id": family,
                "name": prose["name"],
                "regime": prose["regime"],
                "idea": prose["idea"],
                "rules": prose["rules"],
                "params": _params_dict(strategy.config),
                "theme": getattr(strategy, "theme", ""),
                # Lo que decide si su capa se puede evaluar en un backtest historico. Sale de
                # la aritmetica del radar, no de una opinion.
                "blind_in_history": getattr(strategy, "theme", "") in BLIND_THEMES,
            }
        )
    return out


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

    # El regimen preferido de cada primitiva sale del CATALOGO de la vista, no de una tupla
    # escrita aqui: estaba en dos sitios y con ocho familias eso es una divergencia esperando
    # a pasar.
    wanted = {s["id"]: s["regime"] for s in collect_strategies()["strategies"]}
    for strat_type, want in wanted.items():
        preferred = [s["id"] for s in scen if s["regime"] == want]
        # Dos escenarios por familia y no cuatro, por el mismo motivo que RANK_N_PATHS: son
        # ocho familias, el barrido evalua una senal cada dos barras sobre cada (escenario,
        # simbolo), y el chart es ilustrativo.
        scan = (preferred or [s["id"] for s in scen])[:2]
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


def collect_market() -> dict:
    """Capitulo 2.1: la captura de datos REALES.

    No corre nada: son constantes del sistema que opera (universo, proveedor, cache y
    lookback). Es la vista que faltaba -- el dashboard empezaba por el mundo sintetico,
    como si el dato real no existiera, cuando toda la evidencia externa sale de el."""
    from ai_trader.data.cache import CACHE_DIR
    from ai_trader.data.providers.ccxt_crypto import CCXTCryptoConfig
    from ai_trader.synthetic.universe import DEFAULT_UNIVERSE

    config = load_config(ROOT / "config" / "default.toml")
    ccxt_config = CCXTCryptoConfig()
    symbols = list(config.runner.symbols)
    return {
        "symbols": symbols,
        "n_symbols": len(symbols),
        "exchange": ccxt_config.exchange_id,
        "batch": ccxt_config.max_batch_size,
        "timeout_ms": ccxt_config.timeout_ms,
        "lookback_days": config.runner.lookback_days,
        "cache_dir": str(CACHE_DIR).replace("\\", "/"),
        "n_synthetic_assets": len(DEFAULT_UNIVERSE.assets),
        "providers": [
            {"asset_class": "Criptomonedas", "provider": f"ccxt · {ccxt_config.exchange_id}",
             "state": "operado",
             "note": "El universo que se opera. Cotiza 24/7: una barra por día natural."},
            {"asset_class": "Renta variable", "provider": "alpaca", "state": "aparcado",
             "note": "Proveedor implementado y sin estrategia detrás. La clase de activo está "
                     "aparcada a propósito."},
            {"asset_class": "Mercados de predicción", "provider": "polymarket · gamma + CLOB",
             "state": "sin histórico",
             "note": "Precio vivo y libro, pero no hay OHLCV histórico: no se puede backtestear, "
                     "solo capturar hacia adelante."},
        ],
    }


def _usd_es(value: float) -> str:
    """Importe con separador de miles espanol: 1.000 $ y no 1,000 $."""
    return "{:,.0f} $".format(value).replace(",", ".")


def collect_trade() -> dict:
    """Capitulo 3: las constantes que gobiernan UN trade, leidas del config operado.

    Riesgo, coste y capacidad van juntos porque son la misma decision vista en tres
    sitios; separarlos es lo que permitia describir un trade que el sistema no ejecuta."""
    from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY

    config = load_config(ROOT / "config" / "default.toml")
    slippage = config.execution.slippage
    return {
        "starting_equity": DEFAULT_STARTING_EQUITY,
        "fee_rate": config.execution.fee_rate,
        "slippage_bps": config.execution.slippage_bps,
        "max_participation": config.execution.max_participation,
        "vol_coef": slippage.vol_coef,
        "impact_coef": slippage.impact_coef,
        "max_slippage_bps": slippage.max_slippage_bps,
        # Formato espanol para las cifras que se publican tal cual (el JS no puede
        # reformatear un string ya montado).
        "risk": [
            ["Confianza mínima por operación", f"{config.risk.min_confidence_per_trade:.2f}",
             "Que una señal débil abra posición solo porque no había nada mejor."],
            ["Tamaño máximo por posición", _usd_es(config.risk.max_position_size_usd),
             f"Que un trade concentre la cuenta. Con equity conocido manda además la fracción "
             f"de riesgo ({config.risk.risk_fraction_per_trade:.0%} del capital), y el tamaño "
             f"compone."],
            ["Exposición máxima por símbolo",
             _usd_es(config.risk.max_symbol_exposure_usd),
             "Que varias señales del mismo activo se acumulen en una apuesta única."],
            ["Exposición máxima total", _usd_es(config.risk.max_total_exposure_usd),
             "Que el sistema despliegue más de lo que la cuenta soporta."],
            ["Posiciones abiertas simultáneas", str(config.risk.max_open_positions),
             "Que la cartera se convierta en un índice por goteo."],
            ["Pérdida diaria máxima", _usd_es(config.risk.max_daily_loss_usd),
             "Que un mal día siga abriendo posiciones nuevas."],
            ["Stop / objetivo por defecto",
             f"−{config.risk.default_stop_loss_pct:.0f}% / +{config.risk.default_take_profit_pct:.0f}%",
             f"Que una posición quede sin salida definida. La estrategia puede proponer los "
             f"suyos, nunca más lejos de {config.risk.max_stop_distance_pct:.0f}%."],
            ["Vida máxima de una posición", f"{config.runner.max_holding_days} días",
             "Que una posición sin desenlace ocupe exposición indefinidamente. Es el mismo "
             "número que fija la purga de la validación temporal."],
            ["Enfriamiento por símbolo", f"{config.runner.symbol_cooldown_hours} h",
             "Que el sistema reabra lo que acaba de cerrar."],
            ["Operaciones por ciclo", str(config.runner.max_trades_per_cycle),
             "Que un ciclo raro dispare una ráfaga."],
        ],
    }


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
        # Derivado, no literal: un `2` escrito a mano es exactamente el bug que este trabajo
        # destapa —la vista diciendo "dos primitivas" mientras los estudios miden ocho—.
        "n_strategies": len(FAMILIES),
        "n_themed_strategies": len(NEW_FAMILIES),
        "n_blind_themes": len(BLIND_THEMES),
        "n_own_features": len(OWN_ASSET_FEATURES),
        "n_regime_features": len(REGIME_FEATURES),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "commit_count": _git("rev-list", "--count", "HEAD"),
        "generated_at": _git("log", "-1", "--format=%cd", "--date=short"),
    }


def collect_calibration(path: Path = CALIBRATION_REPORT) -> dict | None:
    """Evidencia del estudio que fija lambda y kappa (data/calibration).

    Se LEE del informe publicado; no se recalcula. El estudio son cientos de backtests
    reales y el dashboard debe seguir siendo regenerable en minutos."""
    report = load_calibration_report(ROOT / path)
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


THEMES_REPORT = Path("data") / "themes" / "report.json"
# La familia que se admitio DESPUES, medida aparte. Correrla sola es legitimo porque el estudio
# compara cada familia CONSIGO MISMA: ninguna cifra depende de que otras familias corran.
THEMES_EXTRA = (Path("data") / "themes" / "report_vol_term_structure.json",)


def _merge_skipped(loaded, measured_families: set, coverage: dict) -> list[dict]:
    """Las familias declaradas NO evaluables, de todos los informes y con su numero.

    Se acumulan de todos y no se toman del mas nuevo porque un informe de una sola familia no
    declara omitidas —no evaluo a las demas—, y quedarse con su lista vacia borraria la unica
    exclusion real del estudio. Y se enriquecen con la cobertura MEDIDA porque el informe mas
    antiguo se genero antes de que se midiera: declaraba la exclusion con una frase derivada del
    catalogo, sin cifra que comprobar.
    """
    out: dict[str, dict] = {}
    for _, rep in loaded:
        for skip in rep["plan"]["families_skipped"]:
            if skip["family"] in measured_families:
                continue  # se acabo midiendo en otra corrida: ya no es una exclusion
            entry = dict(skip)
            stat = coverage.get(entry.get("theme", ""))
            if stat and entry.get("max_coverage") is None:
                entry["max_coverage"] = stat["max_coverage"]
                entry["readable_share"] = stat["readable_share"]
                entry["reason"] = (
                    f"el tema '{entry['theme']}' NUNCA alcanza {stat['threshold']:.2f} de "
                    f"cobertura en el archivo: su maximo medido es "
                    f"{stat['max_coverage']:.3f} y es legible en el "
                    f"{100 * stat['readable_share']:.1f}% de las sondas"
                )
            # Gana la entrada CON numero: la que trae medicion describe mejor el mismo hueco.
            if entry["family"] not in out or entry.get("max_coverage") is not None:
                out[entry["family"]] = entry
    return list(out.values())


def collect_themes() -> dict | None:
    """La capa de senal contra ARCHIVO REAL capturado (data/themes).

    Es la unica evidencia del sistema donde la capa tematica se enciende sobre datos de
    mercado de verdad y no sobre un canal sintetico. La comparacion es PAREADA —misma
    configuracion, misma ventana, mismas barras, y lo unico que cambia es el umbral de la
    puerta y si el archivo llega al motor—, asi que la diferencia no arrastra el efecto de
    haber elegido otra estrategia.

    SE FUNDEN VARIOS INFORMES, y hay que explicar las dos mitades de por que:

    - Las FAMILIAS se acumulan y no se recalculan. `vol_term_structure` se admitio despues,
      cuando la cobertura medida contradijo al catalogo, y se corrio aparte. Fundir en vez de
      repetir las 160 unidades ya medidas es correcto por lo mismo que permite correr una
      familia sola: cada veredicto es interno a su familia, y ninguna cifra depende de que
      otras corran.
    - Los METADATOS del tema (cobertura medida, temas ciegos, familias omitidas) se toman del
      informe MAS NUEVO que los traiga, no del primero. El primero se genero antes de que la
      evaluabilidad se midiera, asi que declara sus exclusiones con un motivo derivado del
      catalogo —y para `vol_surface` ese motivo resulto ser FALSO—. Publicar la lista vieja
      seria publicar esa frase otra vez.
    """
    sources = [THEMES_REPORT, *THEMES_EXTRA]
    loaded = [(path, load_report(ROOT / path)) for path in sources]
    loaded = [(path, rep) for path, rep in loaded if rep]
    if not loaded:
        logger.warning("Sin informe tematico: el panel de la capa de senal saldra vacio")
        return None

    families: list[dict] = []
    seen: set[str] = set()
    for _, rep in loaded:
        for fam in rep["families"]:
            if fam["family"] not in seen:
                families.append(fam)
                seen.add(fam["family"])

    # El informe que MIDE la cobertura manda sobre el que la derivaba del catalogo.
    measured = [(path, rep) for path, rep in loaded
                if rep["plan"]["themes"].get("measured")]
    meta_path, meta = (measured or loaded)[-1]
    plan = meta["plan"]
    skipped = _merge_skipped(loaded, seen, plan["themes"].get("measured") or {})

    return {
        "report_path": str(THEMES_REPORT).replace("\\", "/"),
        "metadata_from": str(meta_path).replace("\\", "/"),
        "coverage_is_measured": bool(plan["themes"].get("measured")),
        "families": families,
        "families_skipped": skipped,
        "symbols": plan["symbols"],
        "windows": plan["windows"],
        "themes": plan["themes"],
        "sources_loaded": plan["sources_loaded"],
        "min_paired_windows": plan["min_paired_windows"],
        "arms": plan["arms"],
        "n_failed_units": sum(rep["n_failed_units"] for _, rep in loaded),
        "caveats": meta["caveats"],
    }


def collect_transfer(library: str = TRANSFER_LIBRARY) -> dict | None:
    """¿Ordena el mundo sintetico las estrategias como el real? (data/transfer).

    Es la pregunta que la fidelidad NO responde: un generador puede clavar las colas y
    ordenar al reves. Se LEE del informe publicado; no se recalcula. Son 208 unidades de
    15 ventanas de backtest real cada una y el dashboard tiene que seguir siendo
    regenerable en minutos."""
    report = load_transfer_report(ROOT / transfer_report_path(library))
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


def collect_validation(path: Path = VALIDATION_REPORT) -> dict | None:
    """Comparacion medida entre el corte unico 70/30 y la validacion multiventana
    (data/validation).

    Se LEE del informe publicado; no se recalcula. Cada unidad del estudio son ~20
    ventanas de backtest real y el dashboard tiene que seguir siendo regenerable en
    minutos."""
    report = load_validation_report(ROOT / path)
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


def collect_activity(library: str = TRANSFER_LIBRARY) -> dict | None:
    """El suelo de actividad y la evidencia con la que se eligio (data/activity).

    Se LEE del informe publicado; no se recalcula. El estudio se apoya en las 208 unidades
    del estudio de transferencia (15 ventanas de backtest real cada una) y el dashboard
    tiene que seguir siendo regenerable en minutos. El informe ya trae la regla aplicada y
    el umbral elegido: aqui no se decide nada, se reempaqueta para el render."""
    report = load_activity_report(ROOT / activity_report_path(library))
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


def collect_signal_channel(library: str = SIGNAL_LIBRARY) -> dict | None:
    """El break-even del IC: ¿desde que capacidad predictiva paga una senal? (data/signal_channel).

    Se LEE del informe publicado; no se recalcula. Son 640 unidades de 15 ventanas de
    backtest cada una —tres horas de CPU— y el dashboard tiene que seguir siendo
    regenerable en minutos. Aqui no se decide nada: el criterio de lectura viene declarado
    en el propio informe (`criterion`), escrito en el codigo ANTES de correrlo."""
    report = load_signal_report(ROOT / signal_report_path(library))
    if not report:
        logger.warning("Sin informe del canal sintetico: el panel de break-even saldra vacio")
        return None

    plan = report["plan"]
    certification = {c["cell_id"]: c for c in report["channel_certification"]}
    return {
        "library": plan["library_id"],
        "symbols": plan["symbols"],
        "grid": plan["grid"],
        "sweep": plan["sweep"],
        "synthetic": plan["synthetic"],
        "validation": plan["validation"],
        "criterion": report["criterion"],
        "break_even": report["break_even"],
        "value_of_information": report["value_of_information"],
        "gate_cost": report["gate_cost"],
        "reproduction": report["reproduction"],
        "baseline_invariance": report.get("baseline_invariance"),
        "determinism": report.get("determinism"),
        "n_failed_units": report["n_failed_units"],
        # Una fila por celda: lo declarado, lo medido y lo que decidio.
        "rows": [
            {
                "cell_id": c["cell_id"],
                "arm": c["arm"],
                "rho": c["rho"],
                "lead_days": c["lead_days"],
                "expected_ic": c["expected_ic"],
                "measured_ic": certification.get(c["cell_id"], {}).get("ic_median"),
                "measured_ac1": certification.get(c["cell_id"], {}).get("ac1_median"),
                "past_leak": certification.get(c["cell_id"], {}).get("past_leak_median"),
                "selected": c["selected"],
                "reward_train": c["selected_reward_train"],
                "reward": c["selected_reward_validation"],
                "baseline": c["baseline_reward_validation"],
                "margin": c["margin"],
                "beats": c["beats"],
                "n_beating": c["n_beating_baseline"],
                "n_rankable": c["n_rankable"],
                "fell_back": c.get("selection_fell_back", False),
                "n_configs": c["n_configs"],
                "activity": c["selected_activity_validation"],
                "trades_per_window": c["trades_per_window"],
            }
            for c in report["cells"]
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
    La plataforma de ingesta de senales: catalogo, mapeo de entidades, archivo crudo,
    la PROFUNDIDAD MEDIDA de cada fuente y —desde el 2026-08-12— el RADAR que las lleva a
    la decision.

    Todo se lee de disco y del registro de mediciones (`data/signals/history_depth.json`,
    `data/signals/event_pool.json`); nada de esto toca red al generar el dashboard. Las
    cifras que hay que mirar no son cuantas fuentes hay declaradas, sino cuantas tienen
    `history_from` MEDIDO —esas son las unicas que pueden entrar en un backtest— y cuantos
    EVENTOS POOLED hay detras de las de evento, que es lo que sustituyo a la creencia de que
    eran "muestras de decenas".
    """
    from ai_trader.observation.signal_radar import (
        ASSET_SIGNAL_FEATURES,
        MARKET_SIGNAL_FEATURES,
        MIN_SIGNAL_COVERAGE,
        POLARITY,
        is_market_scoped,
    )
    from ai_trader.signals.adapters.treasuries import (
        COHORT_REPORT,
        load_cohort_report,
    )
    from ai_trader.signals.audit import audit_archive, audit_entities
    from ai_trader.signals.capture import (
        CAPTURE_REPORT,
        connect_adapters,
        entities_for,
        load_capture_report,
    )
    from ai_trader.signals.catalog import CATALOG, catalog_summary
    from ai_trader.signals.depth import DEPTH_LEDGER, load_ledger
    from ai_trader.signals.events import (
        EVENT_POOL_REPORT,
        is_event_source,
        is_price_map_source,
        load_pool_report,
    )
    from ai_trader.signals.liquidity import ADV_LEDGER, liquidity_summary
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

    pool = load_pool_report(ROOT / EVENT_POOL_REPORT) or {}
    pool_by_key = pool.get("sources") or {}
    # La cohorte de tesorerias cotizadas. Es la unica fuente COMPUESTA del catalogo y la
    # unica cuyo N no sale del recuento pooled de `events.py`: ahi la unidad es el evento de
    # una entidad y aqui es la OBSERVACION DE COMPANIA agrupada sobre las doscientas.
    dat = load_cohort_report(ROOT / COHORT_REPORT) or {}

    return {
        "summary": {
            **catalog_summary(),
            "n_connected": len(connected),
            "n_measured": sum(1 for r in depth_by_key.values() if r.get("first_day")),
            "pooled_events": pool.get("pooled_events_total"),
        },
        "universe": universe,
        "normalization": normalization_spec(),
        # El radar: como llegan las veintinueve fuentes a una decision, y con que reglas.
        "radar": {
            "asset_features": list(ASSET_SIGNAL_FEATURES),
            "market_features": list(MARKET_SIGNAL_FEATURES),
            "min_coverage": MIN_SIGNAL_COVERAGE,
            "n_market_sources": sum(1 for s in CATALOG if is_market_scoped(s)),
            "n_asset_sources": sum(1 for s in CATALOG if not is_market_scoped(s)),
            "n_event_sources": sum(1 for s in CATALOG if is_event_source(s)),
            "n_price_map_sources": sum(1 for s in CATALOG if is_price_map_source(s)),
            "n_continuous_sources": sum(
                1 for s in CATALOG if not (is_event_source(s) or is_price_map_source(s))
            ),
            "n_with_polarity": len(POLARITY),
            "event_spec": pool.get("spec"),
        },
        "event_pool": pool_by_key,
        "price_maps": (pool.get("price_maps") or {}).get("sources") or {},
        # Tesorerias cotizadas: la distribucion de mNAV, su N y —lo que la hace auditable—
        # por que se cayo cada companıa que no entro.
        "dat": {
            "generated_at": (dat.get("generated_at") or "")[:10] or None,
            "companies": dat.get("companies"),
            "companies_examined": dat.get("companies_examined"),
            "pooled_observations": dat.get("pooled_observations"),
            "rows": dat.get("rows"),
            "median_lag_days": dat.get("median_disclosure_lag_days"),
            "policy": dat.get("policy") or {},
            "assets": {
                asset: {
                    "n_companies": block.get("n_companies"),
                    "observations": block.get("observations"),
                    "latest": block.get("latest"),
                    # Las companias, para que la distribucion se pueda auditar una a una:
                    # sin esto, "el 33% por debajo de 1" es un numero sin nadie detras.
                    "companies": block.get("companies") or [],
                }
                for asset, block in (dat.get("assets") or {}).items()
            },
            "rejections": dat.get("rejections") or {},
        },
        # El ADV: la cifra que dice si una senal buena admite tamano. Ver signals/liquidity.py.
        "liquidity": liquidity_summary(ROOT / ADV_LEDGER),
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
                # Como entra al espacio de observacion. La codificacion la decide la
                # CADENCIA salvo excepcion declarada (el mapa de precios), y el bloque
                # (mercado o activo) sale del alcance del catalogo.
                "encoding": (
                    "evento" if is_event_source(s)
                    else "mapa de precios" if is_price_map_source(s)
                    else "continua"
                ),
                "block": "mercado" if is_market_scoped(s) else "activo",
                # El ADV tipico de las entidades donde esta senal existe, MEDIDO. None en
                # las de alcance mercado, cuyo eje no es un activo: ahi `adv_note` lo dice.
                "typical_adv_usd": s.typical_adv_usd,
                "adv_note": s.adv_note,
                "snapshots": (
                    ((pool.get("price_maps") or {}).get("sources") or {}).get(s.key) or {}
                ).get("snapshots"),
                "pooled_events": (pool_by_key.get(s.key) or {}).get("pooled_events"),
                "announced": (pool_by_key.get(s.key) or {}).get("announced"),
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


# Regla fija que se anade al FINAL de todos los prompts del roadmap. Vive aqui una sola
# vez y la inyecta `collect_roadmap()`: repetirla en las veinte entradas seria anadir
# duplicacion para combatir la duplicacion, y ademas garantizaria que veinte copias se
# desincronicen en cuanto se retoque el texto.
#
# Va al final y no al principio porque los prompts ya cierran con los recordatorios
# operativos ("Tests + ruff. Regenera dashboard y docs"), y esta es del mismo genero: una
# precondicion del repo, no contexto de la tarea.
REUSE_RULE = """

--- ANTES DE ESCRIBIR CODIGO NUEVO (regla fija de este repo) ---

Declara estas tres cosas POR ESCRITO, en tu respuesta, antes de tocar el primer fichero:

  1. DONDE VA. Modulo y funcion exactos, y por que ahi y no en otro sitio.

  2. QUE YA EXISTE QUE SE LE PARECE. Busca ANTES de escribir, no despues:
       grep -rn "<el concepto>" src/          y tambien por los nombres que ibas a usar
     Mira siempre estos cuatro, que es donde vive lo reutilizable:
       src/ai_trader/shared/        barras, indicadores, reloj, instrumentos, senales
       src/ai_trader/backtest/metrics.py     metricas de resultado
       src/ai_trader/synthetic/fidelity.py   stylized facts, autocorrelacion, colas
       src/ai_trader/scoring/aggregate.py    la recompensa (CVaR)
     Di que encontraste, aunque no encaje. "No busque" no es una respuesta valida.

  3. POR QUE NO LO REUTILIZAS. Si no reutilizas lo que encontraste, da el motivo.
     "Es parecido pero no igual" NO es un motivo: di QUE difiere exactamente y por que
     parametrizar lo que ya existe seria peor que tener dos copias.

Si acabas con un cuerpo identico a otro que ya existe, la salida por defecto es extraerlo
a una funcion comun y dejar los nombres antiguos como ALIAS QUE DELEGAN, para no romper
ninguna llamada. Mueve el cuerpo TAL CUAL, sin reescribirlo: si tocas calculo, el orden de
las operaciones en coma flotante tiene que ser el mismo, y los golden de tests/golden/ son
la prueba de que lo fue.

POR QUE ESTA REGLA: la auditoria del 2026-08-12 (ver DEBT_AUDIT.md y DEBT_BACKLOG.md)
encontro 21 grupos de funciones con el cuerpo IDENTICO, ~135 lineas. Los peores no eran
utilidades sino calculo publicado: el CVaR que DEFINE LA RECOMPENSA esta escrito tres
veces (scoring/aggregate.py:103, scoring/activity_study.py:104,
scoring/transfer_study.py:654) y el cargador de informes, seis. Ninguna de esas copias la
detecta un test, porque cada una tiene los suyos: corregir una y no las otras deja al
sistema puntuando con dos definiciones distintas de la misma metrica."""


def collect_paper() -> dict:
    """
    La vista del paper trading en vivo. DOS fuentes reales y ninguna llamada de red.

    - `data/live/cycles.jsonl` (el diario): la pelicula. De ahi salen la curva marcada a
      mercado, los rechazos del riesgo por familia y el deslizamiento REALMENTE cobrado.
    - `data/runtime_state.json` (el estado): la foto. De ahi salen las posiciones
      cerradas con su PnL neto, que es el registro autoritativo aunque el diario no
      exista todavia (o se haya perdido).

    Las posiciones ABIERTAS se leen del ultimo ciclo del diario y no del estado, porque
    el estado no guarda precio de marca: marcarlas aqui obligaria a que generar el
    dashboard tocara la red, y entonces el dashboard dejaria de ser reproducible.

    Nunca devuelve None: con el sistema recien arrancado devuelve `n_cycles = 0` y la
    vista dice "sin ciclos registrados" en vez de romperse.
    """
    from ai_trader.app.journal import DEFAULT_JOURNAL_PATH, CycleJournal, journal_summary
    from ai_trader.app.state_store import DEFAULT_STATE_PATH, JsonStateStore

    config = load_config(ROOT / "config" / "default.toml")
    journal = CycleJournal(ROOT / DEFAULT_JOURNAL_PATH)
    records = journal.read()
    summary = journal_summary(records)

    state = JsonStateStore(ROOT / DEFAULT_STATE_PATH).load()
    positions = state.get("positions", [])
    closed = [p for p in positions if not p.is_open]

    last = records[-1] if records else {}
    limits = config.risk

    # La curva tiene un punto por ciclo: 96 al dia con el intervalo de 900 s, unos 35.000
    # al ano. Se recorta a 400 puntos para el grafico -mas no se distinguen en pantalla-;
    # las cifras (caida maxima, PnL, comisiones) salen de la curva ENTERA.
    curve = summary["curve"]
    thinned = [curve[i] for i in _pick_indices(len(curve), CURVE_POINTS)]

    return {
        "journal_path": str(DEFAULT_JOURNAL_PATH).replace("\\", "/"),
        "state_path": str(DEFAULT_STATE_PATH).replace("\\", "/"),
        "cycle_interval_seconds": _cycle_interval_seconds(),
        "n_shards": len(journal.shards()),
        "summary": {k: v for k, v in summary.items() if k != "curve"},
        "curve": [
            {
                "t": (p["timestamp"] or "")[:16].replace("T", " "),
                "net": round(p["net_pnl_usd"], 2),
                "realized": round(p["realized_pnl_usd"], 2),
                "exposure": round(p["exposure_usd"], 2),
                "open": p["open_positions"],
            }
            for p in thinned
        ],
        "open_positions": last.get("opened", []),
        "closed_positions": [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "strategy_id": p.strategy_id,
                "size": p.size,
                "entry_price": p.entry_price,
                "exit_price": p.exit_price,
                "fees_usd": round(p.total_fees_usd, 4),
                "realized_pnl_usd": round(p.realized_pnl or 0.0, 2),
                "close_reason": p.close_reason,
                "opened_at": p.opened_at.isoformat()[:10],
                "closed_at": p.closed_at.isoformat()[:10] if p.closed_at else None,
            }
            # Las mas recientes primero, y sin tope: son decenas al ano, no miles.
            for p in sorted(closed, key=lambda p: p.closed_at or p.opened_at, reverse=True)
        ],
        "limits": {
            "max_open_positions": limits.max_open_positions,
            "max_total_exposure_usd": limits.max_total_exposure_usd,
            "max_position_size_usd": limits.max_position_size_usd,
            "max_daily_loss_usd": limits.max_daily_loss_usd,
        },
        "is_paused": bool(state.get("is_paused", False)),
        "n_watched_markets": len(config.runner.prediction_watchlist),
        "watchlist": last.get("watchlist", []),
    }


def collect_divergence() -> dict | None:
    """
    La divergencia live-vs-backtest (data/live/divergence.json).

    Se LEE del informe publicado, no se recalcula: medirla re-simula el periodo entero
    del diario con el motor de backtest, y el dashboard tiene que seguir siendo
    regenerable sin volver a correr un estudio.

    Devuelve algo tambien cuando el estudio dice que NO hay potencia, y eso es lo
    importante de esta vista mientras el diario sea joven: "faltan 28 dias" es una
    afirmacion medida, con su fecha, y sustituye a la prosa de "necesita meses" que
    estaba escrita a mano en la plantilla y no podia equivocarse porque no decia nada.
    """
    report = load_divergence_report(ROOT / DIVERGENCE_REPORT)
    if not report:
        logger.warning("Sin informe de divergencia: el panel del capitulo 5 saldra vacio")
        return None

    measured = report.get("status") == STATUS_MEASURED
    out = {
        "status": report["status"],
        "measured": measured,
        "journal": report["journal"],
        "power": report["power"],
        "thresholds": report["plan"]["thresholds"],
        "reference_cost_bps": report["plan"]["reference_cost_bps"],
        "cycle_interval_seconds": report["plan"].get("cycle_interval_seconds"),
        "report_path": str(DIVERGENCE_REPORT).replace("\\", "/"),
        "generated_at": report["generated_at"][:10],
    }
    if not measured:
        return out

    price = report["fill_price"]
    return {
        **out,
        "resimulation": report["resimulation"],
        "stages": report["decisions"]["stages"],
        "coverage": report["decisions"]["coverage"],
        "total_bps": price.get("total_bps"),
        "components": price.get("components"),
        "n_repriced": price.get("n_repriced", 0),
        "decomposition_ok": price.get("decomposition_ok"),
        "cost": report["cost"],
        "latency": report["latency"],
        "verdict": report["verdict"],
        "ceiling": report["ceiling"],
    }


def _cycle_interval_seconds() -> int | None:
    """El intervalo del ciclo automatico, leido de donde vive. Si el paquete de Telegram
    no esta instalado el dashboard se genera igual: es una cifra de contexto, no un dato
    del que dependa ninguna vista."""
    try:
        from ai_trader.bots.telegram_bot import AUTO_CYCLE_INTERVAL_SECONDS

        return AUTO_CYCLE_INTERVAL_SECONDS
    except Exception:  # noqa: BLE001
        return None


def collect_roadmap() -> list[dict]:
    """Evoluciones pendientes, ordenadas por criticidad, con prompt para Claude Code.

    A cada prompt se le pega `REUSE_RULE` al vuelo. Se hace aqui y no en el texto de cada
    entrada para que la regla exista UNA sola vez en el repo; y sobre copias
    (`{**r, ...}`) para no mutar `ROADMAP`, que si no acumularia la regla en cada llamada.
    """
    from ai_trader.synthetic import retrofit  # noqa: F401  (asegura que el modulo existe)
    ranks = [r["rank"] for r in ROADMAP]
    if sorted(ranks) != list(range(1, len(ROADMAP) + 1)):
        raise ValueError(f"Los rangos del roadmap deben ser 1..N sin huecos: {sorted(ranks)}")
    groups = {r["group"] for r in ROADMAP}
    unknown = groups - {g["key"] for g in ROADMAP_GROUPS}
    if unknown:
        raise ValueError(f"Grupos de roadmap desconocidos: {sorted(unknown)}")
    return [
        {**row, "prompt": row["prompt"] + REUSE_RULE}
        for row in sorted(ROADMAP, key=lambda r: r["rank"])
    ]


def build() -> None:
    store = SyntheticStore(ROOT / "data" / "synthetic")
    logger.info("Captura de datos reales y constantes del trade...")
    market, trade = collect_market(), collect_trade()
    logger.info("Recolectando datos sinteticos...")
    synthetic = collect_synthetic(store)
    logger.info("Stylized facts ai_v1 vs ai_v2...")
    facts = stylized_facts(store)
    logger.info("Fidelidad sintetico vs real (informe publicado)...")
    fidelity = collect_fidelity()
    logger.info("Transferencia de ranking real vs sintetico (informe publicado)...")
    transfer = collect_transfer()
    logger.info("Descomposicion por sesion horaria (informe publicado)...")
    logger.info("Divergencia live-vs-backtest (informe publicado)...")
    logger.info("Suelo de actividad del ranking (informe publicado)...")
    logger.info("Break-even del IC: barrido de rho (informe publicado)...")
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
        "market": market,
        "trade": trade,
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
        "signal_channel": collect_signal_channel(),
        # La rejilla de OCHO familias, AL LADO de la congelada de dos y no en su lugar: lo
        # publicado con dos primitivas sigue siendo cierto sobre lo que midio, y sustituirlo
        # haria imposible ver que cambio al ampliar la rejilla y que cambio al ampliar el mundo.
        "themes": collect_themes(),
        "calibration_v4": collect_calibration(CALIBRATION_REPORT.with_name("report_ai_v4.json")),
        "validation_v4": collect_validation(VALIDATION_REPORT.with_name("report_ai_v4.json")),
        "transfer_v4": collect_transfer(CHANNELS_LIB),
        "activity_v4": collect_activity(CHANNELS_LIB),
        "signal_channel_v4": collect_signal_channel(CHANNELS_LIB),
        "signals_platform": signals_platform,
        "paper": collect_paper(),
        "divergence": collect_divergence(),
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
# ACTUALIZACION 2026-08-12 (2): EL RADAR UNIFICADO ESTA HECHO, y lo primero que hizo fue
# tumbar la creencia que sostenia el diseno anterior. Las seis fuentes de evento tienen
# adaptador y la sonda las midio: 463 ajustes de dificultad desde 2009 (321 dentro de la
# ventana de doce anos que pide la sonda), 621 hacks fechados desde 2016, el calendario del
# FOMC desde 2017. "Muestras de decenas" era falso por un factor de diez, y donde SI es
# corta la muestra la razon medida es otra: el endpoint de unlocks de DefiLlama pasa a ser
# de pago (402) y beaconcha.in pide credencial (401). 917 eventos pooled publicados en
# data/signals/event_pool.json.
#
# Las diecisiete fuentes llegan hoy a la decision por UNA sola via
# (observation/signal_radar.py): seis numeros —tono, intensidad y cobertura, por activo y de
# mercado— con las continuas normalizadas por las dos z y las de evento codificadas aparte
# (dias-al-evento ACOTADO, magnitud sobre su escala declarada, ventana activa), porque una z
# contra una serie que es 99% ceros no significa nada. La cobertura es la feature que
# distingue "no hay evento" de "no se de eventos", y con ella el invariante: ninguna puerta
# bloquea por falta de datos. Nada entro en search_space y la COMPUERTA se cumplio:
# validate_multiwindow devuelve los scores publicados en units_ai_v3.json, identicos hasta el
# ultimo decimal, en las cinco unidades reproducidas.
#
# De paso se cerro el hueco del regimen en vivo (main.py adjunta los dos ensambladores con un
# Mapping perezoso sobre MarketDataService, sin tocar una linea de regime.py) y NO se
# construyo ningun veto: nada impide operar un activo sancionado o deslistado, y esa guarda
# sigue como entrada propia del roadmap. Lo que esta evolucion NO cierra sube al puesto 2: sin
# el break-even de rho no hay test de falsacion, asi que una feature de muestra corta puede
# estar sobreajustada y el sistema no tiene forma de saberlo.
#
# ACTUALIZACION 2026-08-13: EL BREAK-EVEN DEL IC YA ESTA MEDIDO, y con el se cierra el unico
# hueco que el radar de senales dejo declarado por escrito: hoy el sistema SI puede falsar
# que una feature este aportando algo. `scoring/signal_study.py` barre la capacidad
# predictiva de un canal de observacion sintetico -no la senal, el CANAL: cinco numeros
# interpretables- sobre las MISMAS barras, con las 16 configuraciones publicadas y CPCV de
# 15 ventanas. 640 unidades, 3,1 horas, cero fallidas, evidencia en data/signal_channel/.
#
# EL RESULTADO: el break-even esta POR ENCIMA de rho = 0,20 (margen -0,018 en el extremo de
# la rejilla). Un IC diario SOSTENIDO de 0,20 es enorme -la referencia habitual en datos
# alternativos esta un orden de magnitud por debajo, aunque eso es LITERATURA y no una
# medicion de este repositorio: el rho de nuestras diecisiete fuentes sigue sin medir-. Asi
# que la lectura no es "hacen falta senales mejores" sino que el cuello de botella es el
# USO: una PUERTA BINARIA sobre el tono tira toda la informacion salvo un bit, y ademas
# cuesta -1,02 puntos de recompensa por si sola (celda sin canal contra rho=0). Eso reordena
# lo que viene despues: antes que acoplar la senal al estado latente conviene preguntarse si
# el consumo correcto es una puerta o un input continuo del sizing.
#
# LOS TRES CONTROLES SALIERON COMO TENIAN QUE SALIR, que es lo que hace legible el numero:
# (i) rho=0 -el grupo de control- NO bate al baseline, asi que lo medido no es el AR(1) del
# ruido ni el efecto de operar menos; (ii) el canal ENTREGA lo declarado (IC medido 0,004 /
# 0,054 / 0,106 / 0,207) y no correlaciona con retornos ya realizados (0,057); (iii) la
# celda sin canal reproduce 128 unidades de units_ai_v3.json SCORE A SCORE, de modo que la
# costura del canal no movio nada del motor. El valor de la informacion es monotono en rho
# (+0,11 / +0,89 / +1,15 sobre el control), que es la comprobacion de que el instrumento
# mide lo que dice medir.
#
# Lo que este estudio NO contesta, y por eso no se declara cerrado el problema: cuanto rho
# tiene de verdad cada una de las diecisiete fuentes. Eso se mide en el sustrato REAL, con
# la profundidad que la captura vaya comprando dia a dia.
#
# ACTUALIZACION 2026-08-12: las dos evoluciones que ocupaban los puestos 1 y 2 -las senales
# mecanicas como elegibilidad, y el radar de features con su cableado- se FUSIONAN en una
# sola, y de paso cambian de contenido. El motivo no es de agenda sino de diseno: la
# separacion en dos puertas se apoyaba en un tamano de muestra declarado y nunca medido (ver
# "UNA SOLA VIA" mas abajo), y mantener dos caminos distintos hacia la decision obligaba a
# elegir el camino de cada fuente ANTES de saber cuanta historia tiene. Ahora el orden es el
# contrario: se mide la profundidad, y todas las features fluyen igual al motor. En el mismo
# movimiento entra una evolucion NUEVA -el canal de observacion sintetico con barrido de rho-
# que es la que traera el test de falsacion que esta fusion deja pendiente.
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
# UNA SOLA VIA (decision del 2026-08-12, sustituye a las "dos puertas"). Durante una fase el
# plan fue bifurcar: las senales MECANICAS (unlocks, colas de staking, mNAV<1, mapas de
# liquidacion) entrarian como ELEGIBILIDAD en el runner -una guarda que veta operar- y solo
# las ESTADISTICAS como features. Se retira, y el motivo es que la defensa que la sostenia
# -"muestras de decenas, un CEM suelto sobre catorce observaciones construye una estrategia
# preciosa y falsa"- descansaba en un numero QUE NADIE HA MEDIDO. Es una afirmacion de
# folleto, de la misma clase que los history_from que la sonda tuvo que corregir uno por uno,
# y donde se puede comprobar a mano no se sostiene: el ajuste de dificultad de Bitcoin son
# ~900 epocas reconstruibles desde las cabeceras de bloque.
#
# Asi que TODAS las senales fluyen al mismo sitio: features normalizadas al espacio de
# observacion, en backtest Y en vivo, con la codificacion como unica diferencia (evento
# discreto -> dias-al-evento acotado y magnitud sobre float/ADV; serie continua -> las dos
# varas de normalize.py). Lo que reemplaza a la puerta son dos cosas medibles y una que
# falta, y la que falta se declara: (i) NINGUNA feature entra en search_space -- los umbrales
# son constantes declaradas, asi que la huella de las 16 configuraciones publicadas no se
# mueve; (ii) subir N por POOLING del evento normalizado, y medir la profundidad del Tier A
# con la sonda en vez de declararla; (iii) lo que falta es el test de falsacion -- el
# break-even de rho, con rho=0 como control, que es la evolucion 'Canal de observacion
# sintetico'. Hasta que exista, el sistema puede sobreajustar una feature de muestra corta y
# no tiene forma de saberlo.
#
# Y NO HAY VETO: se elimina el concepto de puerta de elegibilidad por senales. Consecuencia
# asumida y escrita, no silenciada: nada impide hoy abrir posicion en un activo sancionado o
# deslistado, y esa guarda -operativa, no de alfa- queda como entrada propia del roadmap.
#
# FOCO: cripto. Renta variable y mercados de prediccion quedan en segundo plano de forma
# EXPLICITA (grupo 'segundo-plano'), no por olvido: toda la evidencia empirica del repo
# -fidelidad contra Binance, calibracion de pesos, estudio de validacion- es cripto, y la
# pata de renta variable no tiene ni un solo dato real detras.

ROADMAP_GROUPS = [
    {
        "key": "ahora",
        "title": "Ahora",
        "subtitle": "Mantener corriendo el PAPER TRADING -que ya deja diario auditable por "
                    "ciclo- porque compra lo único que no se puede comprimir después: tiempo de "
                    "calendario. El test de falsación que faltaba ya está hecho (break-even del "
                    "IC: hace falta rho > 0,20 para batir al baseline con una puerta binaria), y "
                    "lo que abre no es «mejores señales» sino usarlas de otra forma.",
    },
    {
        "key": "despues",
        "title": "Despues",
        "subtitle": "Ampliar el espacio de inputs en los dos mundos y afinar el rigor del juez. Es "
                    "lo que convierte el 'no transfiere' de la vista Ordenación en una conclusión "
                    "en vez de un artefacto del instrumento con el que se midió.",
    },
    {
        "key": "no-prioritario",
        "title": "No prioritario",
        "subtitle": "Trabajo legítimo que no se aborda todavía: añadir candidatos a un juez en el "
                    "que aún no se confía solo multiplica el problema de múltiples pruebas.",
    },
    {
        "key": "segundo-plano",
        "title": "Segundo plano (no cripto)",
        "subtitle": "Renta variable y mercados de predicción, aparcados a propósito hasta que el "
                    "bucle cripto -sintético, real, paper- esté cerrado y medido.",
    },
]

ROADMAP = [
    {
        "id": "paper-trading-live",
        "rank": 1,
        "group": "ahora",
        "priority": "alta",
        "title": "Mantener el proceso vivo hasta que el diario tenga un mes",
        "line": "Live", "status": "pendiente", "impact": "alto", "effort": "bajo",
        "evidence": "La MEDICIÓN ya está hecha y probada, no solo la infraestructura: "
                    "backtest/divergence_study.py re-simula la ventana del diario con el mismo "
                    "motor (enganchándole un MemoryJournal, así que emite el mismo esquema de "
                    "línea), parea por (día, símbolo, estrategia), reparte la diferencia de "
                    "precio en tres sumandos que SUMAN -referencia, coste y cruzado-, compara el "
                    "embudo de decisiones y tasa la latencia contra barras 1H reales. Publica en "
                    "data/live/divergence.json con tres reglas que pueden fallar. Y el diario "
                    "sella ahora los dos instantes que hacen medible la latencia (decided_at en "
                    "la orden y en la salida). Lo único que falta es CALENDARIO: con menos de 30 "
                    "días el estudio se niega a re-simular y dice cuántos faltan.",
        "why": "Sigue siendo el número 1 porque es lo único que no se puede comprimir después, "
               "pero el trabajo cambió de naturaleza: ya no hay nada que construir, hay que "
               "DEJAR CORRER. Cada semana que el proceso no despierta es una semana perdida al "
               "final, y ahora el coste de no hacerlo es visible -la vista de paper trading dice "
               "cuántos días faltan, medidos-. Cuando el diario pase de 30 días, esta entrada se "
               "cierra corriendo un comando.",
        "prompt": (
            "Proyecto ai-trader (Python). NO HAY NADA QUE PROGRAMAR EN ESTA ENTRADA: la medicion "
            "de divergencia live-vs-backtest ya esta escrita, probada y publicada en "
            "src/ai_trader/backtest/divergence_study.py -> data/live/divergence.json, conectada a "
            "la vista 'Paper trading' y a la seccion 5.4 de la documentacion.\n"
            "\n"
            "Lo que falta es CALENDARIO. El estudio exige 30 dias de diario (span de calendario Y "
            "dias con ciclos) y, por debajo de eso, se niega a re-simular y publica el estado "
            "'sin_potencia' diciendo cuantos dias faltan.\n"
            "\n"
            "QUE HACER, en este orden:\n"
            "(1) Comprobar que el proceso en vivo sigue despertando: data/live/cycles.jsonl tiene "
            "que crecer. Si lleva dias parado, arrancarlo (seccion 'Operacion continua' del "
            "README) es TODO el trabajo de esta entrada.\n"
            "(2) Correr el estudio y mirar que dice:\n"
            "    .venv\\Scripts\\python.exe -m ai_trader.backtest.divergence_study --offline\n"
            "(3) Si ya hay potencia y publica cifra, LEERLA en este orden y no en otro: primero "
            "la cobertura de decisiones (si los dos mundos no ven las mismas senales, cualquier "
            "cifra de coste esta explicando la diferencia equivocada), despues el reparto del "
            "precio de llenado, y solo entonces la latencia.\n"
            "(4) Regenerar dashboard y docs, y actualizar esta entrada con el resultado -incluido "
            "si alguna de las tres reglas FALLA, que es un resultado y no un fallo del estudio-.\n"
            "\n"
            "Si al leer la cifra decides cambiar el motor (por ejemplo, modelar la latencia), eso "
            "es OTRA entrada: este estudio mide y declara, no toca execution/."
        ),
    },
    {
        "id": "synthetic-signal-emission",
        "rank": 2,
        "group": "despues",
        "priority": "critica",
        "title": "Que el generador emita las señales, y re-medir la transferencia de forma pareada",
        "line": "B/D", "status": "pendiente", "impact": "alto", "effort": "alto",
        "evidence": "Los FactorShock del generador YA son eventos con día, factor y magnitud "
                    "(synthetic/scenarios.py), descritos como 'un anuncio de la Fed, un "
                    "default, un ataque'. El generador ya sabe QUE pasa y CUANDO: las señales "
                    "sintéticas serían la emisión observable de un estado latente que ya existe. "
                    "Y la mitad barata del problema ya está construida y medida (2026-08-13): el "
                    "canal de observación existe (synthetic/signal_channel.py), se emite en un "
                    "pase aparte y entra por el contrato de producción, así que esta evolución "
                    "hereda toda esa fontanería y sólo tiene que ACOPLAR la emisión al estado "
                    "latente. Lo que el break-even ya contestó: con una PUERTA BINARIA hace falta "
                    "rho > 0,20 para batir al baseline, y la puerta por sí sola cuesta 1,02 "
                    "puntos de recompensa. Antes de acoplar señales conviene decidir si el "
                    "consumo correcto es una puerta o un input continuo del sizing: acoplar mejor "
                    "una señal que se consume tirando toda su información salvo un bit sube el "
                    "coste sin mover el listón.",
        "why": "Es la ficha que VALIDA O REFUTA la tesis entera: si ampliar el espacio de inputs "
               "hace que el ranking transfiera, el sintético se queda en el núcleo; si no, la "
               "contingencia ('mover el sustrato primario del ranking al histórico REAL') se "
               "activa. Y es la más delicada del roadmap por un riesgo "
               "concreto: si las señales sintéticas se emiten del mismo estado latente que los "
               "precios con un acoplamiento limpio, funcionaran demasiado bien en el sintético; y "
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
        "id": "line-d-cpcv-two-stage-cem",
        "rank": 3,
        "group": "despues",
        "priority": "alta",
        "title": "CPCV en dos etapas dentro del optimizador (que el CEM deje de puntuar con el corte único)",
        "line": "D", "status": "pendiente", "impact": "alto", "effort": "medio",
        "evidence": "mom_default en crypto_winter: +2,63 con el corte único y -2,20 con CPCV. En "
                    "crypto_bull_supercycle: -0,18 vs -0,60, con dispersión entre folds de "
                    "sigma ~ 2-3 unidades de Sharpe.",
        "why": "El optimizador sigue puntuando con el corte que su propio estudio desacredita: "
               "run_optimization -> evaluate_sample_detailed -> BacktestEngine.run(split_ratio=0.7). "
               "El corte único no está SESGADO, está ARBITRARIO -y arbitrario es letal para un "
               "optimizador: el CEM escala un paisaje cuyo relieve depende de que tramo de "
               "historia cayó en el 30% de test. La objeción obvia es el coste (CPCV multiplica "
               "x15), y la solución no es elegir sino separar cribado de decisión: dos etapas.",
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
        "rank": 4,
        "group": "despues",
        "priority": "media",
        "title": "Re-correr el estudio de validación con el ensemble completo",
        "line": "D", "status": "pendiente", "impact": "medio", "effort": "bajo",
        "evidence": "El informe publicado corrió con n_paths=1: 8 escenarios x 4 configuraciones "
                    "= 32 muestras.",
        "why": "Las conclusiones sobre el corte único -que es arbitrario y no optimista, y que la "
               "brecha real está contra la cola- son direccionales pero descansan en poca "
               "muestra. Al integrar CPCV en el pipeline (evolución 'CPCV en dos etapas dentro del "
               "optimizador') conviene re-correrlo con "
               "varios caminos por escenario para que la dispersión medida tenga detrás "
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
               "hold-out, colándose dentro del PBO: si el mismo escenario está a los dos lados, "
               "el PBO sale más benigno de lo que debería, porque elegir por train ya sabe algo "
               "del test. Arreglo pequeño, y afecta a una cifra que se publica como garantía.",
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
        "title": "Reportar n_failed junto al reward (la penalización domina la cola)",
        "line": "A", "status": "pendiente", "impact": "medio", "effort": "bajo",
        "evidence": "FAILURE_PENALTY = -5 frente a un headline que vive en el rango -2..2: una "
                    "configuración que falla en 2-3 muestras de 100 tiene el CVaR@25% "
                    "prácticamente capturado por los -5.",
        "why": "Que fallar duela es correcto y no se discute; lo que no puede ser es que el "
               "número final no distinga 'esta config tiene mala cola' de 'esta config es "
               "frágil'. Son dos diagnósticos distintos con dos acciones distintas, y hoy "
               "colapsan en el mismo -5. No hay que cambiar la penalización, hay que publicar el "
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
                    "independientes en el máximo esperado bajo la nula.",
        "why": "Las generaciones tardías del CEM se concentran alrededor de la élite, así que los "
               "intentos efectivos son menos que los contados y el DSR SOBRE-deflacta. El sentido "
               "del error es el prudente, pero mientras no esté escrito alguien puede leer el DSR "
               "como una probabilidad calibrada, que no lo es. Una línea de docstring y una nota "
               "en la metodología.",
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
        "title": "Ordenación de colas y clustering entre activos: el eje que ai_v3 no arregló",
        "line": "B", "status": "pendiente", "impact": "medio", "effort": "medio",
        "evidence": "ai_v3 cumple los umbrales de NIVEL y COBERTURA (98,3% de cobertura media, "
                    "curtosis 3,40 vs 4,19 real, clustering 0,196 vs 0,190) pero su rank_corr "
                    "medio es -0,23: clustering -0,74 y curtosis -0,18. En ai_v2 el clustering "
                    "ordenaba +0,32. El nivel se arregló y la ordenación empeoró.",
        "why": "El generador ya produce la magnitud correcta de cola y de agrupamiento, pero no "
               "sabe QUÉ activo tiene más: en el mercado real son los más ruidosos (DOGE 17,8 de "
               "curtosis y 0,31 de clustering; XRP 13,9) y en el sintético salen de los que menos "
               "ruido propio tienen. La causa es identificable en el motor: el componente "
               "idiosincrático pasa por `_ar1_idio`, un AR(1) con phi negativo en las fases de "
               "rango, que BLANQUEA la estructura de |r| justo en los activos donde ese "
               "componente pesa más (idio_vol alto). Importa para la selección cross-sectional: "
               "una política que elija activos por su régimen de volatilidad está aprendiendo un "
               "ordenamiento invertido respecto al mercado. No es crítico -las tres lecturas se "
               "publican por separado y esta se declara- pero es el último hueco medido del "
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
        "rank": 9,
        "group": "despues",
        "priority": "critica",
        "title": "CONTINGENCIA: mover el sustrato primario del ranking al histórico REAL",
        "line": "B/D", "status": "bloqueada", "impact": "alto", "effort": "alto",
        "evidence": "MEDIDO: Spearman entre el ranking real y el sintético = -0,04 sobre 16 "
                    "configuraciones (IC95% por bloques [-0,44, +0,49], p = 0,89). El top-4 del "
                    "sintético acierta 1 de 4 en la mitad buena del real, peor que el azar (2,0). "
                    "Y sobre las 9 que operan de verdad en los dos mundos el acuerdo es NEGATIVO "
                    "(-0,67). Informe: data/transfer/report_ai_v3.json.",
        "why": "La evidencia de arriba sigue en pie y no se toca. Lo que cambia es la LECTURA: se "
               "midió con estrategias que solo ven precio y volumen, y el único edge del mundo "
               "sintético es un AR(1) colocado a mano por régimen -- rankear momentum sobre eso "
               "mide que configuración ajusta mejor ese AR(1), y no hay motivo para que "
               "transfiera. La hipótesis alternativa (el cuello de botella es el ESPACIO DE "
               "INPUTS, no el generador) es testeable con el mismo instrumento, y es lo que "
               "persiguen los ranks 1-9. Por eso esta ficha pasa de conclusión automática a "
               "CONTINGENCIA: se ejecuta si `synthetic-signal-emission` refuta esa "
               "hipótesis -- es decir, si con el espacio de inputs ampliado el ranking sigue sin "
               "transferir, y en particular si tampoco transfiere en el brazo ORÁCULO, que hace "
               "trampa a propósito. Si ni haciendo trampa transfiere, el problema no son los "
               "inputs y hay que sacar el sintético del criterio de selección. Bloqueada, no "
               "descartada: el código sigue eligiendo hoy con un juez del que se sabe que no "
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
        "rank": 10,
        "group": "despues",
        "priority": "alta",
        "title": "Optimización CEM completa, ya con el juez validado",
        "line": "RL", "status": "bloqueada", "impact": "alto", "effort": "medio",
        "depends": 1,
        "evidence": "El harness CEM está listo, pero cada backtest cuesta ~60 s con 35 activos y "
                    "hoy apuntaría al juez del corte único.",
        "why": "Correr la optimización a escala solo tiene sentido cuando el objetivo que escala "
               "el CEM es el bueno: primero el sustrato (#1) y el juez en dos etapas (#3), "
               "después esto. Hacerlo antes es gastar cómputo caro produciendo un ganador que "
               "habría que tirar.",
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
        "rank": 11,
        "group": "despues",
        "priority": "media",
        "title": "Presupuesto de latencia: el backtest supone que se llena a las 00:00 UTC en punto",
        "line": "Medicion", "status": "pendiente", "impact": "medio", "effort": "bajo",
        "evidence": "MEDIDO (data/sessions/report.json): el hueco entre el cierre que la estrategia "
                    "ve y el open al que se llena es CERO a efectos prácticos -0,07% del rango "
                    "diario, 0,55 pb-, así que la convención de llenado no sesga nada. Pero eso "
                    "solo vale si la orden sale en el instante del open: con UNA hora de retraso el "
                    "precio de llenado ya se ha desplazado 57,9 pb (9,2% del rango del día), que "
                    "son 3,9x el coste de entrada de referencia que el motor si cobra (15 pb).",
        "why": "El estudio de sesiones cerró la pregunta que tenía abierta el backtest (la ventana "
               "ciega no tiene ancho) y abrió otra que hoy no está ni medida ni presupuestada: el "
               "backtest describe un sistema PUNTUAL, y nada en el repo obliga al ciclo real a "
               "serlo. No es un bug del motor -por eso no se tocó- sino un requisito no escrito "
               "que hay que convertir en presupuesto explícito: cuanto puede tardar el ciclo en "
               "ejecutar antes de que el backtest deje de describirlo. Va en 'después' y no en "
               "'ahora' porque solo muerde cuando el paper trading corra en vivo (#6), que es "
               "donde la latencia deja de ser hipotética.",
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
        "rank": 12,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Nuevas estrategias cripto (deliberadamente NO priorizada)",
        "line": "Estrategias", "status": "pendiente", "impact": "medio", "effort": "alto",
        "depends": 1,
        "evidence": "Solo 6 de 32 filas aprueban el gate bajo CPCV. Eso admite dos lecturas -las "
                    "estrategias son flojas, o el juez es ruidoso- y hasta saber cual, añadir "
                    "candidatos multiplica el problema de múltiples pruebas.",
        "why": "Aparece en la lista porque es trabajo real y acordado, pero NO se aborda todavía, "
               "y la razón es una asimetría de coste, no una preferencia: una estrategia añadida "
               "hoy se re-evalúa GRATIS cuando el juez mejore; un juez malo contamina todo lo que "
               "puntúe hoy. Las estrategias son la cosecha; el juez es el suelo. Además cada "
               "candidato nuevo sube el n_trials del DSR, o sea que añadir sin ganar edge "
               "empeora activamente el veredicto de todo lo demás.",
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
        "rank": 13,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Re-medir lambda y kappa con los costes nuevos y más potencia estadística",
        "line": "A/C", "status": "pendiente", "impact": "bajo", "effort": "medio",
        "evidence": "480 backtests ya medidos: la superficie (lambda, kappa) sale PLANA y la "
                    "elección de configuración no cambia. Pero se midió con un único corte 70/30, "
                    "un camino por escenario y el slippage PLANO de 5 bps que ya no existe.",
        "why": "La conclusión publicada -penalizar no estabiliza, y los costes ya dentro del "
               "Sharpe equivalen a un lambda ~ 6,3, muy por encima del 0,25 elegido- se refuerza "
               "con el modelo de costes nuevo, que cobra más fricción. Es decir: re-medirlo casi "
               "seguro no mueve nada. Por eso es de impacto bajo y va aquí, no arriba.",
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
        "rank": 14,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Anotar el modelo de IA en el manifiesto de cada librería",
        "line": "E", "status": "pendiente", "impact": "bajo", "effort": "bajo",
        "evidence": "El manifiesto guarda la CLASE del diseñador ('ClaudeScenarioDesigner'), no "
                    "el identificador del modelo ni la fecha del diseño.",
        "why": "El diseño con IA no es reproducible y ya no puede serlo (los modelos actuales "
               "retiraron los parámetros de muestreo), así que la única trazabilidad posible es "
               "registrar CON QUE se generó. Hoy dos librerías diseñadas con modelos distintos "
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
        "id": "operational-symbol-guard",
        "rank": 15,
        "group": "no-prioritario",
        "priority": "baja",
        "title": "Guarda operativa por símbolo (sanciones, deslistado, halt): lo que deja abierto no tener veto",
        "line": "Riesgo", "status": "pendiente", "impact": "bajo", "effort": "bajo",
        "evidence": "El sistema no tiene ningún concepto de símbolo no operable: grep de "
                    "eligib|veto|blacklist|blackout|halt en src/ no devuelve nada, y las únicas "
                    "guardas por símbolo son la posición abierta y el cooldown "
                    "(app/runner.py:229-235). El radar de señales ya está construido y cableado "
                    "(2026-08-12) y sigue sin vetar nada A PROPÓSITO: toda señal actúa como feature "
                    "y la única puerta que existe falla ABIERTA por diseño, así que este hueco "
                    "queda abierto y escrito en vez de resuelto de tapadillo. NOVEDAD 2026-08-13: "
                    "ya no falta el DATO. `cex_listings` publica altas, bajas y designaciones de "
                    "vigilancia de Upbit fechadas desde 2018-08-02 (523 eventos sobre 343 tokens, "
                    "MEDIDO) y `ofac_sdn` publica la lista de sanciones; las dos entran hoy como "
                    "FEATURE, que es la vía correcta para lo predictivo y no para lo operativo. Lo "
                    "que sigue faltando es exactamente lo que dice el alcance de abajo: la guarda.",
        "why": "No es alfa y por eso se separa: vetar un activo sancionado, deslistado o con el "
               "mercado detenido no es una estrategia, es una restricción operativa, y meterla en "
               "la misma caja que las features fue justamente el error de diseño que la "
               "unificación corrige. Va en 'no prioritario' porque hoy el riesgo es teórico -paper "
               "trading, universo de majors configurado a mano- y se vuelve real el día que haya "
               "dinero de verdad o un universo ancho y rotatorio. Ese día esto sube de rango solo.",
        "prompt": (
            "Proyecto ai-trader (Python). NO EJECUTAR ANTES de que haya dinero real o un universo "
            "que rote sin supervision: hoy el riesgo es teorico y esta declarado. Se deja escrita "
            "para no perder el alcance.\n"
            "\n"
            "ALCANCE, y es deliberadamente estrecho: una guarda por simbolo que impide operar por "
            "razones NO PREDICTIVAS -direccion o token sancionado (OFAC SDN), par deslistado del "
            "venue, mercado detenido o en mantenimiento-. Nada que huela a alfa entra aqui: si una "
            "senal predice retornos, su sitio es el radar de features y no esta puerta. La "
            "distincion no es estetica -- una puerta que veta por motivos predictivos es una "
            "estrategia sin backtest.\n"
            "\n"
            "DONDE: consultada como UNA GUARDA MAS en TradingRunner._process_symbol, junto al "
            "cooldown por simbolo, ANTES de pedir senal. NO dentro de RiskEngine, y las tres "
            "razones importan: (i) RiskEngine no tiene reloj ni colaboradores -- es una funcion "
            "pura de (limites, senal, cartera) y esa pureza vale; (ii) en modo equity-aware, que es "
            "SIEMPRE el del backtest, tres de sus guardas no se ejecutan "
            "(risk/engine.py:108-119), asi que una guarda ahi correria el riesgo de estar "
            "silenciosamente inactiva en toda la evidencia; (iii) _symbol_in_cooldown "
            "(app/runner.py:489-502) es el precedente estructural exacto: la unica guarda que "
            "consulta reloj e historial por simbolo y descarta antes de pedir senal, y ademas es "
            "mas barata porque evita cargar barras y correr estrategias.\n"
            "\n"
            "TRES PROPIEDADES NO NEGOCIABLES:\n"
            "1. Falla ABIERTA: sin datos no hay bloqueo. Un fallo del proveedor no puede parar el "
            "trading, y menos uno cuya unica funcion es una comprobacion administrativa.\n"
            "2. NUNCA en search_space.py, igual que las features: nada de esto es sorteable.\n"
            "3. SE MIDE. Hoy un rechazo es una cadena de texto libre en RiskDecision.reason "
            "(risk/engine.py:239-241): no hay taxonomia ni recuento, asi que una guarda que nunca "
            "dispara y una que dispara siempre son indistinguibles. Cada bloqueo lleva source_key "
            "estructurado y se acumula en SymbolCycleDiagnostics (que ya existe) para reportar "
            "cuantas veces bloqueo cada motivo y sobre que simbolos. Sin esa cifra, una guarda es "
            "una creencia.\n"
            "\n"
            "El orden de las guardas de _process_symbol no esta testeado: congelalo con un test "
            "ANTES de insertar la nueva. Tests + .venv\\Scripts\\python.exe (poetry run esta roto) "
            "+ ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "equities-parked",
        "rank": 16,
        "group": "segundo-plano",
        "priority": "aparcada",
        "title": "Renta variable: aparcada a propósito (no se activa la clase de activo)",
        "line": "Universo", "status": "aparcada", "impact": "bajo", "effort": "alto",
        "evidence": "Cero estrategias de renta variable en el repo (hay proveedor, no estrategia). "
                    "Cero datos reales detrás de la pata de equity del generador. Y el universo "
                    "de 20 megacaps está elegido en 2024-26 entre las que sobrevivieron.",
        "why": "La decisión es SOLO CRIPTO, y no por poco. Toda la evidencia empírica del repo es "
               "cripto: la fidelidad se midió contra Binance y la calibración y la validación "
               "corrieron sobre universos con calendario 365. Activar renta variable sería "
               "construir estrategia + validar el proveedor Alpaca (¿ajusta splits y "
               "dividendos?) + verificar el calendario 252 de punta a punta + resolver un sesgo "
               "de supervivencia estructural (constituyentes point-in-time), todo antes de la "
               "primera señal. Y el edge plausible no está ahí: momentum diario sobre AAPL o SPY "
               "compite contra el segmento más eficiente del planeta. Lo que NO se hace es quitar "
               "los stocks del universo SINTÉTICO: allí GLD, TLT y UUP son lo que hace que los "
               "escenarios de tipos y de dólar signifiquen algo para cripto vía los factores "
               "compartidos. Se genera con los 35, se puntúa y se opera solo cripto.",
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
        "rank": 17,
        "group": "segundo-plano",
        "priority": "aparcada",
        "title": "Polymarket en el backtest: aparcado hasta tener histórico propio",
        "line": "Universo", "status": "aparcada", "impact": "bajo", "effort": "alto",
        "evidence": "Sin OHLCV histórico de mercados de predicción. La estrategia "
                    "polymarket_threshold existe y opera en papel, pero no entra en ningún "
                    "backtest.",
        "why": "No es un olvido ni una limitación del código: no hay histórico que backtestear, y "
               "comprarlo no es barato. La vía realista es la que abre la evolución 'Poner el "
               "paper trading a correr en vivo': con el "
               "paper trading corriendo, el sistema empieza a guardar midpoints por ciclo, y en "
               "unos meses habrá una serie propia. Retomar esto antes de tener esa serie es "
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
