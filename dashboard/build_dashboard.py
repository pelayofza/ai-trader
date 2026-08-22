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
from datetime import datetime, timezone
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
from ai_trader.research.activity_study import (
    activity_report_path,
    load_activity_report,
)
from ai_trader.scoring.sample_eval import evaluate_baselines, evaluate_sample_detailed
from ai_trader.research.signal_study import (
    DEFAULT_LIBRARY_ID as SIGNAL_LIBRARY,
    load_signal_report,
    report_path as signal_report_path,
)
from ai_trader.research.transfer_study import (
    DEFAULT_LIBRARY_ID as TRANSFER_LIBRARY,
    load_transfer_report,
    transfer_report_path,
)
from ai_trader.research.validation_study import (
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
from ai_trader.scoring.families import FAMILIES, NEW_FAMILIES
from ai_trader.strategies import build_strategy
from ai_trader.strategies.mean_reversion import MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy
from ai_trader.research.synthetic.fidelity import (
    FIDELITY_BASELINE_LIBRARY,
    FIDELITY_LIBRARY,
    TARGET_METRIC_KEYS,
    fidelity_report_path,
    load_fidelity_report,
    metric,
)
from ai_trader.research.synthetic.store import SyntheticStore

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
#
# Sobre MERCADO REAL, no sobre una libreria generada. Es el cambio que arrastro el estudio
# de transferencia: el sintetico no ordena como el mercado, asi que un ranking ilustrativo
# calculado sobre el sintetico ilustraba la cosa equivocada.
#
# Sigue siendo una MUESTRA: la evidencia vive en `data/`. Lo que se reduce es el universo y
# el numero de sub-ventanas, no el metodo -- cada muestra se puntua con el mismo
# `evaluate_sample_detailed` que usa el optimizador.
RANK_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
# Cuatro sub-ventanas de 300 dias, las mas RECIENTES del historico cerrado. Las mas
# recientes y no las primeras porque son el regimen que el sistema va a operar, y porque
# ahi el universo reducido esta vivo entero (SOL no cotizaba en 2018).
RANK_N_WINDOWS = 4
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

CHART_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CHART_POINTS = 160  # downsample de las series de precio para el JSON

# Puntos de la curva de paper trading que se embeben en el HTML. El diario crece sin
# limite (un ciclo cada 15 minutos) y el dashboard es un fichero autocontenido.
CURVE_POINTS = 400

# --- panel de costes de ejecucion ---------------------------------------------------
# Muestra transversal del universo OPERADO: dos cripto de primer nivel y cinco altcoins
# de liquidez muy distinta. Los tamanos van de "orden de andar por casa" a institucional,
# que es donde el impacto y el techo de capacidad dejan de ser teoria.
#
# Ya no hay indices ni macro en esta tabla, y no es un recorte cosmetico: se calculaba
# sobre precios de la libreria sintetica, donde SPY o GLD existian por construccion. Sobre
# mercado real solo hay lo que se opera, que es cripto.
COST_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
    "LINK/USDT", "ATOM/USDT", "SEI/USDT",
]
COST_ORDER_USD = [1_000.0, 250_000.0, 25_000_000.0]


# ------------------------------------------------------------------ util ------------


def real_sample() -> dict:
    """
    Las barras REALES que alimentan las tres vistas que se calculan en el build: el
    ranking de muestra, el panel de costes y la demo de señales.

    Se leen UNA vez de la cache en disco (`--offline`), sobre la ventana historica CERRADA
    del proyecto. Cerrada y no "hasta hoy" por una razon operativa: el artefacto se
    compara byte a byte contra el commiteado en cada verificacion, y una ventana que
    avanzara con el calendario haria fallar esa comparacion cada dia.

    Devuelve {} si no hay cache -- las tres vistas degradan solas y lo declaran.
    """
    from ai_trader.data.real_history import (
        DEFAULT_EXCHANGE,
        DEFAULT_REAL_END,
        DEFAULT_REAL_START,
        build_service,
        fetch_real_bars,
    )
    from ai_trader.scoring.real_substrate import real_windows

    start = datetime.fromisoformat(DEFAULT_REAL_START).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(DEFAULT_REAL_END).replace(tzinfo=timezone.utc)
    symbols = sorted(set(RANK_UNIVERSE) | set(COST_SYMBOLS) | set(CHART_SYMBOLS))
    try:
        bars = fetch_real_bars(
            symbols, start, end, build_service(DEFAULT_EXCHANGE, offline=True)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sin barras reales en cache: %s", exc)
        return {}
    if not bars:
        logger.warning("Sin barras reales en cache para %s", ", ".join(symbols))
        return {}

    windows = real_windows(start, end, RANK_WINDOW_DAYS)[-RANK_N_WINDOWS:]
    return {
        "bars": bars,
        "windows": windows,
        "exchange": DEFAULT_EXCHANGE,
        "start": start,
        "end": end,
    }


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
        # Aparte de la lista, y no dentro: no se rankea con ellas. Ver `collect_priority()`.
        "priority": collect_priority(),
    }


# Prosa editorial de la decima, la unica sin nucleo de precio. Va SEPARADA de las ocho de
# arriba en el dato y en la vista, y la separacion es el contenido: las ocho se ordenan entre
# si con evidencia publicada, y esta no compite con nadie porque su sustrato tiene tres dias.
# Meterla en la misma rejilla de tarjetas insinuaria una comparacion que no existe.
PRIORITY_PROSE = {
    "name": "Reporte diario experto",
    "regime": "reporte",
    "idea": "La unica primitiva SIN nucleo de precio: la decision entera sale de las 37 "
            "respuestas categoricas que un agente externo escribe cada manana. El precio solo "
            "aporta el numero al que se entra.",
    "rules": [
        "Frescura: pasadas las horas de la hora de corte, no se opera.",
        "Lado: conviccion absoluta |score| >= umbral Y estar entre los N mejores del dia.",
        "Stop: multiplo de la sigma DIARIA que publica el propio reporte (P32/P33).",
        "Objetivo: multiplo del stop que sube con la conviccion y baja con evento y aglomeracion.",
    ],
}

# Los cuatro papeles que reparten las 37 preguntas, y su motivo en una linea. Las cifras NO
# se escriben: se cuentan de las tablas del modulo, para que anadir una pregunta a la tabla y
# olvidarse de la vista rompa el build en vez de publicar un reparto que no suma 37.
PRIORITY_ROLES = (
    ("Direccion", "suman al score con peso y polaridad"),
    ("Horquilla", "la volatilidad da la UNIDAD del stop y del objetivo, no un voto"),
    ("Moduladores", "profundidad recorta confianza; beta escala el bloque de mercado"),
    ("Benchmark", "P30 es la conclusion del propio redactor y NO se lee"),
)


def collect_priority() -> dict:
    """La estrategia con prioridad forzada: que hace, con que pesos y por que manda.

    TODO lo que sale de aqui es REPRODUCIBLE en cualquier clon: la tabla de pesos vive en
    codigo y el reparto de papeles se cuenta del cuestionario, que esta versionado. Ni una
    cifra de `data/signals_raw/`, que esta fuera de git y cambia cada manana -- la vista de la
    captura es la del capitulo de datos, y esta es la de la decision.
    """
    from ai_trader.observation.daily_report_scores import (
        BENCHMARK_QUESTION,
        BLOCK_LABELS,
        CROWDING_QUESTIONS,
        DIRECTIONAL,
        FULL_CONVICTION_SCORE,
        MIN_COVERAGE,
        TOTAL_WEIGHT,
    )
    from ai_trader.signals.ai_reports import QUESTIONNAIRE_ID, load_contract

    strategy = build_strategy("daily_report_expert")
    contract = load_contract(ROOT) or {}
    n_questions = contract.get("n_questions") or 0

    counts = {
        "Direccion": len(DIRECTIONAL),
        "Horquilla": 2,       # P32 realizada, P33 implicita
        "Moduladores": 2,     # P34 profundidad, P35 beta
        "Benchmark": 1,       # P30
    }
    if n_questions and sum(counts.values()) != n_questions:
        raise ValueError(
            f"Los papeles reparten {sum(counts.values())} preguntas y el cuestionario tiene "
            f"{n_questions}. Anadir una pregunta y olvidar el reparto tiene que ROMPER el "
            "build, no publicar una vista que no suma."
        )

    # Que estrategias operan HOY, leido del config y no escrito a mano: si alguien reactiva
    # una familia aparcada, la vista lo dice sola.
    live = [spec.type for spec in load_config(ROOT / "config" / "default.toml").strategies]

    return {
        "id": "daily_report_expert",
        **PRIORITY_PROSE,
        "params": _params_dict(strategy.config),
        "questionnaire": QUESTIONNAIRE_ID,
        "n_questions": n_questions,
        "roles": [
            {"role": role, "n": counts[role], "why": why} for role, why in PRIORITY_ROLES
        ],
        "total_weight": round(TOTAL_WEIGHT, 2),
        "min_coverage": MIN_COVERAGE,
        "full_conviction": FULL_CONVICTION_SCORE,
        "benchmark": BENCHMARK_QUESTION,
        "contrarian": list(CROWDING_QUESTIONS),
        # La tabla entera, pregunta por pregunta. Es el artefacto que hace auditable un juicio
        # experto: sin ella, "pesos afirmados" es una frase.
        "weights": [
            {
                "id": qid,
                "weight": q.weight,
                "polarity": q.polarity,
                "block": BLOCK_LABELS.get(q.block, q.block),
                "note": q.note,
            }
            for qid, q in DIRECTIONAL.items()
        ],
        "live": live,
        "is_only_live": live == ["daily_report_expert"],
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


def strategy_signals_demo(sample: dict) -> dict:
    """Series de precio REAL anotadas con las entradas de cada primitiva (ilustrativo).

    Para cada primitiva escanea unas pocas (sub-ventana, simbolo) y se queda con la que MAS
    senales dispara, de modo que el chart sea representativo (p.ej. mean-reversion no queda
    vacio si el activo elegido resulto estar en tendencia).

    El docstring decia "series reales" desde siempre y dibujaba velas de la libreria ai_v2.
    Ahora dice la verdad."""
    demo: dict = {}
    if not sample:
        return demo

    windows = sample["windows"]
    # Dos sub-ventanas por familia y no cuatro: son ocho familias, el barrido evalua una
    # senal cada dos barras sobre cada (ventana, simbolo), y el chart es ilustrativo.
    scan = windows[-2:]
    cand_syms = [s for s in CHART_SYMBOLS + RANK_UNIVERSE if s in sample["bars"]]

    # Lista y no `set`: el orden de iteracion fija el orden de las claves del JSON, y el
    # hash de las cadenas esta aleatorizado por proceso. Con un set, el artefacto saldria
    # distinto en cada build y la comparacion contra el commiteado fallaria sin motivo.
    for strat_type in [s["id"] for s in collect_strategies()["strategies"]]:
        strat = build_strategy(strat_type, {})
        best: dict | None = None
        for window in scan:
            for sym in dict.fromkeys(cand_syms):
                df = sample["bars"].get(sym)
                if df is None:
                    continue
                bars = df.loc[window.start : window.end]
                if len(bars) < 80:
                    continue
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
                        "window": window.label,
                        "period": f"{window.start.date()} → {(window.end).date()}",
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


def run_ranking(sample: dict) -> dict:
    """
    Ranking de muestra sobre MERCADO REAL: una sub-ventana del historico = una muestra.

    Rankea por CVaR@25% del HEADLINE score out-of-sample (Sharpe - lambda*turnover -
    kappa*maxDD). Ademas de las estrategias corre los BASELINES pasivos sobre las mismas
    muestras (el gate que hay que batir para 'aprobar') y descuenta el sobreajuste por
    multiples pruebas con PBO y DSR sobre la distribucion de scores del propio ranking.

    Se calculaba sobre caminos de la libreria ai_v2 hasta que el estudio de transferencia
    midio que el sintetico no ordena como el mercado: un ranking ilustrativo sobre el
    sintetico ilustraba la cosa equivocada. El metodo no cambia -- el mismo
    `evaluate_sample_detailed` con el mismo corte 70/30 --, cambia de donde salen las velas.

    Y con eso cambia una propiedad, que hay que decir: cuatro sub-ventanas del MISMO camino
    historico no son cuatro mundos independientes. El PBO y el DSR de esta vista siguen
    siendo el descuento por multiples pruebas, pero sobre una muestra con menos variedad
    que la que tenia el sintetico. La evidencia con potencia vive en `data/`.
    """
    result: dict = {
        "scope": {
            "substrate": "real",
            "universe": RANK_UNIVERSE,
            "n_windows": RANK_N_WINDOWS,
            "window_days": RANK_WINDOW_DAYS,
            "weights": DEFAULT_HEADLINE_WEIGHTS.as_dict(),
        },
        "rows": [],
        "baselines": [],
        "distributions": {},
        "overfit": {},
    }
    if not sample:
        logger.warning("Ranking no disponible: no hay barras reales en cache")
        return result

    base_config = load_config(ROOT / "config" / "default.toml")
    base_config = dataclasses.replace(
        base_config, runner=dataclasses.replace(base_config.runner, symbols=list(RANK_UNIVERSE))
    )
    windows = sample["windows"]
    bars = {s: df for s, df in sample["bars"].items() if s in RANK_UNIVERSE}
    result["scope"]["windows"] = [w.as_dict() for w in windows]
    result["scope"]["exchange"] = sample["exchange"]

    specs = [(label, stype, StrategySpec(type=stype, id=label, params=params))
             for label, stype, params in RANK_CONFIGS]

    # Una sola pasada por muestra: sobre las mismas barras se puntuan todas las
    # configuraciones Y los baselines. Asi la comparacion es pareada (misma ventana para
    # todos), que es lo que el gate necesita.
    scores: dict[str, list[float]] = {label: [] for label, _, _ in specs}
    sharpes: dict[str, list[float]] = {label: [] for label, _, _ in specs}
    # Las operaciones de cada muestra viajan al lado de su score porque el gate las
    # NECESITA: sin ellas `gate()` deja approved=False con activity_checked=False -a
    # proposito, para que un requisito sin comprobar no pase por comprobado-, y la columna
    # "aprueba" de esta vista era estructuralmente 'no' para todo el mundo.
    trades: dict[str, list[int]] = {label: [] for label, _, _ in specs}
    baseline_scores: dict[str, list[float]] = {}
    baseline_stats: dict[str, list] = {}
    oos_obs: list[int] = []

    for window in windows:
        start, end = window.start, window.end

        for label, _, spec in specs:
            try:
                ev = evaluate_sample_detailed(
                    base_config, spec, bars, start, end, split_ratio=0.7
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("eval fallo %s/%s: %s", label, window.label, exc)
                continue
            scores[label].append(ev.score)
            sharpes[label].append(ev.sharpe)
            trades[label].append(ev.num_trades)
            oos_obs.append(ev.oos_observations)

        try:
            for name, baseline in evaluate_baselines(
                base_config, bars, start, end, split_ratio=0.7
            ).items():
                baseline_scores.setdefault(name, []).append(baseline.score)
                baseline_stats.setdefault(name, []).append(baseline)
        except Exception as exc:  # noqa: BLE001
            logger.warning("baselines fallaron %s: %s", window.label, exc)

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
        stats = aggregate_reward(scores[label], trades=trades[label])
        verdict = gate(scores[label], usable_baselines, trades=trades[label])
        activity = stats.activity
        result["rows"].append(
            {
                "label": label,
                "type": stype,
                "approved": verdict.approved,
                "rankable": verdict.eligible,
                "beats_baselines": verdict.beats_baselines,
                "trades_per_window": None if activity is None else round(
                    activity.trades_per_window, 1
                ),
                "zero_window_pct": None if activity is None else round(
                    activity.zero_window_pct, 0
                ),
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
    from ai_trader.research.synthetic.universe import DEFAULT_UNIVERSE

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


def collect_costs(sample: dict) -> dict:
    """Lo que cuesta EJECUTAR en cada mercado del universo, con LIQUIDEZ REAL.

    No corre backtests: resuelve la liquidez de cada simbolo con la misma costura que usa
    el motor (mediana de volumen y volatilidad de las ultimas barras cerradas) y evalua el
    modelo para ordenes de tamano creciente. Es la evidencia de que la friccion dejo de ser
    una constante.

    Se calculaba sobre un escenario de la libreria sintetica, y eso lo invalidaba como
    panel de COSTES: el volumen de un activo generado es el que el generador le puso, asi
    que la capacidad y el impacto que salian de aqui eran los de un mercado inventado. Con
    barras reales, la capacidad en dolares es la que de verdad hay."""
    out: dict = {"substrate": "real", "sizes_usd": COST_ORDER_USD, "rows": []}
    if not sample:
        logger.warning("Panel de costes no disponible: no hay barras reales en cache")
        return out

    config = load_config(ROOT / "config" / "default.toml")
    # La liquidez se mide sobre el tramo mas reciente, que es el que describe el mercado
    # que se va a operar: promediar ocho anos mezclaria el volumen de 2018 con el de hoy.
    window = sample["windows"][-1]
    bars = {
        s: df.loc[window.start : window.end]
        for s, df in sample["bars"].items()
        if s in COST_SYMBOLS
    }
    symbols = [s for s in COST_SYMBOLS if s in bars and len(bars[s]) > 2]
    if not symbols:
        logger.warning("Panel de costes: ningun simbolo de la muestra tiene barras reales")
        return out
    out["period"] = f"{window.start.date()} → {window.end.date()}"

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
       src/ai_trader/research/research/synthetic/fidelity.py   stylized facts, autocorrelacion, colas
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
veces (scoring/aggregate.py:103, research/activity_study.py:104,
research/transfer_study.py:654) y el cargador de informes, seis. Ninguna de esas copias la
detecta un test, porque cada una tiene los suyos: corregir una y no las otras deja al
sistema puntuando con dos definiciones distintas de la misma metrica."""


def collect_daily_reports() -> dict:
    """
    La SEGUNDA via de captura: el reporte diario por activo que escribe un agente externo.

    Dos bloques, y separarlos es el contenido de la vista:

    - `contract`: sale de `config/`, que esta VERSIONADO. Es lo que el pipeline promete y
      es reproducible en cualquier clon.
    - `last_run`: sale de `data/signals_raw/ai_reports/`, que esta en el .gitignore y crece
      cada manana a las ocho. Es lo que una ejecucion MIDIO, y cambia solo con que pase un
      dia -- igual que el bloque de paper trading en vivo, y por el mismo motivo la
      caracterizacion lo enmascara en vez de congelarlo (ver tests/golden_support.py).

    Confundir los dos seria repetir el error que la propia v2 del cuestionario vino a
    arreglar: dar por medido lo que solo estaba declarado.

    `last_run` va DENTRO y al final del dict a proposito: el scrubber lo recorta por su
    clave, y para eso necesita que el borde sea estable.
    """
    from ai_trader.signals.ai_reports import (
        AI_REPORTS_DIR,
        AGENT_INSTRUCTIONS,
        contract_problems,
        load_contract,
        load_last_run,
    )

    return {
        "contract": load_contract(ROOT),
        # Vacia mientras el contrato sea coherente. Si algun dia no lo es, la vista lo dice
        # en vez de ensenar cifras que ya no cuadran con lo que el agente va a leer manana.
        "problems": contract_problems(ROOT),
        "instructions_path": AGENT_INSTRUCTIONS.as_posix(),
        "archive_path": AI_REPORTS_DIR.as_posix(),
        "last_run": load_last_run(ROOT),
    }


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

    Las entradas RETIRADAS no llevan prompt, y no es un olvido: no hay nada que ejecutar.
    Estan en la lista para que se sepa cuales eran y por que dejaron de estar -- borrarlas
    dejaria el roadmap mas corto y la historia sin explicar.
    """
    ranks = [r["rank"] for r in ROADMAP]
    if sorted(ranks) != list(range(1, len(ROADMAP) + 1)):
        raise ValueError(f"Los rangos del roadmap deben ser 1..N sin huecos: {sorted(ranks)}")
    groups = {r["group"] for r in ROADMAP}
    unknown = groups - {g["key"] for g in ROADMAP_GROUPS}
    if unknown:
        raise ValueError(f"Grupos de roadmap desconocidos: {sorted(unknown)}")
    retired_with_prompt = [
        r["id"] for r in ROADMAP if r["group"] == "retiradas" and r.get("prompt")
    ]
    if retired_with_prompt:
        raise ValueError(f"Una entrada retirada no lleva prompt: {retired_with_prompt}")
    return [
        {**row, "prompt": row["prompt"] + REUSE_RULE} if row.get("prompt") else dict(row)
        for row in sorted(ROADMAP, key=lambda r: r["rank"])
    ]


def build() -> None:
    # Las tres vistas que se CALCULAN en el build (ranking, costes y demo de señales)
    # salen de aqui: barras reales, leidas una vez de la cache en disco. La libreria
    # sintetica ya solo se lee para el capitulo de investigacion archivada.
    logger.info("Barras reales de la cache (ventana historica cerrada)...")
    sample = real_sample()
    store = SyntheticStore(ROOT / "data" / "synthetic")
    logger.info("Captura de datos reales y constantes del trade...")
    market, trade = collect_market(), collect_trade()
    logger.info("Investigacion archivada: escenarios sinteticos...")
    synthetic = collect_synthetic(store)
    logger.info("Investigacion archivada: stylized facts ai_v1 vs ai_v2...")
    facts = stylized_facts(store)
    logger.info("Investigacion archivada: fidelidad (informe publicado)...")
    fidelity = collect_fidelity()
    logger.info("Investigacion archivada: transferencia (informe publicado)...")
    transfer = collect_transfer()
    logger.info("Descomposicion por sesion horaria (informe publicado)...")
    logger.info("Divergencia live-vs-backtest (informe publicado)...")
    logger.info("Suelo de actividad del ranking (informe publicado)...")
    logger.info("Break-even del IC: barrido de rho (informe publicado)...")
    logger.info("Catalogo de senales externas y auditoria de cobertura...")
    signals_platform = collect_signals()
    logger.info("Reporte diario por activo: contrato y ultima ejecucion...")
    logger.info("Catalogo de estrategias...")
    strategies = collect_strategies()
    logger.info("Demo de señales sobre precio real...")
    signals = strategy_signals_demo(sample)
    logger.info("Costes de ejecucion con liquidez real...")
    costs = collect_costs(sample)
    logger.info("Ranking sobre sub-ventanas reales (puede tardar unos minutos)...")
    ranking = run_ranking(sample)
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
        "daily_reports": collect_daily_reports(),
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
# ORDEN: `rank` 1..N de mayor a menor criticidad.
#
# ACTUALIZACION 2026-08-20: CAMBIA EL CRITERIO DE ORDENACION, y con el se caen diez
# entradas. El criterio anterior era la asimetria de coste del juez -- "un juez malo
# contamina todo lo que puntue mientras siga malo", asi que el sustrato y el juez van
# delante de la cosecha --. Ese criterio construyo un instrumento excelente y una
# herramienta que no opera nada.
#
# El criterio nuevo es el contrario, y se asume con sus consecuencias escritas: PONER LA
# HERRAMIENTA A FUNCIONAR sobre datos reales, aceptando el riesgo de sobreajuste que eso
# trae. Es mejor tener algo corriendo con sobreajuste -- y atacarlo despues con evidencia
# de calendario, que es la unica que no se puede falsificar -- que seguir refinando el
# juez de un backtest que no decide nada.
#
# Las tres lineas de "Ahora" son las tres piezas de ese bucle, en orden de dependencia:
# capturar mas y mejor senal externa, medir su calidad una por una, y generar estrategias
# con ella sobre el historico real. El paper trading sigue en paralelo por el motivo de
# siempre: es lo unico que compra tiempo de calendario, que no se puede comprimir despues.
#
# Lo que se retira NO se borra: las diez entradas siguen en la lista, en su grupo, con el
# motivo. Casi todas eran mejoras del generador sintetico o afinado del juez de backtest, y
# las dos cosas son afinar el instrumento en vez de usarlo. El historico de decisiones que
# sigue debajo se conserva entero por la misma razon.
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
# que una feature este aportando algo. `research/signal_study.py` barre la capacidad
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
        "subtitle": "Poner la herramienta a FUNCIONAR sobre datos reales. Tres frentes, en este "
                    "orden: capturar más y mejor señal externa, medir su calidad una por una, y "
                    "generar estrategias con ella. Y en paralelo, lo único que no se puede "
                    "comprimir después: que el paper trading siga despertando.",
    },
    {
        "key": "despues",
        "title": "Despues",
        "subtitle": "Lo que hace falta para que lo de arriba se pueda OPERAR de verdad, no solo "
                    "medir: la guarda por símbolo y el presupuesto de latencia.",
    },
    {
        "key": "segundo-plano",
        "title": "Segundo plano (no cripto)",
        "subtitle": "Renta variable y mercados de predicción, aparcados a propósito hasta que el "
                    "bucle cripto -captura, señal, estrategia, paper- esté cerrado y medido.",
    },
    {
        "key": "retiradas",
        "title": "Retiradas",
        "subtitle": "Diez entradas que salen de la lista al aparcar la línea sintética. No se "
                    "borran de la historia: se declara cuáles eran y por qué dejaron de estar. "
                    "Casi todas eran o mejoras del generador o afinado del juez de backtest, y "
                    "las dos cosas son afinar el instrumento en vez de usarlo.",
    },
]

ROADMAP = [
    {
        "id": "signal-capture-depth",
        "rank": 1,
        "group": "ahora",
        "priority": "critica",
        "title": "Capturar de verdad: profundidad histórica de las señales externas",
        "line": "Datos", "status": "pendiente", "impact": "alto", "effort": "medio",
        "evidence": "MEDIDO: el catálogo declara 30 fuentes y solo 14 son backtesteables "
                    "(`signals/catalog.py`, campo `history_from`, poblado por sonda y no a "
                    "mano). De los cinco temas, `liquidation` (1 de 4 fuentes con historia) y "
                    "`vol_surface` (1 de 2) NO alcanzan el mínimo de cobertura, y por eso el "
                    "estudio de la capa temática los declara no evaluables en vez de puntuarlos.",
        "why": "Es el cuello de botella de todo lo demás, y ahora se ve: el ranking decide sobre "
               "histórico real, y una señal sin histórico no puede entrar en esa decisión por "
               "mucho que esté conectada en vivo. Cada mes que el archivo crece hacia adelante es "
               "un mes que no se recupera; cada fuente que admita descarga hacia atrás es "
               "profundidad que se gana hoy.",
        "prompt": (
            "Proyecto ai-trader (Python). El archivo de senales externas vive en "
            "data/signals_raw/ y su catalogo en src/ai_trader/signals/catalog.py, con el campo "
            "`history_from` que dice desde cuando hay profundidad MEDIDA (lo puebla la sonda, "
            "`ai-trader signals depth`, no se escribe a mano).\n"
            "\n"
            "ESTADO: 30 fuentes declaradas, 30 conectadas, 14 backtesteables. Dos temas de cinco "
            "no llegan al minimo de cobertura y el estudio de la capa tematica los declara no "
            "evaluables.\n"
            "\n"
            "TAREA: subir la profundidad historica utilizable, por este orden:\n"
            "(1) Para CADA fuente sin `history_from` o con poca: comprobar si su endpoint admite "
            "rango historico. Varias capturas piden 30 dias por defecto (ver `signals/capture.py`) "
            "cuando el proveedor sirve anos. Eso es profundidad gratis.\n"
            "(2) Priorizar las fuentes de `liquidation` y `vol_surface`: son los dos temas que hoy "
            "no se pueden evaluar hacia atras, asi que cada una de las suyas vale mas que una de "
            "un tema que ya llega.\n"
            "(3) Re-correr la sonda y dejar que `history_from` se actualice SOLO. Si una fuente no "
            "admite historia, se declara y no se toca: un `history_from` inventado convierte un "
            "backtest en ficcion.\n"
            "(4) Publicar la auditoria (`ai-trader signals audit`) y regenerar dashboard y docs "
            "con el recuento nuevo de backtesteables.\n"
            "\n"
            "NO conectes fuentes nuevas en esta entrada. Primero exprime las 30 que ya hay: una "
            "fuente conectada sin historia no aporta a la decision, y anadir la 31 solo aumenta el "
            "problema de multiples pruebas.\n"
            "\n"
            "Tests + .venv\\Scripts\\python.exe (poetry run esta roto) + ruff."
        ),
    },
    {
        "id": "signal-quality-review",
        "rank": 2,
        "group": "ahora",
        "priority": "critica",
        "title": "Análisis individualizado y calidad de dato, fuente por fuente",
        "line": "Datos", "status": "pendiente", "impact": "alto", "effort": "medio",
        "evidence": "Las 30 fuentes se agregan hoy en seis features (tono, intensidad y cobertura, "
                    "por activo y de mercado) y NUNCA se han mirado de una en una. Hay precedentes "
                    "de que bajar al dato cambia la respuesta: identificar el activo de una "
                    "tesorería por su precio implícito daba 25% de falsos positivos, y el efecto "
                    "Upbit resultó vivir en mercados de 248k $/día. Las dos cosas se vieron mirando "
                    "la fuente, no el agregado.",
        "why": "El radar promedia. Un promedio de una fuente rota y cuatro sanas parece sano, y "
               "esa es exactamente la forma en que un dato malo entra en una decisión sin que "
               "nadie lo vea. Antes de generar estrategias sobre estas señales hay que saber "
               "cuáles valen: qué latencia real tiene cada una, cuántos huecos, si su fecha es de "
               "publicación o de referencia, y si su distribución cambia de régimen a mitad de la "
               "serie.",
        "prompt": (
            "Proyecto ai-trader (Python). Las 30 fuentes de signals/catalog.py se normalizan "
            "(signals/normalize.py) y se agregan en seis features del radar "
            "(observation/signal_radar.py). Nadie las ha mirado NUNCA de una en una.\n"
            "\n"
            "TAREA: un informe de calidad POR FUENTE, publicado en data/signals/, que conteste "
            "para cada una y con cifras:\n"
            "(a) COBERTURA: dias con dato / dias del rango, y el hueco mas largo. Un hueco de tres "
            "semanas en una serie diaria no es 'ruido', es que la fuente estuvo caida.\n"
            "(b) LATENCIA REAL: distancia entre la fecha de referencia del dato y la fecha en que "
            "se pudo observar. Ojo con las que se fechan por PUBLICACION (el COT ya lo hace): usar "
            "la fecha de referencia en un backtest es mirar el futuro.\n"
            "(c) ESTABILIDAD: media, dispersion y fraccion de ceros por ano. Una serie cuya "
            "distribucion cambia a mitad del historico esta midiendo dos cosas distintas con el "
            "mismo nombre.\n"
            "(d) REDUNDANCIA: correlacion entre fuentes del mismo tema. Cinco fuentes que dicen lo "
            "mismo son una fuente con cinco votos, y el radar las cuenta como cinco.\n"
            "(e) UN VEREDICTO POR FUENTE que pueda ser NEGATIVO: apta / apta con reservas / no "
            "apta para decidir, con el motivo. Un informe que no puede suspender a nadie no es un "
            "control de calidad.\n"
            "\n"
            "Y engancharlo: las fuentes 'no aptas' tienen que quedar fuera del radar por "
            "CONFIGURACION, no porque alguien se acuerde. Declara cuantas quedan fuera y cuanto "
            "cambia la cobertura de cada tema al quitarlas.\n"
            "\n"
            "Tests + determinismo + .venv\\Scripts\\python.exe + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "strategy-generation-real",
        "rank": 3,
        "group": "ahora",
        "priority": "critica",
        "title": "Generar estrategias con las señales reales dentro, y operar las que aprueben",
        "line": "Estrategias", "status": "pendiente", "impact": "alto", "effort": "alto",
        "evidence": "El sustrato ya está: scoring/real_source.py rankea sobre cinco sub-ventanas "
                    "reales (2018-07 → 2025-12, 24 pares) con CPCV purgado y hold-out temporal, y "
                    "es el sustrato por defecto de run_optimization. Acepta `signals=` -- los "
                    "frames del archivo real -- pero va APAGADO por defecto: armar el radar "
                    "multiplica el coste por 7,9 (medido en el estudio de la capa temática), sobre "
                    "los 121 s por (configuración, unidad) que ya cuesta con el radar apagado.",
        "why": "Es el objetivo de todo lo anterior, y hasta ahora no se podía hacer: el "
               "optimizador puntuaba sobre mundos generados y las señales solo entraban en un "
               "estudio pareado aparte. Ahora las dos piezas encajan. El coste es real, así que la "
               "entrada incluye hacerlo tratable, no solo lanzarlo.",
        "prompt": (
            "Proyecto ai-trader (Python). run_optimization (src/ai_trader/scoring/optimize.py) ya "
            "puntua por defecto sobre el HISTORICO REAL via scoring/real_source.py: cinco "
            "sub-ventanas de 544 dias, CPCV C(6,2)=15 folds por unidad con purga, hold-out "
            "TEMPORAL (la ventana mas reciente no la ve el CEM), 24 pares cripto.\n"
            "\n"
            "TAREA: la cosecha de estrategias, con la capa de senal ARMADA.\n"
            "(1) Enchufa el archivo real: `signals=load_frames(...)` (signals/feed.py) llega hasta "
            "el motor por la costura que ya existe. Mide lo que cuesta ANTES de lanzar la corrida "
            "entera: el estudio tematico midio 7,9x, y sobre este sustrato eso son dias.\n"
            "(2) Hazlo tratable antes de lanzarlo. El cuello esta MEDIDO y no se ha atacado: el "
            "brazo armado reconstruye el radar UNA VEZ POR FOLD, y cada construccion normaliza el "
            "archivo entero. Cachear el radar por (ventana, brazo) es el primer sitio donde mirar. "
            "Paraleliza tambien por unidad si hace falta, y DECLARA por log cualquier subsampleo.\n"
            "(3) Corre las ocho familias, con y sin capa, y publica el resultado con su veredicto "
            "completo: gate de baselines, suelo de actividad, PBO y DSR.\n"
            "(4) HONESTIDAD, que aqui es lo que mas cuesta: son CUATRO unidades de train de un "
            "unico camino historico. El riesgo de sobreajuste se asume a proposito -- es mejor "
            "tener algo corriendo y atacarlo despues con evidencia de calendario que seguir en "
            "modo investigacion --, pero se DECLARA: numero de unidades efectivas, PBO y DSR al "
            "lado de cada cifra, y ni una sola frase que sugiera que este ranking generaliza.\n"
            "(5) Lo que apruebe entra en config/default.toml y empieza a operar en paper. Ese es "
            "el entregable: no un informe, estrategias corriendo.\n"
            "\n"
            "Tests + determinismo + .venv\\Scripts\\python.exe + ruff. Regenera dashboard y docs."
        ),
    },
    {
        "id": "paper-trading-live",
        "rank": 4,
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
        "id": "execution-latency-budget",
        "rank": 5,
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
        "id": "operational-symbol-guard",
        "rank": 6,
        "group": "despues",
        "priority": "media",
        "title": "Guarda operativa por símbolo (sanciones, deslistado, halt): lo que deja abierto no tener veto",
        "line": "Riesgo", "status": "pendiente", "impact": "medio", "effort": "bajo",
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
        "rank": 7,
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
        "rank": 8,
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
    {
        "id": "rl-full-run",
        "rank": 9,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Optimización CEM completa, ya con el juez validado",
        "line": "RL", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "ABSORBIDA por «Generar estrategias con las señales reales dentro». Estaba "
               "bloqueada esperando dos cosas: un sustrato con fidelidad y un juez con CPCV. El "
               "juez llegó (CPCV purgado dentro del optimizador) y el sustrato cambió de mundo -- "
               "ahora es el histórico real. Correr el CEM completo ya no es una entrada aparte: es "
               "el paso 3 de la entrada que lo sustituye, y allí lleva además la capa de señal.",
    },
    {
        "id": "synthetic-signal-emission",
        "rank": 10,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Que el generador emita las señales, y re-medir la transferencia de forma pareada",
        "line": "B/D", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "Ampliar el espacio de inputs del generador solo tenía sentido para salvar la hipótesis de que el sintético podía llegar a ordenar. Esa apuesta se cierra.",
    },
    {
        "id": "line-d-cpcv-two-stage-cem",
        "rank": 11,
        "group": "retiradas",
        "priority": "retirada",
        "title": "CPCV en dos etapas dentro del optimizador",
        "line": "D", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "HECHA a medias, y el resto muere con el sintético: el CEM ya puntúa con CPCV purgado (scoring/real_source.py). Las «dos etapas» eran sintético-criba + real-decide, y ya no hay criba.",
    },
    {
        "id": "validation-study-full-ensemble",
        "rank": 12,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Re-correr el estudio de validación con el ensemble completo",
        "line": "D", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "Re-correr un estudio sobre un sustrato aparcado. Lo publicado sigue describiendo lo que midió, que es todo lo que se le pedía.",
    },
    {
        "id": "pbo-blocks-scenario-aligned",
        "rank": 13,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Alinear los bloques del PBO con las fronteras de escenario",
        "line": "A", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "Los «escenarios» eran del generador. Sobre sustrato real la unidad de bloque es la sub-ventana, y ya lo es por construcción.",
    },
    {
        "id": "report-n-failed-with-reward",
        "rank": 14,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Reportar n_failed junto al reward",
        "line": "A", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "Rigor de informe de un estudio archivado.",
    },
    {
        "id": "dsr-independent-trials-caveat",
        "rank": 15,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Declarar que el DSR asume intentos independientes y el CEM no los produce",
        "line": "A", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "Matiz metodológico legítimo, pero es afinar el juez cuando lo que falta es tener algo en marcha que juzgar.",
    },
    {
        "id": "fidelity-rank-corr-ordering",
        "rank": 16,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Ordenación de colas y clustering entre activos",
        "line": "B", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "Mejorar la fidelidad del generador. Fidelidad no era el problema: se consiguió (98% de cobertura, nueve umbrales aceptados) y aun así no sirvió para ordenar.",
    },
    {
        "id": "real-substrate-primary-ranking",
        "rank": 17,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Mover el sustrato primario del ranking al histórico REAL",
        "line": "B/D", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "HECHA. scoring/real_source.py: sub-ventanas reales con CPCV purgado y hold-out temporal, y es el sustrato por defecto de run_optimization. Era la contingencia declarada, y se disparó.",
    },
    {
        "id": "weights-recalibrate-power",
        "rank": 18,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Re-medir lambda y kappa con más potencia estadística",
        "line": "A/C", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "La calibración se midió sobre la librería sintética. Re-medirla ahí no aporta nada; re-medirla sobre real es parte del ranking, no una entrada aparte.",
    },
    {
        "id": "designer-model-in-manifest",
        "rank": 19,
        "group": "retiradas",
        "priority": "retirada",
        "title": "Anotar el modelo de IA en el manifiesto de cada librería",
        "line": "E", "status": "retirada", "impact": "nulo", "effort": "-",
        "why": "Trazabilidad de un generador que ya no genera.",
    },
]


if __name__ == "__main__":
    build()
