"""
Generador de la documentacion funcional de AI-Trader (HTML -> PDF).

Inyecta CIFRAS VIVAS del repo (baratas: sin backtests) en una plantilla de prosa
orientada a auditoria. Se regenera cuando la herramienta evoluciona:

    .venv\\Scripts\\python.exe -m docs.build_docs

Salida: docs/metodologia.html (autocontenido, CSS print-optimizado; Ctrl+P -> PDF).
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from ai_trader.backtest.divergence_study import (
    DIVERGENCE_REPORT,
    STATUS_MEASURED as DIVERGENCE_MEASURED,
    load_divergence_report,
)
from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.backtest.metrics import DEFAULT_HEADLINE_WEIGHTS
from ai_trader.backtest.session_study import (
    SESSIONS_REPORT,
    US_KEY,
    load_sessions_report,
)
from ai_trader.observation.features import OWN_ASSET_FEATURES
from ai_trader.observation.regime import REGIME_FEATURES
from ai_trader.scoring.activity_study import activity_report_path, load_activity_report
from ai_trader.scoring.signal_study import (
    DEFAULT_LIBRARY_ID as SIGNAL_LIBRARY,
    load_signal_report,
    report_path as signal_report_path,
)
from ai_trader.scoring.search_space import get_space
from ai_trader.scoring.transfer_study import (
    DEFAULT_LIBRARY_ID as TRANSFER_LIBRARY,
    load_transfer_report,
    transfer_report_path,
)
from ai_trader.scoring.validation_study import VALIDATION_REPORT, load_validation_report
from ai_trader.shared.reports import load_report
from ai_trader.scoring.weight_calibration import (
    CALIBRATION_REPORT,
    grid_point,
    load_calibration_report,
)
from ai_trader.config import load_config
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.entities import ENTITY_OVERRIDES
from ai_trader.synthetic.fidelity import (
    CROSS_CORR_KEY,
    FIDELITY_BASELINE_LIBRARY,
    FIDELITY_LIBRARY,
    TARGET_METRIC_KEYS,
    fidelity_report_path,
    load_fidelity_report,
    metric,
)
from ai_trader.scoring.weight_study import NEW_FAMILIES
from ai_trader.strategies.mean_reversion import MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy
from ai_trader.strategies.registry import build_strategy
from ai_trader.synthetic.universe import DEFAULT_UNIVERSE, FACTOR_DESCRIPTIONS
from ai_trader.synthetic.store import SyntheticStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("ai_trader").setLevel(logging.WARNING)
logger = logging.getLogger("docs")

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "metodologia.html"
MACRO = {"GLD", "TLT", "UUP"}
_CONFIG_CACHE = None


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _count_tests() -> int:
    n = 0
    for p in (ROOT / "tests").glob("test_*.py"):
        try:
            n += p.read_text(encoding="utf-8").count("def test_")
        except Exception:  # noqa: BLE001
            pass
    return n


def _stylized(store: SyntheticStore, lib: str, n_paths: int = 2, n_scen: int = 12) -> dict | None:
    try:
        m = store.load_manifest(lib)
    except Exception:  # noqa: BLE001
        return None
    per, absac, exc = [], [], []
    for meta in m.scenarios[:n_scen]:
        acs, aacs, es = [], [], []
        for p in range(min(n_paths, m.n_paths)):
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
            per.append(float(np.mean(acs)))
            absac.append(float(np.mean(aacs)))
            exc.append(float(np.mean(es)))
    if not per:
        return None
    a = np.array(per)
    return {
        "spread": round(float(a.max() - a.min()), 3),
        "revert": int(np.sum(a < -0.05)),
        "trend": int(np.sum(a > 0.05)),
        "total": int(len(a)),
        "clustering": round(float(np.median(absac)), 3),
        "exceed": round(float(np.median(exc)) * 100.0, 2),
    }


def _calibration() -> dict | None:
    """Cifras del estudio de calibracion de los pesos del headline (data/calibration).

    Se leen del informe publicado, no se recalculan: el estudio cuesta cientos de
    backtests reales y la documentacion tiene que ser barata de regenerar. Si el informe
    no esta, la prosa se genera sin cifras en vez de inventarlas."""
    report = load_calibration_report(ROOT / CALIBRATION_REPORT)
    if not report:
        return None

    weights = DEFAULT_HEADLINE_WEIGHTS
    chosen = grid_point(report, weights.lambda_turnover, weights.kappa_maxdd)
    neutral = grid_point(report, 0.0, 0.0)
    if chosen is None or neutral is None:
        logger.warning("El informe de calibracion no cubre los pesos por defecto")
        return None

    audit = report["cost_audit_active"]
    plan = report["plan"]
    return {
        "lambda": weights.lambda_turnover,
        "kappa": weights.kappa_maxdd,
        "n_configs": len(report["configs"]["kept"]),
        "n_samples": report["configs"]["n_samples"],
        "n_backtests": len(report["configs"]["kept"]) * report["configs"]["n_samples"],
        "library": plan["library_id"],
        "n_train": len(report["split"]["train"]),
        "n_validation": len(report["split"]["validation"]),
        "lambdas": report["grid"]["lambdas"],
        "kappas": report["grid"]["kappas"],
        "grid": {
            (p["lambda_turnover"], p["kappa_maxdd"]): p["rank_ic_mean"]
            for p in report["grid"]["points"]
        },
        "ic": chosen["rank_ic_mean"],
        "ic_se": chosen["rank_ic_se"],
        "ic_neutral": neutral["rank_ic_mean"],
        "ic_neutral_se": neutral["rank_ic_se"],
        "gain": chosen["rank_ic_gain"],
        "gain_se": chosen["rank_ic_gain_se"],
        "gap": chosen["selection_gap_norm"],
        "gap_neutral": neutral["selection_gap_norm"],
        "best": report["ranked_by_stability"][0],
        # Cuantas configuraciones distintas gana alguien en toda la rejilla: 1 significa
        # que los pesos no cambian la decision, que es el hallazgo central del estudio.
        "n_winners": len({p["selected_config"] for p in report["grid"]["points"]}),
        "prev": grid_point(report, 0.5, 1.0),
        "worst": grid_point(report, max(report["grid"]["lambdas"]), max(report["grid"]["kappas"])),
        "n_active": len(report["active_subset"]["configs"]),
        "cost_rate": audit["cost_rate"],
        "measured_fee_rate": audit["measured_fee_rate"],
        "implied_lambda": audit["implied_lambda_median"],
        "implied_p25": audit["implied_lambda_p25"],
        "implied_p75": audit["implied_lambda_p75"],
        "median_turnover": audit["median_turnover"],
        "median_vol": audit["median_volatility"],
        "sharpe_drag": audit["median_sharpe_drag"],
        "share_pct": 100.0 * weights.lambda_turnover / audit["implied_lambda_median"],
    }


def _fidelity() -> dict | None:
    """Cifras del estudio de fidelidad sintetico-vs-real (data/fidelity).

    Igual que la calibracion: se leen del informe publicado, no se recalculan. Medirlo
    exige descargar ocho anos de historico real y recorrer la libreria entera; la
    documentacion tiene que ser barata de regenerar. Sin informe, prosa sin cifras."""
    report = load_fidelity_report(ROOT / fidelity_report_path(FIDELITY_LIBRARY))
    if not report:
        logger.warning("Sin informe de fidelidad: la seccion 2.10 saldra degradada")
        return None

    plan = report["plan"]
    by_key = {m["key"]: m for m in report["metrics"]}
    rows = [
        *(by_key[k] for k in TARGET_METRIC_KEYS),
        report["cross_correlation"],
        *(m for m in report["metrics"] if m["key"] not in TARGET_METRIC_KEYS),
    ]

    # La libreria ANTERIOR, medida con el mismo harness y la misma ventana real. Sin el
    # antes, una correccion medida no se distingue de una afirmacion.
    baseline = load_fidelity_report(ROOT / fidelity_report_path(FIDELITY_BASELINE_LIBRARY))
    before = None
    if baseline:
        prev = {m["key"]: m for m in baseline["metrics"]}
        prev[CROSS_CORR_KEY] = baseline["cross_correlation"]
        before = {
            "library": baseline["plan"]["library_id"],
            "by_key": {
                key: {"synth": row["synth_median"], "coverage": row["coverage_pct"]}
                for key, row in prev.items()
            },
            "coverage_mean_pct": baseline["summary"]["coverage_mean_pct"],
            "accepted": baseline["summary"].get("accepted"),
        }
    else:
        logger.warning("Sin informe de %s: la seccion 2.10 no podra comparar",
                       FIDELITY_BASELINE_LIBRARY)

    return {
        "before": before,
        "acceptance": report["acceptance"],
        "library": plan["library_id"],
        "exchange": plan["exchange"],
        "start": plan["real_window"]["start"],
        "end": plan["real_window"]["end"],
        "window_days": plan["window_days"],
        "step_days": plan["step_days"],
        "n_scenarios": plan["n_scenarios"],
        "n_paths": plan["n_paths"],
        "missing": plan["missing_symbols"],
        "rows": [
            {
                "label": row["label"],
                "decimals": 3 if row["key"] == "cross_corr" else metric(row["key"]).decimals,
                "real": row["real_median"],
                "synth": row["synth_median"],
                "ratio": row["ratio"],
                "rank_corr": row["rank_corr"],
                "coverage": row["coverage_pct"],
                "before": (before or {}).get("by_key", {}).get(row["key"]),
                "is_cross": row["key"] == "cross_corr",
                "is_target": row["key"] == "cross_corr" or row["key"] in TARGET_METRIC_KEYS,
            }
            for row in rows
        ],
        "kurtosis": by_key["excess_kurtosis"],
        "exceed": by_key["exceed_3sigma_pct"],
        "clustering": by_key["ac_abs1"],
        "autocorr": by_key["ac1"],
        "vol": by_key["vol_annual_pct"],
        "cross": report["cross_correlation"],
        **report["summary"],
        "generated_at": report["generated_at"][:10],
    }


def _transfer() -> dict | None:
    """Cifras del estudio de transferencia de ranking real-vs-sintetico (data/transfer).

    Es la pregunta que la fidelidad no responde: si el mundo sintetico ORDENA las
    estrategias como el mercado. Se lee del informe publicado, no se recalcula: son 208
    unidades de 15 ventanas de backtest real cada una."""
    report = load_transfer_report(ROOT / transfer_report_path(TRANSFER_LIBRARY))
    if not report:
        logger.warning("Sin informe de transferencia: la seccion 4.11 saldra degradada")
        return None

    plan = report["plan"]
    return {
        "library": plan["library_id"],
        "is_fallback": plan["library_is_fallback"],
        "symbols": plan["symbols"],
        "n_omitted": len(plan["real"]["symbols_omitted"]),
        "min_history_days": plan["real"]["min_history_days"],
        "head_discarded": plan["real"]["head_discarded_days"],
        "n_sub_windows": len(plan["real"]["sub_windows"]),
        "real_start": plan["real"]["window"]["start"][:10],
        "real_end": plan["real"]["window"]["end"][:10],
        "n_samples": plan["synthetic"]["n_samples"],
        "study_seed": plan["grid"]["study_seed"],
        # Las familias salen del INFORME y no de la constante del modulo: el dia que haya dos
        # informes con rejillas distintas al lado, cada uno tiene que describir la suya.
        "families": list(plan["grid"]["families"]),
        "validation": plan["validation"],
        "n_configs": report["transfer"]["n_configs"],
        "rho": report["transfer"]["spearman"],
        "boot": report["transfer"]["bootstrap_blocks"],
        "boot_configs": report["transfer"]["bootstrap_configs"],
        "permutation": report["transfer"]["permutation"],
        "top_k": report["transfer"]["top_k"],
        "discrepancies": report["transfer"]["discrepancies"],
        "activity": report["transfer"]["activity"],
        "verdict": report["verdict"],
        "caveats": report["caveats"],
        "rows": [
            {
                "config_id": c["config_id"],
                "family": c["family"],
                "reward_real": c["reward_real"],
                "reward_synthetic": c["reward_synthetic"],
                "rank_real": c["rank_real"],
                "rank_synthetic": c["rank_synthetic"],
                "delta": c["rank_delta"],
                "trades_real": c["trades_per_fold"]["real"],
                "active": c["active"],
            }
            for c in sorted(report["configs"], key=lambda c: c["rank_real"])
        ],
        "leakage": report["leakage"],
        "generated_at": report["generated_at"][:10],
    }


def _signal_channel() -> dict | None:
    """Cifras del barrido de rho (data/signal_channel): el break-even del IC, su grupo de
    control y la certificacion de que el canal entrega lo que declara. Se lee del informe
    publicado: son 640 unidades de 15 ventanas de backtest cada una."""
    report = load_signal_report(ROOT / signal_report_path(SIGNAL_LIBRARY))
    if not report:
        logger.warning("Sin informe del canal sintetico: la seccion 4.12 saldra degradada")
        return None

    plan = report["plan"]
    certification = {c["cell_id"]: c for c in report["channel_certification"]}
    return {
        "library": plan["library_id"],
        "symbols": plan["symbols"],
        "n_configs": plan["grid"]["n_configs"],
        "channel": plan["sweep"]["channel_fixed"],
        "gate_param": plan["grid"]["injected_param"],
        "gate_value": plan["grid"]["injected_value"],
        "split": plan["synthetic"]["split"],
        "n_samples": plan["synthetic"]["n_samples"],
        "validation": plan["validation"],
        "criterion": report["criterion"],
        "break_even": report["break_even"],
        "gate_cost": report["gate_cost"],
        "reproduction": report["reproduction"],
        "determinism": report.get("determinism"),
        "rows": [
            {
                "cell_id": c["cell_id"],
                "arm": c["arm"],
                "rho": c["rho"],
                "lead_days": c["lead_days"],
                "expected_ic": c["expected_ic"],
                "measured_ic": certification.get(c["cell_id"], {}).get("ic_median"),
                "past_leak": certification.get(c["cell_id"], {}).get("past_leak_median"),
                "selected": c["selected"],
                "reward": c["selected_reward_validation"],
                "baseline": c["baseline_reward_validation"],
                "margin": c["margin"],
                "beats": c["beats"],
                "n_beating": c["n_beating_baseline"],
            }
            for c in report["cells"]
        ],
        "generated_at": report["generated_at"][:10],
    }


def _activity() -> dict | None:
    """Cifras del suelo de actividad (data/activity): las dos condiciones, la regla que
    eligio el umbral y su efecto medido sobre el gate. Se lee del informe publicado."""
    report = load_activity_report(ROOT / activity_report_path(TRANSFER_LIBRARY))
    if not report:
        logger.warning("Sin informe de actividad: la seccion 4.10 saldra degradada")
        return None

    return {
        "library": report["source"]["library_id"],
        "floor": report["floor"],
        "decision": report["decision"],
        "mechanism": report["mechanism"],
        "band": report["band"],
        "sweep": report["sweep"],
        "reproducibility": report["reproducibility"],
        "gate": report["gate"],
        "rows": [
            {
                "config_id": c["config_id"],
                "reward": c["real"]["reward"],
                "trades_per_window": c["real"]["trades_per_window"],
                "zero_window_pct": c["real"]["zero_window_pct"],
                "rankable": c["real"]["rankable"],
            }
            for c in sorted(report["configs"], key=lambda c: -c["real"]["reward"])
        ],
        "generated_at": report["generated_at"][:10],
    }


def _validation() -> dict | None:
    """Cifras del estudio de validacion multiventana (data/validation).

    Igual que la calibracion y la fidelidad: se leen del informe publicado, no se
    recalculan. Cada unidad del estudio son ~20 ventanas de backtest real."""
    report = load_validation_report(ROOT / VALIDATION_REPORT)
    if not report:
        logger.warning("Sin informe de validacion: la seccion 4.8 saldra degradada")
        return None

    plan = report["plan"]
    rows = report["rows"]
    return {
        "library": plan["library_id"],
        "n_units": len(rows),
        "n_configs": len(plan["config_ids"]),
        "n_samples": len(plan["scenario_ids"]) * plan["n_paths"],
        "n_folds_wf": plan["n_folds"],
        "n_folds_cpcv": rows[0]["cpcv"]["n_folds"] if rows else 0,
        "n_groups": plan["n_groups"],
        "n_test_groups": plan["n_test_groups"],
        "purge": plan["purge_days"],
        "embargo": rows[0]["walk_forward"]["embargo_days"] if rows else 0,
        "opt_wf": report["optimism"]["walk_forward"],
        "opt_cpcv": report["optimism"]["cpcv"],
        "opt_tail": report["optimism"]["vs_tail"],
        "std": report["dispersion"]["walk_forward_std"],
        "std_cpcv": report["dispersion"]["cpcv_std"]["median"],
        "range": report["dispersion"]["walk_forward_range"],
        "svn": report["signal_vs_noise"],
        "rank": report["rank_agreement"],
        "flips": report["decision_flips"],
        "leakage": report["leakage"],
        "generated_at": report["generated_at"][:10],
    }


def _sessions() -> dict | None:
    """Cifras del estudio de descomposicion por sesion horaria (data/sessions).

    Es lo que convierte la convencion intrabar de la 3.4 —que hasta ahora se justificaba
    solo por prudencia— en una limitacion MEDIDA con su umbral. Se lee del informe
    publicado, no se recalcula: son seis anos de barras 1H de 24 pares."""
    report = load_sessions_report(ROOT / SESSIONS_REPORT)
    if not report:
        logger.warning("Sin informe de sesiones: la seccion 3.5 saldra degradada")
        return None

    plan = report["plan"]
    latency = {r["hours"]: r for r in report["latency"]["rows"]}
    overall = report["overall"]["sessions"]
    return {
        "window": plan["window"],
        "exchange": plan["exchange"],
        "thresholds": plan["thresholds"],
        "reference_cost_bps": plan["reference_cost_bps"],
        "n_symbols": report["overall"]["n_symbols"],
        "n_days": report["overall"]["n_days"],
        "sessions": report["sessions"],
        "overall": overall,
        "gap": report["gap"],
        "latency_1h": latency.get(1),
        "latency_rows": report["latency"]["rows"],
        "trend": report["trend"],
        "verdicts": report["verdicts"],
        "us": overall[US_KEY],
        "us_key": US_KEY,
        "n_cohort": len(report["cohort"]),
        "generated_at": report["generated_at"][:10],
    }


def _divergence() -> dict | None:
    """Cifras del estudio de divergencia live-vs-backtest (data/live/divergence.json).

    Es lo que llena el capitulo 5, y el unico del documento cuyo estado normal durante
    meses sera "sin potencia". Ese estado tambien se publica: decir cuantos dias faltan,
    medidos, es una afirmacion que se puede comprobar, y la prosa que sustituye ("necesita
    meses de operacion") no lo era."""
    report = load_divergence_report(ROOT / DIVERGENCE_REPORT)
    if not report:
        logger.warning("Sin informe de divergencia: la seccion 5.4 saldra degradada")
        return None

    measured = report.get("status") == DIVERGENCE_MEASURED
    out = {
        "measured": measured,
        "journal": report["journal"],
        "power": report["power"],
        "thresholds": report["plan"]["thresholds"],
        "reference_cost_bps": report["plan"]["reference_cost_bps"],
        "generated_at": report["generated_at"][:10],
    }
    if not measured:
        return out

    price = report["fill_price"]
    return {
        **out,
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


def _app_config():
    """El config operado, cargado una sola vez: lo consumen la 2.1 y el capitulo 3."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config(ROOT / "config" / "default.toml")
    return _CONFIG_CACHE


def _market() -> dict:
    """Seccion 2.1: la captura de datos REALES (proveedores, universo, cache).

    No lee ningun informe: son constantes del sistema que corre. Es la mitad del
    capitulo de datos que hasta ahora no estaba escrita en ningun sitio -- la
    documentacion empezaba por el generador sintetico, como si los datos reales no
    existieran, cuando toda la evidencia publicada (fidelidad, transferencia,
    sesiones) sale de ellos."""
    from ai_trader.data.cache import CACHE_DIR
    from ai_trader.data.providers.ccxt_crypto import CCXTCryptoConfig

    cfg = _app_config()
    ccxt_cfg = CCXTCryptoConfig()
    symbols = list(cfg.runner.symbols)
    return {
        "n_synthetic_assets": len(DEFAULT_UNIVERSE.assets),
        "n_symbols": len(symbols),
        "symbols": symbols,
        "n_crypto": sum(1 for s in symbols if "/" in s),
        "lookback_days": cfg.runner.lookback_days,
        "exchange": ccxt_cfg.exchange_id,
        "batch": ccxt_cfg.max_batch_size,
        "cache_dir": str(CACHE_DIR).replace("\\", "/"),
    }


def _trade() -> dict:
    """Capitulo 3: las constantes que gobiernan UN trade, leidas del config operado.

    Van juntas a proposito: el limite de riesgo, el coste de ejecucion y el techo de
    capacidad son la misma decision vista en tres sitios, y separarlas es lo que
    permitia que la documentacion describiera un trade que el sistema no ejecuta."""
    cfg = _app_config()
    slip = cfg.execution.slippage
    return {
        "fee_rate": cfg.execution.fee_rate,
        "slippage_bps": cfg.execution.slippage_bps,
        "max_participation": cfg.execution.max_participation,
        "vol_coef": slip.vol_coef,
        "impact_coef": slip.impact_coef,
        "max_slippage_bps": slip.max_slippage_bps,
        "starting_equity": DEFAULT_STARTING_EQUITY,
        "max_position_size_usd": cfg.risk.max_position_size_usd,
        "max_open_positions": cfg.risk.max_open_positions,
        "max_symbol_exposure_usd": cfg.risk.max_symbol_exposure_usd,
        "max_total_exposure_usd": cfg.risk.max_total_exposure_usd,
        "max_daily_loss_usd": cfg.risk.max_daily_loss_usd,
        "min_confidence": cfg.risk.min_confidence_per_trade,
        "default_stop_loss_pct": cfg.risk.default_stop_loss_pct,
        "default_take_profit_pct": cfg.risk.default_take_profit_pct,
        "max_stop_distance_pct": cfg.risk.max_stop_distance_pct,
        "risk_fraction_per_trade": cfg.risk.risk_fraction_per_trade,
        "max_holding_days": cfg.runner.max_holding_days,
        "cooldown_hours": cfg.runner.symbol_cooldown_hours,
        "max_trades_per_cycle": cfg.runner.max_trades_per_cycle,
    }


def _signals() -> dict:
    """
    La plataforma de ingesta de senales (seccion 2.2): catalogo, conexion y PROFUNDIDAD.

    A diferencia de los demas bloques, este NO lee un informe de estudio: el catalogo es
    codigo, la auditoria de entidades y de archivo es local, y la profundidad sale del
    registro de mediciones que escribe la sonda (`data/signals/history_depth.json`). Las
    cifras que importan —cuantas fuentes tienen adaptador y cuantas tienen historia
    MEDIDA— cambian escribiendo codigo y corriendo la sonda, no re-corriendo un estudio.
    """
    from ai_trader.observation.signal_radar import (
        MIN_SIGNAL_COVERAGE,
        POLARITY,
        SIGNAL_FEATURES,
        is_market_scoped,
    )
    from ai_trader.signals.audit import audit_archive, audit_entities
    from ai_trader.signals.capture import connect_adapters
    from ai_trader.signals.catalog import CATALOG, catalog_summary
    from ai_trader.signals.depth import DEPTH_LEDGER, load_ledger
    from ai_trader.signals.events import (
        EVENT_POOL_REPORT,
        event_encoding_spec,
        is_event_source,
        is_price_map_source,
        load_pool_report,
    )
    from ai_trader.signals.adapters.treasuries import COHORT_REPORT, load_cohort_report
    from ai_trader.signals.liquidity import ADV_LEDGER, liquidity_summary
    from ai_trader.signals.normalize import normalization_spec
    from ai_trader.signals.source import connected_keys
    from ai_trader.signals.store import SignalStore

    universe = list(load_config(ROOT / "config" / "default.toml").runner.symbols)
    connect_adapters()
    entities = audit_entities(universe)
    archive = audit_archive(SignalStore(ROOT / "data" / "signals_raw"))

    ledger = load_ledger(ROOT / DEPTH_LEDGER) or {}
    depth = {row["source_key"]: row for row in ledger.get("sources") or []}
    pool = load_pool_report(ROOT / EVENT_POOL_REPORT) or {}
    dat = load_cohort_report(ROOT / COHORT_REPORT) or {}
    pool_rows = pool.get("sources") or {}

    return {
        "summary": catalog_summary(),
        "n_connected": len(connected_keys()),
        "n_measured": sum(1 for r in depth.values() if r.get("first_day")),
        "records": archive.records,
        "normalization": normalization_spec(),
        "depth_measured_at": (ledger.get("generated_at") or "")[:10] or "—",
        # El radar: como llegan las veintinueve fuentes a una decision.
        "radar": {
            "features": list(SIGNAL_FEATURES),
            "min_coverage_pct": round(MIN_SIGNAL_COVERAGE * 100),
            "n_polarity": len(POLARITY),
            "n_market": sum(1 for s in CATALOG if is_market_scoped(s)),
            "n_asset": sum(1 for s in CATALOG if not is_market_scoped(s)),
            "n_event": sum(1 for s in CATALOG if is_event_source(s)),
            "n_price_map": sum(1 for s in CATALOG if is_price_map_source(s)),
            "n_continuous": sum(
                1 for s in CATALOG if not (is_event_source(s) or is_price_map_source(s))
            ),
        },
        # El ADV tipico de las entidades donde cada senal existe. Ver signals/liquidity.py.
        "liquidity": {
            **liquidity_summary(ROOT / ADV_LEDGER),
            "rows": [
                (
                    name,
                    str(row.get("n_entities") or 0),
                    str(row.get("n_traded") or 0),
                    f"{round(row.get('median_usd') or 0):,}".replace(",", " "),
                    f"{round(row.get('p10_usd') or 0):,}".replace(",", " "),
                    f"{round(row.get('max_usd') or 0):,}".replace(",", " "),
                )
                for name, row in sorted(
                    (liquidity_summary(ROOT / ADV_LEDGER).get("venues") or {}).items()
                )
            ],
        },
        # Tesorerias cotizadas. La tabla que se publica no es la distribucion —hoy no hay—
        # sino POR QUE no la hay, compania a compania: es la unica forma de que una
        # cobertura baja se distinga de un filtro mal escrito.
        "dat": {
            **{
                key: dat.get(key)
                for key in (
                    "companies", "companies_examined", "pooled_observations", "rows",
                )
            },
            "median_lag_days": dat.get("median_disclosure_lag_days"),
            "policy": dat.get("policy") or {},
            "assets": dat.get("assets") or {},
            "rows": [
                (reason, str(count))
                for reason, count in sorted(
                    (dat.get("rejections") or {}).items(), key=lambda kv: (-kv[1], kv[0])
                )
            ],
        },
        "price_maps": {
            "spec": event_encoding_spec().get("price_map") or {},
            "rows": [
                (
                    key,
                    f"{row.get('snapshots', 0):,}".replace(",", " "),
                    str(row.get("entities", 0)),
                    row.get("first_day") or "—",
                    row.get("last_day") or "—",
                )
                for key, row in sorted(
                    ((pool.get("price_maps") or {}).get("sources") or {}).items()
                )
            ],
        },
        "events": {
            "spec": event_encoding_spec(),
            "pooled_total": pool.get("pooled_events_total", 0),
            "rows": [
                (
                    key,
                    f"{row.get('pooled_events', 0):,}".replace(",", " "),
                    str(row.get("entities", 0)),
                    "si" if row.get("announced") else "no",
                    row.get("first_day") or "—",
                    row.get("last_day") or "—",
                )
                for key, row in sorted(pool_rows.items())
            ],
        },
        "entities": {
            "n_symbols": entities.n_symbols,
            "coverage_pct": round(entities.coverage_pct, 2),
            "by_source": entities.by_source,
            "n_entities": len({r.key for r in entities.refs if r.resolved}),
            "n_overrides": len(ENTITY_OVERRIDES),
        },
        "rows": [
            (
                s.key,
                (
                    "evento" if is_event_source(s)
                    else "mapa precios" if is_price_map_source(s)
                    else "continua"
                ),
                s.pit,
                str(len(s.features)),
                "si" if s.key in set(connected_keys()) else "no",
                s.history_from.isoformat() if s.history_from else "—",
                (depth.get(s.key) or {}).get("first_day") or "—",
                f"{(depth.get(s.key) or {}).get('days', 0):,}".replace(",", " "),
                (
                    f"{round(s.typical_adv_usd):,}".replace(",", " ")
                    if s.typical_adv_usd
                    else "—"
                ),
            )
            for s in CATALOG
        ],
    }


def collect() -> dict:
    store = SyntheticStore(ROOT / "data" / "synthetic")
    facts: dict = {}

    facts["commit"] = _git("rev-parse", "--short", "HEAD")
    facts["commit_count"] = _git("rev-list", "--count", "HEAD")
    facts["date"] = _git("log", "-1", "--format=%cd", "--date=short")
    facts["n_tests"] = _count_tests()

    for lib in ("ai_v1", "ai_v2"):
        try:
            m = store.load_manifest(lib)
            facts[lib] = {"scenarios": m.num_scenarios, "paths": m.n_paths,
                          "samples": m.num_samples, "horizon": m.horizon_days}
        except Exception:  # noqa: BLE001
            facts[lib] = None
    facts["sf_v1"] = _stylized(store, "ai_v1")
    facts["sf_v2"] = _stylized(store, "ai_v2")

    assets = DEFAULT_UNIVERSE.assets
    facts["n_assets"] = len(assets)
    facts["n_crypto"] = sum(1 for a in assets if a.symbol.endswith("/USDT"))
    facts["n_macro"] = sum(1 for a in assets if a.symbol in MACRO)
    facts["n_equity"] = facts["n_assets"] - facts["n_crypto"] - facts["n_macro"]
    facts["factors"] = [(f, FACTOR_DESCRIPTIONS.get(f, "")) for f in DEFAULT_UNIVERSE.factors]

    facts["n_own"] = len(OWN_ASSET_FEATURES)
    facts["n_regime"] = len(REGIME_FEATURES)

    facts["market"] = _market()
    facts["trade"] = _trade()
    facts["calibration"] = _calibration()
    facts["fidelity"] = _fidelity()
    facts["transfer"] = _transfer()
    facts["validation"] = _validation()
    facts["sessions"] = _sessions()
    facts["divergence"] = _divergence()
    facts["activity"] = _activity()
    facts["signal_channel"] = _signal_channel()
    facts["signals"] = _signals()

    facts["mom_params"] = _params(CryptoMomentumStrategy().config)
    facts["mr_params"] = _params(MeanReversionStrategy().config)
    facts["space_mom"] = _space("crypto_momentum")
    facts["space_mr"] = _space("mean_reversion")
    # Las tematicas se ITERAN en vez de tener dos claves cada una: con la forma anterior,
    # anadir la novena familia obligaria a tocar este fichero Y las marcas literales de la
    # plantilla, en dos sitios que se desincronizan a la primera.
    facts["themed"] = _themed_families()
    facts["transfer_extended"] = _transfer_extended()
    facts["themes_real"] = _themes_real()
    facts["extended_grid"] = _extended_grid()

    return facts


THEMES_REPORT = Path("data") / "themes" / "report.json"
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


def _themes_real() -> dict | None:
    """
    La capa tematica medida contra ARCHIVO REAL, que es la unica evidencia del sistema donde
    la senal se enciende sobre mercado y no sobre un canal sintetico.

    Se funden varios informes porque `vol_term_structure` se admitio despues —cuando la
    cobertura medida contradijo al catalogo— y se corrio aparte. Fundir en vez de repetir las
    160 unidades ya medidas es correcto porque cada veredicto es INTERNO a su familia: la
    comparacion es ciega-contra-armada dentro de la misma configuracion, asi que ninguna cifra
    depende de que otras familias corran.

    Los metadatos del tema se toman del informe mas nuevo que traiga cobertura MEDIDA. El
    primero se genero antes de que la evaluabilidad se midiera y declara sus exclusiones con un
    motivo derivado del catalogo que, para `vol_surface`, resulto ser falso.
    """
    loaded = []
    for path in (THEMES_REPORT, *THEMES_EXTRA):
        rep = load_report(ROOT / path)
        if rep:
            loaded.append((path, rep))
    if not loaded:
        return None

    families, seen = [], set()
    for _, rep in loaded:
        for fam in rep["families"]:
            if fam["family"] not in seen:
                families.append(fam)
                seen.add(fam["family"])

    measured = [(path, rep) for path, rep in loaded if rep["plan"]["themes"].get("measured")]
    meta_path, meta = (measured or loaded)[-1]
    plan = meta["plan"]
    return {
        "families": families,
        "families_skipped": _merge_skipped(
            loaded, seen, plan["themes"].get("measured") or {}
        ),
        "n_helps": sum(1 for f in families if f["verdict"] == "la_capa_ayuda"),
        "n_moved": sum(1 for f in families if f["n_windows_where_the_layer_moved"] > 0),
        "themes": plan["themes"],
        "windows": plan["windows"],
        "n_symbols": len(plan["symbols"]),
        "min_paired_windows": plan["min_paired_windows"],
        "n_failed_units": sum(rep["n_failed_units"] for _, rep in loaded),
        "metadata_from": str(meta_path).replace("\\", "/"),
        "coverage_is_measured": bool(plan["themes"].get("measured")),
    }


def _extended_grid() -> dict | None:
    """Los tres estudios que se re-corrieron sobre la rejilla de OCHO familias.

    Van juntos porque cuentan una sola cosa: que cambia al pasar de 16 candidatos a 64. Cada
    uno se lee de su informe y ninguno sustituye al congelado, que sigue publicandose al lado.
    """
    signal = load_signal_report(ROOT / signal_report_path("ai_v4"))
    frozen_signal = load_signal_report(ROOT / signal_report_path(SIGNAL_LIBRARY))
    validation = load_validation_report(ROOT / VALIDATION_REPORT.with_name("report_ai_v4.json"))
    frozen_validation = load_validation_report(ROOT / VALIDATION_REPORT)
    calib = load_calibration_report(ROOT / CALIBRATION_REPORT.with_name("report_ai_v4.json"))
    frozen_calib = load_calibration_report(ROOT / CALIBRATION_REPORT)
    if not (signal and validation and calib):
        return None

    def best(report):
        pts = report["active_subset"]["points"]
        top = max(pts, key=lambda x: x["rank_ic_mean"])
        base = next(x for x in pts
                    if x["lambda_turnover"] == 0.0 and x["kappa_maxdd"] == 0.0)
        return top, base

    top, base = best(calib)
    ftop, fbase = best(frozen_calib) if frozen_calib else (None, None)

    return {
        "signal": {
            "verdict": signal["break_even"]["verdict"],
            "margins": signal["break_even"]["by_lead"][0]["margins"],
            "identical_to_frozen": (
                frozen_signal is not None
                and json.dumps(signal["break_even"], sort_keys=True)
                == json.dumps(frozen_signal["break_even"], sort_keys=True)
            ),
            "n_configs": len(signal["configs"]),
            "gate_cost": signal["gate_cost"]["delta_validation"],
            "reproduction": signal["reproduction"]["identical"],
            "determinism": len(signal["determinism"]["mismatches"]) == 0,
        },
        "validation": {
            "n_configs": len(validation["plan"]["config_ids"]),
            "vs_tail": validation["optimism"]["vs_tail"]["median"],
            "frozen_vs_tail": (frozen_validation or {}).get("optimism", {})
                .get("vs_tail", {}).get("median"),
            "flips": validation["decision_flips"],
            "frozen_flips": (frozen_validation or {}).get("decision_flips"),
            "leakage_clean": validation["leakage"]["clean"],
            "folds_audited": validation["leakage"]["folds_audited"],
        },
        "weights": {
            "n_active": len(calib["active_subset"]["configs"]),
            "n_configs": len(calib["configs"]["kept"]),
            "best": (top["lambda_turnover"], top["kappa_maxdd"]),
            "best_ic": top["rank_ic_mean"],
            "base_ic": base["rank_ic_mean"],
            "gain": top["rank_ic_gain"],
            "gain_se": top["rank_ic_gain_se"],
            "frozen_best": (ftop["lambda_turnover"], ftop["kappa_maxdd"]) if ftop else None,
            "implied_lambda": calib["cost_audit_active"]["implied_lambda_median"],
            "frozen_implied_lambda": (frozen_calib or {}).get("cost_audit_active", {})
                .get("implied_lambda_median"),
        },
    }


def _transfer_extended() -> dict | None:
    """
    El estudio de transferencia con las OCHO familias, y su control de rejilla.

    Se lee aparte del congelado (`_transfer`, que sigue apuntando a ai_v3 con dos familias)
    porque son dos evidencias distintas y mezclarlas seria exactamente lo que la separacion
    aditiva existe para impedir. Si alguno de los dos informes no esta, la seccion se genera
    sin cifras en vez de inventarlas.
    """
    extended = load_transfer_report(ROOT / transfer_report_path("ai_v4"))
    control = load_transfer_report(
        ROOT / transfer_report_path("ai_v3", Path("data") / "transfer" / "control_8f")
    )
    if not extended:
        return None

    activity = load_activity_report(ROOT / activity_report_path("ai_v4"))
    transfer = extended["transfer"]
    plan = extended["plan"]

    # El residuo del control: cuantos campos del informe difieren aparte de la libreria. Es
    # la cifra que convierte "el mundo no aporta" en algo comprobado y no argumentado.
    residue = None
    if control:
        residue = sum(
            1
            for key in ("transfer", "rankings", "eligibility", "blocks", "baselines")
            if json.dumps(extended.get(key), sort_keys=True)
            != json.dumps(control.get(key), sort_keys=True)
        )

    return {
        "library": plan["library_id"],
        "n_configs": plan["grid"]["n_configs"],
        "n_families": len(plan["grid"]["families"]),
        "spearman": transfer["spearman"],
        "verdict": extended["verdict"]["key"],
        "threshold": extended["verdict"]["threshold"],
        "control_residue": residue,
        "activity_floor": (activity or {}).get("floor", {}).get("min_median_trades_per_window"),
    }


# Las seis tematicas, con la prosa que no es derivable —que mira cada una y con que fuentes—
# junto a los parametros y el espacio de busqueda, que si lo son. El `raise` de mas abajo es
# lo que impide publicar una vista que diga "seis" mientras los estudios miden otra cosa.
THEMED_PROSE: dict[str, dict[str, str]] = {
    "liquidation_cascade": {
        "name": "Cascada de liquidaciones",
        "theme": "liquidation",
        "idea": (
            "Cripto no se mueve, se descuelga: el precio entra en un cúmulo de precios de "
            "liquidación, el flujo forzado acelera el movimiento y, cuando el combustible se "
            "agota, retrocede. El precio ve el agotamiento —barra de capitulación: rango muy "
            "por encima del ATR y cierre pegado al extremo—; la señal ve cuánto combustible "
            "queda debajo, y con eso distingue comprar la última capitulación de comprar la "
            "primera de tres."
        ),
    },
    "vol_term_structure": {
        "name": "Estructura temporal de volatilidad",
        "theme": "vol_surface",
        "idea": (
            "La superficie de opciones es lo único del catálogo que cotiza el futuro en vez "
            "de resumir el pasado. La volatilidad realizada se comprime antes de expandirse "
            "—y eso lo ve el precio—, pero la dirección de la expansión es lo que el skew "
            "está pagando, y un vencimiento que concentra el interés abierto es un día con "
            "gamma en el que el precio tiende a quedarse clavado."
        ),
    },
    "event_calendar_drift": {
        "name": "Deriva de calendario",
        "theme": "macro",
        "idea": (
            "Hay días que se saben con meses de antelación y el mercado se coloca antes. "
            "Este tema NO dice hacia dónde —de sus seis fuentes solo una tiene polaridad "
            "declarada, así que su tono es ~0 por construcción— y por eso esta primitiva no "
            "declara umbral de tono: la dirección la pone la deriva del precio y la señal "
            "decide si se opera y cuánto."
        ),
    },
    "attention_ignition": {
        "name": "Ignición de atención",
        "theme": "attention",
        "idea": (
            "La atención minorista es la demanda de último recurso de cripto: llega tarde, "
            "lenta e insensible al precio, así que produce continuación y no reversión. El "
            "precio ve la barra de ignición —volumen múltiplo de su mediana y cierre pegado "
            "al máximo—; la señal ve el listado en Upbit, el diferencial de visibilidad "
            "Corea−EE.UU. y las búsquedas en Naver. Solo largo, por tesis."
        ),
    },
    "flow_persistence": {
        "name": "Persistencia de flujo",
        "theme": "flow",
        "idea": (
            "El tema con mejor materia prima del catálogo —once de doce fuentes con "
            "polaridad razonada, ocho con historia medida— y lo que mide tiene una propiedad "
            "que casi ninguna señal tiene: persistencia. Por eso el núcleo correcto no es "
            "una ruptura sino el retroceso dentro de una tendencia persistente: comprar la "
            "pausa mientras el dinero sigue entrando."
        ),
    },
    "signal_composite": {
        "name": "Compuesto de señales",
        "theme": "los cinco",
        "idea": (
            "La única primitiva que ve los cinco temas a la vez, y por tanto la única que "
            "puede cobrar la raíz de la ley fundamental del gestor activo: el IC agregado de "
            "los cinco canales declarados vale 0,074 frente a 0,048 del mejor tema solo. Su "
            "núcleo de precio es deliberadamente pobre —piso de ATR y un giro reciente de la "
            "media corta—, así que ciega no aporta nada sobre el momentum: toda su tesis "
            "está en la capa."
        ),
    },
}


def _themed_families() -> list[dict]:
    missing = set(NEW_FAMILIES) - set(THEMED_PROSE)
    if missing:
        raise ValueError(
            f"Familias temáticas sin prosa en docs/build_docs.py: {sorted(missing)}. "
            "Añadir una familia al grid y olvidar la documentación tiene que romper el build."
        )
    out = []
    for family in NEW_FAMILIES:
        prose = THEMED_PROSE[family]
        strategy = build_strategy(family)
        out.append(
            {
                "id": family,
                "name": prose["name"],
                "theme": prose["theme"],
                "idea": prose["idea"],
                "params": _params(strategy.config),
                "space": _space(family),
            }
        )
    return out


def _params(cfg) -> list[tuple[str, str]]:
    import dataclasses
    return [(f.name, str(getattr(cfg, f.name))) for f in dataclasses.fields(cfg)
            if f.name != "timeframe"]


def _space(stype: str) -> list[tuple[str, str]]:
    sp = get_space(stype)
    return [(d.name, f"[{d.low:g}, {d.high:g}]{' entero' if d.is_int else ''}") for d in sp.dims]


def build() -> None:
    logger.info("Recolectando cifras vivas...")
    facts = collect()
    from docs.template import render_html
    OUT.write_text(render_html(facts), encoding="utf-8")
    logger.info("Documentacion escrita en %s", OUT)


if __name__ == "__main__":
    build()
