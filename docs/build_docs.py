"""
Generador de la documentacion funcional de AI-Trader (HTML -> PDF).

Inyecta CIFRAS VIVAS del repo (baratas: sin backtests) en una plantilla de prosa
orientada a auditoria. Se regenera cuando la herramienta evoluciona:

    .venv\\Scripts\\python.exe -m docs.build_docs

Salida: docs/metodologia.html (autocontenido, CSS print-optimizado; Ctrl+P -> PDF).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from ai_trader.backtest.metrics import DEFAULT_HEADLINE_WEIGHTS
from ai_trader.backtest.session_study import (
    SESSIONS_REPORT,
    US_KEY,
    load_sessions_report,
)
from ai_trader.observation.features import OWN_ASSET_FEATURES
from ai_trader.observation.regime import REGIME_FEATURES
from ai_trader.scoring.activity_study import activity_report_path, load_activity_report
from ai_trader.scoring.search_space import get_space
from ai_trader.scoring.transfer_study import (
    DEFAULT_LIBRARY_ID as TRANSFER_LIBRARY,
    load_transfer_report,
    transfer_report_path,
)
from ai_trader.scoring.validation_study import VALIDATION_REPORT, load_validation_report
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
from ai_trader.strategies.mean_reversion import MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy
from ai_trader.synthetic.universe import DEFAULT_UNIVERSE, FACTOR_DESCRIPTIONS
from ai_trader.synthetic.store import SyntheticStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("ai_trader").setLevel(logging.WARNING)
logger = logging.getLogger("docs")

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "metodologia.html"
MACRO = {"GLD", "TLT", "UUP"}


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
        logger.warning("Sin informe de fidelidad: la seccion 2.8 saldra degradada")
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
        logger.warning("Sin informe de %s: la seccion 2.8 no podra comparar",
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
        logger.warning("Sin informe de transferencia: la seccion 2.9 saldra degradada")
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


def _activity() -> dict | None:
    """Cifras del suelo de actividad (data/activity): las dos condiciones, la regla que
    eligio el umbral y su efecto medido sobre el gate. Se lee del informe publicado."""
    report = load_activity_report(ROOT / activity_report_path(TRANSFER_LIBRARY))
    if not report:
        logger.warning("Sin informe de actividad: la seccion 5.6b saldra degradada")
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
        logger.warning("Sin informe de validacion: la seccion 3.7 saldra degradada")
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

    Es lo que convierte la convencion intrabar de la 3.3 —que hasta ahora se justificaba
    solo por prudencia— en una limitacion MEDIDA con su umbral. Se lee del informe
    publicado, no se recalcula: son seis anos de barras 1H de 24 pares."""
    report = load_sessions_report(ROOT / SESSIONS_REPORT)
    if not report:
        logger.warning("Sin informe de sesiones: la seccion 3.3 saldra degradada")
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


def _signals() -> dict:
    """
    La plataforma de ingesta de senales (seccion 4.3): catalogo, conexion y PROFUNDIDAD.

    A diferencia de los demas bloques, este NO lee un informe de estudio: el catalogo es
    codigo, la auditoria de entidades y de archivo es local, y la profundidad sale del
    registro de mediciones que escribe la sonda (`data/signals/history_depth.json`). Las
    cifras que importan —cuantas fuentes tienen adaptador y cuantas tienen historia
    MEDIDA— cambian escribiendo codigo y corriendo la sonda, no re-corriendo un estudio.
    """
    from ai_trader.signals.audit import audit_archive, audit_entities
    from ai_trader.signals.capture import connect_adapters
    from ai_trader.signals.catalog import CATALOG, catalog_summary
    from ai_trader.signals.depth import DEPTH_LEDGER, load_ledger
    from ai_trader.signals.normalize import normalization_spec
    from ai_trader.signals.source import connected_keys
    from ai_trader.signals.store import SignalStore

    universe = list(load_config(ROOT / "config" / "default.toml").runner.symbols)
    connect_adapters()
    entities = audit_entities(universe)
    archive = audit_archive(SignalStore(ROOT / "data" / "signals_raw"))

    ledger = load_ledger(ROOT / DEPTH_LEDGER) or {}
    depth = {row["source_key"]: row for row in ledger.get("sources") or []}

    return {
        "summary": catalog_summary(),
        "n_connected": len(connected_keys()),
        "n_measured": sum(1 for r in depth.values() if r.get("first_day")),
        "records": archive.records,
        "normalization": normalization_spec(),
        "depth_measured_at": (ledger.get("generated_at") or "")[:10] or "—",
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
                s.tier,
                s.pit,
                str(len(s.features)),
                "si" if s.key in set(connected_keys()) else "no",
                s.history_from.isoformat() if s.history_from else "—",
                (depth.get(s.key) or {}).get("first_day") or "—",
                f"{(depth.get(s.key) or {}).get('days', 0):,}".replace(",", " "),
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

    facts["calibration"] = _calibration()
    facts["fidelity"] = _fidelity()
    facts["transfer"] = _transfer()
    facts["validation"] = _validation()
    facts["sessions"] = _sessions()
    facts["activity"] = _activity()
    facts["signals"] = _signals()

    facts["mom_params"] = _params(CryptoMomentumStrategy().config)
    facts["mr_params"] = _params(MeanReversionStrategy().config)
    facts["space_mom"] = _space("crypto_momentum")
    facts["space_mr"] = _space("mean_reversion")

    return facts


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
