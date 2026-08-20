"""
Estudio comparativo: que cambia al partir el tiempo de una forma o de otra.

Corre, para cada par (configuracion, muestra) de la libreria sintetica, TRES esquemas
sobre exactamente las mismas barras y la misma ventana:

    single_split   el corte 70/30 historico -> UN headline score OOS
    walk_forward   5 ventanas OOS consecutivas, purgadas y embargadas
    cpcv           C(6,2) = 15 ventanas combinatorias, purgadas y embargadas

y publica la comparacion pareada. Las tres comparten la misma particion del calendario
(walk-forward usa 5+1 grupos y CPCV 6), de modo que el cacheo de tramos del motor evita
volver a correr lo ya corrido.

    .venv\\Scripts\\python.exe -m ai_trader.research.validation_study --workers 7

Salida (data/validation/):
    report_<lib>.json   plan, resultados por (config, muestra) y agregados

RESULTADO DE LA PRIMERA CORRIDA (32 unidades sobre ai_v2), porque contradice la hipotesis
que motivo el estudio y conviene que quien lo re-corra lo sepa: se esperaba que el corte
unico SOBRE-ESTIMARA sistematicamente. No lo hace. Frente a la mediana de las ventanas su
diferencia es -0.04 (IQR -1.40 .. +1.35, rango -3.4 .. +4.7): no esta sesgado al alza,
esta ARBITRARIO. Lo que si sostiene la medicion es (1) +1.35 frente al CVaR de las
ventanas —no por sesgo, sino porque el CVaR de un solo numero es ese numero—, (2) que
mover la ventana mueve el resultado 3.3 veces mas que cambiar de configuracion, y (3) que
la ganadora cambia en 4 de 8 muestras. Un metodo de seleccion no necesita estar sesgado
para ser malo: le basta con ser arbitrario.

Que se mide, y por que cada cosa:

- **Diferencia de nivel** = `single - mediana(ventanas)`, pareado. Deliberadamente NO se
  llama "optimismo": nombrar un estadistico por el resultado que se espera de el es la
  forma mas barata de encontrarlo.
- **Diferencia contra la cola** = `single - CVaR(ventanas)`. Es la comparacion que
  importa, porque el CVaR es lo que rankea el sistema.
- **Dispersion** = desviacion tipica entre ventanas de una misma muestra, y **senal vs
  ruido** = ese rango frente a lo que separa a dos configuraciones. Es lo que el corte
  unico no podia mostrar por construccion.
- **Acuerdo de rangos** (Spearman) entre ordenar configuraciones por el corte unico y
  ordenarlas por la recompensa multiventana. Es la pregunta decisiva: no importa que el
  nivel cambie si la DECISION no cambia; importa si cambia.
- **Cambios de veredicto** del gate de baselines entre un esquema y otro.

Determinismo: la geometria de los folds solo depende de fechas y parametros, el backtest
es determinista dado (config, barras, ventana), y las configuraciones son una lista fija.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.config import StrategySpec, load_config
from ai_trader.scoring.multiwindow import (
    SCHEME_CPCV,
    SCHEME_WALK_FORWARD,
    validate_multiwindow,
)
from ai_trader.research.synthetic_source import (
    DEFAULT_LIBRARY_ID,
    DEFAULT_SYNTHETIC_CONFIG,
)
from ai_trader.scoring.weight_calibration import spearman
from ai_trader.shared.reports import load_report
from ai_trader.research.synthetic.service import study_window
from ai_trader.research.synthetic.store import SyntheticStore

logger = logging.getLogger("validation_study")

OUT_DIR = Path("data") / "validation"
VALIDATION_REPORT = OUT_DIR / "report_ai_v2.json"

# Particion compartida: walk-forward con 5 folds parte en 5+1 = 6 grupos, los mismos que
# CPCV. Asi los dos esquemas reusan tramos ya corridos en vez de duplicarlos.
N_FOLDS = 5
N_GROUPS = 6
N_TEST_GROUPS = 2

# Configuraciones fijas y explicitas (no un hipercubo): el estudio compara ESQUEMAS DE
# VALIDACION, no busca la mejor estrategia. Basta con que sean distinguibles entre si.
STUDY_CONFIGS: tuple[tuple[str, str, dict], ...] = (
    ("mom_default", "crypto_momentum", {}),
    ("mom_fast", "crypto_momentum",
     {"fast_sma_window": 5, "slow_sma_window": 20, "breakout_lookback": 3}),
    ("mr_default", "mean_reversion", {}),
    ("mr_strict", "mean_reversion", {"entry_z": 1.5, "exit_z": 0.2, "lookback": 15}),
    # Las seis tematicas, dos configuraciones cada una y por el mismo motivo que las dos de
    # arriba: aqui no se busca la mejor estrategia, se comparan ESQUEMAS DE VALIDACION, y
    # para eso basta con que las configuraciones sean distinguibles entre si. La variante de
    # cada familia mueve el mando que define su tesis, no un parametro cualquiera.
    ("liq_default", "liquidation_cascade", {}),
    ("liq_deep", "liquidation_cascade", {"entry_stretch_atr": 3.0, "entry_range_atr": 2.0}),
    ("vol_default", "vol_term_structure", {}),
    ("vol_tight", "vol_term_structure", {"max_compression": 0.6, "breakout_lookback": 10}),
    ("cal_default", "event_calendar_drift", {}),
    ("cal_wide", "event_calendar_drift", {"drift_window": 10, "min_drift_pct": 4.0}),
    ("att_default", "attention_ignition", {}),
    ("att_loud", "attention_ignition", {"volume_mult": 4.0, "close_location_min": 0.85}),
    ("flow_default", "flow_persistence", {}),
    ("flow_strict", "flow_persistence", {"min_persistence": 0.70, "pullback_atr": 0.5}),
    ("comp_default", "signal_composite", {}),
    # `min_bars` NO es decorativo aqui: la familia se niega a construirse si no cubre
    # trend_window + cross_lookback + atr_window, y con trend_window=150 el suelo sube a
    # 167. Queda en 170 —por encima del suelo y diez barras por debajo de las 180 de
    # calentamiento— para que ni reviente ni se quede muda en el borde de la ventana.
    ("comp_slow", "signal_composite",
     {"trigger_window": 40, "trend_window": 150, "min_bars": 170}),
)


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Todo lo que define el estudio. Se serializa con los resultados: sin esto las
    cifras no serian ni reproducibles ni auditables."""

    library_id: str
    config_path: str
    config_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    n_paths: int
    n_folds: int
    n_groups: int
    n_test_groups: int
    purge_days: int
    embargo_days: int | None
    starting_equity: float
    start: str
    end: str

    def as_dict(self) -> dict:
        return {
            "library_id": self.library_id,
            "config_path": self.config_path,
            "config_ids": list(self.config_ids),
            "scenario_ids": list(self.scenario_ids),
            "n_paths": self.n_paths,
            "n_folds": self.n_folds,
            "n_groups": self.n_groups,
            "n_test_groups": self.n_test_groups,
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
            "starting_equity": self.starting_equity,
            "start": self.start,
            "end": self.end,
            "n_samples": len(self.scenario_ids) * self.n_paths,
        }


def load_validation_report(path: Path | str = VALIDATION_REPORT) -> dict | None:
    """Lee el informe publicado. Devuelve None si no esta, para que los generadores de
    dashboard y documentacion degraden a prosa sin cifras en vez de romperse."""
    return load_report(path)


# ------------------------------------------------------------------- workers --------

_WORKER: dict = {}


def _init_worker(library_id: str, config_path: str, start: datetime, end: datetime,
                 purge_days: int, starting_equity: float) -> None:
    logging.getLogger("ai_trader").setLevel(logging.ERROR)
    _WORKER.update(
        store=SyntheticStore(),
        library_id=library_id,
        base_config=load_config(config_path),
        start=start,
        end=end,
        purge_days=purge_days,
        starting_equity=starting_equity,
        bars_key=None,
        bars=None,
    )


def _worker_bars(scenario_id: str, path_index: int) -> dict:
    """Cachea las barras de la ultima muestra vista: las tareas llegan agrupadas por
    muestra, asi que una carga de parquet sirve para todas sus configuraciones."""
    key = (scenario_id, path_index)
    if _WORKER["bars_key"] != key:
        _WORKER["bars"] = _WORKER["store"].load_bars(
            _WORKER["library_id"], scenario_id, path_index
        )
        _WORKER["bars_key"] = key
    return _WORKER["bars"]


def _run_task(task: tuple[str, str, int]) -> dict:
    """Evalua UNA (configuracion, muestra) por los tres esquemas, compartiendo el cacheo
    de tramos: walk-forward y CPCV parten el calendario igual, asi que los tramos comunes
    se corren una sola vez."""
    config_id, scenario_id, path_index = task
    spec = _spec_for(config_id)
    bars = _worker_bars(scenario_id, path_index)
    common = dict(
        purge_days=_WORKER["purge_days"],
        starting_equity=_WORKER["starting_equity"],
        block_cache={},
        baseline_cache={},
    )

    wf = validate_multiwindow(
        _WORKER["base_config"], spec, bars, _WORKER["start"], _WORKER["end"],
        scheme=SCHEME_WALK_FORWARD, n_folds=N_FOLDS, **common,
    )
    cpcv = validate_multiwindow(
        _WORKER["base_config"], spec, bars, _WORKER["start"], _WORKER["end"],
        scheme=SCHEME_CPCV, n_groups=N_GROUPS, n_test_groups=N_TEST_GROUPS,
        compare_single_split=False, **common,
    )
    return {
        "config_id": config_id,
        "scenario_id": scenario_id,
        "path_index": path_index,
        # El corte unico se evalua UNA vez por (config, muestra): lo devuelve el esquema
        # walk-forward como referencia y el de CPCV no lo repite.
        "single_score": wf.single_split_score,
        "walk_forward": _scheme_row(wf),
        "cpcv": _scheme_row(cpcv),
    }


def _spec_for(config_id: str) -> StrategySpec:
    for cid, family, params in STUDY_CONFIGS:
        if cid == config_id:
            return StrategySpec(type=family, id=cid, params=dict(params))
    raise KeyError(config_id)


def _scheme_row(result) -> dict:
    s = result.stats
    return {
        "reward": s.reward,
        "mean": s.mean,
        "median": result.median,
        "std": s.std,
        "worst": s.worst,
        "best": s.best,
        "n_folds": s.n,
        "leakage_ok": result.leakage_ok,
        "approved": result.approved,
        "embargo_days": result.embargo_days,
        "scores": [round(x, 4) for x in result.scores],
    }


# ------------------------------------------------------------------ ejecucion -------


def run_study(plan: StudyPlan, workers: int) -> list[dict]:
    tasks = [
        (config_id, scenario_id, path_index)
        for scenario_id in plan.scenario_ids
        for path_index in range(plan.n_paths)
        for config_id in plan.config_ids  # muestra-mayor: el worker reusa las barras
    ]
    total = len(tasks)
    windows = plan.n_folds + (plan.n_groups + plan.n_groups - 1) + 2
    logger.info(
        "Evaluando %d configs x %d escenarios x %d paths = %d unidades "
        "(~%d ventanas cada una) | workers=%d",
        len(plan.config_ids), len(plan.scenario_ids), plan.n_paths, total, windows, workers,
    )

    init_args = (
        plan.library_id, plan.config_path,
        datetime.fromisoformat(plan.start), datetime.fromisoformat(plan.end),
        plan.purge_days, plan.starting_equity,
    )
    started = time.time()
    rows: list[dict] = []
    with mp.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
        for payload in pool.imap_unordered(_run_task, tasks, chunksize=1):
            rows.append(payload)
            elapsed = time.time() - started
            logger.info(
                "  %d/%d (%.0f%%) | %.1f min transcurridos | ETA %.0f min",
                len(rows), total, 100.0 * len(rows) / total, elapsed / 60.0,
                elapsed / len(rows) * (total - len(rows)) / 60.0,
            )
    logger.info("Hecho en %.1f min", (time.time() - started) / 60.0)
    return sorted(rows, key=lambda r: (r["config_id"], r["scenario_id"], r["path_index"]))


def analyze(rows: list[dict], plan: dict) -> dict:
    """Convierte las filas crudas en la comparacion pareada que responde al estudio.

    `plan` es el diccionario del plan (no el dataclass) para que `--analyze-only` pueda
    re-analizar un informe publicado con SU plan, no con el que se reconstruya hoy."""
    by_sample: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        by_sample.setdefault((row["scenario_id"], row["path_index"]), {})[row["config_id"]] = row

    optimism_wf = [
        r["single_score"] - r["walk_forward"]["median"]
        for r in rows if r["single_score"] is not None
    ]
    optimism_cpcv = [
        r["single_score"] - r["cpcv"]["median"]
        for r in rows if r["single_score"] is not None
    ]
    tail_gap = [
        r["single_score"] - r["walk_forward"]["reward"]
        for r in rows if r["single_score"] is not None
    ]

    # Acuerdo de rangos por muestra: ¿el corte unico ordena las configuraciones como las
    # ordena la evidencia multiventana? Es lo unico que decide de verdad.
    rank_wf, rank_cpcv = [], []
    for configs in by_sample.values():
        ids = sorted(configs)
        if len(ids) < 3:
            continue
        single = [configs[i]["single_score"] for i in ids]
        if any(v is None for v in single):
            continue
        rank_wf.append(spearman(single, [configs[i]["walk_forward"]["reward"] for i in ids]))
        rank_cpcv.append(spearman(single, [configs[i]["cpcv"]["reward"] for i in ids]))
    rank_wf = [v for v in rank_wf if v == v]  # descarta NaN (rankings degenerados)
    rank_cpcv = [v for v in rank_cpcv if v == v]

    # Ganadora por muestra segun cada esquema: cuantas veces cambia la eleccion.
    flips_wf = flips_cpcv = comparable = 0
    for configs in by_sample.values():
        ids = sorted(configs)
        if any(configs[i]["single_score"] is None for i in ids):
            continue
        comparable += 1
        best_single = max(ids, key=lambda i: configs[i]["single_score"])
        flips_wf += best_single != max(ids, key=lambda i: configs[i]["walk_forward"]["reward"])
        flips_cpcv += best_single != max(ids, key=lambda i: configs[i]["cpcv"]["reward"])

    # Senal vs ruido: ¿cuanto separa a dos CONFIGURACIONES frente a cuanto separa a dos
    # VENTANAS de la misma configuracion? Si lo segundo domina, elegir con una sola
    # ventana es elegir por el tramo de historia que toco, no por la estrategia.
    config_spread_single, config_spread_wf = [], []
    for configs in by_sample.values():
        singles = [c["single_score"] for c in configs.values() if c["single_score"] is not None]
        if len(singles) > 1:
            config_spread_single.append(max(singles) - min(singles))
        rewards = [c["walk_forward"]["reward"] for c in configs.values()]
        if len(rewards) > 1:
            config_spread_wf.append(max(rewards) - min(rewards))

    window_range = _spread([r["walk_forward"]["best"] - r["walk_forward"]["worst"] for r in rows])
    spread_wf = _spread(config_spread_wf)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "rows": rows,
        "signal_vs_noise": {
            "config_spread_single_split": _spread(config_spread_single),
            "config_spread_walk_forward": spread_wf,
            "window_range_walk_forward": window_range,
            "ratio": (
                round(window_range["median"] / spread_wf["median"], 2)
                if spread_wf["median"] else None
            ),
        },
        "optimism": {
            "walk_forward": _spread(optimism_wf),
            "cpcv": _spread(optimism_cpcv),
            "vs_tail": _spread(tail_gap),
        },
        "dispersion": {
            "walk_forward_std": _spread([r["walk_forward"]["std"] for r in rows]),
            "cpcv_std": _spread([r["cpcv"]["std"] for r in rows]),
            "walk_forward_range": _spread(
                [r["walk_forward"]["best"] - r["walk_forward"]["worst"] for r in rows]
            ),
        },
        "rank_agreement": {
            "walk_forward": _spread(rank_wf),
            "cpcv": _spread(rank_cpcv),
            "n_samples": len(rank_wf),
        },
        "decision_flips": {
            "n_samples": comparable,
            "walk_forward": flips_wf,
            "cpcv": flips_cpcv,
            "walk_forward_pct": round(100.0 * flips_wf / comparable, 1) if comparable else 0.0,
            "cpcv_pct": round(100.0 * flips_cpcv / comparable, 1) if comparable else 0.0,
        },
        "gate": {
            "approved_walk_forward": sum(1 for r in rows if r["walk_forward"]["approved"]),
            "approved_cpcv": sum(1 for r in rows if r["cpcv"]["approved"]),
            "n": len(rows),
        },
        "leakage": {
            "folds_audited": sum(
                r["walk_forward"]["n_folds"] + r["cpcv"]["n_folds"] for r in rows
            ),
            "clean": all(
                r["walk_forward"]["leakage_ok"] and r["cpcv"]["leakage_ok"] for r in rows
            ),
        },
    }


def _spread(values: list[float]) -> dict:
    """Mediana e intercuartilico. Mediana y no media: la distribucion de headline scores
    tiene colas, y una muestra degenerada no deberia mover el resumen."""
    clean = sorted(v for v in values if v is not None and v == v)
    if not clean:
        return {"n": 0, "median": None, "p25": None, "p75": None, "mean": None}
    return {
        "n": len(clean),
        "median": round(statistics.median(clean), 4),
        "p25": round(_quantile(clean, 0.25), 4),
        "p75": round(_quantile(clean, 0.75), 4),
        "mean": round(statistics.fmean(clean), 4),
        "min": round(clean[0], 4),
        "max": round(clean[-1], 4),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--library", default=DEFAULT_LIBRARY_ID)
    parser.add_argument("--config", default=str(DEFAULT_SYNTHETIC_CONFIG))
    parser.add_argument("--scenarios", type=int, default=6, help="Escenarios (default 6).")
    parser.add_argument("--paths", type=int, default=1, help="Caminos por escenario (1).")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=None, help="Ruta del informe (default data/validation).")
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Re-analiza el informe publicado sin volver a correr backtests.",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.out) if args.out else OUT_DIR / f"report_{args.library}.json"

    base_config = load_config(args.config)
    store = SyntheticStore()
    manifest = store.load_manifest(args.library)
    scenario_ids = tuple(s["id"] for s in manifest.scenarios)[: args.scenarios]
    start, end = study_window(manifest, base_config.runner.lookback_days)

    plan = StudyPlan(
        library_id=args.library,
        config_path=args.config,
        config_ids=tuple(c[0] for c in STUDY_CONFIGS),
        scenario_ids=scenario_ids,
        n_paths=min(args.paths, manifest.n_paths),
        n_folds=N_FOLDS,
        n_groups=N_GROUPS,
        n_test_groups=N_TEST_GROUPS,
        purge_days=max(1, base_config.runner.max_holding_days),
        embargo_days=None,  # 1% del rango, resuelto por `resolve_embargo_days`
        starting_equity=DEFAULT_STARTING_EQUITY,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    if args.analyze_only:
        published = load_validation_report(report_path)
        if not published:
            logger.error("No hay informe en %s", report_path)
            return 1
        report = analyze(published["rows"], published["plan"])
    else:
        report = analyze(run_study(plan, args.workers), plan.as_dict())

    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    logger.info("Informe -> %s", report_path)
    _print_summary(report)
    return 0


def _print_summary(report: dict) -> None:
    o = report["optimism"]
    d = report["dispersion"]
    r = report["rank_agreement"]
    f = report["decision_flips"]
    print()
    print("=== VALIDACION MULTIVENTANA vs CORTE UNICO ===")
    print(f"  muestras x configs: {len(report['rows'])}   "
          f"folds auditados: {report['leakage']['folds_audited']}   "
          f"sin fuga: {report['leakage']['clean']}")
    # "Diferencia", no "optimismo": la medicion no encontro sesgo al alza frente a la
    # mediana de las ventanas, solo frente a la cola. Nombrarlo mal seria pre-juzgarlo.
    print(f"  Diferencia del 70/30 vs mediana walk-forward: "
          f"{o['walk_forward']['median']:+.4f} (IQR {o['walk_forward']['p25']:+.3f} .. "
          f"{o['walk_forward']['p75']:+.3f})")
    print(f"  Diferencia del 70/30 vs mediana CPCV:        "
          f"{o['cpcv']['median']:+.4f} (IQR {o['cpcv']['p25']:+.3f} .. {o['cpcv']['p75']:+.3f})")
    print(f"  Diferencia vs la COLA (CVaR walk-forward):   {o['vs_tail']['median']:+.4f} "
          f"(IQR {o['vs_tail']['p25']:+.3f} .. {o['vs_tail']['p75']:+.3f})")
    print(f"  Dispersion entre ventanas (std): wf {d['walk_forward_std']['median']:.4f}   "
          f"cpcv {d['cpcv_std']['median']:.4f}   rango wf {d['walk_forward_range']['median']:.4f}")
    svn = report["signal_vs_noise"]
    print(f"  Senal vs ruido: separa a dos configs {svn['config_spread_walk_forward']['median']:.2f} "
          f"vs a dos ventanas {svn['window_range_walk_forward']['median']:.2f} "
          f"(ruido temporal x{svn['ratio']})")
    print(f"  Acuerdo de rangos con el corte unico: wf {r['walk_forward']['median']:+.3f}   "
          f"cpcv {r['cpcv']['median']:+.3f}  (n={r['n_samples']})")
    print(f"  Cambia la configuracion elegida: wf {f['walk_forward']}/{f['n_samples']} "
          f"({f['walk_forward_pct']}%)   cpcv {f['cpcv']}/{f['n_samples']} ({f['cpcv_pct']}%)")


if __name__ == "__main__":
    raise SystemExit(main())
