"""
Estudio de calibracion de los pesos del headline score sobre una libreria sintetica.

Corre la parte CARA (un backtest real por configuracion y muestra), guarda los
componentes crudos y despues barre la rejilla (lambda, kappa) en memoria. Los
componentes no dependen de los pesos, asi que la rejilla sale gratis y el fichero de
componentes permite re-analizar sin volver a backtestear.

    .venv\\Scripts\\python.exe -m ai_trader.research.weight_study --workers 7
    .venv\\Scripts\\python.exe -m ai_trader.research.weight_study --analyze-only

Salidas (data/calibration/):
    components_<lib>.json  evidencia cruda: metricas train/test por (config, muestra)
    report_<lib>.json      rejilla, auditoria de costes y pesos recomendados

Determinismo: las configuraciones salen de un hipercubo latino con seed fija, la
particion de escenarios de `split_scenarios(seed)`, y el backtest es determinista dado
(config, barras, ventana). `--verify-determinism` lo comprueba re-evaluando muestras ya
calculadas en otro proceso y exigiendo igualdad bit a bit de las metricas.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.backtest.metrics import HeadlineWeights
from ai_trader.config import AppConfig, StrategySpec, load_config
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA
from ai_trader.scoring.families import FAMILIES, STUDY_SEED
from ai_trader.research.synthetic_source import (
    DEFAULT_LIBRARY_ID,
    DEFAULT_SYNTHETIC_CONFIG,
)
from ai_trader.scoring.scenario_split import DEFAULT_VALIDATION_FRACTION, split_scenarios
from ai_trader.scoring.weight_calibration import (
    DEFAULT_KAPPA_GRID,
    DEFAULT_LAMBDA_GRID,
    CalibrationMatrix,
    CalibrationSample,
    candidate_specs,
    evaluate_calibration_sample,
    filter_active_configs,
    sweep_weights,
    turnover_cost_audit,
)
from ai_trader.research.synthetic.service import study_window
from ai_trader.research.synthetic.store import SyntheticStore

logger = logging.getLogger("weight_study")

OUT_DIR = Path("data") / "calibration"
# Umbral de "configuracion que opera de verdad" para el chequeo de robustez: por debajo
# de ~1 operacion cada dos semanas de ventana OOS, el score mide inactividad, no criterio.
ACTIVE_MIN_MEDIAN_TRADES = 20


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Todo lo que define el estudio. Se serializa con los resultados: sin esto, las
    cifras no serian reproducibles ni auditables."""

    library_id: str
    config_path: str
    families: tuple[str, ...]
    configs_per_family: int
    scenario_ids: tuple[str, ...]
    n_paths: int
    split_ratio: float
    starting_equity: float
    validation_fraction: float
    split_seed: int
    study_seed: int
    start: str
    end: str

    def as_dict(self) -> dict:
        return {
            "library_id": self.library_id,
            "config_path": self.config_path,
            "families": list(self.families),
            "configs_per_family": self.configs_per_family,
            "n_scenarios": len(self.scenario_ids),
            "n_paths": self.n_paths,
            "split_ratio": self.split_ratio,
            "starting_equity": self.starting_equity,
            "validation_fraction": self.validation_fraction,
            "split_seed": self.split_seed,
            "study_seed": self.study_seed,
            "window": {"start": self.start, "end": self.end},
        }


# ------------------------------------------------------------------- worker ---------

_WORKER: dict = {}


def _init_worker(
    library_id: str,
    config_path: str,
    specs: list[StrategySpec],
    start: datetime,
    end: datetime,
    split_ratio: float,
    starting_equity: float,
) -> None:
    logging.getLogger("ai_trader").setLevel(logging.ERROR)
    _WORKER.update(
        store=SyntheticStore(),
        library_id=library_id,
        base_config=load_config(config_path),
        specs={spec.id: spec for spec in specs},
        start=start,
        end=end,
        split_ratio=split_ratio,
        starting_equity=starting_equity,
        bars_key=None,
        bars=None,
    )


def _worker_bars(scenario_id: str, path_index: int) -> dict:
    """Cachea las barras de la ultima muestra vista. Las tareas llegan agrupadas por
    muestra (chunksize = nº de configs), asi que una carga sirve para todas sus configs."""
    key = (scenario_id, path_index)
    if _WORKER["bars_key"] != key:
        _WORKER["bars"] = _WORKER["store"].load_bars(_WORKER["library_id"], scenario_id, path_index)
        _WORKER["bars_key"] = key
    return _WORKER["bars"]


def _run_task(task: tuple[str, str, int]) -> dict:
    config_id, scenario_id, path_index = task
    sample = evaluate_calibration_sample(
        _WORKER["base_config"],
        _WORKER["specs"][config_id],
        _worker_bars(scenario_id, path_index),
        _WORKER["start"],
        _WORKER["end"],
        config_id=config_id,
        scenario_id=scenario_id,
        path_index=path_index,
        split_ratio=_WORKER["split_ratio"],
        starting_equity=_WORKER["starting_equity"],
    )
    return sample.as_dict()


# ------------------------------------------------------------------- ejecucion -------


def _build_specs(families: tuple[str, ...], per_family: int) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    for i, family in enumerate(families):
        # Seed distinta por familia para que los hipercubos no queden alineados entre si.
        specs.extend(candidate_specs(family, per_family, seed=STUDY_SEED + i))
    return specs


def run_backtests(plan: StudyPlan, specs: list[StrategySpec], workers: int) -> list[CalibrationSample]:
    """Corre la matriz (configuracion x muestra) en paralelo. Es la unica parte cara."""
    tasks = [
        (spec.id, scenario_id, path_index)
        for scenario_id in plan.scenario_ids
        for path_index in range(plan.n_paths)
        for spec in specs  # muestra-mayor: cada worker reutiliza las barras cargadas
    ]
    total = len(tasks)
    logger.info(
        "Backtests: %d configs x %d escenarios x %d paths = %d | workers=%d",
        len(specs), len(plan.scenario_ids), plan.n_paths, total, workers,
    )

    init_args = (
        plan.library_id, plan.config_path, specs,
        datetime.fromisoformat(plan.start), datetime.fromisoformat(plan.end),
        plan.split_ratio, plan.starting_equity,
    )
    started = time.time()
    done: list[CalibrationSample] = []
    # Chunks de media muestra: bastante grandes para reutilizar las barras cargadas,
    # bastante pequenos para que la cola no deje workers ociosos al final.
    chunksize = max(1, len(specs) // 2)
    with mp.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
        for payload in pool.imap_unordered(_run_task, tasks, chunksize=chunksize):
            done.append(CalibrationSample.from_dict(payload))
            if len(done) % 25 == 0 or len(done) == total:
                elapsed = time.time() - started
                rate = elapsed / len(done)
                logger.info(
                    "  %d/%d (%.0f%%) | %.1fs/backtest | ETA %.0f min",
                    len(done), total, 100.0 * len(done) / total,
                    rate * workers, rate * (total - len(done)) / 60.0,
                )
    logger.info("Backtests done in %.1f min", (time.time() - started) / 60.0)
    return done


def verify_determinism(
    plan: StudyPlan, specs: list[StrategySpec], samples: list[CalibrationSample], n: int
) -> dict:
    """Re-evalua `n` muestras ya calculadas en procesos nuevos y exige metricas
    identicas. Determinismo afirmado sin comprobar es determinismo supuesto."""
    picks = sorted(samples, key=lambda s: (s.config_id, s.scenario_id, s.path_index))
    step = max(1, len(picks) // n)
    chosen = picks[::step][:n]
    tasks = [(s.config_id, s.scenario_id, s.path_index) for s in chosen]

    init_args = (
        plan.library_id, plan.config_path, specs,
        datetime.fromisoformat(plan.start), datetime.fromisoformat(plan.end),
        plan.split_ratio, plan.starting_equity,
    )
    with mp.Pool(min(len(tasks), 4), initializer=_init_worker, initargs=init_args) as pool:
        replays = [CalibrationSample.from_dict(p) for p in pool.map(_run_task, tasks)]

    mismatches = [
        r.config_id
        for original, r in zip(chosen, replays)
        if original.as_dict() != r.as_dict()
    ]
    logger.info(
        "Determinismo: %d/%d muestras reproducidas bit a bit%s",
        len(replays) - len(mismatches), len(replays),
        "" if not mismatches else f" | DISCREPANCIAS: {mismatches}",
    )
    return {"checked": len(replays), "mismatches": mismatches}


# ------------------------------------------------------------------- analisis --------


def _sorted_by_stability(grid: list) -> list:
    """Ordena la rejilla por el criterio de decision: primero estabilidad del ranking
    (rank_ic_mean), y a igualdad practica, menor gap normalizado."""
    return sorted(grid, key=lambda p: (-p.rank_ic_mean, p.selection_gap_norm))


def analyze(
    samples: list[CalibrationSample],
    plan: StudyPlan,
    base_config: AppConfig,
    *,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    kappas: tuple[float, ...] = DEFAULT_KAPPA_GRID,
    alpha: float = DEFAULT_CVAR_ALPHA,
    min_median_trades: int = ACTIVE_MIN_MEDIAN_TRADES,
) -> dict:
    matrix = CalibrationMatrix(samples)
    split = split_scenarios(
        list(plan.scenario_ids),
        validation_fraction=plan.validation_fraction,
        seed=plan.split_seed,
    )
    grid = sweep_weights(matrix, split, lambdas=lambdas, kappas=kappas, alpha=alpha)
    audit = turnover_cost_audit(
        matrix.all_samples(),
        fee_rate=base_config.execution.fee_rate,
        slippage_bps=base_config.execution.slippage_bps,
    )
    active_audit = turnover_cost_audit(
        matrix.all_samples(),
        fee_rate=base_config.execution.fee_rate,
        slippage_bps=base_config.execution.slippage_bps,
        min_trades=min_median_trades,
    )

    # Robustez: la misma rejilla sobre las configuraciones que operan de verdad. Si la
    # recomendacion cambia al quitar las casi inertes, no hay recomendacion.
    active = CalibrationMatrix(
        filter_active_configs(matrix.all_samples(), min_median_trades=min_median_trades)
    )
    active_grid = (
        sweep_weights(active, split, lambdas=lambdas, kappas=kappas, alpha=alpha)
        if active.n_configs >= 3
        else []
    )
    logger.info(
        "Subconjunto activo (mediana de trades OOS >= %d): %d/%d configuraciones",
        min_median_trades, active.n_configs, matrix.n_configs,
    )

    ranked = _sorted_by_stability(grid)
    best = ranked[0]
    logger.info(
        "Rejilla: %d puntos | mejor por estabilidad: lambda=%.2f kappa=%.2f "
        "(rank_ic=%.3f+-%.3f, ganancia pareada vs no penalizar=%.4f+-%.4f, gap_norm=%.2f)",
        len(grid), best.lambda_turnover, best.kappa_maxdd,
        best.rank_ic_mean, best.rank_ic_se,
        best.rank_ic_gain, best.rank_ic_gain_se, best.selection_gap_norm,
    )
    logger.info(
        "Costes: cost_rate=%.4f medido=%.4f | turnover mediano=%.3f | vol mediana=%.2f "
        "=> lambda implicito mediano=%.2f (IQR %.2f-%.2f)",
        audit.cost_rate, audit.measured_fee_rate, audit.median_turnover,
        audit.median_volatility, audit.implied_lambda_median,
        audit.implied_lambda_p25, audit.implied_lambda_p75,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan.as_dict(),
        "split": {
            "train": list(split.train),
            "validation": list(split.validation),
            "seed": split.seed,
        },
        "configs": {
            "kept": matrix.config_ids,
            "dropped": matrix.dropped,
            "n_samples": matrix.n_samples,
        },
        "grid": {
            "lambdas": list(lambdas),
            "kappas": list(kappas),
            "alpha": alpha,
            "points": [p.as_dict() for p in grid],
        },
        "active_subset": {
            "min_median_trades": min_median_trades,
            "configs": active.config_ids,
            "points": [p.as_dict() for p in active_grid],
            "ranked_by_stability": [p.as_dict() for p in _sorted_by_stability(active_grid)[:10]],
        },
        "ranked_by_stability": [p.as_dict() for p in ranked[:10]],
        "cost_audit": audit.as_dict(),
        "cost_audit_active": active_audit.as_dict(),
        "baseline_weights": HeadlineWeights().as_dict(),
    }


def _print_grid(report: dict) -> None:
    points = report["grid"]["points"]
    lambdas = report["grid"]["lambdas"]
    kappas = report["grid"]["kappas"]
    lookup = {(p["lambda_turnover"], p["kappa_maxdd"]): p for p in points}

    for field, title in (
        ("rank_ic_mean", "rank IC medio (estabilidad IS->OOS; mas alto mejor)"),
        ("rank_ic_gain", "ganancia PAREADA de rank IC vs no penalizar (mas alto mejor)"),
        ("rank_ic_gain_se", "error estandar de esa ganancia pareada"),
        ("selection_gap_norm", "gap train-validation normalizado (mas bajo mejor)"),
        ("top1_oos_pct", "percentil OOS del ganador in-sample (mas alto mejor)"),
    ):
        print(f"\n{title}")
        print("  lambda\\kappa " + "".join(f"{k:>9.2f}" for k in kappas))
        for lam in lambdas:
            row = "".join(f"{lookup[(lam, k)][field]:>9.3f}" for k in kappas)
            print(f"  {lam:>11.2f} {row}")

    audit = report["cost_audit"]
    print(
        f"\nCostes ya pagados por la curva: cost_rate={audit['cost_rate']:.4f} "
        f"(medido {audit['measured_fee_rate']:.4f} de fees sobre notional) | "
        f"lambda implicito mediano={audit['implied_lambda_median']:.2f} "
        f"(IQR {audit['implied_lambda_p25']:.2f}-{audit['implied_lambda_p75']:.2f})"
    )


# ----------------------------------------------------------------------- main --------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=DEFAULT_LIBRARY_ID)
    parser.add_argument("--config", default=str(DEFAULT_SYNTHETIC_CONFIG))
    parser.add_argument("--configs-per-family", type=int, default=9)
    parser.add_argument("--paths", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=0, help="0 = todos (pilotos aparte)")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--verify-determinism", type=int, default=0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.ERROR)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    components_path = out_dir / f"components_{args.library}.json"
    report_path = out_dir / f"report_{args.library}.json"

    base_config = load_config(args.config)
    store = SyntheticStore()
    manifest = store.load_manifest(args.library)
    scenario_ids = tuple(s["id"] for s in manifest.scenarios)
    if args.scenarios:
        scenario_ids = scenario_ids[: args.scenarios]

    start, end = study_window(manifest, base_config.runner.lookback_days)
    plan = StudyPlan(
        library_id=args.library,
        config_path=args.config,
        families=FAMILIES,
        configs_per_family=args.configs_per_family,
        scenario_ids=scenario_ids,
        n_paths=min(args.paths, manifest.n_paths),
        split_ratio=0.7,
        starting_equity=DEFAULT_STARTING_EQUITY,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        split_seed=0,
        study_seed=STUDY_SEED,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    specs = _build_specs(plan.families, plan.configs_per_family)

    if args.analyze_only:
        payload = json.loads(components_path.read_text(encoding="utf-8"))
        samples = [CalibrationSample.from_dict(d) for d in payload["samples"]]
        logger.info("Reusando %d backtests de %s", len(samples), components_path)
    else:
        samples = run_backtests(plan, specs, args.workers)
        components_path.write_text(
            json.dumps(
                {"plan": plan.as_dict(), "samples": [s.as_dict() for s in samples]},
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info("Componentes crudos -> %s", components_path)

    report = analyze(samples, plan, base_config)
    if args.verify_determinism:
        report["determinism"] = verify_determinism(plan, specs, samples, args.verify_determinism)

    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    logger.info("Informe -> %s", report_path)
    _print_grid(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
