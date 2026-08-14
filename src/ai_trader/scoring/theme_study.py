"""
LA CAPA TEMATICA CONTRA SENAL REAL: la misma familia, la misma ventana, la puerta abierta y cerrada.

QUE PREGUNTA CONTESTA, Y POR QUE NO LA CONTESTA EL RANKING
-----------------------------------------------------------
El estudio de transferencia ordena ocho familias sobre 2018-2025, pero ninguna dimension de su
rejilla toca un umbral de senal (eso es deliberado: ver `scoring/search_space.py`). Asi que lo
que rankea de las seis tematicas es su NUCLEO DE PRECIO con el filtro abierto de par en par. Su
puesto ahi es una cota, y de la clase mas enganosa: la de la version que no sabe nada.

Este estudio mide lo otro, que es lo unico atribuible a la senal: **la diferencia pareada entre
la misma configuracion con la capa inerte y con la capa armada**, sobre las mismas ventanas y con
el archivo de senales REAL enchufado. Pareada y no comparada entre familias: el ruido entre
ventanas es enorme comparado con el efecto que se busca, y lo unico que lo cancela es que las dos
piernas vean exactamente las mismas barras.

POR QUE SE PUEDE HACER HOY, Y SOLO PARA CUATRO DE LAS SEIS
-----------------------------------------------------------
No es una limitacion de diseno sino de profundidad, y esta medida: de los cinco temas, `macro`
(3 fuentes con `history_from`), `attention` (2) y `flow` (8) alcanzan el minimo de cobertura en
historico; `liquidation` (1 de 4) y `vol_surface` (1 de 2) no. Las familias de esos dos temas
entran en el informe DECLARADAS como no evaluables, con la fecha en la que dejaran de serlo, en
vez de desaparecer de la tabla.

Y hubo que abrir una costura para llegar aqui: `BacktestEngine` aceptaba `signals=` desde
siempre, pero `scoring/multiwindow.py` no lo propagaba, asi que ningun estudio habia mirado nunca
senal capturada de verdad. El lado real era ciego POR CABLEADO, no por falta de datos.

EL VEREDICTO PUEDE SER "SIN POTENCIA", Y ESO ES UN RESULTADO
-------------------------------------------------------------
Igual que en `backtest/divergence_study.py`: si los dias con cobertura no dan muestra, la
respuesta correcta es decirlo y no publicar una diferencia que el ruido explica. Un intervalo por
bloques que contiene el cero con N=12 no dice "la senal no sirve", dice "no se ha medido".
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.config import DEFAULT_CONFIG_PATH, StrategySpec, load_config
from ai_trader.observation.signal_radar import MIN_SIGNAL_COVERAGE
from ai_trader.observation.signal_themes import THEMES, effective_denominator
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA
from ai_trader.scoring.multiwindow import SCHEME_CPCV, validate_multiwindow
from ai_trader.scoring.signal_study import GATE_VALUE_BY_PARAM, gate_param_for
from ai_trader.scoring.transfer_study import (
    N_GROUPS,
    N_TEST_GROUPS,
    RealWindow,
    build_specs,
    crypto_universe,
    real_windows,
)
from ai_trader.scoring.weight_study import NEW_FAMILIES
from ai_trader.shared.reports import write_report
from ai_trader.signals.catalog import CATALOG
from ai_trader.signals.feed import load_frames

logger = logging.getLogger("theme_study")

OUT_DIR = Path("data") / "themes"
REPORT = OUT_DIR / "report.json"

# Configuraciones por familia. Menos que en la rejilla de ranking a proposito: aqui no se
# ordena nada, se mide UNA diferencia pareada, y cada configuracion cuesta el DOBLE (dos
# piernas). Con cuatro por familia el intervalo por bloques ya se puede leer.
CONFIGS_PER_FAMILY = 4

# Los dos brazos. El nombre va en el informe para que no haya que deducirlo del signo.
ARM_BLIND = "ciega"
ARM_ARMED = "armada"

# Minimo de ventanas con la capa realmente ACTIVA para publicar una diferencia. Por debajo el
# veredicto es `sin_potencia`, igual que en el estudio de divergencia.
MIN_PAIRED_WINDOWS = 12

VERDICT_NO_POWER = "sin_potencia"
VERDICT_HELPS = "la_capa_ayuda"
VERDICT_HURTS = "la_capa_resta"
VERDICT_FLAT = "indistinguible"


def evaluable_themes() -> tuple[frozenset[str], frozenset[str]]:
    """
    Que temas alcanzan cobertura en un backtest historico y cuales no. DERIVADO del catalogo.

    Se calcula y no se escribe a mano para que el dia que una fuente gane profundidad medida,
    este estudio deje de excluir a su familia sin que nadie tenga que acordarse.
    """
    backtestable = {source.key for source in CATALOG if source.backtestable}
    evaluable, blind = set(), set()
    for name, spec in THEMES.items():
        denominator = effective_denominator(len(spec.sources), spec.min_sources)
        covered = len(set(spec.sources) & backtestable) / denominator
        (evaluable if covered >= MIN_SIGNAL_COVERAGE else blind).add(name)
    return frozenset(evaluable), frozenset(blind)


@dataclass(frozen=True, slots=True)
class StudyPlan:
    config_path: str
    families: tuple[str, ...]
    families_skipped: tuple[dict, ...]
    configs_per_family: int
    symbols: tuple[str, ...]
    windows: tuple[dict, ...]
    themes_evaluable: tuple[str, ...]
    themes_blind: tuple[str, ...]
    sources_loaded: tuple[str, ...]
    starting_equity: float
    cvar_alpha: float

    def as_dict(self) -> dict:
        return {
            "config_path": self.config_path,
            "grid": {
                "families": list(self.families),
                "configs_per_family": self.configs_per_family,
                "n_configs": len(self.families) * self.configs_per_family,
            },
            "families_skipped": [dict(f) for f in self.families_skipped],
            "symbols": list(self.symbols),
            "windows": [dict(w) for w in self.windows],
            "themes": {
                "evaluable": list(self.themes_evaluable),
                "blind": list(self.themes_blind),
            },
            "sources_loaded": list(self.sources_loaded),
            "arms": {
                ARM_BLIND: "capa de senal en sus valores inertes",
                ARM_ARMED: "umbral de puerta inyectado desde fuera del espacio de busqueda",
            },
            "min_paired_windows": MIN_PAIRED_WINDOWS,
            "starting_equity": self.starting_equity,
            "cvar_alpha": self.cvar_alpha,
        }


_WORKER: dict = {}


def _init_worker(config_path: str, raw_root: str | None) -> None:
    logging.getLogger("ai_trader").setLevel(logging.ERROR)
    config = load_config(config_path)
    _WORKER.update(
        config=config,
        # Los frames se cargan UNA vez por worker: son el archivo entero y releerlos por
        # unidad multiplicaria por cien el coste de I/O del estudio.
        frames=load_frames(raw_root=raw_root or config.signals.raw_root or None),
    )


def _armed_spec(spec: StrategySpec) -> StrategySpec:
    """La misma configuracion con —y solo con— el umbral de su puerta inyectado.

    Se inyecta desde FUERA del espacio de busqueda, exactamente como hace el barrido de rho:
    los umbrales de senal no son sorteables, asi que la unica forma de encenderlos es esta.
    """
    param = gate_param_for(spec.type)
    return dataclasses.replace(spec, params={**spec.params, param: GATE_VALUE_BY_PARAM[param]})


def _run_unit(task: tuple) -> dict:
    spec_dict, window_dict, arm = task
    spec = StrategySpec(**spec_dict)
    config = _WORKER["config"]
    frames = _WORKER["frames"]
    start = datetime.fromisoformat(window_dict["start"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(window_dict["end"]).replace(tzinfo=timezone.utc)

    from ai_trader.data.market_data import MarketDataService

    service = MarketDataService(config)
    bars = {
        symbol: service.get_daily_bars(symbol, start, end)
        for symbol in config.runner.symbols
    }
    bars = {s: b for s, b in bars.items() if b is not None and not b.empty}

    try:
        validation = validate_multiwindow(
            config, bars, start, end,
            spec=_armed_spec(spec) if arm == ARM_ARMED else spec,
            scheme=SCHEME_CPCV,
            n_groups=N_GROUPS,
            n_test_groups=N_TEST_GROUPS,
            with_baselines=False,
            compare_single_split=False,
            # LA COSTURA NUEVA: el archivo real llega al motor. Sin esto las dos piernas
            # serian identicas y el estudio mediria cero por construccion.
            signals=frames if arm == ARM_ARMED else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unidad fallida %s/%s/%s: %s", spec.id, window_dict["label"], arm, exc)
        return {"config_id": spec.id, "family": spec.type, "window": window_dict["label"],
                "arm": arm, "failed": True, "scores": [], "trades": []}

    return {
        "config_id": spec.id,
        "family": spec.type,
        "window": window_dict["label"],
        "arm": arm,
        "failed": False,
        "scores": [round(f.score, 6) for f in validation.folds],
        "trades": [f.num_trades for f in validation.folds],
    }


def _block_bootstrap(diffs: Sequence[float], *, n_samples: int = 2000, seed: int = 20260814) -> dict:
    """Intervalo por remuestreo de las DIFERENCIAS pareadas, no de los niveles."""
    values = np.asarray([d for d in diffs if np.isfinite(d)], dtype=float)
    if values.size < 3:
        return {"mean": None, "lo": None, "hi": None, "n": int(values.size)}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_samples, values.size), replace=True).mean(axis=1)
    return {
        "mean": round(float(values.mean()), 6),
        "lo": round(float(np.percentile(draws, 2.5)), 6),
        "hi": round(float(np.percentile(draws, 97.5)), 6),
        "n": int(values.size),
    }


def analyze(rows: Sequence[dict], plan: StudyPlan) -> dict:
    """La diferencia pareada por familia, con su intervalo y su veredicto."""
    by_key: dict[tuple, dict] = {}
    for row in rows:
        if row["failed"]:
            continue
        by_key[(row["config_id"], row["window"], row["arm"])] = row

    families: list[dict] = []
    for family in plan.families:
        diffs, blind_scores, armed_scores, moved = [], [], [], 0
        for config_id, window, arm in list(by_key):
            if arm != ARM_BLIND or by_key[(config_id, window, arm)]["family"] != family:
                continue
            armed = by_key.get((config_id, window, ARM_ARMED))
            if armed is None:
                continue
            blind = by_key[(config_id, window, ARM_BLIND)]
            if not blind["scores"] or not armed["scores"]:
                continue
            b = float(np.mean(blind["scores"]))
            a = float(np.mean(armed["scores"]))
            blind_scores.append(b)
            armed_scores.append(a)
            diffs.append(a - b)
            # La capa "se movio" si cambio el numero de operaciones: si nunca lo hace, lo que
            # se esta midiendo es ruido de dos corridas identicas y hay que decirlo.
            if blind["trades"] != armed["trades"]:
                moved += 1

        interval = _block_bootstrap(diffs)
        if moved < MIN_PAIRED_WINDOWS:
            verdict = VERDICT_NO_POWER
        elif interval["lo"] is not None and interval["lo"] > 0:
            verdict = VERDICT_HELPS
        elif interval["hi"] is not None and interval["hi"] < 0:
            verdict = VERDICT_HURTS
        else:
            verdict = VERDICT_FLAT
        families.append(
            {
                "family": family,
                "gate_param": gate_param_for(family),
                "n_pairs": len(diffs),
                "n_windows_where_the_layer_moved": moved,
                "blind_mean": round(float(np.mean(blind_scores)), 4) if blind_scores else None,
                "armed_mean": round(float(np.mean(armed_scores)), 4) if armed_scores else None,
                "paired_difference": interval,
                "verdict": verdict,
            }
        )

    return {
        "plan": plan.as_dict(),
        "families": families,
        "n_failed_units": sum(1 for r in rows if r["failed"]),
        "caveats": [
            "La diferencia es PAREADA: misma configuracion, misma ventana, mismas barras. Lo "
            "unico que cambia entre las dos piernas es el umbral de la puerta y si el archivo "
            "de senales llega al motor.",
            "Los umbrales no se optimizan: se inyectan desde fuera del espacio de busqueda con "
            "el valor declarado de su eje, igual que en el barrido de rho.",
            "Las familias de temas sin profundidad medida NO se evaluan, y se declaran en "
            "`plan.families_skipped` con el motivo en vez de desaparecer de la tabla.",
            f"Por debajo de {MIN_PAIRED_WINDOWS} ventanas en las que la capa cambie algo, el "
            "veredicto es 'sin_potencia': no se publica una diferencia que el ruido explica.",
        ],
    }


def build_plan(args: argparse.Namespace) -> tuple[StudyPlan, list[StrategySpec], list[RealWindow]]:
    config = load_config(args.config)
    evaluable, blind = evaluable_themes()

    families, skipped = [], []
    from ai_trader.strategies.registry import build_strategy

    for family in NEW_FAMILIES:
        theme = getattr(build_strategy(family), "theme", "")
        # El compuesto lee los CINCO temas, asi que le basta con que dos sean legibles; eso
        # se cumple con los tres que tienen profundidad.
        readable = theme == "composite" or theme in evaluable
        if readable:
            families.append(family)
        else:
            skipped.append(
                {
                    "family": family,
                    "theme": theme,
                    "reason": (
                        f"el tema '{theme}' no alcanza {MIN_SIGNAL_COVERAGE:.2f} de cobertura "
                        "en historico: sus fuentes empezaron a existir el dia que arranco la "
                        "captura, asi que su capa no se puede evaluar hacia atras todavia"
                    ),
                }
            )

    symbols, first, last = _real_bounds(config)
    windows = real_windows(first, last, args.window_days)
    frames = load_frames(raw_root=config.signals.raw_root or None)

    plan = StudyPlan(
        config_path=str(args.config),
        families=tuple(families),
        families_skipped=tuple(skipped),
        configs_per_family=args.configs_per_family,
        symbols=tuple(symbols),
        windows=tuple(w.as_dict() for w in windows),
        themes_evaluable=tuple(sorted(evaluable)),
        themes_blind=tuple(sorted(blind)),
        sources_loaded=tuple(sorted(frames)),
        starting_equity=DEFAULT_STARTING_EQUITY,
        cvar_alpha=DEFAULT_CVAR_ALPHA,
    )
    return plan, build_specs(tuple(families), args.configs_per_family), windows


def _real_bounds(config) -> tuple[tuple[str, ...], datetime, datetime]:
    """
    Que simbolos hay en cache y que rango cubren.

    Se resuelve aqui y no con `transfer_study.audit_real_symbols` porque esa funcion audita
    una cobertura MINIMA de historia por simbolo para que los dos lados de la transferencia
    sean comparables, y este estudio no compara dos lados: compara dos brazos sobre las MISMAS
    barras. Pedirle prestado el criterio traeria una restriccion que aqui no significa nada.
    """
    from ai_trader.data.market_data import MarketDataService

    service = MarketDataService(config)
    symbols, firsts, lasts = [], [], []
    for symbol in crypto_universe(config):
        try:
            bars = service.get_daily_bars(symbol, None, None)
        except Exception as exc:  # noqa: BLE001
            logger.info("  %s sin barras en cache (%s)", symbol, exc)
            continue
        if bars is None or bars.empty:
            continue
        symbols.append(symbol)
        firsts.append(bars.index[0].to_pydatetime())
        lasts.append(bars.index[-1].to_pydatetime())
    if not symbols:
        raise ValueError(
            "No hay barras reales en cache. Corre una vez sin --offline (o el estudio de "
            "transferencia, que llena la misma cache) antes de este."
        )
    return tuple(symbols), min(firsts), max(lasts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--configs-per-family", type=int, default=CONFIGS_PER_FAMILY)
    parser.add_argument("--window-days", type=int, default=544)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.ERROR)

    plan, specs, windows = build_plan(args)
    logger.info(
        "%d familias evaluables x %d configuraciones x %d ventanas x 2 brazos = %d unidades",
        len(plan.families), args.configs_per_family, len(windows),
        len(specs) * len(windows) * 2,
    )
    for skip in plan.families_skipped:
        logger.info("  OMITIDA %s: %s", skip["family"], skip["reason"])

    tasks = [
        (dataclasses.asdict(spec), window.as_dict(), arm)
        for spec in specs
        for window in windows
        for arm in (ARM_BLIND, ARM_ARMED)
    ]
    config = load_config(args.config)
    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(args.config), config.signals.raw_root or None),
    ) as pool:
        rows = pool.map(_run_unit, tasks)

    report = analyze(rows, plan)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    path = write_report(report, args.out)
    logger.info("Informe -> %s", path)

    for fam in report["families"]:
        diff = fam["paired_difference"]
        band = (
            f"[{diff['lo']:+.3f}, {diff['hi']:+.3f}]" if diff["lo"] is not None else "sin muestra"
        )
        logger.info(
            "  %-22s %-16s diff %s  n=%d  movio en %d ventanas",
            fam["family"], fam["verdict"], band, fam["n_pairs"],
            fam["n_windows_where_the_layer_moved"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
