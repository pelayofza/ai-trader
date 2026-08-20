"""
Transferencia de ranking: ¿ordena el mundo SINTETICO las estrategias como el REAL?

La fidelidad (`synthetic.fidelity_study`) mide hechos estilizados —colas, agrupamiento de
volatilidad, correlacion cruzada— y ya se sabe que ai_v3 los cumple. Pero eso no es lo
que el producto le pide al generador. Lo que le pide es UNA cosa: que si una
configuracion es mejor que otra en el mundo sintetico, tienda a serlo tambien en el
mercado. Un generador puede tener colas perfectas y ordenar al reves; ninguna metrica de
fidelidad lo detectaria.

Este estudio mide exactamente eso: la MISMA rejilla de 16 configuraciones se puntua dos
veces —contra el historico real de Binance y contra la libreria sintetica— y se compara
el orden de los dos rankings.

    .venv\\Scripts\\python.exe -m ai_trader.scoring.transfer_study --offline --workers 7
    .venv\\Scripts\\python.exe -m ai_trader.scoring.transfer_study --analyze-only

Salida: data/transfer/report_<lib>.json

Cinco decisiones que son las que hacen que la cifra signifique algo:

1. LA REJILLA ES LA DEL ESTUDIO DE PESOS. `candidate_specs` con la misma semilla
   (`weight_study.STUDY_SEED`) y las mismas dos familias, 8 configuraciones cada una.
   Reusarla no ahorra computo aqui, pero hace comparables los dos estudios: cuando la
   calibracion dice "los pesos no cambian la eleccion" y este dice "el sintetico ordena
   asi", ambas frases hablan del mismo conjunto de 16 objetos.

2. SOLO CAMBIA EL MUNDO. Los dos lados corren con el MISMO config base, el MISMO universo
   (los simbolos cripto presentes a la vez en el mercado y en la libreria), el mismo
   esquema CPCV, la misma purga y el mismo embargo. Si algo mas cambiara entre los dos
   lados, la correlacion de rangos mediria esa otra cosa.

3. MISMA LONGITUD DE VENTANA. Un camino sintetico dura 544 dias negociables; el historico
   real dura ocho anos. Puntuar el real en ventanas ocho veces mas largas compararia
   estimadores con precisiones distintas, no mundos. Asi que el historico se trocea en
   sub-ventanas DISJUNTAS de la misma longitud que una muestra sintetica, y cada una pasa
   por el mismo CPCV. Es la misma decision que ya tomo el estudio de fidelidad.

4. UNA SOLA COLA, NO DOS. Cada configuracion produce muchos scores (sub-ventana x fold en
   el real, muestra x fold en el sintetico). Todos se ponen en COMUN en una sola
   distribucion y se toma su CVaR@25%. Tomar el CVaR de cada unidad y luego el CVaR de
   esos CVaR compondria dos conservadurismos y dispararia la varianza del estimador.

5. EL INTERVALO ES POR BLOQUES. El historico real es UN camino: no hay ensemble y la
   unica dispersion del lado real son sus folds, que ademas comparten calendario. Un
   bootstrap iid sobre scores fingiria independencia que no existe. Se remuestrean
   BLOQUES enteros (una sub-ventana real, una muestra sintetica) con todos sus folds
   dentro.

6. EL RANKING SE PUBLICA CON Y SIN SUELO DE ACTIVIDAD. Este estudio fue el que destapo que
   el ranking real premiaba no operar (Spearman(recompensa, operaciones) = -0.84), asi que
   ahora cada configuracion lleva su actividad al lado de su recompensa —operaciones por
   ventana OOS y fraccion de ventanas vacias— y el informe saca las dos listas: la completa
   y la de las configuraciones RANKEABLES (`scoring.activity`). Si el suelo cambia al
   ganador, el bloque `eligibility` lo dice con esas palabras. El suelo no penaliza a
   nadie: no resta puntos, decide quien compite.

REGLA DE DECISION, declarada antes de mirar el resultado:
  rho >= +0.30  el generador es un PRE-CRIBADO legitimo, y el flujo definitivo es
                "real como sustrato primario del ranking, sintetico como capa de estres
                y veto".
  rho <= -0.30  el generador ORDENA AL REVES: usarlo para elegir seria peor que no usarlo.
  en medio      NO HAY TRANSFERENCIA: no se disenan estrategias contra el sintetico, y
                eso tiene que decirlo la metodologia.

Determinismo: la rejilla sale de un hipercubo latino con semilla fija, la geometria de las
ventanas solo depende de fechas, el backtest es determinista dado (config, barras,
ventana), y los remuestreos usan un generador con semilla propia. `--verify-determinism`
re-evalua unidades ya calculadas en procesos nuevos y exige scores identicos.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.backtest.metrics import periods_per_year_for_symbols
from ai_trader.backtest.validation import SCHEME_CPCV, resolve_embargo_days
from ai_trader.config import StrategySpec, load_config
from ai_trader.scoring.activity import (
    DEFAULT_ACTIVITY_FLOOR,
    MIN_MEDIAN_TRADES_PER_WINDOW,
    ActivityStats,
    measure_activity,
)
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, aggregate_reward
from ai_trader.scoring.baselines import BASELINE_LABELS, BaselineGate, gate
from ai_trader.scoring.families import (
    CONFIGS_PER_FAMILY,
    FAMILIES,
    STUDY_SEED,
    build_specs,
)
from ai_trader.scoring.multiwindow import resolve_purge_days, validate_multiwindow
from ai_trader.scoring.real_substrate import (
    N_GROUPS,
    N_TEST_GROUPS,
    RealWindow,
    audit_real_symbols,
    crypto_universe,
    real_windows,
)
from ai_trader.scoring.weight_calibration import spearman
from ai_trader.shared.reports import guard_published_grid, load_report
from ai_trader.shared.instruments import AssetClass, detect_asset_class
from ai_trader.data.real_history import (
    DEFAULT_EXCHANGE,
    DEFAULT_REAL_END,
    DEFAULT_REAL_START,
    build_service,
    fetch_real_bars,
)
from ai_trader.synthetic.service import study_window
from ai_trader.synthetic.store import SyntheticStore

logger = logging.getLogger("transfer_study")

OUT_DIR = Path("data") / "transfer"

# Libreria sintetica por defecto: la CORREGIDA (la que acepta los umbrales de fidelidad).
# Si no existiese en disco se cae a la anterior y el informe lo declara: medir la
# transferencia del generador viejo tambien es un resultado, siempre que se sepa cual es.
DEFAULT_LIBRARY_ID = "ai_v3"
FALLBACK_LIBRARY_ID = "ai_v2"
# El config del universo OPERABLE (cripto puro): fija anualizacion 365, no 252, y ademas
# es el unico universo con contraparte real que se pueda descargar.
DEFAULT_CONFIG = Path("config") / "default.toml"

# Regla de decision (ver docstring). Se declara aqui, en el codigo, y no en la prosa del
# informe: un umbral que se elige despues de ver el resultado no es un umbral.
RHO_ACCEPT = 0.30
TOP_K = 4  # cuantas "mejores del sintetico" se persiguen en el ranking real

# Suelo de actividad. Este estudio fue el que destapo el problema (Spearman(recompensa,
# operaciones) = -0.84 en el lado real), pero el umbral ya NO vive aqui: es un requisito de
# elegibilidad de todo el sistema y su casa es `scoring.activity`, donde estan los dos
# numeros, la evidencia con la que se eligieron y el estudio que los mide. Se conserva el
# nombre antiguo porque es el que cita la documentacion publicada.
ACTIVE_MIN_TRADES_PER_FOLD = MIN_MEDIAN_TRADES_PER_WINDOW

BOOTSTRAP_SAMPLES = 2000
PERMUTATION_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260811
CI_PCT = 95.0

VERDICT_TRANSFERS = "pre_cribado_legitimo"
VERDICT_NONE = "sin_transferencia"
VERDICT_INVERTED = "ordenacion_invertida"

FLOW_TRANSFERS = (
    "real como sustrato primario del ranking, sintetico como capa de estres y veto"
)
FLOW_NONE = (
    "no se disenan estrategias contra el sintetico: el ranking lo fija el historico real y "
    "el sintetico se queda como banco de estres, nunca como criterio de seleccion"
)


def transfer_report_path(library_id: str, out_dir: Path | str = OUT_DIR) -> Path:
    return Path(out_dir) / f"report_{library_id}.json"


def load_transfer_report(path: Path | str) -> dict | None:
    """Lee el informe publicado. Devuelve None si no esta, para que los generadores de
    dashboard y documentacion degraden a prosa sin cifras en vez de romperse."""
    return load_report(path)


# ------------------------------------------------------------------- el plan ---------


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Todo lo que define el estudio. Se serializa con los resultados: sin esto las cifras
    no son ni reproducibles ni auditables."""

    library_id: str
    library_is_fallback: bool
    config_path: str
    exchange: str
    offline: bool
    families: tuple[str, ...]
    configs_per_family: int
    study_seed: int
    symbols: tuple[str, ...]
    # Inicio de la DESCARGA, anterior al de la primera sub-ventana: la estrategia necesita
    # `lookback_days` de calentamiento antes de su primer dia negociable, igual que se lo
    # deja `sample_window` al lado sintetico. Sin esto la sub-ventana mas antigua correria
    # ciega y no seria comparable con las demas ni con el sintetico.
    real_source_start: str
    real_start: str
    real_end: str
    real_windows: tuple[dict, ...]
    real_head_discarded_days: int
    real_min_history_days: int
    real_symbols_omitted: tuple[dict, ...]
    library_symbols_omitted: tuple[str, ...]
    synthetic_start: str
    synthetic_end: str
    scenario_ids: tuple[str, ...]
    n_paths: int
    window_days: int
    scheme: str
    n_groups: int
    n_test_groups: int
    n_folds: int
    purge_days: int
    embargo_days: int
    periods_per_year: int
    cvar_alpha: float
    starting_equity: float

    def as_dict(self) -> dict:
        return {
            "library_id": self.library_id,
            "library_is_fallback": self.library_is_fallback,
            "config_path": self.config_path,
            "exchange": self.exchange,
            "offline": self.offline,
            "grid": {
                "families": list(self.families),
                "configs_per_family": self.configs_per_family,
                "n_configs": len(self.families) * self.configs_per_family,
                "study_seed": self.study_seed,
                "sampler": "hipercubo latino sobre el espacio de busqueda real",
            },
            "symbols": list(self.symbols),
            "real": {
                "window": {"start": self.real_start, "end": self.real_end},
                "source_start": self.real_source_start,
                "sub_windows": list(self.real_windows),
                "head_discarded_days": self.real_head_discarded_days,
                "min_history_days": self.real_min_history_days,
                "symbols_omitted": list(self.real_symbols_omitted),
            },
            "synthetic": {
                "window": {"start": self.synthetic_start, "end": self.synthetic_end},
                "scenario_ids": list(self.scenario_ids),
                "n_paths": self.n_paths,
                "n_samples": len(self.scenario_ids) * self.n_paths,
                "library_symbols_omitted": list(self.library_symbols_omitted),
            },
            "validation": {
                "scheme": self.scheme,
                "window_days": self.window_days,
                "n_groups": self.n_groups,
                "n_test_groups": self.n_test_groups,
                "n_folds": self.n_folds,
                "purge_days": self.purge_days,
                "embargo_days": self.embargo_days,
                "periods_per_year": self.periods_per_year,
                "cvar_alpha": self.cvar_alpha,
                "starting_equity": self.starting_equity,
            },
        }


SIDE_REAL = "real"
SIDE_SYNTHETIC = "synthetic"


# ------------------------------------------------------------------- workers ---------

_WORKER: dict = {}


def _init_worker(payload: dict) -> None:
    """
    Estado por proceso. El config base se reconstruye con el UNIVERSO COMUN: es lo que hace
    que los dos lados corran el mismo backtest y que `periods_per_year` sea el mismo.
    """
    logging.getLogger("ai_trader").setLevel(logging.ERROR)
    base = load_config(payload["config_path"])
    _WORKER.update(
        payload=payload,
        config=dataclasses.replace(
            base, runner=dataclasses.replace(base.runner, symbols=list(payload["symbols"]))
        ),
        symbols=list(payload["symbols"]),
        library_id=payload["library_id"],
        purge_days=payload["purge_days"],
        starting_equity=payload["starting_equity"],
        cvar_alpha=payload["cvar_alpha"],
        specs={spec.id: spec for spec in payload["specs"]},
        real_windows={w.label: w for w in payload["real_windows"]},
        synth_start=datetime.fromisoformat(payload["synthetic_start"]),
        synth_end=datetime.fromisoformat(payload["synthetic_end"]),
        store=SyntheticStore(),
        real_bars=None,
        synth_key=None,
        synth_bars=None,
    )


def _real_bars() -> dict[str, pd.DataFrame]:
    """Barras reales del worker, cargadas una sola vez y compartidas por todas las
    sub-ventanas: el motor recorta por fechas y el calentamiento de cada sub-ventana sale
    de su propia historia previa (por eso la descarga empieza antes de la primera)."""
    if _WORKER["real_bars"] is None:
        payload = _WORKER["payload"]
        _WORKER["real_bars"] = fetch_real_bars(
            payload["symbols"],
            datetime.fromisoformat(payload["real_start"]),
            datetime.fromisoformat(payload["real_end"]),
            build_service(payload["exchange"], offline=payload["offline"]),
        )
    return _WORKER["real_bars"]


def _synthetic_bars(scenario_id: str, path_index: int) -> dict[str, pd.DataFrame]:
    """Barras de la ultima muestra vista, cacheadas. Las tareas llegan agrupadas por
    unidad, asi que una lectura de parquet sirve para las 16 configuraciones."""
    key = (scenario_id, path_index)
    if _WORKER["synth_key"] != key:
        bars = _WORKER["store"].load_bars(_WORKER["library_id"], scenario_id, path_index)
        _WORKER["synth_bars"] = {s: df for s, df in bars.items() if s in _WORKER["symbols"]}
        _WORKER["synth_key"] = key
    return _WORKER["synth_bars"]


def _run_task(task: tuple[str, str, str, int]) -> dict:
    """
    Evalua UNA (configuracion, lado, unidad) por CPCV y devuelve sus scores fold a fold.

    Lo caro esta aqui y solo aqui. Un fallo de una unidad no tumba el estudio: se marca
    y la agregacion descarta la configuracion entera (rankear una configuracion sobre un
    subconjunto de unidades distinto al de las demas no es rankear).
    """
    config_id, side, unit_id, path_index = task
    spec = _WORKER["specs"][config_id]

    if side == SIDE_REAL:
        window = _WORKER["real_windows"][unit_id]
        bars = _real_bars()
        start, end = window.start, window.end
    else:
        bars = _synthetic_bars(unit_id, path_index)
        start, end = _WORKER["synth_start"], _WORKER["synth_end"]

    row = {
        "config_id": config_id,
        "side": side,
        "unit_id": unit_id,
        "path_index": path_index,
        "failed": False,
    }
    try:
        result = validate_multiwindow(
            _WORKER["config"], spec, bars, start, end,
            scheme=SCHEME_CPCV,
            n_groups=N_GROUPS,
            n_test_groups=N_TEST_GROUPS,
            purge_days=_WORKER["purge_days"],
            starting_equity=_WORKER["starting_equity"],
            cvar_alpha=_WORKER["cvar_alpha"],
            with_baselines=True,
            # El corte unico 70/30 no interviene en este estudio y cuesta un backtest
            # entero por unidad: se apaga a proposito, no por descuido.
            compare_single_split=False,
            block_cache={},
            baseline_cache={},
        )
    except Exception as exc:  # noqa: BLE001 - una unidad mala no debe tumbar el estudio
        logger.warning("Unidad fallida (%s | %s | %s): %s", config_id, side, unit_id, exc)
        return {**row, "failed": True, "error": str(exc)}

    return {
        **row,
        "scores": [round(s, 6) for s in result.scores],
        "baseline_scores": {
            name: [round(v, 6) for v in values]
            for name, values in result.baseline_scores.items()
        },
        "approved": result.approved,
        "leakage_ok": result.leakage_ok,
        "embargo_days": result.embargo_days,
        "coverage": result.coverage,
        "num_trades": sum(f.num_trades for f in result.folds),
        # Operaciones FOLD A FOLD, en el mismo orden que `scores`. El total no basta: la
        # fraccion de ventanas vacias -que es lo que delata un reward de 0 aritmetico- solo
        # se puede reconstruir con el detalle. Ver `scoring.activity`.
        "trades_by_fold": [f.num_trades for f in result.folds],
    }


# ------------------------------------------------------------------ ejecucion --------


def build_tasks(
    plan: StudyPlan, config_ids: Sequence[str]
) -> list[tuple[str, str, str, int]]:
    """Tareas ordenadas por UNIDAD: el worker reutiliza las barras cargadas para las 16
    configuraciones de la misma muestra."""
    tasks: list[tuple[str, str, str, int]] = []
    for window in plan.real_windows:
        tasks.extend((cid, SIDE_REAL, window["label"], 0) for cid in config_ids)
    for scenario_id in plan.scenario_ids:
        for path_index in range(plan.n_paths):
            tasks.extend(
                (cid, SIDE_SYNTHETIC, scenario_id, path_index) for cid in config_ids
            )
    return tasks


def run_units(
    plan: StudyPlan,
    specs: Sequence[StrategySpec],
    windows: Sequence[RealWindow],
    workers: int,
) -> list[dict]:
    payload = _worker_payload(plan, specs, windows)
    tasks = build_tasks(plan, [s.id for s in specs])
    total = len(tasks)
    logger.info(
        "Evaluando %d configs x (%d sub-ventanas reales + %d muestras sinteticas) = %d "
        "unidades de %d folds | workers=%d",
        len(specs), len(windows), len(plan.scenario_ids) * plan.n_paths, total,
        plan.n_folds, workers,
    )

    started = time.time()
    rows: list[dict] = []
    chunksize = max(1, len(specs) // 2)
    with mp.Pool(workers, initializer=_init_worker, initargs=(payload,)) as pool:
        for row in pool.imap_unordered(_run_task, tasks, chunksize=chunksize):
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == total:
                elapsed = time.time() - started
                logger.info(
                    "  %d/%d (%.0f%%) | %.1f min | ETA %.0f min",
                    len(rows), total, 100.0 * len(rows) / total, elapsed / 60.0,
                    elapsed / len(rows) * (total - len(rows)) / 60.0,
                )
    logger.info("Unidades corridas en %.1f min", (time.time() - started) / 60.0)
    return sorted(rows, key=lambda r: (r["side"], r["unit_id"], r["path_index"], r["config_id"]))


def _worker_payload(
    plan: StudyPlan, specs: Sequence[StrategySpec], windows: Sequence[RealWindow]
) -> dict:
    return {
        "config_path": plan.config_path,
        "symbols": list(plan.symbols),
        "library_id": plan.library_id,
        "exchange": plan.exchange,
        "offline": plan.offline,
        "purge_days": plan.purge_days,
        "starting_equity": plan.starting_equity,
        "cvar_alpha": plan.cvar_alpha,
        "specs": list(specs),
        "real_windows": list(windows),
        "real_start": plan.real_source_start,
        "real_end": plan.real_end,
        "synthetic_start": plan.synthetic_start,
        "synthetic_end": plan.synthetic_end,
    }


def verify_determinism(
    plan: StudyPlan,
    specs: Sequence[StrategySpec],
    windows: Sequence[RealWindow],
    rows: Sequence[dict],
    n: int,
) -> dict:
    """Re-evalua `n` unidades ya calculadas en procesos NUEVOS y exige scores identicos.
    Determinismo afirmado sin comprobar es determinismo supuesto."""
    usable = [r for r in rows if not r["failed"]]
    if not usable or n <= 0:
        return {"checked": 0, "mismatches": []}
    step = max(1, len(usable) // n)
    chosen = usable[::step][:n]
    tasks = [(r["config_id"], r["side"], r["unit_id"], r["path_index"]) for r in chosen]

    payload = _worker_payload(plan, specs, windows)
    with mp.Pool(
        min(len(tasks), 4), initializer=_init_worker, initargs=(payload,)
    ) as pool:
        replays = pool.map(_run_task, tasks)

    mismatches = [
        f"{r['config_id']}/{r['side']}/{r['unit_id']}"
        for r, replay in zip(chosen, replays)
        if r.get("scores") != replay.get("scores")
    ]
    logger.info(
        "Determinismo: %d/%d unidades reproducidas score a score%s",
        len(replays) - len(mismatches), len(replays),
        "" if not mismatches else f" | DISCREPANCIAS: {mismatches}",
    )
    return {"checked": len(replays), "mismatches": mismatches}


# ------------------------------------------------------------------- agregacion ------


def _cvar(scores: Sequence[float], alpha: float) -> float:
    """CVaR@alpha: media del peor alpha-cuantil. Replica exactamente la aritmetica de
    `aggregate_reward` (que es la recompensa del sistema) pero sin construir el dataclass:
    el bootstrap la llama decenas de miles de veces."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0.0
    k = max(1, math.ceil(alpha * arr.size))
    return float(np.sort(arr)[:k].mean())


@dataclass(slots=True)
class SideScores:
    """Los scores de UN lado (real o sintetico), organizados por bloque.

    `blocks` es la unidad de remuestreo: una sub-ventana real o una muestra sintetica, con
    TODOS sus folds dentro. `by_config[config_id][b]` son los scores del bloque b.
    """

    side: str
    block_ids: list[str]
    by_config: dict[str, list[list[float]]]
    baselines: dict[str, list[list[float]]]
    approvals: dict[str, list[bool]]
    dropped: list[str]
    # Operaciones ventana a ventana, en la MISMA forma que `by_config`. Vacio solo si las
    # filas vienen de una corrida anterior a que se instrumentara (`trades_by_fold`).
    trades_by_config: dict[str, list[list[int]]] = dataclasses.field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.block_ids)

    def pooled(self, config_id: str) -> list[float]:
        return [s for block in self.by_config[config_id] for s in block]

    def pooled_trades(self, config_id: str) -> list[int] | None:
        blocks = self.trades_by_config.get(config_id)
        if blocks is None:
            return None
        return [t for block in blocks for t in block]

    def reward(self, config_id: str, alpha: float) -> float:
        return _cvar(self.pooled(config_id), alpha)

    def activity(self, config_id: str) -> ActivityStats | None:
        """La actividad sobre EXACTAMENTE las mismas ventanas que producen `reward`."""
        trades = self.pooled_trades(config_id)
        return None if trades is None else measure_activity(trades)

    def rankable(self, config_id: str) -> bool:
        return DEFAULT_ACTIVITY_FLOOR.eligible(self.activity(config_id))


def _fold_trades(row: dict) -> list[int] | None:
    """Operaciones ventana a ventana de una unidad, o None si esa unidad es de una corrida
    anterior a la instrumentacion. Se exige que haya una cifra por score: media unidad
    medida no es una unidad medida."""
    trades = row.get("trades_by_fold")
    if trades is None or len(trades) != len(row.get("scores", [])):
        return None
    return [int(t) for t in trades]


def collect_side(
    rows: Sequence[dict],
    side: str,
    config_ids: Sequence[str],
    *,
    alpha: float = DEFAULT_CVAR_ALPHA,
) -> SideScores:
    """
    Ordena las filas crudas de un lado en la matriz (configuracion x bloque).

    Una configuracion a la que le falte cualquier bloque, o que tenga alguna unidad
    fallida, se DESCARTA entera y se declara: comparar rankings exige que las 16 se hayan
    medido sobre exactamente el mismo material.
    """
    side_rows = [r for r in rows if r["side"] == side]
    block_ids = sorted({(r["unit_id"], r["path_index"]) for r in side_rows})
    labels = [f"{u}#{p}" if p else u for u, p in block_ids]

    indexed: dict[str, dict[tuple[str, int], dict]] = {}
    for row in side_rows:
        indexed.setdefault(row["config_id"], {})[(row["unit_id"], row["path_index"])] = row

    by_config: dict[str, list[list[float]]] = {}
    trades_by_config: dict[str, list[list[int]]] = {}
    approvals: dict[str, list[bool]] = {}
    dropped: list[str] = []
    for config_id in config_ids:
        cells = indexed.get(config_id, {})
        if len(cells) != len(block_ids) or any(cells[k]["failed"] for k in block_ids):
            dropped.append(config_id)
            continue
        by_config[config_id] = [list(cells[k]["scores"]) for k in block_ids]
        # La actividad se lee ventana a ventana o no se lee: repartir el total del bloque
        # entre sus folds daria una media correcta y una fraccion de ventanas vacias
        # INVENTADA, que es justo el dato que hace falta.
        per_fold = [_fold_trades(cells[k]) for k in block_ids]
        if all(t is not None for t in per_fold):
            trades_by_config[config_id] = [list(t) for t in per_fold]
        # El veredicto por bloque se RECALCULA aqui en vez de arrastrar el que la unidad
        # guardo al correrse. Es la misma cuenta con los mismos datos, pero el gate depende
        # del suelo de actividad vigente: un informe re-analizado no puede seguir
        # publicando el veredicto de una version anterior del suelo sin decirlo.
        approvals[config_id] = [
            gate(
                list(cells[k]["scores"]),
                dict(cells[k].get("baseline_scores", {})),
                alpha=alpha,
                trades=_fold_trades(cells[k]),
            ).approved
            for k in block_ids
        ]

    # Los baselines NO dependen de la configuracion (son carteras pasivas sobre las mismas
    # ventanas), asi que basta con quedarse con una copia por bloque. Si dos filas del
    # mismo bloque no coincidieran, algo no seria determinista y hay que saberlo.
    baselines: dict[str, list[list[float]]] = {}
    for name in sorted(BASELINE_LABELS):
        series: list[list[float]] = []
        for key in block_ids:
            found = [
                r["baseline_scores"][name]
                for r in side_rows
                if not r["failed"]
                and (r["unit_id"], r["path_index"]) == key
                and name in r.get("baseline_scores", {})
            ]
            if not found:
                series = []
                break
            if any(f != found[0] for f in found[1:]):
                logger.warning(
                    "Baseline '%s' no reproducible en el bloque %s del lado %s", name, key, side
                )
            series.append(list(found[0]))
        if series:
            baselines[name] = series

    if dropped:
        logger.warning(
            "Lado %s: %d/%d configuraciones descartadas por unidades fallidas o incompletas: %s",
            side, len(dropped), len(config_ids), ", ".join(dropped),
        )
    if by_config and not trades_by_config:
        logger.warning(
            "Lado %s: las unidades no traen operaciones fold a fold (`trades_by_fold`). Son "
            "de una corrida anterior a la instrumentacion: el informe saldra SIN actividad "
            "ni elegibilidad. Re-corre el estudio para tenerlas.",
            side,
        )
    return SideScores(
        side=side,
        block_ids=labels,
        by_config=by_config,
        baselines=baselines,
        approvals=approvals,
        dropped=dropped,
        trades_by_config=trades_by_config,
    )


def pooled_gate(side: SideScores, config_id: str, alpha: float) -> BaselineGate:
    """Gate de baselines sobre la distribucion PUESTA EN COMUN, no un gate por unidad
    promediado: la recompensa que rankea es la agregada, y el rival tiene que serlo
    tambien."""
    return gate(
        side.pooled(config_id),
        {name: [s for block in blocks for s in block] for name, blocks in side.baselines.items()},
        alpha=alpha,
        missing=tuple(n for n in sorted(BASELINE_LABELS) if n not in side.baselines),
        trades=side.pooled_trades(config_id),
    )


# ---------------------------------------------------------------- transferencia ------


def _ranks_best_first(rewards: Sequence[float]) -> list[int]:
    """Rango 1 = la mejor. Empates: el orden estable por indice (no hay empates reales
    entre CVaR de distribuciones continuas, pero mejor no depender de eso)."""
    order = sorted(range(len(rewards)), key=lambda i: (-rewards[i], i))
    ranks = [0] * len(rewards)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def _percentile_ci(values: Sequence[float], pct: float) -> tuple[float, float]:
    clean = np.asarray([v for v in values if v == v], dtype=float)
    if clean.size < 2:
        return float("nan"), float("nan")
    lo = (100.0 - pct) / 2.0
    return float(np.percentile(clean, lo)), float(np.percentile(clean, 100.0 - lo))


def block_bootstrap_rho(
    real: SideScores,
    synth: SideScores,
    config_ids: Sequence[str],
    *,
    alpha: float,
    n_samples: int,
    seed: int,
) -> dict:
    """
    Intervalo del rho por BOOTSTRAP DE BLOQUES.

    El bloque es la unidad de evidencia de cada lado —una sub-ventana real, una muestra
    sintetica— y se remuestrea ENTERO, con sus folds dentro. Dos razones: los folds de un
    mismo CPCV comparten calendario (reusan los mismos tramos en 15 combinaciones) y los
    retornos dentro de una ventana estan serialmente correlacionados. Un bootstrap iid
    sobre scores sueltos fingiria una independencia que no existe y devolveria un intervalo
    demasiado estrecho.

    Los dos lados se remuestrean de forma INDEPENDIENTE (son dos fuentes de ruido
    distintas) pero las 16 configuraciones comparten los mismos indices dentro de cada
    lado: la comparacion es pareada, como lo es el ranking.
    """
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_samples):
        idx_real = rng.integers(0, real.n_blocks, real.n_blocks)
        idx_synth = rng.integers(0, synth.n_blocks, synth.n_blocks)
        real_rewards = [
            _cvar(np.concatenate([real.by_config[c][i] for i in idx_real]), alpha)
            for c in config_ids
        ]
        synth_rewards = [
            _cvar(np.concatenate([synth.by_config[c][i] for i in idx_synth]), alpha)
            for c in config_ids
        ]
        draws.append(spearman(real_rewards, synth_rewards))

    usable = [d for d in draws if d == d]
    lo, hi = _percentile_ci(usable, CI_PCT)
    return {
        "method": "bootstrap por bloques (sub-ventana real / muestra sintetica entera)",
        "n_samples": n_samples,
        "n_usable": len(usable),
        "n_blocks_real": real.n_blocks,
        "n_blocks_synthetic": synth.n_blocks,
        "seed": seed,
        "ci_pct": CI_PCT,
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "median": round(float(np.median(usable)), 4) if usable else None,
        "excludes_zero": bool(usable and (lo > 0 or hi < 0)),
    }


def config_bootstrap_rho(
    real_rewards: Sequence[float],
    synth_rewards: Sequence[float],
    *,
    n_samples: int,
    seed: int,
) -> dict:
    """
    Intervalo SECUNDARIO: remuestrea las 16 CONFIGURACIONES, no la evidencia.

    Responde a otra pregunta —"¿saldria el mismo rho con otras 16 configuraciones del
    mismo hipercubo?"— y por eso se publica aparte y no como el intervalo principal.
    Con solo 16 puntos, una correlacion de rangos tiene un error grande por si sola; este
    intervalo es el que lo hace visible.
    """
    rng = np.random.default_rng(seed + 1)
    x = np.asarray(real_rewards, dtype=float)
    y = np.asarray(synth_rewards, dtype=float)
    draws: list[float] = []
    for _ in range(n_samples):
        idx = rng.integers(0, x.size, x.size)
        draws.append(spearman(x[idx], y[idx]))
    usable = [d for d in draws if d == d]
    lo, hi = _percentile_ci(usable, CI_PCT)
    return {
        "method": "bootstrap sobre las configuraciones de la rejilla",
        "n_samples": n_samples,
        "n_usable": len(usable),
        "seed": seed + 1,
        "ci_pct": CI_PCT,
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "excludes_zero": bool(usable and (lo > 0 or hi < 0)),
    }


def permutation_p_value(
    real_rewards: Sequence[float],
    synth_rewards: Sequence[float],
    *,
    n_samples: int,
    seed: int,
) -> dict:
    """p-valor a dos colas por PERMUTACION del ranking sintetico. No usa la aproximacion
    normal de Spearman, que con n=16 es floja."""
    rng = np.random.default_rng(seed + 2)
    x = np.asarray(real_rewards, dtype=float)
    y = np.asarray(synth_rewards, dtype=float)
    observed = spearman(x, y)
    if observed != observed:
        return {"observed": None, "p_value": None, "n_samples": n_samples, "seed": seed + 2}
    extreme = 0
    for _ in range(n_samples):
        if abs(spearman(x, rng.permutation(y))) >= abs(observed):
            extreme += 1
    return {
        "observed": round(float(observed), 4),
        # (extremos + 1) / (n + 1): el estimador sin sesgo, que nunca devuelve p = 0
        # exacto (un p de 0 con 20.000 permutaciones seria una promesa que la muestra no
        # puede respaldar).
        "p_value": round((extreme + 1) / (n_samples + 1), 4),
        "n_samples": n_samples,
        "seed": seed + 2,
    }


def _hypergeometric_tail(observed: int, k: int, good: int, total: int) -> float:
    """P(X >= observed) al sacar `k` de `total` con `good` exitos. Es la probabilidad de
    que el solape del top-k salga asi de bien POR AZAR."""
    denom = math.comb(total, k)
    return sum(
        math.comb(good, i) * math.comb(total - good, k - i)
        for i in range(observed, min(k, good) + 1)
        if 0 <= k - i <= total - good
    ) / denom


def top_k_overlap(
    config_ids: Sequence[str],
    rank_real: Sequence[int],
    rank_synth: Sequence[int],
    *,
    k: int = TOP_K,
) -> dict:
    """
    ¿Las k mejores del sintetico caen en la MITAD BUENA del real?

    Es la lectura operativa de la transferencia, y no es la misma pregunta que el rho: un
    pre-cribado no necesita ordenar bien las 16, necesita que lo que asciende no sea
    basura. Se reporta junto a lo que saldria por azar y a la probabilidad de ese solape.
    """
    n = len(config_ids)
    good_half = (n + 1) // 2
    picked = [i for i in range(n) if rank_synth[i] <= k]
    hits = [i for i in picked if rank_real[i] <= good_half]
    return {
        "k": k,
        "n_configs": n,
        "good_half_size": good_half,
        "picked": [
            {
                "config_id": config_ids[i],
                "rank_synthetic": rank_synth[i],
                "rank_real": rank_real[i],
                "in_good_half": rank_real[i] <= good_half,
            }
            for i in sorted(picked, key=lambda i: rank_synth[i])
        ],
        "hits": len(hits),
        "expected_by_chance": round(k * good_half / n, 2),
        "p_value": round(_hypergeometric_tail(len(hits), k, good_half, n), 4),
    }


def rank_discrepancies(
    config_ids: Sequence[str],
    rank_real: Sequence[int],
    rank_synth: Sequence[int],
    *,
    threshold: int | None = None,
) -> dict:
    """
    El SIGNO de los desacuerdos grandes, que no es un detalle: los dos errores cuestan
    cosas distintas.

    `delta = rango_real - rango_sintetico`. Positivo = el sintetico la coloca MEJOR de lo
    que merece (SOBREVALORA): es el error caro, porque asciende basura a la fase real.
    Negativo = la coloca peor (INFRAVALORA): solo cuesta oportunidades perdidas.
    """
    n = len(config_ids)
    limit = n // 2 if threshold is None else threshold
    items = [
        {
            "config_id": config_ids[i],
            "rank_real": rank_real[i],
            "rank_synthetic": rank_synth[i],
            "delta": rank_real[i] - rank_synth[i],
        }
        for i in range(n)
    ]
    large = [it for it in items if abs(it["delta"]) >= limit]
    over = [it for it in large if it["delta"] > 0]
    under = [it for it in large if it["delta"] < 0]
    return {
        "threshold": limit,
        "n_large": len(large),
        "n_overrated_by_synthetic": len(over),
        "n_underrated_by_synthetic": len(under),
        "mean_abs_delta": round(float(np.mean([abs(it["delta"]) for it in items])), 2),
        "max_abs_delta": max((abs(it["delta"]) for it in items), default=0),
        "items": sorted(large, key=lambda it: -abs(it["delta"])),
    }


def trades_per_fold(rows: Sequence[dict], side: str) -> dict[str, float]:
    """Operaciones por ventana OOS de cada configuracion en un lado, en mediana sobre sus
    bloques. Es la medida de si la configuracion OPERA o solo existe."""
    per_config: dict[str, list[float]] = {}
    for row in rows:
        if row["side"] != side or row["failed"]:
            continue
        n_folds = len(row.get("scores", []))
        if n_folds:
            per_config.setdefault(row["config_id"], []).append(row["num_trades"] / n_folds)
    return {c: float(np.median(v)) for c, v in per_config.items()}


def side_activity(side: SideScores, config_ids: Sequence[str]) -> dict[str, ActivityStats]:
    """Actividad ventana a ventana de cada configuracion en un lado. Vacio si las unidades
    no traen el detalle por fold (corridas anteriores a la instrumentacion)."""
    out: dict[str, ActivityStats] = {}
    for config_id in config_ids:
        stats = side.activity(config_id)
        if stats is not None:
            out[config_id] = stats
    return out


def activity_check(
    rows: Sequence[dict],
    real: SideScores,
    synth: SideScores,
    config_ids: Sequence[str],
    *,
    alpha: float,
    min_trades_per_fold: float = ACTIVE_MIN_TRADES_PER_FOLD,
) -> dict:
    """
    ¿Estan los dos rankings midiendo criterio, o estan midiendo quien opera menos?

    Es la comprobacion que decide si un rho ~ 0 significa "el sintetico no transfiere" o
    "ninguno de los dos lados esta rankeando estrategias". Una configuracion que apenas
    abre posiciones deja la curva plana: Sharpe 0, rotacion 0, caida 0 -> headline 0
    EXACTO. Y un cero le gana a cualquier configuracion que arriesgue y pierda, asi que en
    un mercado que cae, la lista ordenada por CVaR premia la inactividad.

    Se reportan tres cosas: la correlacion entre puesto y actividad (si es fuerte, el
    ranking es una lista de inactividad disfrazada), la actividad completa de cada
    configuracion —operaciones por ventana Y fraccion de ventanas vacias, que es la firma
    aritmetica del problema— y el rho restringido a las configuraciones que operan de
    verdad en LOS DOS mundos, que es el unico subconjunto sobre el que la pregunta original
    tiene sentido.

    Ojo con la diferencia entre este subconjunto "activo" y la ELEGIBILIDAD que publica
    `eligibility_block`: aqui se exige actividad en los dos lados porque lo que se
    correlaciona son dos rankings pareados. El suelo de elegibilidad de un ranking se
    aplica sobre el lado que rankea, y solo sobre ese.
    """
    trades = {
        SIDE_REAL: trades_per_fold(rows, SIDE_REAL),
        SIDE_SYNTHETIC: trades_per_fold(rows, SIDE_SYNTHETIC),
    }
    activity = {
        SIDE_REAL: side_activity(real, config_ids),
        SIDE_SYNTHETIC: side_activity(synth, config_ids),
    }
    rewards = {
        SIDE_REAL: [real.reward(c, alpha) for c in config_ids],
        SIDE_SYNTHETIC: [synth.reward(c, alpha) for c in config_ids],
    }
    # Spearman(recompensa, actividad). Negativo = cuanto mas opera, peor puntua, que es la
    # firma de un ranking dominado por la inactividad.
    corr = {
        side: spearman(rewards[side], [trades[side].get(c, 0.0) for c in config_ids])
        for side in (SIDE_REAL, SIDE_SYNTHETIC)
    }

    # "Opera de verdad" tiene UNA sola definicion en todo el sistema: el suelo de
    # `scoring.activity`, aplicado aqui a los dos mundos porque lo que se correlaciona es
    # un ranking pareado. Solo si las unidades no traen el detalle por ventana se cae al
    # criterio antiguo (mediana de operaciones por bloque), y el informe dice cual se uso.
    if activity[SIDE_REAL] and activity[SIDE_SYNTHETIC]:
        active = [c for c in config_ids if real.rankable(c) and synth.rankable(c)]
        criterion = (
            "suelo de actividad de scoring.activity, exigido en los dos mundos "
            f"(>= {DEFAULT_ACTIVITY_FLOOR.min_median_trades_per_window:.0f} operaciones en "
            f"la ventana mediana y <= "
            f"{DEFAULT_ACTIVITY_FLOOR.max_zero_window_pct:.0f}% de ventanas vacias)"
        )
    else:
        active = [
            c for c in config_ids
            if trades[SIDE_REAL].get(c, 0.0) >= min_trades_per_fold
            and trades[SIDE_SYNTHETIC].get(c, 0.0) >= min_trades_per_fold
        ]
        criterion = (
            f"mediana de operaciones por bloque >= {min_trades_per_fold:.0f} (las unidades "
            "no traen el detalle por ventana)"
        )
    rho_active = float("nan")
    boot_active: dict | None = None
    perm_active: dict | None = None
    if len(active) >= 3:
        active_real = [real.reward(c, alpha) for c in active]
        active_synth = [synth.reward(c, alpha) for c in active]
        rho_active = spearman(active_real, active_synth)
        # El subconjunto activo es POST-HOC (se define despues de ver que el ranking real
        # premiaba no operar), y ademas tiene pocos puntos. Publicar su rho sin intervalo
        # ni p-valor invitaria a leerlo con la confianza que no tiene.
        boot_active = block_bootstrap_rho(
            real, synth, active,
            alpha=alpha, n_samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
        )
        perm_active = permutation_p_value(
            active_real, active_synth, n_samples=PERMUTATION_SAMPLES, seed=BOOTSTRAP_SEED
        )
    return {
        "min_trades_per_fold": min_trades_per_fold,
        "active_criterion": criterion,
        "floor": DEFAULT_ACTIVITY_FLOOR.as_dict(),
        "bootstrap_active": boot_active,
        "permutation_active": perm_active,
        "reward_vs_activity_real": None if corr[SIDE_REAL] != corr[SIDE_REAL]
        else round(float(corr[SIDE_REAL]), 4),
        "reward_vs_activity_synthetic": None if corr[SIDE_SYNTHETIC] != corr[SIDE_SYNTHETIC]
        else round(float(corr[SIDE_SYNTHETIC]), 4),
        "active_configs": list(active),
        "n_active": len(active),
        "spearman_active": None if rho_active != rho_active else round(float(rho_active), 4),
        "trades_per_fold": {
            c: {
                "real": round(trades[SIDE_REAL].get(c, 0.0), 2),
                "synthetic": round(trades[SIDE_SYNTHETIC].get(c, 0.0), 2),
            }
            for c in config_ids
        },
        "measured_per_window": bool(activity[SIDE_REAL] and activity[SIDE_SYNTHETIC]),
        "stats": {
            c: {
                side: (activity[side][c].as_dict() if c in activity[side] else None)
                for side in (SIDE_REAL, SIDE_SYNTHETIC)
            }
            for c in config_ids
        },
    }


def _count(items: Sequence | None) -> str:
    """Cuantos, o 'n/d' si la lista es None (no medido). Un hueco no es un cero."""
    return "n/d" if items is None else str(len(items))


def eligibility_block(
    real: SideScores,
    synth: SideScores,
    config_ids: Sequence[str],
    *,
    alpha: float,
) -> dict:
    """
    El ranking CON y SIN el suelo de actividad, y que cambia entre uno y otro.

    Publicar los dos no es un adorno: el suelo es una decision de diseno, y una decision de
    diseno que cambia al ganador tiene que enseñar el ganador que descarta. Se reporta por
    lado (cada mundo rankea con su propia actividad), mas el efecto sobre el gate de
    baselines, que es la otra cosa que el suelo modifica.

    `changes_winner` es la pregunta corta: ¿elegiriamos otra configuracion? Si la respuesta
    es si, hay que decirlo en la primera linea del informe, y por eso viaja en el bloque y
    no en un pie de pagina.
    """
    out: dict = {"floor": DEFAULT_ACTIVITY_FLOOR.as_dict(), "sides": {}}
    for side, scores in ((SIDE_REAL, real), (SIDE_SYNTHETIC, synth)):
        rewards = {c: scores.reward(c, alpha) for c in config_ids}
        activity = side_activity(scores, config_ids)
        measured = len(activity) == len(config_ids)
        rankable = [c for c in config_ids if scores.rankable(c)] if measured else []
        gates = {c: pooled_gate(scores, c, alpha) for c in config_ids}

        def order(subset: Sequence[str]) -> list[str]:
            return sorted(subset, key=lambda c: (-rewards[c], c))

        full_order = order(config_ids)
        floor_order = order(rankable)
        out["sides"][side] = {
            "measured": measured,
            "n_configs": len(config_ids),
            "n_rankable": len(rankable),
            "dropped": [c for c in config_ids if c not in rankable],
            "ranking_all": full_order,
            "ranking_rankable": floor_order,
            "winner_all": full_order[0] if full_order else None,
            "winner_rankable": floor_order[0] if floor_order else None,
            "changes_winner": bool(
                measured and floor_order and full_order and floor_order[0] != full_order[0]
            ) if measured else None,
            # El gate ya incorpora el suelo (`gate.approved`), asi que "sin suelo" es su
            # componente `beats_baselines`: exactamente el veredicto de antes. Sin actividad
            # medida el lado "con suelo" es None y no una lista vacia: "no aprueba ninguna"
            # y "no se pudo comprobar" no son el mismo hecho.
            "approved_without_floor": sorted(c for c in config_ids if gates[c].beats_baselines),
            "approved_with_floor": (
                sorted(c for c in config_ids if gates[c].approved) if measured else None
            ),
            "reasons": {
                c: list(DEFAULT_ACTIVITY_FLOOR.reasons(activity.get(c)))
                for c in config_ids
                if c not in rankable
            },
        }
    return out


def verdict_for(rho: float, bootstrap: dict) -> dict:
    """Aplica la regla de decision declarada en el modulo. El veredicto lo fija el
    estimador puntual (que es lo que la regla dice); el intervalo NO lo cambia, pero se
    reporta si excluye el cero, porque un rho de 0,35 con un intervalo que cruza el cero
    no es la misma evidencia que uno que no lo cruza."""
    if rho != rho:
        key, decision, flow = VERDICT_NONE, "rho no calculable", FLOW_NONE
    elif rho >= RHO_ACCEPT:
        key = VERDICT_TRANSFERS
        decision = f"rho = {rho:+.3f} >= {RHO_ACCEPT:+.2f}"
        flow = FLOW_TRANSFERS
    elif rho <= -RHO_ACCEPT:
        key = VERDICT_INVERTED
        decision = f"rho = {rho:+.3f} <= {-RHO_ACCEPT:+.2f}"
        flow = FLOW_NONE
    else:
        key = VERDICT_NONE
        decision = f"|rho| = {abs(rho):.3f} < {RHO_ACCEPT:.2f}"
        flow = FLOW_NONE
    return {
        "key": key,
        "rho": None if rho != rho else round(float(rho), 4),
        "threshold": RHO_ACCEPT,
        "decision": decision,
        "flow": flow,
        "ci_excludes_zero": bootstrap["excludes_zero"],
        "transfers": key == VERDICT_TRANSFERS,
    }


# --------------------------------------------------------------------- informe -------


def analyze(rows: Sequence[dict], plan: dict, specs: Sequence[StrategySpec]) -> dict:
    """Convierte las unidades crudas en los dos rankings y la medida de transferencia.

    `plan` es el diccionario (no el dataclass) para que `--analyze-only` re-analice el
    informe publicado con SU plan, no con el que se reconstruya hoy."""
    alpha = plan["validation"]["cvar_alpha"]
    config_ids = [s.id for s in specs]
    by_id = {s.id: s for s in specs}

    real = collect_side(rows, SIDE_REAL, config_ids, alpha=alpha)
    synth = collect_side(rows, SIDE_SYNTHETIC, config_ids, alpha=alpha)
    kept = [c for c in config_ids if c in real.by_config and c in synth.by_config]
    dropped = sorted(set(real.dropped) | set(synth.dropped))
    if len(kept) < 3:
        raise ValueError(
            f"Solo {len(kept)} configuraciones completas en los dos lados; no hay ranking "
            "que correlacionar"
        )

    real_rewards = [real.reward(c, alpha) for c in kept]
    synth_rewards = [synth.reward(c, alpha) for c in kept]
    rank_real = _ranks_best_first(real_rewards)
    rank_synth = _ranks_best_first(synth_rewards)
    rho = spearman(real_rewards, synth_rewards)

    bootstrap = block_bootstrap_rho(
        real, synth, kept,
        alpha=alpha, n_samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
    )
    logger.info(
        "Spearman(real, sintetico) = %+.4f | IC%.0f%% por bloques [%+.3f, %+.3f]",
        rho, CI_PCT, bootstrap["lo"], bootstrap["hi"],
    )

    activity = activity_check(rows, real, synth, kept, alpha=alpha)
    logger.info(
        "Actividad: Spearman(recompensa, operaciones) real=%s sintetico=%s | %d/%d configs "
        "operan de verdad en los dos mundos -> rho activo=%s",
        activity["reward_vs_activity_real"], activity["reward_vs_activity_synthetic"],
        activity["n_active"], len(kept), activity["spearman_active"],
    )
    eligibility = eligibility_block(real, synth, kept, alpha=alpha)
    real_elig = eligibility["sides"][SIDE_REAL]
    logger.info(
        "Elegibilidad (lado real): %d/%d rankeables | ganador sin suelo=%s, con suelo=%s "
        "(%s) | aprueban el gate %d -> %d",
        real_elig["n_rankable"], real_elig["n_configs"], real_elig["winner_all"],
        real_elig["winner_rankable"],
        "CAMBIA LA ELECCION" if real_elig["changes_winner"] else "misma eleccion",
        len(real_elig["approved_without_floor"]), _count(real_elig["approved_with_floor"]),
    )

    def side_row(side: SideScores, config_id: str) -> dict:
        pooled = side.pooled(config_id)
        stats = aggregate_reward(pooled, alpha=alpha)
        gate_result = pooled_gate(side, config_id, alpha)
        return {
            **stats.as_dict(),
            "n_blocks": side.n_blocks,
            "approved_blocks": sum(side.approvals[config_id]),
            "approved_pooled": gate_result.approved,
            "gate": gate_result.as_dict(),
            "scores_by_block": [[round(s, 4) for s in block] for block in side.by_config[config_id]],
        }

    configs = [
        {
            "config_id": config_id,
            "family": by_id[config_id].type,
            "params": by_id[config_id].params,
            "rank_real": rank_real[i],
            "rank_synthetic": rank_synth[i],
            "rank_delta": rank_real[i] - rank_synth[i],
            "reward_real": round(real_rewards[i], 4),
            "reward_synthetic": round(synth_rewards[i], 4),
            "trades_per_fold": activity["trades_per_fold"][config_id],
            "activity": activity["stats"][config_id],
            "rankable_real": real.rankable(config_id),
            "rankable_synthetic": synth.rankable(config_id),
            "active": config_id in activity["active_configs"],
            "real": side_row(real, config_id),
            "synthetic": side_row(synth, config_id),
        }
        for i, config_id in enumerate(kept)
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "configs": configs,
        "configs_dropped": dropped,
        "blocks": {"real": real.block_ids, "synthetic": synth.block_ids},
        "baselines": {
            "real": {
                name: round(_cvar([s for b in blocks for s in b], alpha), 4)
                for name, blocks in real.baselines.items()
            },
            "synthetic": {
                name: round(_cvar([s for b in blocks for s in b], alpha), 4)
                for name, blocks in synth.baselines.items()
            },
            "missing": sorted(set(BASELINE_LABELS) - set(real.baselines) - set(synth.baselines)),
        },
        "rankings": {
            "real": [c["config_id"] for c in sorted(configs, key=lambda c: c["rank_real"])],
            "synthetic": [
                c["config_id"] for c in sorted(configs, key=lambda c: c["rank_synthetic"])
            ],
            # El mismo orden, quitando las que no superan el suelo de actividad. Los dos se
            # publican: si la eleccion cambia, se ve sin tener que recalcular nada.
            "real_rankable": eligibility["sides"][SIDE_REAL]["ranking_rankable"],
            "synthetic_rankable": eligibility["sides"][SIDE_SYNTHETIC]["ranking_rankable"],
        },
        "eligibility": eligibility,
        "transfer": {
            "spearman": None if rho != rho else round(float(rho), 4),
            "n_configs": len(kept),
            "bootstrap_blocks": bootstrap,
            "bootstrap_configs": config_bootstrap_rho(
                real_rewards, synth_rewards,
                n_samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
            ),
            "permutation": permutation_p_value(
                real_rewards, synth_rewards,
                n_samples=PERMUTATION_SAMPLES, seed=BOOTSTRAP_SEED,
            ),
            "top_k": top_k_overlap(kept, rank_real, rank_synth),
            "discrepancies": rank_discrepancies(kept, rank_real, rank_synth),
            "activity": activity,
        },
        "verdict": verdict_for(rho, bootstrap),
        "caveats": caveats(plan, real, synth, activity, eligibility),
        "leakage": {
            "folds_audited": sum(len(r.get("scores", [])) for r in rows if not r["failed"]),
            "clean": all(r.get("leakage_ok", True) for r in rows if not r["failed"]),
        },
    }


def _activity_reading(activity: dict) -> str:
    """Que hay que concluir del control de inactividad, ya leido. Se escribe aqui y no en
    el dashboard para que el informe -que es lo que se audita- lleve la conclusion, no la
    instruccion de como sacarla."""
    rho_active = activity["spearman_active"]
    if rho_active is None:
        return (
            "No quedan configuraciones suficientes para rehacer el ranking sin ellas, asi que "
            "el veredicto NO puede separarse de la inactividad: leelo con esa reserva."
        )
    boot = activity["bootstrap_active"] or {}
    cautela = (
        ""
        if boot.get("excludes_zero")
        else (
            " Es un subconjunto POST-HOC y pequeno, y su intervalo por bloques "
            f"[{boot.get('lo')}, {boot.get('hi')}] cruza el cero: senal fuerte, no prueba."
        )
    )
    if rho_active <= -RHO_ACCEPT:
        return (
            "Al quitar la inactividad el acuerdo no aparece: se vuelve NEGATIVO. Entre "
            "configuraciones que actuan de verdad, el sintetico tiende a invertir el orden del "
            "mercado, asi que el veredicto de arriba no es un artefacto de las inactivas —es, si "
            "acaso, mas benigno que la realidad." + cautela
        )
    if rho_active >= RHO_ACCEPT:
        return (
            "Al quitar la inactividad SI aparece acuerdo, lo que cambia la lectura: el veredicto "
            "global esta contaminado por las configuraciones que apenas operan y la transferencia "
            "habria que juzgarla sobre el subconjunto activo." + cautela
        )
    return (
        "El acuerdo sigue siendo nulo al quitar la inactividad, asi que el veredicto de arriba no "
        "es un artefacto de las configuraciones que apenas operan." + cautela
    )


def caveats(
    plan: dict,
    real: SideScores,
    synth: SideScores,
    activity: dict,
    eligibility: dict,
) -> list[dict]:
    """Los limites del estudio, EN el informe y no en una nota al pie que nadie lee.

    Van aqui y no en la prosa del dashboard porque el consumidor de esta cifra es quien
    decide si se disena contra el sintetico, y esa decision necesita saber en que direccion
    empuja cada sesgo."""
    real_plan = plan["real"]
    n_omitted = len(real_plan["symbols_omitted"])
    elig = eligibility["sides"][SIDE_REAL]
    floor = eligibility["floor"]
    return [
        {
            "key": "inactividad_premiada",
            "title": "El CVaR premia no operar, y por eso el ranking tiene un suelo",
            "text": (
                "Una configuracion que apenas abre posiciones deja la curva plana: Sharpe 0, "
                "rotacion 0, caida 0, headline 0 EXACTO. En un periodo en el que casi todo lo "
                "que arriesga pierde, ese cero gana. La correlacion entre recompensa y "
                f"operaciones por ventana lo mide: {activity['reward_vs_activity_real']} en el "
                f"real y {activity['reward_vs_activity_synthetic']} en el sintetico. La "
                "respuesta NO es penalizar la baja rotacion (eso ya lo hace lambda, y el "
                "estudio de pesos midio que no estabiliza nada) sino un REQUISITO DE "
                "ELEGIBILIDAD: para entrar en el ranking hay que operar al menos "
                f"{floor['min_median_trades_per_window']:.0f} veces en la ventana mediana y "
                f"tener como mucho un {floor['max_zero_window_pct']:.0f}% de ventanas vacias "
                f"(ver scoring.activity). En el lado real eso deja {elig['n_rankable']} de "
                f"{elig['n_configs']} configuraciones, cambia al ganador de "
                f"{elig['winner_all']} a {elig['winner_rankable']} y reduce los aprobados del "
                f"gate de {len(elig['approved_without_floor'])} a "
                f"{_count(elig['approved_with_floor'])}. Las que caen conservan su recompensa "
                "publicada: no se les resta nada, se les niega competir. El rho restringido a "
                f"las {activity['n_active']} configuraciones que operan de verdad en los dos "
                f"mundos es {activity['spearman_active']}. " + _activity_reading(activity)
            ),
        },
        {
            "key": "un_solo_camino",
            "title": "El lado real es UN camino, no un ensemble",
            "text": (
                f"El sintetico son {synth.n_blocks} mundos distintos; el real son "
                f"{real.n_blocks} tramos del MISMO pasado. Ese lado no tiene ensemble y su "
                "unica fuente de dispersion son los folds del CPCV, que ademas reusan "
                "calendario. Por eso el intervalo es un bootstrap por bloques y no iid: con "
                f"{real.n_blocks} bloques reales sale ancho, y esa anchura es la verdad "
                "disponible, no un defecto del metodo."
            ),
        },
        {
            "key": "sesgo_supervivencia",
            "title": "Sesgo de supervivencia en el lado real",
            "text": (
                "Los simbolos son los que cotizan HOY en el universo operable: los pares "
                "deslistados o muertos entre 2018 y 2025 no estan en ninguna parte, y el "
                "generador sintetico no tiene ese filtro. El sesgo juega A FAVOR del lado "
                "real (su historia esta seleccionada por haber sobrevivido) y por tanto EN "
                "CONTRA de la hipotesis que este estudio quiere validar: introduce una "
                "diferencia sistematica entre los dos mundos que solo puede bajar el rho."
            ),
        },
        {
            "key": "historico_insuficiente",
            "title": "Simbolos omitidos por historico insuficiente",
            "text": (
                f"{n_omitted} pares del universo operable quedan fuera y {len(plan['symbols'])} "
                f"entran. El criterio no es a ojo: hacen falta {real_plan['min_history_days']} "
                "dias de barras dentro de la ventana (calentamiento de la estrategia mas un "
                "grupo de CPCV), porque por debajo de eso el par no llega a operarse ni en una "
                "sola ventana OOS. Los omitidos se DECLARAN con su recuento y sus fechas; no "
                "se rellenan ni se sustituyen, porque un par listado hace unos meses no tiene "
                "2018 y fabricarselo seria inventar el dato que se quiere medir."
            ),
        },
        {
            "key": "cobertura_creciente",
            "title": "El universo real crece con el tiempo; el sintetico no",
            "text": (
                "Varios de los pares que si entran empiezan a cotizar a mitad de la ventana, "
                "asi que las sub-ventanas antiguas se operan con menos activos que las "
                "recientes. En el mundo sintetico los "
                f"{len(plan['symbols'])} activos existen desde el primer dia. Es una "
                "diferencia estructural entre los dos lados que no se puede eliminar sin "
                "tirar la mitad del historico."
            ),
        },
        {
            "key": "rejilla_pequena",
            "title": "16 configuraciones, y de solo dos familias",
            "text": (
                "Una correlacion de rangos sobre 16 puntos tiene un error de muestreo grande: "
                "sirve para distinguir 'ordena como el mercado' de 'no ordena', no para "
                "comparar 0,35 con 0,45. Ademas la rejilla es un hipercubo latino sobre el "
                "espacio de busqueda, no la distribucion que el CEM visita de verdad al "
                "converger; la transferencia medida es la del espacio, no la de la region "
                "final del optimizador."
            ),
        },
        {
            "key": "folds_dependientes",
            "title": "Los folds de CPCV no son independientes",
            "text": (
                f"C({plan['validation']['n_groups']}, "
                f"{plan['validation']['n_test_groups']}) = "
                f"{plan['validation']['n_folds']} ventanas salen de reusar los mismos tramos "
                "de calendario en combinaciones distintas. Cuentan como evidencia sobre la "
                "estabilidad del resultado, no como observaciones independientes; el "
                "bootstrap por bloques es justo lo que impide tratarlas como tales."
            ),
        },
        {
            "key": "cabecera_descartada",
            "title": "La cabecera del historico se descarta",
            "text": (
                f"{real_plan['head_discarded_days']} dias iniciales no completan una "
                "sub-ventana del tamano de un camino sintetico y quedan fuera. El corte se "
                "hace por el principio a proposito (es el tramo con menos pares vivos), y se "
                "declara porque cambia que anos entran en la comparacion."
            ),
        },
    ]


# ---------------------------------------------------------------------- main ---------


def resolve_library(store: SyntheticStore, requested: str) -> tuple[str, bool]:
    """La libreria pedida si existe; si no, la anterior, dejandolo escrito."""
    available = set(store.list_libraries())
    if requested in available:
        return requested, False
    if FALLBACK_LIBRARY_ID in available:
        logger.warning(
            "La libreria '%s' no existe: el estudio se corre sobre '%s' y el informe lo "
            "declara (library_is_fallback)", requested, FALLBACK_LIBRARY_ID,
        )
        return FALLBACK_LIBRARY_ID, True
    raise FileNotFoundError(f"No hay libreria '{requested}' ni '{FALLBACK_LIBRARY_ID}'")


def build_plan(args: argparse.Namespace) -> tuple[StudyPlan, list[StrategySpec], list[RealWindow]]:
    store = SyntheticStore()
    library_id, is_fallback = resolve_library(store, args.library)
    manifest = store.load_manifest(library_id)
    base_config = load_config(args.config)

    synth_start, synth_end = study_window(manifest, base_config.runner.lookback_days)
    window_days = (synth_end - synth_start).days
    min_history = base_config.runner.lookback_days + window_days // N_GROUPS

    real_start = pd.Timestamp(args.start, tz="UTC")
    real_end = pd.Timestamp(args.end, tz="UTC")
    requested = crypto_universe(base_config)
    logger.info(
        "Barras reales (%s, %s -> %s) para %d pares cripto%s",
        args.exchange, args.start, args.end, len(requested),
        " [offline: solo cache]" if args.offline else "",
    )
    service = build_service(args.exchange, offline=args.offline)
    bars = fetch_real_bars(
        requested, real_start.to_pydatetime(), real_end.to_pydatetime(), service
    )
    kept, dropped = audit_real_symbols(
        bars, requested, start=real_start, end=real_end, min_history_days=min_history
    )
    if not kept:
        raise ValueError("Ningun simbolo real supera el minimo de historico")

    # El universo comun: lo que existe a la vez en el mercado y en la libreria. Es la unica
    # forma de que los dos lados corran el MISMO backtest con lo unico cambiado siendo el
    # mundo del que salen los precios.
    library_symbols = {str(u["symbol"]) for u in manifest.universe}
    symbols = tuple(a.symbol for a in kept if a.symbol in library_symbols)
    excluded_lib = tuple(
        sorted(
            s for s in library_symbols
            if detect_asset_class(s) is AssetClass.CRYPTO and s not in symbols
        )
    )
    no_counterpart = [a for a in kept if a.symbol not in library_symbols]
    if not symbols:
        raise ValueError("El universo comun (mercado ∩ libreria) esta vacio")

    windows = real_windows(
        real_start.to_pydatetime(), real_end.to_pydatetime(), window_days
    )
    head = (windows[0].start - real_start.to_pydatetime()).days

    scenario_ids = tuple(s["id"] for s in manifest.scenarios)
    if args.scenarios:
        scenario_ids = scenario_ids[: args.scenarios]

    omitted = [
        *(a.as_dict() for a in dropped),
        *(
            {**a.as_dict(), "reason": "sin contraparte en la libreria sintetica"}
            for a in no_counterpart
        ),
    ]
    logger.info(
        "Universo comun (%d): %s | omitidos %d",
        len(symbols), ", ".join(symbols), len(omitted),
    )

    plan = StudyPlan(
        library_id=library_id,
        library_is_fallback=is_fallback,
        config_path=str(args.config),
        exchange=args.exchange,
        offline=bool(args.offline),
        families=tuple(args.families),
        configs_per_family=args.configs_per_family,
        study_seed=STUDY_SEED,
        symbols=symbols,
        real_source_start=real_start.to_pydatetime().isoformat(),
        real_start=windows[0].start.isoformat(),
        real_end=windows[-1].end.isoformat(),
        real_windows=tuple(w.as_dict() for w in windows),
        real_head_discarded_days=head,
        real_min_history_days=min_history,
        real_symbols_omitted=tuple(omitted),
        library_symbols_omitted=excluded_lib,
        synthetic_start=synth_start.isoformat(),
        synthetic_end=synth_end.isoformat(),
        scenario_ids=scenario_ids,
        n_paths=min(args.paths, manifest.n_paths),
        window_days=window_days,
        scheme=SCHEME_CPCV,
        n_groups=N_GROUPS,
        n_test_groups=N_TEST_GROUPS,
        n_folds=math.comb(N_GROUPS, N_TEST_GROUPS),
        purge_days=resolve_purge_days(base_config),
        embargo_days=resolve_embargo_days(synth_start, synth_end),
        periods_per_year=periods_per_year_for_symbols(symbols),
        cvar_alpha=DEFAULT_CVAR_ALPHA,
        starting_equity=DEFAULT_STARTING_EQUITY,
    )
    return plan, build_specs(tuple(args.families), args.configs_per_family), windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--library", default=DEFAULT_LIBRARY_ID)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    parser.add_argument("--start", default=DEFAULT_REAL_START)
    parser.add_argument("--end", default=DEFAULT_REAL_END)
    parser.add_argument("--configs-per-family", type=int, default=CONFIGS_PER_FAMILY)
    parser.add_argument(
        "--families", nargs="+", default=list(FAMILIES),
        help=(
            "Familias de la rejilla, EN ORDEN. El orden fija las semillas del hipercubo, asi "
            "que reordenarlas produce otras configuraciones con los mismos nombres."
        ),
    )
    parser.add_argument("--scenarios", type=int, default=8, help="0 = todos")
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument(
        "--offline", action="store_true", help="No llamar al exchange: usar solo la cache."
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--verify-determinism", type=int, default=0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--overwrite-published", action="store_true",
        help="Permite reemplazar un informe publicado con OTRA rejilla. Ver shared/reports.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.ERROR)

    plan, specs, windows = build_plan(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units_path = out_dir / f"units_{plan.library_id}.json"
    report_path = transfer_report_path(plan.library_id, out_dir)

    # ANTES de gastar dos horas de CPU: si el fichero de destino publica otra rejilla, este
    # estudio no lo sustituye. Un `--library` mal tecleado destruiria la evidencia con la que
    # se decidio que el sintetico no ordena como el mercado.
    for target in (units_path, report_path):
        guard_published_grid(target, plan.families, overwrite=args.overwrite_published)

    if args.analyze_only:
        payload = json.loads(units_path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        plan_dict = payload["plan"]
        logger.info("Reusando %d unidades de %s", len(rows), units_path)
    else:
        rows = run_units(plan, specs, windows, args.workers)
        plan_dict = plan.as_dict()
        units_path.write_text(
            json.dumps({"plan": plan_dict, "rows": rows}, indent=1), encoding="utf-8"
        )
        logger.info("Unidades crudas -> %s", units_path)

    report = analyze(rows, plan_dict, specs)
    if args.verify_determinism and not args.analyze_only:
        report["determinism"] = verify_determinism(
            plan, specs, windows, rows, args.verify_determinism
        )

    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Informe -> %s", report_path)
    _print_report(report)
    return 0


def _print_report(report: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    t = report["transfer"]
    v = report["verdict"]
    plan = report["plan"]
    print("\n=== TRANSFERENCIA DE RANKING: REAL vs SINTETICO ===")
    print(
        f"  libreria {plan['library_id']}"
        f"{' (FALLBACK: la pedida no existe)' if plan['library_is_fallback'] else ''} | "
        f"{len(plan['symbols'])} simbolos | {t['n_configs']} configuraciones"
    )
    print(
        f"  real: {len(plan['real']['sub_windows'])} sub-ventanas de "
        f"{plan['validation']['window_days']}d | sintetico: "
        f"{plan['synthetic']['n_samples']} muestras | {plan['validation']['n_folds']} folds CPCV "
        f"(purga {plan['validation']['purge_days']}d, embargo {plan['validation']['embargo_days']}d)"
    )
    print(
        f"\n{'config':<22}{'real':>9}{'sint.':>9}{'r.real':>8}{'r.sint':>8}{'delta':>7}"
        f"{'ops/vent':>10}{'vacias%':>9}{'rank?':>7}"
    )
    for row in sorted(report["configs"], key=lambda c: c["rank_real"]):
        act = (row.get("activity") or {}).get("real") or {}
        print(
            f"{row['config_id']:<22}{row['reward_real']:>9.3f}{row['reward_synthetic']:>9.3f}"
            f"{row['rank_real']:>8}{row['rank_synthetic']:>8}{row['rank_delta']:>+7}"
            f"{act.get('trades_per_window', float('nan')):>10.1f}"
            f"{act.get('zero_window_pct', float('nan')):>9.0f}"
            f"{('si' if row.get('rankable_real') else 'NO'):>7}"
        )

    b, cb, p = t["bootstrap_blocks"], t["bootstrap_configs"], t["permutation"]
    rho_txt = "n/a" if t["spearman"] is None else f"{t['spearman']:+.3f}"
    print(
        f"\n  Spearman = {rho_txt} | IC{b['ci_pct']:.0f}% por bloques "
        f"[{b['lo']:+.3f}, {b['hi']:+.3f}] ({b['n_blocks_real']} bloques reales, "
        f"{b['n_blocks_synthetic']} sinteticos)"
    )
    print(
        f"  IC{cb['ci_pct']:.0f}% remuestreando configuraciones [{cb['lo']:+.3f}, "
        f"{cb['hi']:+.3f}] | p (permutacion) = {p['p_value']}"
    )
    k = t["top_k"]
    print(
        f"  Top-{k['k']} del sintetico en la mitad buena del real: {k['hits']}/{k['k']} "
        f"(azar {k['expected_by_chance']}, p = {k['p_value']})"
    )
    d = t["discrepancies"]
    print(
        f"  Desacuerdos >= {d['threshold']} puestos: {d['n_large']} "
        f"({d['n_overrated_by_synthetic']} sobrevalorados por el sintetico, "
        f"{d['n_underrated_by_synthetic']} infravalorados) | |delta| medio "
        f"{d['mean_abs_delta']}"
    )
    a = t["activity"]
    ba, pa = a["bootstrap_active"], a["permutation_active"]
    print(
        f"  Control de inactividad: Spearman(recompensa, operaciones) = "
        f"{a['reward_vs_activity_real']} real / {a['reward_vs_activity_synthetic']} sintetico"
    )
    print(
        f"  rho sobre las {a['n_active']} que operan en ambos = {a['spearman_active']}"
        + (
            f" | IC{ba['ci_pct']:.0f}% [{ba['lo']:+.3f}, {ba['hi']:+.3f}] | p = {pa['p_value']}"
            if ba and pa else ""
        )
    )
    e = report["eligibility"]["sides"][SIDE_REAL]
    f = report["eligibility"]["floor"]
    print(
        f"\n  SUELO DE ACTIVIDAD (>= {f['min_median_trades_per_window']:.0f} ops en la "
        f"ventana mediana, <= {f['max_zero_window_pct']:.0f}% de ventanas vacias): "
        f"{e['n_rankable']}/{e['n_configs']} rankeables en el lado real"
    )
    print(
        f"  ganador sin suelo: {e['winner_all']} | con suelo: {e['winner_rankable']} -> "
        + ("LA ELECCION CAMBIA" if e["changes_winner"] else "misma eleccion")
    )
    print(
        f"  gate de baselines: aprueban {len(e['approved_without_floor'])} sin suelo y "
        f"{_count(e['approved_with_floor'])} con suelo "
        f"({', '.join(e['approved_with_floor'] or []) or 'ninguna'})"
    )
    print(f"\n  VEREDICTO: {v['key'].upper()} ({v['decision']})")
    print(f"  FLUJO: {v['flow']}")


if __name__ == "__main__":
    raise SystemExit(main())
