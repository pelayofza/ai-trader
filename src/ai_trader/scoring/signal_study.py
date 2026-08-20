"""
BARRIDO DE RHO: ¿a partir de que capacidad predictiva una senal paga sus costes?

El radar de senales (`observation/signal_radar.py`) mete features en la decision y su unica
defensa contra el sobreajuste es negativa: nada entra en `scoring/search_space.py`, asi que
el optimizador no puede sortear umbrales. Eso limita los grados de libertad, pero NO mide
nada. Mientras no exista este estudio, una feature con catorce observaciones puede estar
sobreajustada y el sistema no tiene forma de saberlo. Es el hueco que la evolucion del
radar dejo declarado por escrito, y es el que se cierra aqui.

La pregunta que se contesta es UNA:

    ¿cual es el rho minimo a partir del cual la estrategia bate al baseline DESPUES DE
    COSTES?

Y lo importante es de que clase de cosa es esa respuesta. No es un parametro ajustado a un
historico —el historico no entra en ninguna parte de este estudio, por (g)— sino una
propiedad del DISENO: de esta estrategia, con estos costes, con esta puerta. No se puede
sobreajustar a datos que nunca se usaron. Al mundo real le queda despues una sola pregunta,
binaria: el rho que mide mi senal, ¿esta por encima del break-even, y con margen?

    .venv\\Scripts\\python.exe -m ai_trader.scoring.signal_study --workers 7
    .venv\\Scripts\\python.exe -m ai_trader.scoring.signal_study --analyze-only

Salida: data/signal_channel/report_<lib>.json (+ units_<lib>.json con las filas crudas)

SEIS DECISIONES QUE SON LAS QUE HACEN QUE LA CIFRA SIGNIFIQUE ALGO
------------------------------------------------------------------
1. SE BARRE, NO SE CALIBRA. `rho` no se estima de ningun sitio: se recorre una rejilla. Un
   rho calibrado contra el historico real seria el mismo error que este proyecto lleva
   evitando desde el principio, con un paso mas.

2. rho = 0 ES EL GRUPO DE CONTROL, y es el test de falsacion que hoy no existe en ninguna
   otra parte del repositorio. Si una estrategia que consume la senal puntua bien con
   rho=0, no ha aprendido nada sobre el mundo: esta explotando el AR(1) del ruido, o
   simplemente esta operando menos. El criterio de lectura esta declarado en `CRITERION`,
   en codigo, ANTES de correr; si el control ensucia, el barrido se publica ANULADO y sin
   break-even.

3. LA CELDA `off` ES OTRO CONTROL, y contesta otra cosa. Sin canal y sin puerta es
   exactamente la configuracion publicada, asi que reproduce `data/transfer/units_ai_v3.json`
   score a score. Si no lo reproduce, la costura del canal ha cambiado algo del motor y
   ninguna otra cifra de este informe es creible. Los deltas se leen SIEMPRE contra rho=0
   (misma puerta, sin informacion), nunca contra `off`: entre `off` y rho=0 hay una puerta
   que opera menos, y confundir "operar menos" con "saber algo" es justo el error que este
   estudio existe para no cometer.

4. LAS 16 CONFIGURACIONES SON LAS PUBLICADAS. Se importan de `scoring.families.build_specs`
   y lo unico que se les inyecta —con `dataclasses.replace`— es el umbral de la puerta de
   senales. Ningun `config_id` es nuevo: el barrido se lee contra el ranking ya publicado.

5. LA ELECCION SE HACE FUERA DE LA MUESTRA QUE LA JUZGA. En cada celda gana la
   configuracion RANKEABLE (suelo de `scoring.activity`) con mejor recompensa en los
   escenarios de TRAIN, y lo que se publica es su recompensa en los de VALIDATION, que no
   participaron en la eleccion (`scoring.scenario_split`). Elegir y puntuar sobre las
   mismas muestras convertiria el break-even en el maximo de un ruido.

6. "DESPUES DE COSTES" NO ES UNA CORRECCION POSTERIOR. La curva de equity ya paga
   comisiones y el modelo de microestructura en cada operacion, y los baselines pasivos
   pagan sus dos patas sobre exactamente las mismas ventanas. El liston ya viene con la
   factura dentro.

LO QUE ESTE ESTUDIO NO ES
-------------------------
No mide transferencia de ranking (eso es `scoring/transfer_study.py`) ni toca datos reales.
El sustrato sintetico es el de SELECCION y el real el de VERIFICACION, y nunca el mismo
dato haciendo las dos cosas. Medir el rho de una senal de verdad es la otra mitad del
trabajo, y va en el sustrato real.

Determinismo: la rejilla de configuraciones sale de un hipercubo latino con semilla fija,
la emision del canal de un SHA-256 de (grupo, simbolo) —nunca del `hash()` de Python, que
esta salado por proceso—, la geometria de los folds solo depende de fechas y el backtest es
determinista dado (config, barras, ventana). `--verify-determinism` re-evalua unidades ya
calculadas en procesos NUEVOS y exige scores identicos.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
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
from ai_trader.backtest.validation import SCHEME_CPCV
from ai_trader.config import AppConfig, StrategySpec, load_config
from ai_trader.scoring.activity import DEFAULT_ACTIVITY_FLOOR, ActivityStats, measure_activity
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, aggregate_reward
from ai_trader.scoring.baselines import BASELINE_LABELS
from ai_trader.scoring.multiwindow import resolve_purge_days, validate_multiwindow
from ai_trader.scoring.scenario_split import ScenarioSplit, split_scenarios
from ai_trader.scoring.signal_gate import (
    GATE_MIN_TONE,
    GATE_PARAM,
    GATE_VALUE_BY_PARAM,
    gate_param_for,
)
from ai_trader.scoring.real_substrate import N_GROUPS, N_TEST_GROUPS, crypto_universe
from ai_trader.scoring.families import (
    CONFIGS_PER_FAMILY,
    FAMILIES,
    STUDY_SEED,
    build_specs,
)
from ai_trader.shared.checkpoint import UnitCheckpoint, fingerprint
from ai_trader.shared.reports import guard_published_grid
from ai_trader.shared import bars as bar_schema
from ai_trader.synthetic.fidelity import CHANNEL_MAX_LAG, channel_facts
from ai_trader.synthetic.scenarios import SignalChannel
from ai_trader.synthetic.service import study_window
from ai_trader.synthetic.signal_channel import DEFAULT_EMISSION_SEED, emit_signals
from ai_trader.synthetic.store import SyntheticStore

logger = logging.getLogger("signal_study")

OUT_DIR = Path("data") / "signal_channel"
DEFAULT_LIBRARY_ID = "ai_v3"
DEFAULT_CONFIG = Path("config") / "default.toml"

# Filas crudas del estudio de transferencia, contra las que la celda `off` se reproduce.
def transfer_units_path(library_id: str) -> Path:
    """
    Las unidades de transferencia de UNA libreria, que es contra lo que reproduce la celda
    de control.

    Se deriva de `--library` y no es una constante, y el motivo no es de estilo: la celda
    `off` compara score a score contra unidades producidas con las BARRAS de la libreria que
    se este corriendo. Apuntar siempre a ai_v3 hacia que correr el barrido sobre otro mundo
    fallara la reproduccion por el mundo y no por la costura, que es exactamente el rojo mas
    dificil de leer. Para `ai_v3` la ruta derivada es literalmente la de siempre, asi que el
    control publicado sigue siendo byte a byte el mismo.
    """
    return Path("data") / "transfer" / f"units_{library_id}.json"


# El nombre de siempre, que la prosa de `CRITERION` cita y varios tests importan.
TRANSFER_UNITS = transfer_units_path("ai_v3")

# --- la rejilla que se barre ---------------------------------------------------------
#
# rho: de "no sabe nada" a "sabe mucho". 0,20 de IC diario sostenido no existe en ninguna
# senal publica conocida; esta para que el barrido tenga un extremo en el que, si tampoco
# ahi se bate al baseline, el problema no sea la senal sino la estrategia que la consume.
DEFAULT_RHO_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
# h: dias de adelanto. Se empieza por UNO, que es el caso mas favorable a la senal (el
# retorno que anticipa es exactamente el del dia que se opera) y por tanto la cota INFERIOR
# del break-even. Si con h=1 el break-even ya sale alto, con h mayor solo puede empeorar.
DEFAULT_LEAD_GRID: tuple[int, ...] = (1,)

ARM_OFF = "off"  # sin canal y sin puerta: la configuracion publicada, tal cual

# El canal del barrido, con todo lo que NO se barre fijado y declarado:
#
# - `noise_ar = 0.30`: el ruido tiene que ser una SERIE. Con phi=0 el canal es confeti y la
#   puerta se enfrenta a un problema mas facil que el real; con phi alto, el control de
#   rho=0 se vuelve el escenario mas exigente posible (una serie muy persistente es lo mas
#   parecido a una senal que se puede construir sin informacion). 0,30 es la persistencia
#   tipica de una serie de atencion o de sentimiento diaria.
# - `informative_share = 1.0` y `coverage = 1.0`: TODO dia informativo y observado. No es
#   realismo, es aislamiento: con estos dos mandos al maximo, `rho` es literalmente el IC
#   entregado y el eje del barrido queda limpio. Bajarlos solo puede EMPEORAR el
#   break-even, asi que lo que se publica es la cota optimista.
# - UN solo canal, asi que el grupo de correlacion no muerde todavia. La breadth —cinco
#   canales de rho pequeno contra uno grande— es la primera extension natural y por eso el
#   mando ya existe; medirla con un solo canal seria imposible.
CHANNEL_NAME = "sweep"
CHANNEL_NOISE_AR = 0.30
CHANNEL_INFORMATIVE_SHARE = 1.0
CHANNEL_COVERAGE = 1.0
CHANNEL_CORR_GROUP = "sweep"

DETERMINISM_CHECKS = 6

# --- criterio de lectura, DECLARADO ANTES DE CORRER ----------------------------------
#
# Se escribe aqui, en el codigo y en el informe, y no en la prosa que se redacta despues:
# un criterio elegido a la vista del resultado no es un criterio.
CRITERION: dict[str, str] = {
    "seleccion": (
        "en cada celda gana la configuracion RANKEABLE (suelo de scoring.activity) con "
        "mayor recompensa CVaR@25% en los escenarios de TRAIN; si la puerta deja a todas "
        "por debajo del suelo se elige entre las que hay y la celda lo declara "
        "(`selection_fell_back`)"
    ),
    "lectura": (
        "se publica la recompensa de la elegida en los escenarios de VALIDATION, que no "
        "participaron en la eleccion"
    ),
    "batir": (
        "bate si esa recompensa supera la del MEJOR baseline pasivo sobre las mismas "
        "ventanas; los baselines ya pagan comisiones y deslizamiento, asi que 'despues de "
        "costes' esta dentro de la curva y no es una correccion posterior"
    ),
    "break_even": (
        "el menor rho de la rejilla que bate. Si ninguno bate, la respuesta es "
        "'> max(rejilla)' y eso YA contesta la pregunta de la evolucion siguiente"
    ),
    "control": (
        "rho=0 NO puede batir. Si bate, el barrido queda ANULADO y no se publica "
        "break-even: la estrategia estaria explotando el ruido persistente del canal o el "
        "hecho de operar menos, no informacion"
    ),
    "control_off": (
        "la celda 'off' (sin canal ni puerta) tiene que reproducir units_ai_v3.json score "
        "a score; es la prueba de que la costura del canal no cambia el motor"
    ),
    "delta": (
        "el valor de la informacion se lee SIEMPRE contra rho=0 (misma puerta, sin "
        "informacion), nunca contra 'off': entre 'off' y rho=0 hay una puerta que opera "
        "menos, y operar menos no es saber algo"
    ),
    "coste_de_la_puerta": (
        "'off' es el CERO del eje, no una celda rival: la puerta solo puede quitar "
        "entradas, asi que la curva de rho arranca por debajo y sube. Su coste se publica "
        "en los dos lados del hold-out porque el SIGNO depende del regimen (filtrar reduce "
        "exposicion: ayuda donde el mercado cae, estorba donde sube); lo que no depende del "
        "regimen es la monotonia en rho"
    ),
}


# ------------------------------------------------------------------- las celdas ------


@dataclass(frozen=True, slots=True)
class Cell:
    """Una celda del barrido: un mundo de observacion, o la ausencia de uno."""

    cell_id: str
    arm: str
    rho: float | None = None
    lead_days: int | None = None

    @property
    def is_off(self) -> bool:
        return self.arm == ARM_OFF

    @property
    def channel(self) -> SignalChannel | None:
        if self.is_off:
            return None
        return SignalChannel(
            name=CHANNEL_NAME,
            rho=float(self.rho),
            lead_days=int(self.lead_days),
            noise_ar=CHANNEL_NOISE_AR,
            informative_share=CHANNEL_INFORMATIVE_SHARE,
            coverage=CHANNEL_COVERAGE,
            corr_group=CHANNEL_CORR_GROUP,
        )

    def spec_for(self, spec: StrategySpec) -> StrategySpec:
        """La configuracion publicada con —y solo con— el umbral de la puerta inyectado.

        El PARAMETRO depende de la familia (ver `GATE_PARAM_BY_FAMILY`), pero el valor sigue
        sin tener un grado de libertad que ajustar: es el corte declarado de su eje.
        """
        if self.is_off:
            return spec
        param = gate_param_for(spec.type)
        return dataclasses.replace(
            spec, params={**spec.params, param: GATE_VALUE_BY_PARAM[param]}
        )

    def as_dict(self) -> dict:
        channel = self.channel
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "rho": self.rho,
            "lead_days": self.lead_days,
            "gate_min_tone": None if self.is_off else GATE_MIN_TONE,
            "channel": None if channel is None else channel.to_dict(),
            "expected_ic": None if channel is None else round(channel.expected_ic, 4),
        }


def build_cells(
    rho_grid: Sequence[float] = DEFAULT_RHO_GRID,
    lead_grid: Sequence[int] = DEFAULT_LEAD_GRID,
) -> list[Cell]:
    """La celda `off` y una por (rho, h). El orden es el del informe."""
    cells = [Cell(cell_id=ARM_OFF, arm=ARM_OFF)]
    for lead in lead_grid:
        for rho in rho_grid:
            cells.append(
                Cell(cell_id=f"rho{rho:g}_h{lead}", arm="on", rho=float(rho), lead_days=int(lead))
            )
    return cells


# --------------------------------------------------------------------- el plan -------


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Todo lo que define el estudio. Se serializa con los resultados: sin esto las cifras
    no son ni reproducibles ni auditables."""

    library_id: str
    config_path: str
    symbols: tuple[str, ...]
    library_symbols_omitted: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    train_scenarios: tuple[str, ...]
    validation_scenarios: tuple[str, ...]
    split_seed: int
    n_paths: int
    start: str
    end: str
    window_days: int
    cells: tuple[Cell, ...]
    families: tuple[str, ...]
    configs_per_family: int
    study_seed: int
    emission_seed: int
    scheme: str
    n_groups: int
    n_test_groups: int
    n_folds: int
    purge_days: int
    periods_per_year: int
    cvar_alpha: float
    starting_equity: float

    @property
    def n_samples(self) -> int:
        return len(self.scenario_ids) * self.n_paths

    def as_dict(self) -> dict:
        return {
            "library_id": self.library_id,
            "config_path": self.config_path,
            "symbols": list(self.symbols),
            "library_symbols_omitted": list(self.library_symbols_omitted),
            "grid": {
                "families": list(self.families),
                "configs_per_family": self.configs_per_family,
                "n_configs": len(self.families) * self.configs_per_family,
                "study_seed": self.study_seed,
                "sampler": "hipercubo latino sobre el espacio de busqueda real",
                "injected_param": GATE_PARAM,
                "injected_value": GATE_MIN_TONE,
            },
            "sweep": {
                "cells": [c.as_dict() for c in self.cells],
                "emission_seed": self.emission_seed,
                "channel_fixed": {
                    "name": CHANNEL_NAME,
                    "noise_ar": CHANNEL_NOISE_AR,
                    "informative_share": CHANNEL_INFORMATIVE_SHARE,
                    "coverage": CHANNEL_COVERAGE,
                    "corr_group": CHANNEL_CORR_GROUP,
                },
            },
            "synthetic": {
                "window": {"start": self.start, "end": self.end},
                "window_days": self.window_days,
                "scenario_ids": list(self.scenario_ids),
                "n_paths": self.n_paths,
                "n_samples": self.n_samples,
                "split": {
                    "train": list(self.train_scenarios),
                    "validation": list(self.validation_scenarios),
                    "seed": self.split_seed,
                    "unit": "escenario entero (arquetipo), no path",
                },
            },
            "validation": {
                "scheme": self.scheme,
                "n_groups": self.n_groups,
                "n_test_groups": self.n_test_groups,
                "n_folds": self.n_folds,
                "purge_days": self.purge_days,
                "periods_per_year": self.periods_per_year,
                "cvar_alpha": self.cvar_alpha,
                "starting_equity": self.starting_equity,
            },
        }


def build_plan(args: argparse.Namespace) -> tuple[StudyPlan, list[StrategySpec]]:
    store = SyntheticStore()
    manifest = store.load_manifest(args.library)
    base_config = load_config(args.config)

    # El universo comun se calcula contra la LIBRERIA, no contra el mercado: por (g) aqui
    # no entra un solo dato real. Da los mismos simbolos que el estudio de transferencia
    # (que ademas los audito contra el exchange), y si algun dia no coincidiera, la
    # reproduccion de la celda `off` lo diria a gritos.
    library_symbols = {str(u["symbol"]) for u in manifest.universe}
    symbols = tuple(s for s in crypto_universe(base_config) if s in library_symbols)
    if not symbols:
        raise ValueError("El universo comun (config ∩ libreria) esta vacio")
    omitted = tuple(sorted(library_symbols - set(symbols)))

    start, end = study_window(manifest, base_config.runner.lookback_days)
    scenario_ids = tuple(s["id"] for s in manifest.scenarios)
    if args.scenarios:
        scenario_ids = scenario_ids[: args.scenarios]
    split = split_scenarios(list(scenario_ids), seed=args.split_seed)

    specs = build_specs(tuple(args.families), args.configs_per_family)
    cells = build_cells(args.rho, args.lead)

    plan = StudyPlan(
        library_id=args.library,
        config_path=str(args.config),
        symbols=symbols,
        library_symbols_omitted=omitted,
        scenario_ids=scenario_ids,
        train_scenarios=split.train,
        validation_scenarios=split.validation,
        split_seed=args.split_seed,
        n_paths=args.paths,
        start=start.isoformat(),
        end=end.isoformat(),
        window_days=(end - start).days,
        cells=tuple(cells),
        families=tuple(sorted({s.type for s in specs})),
        configs_per_family=args.configs_per_family,
        study_seed=STUDY_SEED,
        emission_seed=args.emission_seed,
        scheme=SCHEME_CPCV,
        n_groups=N_GROUPS,
        n_test_groups=N_TEST_GROUPS,
        n_folds=15,
        purge_days=resolve_purge_days(base_config),
        periods_per_year=periods_per_year_for_symbols(symbols),
        cvar_alpha=DEFAULT_CVAR_ALPHA,
        starting_equity=DEFAULT_STARTING_EQUITY,
    )
    return plan, specs


def config_for(plan: StudyPlan) -> AppConfig:
    """El config base con el universo comun. Lo unico que cambia entre celdas es el mundo
    de observacion; riesgo, ejecucion, costes y universo son los del sistema en vivo."""
    base = load_config(plan.config_path)
    return dataclasses.replace(
        base, runner=dataclasses.replace(base.runner, symbols=list(plan.symbols))
    )


# ------------------------------------------------------------------- workers ---------

_WORKER: dict = {}


def _init_worker(payload: dict) -> None:
    logging.getLogger("ai_trader").setLevel(logging.ERROR)
    plan: StudyPlan = payload["plan"]
    _WORKER.update(
        plan=plan,
        config=config_for(plan),
        specs={spec.id: spec for spec in payload["specs"]},
        cells={cell.cell_id: cell for cell in plan.cells},
        store=SyntheticStore(),
        start=datetime.fromisoformat(plan.start),
        end=datetime.fromisoformat(plan.end),
        bars_key=None,
        bars=None,
        panel_key=None,
        panel=None,
    )


def _bars(scenario_id: str, path_index: int) -> dict[str, pd.DataFrame]:
    """Barras de la ultima muestra vista, cacheadas: las tareas llegan agrupadas por
    (celda, muestra), asi que una lectura de parquet sirve para las 16 configuraciones."""
    key = (scenario_id, path_index)
    if _WORKER["bars_key"] != key:
        loaded = _WORKER["store"].load_bars(_WORKER["plan"].library_id, scenario_id, path_index)
        _WORKER["bars"] = {s: df for s, df in loaded.items() if s in _WORKER["plan"].symbols}
        _WORKER["bars_key"] = key
        _WORKER["panel_key"] = None
    return _WORKER["bars"]


def _panel(cell: Cell, scenario_id: str, path_index: int):
    """El canal de esta (celda, muestra), emitido una vez y reutilizado por las 16
    configuraciones. La emision es determinista, asi que cachearla no cambia ningun
    numero: solo evita repetirla dieciseis veces."""
    key = (cell.cell_id, scenario_id, path_index)
    if _WORKER["panel_key"] != key:
        channel = cell.channel
        _WORKER["panel"] = (
            None
            if channel is None
            else emit_signals(
                _bars(scenario_id, path_index),
                [channel],
                seed=_WORKER["plan"].emission_seed,
            )
        )
        _WORKER["panel_key"] = key
    return _WORKER["panel"]


def _run_task(task: tuple[str, str, str, int]) -> dict:
    """Evalua UNA (configuracion, celda, muestra) por CPCV y devuelve sus scores fold a
    fold. Un fallo se marca y no tumba el barrido; la agregacion descarta despues la
    configuracion entera, porque comparar celdas exige que las 16 se hayan medido sobre
    exactamente el mismo material."""
    config_id, cell_id, scenario_id, path_index = task
    cell: Cell = _WORKER["cells"][cell_id]
    spec = cell.spec_for(_WORKER["specs"][config_id])

    row = {
        "config_id": config_id,
        "cell_id": cell_id,
        "scenario_id": scenario_id,
        "path_index": path_index,
        "failed": False,
    }
    try:
        bars = _bars(scenario_id, path_index)
        panel = _panel(cell, scenario_id, path_index)
        result = validate_multiwindow(
            _WORKER["config"], spec, bars, _WORKER["start"], _WORKER["end"],
            scheme=SCHEME_CPCV,
            n_groups=N_GROUPS,
            n_test_groups=N_TEST_GROUPS,
            purge_days=_WORKER["plan"].purge_days,
            starting_equity=_WORKER["plan"].starting_equity,
            cvar_alpha=_WORKER["plan"].cvar_alpha,
            with_baselines=True,
            # El corte unico no interviene y cuesta un backtest entero por unidad.
            compare_single_split=False,
            block_cache={},
            baseline_cache={},
            signal_provider_factory=None if panel is None else panel.provider_factory(),
        )
    except Exception as exc:  # noqa: BLE001 - una unidad mala no debe tumbar el estudio
        logger.warning("Unidad fallida (%s | %s | %s): %s", config_id, cell_id, scenario_id, exc)
        return {**row, "failed": True, "error": str(exc)}

    return {
        **row,
        "scores": [round(s, 6) for s in result.scores],
        "trades_by_fold": list(result.trades),
        "baseline_scores": {
            name: [round(v, 6) for v in values]
            for name, values in result.baseline_scores.items()
        },
        "leakage_ok": result.leakage_ok,
        "embargo_days": result.embargo_days,
    }


def build_tasks(plan: StudyPlan, config_ids: Sequence[str]) -> list[tuple[str, str, str, int]]:
    """Tareas ordenadas por (celda, muestra): el worker reutiliza las barras cargadas y el
    canal emitido para las 16 configuraciones."""
    return [
        (config_id, cell.cell_id, scenario_id, path_index)
        for cell in plan.cells
        for scenario_id in plan.scenario_ids
        for path_index in range(plan.n_paths)
        for config_id in config_ids
    ]


def run_units(
    plan: StudyPlan,
    specs: Sequence[StrategySpec],
    workers: int,
    checkpoint: UnitCheckpoint | None = None,
) -> list[dict]:
    payload = {"plan": plan, "specs": list(specs)}
    tasks = build_tasks(plan, [s.id for s in specs])
    total = len(tasks)

    # Lo ya corrido en un intento anterior. Reanudar es exacto y no una aproximacion: cada
    # unidad es independiente y determinista, y las filas se ordenan al final, asi que el
    # fichero de unidades sale identico se haya corrido de una vez o en cinco tramos.
    done = checkpoint.load() if checkpoint is not None else {}
    pending = [t for t in tasks if tuple(t) not in done]

    logger.info(
        "Evaluando %d configuraciones x %d celdas x %d muestras = %d unidades de %d folds "
        "| workers=%d%s",
        len(specs), len(plan.cells), plan.n_samples, total, plan.n_folds, workers,
        f" | {len(done)} ya corridas, quedan {len(pending)}" if done else "",
    )
    if not pending:
        logger.info("No queda ninguna unidad por correr: todo estaba en el punto de guardado")
        return sorted(
            done.values(),
            key=lambda r: (r["cell_id"], r["scenario_id"], r["path_index"], r["config_id"]),
        )

    started = time.time()
    rows: list[dict] = list(done.values())
    fresh = 0
    chunksize = max(1, len(specs) // 2)
    # `imap_unordered` y no `imap`: las filas se ordenan igualmente al final, y con el
    # ordenado el contador solo avanza cuando termina el PREFIJO completo, asi que con
    # siete workers el progreso que se imprime va muy por detras del real y su ETA es
    # falsa por un factor de dos o tres.
    #
    # `pending` conserva el orden de `build_tasks` —es una subsecuencia filtrada—, asi que la
    # localidad que hace que el worker reutilice las barras y el canal emitido para las 16
    # configuraciones se mantiene al reanudar.
    with mp.Pool(workers, initializer=_init_worker, initargs=(payload,)) as pool:
        for row in pool.imap_unordered(_run_task, pending, chunksize=chunksize):
            rows.append(row)
            fresh += 1
            if checkpoint is not None:
                checkpoint.append(
                    (row["config_id"], row["cell_id"], row["scenario_id"], row["path_index"]),
                    row,
                )
            if fresh % 16 == 0 or fresh == len(pending):
                elapsed = time.time() - started
                logger.info(
                    "  %d/%d (%.0f%%) | %.1f min | ETA %.0f min",
                    len(rows), total, 100.0 * len(rows) / total, elapsed / 60.0,
                    elapsed / fresh * (len(pending) - fresh) / 60.0,
                )
    logger.info("Unidades corridas en %.1f min", (time.time() - started) / 60.0)
    return sorted(
        rows,
        key=lambda r: (r["cell_id"], r["scenario_id"], r["path_index"], r["config_id"]),
    )


def verify_determinism(
    plan: StudyPlan, specs: Sequence[StrategySpec], rows: Sequence[dict], n: int
) -> dict:
    """Re-evalua `n` unidades en procesos NUEVOS y exige scores identicos. Determinismo
    afirmado sin comprobar es determinismo supuesto."""
    usable = [r for r in rows if not r["failed"]]
    if not usable or n <= 0:
        return {"checked": 0, "mismatches": []}
    step = max(1, len(usable) // n)
    chosen = usable[::step][:n]
    tasks = [(r["config_id"], r["cell_id"], r["scenario_id"], r["path_index"]) for r in chosen]

    payload = {"plan": plan, "specs": list(specs)}
    with mp.Pool(
        min(len(tasks), 4), initializer=_init_worker, initargs=(payload,)
    ) as pool:
        replays = pool.map(_run_task, tasks)

    mismatches = [
        f"{r['config_id']}/{r['cell_id']}/{r['scenario_id']}"
        for r, replay in zip(chosen, replays)
        if r.get("scores") != replay.get("scores")
    ]
    logger.info(
        "Determinismo: %d/%d unidades reproducidas score a score%s",
        len(replays) - len(mismatches), len(replays),
        "" if not mismatches else f" | DISCREPANCIAS: {mismatches}",
    )
    return {"checked": len(replays), "mismatches": mismatches}


# ----------------------------------------------------------------- agregacion --------


@dataclass(slots=True)
class CellScores:
    """Los scores de UNA celda, separados por el hold-out de escenarios.

    `by_config[config_id][side]` son los scores de todas las ventanas OOS de ese lado
    puestos EN COMUN. En comun y no promediando por muestra: la recompensa del sistema es
    el CVaR de una distribucion, y tomar el CVaR de cada muestra y luego el de esos CVaR
    compondria dos conservadurismos.
    """

    cell_id: str
    by_config: dict[str, dict[str, list[float]]]
    trades: dict[str, dict[str, list[int]]]
    baselines: dict[str, dict[str, list[float]]]
    dropped: list[str]

    def reward(self, config_id: str, side: str, alpha: float) -> float:
        return aggregate_reward(self.by_config[config_id][side], alpha=alpha).reward

    def activity(self, config_id: str, side: str) -> ActivityStats:
        return measure_activity(self.trades[config_id][side])

    def rankable(self, config_id: str, side: str) -> bool:
        return DEFAULT_ACTIVITY_FLOOR.eligible(self.activity(config_id, side))

    def best_baseline(self, side: str, alpha: float) -> tuple[str | None, float]:
        rewards = {
            name: aggregate_reward(values[side], alpha=alpha).reward
            for name, values in self.baselines.items()
            if values[side]
        }
        if not rewards:
            return None, float("-inf")
        best = min(rewards, key=lambda n: (-rewards[n], n))
        return best, rewards[best]


SIDE_TRAIN = "train"
SIDE_VALIDATION = "validation"


def collect_cell(
    rows: Sequence[dict], cell_id: str, config_ids: Sequence[str], split: ScenarioSplit
) -> CellScores:
    """Ordena las filas crudas de una celda en (configuracion x lado del hold-out)."""
    side_of = {s: SIDE_TRAIN for s in split.train} | {
        s: SIDE_VALIDATION for s in split.validation
    }
    cell_rows = [r for r in rows if r["cell_id"] == cell_id]
    samples = sorted({(r["scenario_id"], r["path_index"]) for r in cell_rows})

    indexed: dict[str, dict[tuple[str, int], dict]] = {}
    for row in cell_rows:
        indexed.setdefault(row["config_id"], {})[(row["scenario_id"], row["path_index"])] = row

    by_config: dict[str, dict[str, list[float]]] = {}
    trades: dict[str, dict[str, list[int]]] = {}
    dropped: list[str] = []
    for config_id in config_ids:
        cells = indexed.get(config_id, {})
        if len(cells) != len(samples) or any(cells[k]["failed"] for k in samples):
            dropped.append(config_id)
            continue
        scores = {SIDE_TRAIN: [], SIDE_VALIDATION: []}
        counts = {SIDE_TRAIN: [], SIDE_VALIDATION: []}
        for key in samples:
            side = side_of[key[0]]
            scores[side].extend(cells[key]["scores"])
            counts[side].extend(cells[key].get("trades_by_fold", []))
        by_config[config_id] = scores
        trades[config_id] = counts

    # Los baselines no dependen de la configuracion (son carteras pasivas sobre las mismas
    # ventanas): basta una copia por muestra.
    baselines: dict[str, dict[str, list[float]]] = {}
    for name in sorted(BASELINE_LABELS):
        series = {SIDE_TRAIN: [], SIDE_VALIDATION: []}
        complete = True
        for key in samples:
            found = next(
                (
                    r["baseline_scores"][name]
                    for r in cell_rows
                    if not r["failed"]
                    and (r["scenario_id"], r["path_index"]) == key
                    and name in r.get("baseline_scores", {})
                ),
                None,
            )
            if found is None:
                complete = False
                break
            series[side_of[key[0]]].extend(found)
        if complete:
            baselines[name] = series

    if dropped:
        logger.warning(
            "Celda %s: %d/%d configuraciones descartadas por unidades fallidas: %s",
            cell_id, len(dropped), len(config_ids), ", ".join(dropped),
        )
    return CellScores(
        cell_id=cell_id,
        by_config=by_config,
        trades=trades,
        baselines=baselines,
        dropped=dropped,
    )


def cell_verdict(cell: Cell, scores: CellScores, config_ids: Sequence[str], alpha: float) -> dict:
    """
    La lectura de UNA celda, siguiendo `CRITERION` al pie de la letra.

    Se publican tambien las cifras que permiten desmontarla: cuantas configuraciones baten
    al baseline (no solo la elegida), y la actividad de la elegida, porque una recompensa
    alta con la curva plana no es una recompensa.
    """
    usable = [c for c in config_ids if c in scores.by_config]
    baseline_name, baseline_val = scores.best_baseline(SIDE_VALIDATION, alpha)
    _, baseline_train = scores.best_baseline(SIDE_TRAIN, alpha)

    ranked = [c for c in usable if scores.rankable(c, SIDE_TRAIN)]
    # Si la puerta deja a TODAS por debajo del suelo de actividad, se elige entre las que
    # hay y se DECLARA: publicar una celda sin elegida escondería el hecho, y elegir en
    # silencio entre inelegibles lo escondería aun mejor.
    pool = ranked or usable
    selected = (
        min(pool, key=lambda c: (-scores.reward(c, SIDE_TRAIN, alpha), c)) if pool else None
    )

    beating = [
        c for c in usable if scores.reward(c, SIDE_VALIDATION, alpha) > baseline_val
    ]
    beating_ranked = [c for c in beating if c in ranked]

    reward_val = None if selected is None else scores.reward(selected, SIDE_VALIDATION, alpha)
    reward_train = None if selected is None else scores.reward(selected, SIDE_TRAIN, alpha)
    activity = None if selected is None else scores.activity(selected, SIDE_VALIDATION)
    return {
        **cell.as_dict(),
        "n_configs": len(usable),
        "n_rankable": len(ranked),
        "selection_fell_back": bool(usable and not ranked),
        "dropped": list(scores.dropped),
        "selected": selected,
        "selected_reward_train": _round(reward_train),
        "selected_reward_validation": _round(reward_val),
        "selected_rankable_validation": (
            None if selected is None else scores.rankable(selected, SIDE_VALIDATION)
        ),
        "selected_activity_validation": None if activity is None else activity.as_dict(),
        "baseline_name": baseline_name,
        "baseline_reward_validation": _round(baseline_val),
        "baseline_reward_train": _round(baseline_train),
        "margin": _round(None if reward_val is None else reward_val - baseline_val),
        "beats": bool(reward_val is not None and reward_val > baseline_val),
        "n_beating_baseline": len(beating),
        "n_beating_baseline_rankable": len(beating_ranked),
        "rewards_validation": {
            c: _round(scores.reward(c, SIDE_VALIDATION, alpha)) for c in usable
        },
        "rewards_train": {c: _round(scores.reward(c, SIDE_TRAIN, alpha)) for c in usable},
        "trades_per_window": {
            c: round(scores.activity(c, SIDE_VALIDATION).trades_per_window, 2) for c in usable
        },
    }


def break_even(verdicts: Sequence[dict]) -> dict:
    """
    El resultado del estudio, leido con `CRITERION` y no con lo que se vea.

    Un barrido cuyo control ensucia no tiene break-even que publicar, y decirlo es el
    resultado: significa que la ganancia no venia de la informacion.
    """
    on = [v for v in verdicts if v["arm"] != ARM_OFF]
    control = next((v for v in on if v["rho"] == 0.0), None)
    control_clean = bool(control is not None and not control["beats"])

    by_lead: dict[int, dict] = {}
    for lead in sorted({v["lead_days"] for v in on}):
        cells = sorted((v for v in on if v["lead_days"] == lead), key=lambda v: v["rho"])
        winners = [v for v in cells if v["beats"] and v["rho"] > 0.0]
        rho_max = max((v["rho"] for v in cells), default=None)
        by_lead[lead] = {
            "lead_days": lead,
            "rho_grid": [v["rho"] for v in cells],
            "beats": [v["beats"] for v in cells],
            "break_even_rho": winners[0]["rho"] if winners else None,
            "break_even_above": None if winners else rho_max,
            "margins": {str(v["rho"]): v["margin"] for v in cells},
        }

    return {
        "control_clean": control_clean,
        "control": None if control is None else {
            "rho": control["rho"],
            "beats": control["beats"],
            "margin": control["margin"],
            "selected": control["selected"],
        },
        "verdict": (
            "anulado_por_el_control" if not control_clean
            else "break_even_encontrado"
            if any(v["break_even_rho"] is not None for v in by_lead.values())
            else "no_alcanzado_en_la_rejilla"
        ),
        "by_lead": [by_lead[lead] for lead in sorted(by_lead)],
    }


def value_of_information(verdicts: Sequence[dict]) -> list[dict]:
    """Delta de recompensa contra rho=0 con la MISMA puerta, que es la unica comparacion
    en la que lo unico que cambia es la informacion."""
    out: list[dict] = []
    for lead in sorted({v["lead_days"] for v in verdicts if v["arm"] != ARM_OFF}):
        cells = [v for v in verdicts if v["lead_days"] == lead]
        zero = next((v for v in cells if v["rho"] == 0.0), None)
        if zero is None or zero["selected_reward_validation"] is None:
            continue
        for cell in sorted(cells, key=lambda v: v["rho"]):
            if cell["selected_reward_validation"] is None:
                continue
            out.append({
                "cell_id": cell["cell_id"],
                "rho": cell["rho"],
                "lead_days": lead,
                "reward": cell["selected_reward_validation"],
                "delta_vs_rho0": _round(
                    cell["selected_reward_validation"] - zero["selected_reward_validation"]
                ),
            })
    return out


def gate_cost(verdicts: Sequence[dict], config_ids: Sequence[str] = ()) -> dict:
    """
    Cuanto cuesta la PUERTA por si misma: 'off' contra rho=0.

    Es la cifra que impide leer como informacion lo que solo es operar menos, y se publica
    EN LOS DOS LADOS del hold-out porque su signo DEPENDE DEL REGIMEN: filtrar entradas al
    azar reduce exposicion, asi que ayuda en los escenarios que caen y estorba en los que
    suben. Publicar solo el lado de validacion invitaria a leer "la puerta cuesta X" como
    una constante del sistema cuando es una propiedad del tramo.

    Lo que SI es independiente del regimen —y es lo que sostiene el break-even— es la
    monotonia en rho, que se lee en `value_of_information` y, sin el ruido de que la
    configuracion elegida cambie entre celdas, en `by_config`: la misma configuracion en
    las cinco celdas, con sus operaciones por ventana al lado. Si esas operaciones apenas
    se mueven entre celdas, la diferencia de recompensa no puede ser "opera menos".
    """
    off = next((v for v in verdicts if v["arm"] == ARM_OFF), None)
    zero = next((v for v in verdicts if v["arm"] != ARM_OFF and v["rho"] == 0.0), None)
    if off is None or zero is None:
        return {}

    def _delta(side: str) -> float | None:
        a, b = off[f"selected_reward_{side}"], zero[f"selected_reward_{side}"]
        return None if a is None or b is None else _round(b - a)

    return {
        "off_selected": off["selected"],
        "off_reward_validation": off["selected_reward_validation"],
        "off_reward_train": off["selected_reward_train"],
        "rho0_selected": zero["selected"],
        "rho0_reward_validation": zero["selected_reward_validation"],
        "rho0_reward_train": zero["selected_reward_train"],
        "delta": _delta("validation"),
        "delta_validation": _delta("validation"),
        "delta_train": _delta("train"),
        "regime_dependent": (
            "el signo depende del tramo: filtrar entradas reduce exposicion, asi que la "
            "puerta ciega ayuda donde el mercado cae y estorba donde sube"
        ),
        "off_trades_per_window": off["trades_per_window"],
        "rho0_trades_per_window": zero["trades_per_window"],
        # La MISMA configuracion a traves de las celdas: la lectura sin el ruido de que la
        # elegida cambie. Es la que separa "opera menos" de "opera mejor".
        "by_config": {
            config_id: {
                "rewards": {
                    v["cell_id"]: v["rewards_validation"].get(config_id) for v in verdicts
                },
                "trades_per_window": {
                    v["cell_id"]: v["trades_per_window"].get(config_id) for v in verdicts
                },
            }
            for config_id in config_ids
        },
    }


# --------------------------------------------------- certificacion del canal ---------


def certify_channels(plan: StudyPlan, store: SyntheticStore | None = None) -> list[dict]:
    """
    ¿Entrega cada celda el canal que declara? Se mide, no se supone.

    Es la extension del loop de calibracion legitimo a las senales: IC empirico,
    autocorrelacion y perfil lead-lag son propiedades DEL MUNDO que ninguna estrategia
    optimiza, asi que medirlas no es cerrar el loop sobre el rendimiento. Solo cuesta la
    emision (no hay backtest), asi que se hace sobre todas las muestras del plan.

    `past_leak` es el test de falsacion del propio canal: la senal no puede correlacionar
    con retornos YA REALIZADOS. Si lo hiciera, la puerta estaria comprando informacion que
    la estrategia ya tiene en las barras y el break-even seria optimista.
    """
    store = store or SyntheticStore()
    channels = [(cell, cell.channel) for cell in plan.cells if cell.channel is not None]
    measured: dict[str, dict[str, list]] = {
        cell.cell_id: {"ic": [], "ac1": [], "leak": [], "profile": []} for cell, _ in channels
    }

    # El bucle exterior es la MUESTRA y no la celda: las barras son las mismas para todas
    # (la senal se deriva de ellas, no al reves), asi que leer el parquet una vez por celda
    # seria leer cuatro veces lo mismo.
    for scenario_id in plan.scenario_ids:
        for path_index in range(plan.n_paths):
            bars = {
                s: df
                for s, df in store.load_bars(plan.library_id, scenario_id, path_index).items()
                if s in plan.symbols
            }
            closes = {
                s: bar_schema.series(df, bar_schema.CLOSE).to_numpy(dtype=float)
                for s, df in bars.items()
            }
            for cell, channel in channels:
                panel = emit_signals(bars, [channel], seed=plan.emission_seed)
                bucket = measured[cell.cell_id]
                for symbol, series in closes.items():
                    values = panel.values.get((channel.name, symbol))
                    if values is None:
                        continue
                    facts = channel_facts(
                        values, series, lead_days=channel.lead_days, max_lag=CHANNEL_MAX_LAG
                    )
                    if facts is None:
                        continue
                    bucket["ic"].append(facts.ic)
                    bucket["ac1"].append(facts.ac1)
                    bucket["leak"].append(facts.past_leak)
                    bucket["profile"].append(dict(facts.lead_lag))

    out: list[dict] = []
    for cell, channel in channels:
        bucket = measured[cell.cell_id]
        ics, ac1s, leaks, profiles = (
            bucket["ic"], bucket["ac1"], bucket["leak"], bucket["profile"]
        )
        out.append({
            "cell_id": cell.cell_id,
            "declared_ic": round(channel.expected_ic, 4),
            "declared_noise_ar": channel.noise_ar,
            "n_series": len(ics),
            "ic_median": _round(float(np.median(ics)) if ics else None),
            "ic_p10": _round(float(np.percentile(ics, 10)) if ics else None),
            "ic_p90": _round(float(np.percentile(ics, 90)) if ics else None),
            "ac1_median": _round(float(np.median(ac1s)) if ac1s else None),
            "past_leak_median": _round(float(np.median(leaks)) if leaks else None),
            "lead_lag_median": {
                str(lag): _round(float(np.median([p[lag] for p in profiles])))
                for lag in range(-CHANNEL_MAX_LAG, CHANNEL_MAX_LAG + 1)
            } if profiles else {},
        })
    return out


# ------------------------------------------------------- reproduccion del control ----


def reproduction_check(
    plan: StudyPlan, rows: Sequence[dict], units_path: Path | str = TRANSFER_UNITS
) -> dict:
    """
    ¿Reproduce la celda `off` los scores publicados del estudio de transferencia?

    Es el control de que la costura del canal —la factoria de proveedores en el motor, la
    tabla de polaridad inyectable, el catalogo de fuentes simuladas— no ha movido nada del
    sistema. Si esto sale rojo, ninguna otra cifra del informe es creible.

    Sin el fichero de unidades no se falla ni se aprueba: se declara `available=False`.

    Y hay una condicion que parece burocracia y es justo lo contrario: la comparacion solo
    vale si la rejilla es LITERALMENTE la publicada. `candidate_specs` muestrea un hipercubo
    latino de n puntos, y el hipercubo de 2 no contiene al de 8: con `--configs-per-family`
    distinto, `crypto_momentum#00` es OTRA configuracion con el mismo nombre. Comparar sus
    scores daria discrepancias reales que no significan nada, que es la peor clase de
    alarma. Por eso el estudio corta aqui en vez de fiarse del identificador.
    """
    if plan.configs_per_family != CONFIGS_PER_FAMILY:
        return {
            "available": False,
            "reason": (
                f"la rejilla de esta corrida ({plan.configs_per_family} por familia) no es "
                f"la publicada ({CONFIGS_PER_FAMILY}): el hipercubo latino de n puntos no "
                "contiene al de m, asi que los config_id no designan las mismas "
                "configuraciones y compararlos no mediria nada"
            ),
        }
    path = Path(units_path)
    if not path.exists():
        return {"available": False, "reason": f"no esta {path}"}

    published = {
        (r["config_id"], r["unit_id"], r["path_index"]): r["scores"]
        for r in json.loads(path.read_text(encoding="utf-8"))["rows"]
        if r.get("side") == "synthetic" and not r.get("failed") and "scores" in r
    }
    compared = 0
    mismatches: list[dict] = []
    for row in rows:
        if row["cell_id"] != ARM_OFF or row["failed"]:
            continue
        key = (row["config_id"], row["scenario_id"], row["path_index"])
        expected = published.get(key)
        if expected is None:
            continue
        compared += 1
        if expected != row["scores"]:
            mismatches.append({
                "config_id": row["config_id"],
                "scenario_id": row["scenario_id"],
                "max_abs_diff": max(
                    abs(a - b) for a, b in zip(expected, row["scores"])
                ) if len(expected) == len(row["scores"]) else None,
            })
    return {
        "available": True,
        "source": str(path),
        "compared": compared,
        "identical": compared > 0 and not mismatches,
        "mismatches": mismatches[:10],
        "n_mismatches": len(mismatches),
    }


def baseline_invariance(cells: dict[str, CellScores]) -> dict:
    """Los baselines son carteras pasivas: no consultan senales, asi que tienen que salir
    IDENTICOS en todas las celdas. Si no lo son, el canal se ha filtrado al liston y las
    comparaciones no valen."""
    reference = cells.get(ARM_OFF) or next(iter(cells.values()), None)
    if reference is None:
        return {}
    differing: list[str] = []
    for cell_id, scores in cells.items():
        for name, series in reference.baselines.items():
            other = scores.baselines.get(name)
            if other is None or other != series:
                differing.append(f"{cell_id}/{name}")
    return {
        "reference_cell": reference.cell_id,
        "identical": not differing,
        "differing": differing[:10],
    }


# ------------------------------------------------------------------- informe ---------


def _round(value, decimals: int = 4):
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, decimals)


def build_report(
    plan: StudyPlan,
    specs: Sequence[StrategySpec],
    rows: Sequence[dict],
    units_path: Path | str | None = None,
) -> dict:
    config_ids = [s.id for s in specs]
    split = ScenarioSplit(
        train=plan.train_scenarios, validation=plan.validation_scenarios, seed=plan.split_seed
    )
    cells = {
        cell.cell_id: collect_cell(rows, cell.cell_id, config_ids, split) for cell in plan.cells
    }
    verdicts = [
        cell_verdict(cell, cells[cell.cell_id], config_ids, plan.cvar_alpha)
        for cell in plan.cells
    ]
    return {
        "plan": plan.as_dict(),
        "criterion": dict(CRITERION),
        "configs": config_ids,
        "cells": verdicts,
        "break_even": break_even(verdicts),
        "value_of_information": value_of_information(verdicts),
        "gate_cost": gate_cost(verdicts, config_ids),
        "channel_certification": certify_channels(plan),
        "reproduction": reproduction_check(
            plan, rows, units_path or transfer_units_path(plan.library_id)
        ),
        "baseline_invariance": baseline_invariance(cells),
        "n_failed_units": sum(1 for r in rows if r["failed"]),
    }


def _print_report(report: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print(f"\n{'celda':<14}{'IC decl.':>9}{'IC medido':>11}{'elegida':>22}"
          f"{'recompensa':>12}{'baseline':>10}{'margen':>9}{'bate':>7}")
    certification = {c["cell_id"]: c for c in report["channel_certification"]}
    for cell in report["cells"]:
        cert = certification.get(cell["cell_id"], {})
        declared = _fmt(cell["expected_ic"], 3)
        measured = _fmt(cert.get("ic_median"), 3)
        selected = str(cell["selected"])
        print(
            f"{cell['cell_id']:<14}{declared:>9}{measured:>11}{selected:>22}"
            f"{_fmt(cell['selected_reward_validation']):>12}"
            f"{_fmt(cell['baseline_reward_validation']):>10}"
            f"{_fmt(cell['margin']):>9}"
            f"{'SI' if cell['beats'] else 'no':>7}"
        )

    be = report["break_even"]
    print(f"\nControl rho=0: {'LIMPIO' if be['control_clean'] else 'ENSUCIADO'}")
    for lead in be["by_lead"]:
        if lead["break_even_rho"] is not None:
            print(f"  h={lead['lead_days']}: break-even rho = {lead['break_even_rho']:.2f}")
        else:
            print(f"  h={lead['lead_days']}: break-even rho > {lead['break_even_above']:.2f} "
                  "(no se alcanza en la rejilla)")
    print("VEREDICTO:", be["verdict"])

    rep = report["reproduction"]
    if rep.get("available"):
        outcome = "IDENTICAS" if rep["identical"] else f"{rep['n_mismatches']} DISCREPANCIAS"
        print(
            f"\nReproduccion de la celda 'off' contra {rep['source']}: "
            f"{rep['compared']} unidades, {outcome}"
        )
    inv = report.get("baseline_invariance") or {}
    if inv:
        print(f"Baselines identicos entre celdas: {'si' if inv['identical'] else 'NO'}")


def _fmt(value, decimals: int = 4) -> str:
    return "—" if value is None else f"{value:+.{decimals}f}"


def report_path(library_id: str, out_dir: Path | str = OUT_DIR) -> Path:
    return Path(out_dir) / f"report_{library_id}.json"


def units_path(library_id: str, out_dir: Path | str = OUT_DIR) -> Path:
    return Path(out_dir) / f"units_{library_id}.json"


def load_signal_report(path: Path | str) -> dict | None:
    """Lee el informe publicado. None si no esta, para que los generadores de dashboard y
    documentacion degraden a prosa sin cifras en vez de romperse."""
    report = Path(path)
    if not report.exists():
        return None
    return json.loads(report.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=DEFAULT_LIBRARY_ID)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scenarios", type=int, default=8, help="0 = todos")
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--configs-per-family", type=int, default=CONFIGS_PER_FAMILY)
    parser.add_argument(
        "--families", nargs="+", default=list(FAMILIES),
        help="Familias de la rejilla, EN ORDEN (el orden fija las semillas del hipercubo).",
    )
    parser.add_argument(
        "--transfer-units", default=None,
        help=(
            "Unidades contra las que reproduce la celda de control. Por defecto, las de la "
            "misma libreria (data/transfer/units_<library>.json)."
        ),
    )
    parser.add_argument(
        "--overwrite-published", action="store_true",
        help="Permite reemplazar un informe publicado con OTRA rejilla. Ver shared/reports.",
    )
    parser.add_argument(
        "--rho", type=float, nargs="+", default=list(DEFAULT_RHO_GRID),
        help="Rejilla de capacidad predictiva. Tiene que incluir 0: es el control.",
    )
    parser.add_argument("--lead", type=int, nargs="+", default=list(DEFAULT_LEAD_GRID))
    parser.add_argument("--split-seed", type=int, default=STUDY_SEED)
    parser.add_argument("--emission-seed", type=int, default=DEFAULT_EMISSION_SEED)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Re-analiza las unidades ya guardadas, sin correr backtests.",
    )
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignora el punto de guardado y corre las unidades desde cero.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("ai_trader").setLevel(logging.ERROR)

    if 0.0 not in [float(r) for r in args.rho]:
        raise SystemExit(
            "La rejilla de rho TIENE que incluir 0: es el grupo de control y sin el no hay "
            "test de falsacion que publicar."
        )

    plan, specs = build_plan(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units_file = units_path(args.library, out_dir)
    target = report_path(args.library, out_dir)
    if not args.analyze_only:
        # Antes de gastar cinco horas: no se sustituye un informe que publica otra rejilla.
        for path in (units_file, target):
            guard_published_grid(path, plan.families, overwrite=args.overwrite_published)

    if args.analyze_only:
        if not units_file.exists():
            raise SystemExit(f"No hay unidades guardadas en {units_file}")
        stored = json.loads(units_file.read_text(encoding="utf-8"))
        rows = stored["rows"]
        logger.info("Re-analizando %d unidades de %s", len(rows), units_file)
    else:
        # El punto de guardado va al lado de las unidades y lleva la huella del plan y de las
        # configuraciones: reanudar sobre otro plan mezclaria dos estudios en un informe que
        # dice ser uno, asi que un fichero con otra huella se aparta en vez de reutilizarse.
        checkpoint = None
        if not args.no_resume:
            checkpoint = UnitCheckpoint(
                units_file.with_name(f"progress_{args.library}.jsonl"),
                fingerprint(plan.as_dict(), [s.id for s in specs],
                            [dataclasses.asdict(s) for s in specs]),
            )
        rows = run_units(plan, specs, args.workers, checkpoint)
        units_file.write_text(
            json.dumps({"plan": plan.as_dict(), "rows": rows}, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Unidades -> %s", units_file)
        # A partir de aqui el fichero bueno es el de unidades: dejar el punto de guardado solo
        # invita a reanudar un barrido que ya termino.
        if checkpoint is not None:
            checkpoint.discard()

    report = build_report(plan, specs, rows, args.transfer_units)
    if args.verify_determinism and not args.analyze_only:
        report["determinism"] = verify_determinism(plan, specs, rows, DETERMINISM_CHECKS)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    target = report_path(args.library, out_dir)
    target.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Informe -> %s", target)
    _print_report(report)

    # El codigo de salida NO depende de que se encuentre break-even: "no se alcanza en la
    # rejilla" es un resultado, y de los mas informativos. Lo que SI falla es que el
    # estudio no se pueda leer: control ensuciado, o la celda `off` sin reproducir.
    rep = report["reproduction"]
    if rep.get("available") and not rep["identical"]:
        logger.error("La celda 'off' NO reproduce las unidades publicadas")
        return 1
    if not report["break_even"]["control_clean"]:
        logger.error("Control rho=0 ensuciado: el barrido no publica break-even")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
