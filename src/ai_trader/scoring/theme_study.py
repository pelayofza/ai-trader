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

EL COSTE, MEDIDO, Y POR QUE ES TAN ALTO
---------------------------------------
Medido unidad a unidad sobre la ventana mas poblada (w5, 24 simbolos): **104 s el brazo ciego
y 817 s el armado**. Ponderando las cinco ventanas por su universo real (8, 17, 18, 23 y 24
simbolos) salen ~15 h de CPU para las 160 unidades de la configuracion por defecto (4 familias
x 4 configuraciones x 5 ventanas x 2 brazos), o alrededor de 3,5 h de reloj con siete workers
sobre cuatro nucleos fisicos. Esta escrito aqui porque lo estime tres veces por lo bajo antes
de medirlo: **no lances este estudio esperando una hora**.

Una medicion anterior daba ~610 s por unidad de media, el doble. Estaba contaminada: aquella
corrida tenia los workers pidiendo barras al exchange —`--offline` no llegaba hasta ellos— y lo
que media era la red, no el estudio.

De donde sale el coste que queda, que no es donde parece. Frente a las ~88 s/unidad del estudio
de transferencia hay dos factores y el segundo es el gordo:

1. El universo es de 24 simbolos y no de 11 (ver `_real_bounds`): ~2,2x.
2. **El brazo ARMADO reconstruye el radar tematico una vez por fold**, y cada construccion
   normaliza las 21 fuentes del archivo entero (`normalize_features` sobre cada frame). Con 15
   folds de CPCV eso son quince normalizaciones completas por unidad. El estudio de
   transferencia no lo paga porque corre con el radar VACIO. Es exactamente el factor 7,9x que
   se mide entre los dos brazos de la misma unidad (817 s frente a 104 s).

La optimizacion evidente —cachear el radar por (ventana, brazo) en vez de reconstruirlo por
fold— no esta hecha y es el primer sitio donde mirar si este estudio hay que repetirlo a
menudo. Mientras tanto, para una pasada rapida: `--configs-per-family 2` lo deja en la mitad,
con el intervalo por bloques mas ancho.
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
import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.config import DEFAULT_CONFIG_PATH, StrategySpec, load_config
from ai_trader.observation.signal_radar import MIN_SIGNAL_COVERAGE
from ai_trader.observation.signal_themes import (
    THEMES,
    ThemedSignalRadarProvider,
    effective_denominator,
    theme_features,
)
from ai_trader.shared.clock import HistoricalClock
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA
from ai_trader.scoring.multiwindow import SCHEME_CPCV, validate_multiwindow
from ai_trader.scoring.signal_gate import GATE_VALUE_BY_PARAM, gate_param_for
from ai_trader.scoring.real_substrate import (
    N_GROUPS,
    N_TEST_GROUPS,
    RealWindow,
    audit_real_symbols,
    crypto_universe,
    real_windows,
)
from ai_trader.scoring.families import NEW_FAMILIES, build_specs
from ai_trader.shared.reports import write_report
from ai_trader.signals.catalog import CATALOG
# El cargador de barras reales lo comparten todos los estudios que tocan mercado: los
# simbolos que el exchange no sirve se OMITEN y se declaran, no se rellenan.
from ai_trader.data.real_history import (
    DEFAULT_EXCHANGE,
    DEFAULT_REAL_END,
    DEFAULT_REAL_START,
    build_service,
    fetch_real_bars,
)
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
    Lo que el CATALOGO DECLARA sobre que temas se pueden evaluar hacia atras.

    Se calcula y no se escribe a mano para que el dia que una fuente gane profundidad medida,
    este estudio deje de excluir a su familia sin que nadie tenga que acordarse.

    OJO: esto es la declaracion, no la medicion, y las dos NO coinciden. Quien decide que
    familias entran en el estudio es `measured_themes`, que sondea el radar de verdad. Esta
    funcion se conserva porque el desacuerdo entre ambas es un dato que el informe publica.
    """
    backtestable = {source.key for source in CATALOG if source.backtestable}
    evaluable, blind = set(), set()
    for name, spec in THEMES.items():
        denominator = effective_denominator(len(spec.sources), spec.min_sources)
        covered = len(set(spec.sources) & backtestable) / denominator
        (evaluable if covered >= MIN_SIGNAL_COVERAGE else blind).add(name)
    return frozenset(evaluable), frozenset(blind)


# Sondas del radar para medir la cobertura real a lo largo del rango del estudio. Ocho porque
# la construccion del radar es lo caro y las consultas no: se construye UNA vez y se mueve el
# reloj, que es exactamente lo que hace `HistoricalClock.set`.
THEME_PROBES = 8


def measured_themes(
    frames,
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    probes: int = THEME_PROBES,
) -> tuple[frozenset[str], frozenset[str], dict]:
    """
    Que temas alcanzan cobertura DE VERDAD en el archivo. MEDIDO sondeando el radar.

    Existe porque la derivacion del catalogo (`evaluable_themes`) se equivoca en los DOS
    sentidos, y las dos veces con consecuencias caras:

    - Cuenta fuentes que no cubren al simbolo que se opera. `cex_listings` es backtestable y
      esta en el tema `attention`, pero es un calendario de LISTADOS y BTC no se lista: no
      produce lectura. Medido, `attention` cubre 0,143 (una fuente de siete declaradas, solo
      `wikipedia_pageviews`) en todos los anyos del historico, por debajo del 0,25 de la
      puerta. La familia entro en el estudio y no pudo mover nada: cuarenta unidades tiradas.
    - Descarta fuentes con archivo profundo que el catalogo aun no ha certificado.
      `deribit_volatility` publica desde 2021-03-24, pero su profundidad medida todavia no
      llega a los 365 dias que exige `depth.MIN_MEASURED_DAYS`, asi que figura como no
      backtestable. `vol_surface` se excluyo diciendo que "sus fuentes empezaron a existir el
      dia que arranco la captura", y eso es FALSO.

    El criterio de exclusion es el minimo defendible: se descarta un tema solo si NUNCA
    alcanza el umbral de cobertura en ninguna sonda, porque entonces la puerta no puede atar
    y medirlo es gastar CPU en un cero conocido. Todo lo demas entra y se mide, aunque su
    cuota de sondas legibles sea baja: para eso esta `MIN_PAIRED_WINDOWS`, que convierte la
    falta de muestra en `sin_potencia` en vez de en una diferencia inventada.
    """
    clock = HistoricalClock(start)
    provider = ThemedSignalRadarProvider(frames, clock)
    step = (end - start) / max(1, probes - 1)

    seen = {name: 0 for name in THEMES}
    best = {name: 0.0 for name in THEMES}
    total = 0
    for i in range(probes):
        clock.set(start + step * i)
        for symbol in symbols:
            feats = provider.features(symbol)
            total += 1
            for name in THEMES:
                cov = feats.get(theme_features(name)[2], 0.0)
                best[name] = max(best[name], cov)
                if cov >= MIN_SIGNAL_COVERAGE:
                    seen[name] += 1

    evaluable = frozenset(name for name in THEMES if seen[name] > 0)
    blind = frozenset(THEMES) - evaluable
    detail = {
        name: {
            "max_coverage": round(best[name], 4),
            "readable_share": round(seen[name] / total, 4) if total else 0.0,
            "threshold": MIN_SIGNAL_COVERAGE,
        }
        for name in sorted(THEMES)
    }
    return evaluable, blind, detail


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
    themes_declared_evaluable: tuple[str, ...]
    themes_measured: dict
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
                # Lo que el catalogo DECLARA, al lado de lo medido. Los dos, y no solo el que
                # manda, porque el desacuerdo entre ambos es el dato: el catalogo cuenta
                # fuentes backtestables y el radar lee las que de verdad cubren al simbolo.
                "declared_evaluable": list(self.themes_declared_evaluable),
                "measured": dict(self.themes_measured),
                "disagreement": sorted(
                    set(self.themes_declared_evaluable) ^ set(self.themes_evaluable)
                ),
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


def _init_worker(
    config_path: str,
    raw_root: str | None,
    symbols: Sequence[str],
    offline: bool,
    exchange: str,
) -> None:
    logging.getLogger("ai_trader").setLevel(logging.ERROR)
    config = load_config(config_path)
    _WORKER.update(
        config=config,
        # El universo del PLAN, no `config.runner.symbols`: ese trae ademas la renta variable,
        # que va por otro proveedor y otra sesion de mercado, y pedirsela al servicio de cripto
        # ensuciaria cada unidad con excepciones que no significan nada.
        symbols=tuple(symbols),
        # `--offline` TIENE que llegar hasta aqui. Sin esto cada worker construia su propio
        # servicio contra el exchange: siete procesos pidiendo veinticuatro simbolos a la vez,
        # el exchange contestando vacio a la mitad, y los siete peleandose por renombrar el
        # mismo `.parquet.tmp`. Lo caro no era la red: era que un simbolo caido se DESCARTA,
        # asi que el universo de cada unidad acababa dependiendo de como cayera la carrera.
        offline=bool(offline),
        exchange=exchange,
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


def _window_bars(
    config, label: str, start: datetime, end: datetime, symbols: Sequence[str]
) -> dict:
    """
    Las barras de UNA ventana, cacheadas por worker.

    Sin esto cada unidad releia el historico de los simbolos del universo, y como las tareas
    llegan agrupadas por ventana eso multiplicaba por treinta y dos el trabajo de lectura: la
    unidad pasaba de segundos a MINUTOS. Es el mismo patron que `transfer_study._real_bars`,
    y su ausencia aqui fue un descuido con un coste medible, no una diferencia de diseno.

    Se cachea UNA ventana y no todas: son 544 dias por veinticuatro simbolos, y siete workers
    reteniendo las cinco a la vez es memoria que no hace falta.

    El universo llega DECLARADO desde el plan y no se deduce de lo que conteste el proveedor.
    Es lo que hace que el brazo ciego y el armado se comparen sobre lo mismo: se calcula una
    vez en el padre, con el mismo criterio de historico que la transferencia, y aqui solo se
    comprueba. Si falta un simbolo declarado la unidad revienta en vez de correr con menos, que
    es como se cuela una diferencia pareada entre dos universos distintos.
    """
    cached = _WORKER.get("bars_label")
    if cached == label:
        return _WORKER["bars"]

    service = build_service(_WORKER["exchange"], offline=_WORKER["offline"])
    bars = fetch_real_bars(tuple(symbols), start, end, service)
    bars = {s: b for s, b in bars.items() if b is not None and not b.empty}
    missing = sorted(set(symbols) - set(bars))
    if missing:
        raise RuntimeError(
            f"La ventana {label} no sirve barras para {missing}, y el plan las declara. "
            "Correr con menos simbolos de los declarados haria que los dos brazos dejaran "
            "de compararse sobre el mismo universo."
        )
    _WORKER["bars_label"] = label
    _WORKER["bars"] = bars
    return bars


def _run_unit(task: tuple) -> dict:
    spec_dict, window_dict, arm = task
    spec = StrategySpec(**spec_dict)
    config = _WORKER["config"]
    frames = _WORKER["frames"]
    start = datetime.fromisoformat(window_dict["start"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(window_dict["end"]).replace(tzinfo=timezone.utc)

    bars = _window_bars(config, window_dict["label"], start, end, window_dict["symbols"])

    try:
        validation = validate_multiwindow(
            config,
            # `spec` es el SEGUNDO posicional, no una kwarg detras de las fechas.
            _armed_spec(spec) if arm == ARM_ARMED else spec,
            bars,
            start,
            end,
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
            "Las familias cuyo tema NUNCA alcanza cobertura no se evaluan, y se declaran en "
            "`plan.families_skipped` con su cobertura maxima medida en vez de desaparecer de "
            "la tabla. El criterio es la cobertura MEDIDA sobre el archivo, no el flag "
            "`backtestable` del catalogo: `plan.themes.disagreement` lista donde difieren.",
            f"Por debajo de {MIN_PAIRED_WINDOWS} ventanas en las que la capa cambie algo, el "
            "veredicto es 'sin_potencia': no se publica una diferencia que el ruido explica.",
        ],
    }


def build_plan(args: argparse.Namespace) -> tuple[StudyPlan, list[StrategySpec], list[RealWindow]]:
    config = load_config(args.config)

    # El mismo minimo que la transferencia: calentamiento de la estrategia + un grupo de CPCV.
    min_history = config.runner.lookback_days + args.window_days // N_GROUPS
    symbols, first, last, bars = _real_bounds(
        config, args.start, args.end, min_history,
        offline=bool(args.offline), exchange=args.exchange,
    )
    windows = _windows_with_universe(real_windows(first, last, args.window_days), bars, symbols,
                                     min_history_days=min_history)
    frames = load_frames(raw_root=config.signals.raw_root or None)

    # La cobertura se MIDE sobre el archivo y el universo reales, no se deduce del catalogo.
    declared, _ = evaluable_themes()
    evaluable, blind, measured = measured_themes(frames, symbols, start=first, end=last)
    for name in sorted(set(declared) ^ set(evaluable)):
        logger.info(
            "  DESACUERDO en '%s': catalogo dice %s, medido dice %s (cobertura maxima %.3f)",
            name,
            "evaluable" if name in declared else "ciego",
            "evaluable" if name in evaluable else "ciego",
            measured[name]["max_coverage"],
        )

    families, skipped = [], []
    from ai_trader.strategies.registry import build_strategy

    # Correr un subconjunto es LEGITIMO aqui y no lo seria en el ranking: este estudio compara
    # cada familia CONSIGO MISMA (ciega contra armada, misma configuracion, misma ventana), asi
    # que ninguna cifra depende de que otras familias esten en la corrida. Por eso una familia
    # admitida despues se puede medir sola, sin repetir las que ya estan medidas.
    requested = tuple(args.families) if getattr(args, "families", None) else NEW_FAMILIES
    unknown = [f for f in requested if f not in NEW_FAMILIES]
    if unknown:
        raise SystemExit(f"Familias que no son tematicas: {unknown}")

    for family in requested:
        theme = getattr(build_strategy(family), "theme", "")
        # El compuesto lee los CINCO temas, asi que le basta con que dos sean legibles.
        readable = theme == "composite" or theme in evaluable
        if readable:
            families.append(family)
        else:
            stat = measured.get(theme, {})
            skipped.append(
                {
                    "family": family,
                    "theme": theme,
                    "reason": (
                        f"el tema '{theme}' NUNCA alcanza {MIN_SIGNAL_COVERAGE:.2f} de "
                        f"cobertura en el archivo: su maximo medido sobre {len(symbols)} "
                        f"simbolos y {THEME_PROBES} sondas es "
                        f"{stat.get('max_coverage', 0.0):.3f}, asi que su puerta no puede "
                        "atar y medirla seria gastar CPU en un cero conocido"
                    ),
                    "max_coverage": stat.get("max_coverage"),
                }
            )

    plan = StudyPlan(
        config_path=str(args.config),
        families=tuple(families),
        families_skipped=tuple(skipped),
        configs_per_family=args.configs_per_family,
        symbols=tuple(symbols),
        windows=tuple(windows),
        themes_evaluable=tuple(sorted(evaluable)),
        themes_blind=tuple(sorted(blind)),
        themes_declared_evaluable=tuple(sorted(declared)),
        themes_measured=measured,
        sources_loaded=tuple(sorted(frames)),
        starting_equity=DEFAULT_STARTING_EQUITY,
        cvar_alpha=DEFAULT_CVAR_ALPHA,
    )
    return plan, build_specs(tuple(families), args.configs_per_family), windows


def _windows_with_universe(
    windows: Sequence[RealWindow],
    bars: dict,
    symbols: Sequence[str],
    *,
    min_history_days: int,
) -> list[dict]:
    """
    Cada ventana, con el universo que le corresponde DECLARADO dentro.

    El universo global son los simbolos con historia suficiente en TODO el rango, y ese no es
    el universo de cada sub-ventana: media cripto de hoy no cotizaba en 2019, asi que en la
    ventana mas antigua faltan los pares jovenes. Que falten es correcto; lo que no vale es que
    se decida solo, dentro del worker y segun lo que conteste el proveedor. Aqui se resuelve
    una vez, con el MISMO criterio de `audit_real_symbols` que usa la transferencia, y viaja en
    la tarea: el brazo ciego y el armado reciben la misma lista por construccion, no por suerte.

    Una ventana sin simbolos se declara y se cae de la muestra; correrla daria un motor sin
    barras y un cero que parece una medicion.
    """
    out: list[dict] = []
    for window in windows:
        kept, dropped = audit_real_symbols(
            bars,
            symbols,
            start=pd.Timestamp(window.start),
            end=pd.Timestamp(window.end),
            min_history_days=min_history_days,
        )
        payload = window.as_dict()
        payload["symbols"] = [a.symbol for a in kept]
        payload["symbols_dropped"] = [
            {"symbol": a.symbol, "reason": a.reason} for a in dropped
        ]
        logger.info(
            "  %s (%s -> %s): %d simbolos de %d",
            payload["label"], payload["start"], payload["end"], len(kept), len(symbols),
        )
        if not kept:
            logger.warning("  %s se cae: ningun simbolo llega al minimo", payload["label"])
            continue
        out.append(payload)
    if not out:
        raise ValueError("Ninguna sub-ventana tiene universo: no hay estudio que correr")
    return out


def _real_bounds(
    config, start: str, end: str, min_history_days: int, *, offline: bool, exchange: str
) -> tuple[tuple[str, ...], datetime, datetime, dict]:
    """
    Que simbolos hay y que rango cubren, con el MISMO cargador que el resto de estudios.

    El rango se pide explicito y no se deduce del cache: `get_daily_bars` necesita fechas
    reales —con `None` revienta con un `NaTType`— y ademas la ventana tiene que ser la misma
    con la que se publicaron los demas estudios, o las sub-ventanas no caerian en los mismos
    sitios y las cifras dejarian de ser comparables sin que nada avise.

    Y el universo se audita con el MISMO criterio que el estudio de transferencia
    (`audit_real_symbols`): un simbolo con menos historia que el calentamiento de la estrategia
    mas un grupo de CPCV no puede llegar a operarse en una sola ventana OOS, asi que no aporta
    nada y si ruido.

    OJO CON UNA DIFERENCIA QUE PARECE UN BUG Y NO LO ES: aqui el universo sale mas grande que
    en la transferencia (24 frente a 11). Los 11 de alli no son los que tienen historia, son
    los que ademas EXISTEN EN LA LIBRERIA SINTETICA, porque aquel estudio compara dos mundos y
    necesita contraparte en los dos. Este no toca el sintetico: compara dos brazos sobre las
    mismas barras reales, asi que no hay contraparte que exigir y quedarse con 11 seria tirar
    la mitad de la muestra sin motivo. El precio es que cada unidad cuesta aproximadamente el
    doble, y esta contado en el coste del estudio.
    """
    service = build_service(exchange, offline=offline)
    window_start = pd.Timestamp(start, tz="UTC")
    window_end = pd.Timestamp(end, tz="UTC")
    requested = crypto_universe(config)
    bars = fetch_real_bars(
        requested, window_start.to_pydatetime(), window_end.to_pydatetime(), service
    )
    if not bars:
        raise ValueError(
            "No hay barras reales en cache. Corre una vez sin --offline (o el estudio de "
            "transferencia, que llena la misma cache) antes de este."
        )
    kept, dropped = audit_real_symbols(
        bars, requested, start=window_start, end=window_end, min_history_days=min_history_days
    )
    if not kept:
        raise ValueError("Ningun simbolo real supera el minimo de historico")
    for audit in dropped:
        logger.info("  omitido %s: %s", audit.symbol, audit.reason)

    usable = {a.symbol: bars[a.symbol] for a in kept}
    firsts = [b.index[0].to_pydatetime() for b in usable.values()]
    lasts = [b.index[-1].to_pydatetime() for b in usable.values()]
    logger.info("Universo real: %d de %d simbolos con historico suficiente", len(kept), len(requested))
    return tuple(sorted(usable)), min(firsts), max(lasts), usable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--configs-per-family", type=int, default=CONFIGS_PER_FAMILY)
    parser.add_argument(
        "--families", nargs="+", default=None,
        help=(
            "Subconjunto de familias tematicas. Legitimo porque cada familia se compara "
            "consigo misma: ninguna cifra depende de quien mas corra."
        ),
    )
    parser.add_argument("--start", default=DEFAULT_REAL_START)
    parser.add_argument("--end", default=DEFAULT_REAL_END)
    parser.add_argument("--window-days", type=int, default=544)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument(
        "--offline", action="store_true",
        help="No llamar al exchange: usar solo la cache. Llega hasta los workers.",
    )
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
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

    # Agrupadas POR VENTANA y no por configuracion: `pool.map` reparte trozos contiguos, asi
    # que con este orden cada worker ve una ventana entera seguida y el cache de barras
    # acierta. Con el orden contrario cambiaba de ventana en cada unidad y no acertaba nunca.
    tasks = [
        (dataclasses.asdict(spec), window, arm)
        for window in windows
        for spec in specs
        for arm in (ARM_BLIND, ARM_ARMED)
    ]
    config = load_config(args.config)
    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            str(args.config), config.signals.raw_root or None, plan.symbols,
            bool(args.offline), args.exchange,
        ),
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
