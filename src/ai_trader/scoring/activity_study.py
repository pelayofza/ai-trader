"""
De donde sale el suelo de actividad: la evidencia, no el criterio de nadie.

`scoring.activity` declara dos condiciones para que una configuracion sea RANKEABLE. Una
esta derivada y la otra hay que medirla, y este estudio es el que la mide:

- `max_zero_window_pct` = alpha = 25%. DERIVADA: la recompensa del sistema es el CVaR@25%,
  la media del peor cuartil. Una ventana OOS sin operaciones deja la curva plana y puntua 0
  EXACTO (Sharpe 0, rotacion 0, caida 0). Si mas de un cuarto de las ventanas esta vacio,
  el cuartil que FIJA la recompensa puede estar hecho de ceros estructurales, y entonces la
  cifra no es una medicion. No hay nada que elegir aqui: el numero sale de alpha.
- `min_median_trades_per_window` = T. MEDIDA: cuantas operaciones tiene que contener la
  ventana mediana. Es el proxy robusto de la condicion anterior —una fraccion de ventanas
  vacias estimada sobre pocas ventanas es ruidosa, y hay configuraciones que operan poco
  SIN dejar ventanas vacias— y por eso se calibra CONTRA ella en vez de inventarse.

REGLA DE DECISION, declarada antes de mirar el resultado (misma disciplina que
`transfer_study.RHO_ACCEPT`): sobre los pares (configuracion, mundo) del estudio de
transferencia, se llama INFORMATIVA a la que cumple la condicion derivada (<= alpha de
ventanas vacias). T es el valor de la rejilla {1, 2, 3, 5, 8, 13, 21} que reproduce esa
clasificacion con MENOS desacuerdos; si varios empatan, el MAYOR, que es el mas estricto de
los que no tiran ninguna medicion valida. La rejilla es gruesa a proposito: la evidencia
disponible separa ordenes de magnitud, no decimales.

Lo que este estudio NO usa para elegir, y conviene que este escrito porque parecia la
metrica obvia: la REPRODUCIBILIDAD del ranking entre mitades del historico. Se mide y se
publica (`reproducibility`), pero como control, no como criterio: sale MAS ALTA con las
configuraciones inactivas dentro que fuera. Es facil de ver una vez medido —una
configuracion que no opera puntua 0 en todos los bloques, asi que su puesto no se mueve
jamas— y es una trampa perfecta: usar la estabilidad para elegir el suelo habria premiado
justo lo que el suelo existe para quitar.

    .venv\\Scripts\\python.exe -m ai_trader.scoring.activity_study

Entrada: `data/transfer/units_<lib>.json` (las unidades crudas del estudio de
transferencia, con sus operaciones fold a fold). Salida: `data/activity/report_<lib>.json`.

Determinista: todo sale de las unidades publicadas y el unico azar —el control de tamano
del ranking— usa un generador con semilla fija.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai_trader.scoring.activity import (
    MAX_ZERO_WINDOW_PCT,
    MIN_MEDIAN_TRADES_PER_WINDOW,
    ActivityFloor,
    ActivityStats,
)
from ai_trader.scoring.transfer_study import (
    DEFAULT_LIBRARY_ID,
    SIDE_REAL,
    SIDE_SYNTHETIC,
    SideScores,
    build_specs,
    collect_side,
    pooled_gate,
)
from ai_trader.scoring.weight_calibration import spearman
from ai_trader.shared.reports import load_report

logger = logging.getLogger("activity_study")

OUT_DIR = Path("data") / "activity"
UNITS_DIR = Path("data") / "transfer"

# Rejilla de umbrales candidatos, en operaciones por ventana OOS. Gruesa a proposito: la
# evidencia separa "opera" de "no opera" (un orden de magnitud), no decimales.
THRESHOLD_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0)

# Control de tamano del ranking: subconjuntos aleatorios del MISMO tamano que el elegible.
# Sin el, comparar la estabilidad de 16 configuraciones con la de 9 mide el tamano.
CONTROL_DRAWS = 300
CONTROL_SEED = 20260811

SIDES = (SIDE_REAL, SIDE_SYNTHETIC)


def activity_report_path(library_id: str, out_dir: Path | str = OUT_DIR) -> Path:
    return Path(out_dir) / f"report_{library_id}.json"


def load_activity_report(path: Path | str) -> dict | None:
    """Lee el informe publicado; None si no esta, para que dashboard y documentacion
    degraden a prosa sin cifras en vez de romperse."""
    return load_report(path)


# ------------------------------------------------------------------ mecanismo --------


def _cvar(scores: Sequence[float], alpha: float) -> float:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0.0
    k = max(1, math.ceil(alpha * arr.size))
    return float(np.sort(arr)[:k].mean())


def mechanism_check(sides: dict[str, SideScores], config_ids: Sequence[str], alpha: float) -> dict:
    """
    La aritmetica del problema, COMPROBADA sobre los datos en vez de razonada.

    Afirmacion: una ventana sin operaciones puntua 0 exacto, y por eso un reward de 0 puede
    ser "no perdio" o "no jugo". Aqui se cuentan las ventanas vacias, cuantas de ellas
    puntuan exactamente 0 (deberian ser todas) y cuantas ventanas con operaciones puntuan
    exactamente 0 (deberian ser ninguna). Ademas se mide la firma del ranking degenerado:
    Spearman(recompensa, operaciones por ventana), que en el lado real salio -0.84.

    Y se mira donde duele: cuantas de las ventanas que FIJAN la recompensa (el peor cuartil,
    el que promedia el CVaR) estan vacias. Esa es la contaminacion concreta.
    """
    out: dict = {"sides": {}}
    for side_name in SIDES:
        side = sides[side_name]
        empty = zero_scored = nonempty_zero = total = 0
        tail_windows = tail_empty = 0
        for config_id in config_ids:
            scores = side.pooled(config_id)
            trades = side.pooled_trades(config_id) or []
            total += len(scores)
            for score, n_trades in zip(scores, trades):
                if n_trades == 0:
                    empty += 1
                    zero_scored += int(score == 0.0)
                elif score == 0.0:
                    nonempty_zero += 1
            order = np.argsort(np.asarray(scores, dtype=float))
            k = max(1, math.ceil(alpha * len(scores)))
            tail_windows += k
            tail_empty += sum(1 for i in order[:k] if trades[i] == 0)

        rewards = [_cvar(side.pooled(c), alpha) for c in config_ids]
        activity = [_stats(side, c).trades_per_window for c in config_ids]
        out["sides"][side_name] = {
            "windows": total,
            "empty_windows": empty,
            "empty_windows_scoring_exactly_zero": zero_scored,
            "non_empty_windows_scoring_exactly_zero": nonempty_zero,
            # La afirmacion "sin operaciones -> headline 0 exacto" se cumple en las dos
            # direcciones o el mecanismo descrito no es el que esta actuando.
            "holds": zero_scored == empty and nonempty_zero == 0,
            "cvar_tail_windows": tail_windows,
            "cvar_tail_empty_windows": tail_empty,
            "cvar_tail_empty_pct": round(100.0 * tail_empty / tail_windows, 2) if tail_windows else 0.0,
            "spearman_reward_activity": _round(spearman(rewards, activity)),
        }
    return out


def _stats(side: SideScores, config_id: str) -> ActivityStats:
    stats = side.activity(config_id)
    if stats is None:
        raise ValueError(
            f"Las unidades no traen operaciones ventana a ventana para {config_id} "
            f"({side.side}). Re-corre `ai_trader.scoring.transfer_study`: sin ese detalle "
            "no se puede medir la fraccion de ventanas vacias, que es la mitad del suelo."
        )
    return stats


# -------------------------------------------------------------------- barrido --------


def sweep(
    sides: dict[str, SideScores],
    config_ids: Sequence[str],
    *,
    alpha: float,
    grid: Sequence[float] = THRESHOLD_GRID,
) -> list[dict]:
    """
    Que pasaria con cada umbral candidato: quien queda dentro, quien gana, quien aprueba.

    Se publica entero, no solo el elegido. Un umbral que solo se puede defender enseñando
    su resultado y escondiendo los de al lado no es un umbral defendible.
    """
    rows: list[dict] = []
    for threshold in grid:
        floor = ActivityFloor(
            min_median_trades_per_window=threshold, max_zero_window_pct=MAX_ZERO_WINDOW_PCT
        )
        row: dict = {"threshold": threshold, "sides": {}, "disagreements": 0}
        for side_name in SIDES:
            side = sides[side_name]
            rewards = {c: _cvar(side.pooled(c), alpha) for c in config_ids}
            eligible = [c for c in config_ids if floor.eligible(_stats(side, c))]
            # Desacuerdos con la condicion DERIVADA: la mediana de operaciones deja fuera a
            # una informativa, o deja dentro a una que no lo es.
            disagreements = [
                c for c in config_ids
                if (_stats(side, c).zero_window_pct <= MAX_ZERO_WINDOW_PCT)
                != (_stats(side, c).median_trades_per_window >= threshold)
            ]
            gates = {c: pooled_gate(side, c, alpha) for c in config_ids}
            ordered = sorted(config_ids, key=lambda c: (-rewards[c], c))
            ordered_eligible = sorted(eligible, key=lambda c: (-rewards[c], c))
            row["sides"][side_name] = {
                "n_eligible": len(eligible),
                "eligible": ordered_eligible,
                "winner": ordered_eligible[0] if ordered_eligible else None,
                "winner_changes": bool(
                    ordered_eligible and ordered and ordered_eligible[0] != ordered[0]
                ),
                "approved": sorted(
                    c for c in eligible if gates[c].beats_baselines
                ),
                "spearman_reward_activity": _round(
                    spearman(
                        [rewards[c] for c in eligible],
                        [_stats(side, c).trades_per_window for c in eligible],
                    )
                ) if len(eligible) >= 3 else None,
                "max_zero_window_pct_eligible": round(
                    max((_stats(side, c).zero_window_pct for c in eligible), default=0.0), 2
                ),
                "disagreements": disagreements,
            }
            row["disagreements"] += len(disagreements)
        rows.append(row)
    return rows


def choose_threshold(rows: Sequence[dict]) -> dict:
    """Aplica la regla declarada en el docstring del modulo: menos desacuerdos con la
    condicion derivada y, a igualdad, el umbral MAYOR."""
    best = min(r["disagreements"] for r in rows)
    tied = [r["threshold"] for r in rows if r["disagreements"] == best]
    chosen = max(tied)
    return {
        "rule": (
            "el valor de la rejilla que reproduce con menos desacuerdos la condicion "
            f"derivada (<= {MAX_ZERO_WINDOW_PCT:.0f}% de ventanas vacias, que es alpha); a "
            "igualdad, el mayor"
        ),
        "grid": list(THRESHOLD_GRID),
        "chosen": chosen,
        "disagreements": best,
        "tied": tied,
        "published": MIN_MEDIAN_TRADES_PER_WINDOW,
        "matches_published": chosen == MIN_MEDIAN_TRADES_PER_WINDOW,
    }


def stability_band(
    sides: dict[str, SideScores], config_ids: Sequence[str], chosen: float
) -> dict:
    """
    Cuanto depende lo publicado del NUMERO elegido.

    Dos medidas distintas y las dos importan:

    - `same_partition`: que umbrales de la rejilla dan EXACTAMENTE el mismo conjunto
      rankeable que el elegido. Si son varios, la discusion sobre el decimal esta vacia: lo
      que se publica no cambia. Es la comprobacion honesta de un umbral, mejor que
      defenderlo con prosa.
    - `largest_gap`: el hueco mas ancho de la distribucion de actividad, para ver de que
      tamano es la separacion entre "opera" y "no opera" en esta rejilla.
    """
    out: dict = {}
    for side_name in SIDES:
        side = sides[side_name]
        medians = sorted(_stats(side, c).median_trades_per_window for c in config_ids)
        gaps = [
            (medians[i + 1] - medians[i], medians[i], medians[i + 1])
            for i in range(len(medians) - 1)
        ]
        width, lo, hi = max(gaps, default=(0.0, 0.0, 0.0))

        def eligible_at(threshold: float) -> list[str]:
            floor = ActivityFloor(
                min_median_trades_per_window=threshold,
                max_zero_window_pct=MAX_ZERO_WINDOW_PCT,
            )
            return [c for c in config_ids if floor.eligible(_stats(side, c))]

        reference = eligible_at(chosen)
        same = [t for t in THRESHOLD_GRID if eligible_at(t) == reference]
        out[side_name] = {
            "medians": medians,
            "largest_gap": round(width, 2),
            "gap_between": [round(lo, 2), round(hi, 2)],
            "same_partition": same,
            "n_rankable": len(reference),
            "note": (
                f"con {'/'.join(f'{t:.0f}' for t in same)} operaciones por ventana sale el "
                "MISMO conjunto rankeable en este lado"
            ),
        }
    return out


# ------------------------------------------------------------- reproducibilidad ------


def split_half_stability(
    side: SideScores,
    config_ids: Sequence[str],
    *,
    alpha: float,
    subsets: Sequence[Sequence[str]] | None = None,
) -> float:
    """
    Spearman entre el ranking de una MITAD de los bloques y el de la otra.

    Mide si el orden se sostiene al cambiar el tramo de historia. Se promedia sobre todas
    las particiones equilibradas de los bloques.
    """
    n_blocks = side.n_blocks
    if n_blocks < 2 or len(config_ids) < 3:
        return float("nan")
    half = n_blocks // 2
    values: list[float] = []
    for left in itertools.combinations(range(n_blocks), half):
        right = [i for i in range(n_blocks) if i not in left]
        a = [_cvar([s for i in left for s in side.by_config[c][i]], alpha) for c in config_ids]
        b = [_cvar([s for i in right for s in side.by_config[c][i]], alpha) for c in config_ids]
        rho = spearman(a, b)
        if rho == rho:
            values.append(rho)
    return float(np.mean(values)) if values else float("nan")


def reproducibility_control(
    sides: dict[str, SideScores],
    config_ids: Sequence[str],
    *,
    alpha: float,
    floor: ActivityFloor,
) -> dict:
    """
    El control que explica por que la estabilidad NO puede elegir el umbral.

    Para cada lado: estabilidad del ranking completo, la del subconjunto rankeable y la de
    subconjuntos ALEATORIOS del mismo tamano (que es el control necesario: un ranking de 9
    y otro de 16 no tienen la misma varianza por azar). Si el conjunto completo sale mas
    estable que el rankeable, la estabilidad esta midiendo inmovilidad, no acuerdo.
    """
    rng = np.random.default_rng(CONTROL_SEED)
    out: dict = {"draws": CONTROL_DRAWS, "seed": CONTROL_SEED, "sides": {}}
    for side_name in SIDES:
        side = sides[side_name]
        eligible = [c for c in config_ids if floor.eligible(_stats(side, c))]
        full = split_half_stability(side, config_ids, alpha=alpha)
        restricted = split_half_stability(side, eligible, alpha=alpha)
        control: list[float] = []
        for _ in range(CONTROL_DRAWS):
            pick = rng.choice(len(config_ids), size=len(eligible), replace=False)
            value = split_half_stability(
                side, [config_ids[i] for i in sorted(pick)], alpha=alpha
            )
            if value == value:
                control.append(value)
        mean = float(np.mean(control)) if control else float("nan")
        std = float(np.std(control)) if control else float("nan")
        out["sides"][side_name] = {
            "all_configs": _round(full),
            "rankable_only": _round(restricted),
            "random_same_size_mean": _round(mean),
            "random_same_size_std": _round(std),
            "z_vs_random": _round((restricted - mean) / std) if std and std == std and std > 0 else None,
            "inactivity_inflates_stability": bool(
                full == full and restricted == restricted and full > restricted
            ),
        }
    return out


# ---------------------------------------------------------------------- gate ---------


def gate_effect(sides: dict[str, SideScores], config_ids: Sequence[str], *, alpha: float) -> dict:
    """Cuantas configuraciones dejan de aprobar el gate al exigir actividad, y cuales.

    `beats_baselines` es el veredicto ANTIGUO (solo batir a los pasivos) y `approved` el
    nuevo (batirlos Y ser rankeable), asi que la diferencia se lee sin recalcular nada."""
    out: dict = {}
    for side_name in SIDES:
        side = sides[side_name]
        gates = {c: pooled_gate(side, c, alpha) for c in config_ids}
        without = sorted(c for c in config_ids if gates[c].beats_baselines)
        with_floor = sorted(c for c in config_ids if gates[c].approved)
        lost = [c for c in without if c not in with_floor]
        out[side_name] = {
            "approved_without_floor": without,
            "approved_with_floor": with_floor,
            "lost": lost,
            "n_lost": len(lost),
            "lost_detail": [
                {
                    "config_id": c,
                    "trades_per_window": round(_stats(side, c).trades_per_window, 2),
                    "zero_window_pct": round(_stats(side, c).zero_window_pct, 1),
                    "reward": round(_cvar(side.pooled(c), alpha), 4),
                    "reasons": list(gates[c].ineligible_reasons),
                }
                for c in lost
            ],
        }
    return out


# ------------------------------------------------------------------- informe ---------


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or value != value:
        return None
    return round(float(value), digits)


def analyze(rows: Sequence[dict], plan: dict, config_ids: Sequence[str]) -> dict:
    alpha = plan["validation"]["cvar_alpha"]
    sides = {name: collect_side(rows, name, config_ids, alpha=alpha) for name in SIDES}
    kept = [c for c in config_ids if all(c in sides[s].by_config for s in SIDES)]
    if len(kept) < 3:
        raise ValueError("Menos de 3 configuraciones completas en los dos lados")

    grid_rows = sweep(sides, kept, alpha=alpha)
    decision = choose_threshold(grid_rows)
    floor = ActivityFloor(
        min_median_trades_per_window=decision["chosen"], max_zero_window_pct=MAX_ZERO_WINDOW_PCT
    )

    configs = []
    for config_id in kept:
        entry: dict = {"config_id": config_id}
        for side_name in SIDES:
            side = sides[side_name]
            stats = _stats(side, config_id)
            entry[side_name] = {
                **stats.as_dict(),
                "reward": round(_cvar(side.pooled(config_id), alpha), 4),
                "rankable": floor.eligible(stats),
                "informative": stats.zero_window_pct <= MAX_ZERO_WINDOW_PCT,
            }
        configs.append(entry)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "units": str(UNITS_DIR / f"units_{plan['library_id']}.json"),
            "library_id": plan["library_id"],
            "n_configs": len(kept),
            "cvar_alpha": alpha,
            "blocks": {name: sides[name].n_blocks for name in SIDES},
            "windows_per_block": plan["validation"]["n_folds"],
        },
        "floor": {
            **floor.as_dict(),
            "derived": {
                "max_zero_window_pct": (
                    "es alpha en porcentaje: por encima de esa fraccion de ventanas vacias, "
                    "el cuartil que promedia el CVaR puede estar hecho de ceros estructurales"
                ),
            },
        },
        "mechanism": mechanism_check(sides, kept, alpha),
        "configs": configs,
        "sweep": grid_rows,
        "decision": decision,
        "band": stability_band(sides, kept, decision["chosen"]),
        "reproducibility": {
            **reproducibility_control(sides, kept, alpha=alpha, floor=floor),
            "used_as_criterion": False,
            "why_not": (
                "Sale MAS ALTA con las inactivas dentro: una configuracion que no opera "
                "puntua 0 en todos los bloques y su puesto no se mueve nunca. Elegir el "
                "suelo por estabilidad habria premiado exactamente lo que el suelo quita."
            ),
        },
        "gate": gate_effect(sides, kept, alpha=alpha),
    }


def _print_report(report: dict) -> None:
    src, dec, floor = report["source"], report["decision"], report["floor"]
    print("\n=== SUELO DE ACTIVIDAD: DE DONDE SALE EL UMBRAL ===")
    print(
        f"  libreria {src['library_id']} | {src['n_configs']} configuraciones | "
        f"{src['blocks']['real']} bloques reales x {src['windows_per_block']} ventanas, "
        f"{src['blocks']['synthetic']} sinteticos"
    )
    for side in SIDES:
        m = report["mechanism"]["sides"][side]
        print(
            f"  [{side:<9}] {m['empty_windows']}/{m['windows']} ventanas vacias, y "
            f"{m['empty_windows_scoring_exactly_zero']} de ellas puntuan 0 EXACTO "
            f"({'la aritmetica se cumple' if m['holds'] else 'OJO: no se cumple'}) | "
            f"Spearman(recompensa, operaciones) = {m['spearman_reward_activity']}"
        )
        print(
            f"              cola del CVaR: {m['cvar_tail_empty_windows']}/"
            f"{m['cvar_tail_windows']} ventanas vacias ({m['cvar_tail_empty_pct']}%)"
        )

    print(f"\n{'config':<22}{'ops/vent':>9}{'vacias%':>9}{'reward':>9}{'rank?':>7}   (lado real)")
    for row in sorted(report["configs"], key=lambda c: -c[SIDE_REAL]["reward"]):
        r = row[SIDE_REAL]
        print(
            f"{row['config_id']:<22}{r['trades_per_window']:>9.2f}{r['zero_window_pct']:>9.1f}"
            f"{r['reward']:>9.3f}{('si' if r['rankable'] else 'NO'):>7}"
        )

    print(f"\n{'T':>5}{'elegibles':>11}{'desacuerdos':>13}{'ganador (real)':>24}{'aprueban':>10}")
    for row in report["sweep"]:
        real = row["sides"][SIDE_REAL]
        print(
            f"{row['threshold']:>5.0f}{real['n_eligible']:>11}{row['disagreements']:>13}"
            f"{str(real['winner']):>24}{len(real['approved']):>10}"
        )
    print(
        f"\n  REGLA: {dec['rule']}\n  -> T = {dec['chosen']:.0f} operaciones por ventana "
        f"(desacuerdos: {dec['disagreements']}; empatan {dec['tied']})"
    )
    band = report["band"][SIDE_REAL]
    print(
        f"  el numero casi no importa (lado real): {band['note']} "
        f"({band['n_rankable']} de {src['n_configs']})"
    )
    print(
        f"  suelo publicado: >= {floor['min_median_trades_per_window']:.0f} ops en la "
        f"ventana mediana y <= {floor['max_zero_window_pct']:.0f}% de ventanas vacias"
    )

    rep = report["reproducibility"]["sides"][SIDE_REAL]
    print(
        f"\n  CONTROL (no es el criterio): reproducibilidad del ranking real = "
        f"{rep['all_configs']} con todas, {rep['rankable_only']} solo rankeables, "
        f"{rep['random_same_size_mean']} en subconjuntos aleatorios del mismo tamano"
    )
    g = report["gate"][SIDE_REAL]
    print(
        f"  GATE (lado real): aprueban {len(g['approved_without_floor'])} sin suelo -> "
        f"{len(g['approved_with_floor'])} con suelo; pierden la aprobacion {g['n_lost']}: "
        f"{', '.join(g['lost']) or 'ninguna'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--library", default=DEFAULT_LIBRARY_ID)
    parser.add_argument("--units-dir", default=str(UNITS_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--configs-per-family", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    units_path = Path(args.units_dir) / f"units_{args.library}.json"
    payload = json.loads(units_path.read_text(encoding="utf-8"))
    specs = build_specs(per_family=args.configs_per_family)

    report = analyze(payload["rows"], payload["plan"], [s.id for s in specs])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = activity_report_path(args.library, out_dir)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Informe -> %s", path)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
