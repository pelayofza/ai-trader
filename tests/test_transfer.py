"""
Tests del estudio de transferencia de ranking (real vs sintetico).

El estudio produce UNA cifra —el Spearman entre los dos rankings— de la que cuelga una
decision de arquitectura, asi que lo que hay que blindar no es "que corra", sino las
cuatro propiedades sin las cuales esa cifra no significaria nada:

- **La rejilla es la misma que la del estudio de pesos.** Si `build_specs` derivara, los
  dos estudios hablarian de conjuntos distintos de configuraciones sin avisar.
- **Los dos lados son comparables.** Sub-ventanas reales del mismo tamano que un camino
  sintetico, disjuntas, y simbolos sin historico omitidos en vez de rellenados.
- **Una sola cola, no dos.** La recompensa es el CVaR de TODOS los scores puestos en
  comun; el CVaR de CVaR es otra cosa, y el test lo demuestra midiendo la diferencia.
- **El veredicto sale de un umbral declarado**, no de la prosa que lo acompana.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, aggregate_reward
from ai_trader.scoring.multiwindow import validate_multiwindow
from ai_trader.scoring.transfer_study import (
    FLOW_NONE,
    FLOW_TRANSFERS,
    RHO_ACCEPT,
    SIDE_REAL,
    SIDE_SYNTHETIC,
    VERDICT_INVERTED,
    VERDICT_NONE,
    VERDICT_TRANSFERS,
    RealWindow,
    _activity_reading,
    _cvar,
    _hypergeometric_tail,
    _ranks_best_first,
    activity_check,
    analyze,
    audit_real_symbols,
    block_bootstrap_rho,
    build_specs,
    build_tasks,
    collect_side,
    config_bootstrap_rho,
    permutation_p_value,
    rank_discrepancies,
    real_windows,
    top_k_overlap,
    trades_per_fold,
    verdict_for,
)
from ai_trader.scoring.weight_calibration import candidate_specs, spearman
from ai_trader.scoring.weight_study import FAMILIES, STUDY_SEED
from test_backtest_engine import FakeService, make_config, trending_df

UTC = timezone.utc


# --------------------------------------------------------------------- rejilla -------


class TestGrid:
    """La rejilla tiene que ser, literalmente, la del estudio de pesos."""

    def test_es_la_del_estudio_de_pesos(self):
        specs = build_specs(FAMILIES, 8)
        expected = [
            spec
            for i, family in enumerate(FAMILIES)
            for spec in candidate_specs(family, 8, seed=STUDY_SEED + i)
        ]
        assert [s.id for s in specs] == [s.id for s in expected]
        assert [s.params for s in specs] == [s.params for s in expected]

    def test_dieciseis_configuraciones_de_dos_familias(self):
        specs = build_specs(FAMILIES, 8)
        assert len(specs) == 16
        assert sorted({s.type for s in specs}) == sorted(FAMILIES)
        assert len({s.id for s in specs}) == 16

    def test_determinista(self):
        assert [s.params for s in build_specs()] == [s.params for s in build_specs()]


# ------------------------------------------------------------ geometria del real -----


class TestRealWindows:
    """Las sub-ventanas reales son la mitad de la comparabilidad: mismo tamano que un
    camino sintetico, sin solaparse y sin inventar calendario."""

    def test_mismo_tamano_disjuntas_y_ancladas_al_final(self):
        start, end = datetime(2018, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
        windows = real_windows(start, end, 544)

        assert [w.days for w in windows] == [544] * len(windows)
        assert windows[-1].end == end  # ancladas al FINAL: el resto cae en la cabecera
        assert windows[0].start > start
        for previous, following in zip(windows, windows[1:]):
            assert previous.end == following.start  # contiguas y semiabiertas: sin solape

    def test_la_cabecera_descartada_es_el_resto_exacto(self):
        start, end = datetime(2018, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
        total = (end - start).days
        windows = real_windows(start, end, 544)
        assert (windows[0].start - start).days == total % 544

    def test_historico_corto_falla_en_vez_de_encoger_la_ventana(self):
        with pytest.raises(ValueError, match="no llega"):
            real_windows(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC), 544)

    def test_serializa_el_ultimo_dia_incluido(self):
        window = RealWindow("w1", datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 11, tzinfo=UTC))
        assert window.as_dict() == {
            "label": "w1", "start": "2020-01-01", "end": "2020-01-10", "days": 10
        }


def _bars_from(first_day: str, n: int) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [pd.Timestamp(first_day, tz="UTC") + timedelta(days=i) for i in range(n)],
        name="timestamp",
    )
    return pd.DataFrame({"close": np.linspace(100.0, 200.0, n)}, index=index)


class TestSymbolAudit:
    """Un par sin historico se declara y se omite. Nunca se rellena."""

    START = pd.Timestamp("2018-01-01", tz="UTC")
    END = pd.Timestamp("2024-01-01", tz="UTC")

    def test_separa_por_el_umbral_declarado(self):
        bars = {
            "BTC/USDT": _bars_from("2018-01-01", 2000),
            "NEW/USDT": _bars_from("2023-10-01", 90),
        }
        kept, dropped = audit_real_symbols(
            bars, ["BTC/USDT", "NEW/USDT"], start=self.START, end=self.END, min_history_days=270
        )
        assert [a.symbol for a in kept] == ["BTC/USDT"]
        assert [a.symbol for a in dropped] == ["NEW/USDT"]

    def test_el_motivo_lleva_las_cifras_que_lo_justifican(self):
        bars = {"NEW/USDT": _bars_from("2023-10-01", 90)}
        _, dropped = audit_real_symbols(
            bars, ["NEW/USDT"], start=self.START, end=self.END, min_history_days=270
        )
        audit = dropped[0]
        assert audit.n_bars == 90
        assert "90" in audit.reason and "270" in audit.reason
        assert audit.first_bar == "2023-10-01"

    def test_simbolo_sin_barras_se_declara_no_se_inventa(self):
        kept, dropped = audit_real_symbols(
            {}, ["GONE/USDT"], start=self.START, end=self.END, min_history_days=270
        )
        assert kept == []
        assert dropped[0].n_bars == 0 and dropped[0].first_bar is None
        assert "cache" in dropped[0].reason

    def test_solo_cuentan_las_barras_dentro_de_la_ventana(self):
        # Historia larguisima pero casi toda ANTERIOR al estudio: no sirve para puntuar.
        bars = {"OLD/USDT": _bars_from("2014-01-01", 1600)}
        kept, dropped = audit_real_symbols(
            bars, ["OLD/USDT"], start=self.START, end=self.END, min_history_days=270
        )
        assert [a.symbol for a in dropped] == ["OLD/USDT"]
        assert kept == []


# ------------------------------------------------------- una sola cola, no dos -------


class TestPooling:
    def test_cvar_replica_la_recompensa_del_sistema(self):
        scores = [3.0, -1.0, 0.5, -2.5, 1.25, 0.0, -0.75, 2.0]
        assert _cvar(scores, DEFAULT_CVAR_ALPHA) == pytest.approx(
            aggregate_reward(scores, alpha=DEFAULT_CVAR_ALPHA).reward
        )

    def test_poner_en_comun_no_es_cvar_de_cvar(self):
        """La razon por la que el estudio agrega en comun: componer dos colas da otro
        numero, y sistematicamente mas pesimista."""
        blocks = [[2.0, 1.0, 0.0, -1.0], [3.0, 2.5, -4.0, 1.0], [0.5, 0.5, 0.5, -6.0]]
        pooled = _cvar([s for b in blocks for s in b], DEFAULT_CVAR_ALPHA)
        nested = _cvar([_cvar(b, DEFAULT_CVAR_ALPHA) for b in blocks], DEFAULT_CVAR_ALPHA)
        assert pooled != pytest.approx(nested)
        assert nested < pooled

    def test_cvar_vacio_es_cero_y_no_revienta(self):
        assert _cvar([], DEFAULT_CVAR_ALPHA) == 0.0


class TestRanks:
    def test_rango_uno_es_la_mejor(self):
        assert _ranks_best_first([0.1, 5.0, -3.0]) == [2, 1, 3]

    def test_empates_resueltos_de_forma_estable(self):
        assert _ranks_best_first([1.0, 1.0, 0.0]) == [1, 2, 3]


# -------------------------------------------------------------- filas -> matriz ------


def _row(
    config_id: str,
    side: str,
    unit: str,
    scores: list[float],
    *,
    failed: bool = False,
    num_trades: int = 1000,
) -> dict:
    return {
        "config_id": config_id,
        "side": side,
        "unit_id": unit,
        "path_index": 0,
        "failed": failed,
        "scores": scores,
        "baseline_scores": {"btc_hold": [0.5] * len(scores)},
        "approved": True,
        "leakage_ok": True,
        "embargo_days": 5,
        "coverage": {},
        "num_trades": num_trades,
    }


class TestCollectSide:
    def test_pooled_concatena_los_bloques_en_orden(self):
        rows = [
            _row("a", SIDE_REAL, "w1", [1.0, 2.0]),
            _row("a", SIDE_REAL, "w2", [3.0, 4.0]),
        ]
        side = collect_side(rows, SIDE_REAL, ["a"])
        assert side.block_ids == ["w1", "w2"]
        assert side.pooled("a") == [1.0, 2.0, 3.0, 4.0]

    def test_config_incompleta_se_descarta_entera_y_se_declara(self):
        rows = [
            _row("a", SIDE_REAL, "w1", [1.0]), _row("a", SIDE_REAL, "w2", [2.0]),
            _row("b", SIDE_REAL, "w1", [1.0]),  # le falta w2
        ]
        side = collect_side(rows, SIDE_REAL, ["a", "b"])
        assert list(side.by_config) == ["a"]
        assert side.dropped == ["b"]

    def test_unidad_fallida_descarta_la_config(self):
        rows = [
            _row("a", SIDE_REAL, "w1", [1.0]),
            _row("a", SIDE_REAL, "w2", [], failed=True),
            _row("b", SIDE_REAL, "w1", [1.0]), _row("b", SIDE_REAL, "w2", [2.0]),
        ]
        side = collect_side(rows, SIDE_REAL, ["a", "b"])
        assert side.dropped == ["a"] and list(side.by_config) == ["b"]

    def test_los_baselines_se_guardan_una_vez_por_bloque(self):
        """No dependen de la configuracion: son carteras pasivas sobre las mismas
        ventanas. Guardarlos 16 veces seria contar 16 veces la misma evidencia."""
        rows = [_row(c, SIDE_REAL, "w1", [1.0]) for c in ("a", "b")]
        side = collect_side(rows, SIDE_REAL, ["a", "b"])
        assert side.baselines["btc_hold"] == [[0.5]]


# ------------------------------------------------------------------ transferencia ----


class TestTopK:
    def test_acuerdo_perfecto_acierta_todas(self):
        ranks = list(range(1, 17))
        out = top_k_overlap([f"c{i}" for i in range(16)], ranks, ranks, k=4)
        assert out["hits"] == 4
        assert out["expected_by_chance"] == 2.0
        assert out["p_value"] < 0.05

    def test_orden_invertido_no_acierta_ninguna(self):
        ranks = list(range(1, 17))
        out = top_k_overlap([f"c{i}" for i in range(16)], ranks[::-1], ranks, k=4)
        assert out["hits"] == 0
        assert out["p_value"] == pytest.approx(1.0)

    def test_hipergeometrica_conocida(self):
        assert _hypergeometric_tail(0, 4, 8, 16) == pytest.approx(1.0)
        assert _hypergeometric_tail(4, 4, 8, 16) == pytest.approx(
            math.comb(8, 4) / math.comb(16, 4)
        )


class TestDiscrepancies:
    def test_delta_positivo_significa_que_el_sintetico_sobrevalora(self):
        # c0: el sintetico la pone la primera y el real la ultima -> sobrevalorada.
        out = rank_discrepancies(["c0", "c1"], rank_real=[2, 1], rank_synth=[1, 2], threshold=1)
        over = [i for i in out["items"] if i["config_id"] == "c0"][0]
        assert over["delta"] == 1
        assert out["n_overrated_by_synthetic"] == 1
        assert out["n_underrated_by_synthetic"] == 1

    def test_acuerdo_perfecto_no_deja_desacuerdos(self):
        ranks = list(range(1, 9))
        out = rank_discrepancies([f"c{i}" for i in range(8)], ranks, ranks)
        assert out["n_large"] == 0 and out["max_abs_delta"] == 0


class TestVerdict:
    def _boot(self, excludes_zero: bool = True) -> dict:
        return {"excludes_zero": excludes_zero, "lo": 0.1, "hi": 0.6}

    def test_por_encima_del_umbral_es_pre_cribado(self):
        v = verdict_for(RHO_ACCEPT, self._boot())
        assert v["key"] == VERDICT_TRANSFERS and v["transfers"] is True
        assert v["flow"] == FLOW_TRANSFERS

    def test_justo_por_debajo_no_transfiere(self):
        v = verdict_for(RHO_ACCEPT - 0.01, self._boot())
        assert v["key"] == VERDICT_NONE and v["transfers"] is False
        assert v["flow"] == FLOW_NONE

    def test_ordenacion_invertida_se_nombra_aparte(self):
        v = verdict_for(-0.5, self._boot())
        assert v["key"] == VERDICT_INVERTED and v["flow"] == FLOW_NONE

    def test_rho_no_calculable_no_transfiere(self):
        v = verdict_for(float("nan"), self._boot())
        assert v["key"] == VERDICT_NONE and v["rho"] is None

    def test_el_intervalo_se_reporta_pero_no_cambia_el_veredicto(self):
        assert verdict_for(0.5, self._boot(False))["key"] == VERDICT_TRANSFERS
        assert verdict_for(0.5, self._boot(False))["ci_excludes_zero"] is False


def _side_rows(side: str, config_ids, blocks, rewards_by_config, trades=None) -> list[dict]:
    """Filas sinteticas con scores construidos para que el CVaR agregado sea el pedido."""
    return [
        _row(
            cid, side, block, [rewards_by_config[cid] + 0.01 * b] * 4,
            num_trades=1000 if trades is None else trades[cid],
        )
        for cid in config_ids
        for b, block in enumerate(blocks)
    ]


class TestActivityControl:
    """Un CVaR premia no operar: una curva plana puntúa 0 exacto y un 0 gana a cualquier
    cosa que arriesgue y pierda. Sin este control, un rho ~ 0 no se distingue de "ninguno
    de los dos lados estaba rankeando estrategias"."""

    CONFIGS = [f"c{i}" for i in range(6)]

    def _sides(self, real_trades, synth_trades, real_levels=None, synth_levels=None):
        real_levels = real_levels or {c: float(i) for i, c in enumerate(self.CONFIGS)}
        synth_levels = synth_levels or dict(real_levels)
        rows = [
            *_side_rows(SIDE_REAL, self.CONFIGS, ["w1", "w2"], real_levels, real_trades),
            *_side_rows(SIDE_SYNTHETIC, self.CONFIGS, ["s1", "s2"], synth_levels, synth_trades),
        ]
        return (
            rows,
            collect_side(rows, SIDE_REAL, self.CONFIGS),
            collect_side(rows, SIDE_SYNTHETIC, self.CONFIGS),
        )

    def test_operaciones_por_ventana_es_la_mediana_sobre_bloques(self):
        rows, _, _ = self._sides({c: 40 for c in self.CONFIGS}, {c: 40 for c in self.CONFIGS})
        # 40 operaciones repartidas en los 4 folds de cada bloque -> 10 por ventana.
        assert trades_per_fold(rows, SIDE_REAL) == {c: 10.0 for c in self.CONFIGS}

    def test_detecta_un_ranking_dominado_por_la_inactividad(self):
        # Recompensa perfectamente decreciente en actividad: el que menos opera, gana.
        busy = {c: (i + 1) * 400 for i, c in enumerate(self.CONFIGS)}
        rows, real, synth = self._sides(busy, busy)
        out = activity_check(rows, real, synth, self.CONFIGS, alpha=DEFAULT_CVAR_ALPHA)
        assert out["reward_vs_activity_real"] == pytest.approx(1.0)

    def test_el_subconjunto_activo_exige_operar_en_LOS_DOS_mundos(self):
        real_trades = {c: 400 for c in self.CONFIGS}          # 100 por ventana: activas
        synth_trades = {**{c: 400 for c in self.CONFIGS}, "c0": 4, "c1": 4}  # 1 por ventana
        rows, real, synth = self._sides(real_trades, synth_trades)
        out = activity_check(
            rows, real, synth, self.CONFIGS, alpha=DEFAULT_CVAR_ALPHA, min_trades_per_fold=5
        )
        assert out["active_configs"] == ["c2", "c3", "c4", "c5"]
        assert out["n_active"] == 4

    def test_el_rho_activo_llega_con_intervalo_y_p_valor(self):
        """Es un subconjunto POST-HOC y pequeño: publicarlo desnudo invitaría a leerlo con
        una confianza que no tiene."""
        busy = {c: 400 for c in self.CONFIGS}
        rows, real, synth = self._sides(busy, busy)
        out = activity_check(rows, real, synth, self.CONFIGS, alpha=DEFAULT_CVAR_ALPHA)
        assert out["spearman_active"] == pytest.approx(1.0)
        assert out["bootstrap_active"]["lo"] <= out["spearman_active"]
        assert out["permutation_active"]["p_value"] > 0

    @pytest.mark.parametrize(
        "rho_active, expected",
        [
            (-0.67, "se vuelve NEGATIVO"),
            (0.05, "sigue siendo nulo"),
            (0.60, "SI aparece acuerdo"),
            (None, "NO puede separarse de la inactividad"),
        ],
    )
    def test_el_informe_lleva_la_lectura_hecha_no_la_instruccion(self, rho_active, expected):
        """La conclusión del control vive en el informe, que es lo que se audita; el
        dashboard la muestra, no la deduce."""
        text = _activity_reading({
            "spearman_active": rho_active,
            "bootstrap_active": {"excludes_zero": False, "lo": -0.88, "hi": 0.23},
        })
        assert expected in text

    def test_sin_configuraciones_activas_no_se_inventa_un_rho(self):
        quiet = {c: 4 for c in self.CONFIGS}
        rows, real, synth = self._sides(quiet, quiet)
        out = activity_check(
            rows, real, synth, self.CONFIGS, alpha=DEFAULT_CVAR_ALPHA, min_trades_per_fold=5
        )
        assert out["n_active"] == 0
        assert out["spearman_active"] is None
        assert out["bootstrap_active"] is None


class TestBootstrap:
    CONFIGS = [f"c{i}" for i in range(8)]

    def _sides(self, agreement: bool):
        real_levels = {c: float(i) for i, c in enumerate(self.CONFIGS)}
        synth_levels = (
            dict(real_levels)
            if agreement
            else {c: float(len(self.CONFIGS) - i) for i, c in enumerate(self.CONFIGS)}
        )
        real = collect_side(
            _side_rows(SIDE_REAL, self.CONFIGS, ["w1", "w2", "w3"], real_levels),
            SIDE_REAL, self.CONFIGS,
        )
        synth = collect_side(
            _side_rows(SIDE_SYNTHETIC, self.CONFIGS, ["s1", "s2", "s3", "s4"], synth_levels),
            SIDE_SYNTHETIC, self.CONFIGS,
        )
        return real, synth

    def test_determinista_con_la_misma_semilla(self):
        real, synth = self._sides(agreement=True)
        kwargs = dict(alpha=DEFAULT_CVAR_ALPHA, n_samples=200, seed=7)
        first = block_bootstrap_rho(real, synth, self.CONFIGS, **kwargs)
        assert first == block_bootstrap_rho(real, synth, self.CONFIGS, **kwargs)

    def test_declara_cuantos_bloques_hay_en_cada_lado(self):
        real, synth = self._sides(agreement=True)
        out = block_bootstrap_rho(
            real, synth, self.CONFIGS, alpha=DEFAULT_CVAR_ALPHA, n_samples=100, seed=7
        )
        assert out["n_blocks_real"] == 3 and out["n_blocks_synthetic"] == 4

    def test_acuerdo_perfecto_da_intervalo_positivo(self):
        real, synth = self._sides(agreement=True)
        out = block_bootstrap_rho(
            real, synth, self.CONFIGS, alpha=DEFAULT_CVAR_ALPHA, n_samples=300, seed=7
        )
        assert out["lo"] > 0 and out["excludes_zero"] is True

    def test_orden_invertido_da_intervalo_negativo(self):
        real, synth = self._sides(agreement=False)
        out = block_bootstrap_rho(
            real, synth, self.CONFIGS, alpha=DEFAULT_CVAR_ALPHA, n_samples=300, seed=7
        )
        assert out["hi"] < 0 and out["excludes_zero"] is True

    def test_bootstrap_de_configuraciones_es_otra_pregunta_y_va_aparte(self):
        rewards = [float(i) for i in range(8)]
        out = config_bootstrap_rho(rewards, rewards, n_samples=200, seed=7)
        assert "configuraciones" in out["method"] and out["lo"] > 0

    def test_permutacion_detecta_el_acuerdo_perfecto(self):
        rewards = [float(i) for i in range(10)]
        out = permutation_p_value(rewards, rewards, n_samples=500, seed=7)
        assert out["observed"] == pytest.approx(1.0)
        assert 0 < out["p_value"] < 0.01  # nunca exactamente 0: (extremos+1)/(n+1)


# -------------------------------------------------------------- informe completo -----


def _plan_dict(symbols=("BTC/USDT",)) -> dict:
    return {
        "library_id": "ai_v3",
        "library_is_fallback": False,
        "symbols": list(symbols),
        "real": {
            "window": {"start": "2018-07-22", "end": "2026-01-01"},
            "sub_windows": [{"label": "w1"}, {"label": "w2"}, {"label": "w3"}],
            "head_discarded_days": 324,
            "min_history_days": 270,
            "symbols_omitted": [{"symbol": "NEW/USDT"}],
        },
        "synthetic": {"n_samples": 4, "scenario_ids": ["s1"], "n_paths": 1},
        "validation": {
            "scheme": "cpcv", "window_days": 544, "n_groups": 6, "n_test_groups": 2,
            "n_folds": 15, "purge_days": 10, "embargo_days": 5,
            "periods_per_year": 365, "cvar_alpha": DEFAULT_CVAR_ALPHA,
            "starting_equity": 10_000.0,
        },
    }


class _Spec:
    def __init__(self, config_id: str) -> None:
        self.id = config_id
        self.type = "crypto_momentum"
        self.params = {"fast_sma_window": 10}


class TestAnalyze:
    CONFIGS = [f"c{i}" for i in range(6)]

    def _rows(self, agreement: bool) -> list[dict]:
        real_levels = {c: float(i) for i, c in enumerate(self.CONFIGS)}
        synth_levels = (
            dict(real_levels)
            if agreement
            else {c: float(len(self.CONFIGS) - i) for i, c in enumerate(self.CONFIGS)}
        )
        return [
            *_side_rows(SIDE_REAL, self.CONFIGS, ["w1", "w2", "w3"], real_levels),
            *_side_rows(SIDE_SYNTHETIC, self.CONFIGS, ["s1", "s2"], synth_levels),
        ]

    def test_acuerdo_perfecto_transfiere(self):
        report = analyze(
            self._rows(agreement=True), _plan_dict(), [_Spec(c) for c in self.CONFIGS]
        )
        assert report["transfer"]["spearman"] == pytest.approx(1.0)
        assert report["verdict"]["key"] == VERDICT_TRANSFERS
        assert report["rankings"]["real"] == report["rankings"]["synthetic"]

    def test_orden_invertido_se_detecta_como_tal(self):
        report = analyze(
            self._rows(agreement=False), _plan_dict(), [_Spec(c) for c in self.CONFIGS]
        )
        assert report["transfer"]["spearman"] == pytest.approx(-1.0)
        assert report["verdict"]["key"] == VERDICT_INVERTED
        assert report["transfer"]["discrepancies"]["n_overrated_by_synthetic"] > 0

    def test_el_informe_declara_sus_limites(self):
        report = analyze(
            self._rows(agreement=True), _plan_dict(), [_Spec(c) for c in self.CONFIGS]
        )
        keys = {c["key"] for c in report["caveats"]}
        assert {
            "un_solo_camino", "sesgo_supervivencia", "historico_insuficiente",
            "inactividad_premiada",
        } <= keys
        superviv = next(c for c in report["caveats"] if c["key"] == "sesgo_supervivencia")
        # El sesgo tiene DIRECCION y hay que decirla: juega contra la hipotesis.
        assert "EN CONTRA" in superviv["text"]

    def test_determinista(self):
        rows, plan, specs = self._rows(True), _plan_dict(), [_Spec(c) for c in self.CONFIGS]
        first = analyze(rows, plan, specs)
        second = analyze(rows, plan, specs)
        first.pop("generated_at"), second.pop("generated_at")
        assert first == second

    def test_los_dos_lados_se_miden_sobre_el_mismo_material(self):
        rows = self._rows(agreement=True)
        # A c0 le falta un bloque real: no puede rankearse contra las demas.
        rows = [r for r in rows if not (r["config_id"] == "c0" and r["unit_id"] == "w2")]
        report = analyze(rows, _plan_dict(), [_Spec(c) for c in self.CONFIGS])
        assert "c0" in report["configs_dropped"]
        assert {c["config_id"] for c in report["configs"]} == set(self.CONFIGS) - {"c0"}

    def test_menos_de_tres_configuraciones_no_es_un_ranking(self):
        rows = [r for r in self._rows(True) if r["config_id"] in ("c0", "c1")]
        with pytest.raises(ValueError, match="ranking"):
            analyze(rows, _plan_dict(), [_Spec(c) for c in ("c0", "c1")])


class TestTaskPlan:
    def test_una_tarea_por_configuracion_y_unidad_agrupadas_por_unidad(self):
        class _Plan:
            real_windows = ({"label": "w1"}, {"label": "w2"})
            scenario_ids = ("s1", "s2")
            n_paths = 2

        tasks = build_tasks(_Plan(), ["a", "b"])
        assert len(tasks) == 2 * (2 + 2 * 2)
        # Agrupadas por unidad: el worker carga las barras una vez por muestra.
        assert tasks[0][2] == tasks[1][2] and tasks[0][0] != tasks[1][0]
        assert sum(1 for t in tasks if t[1] == SIDE_REAL) == 4


# ------------------------------------------------ contrato con la validacion ---------


class TestBaselineSeriesContract:
    """`transfer_study` agrega los baselines de varias validaciones en una sola cola, y
    para eso necesita sus scores fold a fold. Este test fija ese contrato nuevo."""

    def test_las_series_crudas_agregan_al_estadistico_publicado(self):
        bars = {"BTC/USDT": trending_df(500)}
        engine_config = make_config()
        result = validate_multiwindow(
            engine_config, None, FakeService(bars)._bars,
            datetime(2024, 3, 1, tzinfo=UTC), datetime(2025, 3, 1, tzinfo=UTC),
            scheme="cpcv", n_groups=4, n_test_groups=2,
            purge_days=5, compare_single_split=False,
        )
        assert result.baseline_scores, "el gate corrio pero no expuso las series crudas"
        for name, stats in result.baselines.items():
            series = result.baseline_scores[name]
            assert len(series) == len(result.folds)
            assert aggregate_reward(series).reward == pytest.approx(stats.reward)

    def test_el_spearman_del_estudio_es_el_del_resto_del_sistema(self):
        # No hay una segunda implementacion de Spearman en el repo: el estudio usa la del
        # modulo de calibracion. Si alguien la duplicara, este test dejaria de tener sentido.
        assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
