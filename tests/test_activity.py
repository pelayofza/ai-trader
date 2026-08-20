"""
Tests del suelo de actividad: el requisito de ELEGIBILIDAD del ranking.

Lo que hay que blindar aqui no es "que el dataclass sume bien", sino las cuatro
propiedades sin las cuales el suelo no significaria nada:

- **La degeneracion existe y es aritmetica.** Una ventana sin operaciones puntua 0 EXACTO
  y el CVaR de una lista de ceros es 0, que gana a cualquier cosa que arriesgue y pierda.
  El test lo reproduce con el motor de metricas real, no con prosa.
- **La actividad viaja con la recompensa, o no viaja.** `aggregate_reward` empareja scores
  y operaciones ventana a ventana y se niega a emparejar conjuntos distintos.
- **El suelo es elegibilidad, no penalizacion.** Una configuracion que no lo supera
  conserva su recompensa intacta; lo unico que pierde es competir y aprobar el gate.
- **Los dos numeros del suelo estan atados a su evidencia** (`data/activity/`), asi que
  moverlos obliga a re-correr el estudio y republicarla.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_trader.backtest.metrics import EquityPoint, compute_metrics, headline_score
from ai_trader.scoring.activity import (
    DEFAULT_ACTIVITY_FLOOR,
    MAX_ZERO_WINDOW_PCT,
    MIN_MEDIAN_TRADES_PER_WINDOW,
    ActivityFloor,
    ActivityStats,
    eligibility_dict,
    measure_activity,
)
from ai_trader.research.activity_study import (
    THRESHOLD_GRID,
    activity_report_path,
    choose_threshold,
    load_activity_report,
)
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, aggregate_reward
from ai_trader.scoring.baselines import BASELINE_BTC, BASELINE_SPY, gate

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_LIBRARY = "ai_v3"


def _curve(values: list[float]) -> list[EquityPoint]:
    from datetime import datetime, timedelta, timezone

    day0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [EquityPoint(day=day0 + timedelta(days=i), equity=v) for i, v in enumerate(values)]


class TestLaDegeneracionQueMotivaElSuelo:
    """La aritmetica del problema, reproducida con el motor real."""

    def test_una_ventana_sin_operaciones_puntua_cero_exacto(self):
        flat = compute_metrics(_curve([1000.0] * 20), [])

        assert flat.sharpe == 0.0 and flat.turnover == 0.0 and flat.max_drawdown_pct == 0.0
        assert headline_score(flat) == 0.0  # exacto, no aproximado

    def test_no_jugar_le_gana_a_jugar_y_perder(self):
        """El motivo entero del suelo: sin el, el ranking ordena por inactividad."""
        idle = aggregate_reward([0.0] * 8, trades=[0] * 8)
        risky = aggregate_reward([1.5, -1.0, 0.8, -1.4, 0.9, -0.7, 1.1, -0.9], trades=[9] * 8)

        assert idle.reward > risky.reward
        assert idle.mean < risky.mean  # y ademas rinde menos en el centro
        assert not DEFAULT_ACTIVITY_FLOOR.eligible(idle.activity)
        assert DEFAULT_ACTIVITY_FLOOR.eligible(risky.activity)


class TestActivityStats:
    def test_resume_las_ventanas_una_a_una(self):
        stats = measure_activity([0, 4, 6, 10])

        assert stats.n_windows == 4 and stats.trades == 20
        assert stats.trades_per_window == 5.0
        assert stats.median_trades_per_window == 5.0
        assert stats.zero_windows == 1 and stats.zero_window_pct == 25.0

    def test_sin_ventanas_no_es_cero_es_no_medido(self):
        """Distinguirlo importa: 'no opero' es un hecho, 'no se midio' es una laguna."""
        stats = measure_activity([])

        assert stats.n_windows == 0 and not stats.measured
        assert not DEFAULT_ACTIVITY_FLOOR.eligible(stats)
        assert DEFAULT_ACTIVITY_FLOOR.reasons(stats) == ("actividad no medida",)


class TestActivityFloor:
    def test_las_dos_condiciones_son_necesarias(self):
        rafagas = measure_activity([0, 0, 0, 30, 30, 30, 30, 30])  # mediana alta, 37% vacias
        goteo = measure_activity([2] * 8)  # ninguna vacia, pero opera casi nada

        assert rafagas.median_trades_per_window >= MIN_MEDIAN_TRADES_PER_WINDOW
        assert not DEFAULT_ACTIVITY_FLOOR.eligible(rafagas)
        assert goteo.zero_window_pct == 0.0
        assert not DEFAULT_ACTIVITY_FLOOR.eligible(goteo)
        assert DEFAULT_ACTIVITY_FLOOR.eligible(measure_activity([5] * 8))

    def test_el_motivo_del_rechazo_se_declara(self):
        reasons = DEFAULT_ACTIVITY_FLOOR.reasons(measure_activity([0, 0, 1, 1]))

        assert len(reasons) == 2
        assert "ventana mediana" in reasons[0] and "vacias" in reasons[1]

    def test_la_tolerancia_de_ventanas_vacias_ES_alpha(self):
        """No es un numero nuevo: por encima de alpha, el cuartil que promedia el CVaR
        puede estar hecho de ceros estructurales."""
        assert MAX_ZERO_WINDOW_PCT == pytest.approx(DEFAULT_CVAR_ALPHA * 100.0)

    def test_el_suelo_no_toca_la_recompensa(self):
        """Elegibilidad != penalizacion. Es la diferencia con lambda, que si resta."""
        quiet = aggregate_reward([0.4, 0.0, 0.0, 0.0], trades=[1, 0, 0, 0])

        assert quiet.reward == pytest.approx(aggregate_reward([0.4, 0.0, 0.0, 0.0]).reward)
        assert not DEFAULT_ACTIVITY_FLOOR.eligible(quiet.activity)

    def test_el_veredicto_publicado_lleva_suelo_y_motivos(self):
        out = eligibility_dict(measure_activity([0, 0, 0, 9]))

        assert out["rankable"] is False
        assert out["floor"]["min_median_trades_per_window"] == MIN_MEDIAN_TRADES_PER_WINDOW
        assert out["reasons"] and out["activity"]["zero_window_pct"] == 75.0


class TestRewardStatsLlevaLaActividad:
    def test_la_actividad_sale_en_el_diccionario_publicado(self):
        stats = aggregate_reward([1.0, -1.0, 0.5, 0.0], trades=[7, 8, 0, 9])

        assert stats.as_dict()["activity"]["zero_windows"] == 1
        assert stats.as_dict()["activity"]["n_windows"] == 4

    def test_emparejar_ventanas_distintas_es_un_error_no_un_aviso(self):
        with pytest.raises(ValueError, match="mismas ventanas"):
            aggregate_reward([1.0, 2.0, 3.0], trades=[5, 5])

    def test_sin_operaciones_la_actividad_es_none_no_cero(self):
        assert aggregate_reward([1.0, 2.0]).activity is None


class TestGateExigeActividad:
    """Batir a los baselines con una curva plana es batirlos por no jugar."""

    BASELINES = {BASELINE_BTC: [-1.4] * 8, BASELINE_SPY: [-1.3] * 8}

    def test_una_curva_plana_bate_a_los_pasivos_y_aun_asi_no_aprueba(self):
        verdict = gate([0.0] * 8, self.BASELINES, trades=[0] * 8)

        assert verdict.beats_baselines is True  # el veredicto antiguo decia que si
        assert verdict.eligible is False
        assert verdict.approved is False
        assert verdict.ineligible_reasons

    def test_una_estrategia_que_opera_y_gana_si_aprueba(self):
        verdict = gate([0.5] * 8, self.BASELINES, trades=[9] * 8)

        assert verdict.approved and verdict.beats_baselines and verdict.eligible

    def test_el_suelo_no_puede_rescatar_a_quien_no_bate_a_los_pasivos(self):
        verdict = gate([-2.0] * 8, self.BASELINES, trades=[9] * 8)

        assert verdict.eligible and not verdict.beats_baselines and not verdict.approved

    def test_sin_actividad_medida_el_gate_no_da_por_buena_la_elegibilidad(self):
        """Un requisito que no se ha comprobado no se concede: se declara sin comprobar."""
        verdict = gate([0.5] * 8, self.BASELINES)

        assert verdict.beats_baselines and not verdict.approved
        assert verdict.activity_checked is False
        assert verdict.as_dict()["ineligible_reasons"] == ["actividad no medida"]

    def test_el_veredicto_se_publica_descompuesto(self):
        published = gate([0.0] * 8, self.BASELINES, trades=[0] * 8).as_dict()

        assert published["approved"] is False and published["beats_baselines"] is True
        assert published["activity"]["zero_window_pct"] == 100.0
        assert published["activity_floor"]["max_zero_window_pct"] == MAX_ZERO_WINDOW_PCT

    def test_el_suelo_es_un_parametro_no_una_constante_escondida(self):
        laxo = ActivityFloor(min_median_trades_per_window=1.0, max_zero_window_pct=90.0)
        verdict = gate([0.0] * 8, self.BASELINES, trades=[1] * 8, activity_floor=laxo)

        assert verdict.approved is True


class TestDecisionRule:
    def test_menos_desacuerdos_gana_y_a_igualdad_el_umbral_mayor(self):
        rows = [
            {"threshold": 1.0, "disagreements": 3},
            {"threshold": 3.0, "disagreements": 1},
            {"threshold": 5.0, "disagreements": 1},
            {"threshold": 8.0, "disagreements": 4},
        ]
        out = choose_threshold(rows)

        assert out["chosen"] == 5.0 and out["disagreements"] == 1
        assert out["tied"] == [3.0, 5.0]

    def test_el_umbral_publicado_esta_en_la_rejilla_barrida(self):
        assert MIN_MEDIAN_TRADES_PER_WINDOW in THRESHOLD_GRID


class TestElSueloEstaAtadoASuEvidencia:
    """
    Congela los dos numeros Y los ata al informe que los justifica.

    Un test que solo comprobara las constantes seria un espejo del codigo: pasaria igual
    si alguien las cambiara a ojo. Estos exigen que el estudio publicado
    (`data/activity/`) haya elegido EXACTAMENTE ese umbral con su regla declarada, asi que
    moverlo obliga a re-correr el estudio y republicar la evidencia.
    """

    @pytest.fixture(scope="class")
    def report(self) -> dict:
        published = load_activity_report(ROOT / activity_report_path(ACTIVITY_LIBRARY))
        assert published is not None, (
            "Falta la evidencia del suelo de actividad en data/activity/. "
            "Regenerala con: python -m ai_trader.research.activity_study"
        )
        return published

    def test_los_numeros_del_suelo_estan_congelados(self):
        assert DEFAULT_ACTIVITY_FLOOR.min_median_trades_per_window == 3.0
        assert DEFAULT_ACTIVITY_FLOOR.max_zero_window_pct == 25.0

    def test_la_regla_declarada_elige_el_umbral_publicado(self, report):
        assert report["decision"]["chosen"] == MIN_MEDIAN_TRADES_PER_WINDOW
        assert report["decision"]["matches_published"] is True

    def test_la_aritmetica_afirmada_se_cumple_en_los_datos_reales(self, report):
        """'Sin operaciones -> headline 0 exacto' es una afirmacion comprobable, y el
        informe la comprueba en los dos mundos en vez de razonarla."""
        for side in ("real", "synthetic"):
            assert report["mechanism"]["sides"][side]["holds"] is True

    def test_el_estudio_midio_sobre_material_suficiente(self, report):
        assert report["source"]["n_configs"] >= 8
        assert report["source"]["blocks"]["real"] >= 3
        assert report["mechanism"]["sides"]["real"]["windows"] >= 200

    def test_la_estabilidad_no_se_uso_para_elegir_y_se_explica_por_que(self, report):
        """El control que parecia la metrica obvia y habria elegido justo lo contrario."""
        assert report["reproducibility"]["used_as_criterion"] is False
        assert report["reproducibility"]["sides"]["real"]["inactivity_inflates_stability"]

    def test_el_umbral_elegido_no_es_el_que_sostiene_la_particion(self, report):
        """Lo que se publica no depende del decimal: varios umbrales de la rejilla dan el
        MISMO conjunto rankeable en el lado real. Si algun dia deja de ser cierto, la
        eleccion del numero pasa a ser una decision de verdad y hay que rehacerla."""
        same = report["band"]["real"]["same_partition"]

        assert MIN_MEDIAN_TRADES_PER_WINDOW in same
        assert len(same) > 1


class TestEfectoMedidoSobreElGate:
    """La pregunta (c): cuantas configuraciones dejan de aprobar al exigir actividad."""

    @pytest.fixture(scope="class")
    def report(self) -> dict:
        published = load_activity_report(ROOT / activity_report_path(ACTIVITY_LIBRARY))
        assert published is not None
        return published

    def test_el_informe_publica_los_dos_veredictos_y_su_diferencia(self, report):
        real = report["gate"]["real"]

        assert set(real["lost"]) == set(real["approved_without_floor"]) - set(
            real["approved_with_floor"]
        )
        assert real["n_lost"] == len(real["lost"])

    def test_las_que_pierden_la_aprobacion_eran_inactivas(self, report):
        """Si alguna de las que caen operara de verdad, el suelo estaria filtrando otra
        cosa distinta de la que dice filtrar."""
        for lost in report["gate"]["real"]["lost_detail"]:
            assert lost["reasons"]
            assert (
                lost["trades_per_window"] < MIN_MEDIAN_TRADES_PER_WINDOW
                or lost["zero_window_pct"] > MAX_ZERO_WINDOW_PCT
            )


class TestActivityStatsContract:
    def test_el_porcentaje_de_vacias_no_divide_por_cero(self):
        assert ActivityStats(0, 0, 0.0, 0.0, 0).zero_window_pct == 0.0
