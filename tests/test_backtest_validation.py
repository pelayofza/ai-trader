"""
Tests de la validacion multiventana.

El nucleo son los de FUGA TEMPORAL (`TestNoTemporalLeakage` y
`TestNoLeakageInExecution`): la purga y el embargo son promesas, y una promesa sobre el
tiempo que nadie verifica es exactamente el tipo de cosa que hace que un backtest
parezca mejor de lo que es. Se comprueban en dos planos:

- **Geometria**: dia a dia, ningun fold comparte fechas entre train y test, y los huecos
  cumplen la purga y el embargo declarados.
- **Ejecucion**: el resultado de una ventana no cambia si se borran las barras
  posteriores (no se mira al futuro) ni si se cambia el train del fold (no hay estado
  que cruce el corte), que es lo que legitima cachear tramos entre folds.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import pytest

from ai_trader.backtest.engine import BacktestEngine
from ai_trader.backtest.metrics import EquityPoint, chain_equity_curves, compute_metrics
from ai_trader.backtest.validation import (
    MIN_BLOCK_DAYS,
    SCHEME_CPCV,
    SCHEME_WALK_FORWARD,
    Block,
    Fold,
    assert_no_leakage,
    build_folds,
    coverage,
    cpcv_folds,
    merge_adjacent,
    partition,
    purge_and_embargo,
    resolve_embargo_days,
    resolve_split_cutoff,
    single_split_folds,
    subtract,
    walk_forward_folds,
)
from ai_trader.scoring import validation_study
from ai_trader.scoring.multiwindow import resolve_purge_days, validate_multiwindow
from ai_trader.strategies.registry import build_strategy
from test_backtest_engine import FakeService, make_config, trending_df

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Ventana con margen para que cada bloque tenga dias de sobra en los tests de motor.
RUN_START = datetime(2024, 4, 1, tzinfo=timezone.utc)
RUN_END = datetime(2025, 4, 1, tzinfo=timezone.utc)


def days_of(blocks) -> set[datetime]:
    """Conjunto explicito de fechas cubiertas. Es la forma menos ingeniosa —y por eso la
    mas fiable— de preguntar si dos ventanas comparten un dia."""
    out: set[datetime] = set()
    for block in blocks:
        day = block.start
        while day < block.end:
            out.add(day)
            day += timedelta(days=1)
    return out


@pytest.fixture
def engine():
    bars = {"BTC/USDT": trending_df(600)}
    return BacktestEngine(make_config(), FakeService(bars), starting_equity=10_000.0)


class TestBlockAlgebra:
    """Los bloques son semiabiertos [start, end): es lo que hace que 'sin solape'
    signifique algo. La particion antigua era cerrada y por eso el dia del corte caia
    en train Y en test."""

    def test_partition_covers_the_range_exactly_once(self):
        groups = partition(START, END, 6)

        assert len(groups) == 6
        assert groups[0].start == START
        assert groups[-1].end == END
        # Sin huecos ni solapes: el final de cada grupo es el principio del siguiente.
        assert all(a.end == b.start for a, b in zip(groups, groups[1:]))
        assert sum(g.days for g in groups) == (END - START).days
        assert len(days_of(groups)) == (END - START).days

    def test_partition_is_deterministic(self):
        assert partition(START, END, 7) == partition(START, END, 7)

    def test_partition_refuses_groups_too_small_to_measure(self):
        with pytest.raises(ValueError, match="at least"):
            partition(START, START + timedelta(days=5), 4)

    def test_merge_adjacent_fuses_contiguous_blocks(self):
        a = Block(START, START + timedelta(days=10))
        b = Block(START + timedelta(days=10), START + timedelta(days=20))
        far = Block(START + timedelta(days=40), START + timedelta(days=50))

        assert merge_adjacent([far, a, b]) == (Block(a.start, b.end), far)

    def test_subtract_punches_a_hole_in_the_middle(self):
        block = Block(START, START + timedelta(days=30))
        hole = Block(START + timedelta(days=10), START + timedelta(days=20))

        assert subtract(block, (hole,)) == [
            Block(START, START + timedelta(days=10)),
            Block(START + timedelta(days=20), START + timedelta(days=30)),
        ]

    def test_purge_drops_fragments_too_short_to_produce_a_return(self):
        train = (Block(START, START + timedelta(days=30)),)
        # El test empieza en el dia 31: purgar 30 dias deja un fragmento de 1 dia.
        test = (Block(START + timedelta(days=31), START + timedelta(days=60)),)

        kept = purge_and_embargo(train, test, purge_days=30, embargo_days=0)

        assert kept == ()
        assert all(b.days >= MIN_BLOCK_DAYS for b in kept)


class TestNoTemporalLeakage:
    """La comprobacion central: ningun dia de test aparece en el train de su fold, y los
    huecos respetan la purga y el embargo declarados. Se mide sobre fechas explicitas,
    no sobre los mismos intervalos que construyeron el fold."""

    @pytest.mark.parametrize(
        "folds",
        [
            walk_forward_folds(START, END, n_folds=4, purge_days=10),
            walk_forward_folds(START, END, n_folds=3, purge_days=21, anchored=False),
            cpcv_folds(START, END, n_groups=6, n_test_groups=2, purge_days=10),
            cpcv_folds(START, END, n_groups=5, n_test_groups=1, purge_days=30, embargo_days=7),
            single_split_folds(START, END, purge_days=10),
        ],
        ids=["wf-anchored", "wf-rolling", "cpcv-6x2", "cpcv-5x1", "single"],
    )
    def test_train_and_test_never_share_a_calendar_day(self, folds):
        for fold in folds:
            assert not (days_of(fold.train) & days_of(fold.test)), fold.label
            assert fold.leakage().overlap_days == 0

    @pytest.mark.parametrize("purge", [0, 5, 15, 45])
    def test_the_gap_before_each_test_block_honours_the_purge(self, purge):
        for fold in cpcv_folds(START, END, purge_days=purge, embargo_days=0):
            for test in fold.test:
                for train in fold.train:
                    if train.end <= test.start:  # el train precede al test
                        assert (test.start - train.end).days >= purge, fold.label

    @pytest.mark.parametrize("embargo", [0, 3, 20])
    def test_the_gap_after_each_test_block_honours_the_embargo(self, embargo):
        folds = cpcv_folds(START, END, purge_days=0, embargo_days=embargo)
        # CPCV es donde el embargo muerde: hay train DESPUES del test.
        assert any(
            any(r.start >= t.end for r in f.train for t in f.test) for f in folds
        ), "el escenario no ejercita el embargo"

        for fold in folds:
            for test in fold.test:
                for train in fold.train:
                    if train.start >= test.end:  # el train sigue al test
                        assert (train.start - test.end).days >= embargo, fold.label

    def test_a_bigger_purge_removes_more_training_history(self):
        loose = walk_forward_folds(START, END, n_folds=4, purge_days=0, embargo_days=0)
        tight = walk_forward_folds(START, END, n_folds=4, purge_days=30, embargo_days=0)

        assert all(t.train_days < loose_f.train_days for t, loose_f in zip(tight, loose))
        # ...y no toca el test: purgar recorta entrenamiento, no ventana de evaluacion.
        assert [f.test_days for f in tight] == [f.test_days for f in loose]

    def test_walk_forward_tests_are_disjoint_and_move_forward(self):
        folds = walk_forward_folds(START, END, n_folds=5, purge_days=7)
        tests = [f.test[0] for f in folds]

        assert all(a.end <= b.start for a, b in zip(tests, tests[1:]))
        assert coverage(folds)["reuse_factor"] == 1.0

    def test_cpcv_reuses_each_stretch_in_several_contexts(self):
        folds = cpcv_folds(START, END, n_groups=6, n_test_groups=2, purge_days=10)

        assert len(folds) == 15  # C(6,2)
        assert coverage(folds)["reuse_factor"] == 5.0  # cada tramo entra en 5 folds

    def test_the_auditor_catches_a_leaky_fold(self):
        leaky = Fold(
            label="broken",
            scheme="manual",
            train=(Block(START, START + timedelta(days=100)),),
            test=(Block(START + timedelta(days=50), START + timedelta(days=150)),),
            purge_days=10,
            embargo_days=0,
        )

        assert not leaky.leakage().ok
        assert leaky.leakage().overlap_days == 50
        with pytest.raises(ValueError, match="Temporal leakage"):
            assert_no_leakage([leaky])

    def test_the_auditor_catches_an_insufficient_purge(self):
        # Sin solape, pero el train acaba 3 dias antes de un test que exige 10 de purga.
        tight = Fold(
            label="tight",
            scheme="manual",
            train=(Block(START, START + timedelta(days=97)),),
            test=(Block(START + timedelta(days=100), START + timedelta(days=150)),),
            purge_days=10,
            embargo_days=0,
        )

        assert tight.leakage().overlap_days == 0
        assert tight.leakage().purge_gap_days == 3
        with pytest.raises(ValueError, match="purga insuficiente"):
            assert_no_leakage([tight])

    def test_the_single_split_no_longer_shares_the_cutoff_day(self):
        """El corte antiguo entregaba [start, cutoff] y [cutoff, end]: el dia del corte
        estaba en las dos ventanas. Expresado como fold, ese solape desaparece."""
        (fold,) = single_split_folds(START, END, split_ratio=0.7)
        cutoff = resolve_split_cutoff(START, END, split_ratio=0.7)

        assert fold.train[-1].end == cutoff
        assert fold.test[0].start == cutoff
        assert not (days_of(fold.train) & days_of(fold.test))


class TestFoldPlans:
    def test_build_folds_dispatches_by_scheme(self):
        assert len(build_folds(START, END, scheme=SCHEME_WALK_FORWARD, n_folds=3)) == 3
        assert len(build_folds(START, END, scheme=SCHEME_CPCV, n_groups=5, n_test_groups=2)) == 10
        assert len(build_folds(START, END, scheme="single_split")) == 1

    def test_build_folds_rejects_an_unknown_scheme(self):
        with pytest.raises(ValueError, match="Unknown validation scheme"):
            build_folds(START, END, scheme="montecarlo")

    def test_the_default_embargo_is_one_percent_of_the_range(self):
        assert resolve_embargo_days(START, END) == 4  # 1% de 366 dias
        # Nunca decorativo: en rangos cortos sigue siendo al menos un dia.
        assert resolve_embargo_days(START, START + timedelta(days=20)) == 1

    def test_the_default_purge_is_the_holding_horizon(self):
        config = make_config()

        assert resolve_purge_days(config) == config.runner.max_holding_days


class TestChainedCurves:
    """Los tramos de un fold se COMPONEN por retornos, no se pegan: cada tramo arranca
    del mismo capital, asi que lo unico comparable entre ellos son sus retornos."""

    def _curve(self, day0: datetime, equities: list[float]) -> list[EquityPoint]:
        return [
            EquityPoint(day=day0 + timedelta(days=i), equity=e) for i, e in enumerate(equities)
        ]

    def test_returns_compound_across_blocks(self):
        first = self._curve(START, [100.0, 110.0])  # +10%
        second = self._curve(START + timedelta(days=90), [200.0, 220.0])  # +10%

        chained = chain_equity_curves([first, second])

        assert [round(p.equity, 6) for p in chained.points] == [100.0, 110.0, 121.0]

    def test_the_gap_between_blocks_is_not_counted_as_time_traded(self):
        first = self._curve(START, [100.0, 110.0])
        second = self._curve(START + timedelta(days=90), [200.0, 220.0])

        chained = chain_equity_curves([first, second])

        # Dos dias operados, aunque la curva abarque 91 dias de calendario.
        assert chained.active_days == 2
        assert (chained.points[-1].day - chained.points[0].day).days == 91

    def test_turnover_uses_days_traded_not_the_calendar_span(self, make_position):
        first = self._curve(START, [100.0, 110.0])
        second = self._curve(START + timedelta(days=90), [100.0, 110.0])
        chained = chain_equity_curves([first, second])
        trade = make_position(size=1.0, entry_price=50.0)
        trade.exit_price = 50.0

        honest = compute_metrics(
            chained.points, [trade], active_days=chained.active_days
        ).turnover
        diluted = compute_metrics(chained.points, [trade]).turnover

        # Contar los 89 dias de hueco diluiria la rotacion ~45x.
        assert honest > diluted
        assert honest == pytest.approx(100.0 / (100.0 * 2))

    def test_a_single_block_chains_to_itself(self):
        curve = self._curve(START, [100.0, 110.0, 99.0])

        chained = chain_equity_curves([curve])

        assert [round(p.equity, 6) for p in chained.points] == [100.0, 110.0, 99.0]
        assert chained.active_days == 2

    def test_chaining_nothing_is_an_error_not_a_zero(self):
        with pytest.raises(ValueError, match="no equity curves"):
            chain_equity_curves([])


class TestNoLeakageInExecution:
    """La geometria puede ser perfecta y el motor mirar al futuro igualmente. Estos dos
    tests cierran esa puerta y, de paso, justifican el cacheo de tramos entre folds."""

    def test_a_window_ignores_every_bar_after_its_last_day(self):
        """Si el motor se asomara al futuro, borrar las barras posteriores cambiaria el
        resultado. No cambia."""
        full = trending_df(600)
        block = Block(RUN_START, RUN_START + timedelta(days=120))
        fold = Fold("solo", "manual", (), (block,), purge_days=0, embargo_days=0)

        truncated = full[full.index <= block.last_day]
        with_future = BacktestEngine.from_bars(make_config(), {"BTC/USDT": full}, 10_000.0)
        without_future = BacktestEngine.from_bars(
            make_config(), {"BTC/USDT": truncated}, 10_000.0
        )

        a = with_future.run_folds([fold])[0]
        b = without_future.run_folds([fold])[0]

        assert [round(p.equity, 6) for p in a.test.equity_curve] == [
            round(p.equity, 6) for p in b.test.equity_curve
        ]
        assert a.headline_score == pytest.approx(b.headline_score)

    def test_the_training_window_does_not_touch_the_test_window(self, engine):
        """Dentro de un backtest no se ajusta nada: dos folds con el MISMO test y trains
        distintos tienen que dar exactamente el mismo resultado OOS. Es la propiedad que
        legitima reusar un tramo ya corrido en otro fold."""
        test = Block(RUN_START + timedelta(days=200), RUN_START + timedelta(days=290))
        short = Fold(
            "short", "manual",
            (Block(RUN_START, RUN_START + timedelta(days=60)),), (test,), 10, 0,
        )
        long = Fold(
            "long", "manual",
            (Block(RUN_START, RUN_START + timedelta(days=180)),), (test,), 10, 0,
        )

        results = engine.run_folds([short, long], run_train=True)

        assert results[0].train.metrics.num_trades != results[1].train.metrics.num_trades
        assert results[0].headline_score == pytest.approx(results[1].headline_score)

    def test_purging_changes_the_training_side_and_only_the_training_side(self, engine):
        """Lo que la purga hace, y lo que NO hace, escrito como invariante.

        Purgar recorta el train, asi que la ventana in-sample cambia. La ventana OOS es
        identica, y tiene que serlo: si purgar moviera el score out-of-sample, seria
        senal de que el train se estaba filtrando en el test por alguna via.

        Corolario incomodo pero honesto: mientras dentro del backtest no se ajuste nada,
        purgar no mejora ninguna cifra publicada. Es la geometria correcta para cuando
        algo SI se ajuste sobre el train (el bucle exterior, una politica aprendida), y
        para que la referencia in-sample que se reporta no este contaminada."""
        loose, tight = (
            walk_forward_folds(
                RUN_START, RUN_START + timedelta(days=330), n_folds=3,
                purge_days=purge, embargo_days=0,
            )
            for purge in (0, 40)
        )

        a = engine.run_folds(loose, run_train=True)
        b = engine.run_folds(tight, run_train=True)

        assert [r.headline_score for r in a] == [r.headline_score for r in b]
        assert [r.train_headline_score for r in a] != [r.train_headline_score for r in b]

    def test_the_engine_refuses_a_leaky_plan_before_spending_compute(self, engine):
        leaky = Fold(
            "broken", "manual",
            (Block(RUN_START, RUN_START + timedelta(days=200)),),
            (Block(RUN_START + timedelta(days=100), RUN_START + timedelta(days=250)),),
            purge_days=10, embargo_days=0,
        )

        with pytest.raises(ValueError, match="Temporal leakage"):
            engine.run_folds([leaky])

    def test_running_no_folds_is_an_error(self, engine):
        with pytest.raises(ValueError, match="at least one fold"):
            engine.run_folds([])

    def test_a_fold_without_a_test_window_is_rejected_at_construction(self):
        """No a mitad de ejecucion, cuando ya se han gastado minutos de backtest."""
        with pytest.raises(ValueError, match="no test block"):
            Fold("empty", "manual", (Block(RUN_START, RUN_END),), (), 0, 0)


class TestMultiWindowAggregation:
    """El corte unico daba UN numero, y el CVaR de un solo numero es ese numero. Con
    varias ventanas el estadistico robusto por fin tiene sobre que ser robusto."""

    @pytest.fixture
    def bars(self):
        return {"BTC/USDT": trending_df(600)}

    def _validate(self, bars, **kwargs):
        return validate_multiwindow(
            make_config(), None, bars, RUN_START, RUN_END, with_baselines=False, **kwargs
        )

    def test_walk_forward_produces_one_score_per_window(self, bars):
        result = self._validate(bars, scheme=SCHEME_WALK_FORWARD, n_folds=4)

        assert len(result.folds) == 4
        assert result.stats.n == 4
        assert result.leakage_ok
        assert result.purge_days == make_config().runner.max_holding_days

    def test_cpcv_produces_one_score_per_combination(self, bars):
        result = self._validate(bars, scheme=SCHEME_CPCV, n_groups=5, n_test_groups=2)

        assert len(result.folds) == 10  # C(5,2)
        assert result.stats.n == 10
        assert result.coverage["reuse_factor"] == 4.0

    def test_the_reward_is_the_bad_tail_of_the_windows(self, bars):
        result = self._validate(bars, scheme=SCHEME_WALK_FORWARD, n_folds=4)
        scores = sorted(result.scores)

        # CVaR@25% de 4 ventanas = la peor.
        assert result.stats.reward == pytest.approx(scores[0])
        assert result.stats.reward <= result.stats.mean

    def test_the_distribution_exposes_dispersion_a_single_split_cannot(self, bars):
        result = self._validate(bars, scheme=SCHEME_CPCV, n_groups=6, n_test_groups=2)

        assert result.stats.std > 0
        assert result.stats.worst < result.stats.best
        # La comparacion con el corte antiguo se reporta, no se esconde.
        assert result.single_split_score is not None
        assert result.optimism == pytest.approx(result.single_split_score - result.median)

    def test_folds_with_two_stretches_are_scored_as_one_chained_window(self, bars):
        result = self._validate(bars, scheme=SCHEME_CPCV, n_groups=6, n_test_groups=2)
        split = [f for f in result.folds if f.n_test_blocks == 2]

        assert split, "CPCV deberia producir folds con tramos no contiguos"
        for fold in split:
            assert fold.oos_observations > 0
            assert fold.num_trades > 0

    def test_two_identical_validations_agree_to_the_last_decimal(self, bars):
        a = self._validate(bars, scheme=SCHEME_CPCV, n_groups=5, n_test_groups=2)
        b = self._validate(bars, scheme=SCHEME_CPCV, n_groups=5, n_test_groups=2)

        assert a.scores == b.scores
        assert a.stats.as_dict() == b.stats.as_dict()

    def test_the_scheme_does_not_change_the_world_being_evaluated(self, bars):
        """Los esquemas tienen que ver el MISMO mundo: si cada uno evaluara un rango
        distinto, la comparacion de optimismo mediria el rango, no el esquema. Lo que si
        difiere es CUANTO de ese rango llega a puntuarse."""
        wf = self._validate(bars, scheme=SCHEME_WALK_FORWARD, n_folds=5)
        cpcv = self._validate(bars, scheme=SCHEME_CPCV, n_groups=6, n_test_groups=2)

        # Misma referencia de corte unico = mismas barras y misma ventana global.
        assert wf.single_split_score == pytest.approx(cpcv.single_split_score)
        # Ambos llegan al final del rango...
        assert wf.folds[-1].test_end == cpcv.folds[-1].test_end
        # ...pero el walk-forward sacrifica el primer grupo como train inicial, asi que
        # empieza a puntuar mas tarde y cubre menos calendario. Es el precio de que cada
        # ventana tenga pasado propio, y CPCV es justo lo que lo evita.
        assert wf.folds[0].test_start > cpcv.folds[0].test_start
        assert (
            wf.coverage["calendar_days_covered"] < cpcv.coverage["calendar_days_covered"]
        )

    def test_baselines_are_measured_in_the_very_same_windows(self, bars):
        result = validate_multiwindow(
            make_config(), None, bars, RUN_START, RUN_END,
            scheme=SCHEME_WALK_FORWARD, n_folds=4, with_baselines=True,
        )

        assert result.baseline_gate is not None
        # Un baseline por fold, exactamente: mismas ventanas, misma n.
        for stats in result.baselines.values():
            assert stats.n == len(result.folds)
        assert "btc_hold" in result.baselines


class TestStudyConfigsAreConstructible:
    """Las 16 configuraciones del estudio se construyen. Parece obvio y no lo es: son
    diccionarios escritos A MANO que no pasan por `finalize`, asi que nada las obligaba a
    ser coherentes. El hipercubo tiene quien lo defienda —`test_scoring` exige que todo
    vector dentro de rango construya— pero estas no venian de ningun vector.

    Y el precio de no tenerlo se pago entero: una variante con `trend_window` subido y
    `min_bars` sin subir tumbo el estudio a los veinte minutos y en la unidad 9 de 128,
    despues de que seis workers hubieran hecho su trabajo para nada. El fallo tardaba
    veinte minutos en aparecer y tarda milisegundos en descartarse."""

    @pytest.mark.parametrize(
        ("config_id", "family", "params"),
        [(c[0], c[1], c[2]) for c in validation_study.STUDY_CONFIGS],
        ids=[c[0] for c in validation_study.STUDY_CONFIGS],
    )
    def test_every_study_config_builds(self, config_id, family, params):
        build_strategy(family, dict(params), strategy_id=config_id)

    def test_every_family_of_the_grid_is_represented(self):
        # Si alguien anade una familia a la rejilla y se olvida de este estudio, el
        # esquema de validacion se seguiria publicando sobre las de antes sin avisar.
        from ai_trader.scoring.families import FAMILIES

        assert {c[1] for c in validation_study.STUDY_CONFIGS} == set(FAMILIES)


class TestStudyAnalysis:
    """La aritmetica del informe publicado, sin correr un solo backtest: si el resumen
    miente, las cifras de la documentacion mienten con el."""

    def _row(self, config_id: str, scenario: str, single: float, wf: list[float],
             cpcv: list[float]) -> dict:
        def scheme(scores: list[float]) -> dict:
            ordered = sorted(scores)
            return {
                "reward": ordered[0],
                "mean": sum(scores) / len(scores),
                "median": statistics.median(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "worst": ordered[0],
                "best": ordered[-1],
                "n_folds": len(scores),
                "leakage_ok": True,
                "approved": False,
                "embargo_days": 4,
                "scores": scores,
            }

        return {
            "config_id": config_id,
            "scenario_id": scenario,
            "path_index": 0,
            "single_score": single,
            "walk_forward": scheme(wf),
            "cpcv": scheme(cpcv),
        }

    def test_optimism_is_the_paired_gap_against_the_window_median(self):
        rows = [
            self._row("a", "s1", 3.0, [1.0, 1.0, 1.0], [1.0]),
            self._row("b", "s1", 2.0, [0.0, 0.0, 0.0], [0.0]),
        ]

        report = validation_study.analyze(rows, {"scenario_ids": ["s1"], "n_paths": 1})

        # gaps 2.0 y 2.0 -> mediana 2.0; contra la cola (peor ventana), lo mismo aqui.
        assert report["optimism"]["walk_forward"]["median"] == pytest.approx(2.0)
        assert report["optimism"]["vs_tail"]["median"] == pytest.approx(2.0)

    def test_a_reordering_counts_as_a_decision_flip(self):
        # El corte unico corona a 'a'; las ventanas coronan a 'c'.
        rows = [
            self._row("a", "s1", 5.0, [0.0], [0.0]),
            self._row("b", "s1", 3.0, [1.0], [1.0]),
            self._row("c", "s1", 1.0, [2.0], [2.0]),
        ]

        report = validation_study.analyze(rows, {"scenario_ids": ["s1"], "n_paths": 1})

        assert report["decision_flips"]["walk_forward"] == 1
        assert report["decision_flips"]["n_samples"] == 1
        # Orden exactamente invertido -> acuerdo de rangos -1.
        assert report["rank_agreement"]["walk_forward"]["median"] == pytest.approx(-1.0)

    def test_an_unchanged_ranking_is_not_a_flip(self):
        rows = [
            self._row("a", "s1", 5.0, [2.0], [2.0]),
            self._row("b", "s1", 3.0, [1.0], [1.0]),
            self._row("c", "s1", 1.0, [0.0], [0.0]),
        ]

        report = validation_study.analyze(rows, {"scenario_ids": ["s1"], "n_paths": 1})

        assert report["decision_flips"]["walk_forward"] == 0
        assert report["rank_agreement"]["walk_forward"]["median"] == pytest.approx(1.0)

    def test_signal_vs_noise_compares_configs_against_windows(self):
        """Cuanto separa a dos configuraciones frente a cuanto separa a dos ventanas de la
        misma configuracion. Si el ruido temporal domina, elegir con una sola ventana es
        elegir por el tramo de historia que toco."""
        rows = [
            # Dos configs separadas por 1.0 en reward; cada una oscila 4.0 entre ventanas.
            self._row("a", "s1", 5.0, [3.0, 7.0], [3.0]),
            self._row("b", "s1", 4.0, [2.0, 6.0], [2.0]),
        ]

        svn = validation_study.analyze(
            rows, {"scenario_ids": ["s1"], "n_paths": 1}
        )["signal_vs_noise"]

        assert svn["config_spread_walk_forward"]["median"] == pytest.approx(1.0)
        assert svn["window_range_walk_forward"]["median"] == pytest.approx(4.0)
        assert svn["ratio"] == pytest.approx(4.0)

    def test_the_leakage_audit_is_reported_not_assumed(self):
        rows = [self._row("a", "s1", 1.0, [1.0, 2.0], [1.0])]
        rows[0]["cpcv"]["leakage_ok"] = False

        report = validation_study.analyze(rows, {"scenario_ids": ["s1"], "n_paths": 1})

        assert report["leakage"]["clean"] is False
        assert report["leakage"]["folds_audited"] == 3

    def test_the_summary_degrades_instead_of_inventing_numbers(self):
        assert validation_study._spread([])["median"] is None
        assert validation_study._spread([float("nan"), 1.0])["n"] == 1

    def test_quantiles_interpolate_between_order_statistics(self):
        assert validation_study._quantile([0.0, 1.0, 2.0, 3.0], 0.5) == pytest.approx(1.5)
        assert validation_study._quantile([5.0], 0.75) == pytest.approx(5.0)
