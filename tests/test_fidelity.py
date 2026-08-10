"""
Tests del harness de fidelidad sintetico-vs-real.

Todo lo que se comprueba aqui es NUMERICO y sin red: las series se construyen con
propiedades conocidas (AR(1) de signo dado, colas t-Student, volatilidad agrupada) y se
exige que el medidor las recupere; el estudio se ejercita contra un proveedor de barras
falso. Si estos tests pasan, el informe publicado mide lo que dice medir.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.synthetic.fidelity import (
    Estimate,
    aggregate_facts,
    aggregate_pairs,
    autocorrelation,
    compare,
    compare_cross_correlations,
    compare_metrics,
    load_fidelity_report,
    log_returns,
    pair_correlations,
    pair_key,
    series_facts,
)
from ai_trader.synthetic.fidelity_study import (
    CachedBarsProvider,
    FactsCollector,
    StudyPlan,
    build_report,
    calendar_windows,
    crypto_symbols,
    fetch_real_bars,
    measure_real,
    returns_frame,
    run_study,
)

DAY = pd.Timedelta(days=1)


# ----------------------------------------------------------------- utilidades --------


def _index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC", name="timestamp")


def _bars(returns: np.ndarray, start: str = "2020-01-01", price: float = 100.0):
    """OHLCV canonico a partir de una serie de retornos log."""
    closes = price * np.exp(np.cumsum(np.concatenate(([0.0], returns))))
    index = _index(len(closes), start)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(len(closes), 1_000.0),
        },
        index=index,
    )


def _iid(n: int, seed: int, scale: float = 0.02) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n) * scale


def _ar1(n: int, phi: float, seed: int, scale: float = 0.02) -> np.ndarray:
    noise = np.random.default_rng(seed).standard_normal(n) * scale
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def _clustered(n: int, seed: int, persistence: float = 0.9) -> np.ndarray:
    """Retornos con volatilidad agrupada (GARCH-like): la escala sigue un AR(1)."""
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.02)
    for i in range(1, n):
        vol[i] = np.exp(persistence * np.log(vol[i - 1]) + (1 - persistence) * np.log(0.02)
                        + 0.25 * rng.standard_normal())
    return vol * rng.standard_normal(n)


class TestAutocorrelation:
    def test_recovers_the_sign_and_size_of_an_ar1(self):
        assert autocorrelation(_ar1(4_000, 0.4, seed=1), 1) == pytest.approx(0.4, abs=0.05)
        assert autocorrelation(_ar1(4_000, -0.4, seed=2), 1) == pytest.approx(-0.4, abs=0.05)

    def test_iid_returns_are_uncorrelated(self):
        assert autocorrelation(_iid(4_000, seed=3), 1) == pytest.approx(0.0, abs=0.05)

    def test_constant_or_short_series_give_nan_not_zero(self):
        # NaN se propaga y se filtra; un 0 se leeria como "medido y no hay estructura".
        assert np.isnan(autocorrelation(np.ones(500), 1))
        assert np.isnan(autocorrelation(np.array([1.0, 2.0]), 5))

    def test_rejects_a_non_positive_lag(self):
        with pytest.raises(ValueError):
            autocorrelation(_iid(100, seed=4), 0)


class TestSeriesFacts:
    def test_fat_tails_raise_kurtosis_and_exceedances(self):
        rng = np.random.default_rng(11)
        gauss = series_facts(rng.standard_normal(4_000) * 0.02)
        student = series_facts(rng.standard_t(4, size=4_000) * 0.01)

        assert student.excess_kurtosis > gauss.excess_kurtosis + 2
        assert student.exceed_3sigma_pct > gauss.exceed_3sigma_pct

    def test_gaussian_exceedances_land_near_the_theoretical_027_pct(self):
        facts = series_facts(_iid(20_000, seed=12))
        assert facts.exceed_3sigma_pct == pytest.approx(0.27, abs=0.15)

    def test_clustering_shows_up_in_the_autocorrelation_of_absolute_returns(self):
        clustered = series_facts(_clustered(4_000, seed=13))
        flat = series_facts(_iid(4_000, seed=14))

        assert clustered.ac_abs1 > 0.15
        assert clustered.ac_abs_mean > flat.ac_abs_mean + 0.1

    def test_annualised_volatility_matches_the_input_scale(self):
        facts = series_facts(_iid(8_000, seed=15, scale=0.02))
        assert facts.vol_annual_pct == pytest.approx(2.0 * np.sqrt(365.0), rel=0.05)

    def test_returns_none_when_there_is_not_enough_evidence(self):
        assert series_facts(_iid(50, seed=16)) is None
        assert series_facts(np.zeros(500)) is None

    def test_rejects_a_lag_window_that_measures_nothing(self):
        with pytest.raises(ValueError):
            series_facts(_iid(500, seed=17), max_lag=0)

    def test_log_returns_ignore_non_positive_prices(self):
        assert len(log_returns([100.0, 0.0, 110.0, -5.0])) == 1
        assert len(log_returns([100.0])) == 0


class TestPairCorrelations:
    def test_identical_series_correlate_perfectly_and_the_key_is_sorted(self):
        values = _iid(500, seed=21)
        frame = pd.DataFrame({"B": values, "A": values}, index=_index(500))

        pairs = pair_correlations(frame)
        assert pairs == {pair_key("A", "B"): pytest.approx(1.0)}

    def test_pairs_without_enough_overlap_are_dropped(self):
        frame = pd.DataFrame(
            {"A": _iid(500, seed=22), "B": np.concatenate((np.full(450, np.nan), _iid(50, 23)))},
            index=_index(500),
        )
        assert pair_correlations(frame, min_overlap=120) == {}
        assert pair_key("A", "B") in pair_correlations(frame, min_overlap=20)

    def test_a_single_column_has_no_pairs(self):
        assert pair_correlations(pd.DataFrame({"A": _iid(300, seed=24)})) == {}


class TestEstimate:
    def test_summarises_a_set_of_measurements(self):
        est = Estimate(median=1.0, p10=0.5, p90=1.5, n=10)
        assert est.contains(0.7) and est.contains(1.5)
        assert not est.contains(1.6)
        assert not est.contains(float("nan"))

    def test_aggregation_uses_the_median_and_drops_nan(self):
        facts = [
            series_facts(_iid(400, seed=s), min_observations=200) for s in (31, 32, 33)
        ]
        aggregated = aggregate_facts(facts)
        assert aggregated["ac1"].n == 3
        assert aggregated["ac1"].p10 <= aggregated["ac1"].median <= aggregated["ac1"].p90

    def test_pair_aggregation_groups_by_pair(self):
        aggregated = aggregate_pairs([{"A|B": 0.2}, {"A|B": 0.6}, {"A|C": 0.1}])
        assert aggregated["A|B"].median == pytest.approx(0.4)
        assert aggregated["A|B"].n == 2
        assert aggregated["A|C"].n == 1


class TestComparison:
    def _est(self, value, spread=0.1, n=5):
        return Estimate(median=value, p10=value - spread, p90=value + spread, n=n)

    def test_perfect_ordering_gives_rank_corr_one_even_with_wrong_level(self):
        real = {"A": self._est(1.0), "B": self._est(2.0), "C": self._est(3.0)}
        synth = {"A": self._est(10.0), "B": self._est(20.0), "C": self._est(30.0)}

        result = compare(real, synth, key="k", label="l")
        assert result.rank_corr == pytest.approx(1.0)
        assert result.ratio == pytest.approx(10.0)  # el nivel SI se reporta mal
        assert result.coverage_pct == 0.0

    def test_reversed_ordering_gives_rank_corr_minus_one(self):
        real = {"A": self._est(1.0), "B": self._est(2.0), "C": self._est(3.0)}
        synth = {"A": self._est(3.0), "B": self._est(2.0), "C": self._est(1.0)}
        assert compare(real, synth, key="k", label="l").rank_corr == pytest.approx(-1.0)

    def test_coverage_counts_real_values_inside_the_synthetic_range(self):
        real = {"A": self._est(1.0), "B": self._est(5.0)}
        synth = {"A": self._est(1.05, spread=0.2), "B": self._est(2.0, spread=0.2)}

        result = compare(real, synth, key="k", label="l")
        assert result.coverage_pct == pytest.approx(50.0)
        assert [i.inside for i in result.items] == [True, False]

    def test_only_names_present_on_both_sides_are_compared(self):
        real = {"A": self._est(1.0), "B": self._est(2.0), "ONLY_REAL": self._est(9.0)}
        synth = {"A": self._est(1.0), "B": self._est(2.0), "ONLY_SYNTH": self._est(9.0)}

        result = compare(real, synth, key="k", label="l")
        assert result.n == 2
        assert [i.name for i in result.items] == ["A", "B"]

    def test_a_ratio_against_a_near_zero_real_level_is_declared_unknown(self):
        # La autocorrelacion real ronda cero: el cociente no significaria nada.
        real = {"A": self._est(0.0001), "B": self._est(0.0002)}
        synth = {"A": self._est(0.05), "B": self._est(0.06)}
        assert np.isnan(compare(real, synth, key="k", label="l").ratio)

    def test_an_empty_intersection_degrades_instead_of_exploding(self):
        result = compare({"A": self._est(1.0)}, {"B": self._est(1.0)}, key="k", label="l")
        assert result.n == 0 and result.items == ()
        assert result.as_dict()["rank_corr"] is None

    def test_nan_is_serialised_as_null_not_as_zero(self):
        payload = compare({}, {}, key="k", label="l").as_dict()
        assert payload["real_median"] is None and payload["ratio"] is None


# ------------------------------------------------------------------ estudio ----------


class FakeBarsService:
    """Proveedor de barras en memoria: el estudio no toca la red en los tests."""

    def __init__(self, bars: dict, failing: tuple[str, ...] = ()) -> None:
        self.bars = bars
        self.failing = failing
        self.calls: list[str] = []

    def get_daily_bars(self, symbol, start, end):
        self.calls.append(symbol)
        if symbol in self.failing:
            raise RuntimeError("exchange caido")
        df = self.bars.get(symbol)
        return None if df is None else df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


class TestCalendarWindows:
    def test_non_overlapping_windows_tile_the_period(self):
        start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC")
        windows = calendar_windows(start, end, 365, 365)

        assert len(windows) == 3
        assert windows[0][1] == windows[1][0]  # pegadas, sin solape
        assert windows[-1][1] <= end  # ninguna se sale del periodo declarado

    def test_a_half_step_adds_windows_by_overlapping_them(self):
        start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")

        assert len(calendar_windows(start, end, 730, 730)) == 2
        assert len(calendar_windows(start, end, 730, 365)) == 3

    def test_rejects_non_positive_sizes(self):
        start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")
        with pytest.raises(ValueError):
            calendar_windows(start, end, 0, 10)


class TestFetchRealBars:
    def _service(self, **kwargs):
        bars = {"BTC/USDT": _bars(_iid(600, seed=41)), "ETH/USDT": _bars(_iid(600, seed=42))}
        return FakeBarsService(bars, **kwargs)

    def test_missing_symbols_are_skipped_not_invented(self):
        service = self._service()
        got = fetch_real_bars(
            ["BTC/USDT", "ETH/USDT", "MATIC/USDT"],
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2021, 6, 1, tzinfo=timezone.utc),
            service,
        )
        assert sorted(got) == ["BTC/USDT", "ETH/USDT"]

    def test_a_broken_symbol_does_not_bring_the_study_down(self):
        service = self._service(failing=("ETH/USDT",))
        got = fetch_real_bars(
            ["BTC/USDT", "ETH/USDT"],
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2021, 6, 1, tzinfo=timezone.utc),
            service,
        )
        assert sorted(got) == ["BTC/USDT"]

    def test_the_offline_provider_stops_at_the_last_closed_bar(self, tmp_path, monkeypatch):
        from ai_trader.data import cache as cache_module

        monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)
        cache_module.save_bars("crypto::BTC/USDT", _bars(_iid(30, seed=43)), timeframe="1D")

        got = CachedBarsProvider().get_daily_bars(
            "BTC/USDT",
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 11, tzinfo=timezone.utc),
        )
        # La barra del 11 (dia de cierre de la ventana) queda fuera: aun no ha cerrado.
        assert got.index.max() == pd.Timestamp("2020-01-10", tz="UTC")

    def test_the_offline_provider_returns_none_without_cache(self, tmp_path, monkeypatch):
        from ai_trader.data import cache as cache_module

        monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)
        assert CachedBarsProvider().get_daily_bars(
            "NOPE/USDT",
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 2, 1, tzinfo=timezone.utc),
        ) is None


class TestFactsCollector:
    def _frame(self, n=400, seeds=(51, 52)):
        return pd.DataFrame(
            {f"S{i}": _iid(n, seed=s) for i, s in enumerate(seeds)}, index=_index(n)
        )

    def test_each_window_adds_one_estimate_per_symbol(self):
        collector = FactsCollector(min_observations=200)
        collector.add(self._frame())
        collector.add(self._frame(seeds=(53, 54)))

        assert collector.n_windows == 2
        assert all(len(v) == 2 for v in collector.by_symbol.values())
        assert collector.symbols()["S0"]["ac1"].n == 2

    def test_windows_that_are_too_short_produce_no_facts(self):
        collector = FactsCollector(min_observations=200)
        collector.add(self._frame(n=100))

        assert collector.n_windows == 1  # la ventana existio...
        assert collector.by_symbol == {}  # ...pero no dejo evidencia utilizable

    def test_returns_frame_aligns_symbols_by_date(self):
        bars = {
            "A": _bars(_iid(100, seed=55), start="2020-01-01"),
            "B": _bars(_iid(100, seed=56), start="2020-02-01"),
        }
        frame = returns_frame(bars)
        assert list(frame.columns) == ["A", "B"]
        assert frame["B"].isna().sum() > 0  # B empieza mas tarde: hueco declarado, no relleno


class TestStudyEndToEnd:
    """El estudio completo contra una libreria sintetica minuscula y datos 'reales' falsos."""

    LIBRARY = "lib_fid"

    @pytest.fixture
    def workspace(self, tmp_path):
        from ai_trader.synthetic.designer import TemplateScenarioDesigner
        from ai_trader.synthetic.service import SyntheticDataService
        from ai_trader.synthetic.store import SyntheticStore

        root = tmp_path / "synthetic"
        service = SyntheticDataService(TemplateScenarioDesigner(), store=SyntheticStore(root))
        service.generate(
            self.LIBRARY, n_scenarios=2, n_paths=2, horizon_days=400, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        return root, tmp_path

    def _args(self, root, out_dir, **overrides):
        base = dict(
            library=self.LIBRARY, synthetic_root=str(root), exchange="fake",
            start="2020-01-01", end="2022-12-31", paths=2, scenarios=0,
            window_days=400, step_days=400, max_lag=5, sigma=3.0,
            min_observations=200, min_pair_overlap=100, offline=False,
            verify_determinism=False, out_dir=str(out_dir),
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def _patch_service(self, monkeypatch, symbols):
        bars = {
            symbol: _bars(_iid(1_200, seed=60 + i), start="2020-01-01")
            for i, symbol in enumerate(symbols)
        }
        service = FakeBarsService(bars)
        monkeypatch.setattr(
            "ai_trader.synthetic.fidelity_study.build_service",
            lambda exchange, *, offline: service,
        )
        return service

    def test_produces_a_complete_report(self, workspace, monkeypatch):
        root, tmp_path = workspace
        self._patch_service(monkeypatch, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

        report = run_study(self._args(root, tmp_path / "out"))

        assert report["plan"]["symbols"] == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        assert report["summary"]["n_symbols"] == 3
        assert report["summary"]["n_pairs"] == 3  # 3 activos -> 3 pares
        assert report["summary"]["n_synthetic_samples"] == 4  # 2 escenarios x 2 caminos
        assert {m["key"] for m in report["metrics"]} >= {"ac1", "excess_kurtosis"}
        assert report["cross_correlation"]["n"] == 3

    def test_symbols_without_real_history_are_declared_missing(self, workspace, monkeypatch):
        root, tmp_path = workspace
        self._patch_service(monkeypatch, ["BTC/USDT", "ETH/USDT"])

        report = run_study(self._args(root, tmp_path / "out"))
        missing = report["plan"]["missing_symbols"]

        assert "SOL/USDT" in missing and "BTC/USDT" not in missing
        # Y no se cuelan en la comparacion por la puerta de atras.
        assert all("SOL/USDT" not in i["name"] for i in report["cross_correlation"]["items"])

    def test_is_deterministic(self, workspace, monkeypatch):
        root, tmp_path = workspace
        self._patch_service(monkeypatch, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        args = self._args(root, tmp_path / "out")

        first = run_study(args)
        second = run_study(args)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_the_cli_writes_the_report_to_disk(self, workspace, monkeypatch):
        from ai_trader.synthetic import fidelity_study

        root, tmp_path = workspace
        self._patch_service(monkeypatch, ["BTC/USDT", "ETH/USDT"])
        out_dir = tmp_path / "out"

        code = fidelity_study.main([
            "--library", self.LIBRARY, "--synthetic-root", str(root),
            "--start", "2020-01-01", "--end", "2022-12-31", "--paths", "2",
            "--window-days", "400", "--step-days", "400", "--out-dir", str(out_dir),
        ])

        assert code == 0
        published = load_fidelity_report(out_dir / f"report_{self.LIBRARY}.json")
        assert published["plan"]["library_id"] == self.LIBRARY
        assert "generated_at" in published


class TestReportPlumbing:
    def test_crypto_symbols_filters_by_asset_class_and_sorts(self):
        universe = [
            {"symbol": "SPY", "asset_class": "stock"},
            {"symbol": "ETH/USDT", "asset_class": "crypto"},
            {"symbol": "BTC/USDT", "asset_class": "crypto"},
        ]
        assert crypto_symbols(universe) == ["BTC/USDT", "ETH/USDT"]

    def test_the_summary_only_averages_target_metrics(self):
        frame = pd.DataFrame(
            {"A": _iid(400, seed=71), "B": _iid(400, seed=72)}, index=_index(400)
        )
        real = measure_real(
            {"A": _bars(_iid(400, seed=71)), "B": _bars(_iid(400, seed=72))},
            [(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC") + 400 * DAY)],
        )
        synth = FactsCollector()
        synth.add(frame)

        plan = StudyPlan(
            library_id="lib", exchange="fake", timeframe="1d", real_start="2020-01-01",
            real_end="2021-01-01", window_days=400, step_days=400, n_scenarios=1, n_paths=1,
            symbols=("A", "B"), missing_symbols=(), max_lag=5, sigma=3.0,
            min_observations=200, min_pair_overlap=100, offline=True,
        )
        report = build_report(plan, real, synth)

        assert report["summary"]["n_symbols"] == 2
        assert report["summary"]["n_pairs"] == 1
        assert report["plan"]["windows_overlap"] is False

    def test_load_returns_none_when_the_report_is_not_published(self, tmp_path):
        assert load_fidelity_report(tmp_path / "nope.json") is None

    def test_metric_comparisons_cover_every_declared_metric(self):
        from ai_trader.synthetic.fidelity import METRIC_KEYS

        facts = {"A": aggregate_facts([series_facts(_iid(400, seed=81), min_observations=200)])}
        comparisons = compare_metrics(facts, facts)

        assert [c.key for c in comparisons] == list(METRIC_KEYS)
        assert all(c.rank_corr is not None for c in comparisons)

    def test_cross_correlation_comparison_uses_pairs_as_the_cross_section(self):
        real = aggregate_pairs([{"A|B": 0.8, "A|C": 0.2}])
        synth = aggregate_pairs([{"A|B": 0.5, "A|C": 0.1}])

        result = compare_cross_correlations(real, synth)
        assert result.n == 2
        assert result.rank_corr == pytest.approx(1.0)
