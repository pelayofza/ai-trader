from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from ai_trader.synthetic.designer import (
    TemplateScenarioDesigner,
    normalize_spec,
    parse_scenarios,
)
from ai_trader.synthetic.engine import PathEngine, generate_paths
from ai_trader.synthetic.scenarios import FactorPhase, FactorShock, ScenarioSpec
from ai_trader.synthetic.service import SyntheticDataService, sample_window
from ai_trader.synthetic.store import SyntheticStore
from ai_trader.synthetic.universe import DEFAULT_UNIVERSE, EQUITY


def _spec(phases, shocks=(), tilts=None, id="sc", horizon_name="test"):
    return ScenarioSpec(
        id=id,
        name="Test scenario",
        narrative="unit test",
        phases=tuple(phases),
        shocks=tuple(shocks),
        asset_tilts=tilts or {},
    )


def _equity_only(vol=0.02, days=500):
    """Escenario dominado por el factor EQUITY: aisla la estructura de correlacion."""
    return _spec([FactorPhase(length_days=days, drift={}, vol={EQUITY: vol})])


def _log_returns(df):
    return np.diff(np.log(df["close"].to_numpy()))


class TestEngineOHLCV:
    def test_is_deterministic_for_a_fixed_seed(self):
        engine = PathEngine(DEFAULT_UNIVERSE)
        a = engine.generate(_equity_only(days=120), seed=7)
        b = engine.generate(_equity_only(days=120), seed=7)

        for symbol in a:
            assert np.allclose(a[symbol].to_numpy(), b[symbol].to_numpy())

    def test_different_seeds_diverge(self):
        engine = PathEngine(DEFAULT_UNIVERSE)
        a = engine.generate(_equity_only(days=120), seed=1)
        b = engine.generate(_equity_only(days=120), seed=2)

        assert not np.allclose(a["SPY"]["close"].to_numpy(), b["SPY"]["close"].to_numpy())

    def test_bars_are_valid_ohlc(self):
        bars = PathEngine(DEFAULT_UNIVERSE).generate(_equity_only(days=200), seed=3)

        for symbol, df in bars.items():
            assert len(df) == 200
            o, h, low, c = df["open"], df["high"], df["low"], df["close"]
            assert (h >= np.maximum(o, c) - 1e-9).all(), symbol
            assert (low <= np.minimum(o, c) + 1e-9).all(), symbol
            assert (df[["open", "high", "low", "close"]] > 0).all().all(), symbol

    def test_index_is_utc_and_ordered(self):
        df = PathEngine(DEFAULT_UNIVERSE).generate(_equity_only(days=50), seed=0)["BTC/USDT"]

        assert df.index.name == "timestamp"
        assert str(df.index.tz) == "UTC"
        assert df.index.is_monotonic_increasing

    def test_starts_near_configured_price(self):
        df = PathEngine(DEFAULT_UNIVERSE).generate(_equity_only(vol=0.001, days=30), seed=0)["SPY"]
        # Con vol minima el primer cierre no se aleja mucho del precio inicial (450).
        assert 400 < df["close"].iloc[0] < 500


class TestFactorCorrelations:
    def test_shared_factor_loading_creates_positive_correlation(self):
        # Escenario solo-EQUITY: SPY y QQQ cargan fuerte y positivo; TLT casi nada.
        bars = PathEngine(DEFAULT_UNIVERSE).generate(_equity_only(vol=0.02, days=600), seed=11)

        spy, qqq, tlt = _log_returns(bars["SPY"]), _log_returns(bars["QQQ"]), _log_returns(bars["TLT"])
        corr_spy_qqq = np.corrcoef(spy, qqq)[0, 1]
        corr_spy_tlt = np.corrcoef(spy, tlt)[0, 1]

        assert corr_spy_qqq > 0.8
        assert corr_spy_qqq > corr_spy_tlt

    def test_shock_moves_loaded_asset(self):
        spec = _spec(
            [FactorPhase(length_days=40, drift={}, vol={EQUITY: 1e-6})],
            shocks=[FactorShock(day=20, factor=EQUITY, magnitude=-0.10)],
        )
        bars = PathEngine(DEFAULT_UNIVERSE).generate(spec, seed=5)

        # SPY carga EQUITY 1.0: el retorno del dia del shock debe ser fuertemente negativo.
        ret = _log_returns(bars["SPY"])
        assert ret[19] < -0.05  # diff en indice 19 = close[20]/close[19]


class TestNormalizeSpec:
    def test_drops_unknown_factors_and_symbols(self):
        spec = _spec(
            [FactorPhase(length_days=100, drift={"BOGUS": 0.5}, vol={EQUITY: 0.01, "BOGUS": 9.0})],
            tilts={"FAKECOIN": 0.01, "SPY": 0.001},
        )
        clean = normalize_spec(spec, DEFAULT_UNIVERSE, horizon_days=100)

        assert "BOGUS" not in clean.phases[0].vol
        assert "BOGUS" not in clean.phases[0].drift
        assert "FAKECOIN" not in clean.asset_tilts
        assert "SPY" in clean.asset_tilts

    def test_fits_horizon_exactly(self):
        spec = _spec(
            [FactorPhase(length_days=30, drift={}, vol={}),
             FactorPhase(length_days=30, drift={}, vol={})]
        )
        clean = normalize_spec(spec, DEFAULT_UNIVERSE, horizon_days=200)

        assert clean.horizon_days == 200

    def test_empty_phases_get_a_default_phase(self):
        clean = normalize_spec(_spec([]), DEFAULT_UNIVERSE, horizon_days=120)

        assert clean.horizon_days == 120
        assert len(clean.phases) == 1


class TestParseScenarios:
    def test_parses_fenced_json(self):
        payload = """Here you go:
```json
{"scenarios": [
  {"id": "a", "name": "A", "narrative": "n",
   "phases": [{"length_days": 100, "drift": {"EQUITY": 0.001}, "vol": {"EQUITY": 0.01}}]}
]}
```"""
        specs = parse_scenarios(payload, DEFAULT_UNIVERSE, horizon_days=100)

        assert len(specs) == 1
        assert specs[0].id == "a"
        assert specs[0].horizon_days == 100

    def test_skips_malformed_but_keeps_valid(self):
        payload = json.dumps(
            {"scenarios": [
                {"name": "no phases"},  # sin phases -> se normaliza a fase por defecto
                {"id": "ok", "name": "ok", "narrative": "",
                 "phases": [{"length_days": 50, "drift": {}, "vol": {"EQUITY": 0.01}}]},
            ]}
        )
        specs = parse_scenarios(payload, DEFAULT_UNIVERSE, horizon_days=50)

        assert any(s.id == "ok" for s in specs)

    def test_raises_on_non_list(self):
        with pytest.raises(ValueError):
            parse_scenarios('{"foo": 1}', DEFAULT_UNIVERSE, horizon_days=50)


class TestTemplateDesigner:
    def test_produces_requested_count_with_fitted_horizon(self):
        specs = TemplateScenarioDesigner().design(DEFAULT_UNIVERSE, n_scenarios=10, horizon_days=300)

        assert len(specs) == 10
        assert all(s.horizon_days == 300 for s in specs)

    def test_ids_are_unique(self):
        specs = TemplateScenarioDesigner().design(DEFAULT_UNIVERSE, n_scenarios=20, horizon_days=200)

        assert len({s.id for s in specs}) == 20


class TestGeneratePaths:
    def test_generates_an_ensemble(self):
        paths = generate_paths(
            _equity_only(days=100), DEFAULT_UNIVERSE, n_paths=5, seed_base=1000
        )

        assert len(paths) == 5
        # Paths distintos (semillas contiguas), mismo universo.
        assert not np.allclose(paths[0]["SPY"]["close"], paths[1]["SPY"]["close"])
        assert set(paths[0]) == set(DEFAULT_UNIVERSE.symbols)


class TestStoreRoundTrip:
    def _service(self, tmp_path):
        store = SyntheticStore(tmp_path)
        return SyntheticDataService(TemplateScenarioDesigner(), store=store), store

    def test_save_and_load_bars(self, tmp_path):
        service, store = self._service(tmp_path)
        manifest = service.generate(
            "lib1", n_scenarios=3, n_paths=2, horizon_days=150, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        assert manifest.num_samples == 6
        assert "lib1" in store.list_libraries()

        first_scenario = manifest.scenarios[0]["id"]
        bars = store.load_bars("lib1", first_scenario, 0)
        assert set(bars) == set(DEFAULT_UNIVERSE.symbols)
        assert len(bars["BTC/USDT"]) == 150
        assert list(bars["BTC/USDT"].columns) == ["open", "high", "low", "close", "volume"]

    def test_iter_samples_covers_everything(self, tmp_path):
        service, store = self._service(tmp_path)
        service.generate(
            "lib2", n_scenarios=4, n_paths=3, horizon_days=120, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        samples = list(store.iter_samples("lib2"))
        assert len(samples) == 12

    def test_regeneration_is_reproducible(self, tmp_path):
        # Misma semilla + disenador determinista -> mismas barras tras round-trip.
        s1, store1 = self._service(tmp_path / "a")
        s2, store2 = self._service(tmp_path / "b")
        kwargs = dict(
            n_scenarios=2, n_paths=2, horizon_days=100, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        m1 = s1.generate("lib", **kwargs)
        s2.generate("lib", **kwargs)

        sc = m1.scenarios[0]["id"]
        b1 = store1.load_bars("lib", sc, 0)["ETH/USDT"]["close"].to_numpy()
        b2 = store2.load_bars("lib", sc, 0)["ETH/USDT"]["close"].to_numpy()
        assert np.allclose(b1, b2)


class TestSampleWindow:
    def test_leaves_warmup_room(self, tmp_path):
        service = SyntheticDataService(TemplateScenarioDesigner(), store=SyntheticStore(tmp_path))
        manifest = service.generate(
            "libw", n_scenarios=1, n_paths=1, horizon_days=300, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        start, end = sample_window(manifest, warmup_days=90)

        assert start < end
        assert (start - datetime.fromisoformat(manifest.anchor)).days == 90

    def test_raises_when_warmup_exceeds_horizon(self, tmp_path):
        service = SyntheticDataService(TemplateScenarioDesigner(), store=SyntheticStore(tmp_path))
        manifest = service.generate(
            "libw2", n_scenarios=1, n_paths=1, horizon_days=100, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        with pytest.raises(ValueError):
            sample_window(manifest, warmup_days=200)


class TestBacktestBridge:
    """El output del generador es legible por el motor de backtest via from_bars."""

    def test_runs_a_backtest_over_a_synthetic_sample(self, tmp_path):
        from ai_trader.app.runner import RunnerConfig
        from ai_trader.backtest.engine import BacktestEngine
        from ai_trader.config import AppConfig, StrategySpec
        from ai_trader.execution.paper import PaperExecutionConfig
        from ai_trader.risk.engine import RiskLimits

        store = SyntheticStore(tmp_path)
        service = SyntheticDataService(TemplateScenarioDesigner(), store=store)
        manifest = service.generate(
            "btlib", n_scenarios=1, n_paths=1, horizon_days=260, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        config = AppConfig(
            runner=RunnerConfig(symbols=["BTC/USDT"], lookback_days=40, max_holding_days=10),
            risk=RiskLimits(risk_fraction_per_trade=0.10, min_confidence_per_trade=0.50),
            execution=PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0),
            strategies=[StrategySpec(type="crypto_momentum", id="mom", params={"min_bars": 30})],
        )

        sc = manifest.scenarios[0]["id"]
        bars = store.load_bars("btlib", sc, 0)
        start, end = sample_window(manifest, warmup_days=config.runner.lookback_days + 30)

        result = BacktestEngine.from_bars(config, bars, starting_equity=10_000.0).run(
            start=start, end=end
        )

        # No exigimos que gane dinero: exigimos que el bucle completo funcione.
        assert result.train.equity_curve
        assert result.test.equity_curve
        assert result.starting_equity == 10_000.0
