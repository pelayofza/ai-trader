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
from ai_trader.synthetic.retrofit import enrich_phase, enrich_spec
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

    def test_intrabar_range_widens_in_high_vol_phases(self):
        # B1: la vol diaria es POR FASE, asi que el rango intradia (mechas/gaps) debe
        # ensancharse en la fase de panico. Antes se promediaba y el ATR no reaccionaba.
        spec = _spec([
            FactorPhase(length_days=100, drift={}, vol={EQUITY: 0.005}),  # calma
            FactorPhase(length_days=100, drift={}, vol={EQUITY: 0.05}),   # panico
        ])
        spy = PathEngine(DEFAULT_UNIVERSE).generate(spec, seed=0)["SPY"]

        rel_range = ((spy["high"] - spy["low"]) / spy["close"]).to_numpy()
        calm = rel_range[:100].mean()
        panic = rel_range[100:].mean()

        assert panic > 3 * calm


def _ar_spec(idio_ar, days=500):
    """Escenario con factores en el suelo (vol~0): aisla el componente idiosincratico."""
    return _spec([FactorPhase(length_days=days, drift={}, vol={}, idio_ar=idio_ar)])


def _lag1_autocorr(returns):
    return float(np.corrcoef(returns[:-1], returns[1:])[0, 1])


class TestIdioAutocorrelation:
    def test_negative_ar_makes_returns_mean_revert(self):
        # B5: idio_ar<0 => autocorrelacion de lag-1 negativa (whipsaw / reversion),
        # que es el edge que necesita la primitiva de mean-reversion.
        btc = PathEngine(DEFAULT_UNIVERSE).generate(_ar_spec(-0.4), seed=0)["BTC/USDT"]
        assert _lag1_autocorr(_log_returns(btc)) < -0.2

    def test_positive_ar_makes_returns_trend(self):
        btc = PathEngine(DEFAULT_UNIVERSE).generate(_ar_spec(0.4), seed=0)["BTC/USDT"]
        assert _lag1_autocorr(_log_returns(btc)) > 0.2

    def test_neutral_ar_is_uncorrelated(self):
        btc = PathEngine(DEFAULT_UNIVERSE).generate(_ar_spec(0.0), seed=0)["BTC/USDT"]
        assert abs(_lag1_autocorr(_log_returns(btc))) < 0.12

    def test_autocorrelation_is_variance_matched(self):
        # Anadir estructura serial NO debe cambiar la vol total: std similar para
        # cualquier phi y cercana a idio_vol (0.02 para BTC).
        engine = PathEngine(DEFAULT_UNIVERSE)
        rev = _log_returns(engine.generate(_ar_spec(-0.4), seed=0)["BTC/USDT"]).std()
        mom = _log_returns(engine.generate(_ar_spec(0.4), seed=0)["BTC/USDT"]).std()
        flat = _log_returns(engine.generate(_ar_spec(0.0), seed=0)["BTC/USDT"]).std()

        assert np.isclose(rev, flat, rtol=0.2)
        assert np.isclose(mom, flat, rtol=0.2)
        assert 0.015 < flat < 0.026


def _tail_spec(tail_dof, days=2000):
    return _spec([FactorPhase(length_days=days, drift={}, vol={}, tail_dof=tail_dof)])


def _garch_spec(persistence, days=2000):
    return _spec([FactorPhase(length_days=days, drift={}, vol={}, vol_persistence=persistence)])


def _tail_exceedances(returns, k=3.0):
    z = (returns - returns.mean()) / returns.std()
    return int(np.sum(np.abs(z) > k))


def _abs_autocorr(returns):
    a = np.abs(returns)
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


class TestFatTails:
    def test_student_t_produces_more_tail_exceedances(self):
        # B2: colas gruesas -> mas dias mas alla de 3 sigma que la gaussiana (que espera
        # ~0.27% -> ~5 en 2000). Metrica robusta (la kurtosis muestral de una t es ruidosa).
        engine = PathEngine(DEFAULT_UNIVERSE)
        gauss = _tail_exceedances(_log_returns(engine.generate(_tail_spec(0.0), seed=0)["BTC/USDT"]))
        fat = _tail_exceedances(_log_returns(engine.generate(_tail_spec(5.0), seed=0)["BTC/USDT"]))

        assert fat > gauss
        assert fat >= 15  # claramente leptocurtico vs ~5 gaussiano

    def test_fat_tails_stay_variance_matched(self):
        engine = PathEngine(DEFAULT_UNIVERSE)
        gauss = _log_returns(engine.generate(_tail_spec(0.0), seed=0)["BTC/USDT"]).std()
        fat = _log_returns(engine.generate(_tail_spec(5.0), seed=0)["BTC/USDT"]).std()

        assert np.isclose(fat, gauss, rtol=0.25)


class TestVolClustering:
    def test_persistence_clusters_volatility(self):
        # B3: la persistencia GARCH crea autocorrelacion en |retornos| (clustering),
        # ausente en el mundo iid.
        engine = PathEngine(DEFAULT_UNIVERSE)
        flat = _abs_autocorr(_log_returns(engine.generate(_garch_spec(0.0), seed=0)["BTC/USDT"]))
        clustered = _abs_autocorr(
            _log_returns(engine.generate(_garch_spec(0.9), seed=0)["BTC/USDT"])
        )

        assert clustered > flat + 0.05
        assert clustered > 0.1

    def test_clustering_stays_variance_matched(self):
        engine = PathEngine(DEFAULT_UNIVERSE)
        flat = _log_returns(engine.generate(_garch_spec(0.0), seed=0)["BTC/USDT"]).std()
        clustered = _log_returns(engine.generate(_garch_spec(0.9), seed=0)["BTC/USDT"]).std()

        assert np.isclose(clustered, flat, rtol=0.3)


def _overnight_gaps(df):
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    return o[1:] / c[:-1] - 1.0  # hueco de apertura vs cierre anterior


class TestJumps:
    def test_jumps_produce_large_overnight_gaps(self):
        # B4: con saltos aparecen huecos de apertura mucho mayores que el gap gaussiano,
        # que es lo que hace que un stop pueda saltarse (perdida de cola realista).
        engine = PathEngine(DEFAULT_UNIVERSE)
        calm = _spec([FactorPhase(length_days=1500, drift={}, vol={EQUITY: 0.01})])
        jumpy = _spec([FactorPhase(
            length_days=1500, drift={}, vol={EQUITY: 0.01},
            jump_intensity=0.05, jump_scale=6.0,
        )])

        no_jump = np.abs(_overnight_gaps(engine.generate(calm, seed=0)["BTC/USDT"])).max()
        with_jump = np.abs(_overnight_gaps(engine.generate(jumpy, seed=0)["BTC/USDT"])).max()

        assert with_jump > 5 * no_jump

    def test_no_jumps_keeps_bars_valid(self):
        # Sanity: con saltos las velas siguen siendo OHLC validas.
        jumpy = _spec([FactorPhase(
            length_days=300, drift={}, vol={EQUITY: 0.02},
            jump_intensity=0.1, jump_scale=8.0,
        )])
        df = PathEngine(DEFAULT_UNIVERSE).generate(jumpy, seed=1)["SPY"]
        o, h, low, c = df["open"], df["high"], df["low"], df["close"]

        assert (h >= np.maximum(o, c) - 1e-9).all()
        assert (low <= np.minimum(o, c) + 1e-9).all()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()


class TestShockJitter:
    def _crash_day(self, df):
        # El dia del crash = indice del retorno mas negativo.
        return int(np.argmin(_log_returns(df)))

    def test_jitter_moves_the_shock_day_across_paths(self):
        engine = PathEngine(DEFAULT_UNIVERSE)
        phase = FactorPhase(length_days=400, drift={}, vol={EQUITY: 1e-6})
        jittered = _spec(
            [phase], shocks=[FactorShock(day=200, factor=EQUITY, magnitude=-0.20, jitter_days=30)]
        )

        days = {self._crash_day(engine.generate(jittered, seed=1000 + i)["SPY"]) for i in range(5)}
        assert len(days) > 1  # el crash NO cae siempre el mismo dia

    def test_without_jitter_shock_day_is_fixed(self):
        engine = PathEngine(DEFAULT_UNIVERSE)
        phase = FactorPhase(length_days=400, drift={}, vol={EQUITY: 1e-6})
        fixed = _spec(
            [phase], shocks=[FactorShock(day=200, factor=EQUITY, magnitude=-0.20)]
        )

        days = {self._crash_day(engine.generate(fixed, seed=1000 + i)["SPY"]) for i in range(5)}
        assert len(days) == 1  # sin jitter, mismo dia en todos los paths


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


class TestMicrostructureSchema:
    def test_round_trips_microstructure_fields(self):
        phase = FactorPhase(
            length_days=100,
            vol={EQUITY: 0.01},
            idio_ar=-0.2,
            tail_dof=4.0,
            vol_persistence=0.85,
            jump_intensity=0.02,
            jump_scale=3.0,
        )
        assert FactorPhase.from_dict(phase.to_dict()) == phase

    def test_neutral_fields_are_not_serialized(self):
        # Los spec.json de ai_v1 no deben cambiar: los campos neutros no se emiten.
        neutral = FactorPhase(length_days=10, vol={EQUITY: 0.01})
        payload = neutral.to_dict()

        assert set(payload) == {"length_days", "drift", "vol"}

    def test_normalize_preserves_microstructure(self):
        phase = FactorPhase(length_days=100, vol={EQUITY: 0.01}, idio_ar=-0.3, tail_dof=5.0)
        clean = normalize_spec(_spec([phase]), DEFAULT_UNIVERSE, horizon_days=100)

        assert clean.phases[0].idio_ar == -0.3
        assert clean.phases[0].tail_dof == 5.0

    def test_normalize_carries_microstructure_through_horizon_refit(self):
        # Dos fases con micro, horizonte distinto: _fit_horizon reescala longitudes
        # pero debe arrastrar idio_ar/tail_dof (no resetearlos).
        phases = [
            FactorPhase(length_days=30, vol={EQUITY: 0.01}, idio_ar=0.2),
            FactorPhase(length_days=30, vol={EQUITY: 0.03}, idio_ar=-0.2),
        ]
        clean = normalize_spec(_spec(phases), DEFAULT_UNIVERSE, horizon_days=200)

        assert clean.horizon_days == 200
        assert clean.phases[0].idio_ar == 0.2
        assert clean.phases[1].idio_ar == -0.2


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


class TestUniverseReconstruction:
    def test_summary_is_self_contained_round_trip(self):
        from ai_trader.synthetic.universe import universe_from_summary, universe_summary

        summary = universe_summary(DEFAULT_UNIVERSE)
        assert all("start_price" in item for item in summary)

        rebuilt = universe_from_summary(summary, DEFAULT_UNIVERSE.factors)
        assert rebuilt.symbols == DEFAULT_UNIVERSE.symbols
        for original in DEFAULT_UNIVERSE.assets:
            clone = rebuilt.asset(original.symbol)
            assert clone.start_price == original.start_price
            assert clone.idio_vol == original.idio_vol
            assert clone.loadings == original.loadings

    def test_missing_start_price_raises(self):
        from ai_trader.synthetic.universe import universe_from_summary

        bad = [{"symbol": "X", "asset_class": "crypto", "loadings": {}, "idio_vol": 0.01}]
        with pytest.raises(KeyError):
            universe_from_summary(bad, DEFAULT_UNIVERSE.factors)


class TestResynthesize:
    """Ampliar el ensemble desde escenarios guardados, sin volver a llamar a la IA."""

    def _service(self, tmp_path):
        store = SyntheticStore(tmp_path)
        return SyntheticDataService(TemplateScenarioDesigner(), store=store), store

    def test_adds_paths_keeping_existing_identical(self, tmp_path):
        service, store = self._service(tmp_path)
        m0 = service.generate(
            "lib", n_scenarios=2, n_paths=2, horizon_days=100, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        sc = m0.scenarios[0]["id"]
        before = store.load_bars("lib", sc, 0)["BTC/USDT"]["close"].to_numpy()

        m1 = service.resynthesize("lib", n_paths=5)

        assert m1.n_paths == 5
        assert len(list(store.iter_samples("lib"))) == 2 * 5
        # Los paths previos (mismas semillas) no cambian; solo se anaden nuevos.
        after = store.load_bars("lib", sc, 0)["BTC/USDT"]["close"].to_numpy()
        assert np.allclose(before, after)

    def test_regenerates_from_specs_only(self, tmp_path):
        # Borrar el parquet y reconstruir: solo los spec.json son insustituibles.
        service, store = self._service(tmp_path)
        m0 = service.generate(
            "lib", n_scenarios=2, n_paths=3, horizon_days=100, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        sc = m0.scenarios[0]["id"]
        original = store.load_bars("lib", sc, 1)["ETH/USDT"]["close"].to_numpy()

        (tmp_path / "lib" / "scenarios" / sc / "paths.parquet").unlink()
        service.resynthesize("lib", n_paths=3)

        rebuilt = store.load_bars("lib", sc, 1)["ETH/USDT"]["close"].to_numpy()
        assert np.allclose(original, rebuilt)


class TestRetrofit:
    def test_directional_phase_gets_positive_autocorrelation(self):
        trending = FactorPhase(length_days=100, drift={EQUITY: 0.002}, vol={EQUITY: 0.007})
        assert enrich_phase(trending).idio_ar > 0

    def test_flat_phase_gets_mean_reversion(self):
        flat = FactorPhase(length_days=100, drift={}, vol={EQUITY: 0.012})
        assert enrich_phase(flat).idio_ar < 0

    def test_crisis_phase_gets_tails_and_jumps(self):
        crisis = FactorPhase(length_days=100, drift={EQUITY: -0.004}, vol={EQUITY: 0.03})
        ep = enrich_phase(crisis)

        assert ep.tail_dof > 0
        assert ep.jump_intensity > 0
        assert ep.vol_persistence > 0

    def test_enrich_spec_adds_shock_jitter(self):
        spec = _spec(
            [FactorPhase(length_days=100, drift={}, vol={EQUITY: 0.012})],
            shocks=[FactorShock(day=50, factor=EQUITY, magnitude=-0.1)],
        )
        assert enrich_spec(spec).shocks[0].jitter_days > 0

    def test_enrich_preserves_designed_drift_and_vol(self):
        # El retrofit solo toca microestructura: drift/vol disenados por la IA no cambian.
        phase = FactorPhase(length_days=100, drift={EQUITY: 0.001}, vol={EQUITY: 0.02})
        enriched = enrich_phase(phase)

        assert enriched.drift == phase.drift
        assert enriched.vol == phase.vol


class TestDeriveLibrary:
    def test_derives_enriched_library_keeping_source_intact(self, tmp_path):
        store = SyntheticStore(tmp_path)
        service = SyntheticDataService(TemplateScenarioDesigner(), store=store)
        service.generate(
            "src", n_scenarios=3, n_paths=2, horizon_days=200, seed_base=1000,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        manifest = service.derive_library("src", "dst", enricher=enrich_spec)

        assert "dst" in store.list_libraries()
        assert manifest.num_samples == 6

        # El destino tiene microestructura; la fuente sigue neutra.
        dst_phases = [p for s in store.load_specs("dst") for p in s.phases]
        src_phases = [p for s in store.load_specs("src") for p in s.phases]
        assert any(p.vol_persistence != 0 or p.idio_ar != 0 for p in dst_phases)
        assert all(p.vol_persistence == 0 and p.idio_ar == 0 for p in src_phases)

        # Barras validas y del tamano esperado.
        bars = store.load_bars("dst", manifest.scenarios[0]["id"], 0)
        assert len(bars["BTC/USDT"]) == 200


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
