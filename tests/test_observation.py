from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ai_trader.observation.features import (
    ASSET_CLASS_ONEHOT,
    OWN_ASSET_FEATURES,
    Observation,
    asset_class_onehot,
    build_own_asset_features,
)
from ai_trader.observation.regime import REGIME_FEATURES, MarketRegimeProvider
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.instruments import AssetClass
from tests.conftest import build_bars


class TestOwnAssetFeatures:
    def test_returns_every_canonical_key(self):
        feats = build_own_asset_features(build_bars([100.0] * 80))

        assert set(feats) == set(OWN_ASSET_FEATURES)
        assert all(np.isfinite(v) for v in feats.values())

    def test_flat_series_is_neutral(self):
        feats = build_own_asset_features(build_bars([100.0] * 80))

        assert feats["ret_1d"] == 0.0
        assert feats["ret_60d"] == 0.0
        assert feats["drawdown_from_peak"] == 0.0

    def test_uptrend_shows_positive_returns_and_no_drawdown(self):
        feats = build_own_asset_features(build_bars([float(100 + i) for i in range(80)]))

        assert feats["ret_1d"] > 0.0
        assert feats["ret_20d"] > 0.0
        assert feats["drawdown_from_peak"] == 0.0
        # El cierre esta en la parte alta del canal Donchian.
        assert feats["donchian_pos"] > 0.9

    def test_recent_drop_registers_drawdown(self):
        feats = build_own_asset_features(build_bars([100.0] * 79 + [90.0]))

        assert feats["drawdown_from_peak"] > 0.0
        assert feats["ret_1d"] < 0.0

    def test_short_window_degrades_to_zero_not_nan(self):
        feats = build_own_asset_features(build_bars([100.0, 101.0, 102.0]))

        assert feats["ret_60d"] == 0.0
        assert all(np.isfinite(v) for v in feats.values())


class TestAssetClassOneHot:
    def test_crypto(self):
        assert asset_class_onehot(AssetClass.CRYPTO) == {
            "is_crypto": 1.0,
            "is_stock": 0.0,
            "is_macro": 0.0,
        }

    def test_unknown_is_macro(self):
        assert asset_class_onehot(None)["is_macro"] == 1.0


class TestObservationVector:
    def test_vector_has_stable_length_and_order(self):
        obs = Observation(
            symbol="BTC/USDT",
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            own_asset=build_own_asset_features(build_bars([100.0] * 80)),
            asset_class=asset_class_onehot(AssetClass.CRYPTO),
        )

        names = obs.feature_names()
        vector = obs.to_vector()

        expected_len = len(OWN_ASSET_FEATURES) + len(REGIME_FEATURES) + len(ASSET_CLASS_ONEHOT)
        assert len(names) == expected_len
        assert len(vector) == expected_len
        # Sin provider, los slots de regimen quedan reservados a 0.0 (longitud estable).
        regime_slice = vector[len(OWN_ASSET_FEATURES) : len(OWN_ASSET_FEATURES) + len(REGIME_FEATURES)]
        assert np.all(regime_slice == 0.0)


def _universe(n_bars: int = 80) -> dict:
    # A sube fuerte, B plano, C baja: regimenes distintos para probar el cross-sectional.
    return {
        "A": build_bars([float(100 + i) for i in range(n_bars)]),
        "B": build_bars([100.0] * n_bars),
        "C": build_bars([float(100 - i * 0.2) for i in range(n_bars)]),
    }


class TestMarketRegimeProvider:
    def test_features_are_finite_and_complete(self):
        bars = _universe()
        clock = HistoricalClock(bars["A"].index[-1].to_pydatetime())
        provider = MarketRegimeProvider(bars, clock, breadth_sma=10, rs_window=5, corr_window=10)

        feats = provider.features("A")

        assert set(feats) == set(REGIME_FEATURES)
        assert all(np.isfinite(v) for v in feats.values())
        assert 0.0 <= feats["breadth"] <= 1.0

    def test_strong_asset_beats_the_market(self):
        bars = _universe()
        clock = HistoricalClock(bars["A"].index[-1].to_pydatetime())
        provider = MarketRegimeProvider(bars, clock, breadth_sma=10, rs_window=5, corr_window=10)

        assert provider.features("A")["relative_strength"] > 0.0
        assert provider.features("C")["relative_strength"] < 0.0

    def test_breadth_reflects_trend_composition(self):
        bars = _universe()
        clock = HistoricalClock(bars["A"].index[-1].to_pydatetime())
        provider = MarketRegimeProvider(bars, clock, breadth_sma=10, rs_window=5, corr_window=10)

        # A por encima de su SMA, C por debajo, B en su media: amplitud intermedia.
        assert 0.0 < provider.features("A")["breadth"] < 1.0

    def test_today_bar_is_excluded_but_yesterday_is_visible(self):
        base = _universe(80)
        as_of = base["A"].index[50].to_pydatetime()

        def provider_with(a_frame):
            bars = dict(base, A=a_frame)
            return MarketRegimeProvider(
                bars, HistoricalClock(as_of), breadth_sma=10, rs_window=5, corr_window=10
            )

        baseline = provider_with(base["A"]).features("A")

        # Un pico enorme en la barra de HOY (index 50) no debe cambiar nada: no ha cerrado.
        spiked_today = base["A"].copy()
        spiked_today.iloc[50, spiked_today.columns.get_loc("close")] = 10_000.0
        assert provider_with(spiked_today).features("A") == baseline

        # El mismo pico en AYER (index 49) si es visible y mueve la fuerza relativa.
        spiked_yday = base["A"].copy()
        spiked_yday.iloc[49, spiked_yday.columns.get_loc("close")] = 10_000.0
        assert provider_with(spiked_yday).features("A")["relative_strength"] != baseline[
            "relative_strength"
        ]

    def test_anti_look_ahead_ignores_future_bars(self):
        full = _universe(80)
        as_of = full["A"].index[40].to_pydatetime()

        clock = HistoricalClock(as_of)
        provider_full = MarketRegimeProvider(full, clock, breadth_sma=10, rs_window=5, corr_window=10)

        # Mismo universo pero recortado a as_of: el provider completo, situado en as_of,
        # NO puede usar nada posterior, asi que ambos deben coincidir exactamente.
        truncated = {sym: df[df.index <= as_of] for sym, df in full.items()}
        provider_trunc = MarketRegimeProvider(
            truncated, HistoricalClock(as_of), breadth_sma=10, rs_window=5, corr_window=10
        )

        assert provider_full.features("A") == provider_trunc.features("A")
