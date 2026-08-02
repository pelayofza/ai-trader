from __future__ import annotations

import pytest

from ai_trader.shared.instruments import AssetClass, OutcomeToken, PredictionMarket, Venue
from ai_trader.shared.schemas import Side
from ai_trader.strategies import build_strategy
from ai_trader.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumConfig, CryptoMomentumStrategy
from ai_trader.strategies.polymarket_threshold import (
    PolymarketThresholdConfig,
    PolymarketThresholdStrategy,
)
from tests.conftest import build_bars


class _StubRegime:
    """Provider de regimen de prueba: devuelve features fijas para cualquier simbolo."""

    def __init__(self, *, breadth: float, relative_strength: float) -> None:
        self._features = {
            "breadth": breadth,
            "relative_strength": relative_strength,
            "corr_to_market": 0.0,
            "agg_vol": 0.0,
        }

    def features(self, symbol: str) -> dict[str, float]:
        return self._features


class TestCryptoMomentum:
    def test_no_signal_without_enough_bars(self):
        strategy = CryptoMomentumStrategy()

        assert strategy.generate_signal("BTC/USDT", build_bars([100.0] * 10)) is None

    def test_no_signal_in_a_downtrend(self):
        strategy = CryptoMomentumStrategy()
        closes = [float(200 - i) for i in range(60)]  # bajando

        assert strategy.generate_signal("BTC/USDT", build_bars(closes)) is None

    def test_signal_on_an_uptrend_breakout(self):
        strategy = CryptoMomentumStrategy()
        closes = [float(100 + i * 2) for i in range(60)]  # subiendo y rompiendo maximos

        signal = strategy.generate_signal("BTC/USDT", build_bars(closes))

        assert signal is not None
        assert signal.side == Side.BUY
        assert signal.entry_price == pytest.approx(closes[-1])
        assert 0.0 <= signal.confidence <= 1.0

    def test_breakout_gate_is_enforced_when_required(self):
        """
        El filtro de ruptura estaba anulado a mano (`breakout_ok = True`), asi que la
        estrategia operaba sin el pese a llamarse "breakout".
        """
        # Tendencia alcista, pero el ultimo cierre no supera el maximo reciente.
        closes = [float(100 + i) for i in range(59)] + [130.0]
        bars = build_bars(closes)

        strict = CryptoMomentumStrategy(CryptoMomentumConfig(require_breakout=True))
        lenient = CryptoMomentumStrategy(CryptoMomentumConfig(require_breakout=False))

        assert strict.generate_signal("BTC/USDT", bars) is None
        assert lenient.generate_signal("BTC/USDT", bars) is not None

    def test_does_not_support_prediction_symbols(self):
        strategy = CryptoMomentumStrategy()

        assert strategy.supports_symbol("BTC/USDT")
        assert not strategy.supports_symbol("PM::some-market")

    def test_regime_breadth_gate_blocks_when_active(self):
        closes = [float(100 + i * 2) for i in range(60)]  # senal de momentum clara
        bars = build_bars(closes)
        strategy = CryptoMomentumStrategy(CryptoMomentumConfig(min_breadth=0.5))

        strategy.attach_regime_provider(_StubRegime(breadth=0.2, relative_strength=1.0))
        assert strategy.generate_signal("BTC/USDT", bars) is None

        strategy.attach_regime_provider(_StubRegime(breadth=0.8, relative_strength=1.0))
        assert strategy.generate_signal("BTC/USDT", bars) is not None

    def test_regime_filter_inactive_by_default_ignores_provider(self):
        closes = [float(100 + i * 2) for i in range(60)]
        bars = build_bars(closes)
        strategy = CryptoMomentumStrategy()  # defaults permisivos: filtro apagado

        # Aunque el provider diria "mercado debil", el filtro esta desactivado.
        strategy.attach_regime_provider(_StubRegime(breadth=0.0, relative_strength=-5.0))
        assert strategy.generate_signal("BTC/USDT", bars) is not None


class TestMeanReversion:
    def test_no_signal_without_enough_bars(self):
        strategy = MeanReversionStrategy()

        assert strategy.generate_signal("BTC/USDT", build_bars([100.0] * 10)) is None

    def test_signal_when_price_is_stretched_below_the_mean(self):
        strategy = MeanReversionStrategy()
        # Precio plano en 100 y una caida brusca a 90 en el ultimo cierre: z << -2.
        closes = [100.0] * 39 + [90.0]

        signal = strategy.generate_signal("BTC/USDT", build_bars(closes))

        assert signal is not None
        assert signal.side == Side.BUY
        assert signal.entry_price == pytest.approx(90.0)
        assert 0.0 <= signal.confidence <= 1.0
        # El objetivo de reversion (exit_z=0 por defecto) queda en la media, por encima
        # de la entrada; el stop, por debajo.
        assert signal.take_profit > signal.entry_price
        assert signal.stop_loss < signal.entry_price
        assert signal.features["z_score"] < -strategy.config.entry_z

    def test_no_signal_when_price_rides_above_the_mean(self):
        strategy = MeanReversionStrategy()
        # Tendencia alcista: el precio va por encima de su media, z positivo, no compra.
        closes = [float(100 + i) for i in range(40)]

        assert strategy.generate_signal("BTC/USDT", build_bars(closes)) is None

    def test_does_not_short_an_overbought_stretch(self):
        strategy = MeanReversionStrategy()
        # Estirado por ARRIBA: la reversion es long-only, asi que no genera senal.
        closes = [100.0] * 39 + [110.0]

        assert strategy.generate_signal("BTC/USDT", build_bars(closes)) is None

    def test_entry_z_gates_the_signal(self):
        # Una caida moderada dispara con umbral laxo pero no con umbral estricto.
        closes = [100.0] * 39 + [97.0]
        bars = build_bars(closes)

        lax = MeanReversionStrategy(MeanReversionConfig(entry_z=1.0))
        strict = MeanReversionStrategy(MeanReversionConfig(entry_z=5.0))

        assert lax.generate_signal("BTC/USDT", bars) is not None
        assert strict.generate_signal("BTC/USDT", bars) is None

    def test_does_not_support_prediction_symbols(self):
        strategy = MeanReversionStrategy()

        assert strategy.supports_symbol("BTC/USDT")
        assert not strategy.supports_symbol("PM::some-market")

    def test_config_rejects_exit_z_above_entry_z(self):
        with pytest.raises(ValueError, match="exit_z must be < entry_z"):
            MeanReversionConfig(entry_z=1.0, exit_z=2.0)

    def test_regime_relative_strength_ceiling_blocks_strong_assets(self):
        closes = [100.0] * 39 + [90.0]  # sobreventa clara
        bars = build_bars(closes)
        # Solo compra rezagados: techo de fuerza relativa en 0.
        strategy = MeanReversionStrategy(MeanReversionConfig(max_relative_strength=0.0))

        # Activo mas fuerte que el mercado (rs=0.5 > techo 0.0): bloqueado.
        strategy.attach_regime_provider(_StubRegime(breadth=1.0, relative_strength=0.5))
        assert strategy.generate_signal("BTC/USDT", bars) is None

        # Rezagado (rs=-0.5): pasa.
        strategy.attach_regime_provider(_StubRegime(breadth=1.0, relative_strength=-0.5))
        assert strategy.generate_signal("BTC/USDT", bars) is not None


def make_market(yes_price: float, slug: str = "test-market") -> PredictionMarket:
    return PredictionMarket(
        market_id="1",
        question="Will it happen?",
        slug=slug,
        active=True,
        closed=False,
        archived=False,
        enable_order_book=True,
        outcomes=[
            OutcomeToken(outcome="Yes", token_id="tok-yes", price=yes_price),
            OutcomeToken(outcome="No", token_id="tok-no", price=1.0 - yes_price),
        ],
    )


class TestPolymarketThreshold:
    @pytest.fixture
    def strategy(self) -> PolymarketThresholdStrategy:
        return PolymarketThresholdStrategy(
            PolymarketThresholdConfig(slug="test-market", outcome="yes", buy_below_price=0.40)
        )

    def test_no_signal_above_the_threshold(self, strategy):
        assert strategy.generate_signal("PM::test-market", make_market(0.55)) is None

    def test_signal_below_the_threshold(self, strategy):
        signal = strategy.generate_signal("PM::test-market", make_market(0.30))

        assert signal is not None
        assert signal.side == Side.BUY
        assert signal.entry_price == pytest.approx(0.30)

    def test_signal_carries_the_instrument_identity(self, strategy):
        """
        Emitir una Signal (y no una OrderRequest) es lo que hace que estas operaciones
        pasen por el motor de riesgo, del que antes se escapaban por completo.
        """
        signal = strategy.generate_signal("PM::test-market", make_market(0.30))

        assert signal.venue == Venue.POLYMARKET
        assert signal.asset_class == AssetClass.PREDICTION
        assert signal.instrument_id == "tok-yes"
        assert signal.outcome == "Yes"

    def test_supports_only_its_own_market(self, strategy):
        assert strategy.supports_symbol("PM::test-market")
        assert not strategy.supports_symbol("PM::another-market")
        assert not strategy.supports_symbol("BTC/USDT")


class TestRegistry:
    def test_builds_a_strategy_from_type_and_params(self):
        strategy = build_strategy("crypto_momentum", {"fast_sma_window": 5, "slow_sma_window": 15})

        assert isinstance(strategy, CryptoMomentumStrategy)
        assert strategy.config.fast_sma_window == 5

    def test_builds_mean_reversion_from_type_and_params(self):
        strategy = build_strategy("mean_reversion", {"lookback": 15, "entry_z": 1.5})

        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.config.lookback == 15
        assert strategy.config.entry_z == 1.5

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown strategy type"):
            build_strategy("does_not_exist")

    def test_invalid_params_are_rejected(self):
        with pytest.raises(ValueError, match="Invalid params"):
            build_strategy("crypto_momentum", {"not_a_real_param": 1})
