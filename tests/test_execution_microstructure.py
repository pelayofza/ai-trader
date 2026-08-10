from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.execution.microstructure import (
    EMPTY_SNAPSHOT,
    BarLiquidityProvider,
    LiquiditySnapshot,
    SlippageModel,
)
from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import OrderRequest, OrderStatus, OrderType, Side

# Dos mundos deliberadamente contrastados: el mismo tamano de orden en USD es
# irrelevante para BTC y una fraccion visible de la barra del altcoin.
BTC = LiquiditySnapshot(bar_volume=500_000.0, recent_volatility=0.025)
ALTCOIN = LiquiditySnapshot(bar_volume=2_000.0, recent_volatility=0.070)


def order(**overrides) -> OrderRequest:
    defaults = dict(
        symbol="BTC/USDT",
        side=Side.BUY,
        size=1.0,
        order_type=OrderType.MARKET,
        strategy_id="test_strategy",
        asset_class=AssetClass.CRYPTO,
    )
    return OrderRequest(**{**defaults, **overrides})


class TestSpreadBase:
    """El suelo del coste depende del simbolo, no de una constante global."""

    def test_btc_is_the_tightest_symbol_of_the_universe(self):
        model = SlippageModel()

        btc = model.base_spread_bps("BTC/USDT", AssetClass.CRYPTO)
        alt = model.base_spread_bps("SEI/USDT", AssetClass.CRYPTO)

        assert btc < alt

    def test_unknown_symbol_falls_back_to_its_asset_class(self):
        model = SlippageModel()

        unknown = model.base_spread_bps("WHATEVER/USDT", AssetClass.CRYPTO)

        # Desconocido no es barato: se trata como poco liquido, nunca como BTC.
        assert unknown > model.base_spread_bps("BTC/USDT", AssetClass.CRYPTO)

    def test_config_override_wins_over_the_table(self):
        model = SlippageModel(spread_bps={"btc/usdt": 40.0})

        assert model.base_spread_bps("BTC/USDT", AssetClass.CRYPTO) == 40.0


class TestSlippageModel:
    def test_an_illiquid_altcoin_pays_more_than_btc(self):
        """El requisito central: el mismo tamano cuesta mas donde hay menos libro."""
        model = SlippageModel()

        btc = model.slippage_bps("BTC/USDT", 1.0, BTC, AssetClass.CRYPTO)
        alt = model.slippage_bps("SEI/USDT", 1.0, ALTCOIN, AssetClass.CRYPTO)

        assert alt > btc

    def test_the_gap_survives_an_identical_spread(self):
        """Aunque se igualen los spreads, volatilidad y profundidad siguen separandolos:
        el modelo no depende solo de la tabla."""
        model = SlippageModel(spread_bps={"BTC/USDT": 10.0, "SEI/USDT": 10.0})

        btc = model.slippage_bps("BTC/USDT", 50.0, BTC, AssetClass.CRYPTO)
        alt = model.slippage_bps("SEI/USDT", 50.0, ALTCOIN, AssetClass.CRYPTO)

        assert alt > btc

    def test_bigger_orders_pay_more(self):
        model = SlippageModel()

        small = model.slippage_bps("SEI/USDT", 10.0, ALTCOIN, AssetClass.CRYPTO)
        big = model.slippage_bps("SEI/USDT", 200.0, ALTCOIN, AssetClass.CRYPTO)

        assert big > small

    def test_impact_is_concave_in_size(self):
        """Ley de raiz cuadrada: 4x tamano NO es 4x impacto, es 2x."""
        model = SlippageModel(spread_bps={"X": 0.0}, vol_coef=0.0)
        snap = LiquiditySnapshot(bar_volume=1_000.0, recent_volatility=0.02)

        one = model.slippage_bps("X", 10.0, snap)
        four = model.slippage_bps("X", 40.0, snap)

        assert four == pytest.approx(2.0 * one, rel=1e-6)

    def test_volatility_widens_the_cost(self):
        model = SlippageModel()
        calm = LiquiditySnapshot(bar_volume=10_000.0, recent_volatility=0.01)
        panic = LiquiditySnapshot(bar_volume=10_000.0, recent_volatility=0.09)

        assert model.slippage_bps("BTC/USDT", 5.0, panic) > model.slippage_bps(
            "BTC/USDT", 5.0, calm
        )

    def test_a_thinner_bar_costs_more_for_the_same_order(self):
        model = SlippageModel()
        deep = LiquiditySnapshot(bar_volume=100_000.0, recent_volatility=0.03)
        thin = LiquiditySnapshot(bar_volume=1_000.0, recent_volatility=0.03)

        assert model.slippage_bps("BTC/USDT", 50.0, thin) > model.slippage_bps(
            "BTC/USDT", 50.0, deep
        )

    def test_without_liquidity_data_there_is_no_invented_impact(self):
        """Sin volumen no se extrapola impacto: solo spread y volatilidad de referencia."""
        model = SlippageModel()

        blind = model.slippage_bps("BTC/USDT", 10_000.0, EMPTY_SNAPSHOT, AssetClass.CRYPTO)
        expected = 0.5 * model.base_spread_bps("BTC/USDT", AssetClass.CRYPTO) + (
            model.vol_coef * model.reference_volatility * 10_000.0
        )

        assert blind == pytest.approx(expected)

    def test_the_cost_is_capped(self):
        model = SlippageModel(max_slippage_bps=120.0)
        dust = LiquiditySnapshot(bar_volume=0.001, recent_volatility=0.5)

        assert model.slippage_bps("SEI/USDT", 10_000.0, dust) == 120.0

    def test_is_deterministic(self):
        model = SlippageModel()
        args = ("SEI/USDT", 37.5, ALTCOIN, AssetClass.CRYPTO)

        assert model.slippage_bps(*args) == model.slippage_bps(*args)

    def test_prediction_markets_are_not_treated_as_free(self):
        model = SlippageModel()

        pm = model.slippage_bps("PM::some-market", 100.0, EMPTY_SNAPSHOT, AssetClass.PREDICTION)

        assert pm > model.slippage_bps("BTC/USDT", 100.0, EMPTY_SNAPSHOT, AssetClass.CRYPTO)


class _FixedLiquidity:
    """Proveedor de liquidez de laboratorio: dice lo que se le pide, sin barras."""

    def __init__(self, by_symbol: dict[str, LiquiditySnapshot]):
        self.by_symbol = by_symbol

    def snapshot(self, symbol: str) -> LiquiditySnapshot:
        return self.by_symbol.get(symbol.strip().upper(), EMPTY_SNAPSHOT)


def engine_with(snapshots: dict[str, LiquiditySnapshot], **config) -> PaperExecutionEngine:
    return PaperExecutionEngine(
        PaperExecutionConfig(**config),
        liquidity_provider=_FixedLiquidity(snapshots),
    )


class TestEngineCharges:
    def test_an_illiquid_altcoin_fills_worse_than_btc(self):
        """Mismo notional (1.000 USD), mismo lado, mundos distintos."""
        engine = engine_with({"BTC/USDT": BTC, "SEI/USDT": ALTCOIN})

        btc = engine.execute(order(symbol="BTC/USDT", size=0.025), market_price=40_000.0)
        alt = engine.execute(order(symbol="SEI/USDT", size=1_000.0), market_price=1.0)

        assert alt.slippage_bps > btc.slippage_bps
        # Y se paga donde importa: el precio de compra se aleja mas del midpoint.
        assert alt.filled_price / 1.0 > btc.filled_price / 40_000.0

    def test_the_result_reports_the_slippage_actually_applied(self):
        engine = engine_with({"SEI/USDT": ALTCOIN})

        result = engine.execute(order(symbol="SEI/USDT", size=50.0), market_price=2.0)

        assert result.slippage_bps > 0
        assert result.filled_price == pytest.approx(2.0 * (1 + result.slippage_bps / 10_000))

    def test_selling_pays_the_same_cost_on_the_other_side(self):
        engine = engine_with({"SEI/USDT": ALTCOIN})

        buy = engine.execute(order(symbol="SEI/USDT", side=Side.BUY, size=50.0), market_price=2.0)
        sell = engine.execute(order(symbol="SEI/USDT", side=Side.SELL, size=50.0), market_price=2.0)

        assert buy.slippage_bps == sell.slippage_bps
        assert sell.filled_price < 2.0 < buy.filled_price


class TestCapacity:
    def test_an_order_above_the_cap_fills_partially(self):
        # 2.000 unidades por barra x 10% = 200 unidades de capacidad.
        engine = engine_with({"SEI/USDT": ALTCOIN}, max_participation=0.10)

        result = engine.execute(order(symbol="SEI/USDT", size=1_000.0), market_price=2.0)

        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.filled_size == pytest.approx(200.0)
        assert "requested=1000" in result.message

    def test_an_order_below_the_cap_fills_whole(self):
        engine = engine_with({"SEI/USDT": ALTCOIN}, max_participation=0.10)

        result = engine.execute(order(symbol="SEI/USDT", size=150.0), market_price=2.0)

        assert result.status == OrderStatus.FILLED
        assert result.filled_size == pytest.approx(150.0)

    def test_fees_follow_the_filled_size_not_the_requested_one(self):
        engine = engine_with({"SEI/USDT": ALTCOIN}, fee_rate=0.001, max_participation=0.10)

        result = engine.execute(order(symbol="SEI/USDT", size=1_000.0), market_price=2.0)

        assert result.fees == pytest.approx(result.filled_price * 200.0 * 0.001)

    def test_a_bar_without_any_capacity_rejects_the_order(self):
        dust = LiquiditySnapshot(bar_volume=1e-12, recent_volatility=0.05)
        engine = engine_with({"SEI/USDT": dust})

        result = engine.execute(order(symbol="SEI/USDT", size=10.0), market_price=2.0)

        assert not result.success
        assert result.status == OrderStatus.REJECTED
        assert result.filled_size == 0.0

    def test_exits_are_exempt_from_the_cap(self):
        """Entrar es opcional; salir no. Un cierre se llena entero y paga el impacto."""
        engine = engine_with({"SEI/USDT": ALTCOIN}, max_participation=0.10)

        entry = engine.execute(order(symbol="SEI/USDT", size=1_000.0), market_price=2.0)
        exit_ = engine.execute(
            order(symbol="SEI/USDT", size=1_000.0, side=Side.SELL, reduce_only=True),
            market_price=2.0,
        )

        assert exit_.status == OrderStatus.FILLED
        assert exit_.filled_size == pytest.approx(1_000.0)
        # Cruza 5x mas libro que el fill parcial de entrada, y se le cobra por ello.
        assert exit_.slippage_bps > entry.slippage_bps

    def test_partial_fills_can_be_switched_off(self):
        engine = engine_with({"SEI/USDT": ALTCOIN}, allow_partial_fills=False)

        result = engine.execute(order(symbol="SEI/USDT", size=1_000.0), market_price=2.0)

        assert result.status == OrderStatus.FILLED
        assert result.filled_size == pytest.approx(1_000.0)

    def test_symbols_without_liquidity_data_are_not_capped(self):
        engine = engine_with({})

        result = engine.execute(order(size=10_000.0), market_price=40_000.0)

        assert result.status == OrderStatus.FILLED

    def test_prediction_orders_never_ask_the_provider(self):
        """No hay OHLCV de un mercado de prediccion: se cobra el spread de su clase y
        no se inventa un techo de capacidad."""

        class _Exploding:
            def snapshot(self, symbol):  # pragma: no cover - debe no llamarse
                raise AssertionError("prediction orders must not query bar liquidity")

        engine = PaperExecutionEngine(PaperExecutionConfig(), liquidity_provider=_Exploding())

        result = engine.execute(
            order(
                symbol="PM::some-market",
                asset_class=AssetClass.PREDICTION,
                venue=Venue.POLYMARKET,
                instrument_id="token-123",
                outcome="yes",
                size=100.0,
            ),
            market_price=0.40,
        )

        assert result.status == OrderStatus.FILLED
        assert result.slippage_bps > 0


class _Bars:
    def __init__(self, bars: dict[str, pd.DataFrame], clock):
        self._bars = bars
        self.clock = clock

    def get_daily_bars(self, symbol, start, end):
        df = self._bars.get(symbol.strip().upper())
        if df is None:
            return None
        # Mismo corte anti look-ahead que la fuente real: la barra de hoy no ha cerrado.
        today = pd.Timestamp(self.clock.now()).tz_convert("UTC").normalize()
        window = df[(df.index >= pd.Timestamp(start).tz_convert("UTC")) & (df.index < today)]
        return window if not window.empty else None


def bars_with(volumes: list[float], closes: list[float]) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex(
        [start + timedelta(days=i) for i in range(len(closes))], name="timestamp"
    )
    closes_arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes_arr,
            "high": closes_arr * 1.01,
            "low": closes_arr * 0.99,
            "close": closes_arr,
            "volume": np.array(volumes, dtype=float),
        },
        index=index,
    )


class TestBarLiquidityProvider:
    def test_reads_typical_volume_and_recent_volatility(self):
        closes = [100.0 * (1.01 if i % 2 else 0.99) ** i for i in range(30)]
        clock = HistoricalClock(datetime(2024, 1, 31, tzinfo=timezone.utc))
        provider = BarLiquidityProvider(
            _Bars({"X/USDT": bars_with([1_000.0] * 30, closes)}, clock), clock
        )

        snap = provider.snapshot("X/USDT")

        assert snap.bar_volume == pytest.approx(1_000.0)
        assert snap.recent_volatility > 0

    def test_uses_the_median_not_the_spike(self):
        """Un dia de panico multiplica el volumen; tomar ese pico como capacidad
        regalaria liquidez justo cuando escasea."""
        volumes = [1_000.0] * 29 + [50_000.0]
        clock = HistoricalClock(datetime(2024, 1, 31, tzinfo=timezone.utc))
        provider = BarLiquidityProvider(
            _Bars({"X/USDT": bars_with(volumes, [100.0 + i for i in range(30)])}, clock), clock
        )

        assert provider.snapshot("X/USDT").bar_volume == pytest.approx(1_000.0)

    def test_does_not_look_ahead(self):
        """La liquidez de hoy se estima con barras cerradas: la de hoy no cuenta."""
        volumes = [1_000.0] * 29 + [999_999.0]
        closes = [100.0 + i for i in range(30)]
        day_of_the_spike = datetime(2024, 1, 30, tzinfo=timezone.utc)
        clock = HistoricalClock(day_of_the_spike)
        provider = BarLiquidityProvider(
            _Bars({"X/USDT": bars_with(volumes, closes)}, clock), clock
        )

        assert provider.snapshot("X/USDT").bar_volume == pytest.approx(1_000.0)

    def test_a_symbol_without_bars_gives_an_empty_snapshot(self):
        clock = HistoricalClock(datetime(2024, 1, 31, tzinfo=timezone.utc))
        provider = BarLiquidityProvider(_Bars({}, clock), clock)

        assert provider.snapshot("PM::whatever") == EMPTY_SNAPSHOT

    def test_too_few_bars_leave_volatility_unknown(self):
        clock = HistoricalClock(datetime(2024, 1, 5, tzinfo=timezone.utc))
        provider = BarLiquidityProvider(
            _Bars({"X/USDT": bars_with([1_000.0] * 3, [100.0, 101.0, 102.0])}, clock), clock
        )

        snap = provider.snapshot("X/USDT")

        assert snap.recent_volatility is None
        assert snap.bar_volume == pytest.approx(1_000.0)

    def test_answers_are_cached_within_the_same_day(self):
        clock = HistoricalClock(datetime(2024, 1, 31, tzinfo=timezone.utc))
        source = _Bars({"X/USDT": bars_with([1_000.0] * 30, [100.0 + i for i in range(30)])}, clock)
        calls = {"n": 0}
        inner = source.get_daily_bars

        def counting(symbol, start, end):
            calls["n"] += 1
            return inner(symbol, start, end)

        source.get_daily_bars = counting
        provider = BarLiquidityProvider(source, clock)

        provider.snapshot("X/USDT")
        provider.snapshot("X/USDT")

        assert calls["n"] == 1
