from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest.engine import BacktestEngine, resolve_split_cutoff
from ai_trader.backtest.metrics import HeadlineWeights
from ai_trader.config import AppConfig, StrategySpec
from ai_trader.execution.paper import PaperExecutionConfig
from ai_trader.risk.engine import RiskLimits
from ai_trader.app.runner import RunnerConfig


def trending_df(n_days: int, start_price: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """
    Rampa alcista lineal y determinista. El paso diario (step) supera el rango
    intradia, de modo que cada cierre marca un maximo nuevo y el filtro de breakout
    del momentum deja entrar. El open del dia es el cierre del dia anterior.
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex([start + pd.Timedelta(days=i) for i in range(n_days)], name="timestamp")

    closes = np.array([start_price + i * step for i in range(n_days)])
    opens = np.array([start_price] + [closes[i - 1] for i in range(1, n_days)])
    highs = np.maximum(opens, closes) + 0.1
    lows = np.minimum(opens, closes) - 0.1

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=index,
    )


class FakeService:
    """Sirve el historico completo desde memoria; el clamping lo hace la fuente."""

    def __init__(self, bars: dict[str, pd.DataFrame]):
        self._bars = bars

    def get_daily_bars(self, symbol, start, end):
        return self._bars.get(symbol.strip().upper())


def make_config() -> AppConfig:
    return AppConfig(
        runner=RunnerConfig(
            symbols=["BTC/USDT"],
            lookback_days=120,
            max_holding_days=10,
            symbol_cooldown_hours=0,  # re-entrada libre para generar varios trades
            max_trades_per_cycle=5,
        ),
        risk=RiskLimits(
            min_confidence_per_trade=0.50,
            risk_fraction_per_trade=0.10,
            max_symbol_exposure_usd=1_000_000.0,  # generoso: en modo equity no debe estorbar
            max_total_exposure_usd=1_000_000.0,
            max_daily_loss_usd=1_000_000.0,
        ),
        execution=PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0),
        strategies=[
            StrategySpec(
                type="crypto_momentum",
                id="mom",
                params={"min_bars": 30, "require_breakout": True, "min_atr_pct": 0.1},
            )
        ],
    )


@pytest.fixture
def engine():
    bars = {"BTC/USDT": trending_df(400)}
    return BacktestEngine(make_config(), FakeService(bars), starting_equity=10_000.0)


class TestDataSources:
    """El motor corre igual con datos por proveedor (real/sintetico bajo demanda) o
    con series ya generadas en memoria (datos simulados en bloque)."""

    def test_runs_from_a_bar_provider(self):
        bars = {"BTC/USDT": trending_df(400)}
        engine = BacktestEngine(make_config(), FakeService(bars), starting_equity=10_000.0)

        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert result.train.metrics.num_trades > 0

    def test_runs_from_preloaded_simulated_bars(self):
        # Via directa del futuro generador sintetico: series ya en memoria, sin proveedor.
        simulated = {"BTC/USDT": trending_df(400)}
        engine = BacktestEngine.from_bars(make_config(), simulated, starting_equity=10_000.0)

        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert result.train.metrics.num_trades > 0

    def test_both_data_paths_give_the_same_result(self):
        bars = {"BTC/USDT": trending_df(400)}
        args = dict(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        via_provider = BacktestEngine(make_config(), FakeService(bars), 10_000.0).run(**args)
        via_bars = BacktestEngine.from_bars(make_config(), bars, 10_000.0).run(**args)

        c1 = [round(p.equity, 6) for p in via_provider.test.equity_curve]
        c2 = [round(p.equity, 6) for p in via_bars.test.equity_curve]
        assert c1 == c2

    def test_requires_a_data_source(self):
        with pytest.raises(ValueError, match="data_provider or preloaded bars"):
            BacktestEngine(make_config())


class TestSplit:
    def test_produces_train_and_test_windows(self, engine):
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            split_ratio=0.7,
        )

        assert result.train.label == "train"
        assert result.test.label == "test"
        # El train termina antes de que empiece el test.
        assert result.train.end <= result.test.start

    def test_explicit_cutoff_overrides_ratio(self, engine):
        cutoff = datetime(2024, 10, 1, tzinfo=timezone.utc)
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            split_date=cutoff,
        )

        assert result.test.start >= cutoff

    def test_rejects_cutoff_outside_range(self, engine):
        with pytest.raises(ValueError, match="split_date"):
            engine.run(
                start=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc),
                split_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            )


class TestExecution:
    def test_an_uptrend_produces_trades_and_grows_equity(self, engine):
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        assert result.train.metrics.num_trades > 0
        # Serie claramente alcista: el equity de entrenamiento deberia crecer.
        assert result.train.metrics.ending_equity > result.train.metrics.starting_equity

    def test_equity_curve_has_one_point_per_trading_day(self, engine):
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = engine.run(start=start, end=end, split_ratio=0.7)

        assert len(result.test.equity_curve) >= 2
        # Monotonia temporal de la curva.
        days = [p.day for p in result.test.equity_curve]
        assert days == sorted(days)

    def test_headline_score_is_the_penalized_out_of_sample_sharpe(self, engine):
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        m = result.test.metrics
        w = result.headline_weights
        expected = (
            m.sharpe - w.lambda_turnover * m.turnover - w.kappa_maxdd * m.max_drawdown_pct / 100.0
        )
        assert result.headline_score == pytest.approx(expected)
        # Es OUT-OF-SAMPLE: el train no entra en la cabecera.
        assert result.headline_score != pytest.approx(result.train.metrics.sharpe)

    def test_headline_weights_are_configurable(self):
        bars = {"BTC/USDT": trending_df(400)}
        args = dict(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        neutral = BacktestEngine(
            make_config(), FakeService(bars), 10_000.0,
            headline_weights=HeadlineWeights(lambda_turnover=0.0, kappa_maxdd=0.0),
        ).run(**args)
        strict = BacktestEngine(
            make_config(), FakeService(bars), 10_000.0,
            headline_weights=HeadlineWeights(lambda_turnover=10.0, kappa_maxdd=10.0),
        ).run(**args)

        # Sin penalizaciones el headline es el Sharpe puro; penalizar solo puede restar.
        assert neutral.headline_score == pytest.approx(neutral.test.metrics.sharpe)
        assert strict.headline_score <= neutral.headline_score


class TestSplitCutoff:
    """La frontera train/test es una funcion de modulo compartida: los baselines la
    reutilizan para puntuarse en la MISMA ventana out-of-sample que la estrategia."""

    def test_engine_windows_respect_the_shared_cutoff(self, engine):
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)

        result = engine.run(start=start, end=end, split_ratio=0.7)
        cutoff = resolve_split_cutoff(start, end, split_ratio=0.7)

        assert result.train.end <= cutoff <= result.test.start

    def test_explicit_split_date_must_fall_inside_the_range(self):
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError):
            resolve_split_cutoff(start, end, split_date=datetime(2025, 6, 1, tzinfo=timezone.utc))


class TestDeterminism:
    def test_two_identical_runs_give_the_same_equity_curve(self):
        bars = {"BTC/USDT": trending_df(400)}
        args = dict(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            split_ratio=0.7,
        )

        r1 = BacktestEngine(make_config(), FakeService(bars), 10_000.0).run(**args)
        r2 = BacktestEngine(make_config(), FakeService(bars), 10_000.0).run(**args)

        c1 = [round(p.equity, 6) for p in r1.test.equity_curve]
        c2 = [round(p.equity, 6) for p in r2.test.equity_curve]
        assert c1 == c2
        assert r1.headline_score == r2.headline_score


class TestLiquidityIsWired:
    """El motor inyecta la costura de liquidez en la ejecucion: dos mercados identicos
    salvo por la profundidad de sus barras NO pueden costar lo mismo."""

    def _run(self, symbol: str, volume: float):
        bars = {symbol: trending_df(400).assign(volume=volume)}
        config = make_config()
        config.runner.symbols = [symbol]
        return BacktestEngine.from_bars(config, bars, 10_000.0).run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

    def test_a_thin_market_erodes_more_equity_than_a_deep_one(self):
        # Mismo simbolo (mismo spread base) y mismo camino de precios: solo cambia el
        # volumen, asi que la unica diferencia posible es el impacto de mercado.
        deep = self._run("AAA/USDT", 1_000_000.0)
        thin = self._run("AAA/USDT", 100.0)  # capacidad 10 u: aun no recorta el tamano

        assert deep.train.metrics.num_trades == thin.train.metrics.num_trades
        assert thin.train.metrics.ending_equity < deep.train.metrics.ending_equity

    def test_capacity_caps_what_a_thin_market_can_absorb(self):
        """Con barras muy finas el techo recorta el tamano: se opera menos notional que
        el que el riesgo autoriza, que es justo lo que un backtest sin capacidad ignora."""
        deep = self._run("AAA/USDT", 1_000_000.0)
        thin = self._run("AAA/USDT", 10.0)  # capacidad 1 u por barra

        biggest_deep = max(p.size for p in deep.train.trades)
        biggest_thin = max(p.size for p in thin.train.trades)

        assert biggest_thin < biggest_deep
        assert biggest_thin <= 10.0 * 0.10 + 1e-9  # max_participation del config


class TestCompounding:
    def test_larger_starting_capital_scales_position_sizes(self):
        bars = {"BTC/USDT": trending_df(400)}
        args = dict(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        small = BacktestEngine(make_config(), FakeService(bars), 10_000.0).run(**args)
        big = BacktestEngine(make_config(), FakeService(bars), 100_000.0).run(**args)

        # Con sizing por fraccion de equity, 10x capital -> ~10x notional por trade,
        # asi que el retorno PORCENTUAL debe ser parecido (no el absoluto).
        assert big.train.metrics.ending_equity > 5 * small.train.metrics.ending_equity
        assert math.isclose(
            big.train.metrics.total_return_pct,
            small.train.metrics.total_return_pct,
            rel_tol=0.05,
        )
