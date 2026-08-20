"""
El sustrato REAL del optimizador: sub-ventanas de mercado con CPCV y hold-out temporal.

Estos tests no tocan ni la red ni la cache: construyen `RealWindowSource` con barras en
memoria. Lo que se prueba es lo que cambia la decision -- que el hold-out sea el FUTURO y
no un sorteo, que una unidad caida penalice en vez de desaparecer, y que el sustrato se
declare -- no la calidad de unas barras concretas.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.app.runner import RunnerConfig
from ai_trader.config import AppConfig, StrategySpec
from ai_trader.execution.paper import PaperExecutionConfig
from ai_trader.risk.engine import RiskLimits
from ai_trader.scoring.optimize import run_optimization
from ai_trader.scoring.cem import CEMConfig
from ai_trader.scoring.real_source import (
    DEFAULT_WINDOW_DAYS,
    RealWindowSource,
    TemporalSplit,
    split_windows_by_time,
)
from ai_trader.scoring.real_substrate import real_windows
from ai_trader.scoring.sample_eval import FAILURE_PENALTY

ANCHOR = datetime(2020, 1, 1, tzinfo=timezone.utc)
WINDOW_DAYS = 400


def _wavy(n_days: int, *, period: float, amplitude: float, drift: float) -> pd.DataFrame:
    """Barras con onda + deriva: hay tendencia que seguir Y reversion que explotar, asi que
    ni momentum ni mean-reversion se quedan sin operar en ninguna sub-ventana."""
    index = pd.DatetimeIndex(
        [ANCHOR + timedelta(days=i) for i in range(n_days)], name="timestamp"
    )
    t = np.arange(n_days, dtype=float)
    closes = 100.0 + drift * t + amplitude * np.sin(2 * np.pi * t / period)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 1.0,
            "low": np.minimum(opens, closes) - 1.0,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=index,
    )


def _config() -> AppConfig:
    return AppConfig(
        runner=RunnerConfig(
            symbols=["BTC/USDT", "ETH/USDT"],
            lookback_days=30,
            max_holding_days=10,
            symbol_cooldown_hours=0,
            max_trades_per_cycle=5,
        ),
        risk=RiskLimits(
            min_confidence_per_trade=0.50,
            risk_fraction_per_trade=0.10,
            max_symbol_exposure_usd=1_000_000.0,
            max_total_exposure_usd=1_000_000.0,
            max_daily_loss_usd=1_000_000.0,
        ),
        execution=PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0),
        strategies=[],
    )


def _source(*, n_days: int = 1200, signals: dict | None = None) -> RealWindowSource:
    end = ANCHOR + timedelta(days=n_days)
    windows = real_windows(ANCHOR, end, WINDOW_DAYS)
    bars = {
        "BTC/USDT": _wavy(n_days, period=40, amplitude=8.0, drift=0.05),
        "ETH/USDT": _wavy(n_days, period=55, amplitude=6.0, drift=-0.02),
    }
    return RealWindowSource(
        _config(), bars, windows, split_windows_by_time(windows),
        symbols=("BTC/USDT", "ETH/USDT"),
        signals=signals,
    )


class TestTemporalSplit:
    """El hold-out es el FUTURO, no una muestra al azar."""

    def _windows(self, n: int):
        return real_windows(ANCHOR, ANCHOR + timedelta(days=WINDOW_DAYS * n), WINDOW_DAYS)

    def test_validation_is_the_most_recent_windows(self):
        windows = self._windows(5)

        split = split_windows_by_time(windows, n_validation=1)

        assert split.validation == ("w5",)
        assert split.train == ("w1", "w2", "w3", "w4")
        # Y la validacion empieza DESPUES de que acabe todo el train.
        last_train = max(w.end for w in windows if w.label in split.train)
        first_val = min(w.start for w in windows if w.label in split.validation)
        assert first_val >= last_train

    def test_more_validation_windows_still_come_off_the_end(self):
        split = split_windows_by_time(self._windows(5), n_validation=2)

        assert split.validation == ("w4", "w5")
        assert split.train == ("w1", "w2", "w3")

    def test_always_leaves_at_least_one_window_to_train_on(self):
        split = split_windows_by_time(self._windows(3), n_validation=99)

        assert split.n_train >= 1
        assert split.n_validation >= 1

    def test_a_single_window_cannot_be_split(self):
        with pytest.raises(ValueError, match="al menos 2"):
            split_windows_by_time(self._windows(1))

    def test_there_is_no_seed_to_choose_because_nothing_is_drawn(self):
        """La ausencia de semilla es el contenido: un hold-out temporal no se sortea."""
        assert TemporalSplit(train=("w1",), validation=("w2",)).seed is None

    def test_the_published_window_size_matches_the_transfer_study(self):
        """544 dias no es un numero redondo: es el del estudio de transferencia publicado,
        para que las cifras de los dos sitios se comparen sin traducir nada."""
        assert DEFAULT_WINDOW_DAYS == 544


class TestRealWindowSource:
    def test_a_unit_yields_one_sample_per_cpcv_fold(self):
        source = _source()
        spec = StrategySpec(type="crypto_momentum", id="probe", params={})

        evaluations = source.evaluations(spec, ("w1",))

        assert len(evaluations) == 15  # C(6,2)
        assert source.describe()["folds_per_unit"] == 15

    def test_scoring_two_units_concatenates_their_folds(self):
        source = _source()
        spec = StrategySpec(type="crypto_momentum", id="probe", params={})

        assert len(source.evaluations(spec, ("w1", "w2"))) == 30

    def test_a_broken_unit_penalises_instead_of_vanishing(self):
        """Si una unidad caida desapareciera, esa configuracion se estaria puntuando sobre
        menos unidades que las demas y el ranking dejaria de ser un ranking."""
        source = _source()
        source._windows["w1"] = source._windows["w1"].__class__(
            label="w1",
            start=ANCHOR - timedelta(days=5_000),   # fuera de las barras: no hay nada
            end=ANCHOR - timedelta(days=4_600),
        )
        spec = StrategySpec(type="crypto_momentum", id="probe", params={})

        evaluations = source.evaluations(spec, ("w1",))

        assert len(evaluations) == 15
        assert all(e.failed for e in evaluations)
        assert all(e.score == FAILURE_PENALTY for e in evaluations)

    def test_baselines_come_fold_by_fold_and_are_cached(self):
        source = _source()

        first = source.baseline_scores(("w1", "w2"))
        again = source.baseline_scores(("w1", "w2"))

        assert first  # hay al menos un baseline pasivo disponible
        assert all(len(values) == 30 for values in first.values())
        assert again == first
        assert set(source._baseline_cache) == {"w1", "w2"}

    def test_a_baseline_missing_in_one_unit_is_dropped_from_all(self):
        """Comparar contra una serie con huecos seria comparar contra otra cosa."""
        source = _source()
        source._baseline_cache["w1"] = {"btc_hold": [0.1] * 15, "equal_weight": [0.2] * 15}
        source._baseline_cache["w2"] = {"btc_hold": [0.3] * 15}

        out = source.baseline_scores(("w1", "w2"))

        assert set(out) == {"btc_hold"}
        assert len(out["btc_hold"]) == 30

    def test_describes_the_substrate_it_ranked_on(self):
        described = _source().describe()

        assert described["substrate"] == "real"
        assert described["holdout"].startswith("temporal")
        assert described["n_units"] == 3
        assert described["n_train_units"] == 2
        assert described["n_validation_units"] == 1
        assert described["scheme"] == "cpcv"
        assert described["signals_armed"] is False

    def test_declares_when_the_signal_archive_is_armed(self):
        """Encender el archivo real multiplica el coste por ~8: no puede pasar sin que se
        vea en el informe."""
        assert _source(signals={}).describe()["signals_armed"] is True


class TestOptimizerOnRealWindows:
    def test_runs_end_to_end_and_reports_the_real_substrate(self):
        result = run_optimization(
            "crypto_momentum",
            source=_source(),
            cem_config=CEMConfig(population=2, iterations=1, seed=0),
        )

        assert result.substrate["substrate"] == "real"
        assert result.split.validation == ("w3",)
        assert result.train.n == 2 * 15   # dos unidades de train x 15 folds
        assert result.validation.n == 1 * 15
        assert isinstance(result.approved, bool)

    def test_the_serialized_split_says_there_was_no_draw(self):
        result = run_optimization(
            "crypto_momentum",
            source=_source(),
            cem_config=CEMConfig(population=2, iterations=1, seed=0),
        )

        assert result.as_dict()["split"] == {
            "n_train": 2, "n_validation": 1, "seed": None
        }
