"""
EL CABLEADO EN VIVO: lo que hasta hoy solo existia en backtest.

`main.py::build_runner` construia las estrategias y no les adjuntaba NADA. La consecuencia
no era teorica: cualquier configuracion que el optimizador eligiera con filtros de regimen
activos (`min_breadth`, `min/max_relative_strength`) se comportaba distinto en paper que en
el backtest que la selecciono, y esa discrepancia no aparecia en ninguna metrica.

Aqui se congelan las tres propiedades del arreglo:

1. Los DOS ensambladores —regimen y senales— se adjuntan en vivo, con el mismo bucle
   duck-typed que usa `backtest/engine.py`.
2. `observation/regime.py` no cambia ni una linea: el adaptador `LiveUniverseBars` cumple
   el contrato `Mapping` que ya esperaba. Que no haga falta tocarlo es la comprobacion de
   que el adaptador es correcto.
3. SIN CREDENCIALES, SIN ARCHIVO Y SIN SECCION `[signals]`, EL SISTEMA ARRANCA IGUAL:
   radar vacio, cobertura 0, puertas saltadas y un aviso explicito.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ai_trader.app.runner import RunnerConfig
from ai_trader.config import AppConfig, SignalsConfig, StrategySpec, load_config
from ai_trader.execution.paper import PaperExecutionConfig
from ai_trader.main import LiveUniverseBars, build_runner, build_signal_radar
from ai_trader.notifications.base import NullNotifier
from ai_trader.observation.regime import MarketRegimeProvider
from ai_trader.risk.engine import RiskLimits
from ai_trader.shared.clock import HistoricalClock
from tests.conftest import build_bars

UTC = timezone.utc


class FakeMarketDataService:
    """Lo minimo que `LiveUniverseBars` y el runner le piden al servicio real."""

    def __init__(self, bars: dict[str, pd.DataFrame]) -> None:
        self.bars = bars
        self.calls: list[tuple[str, datetime, datetime]] = []

    def get_daily_bars(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        return self.bars.get(symbol)

    def detect_asset_class(self, symbol):
        from ai_trader.shared.instruments import AssetClass

        return AssetClass.CRYPTO

    def get_prediction_market(self, slug):
        return None

    def get_prediction_midpoint(self, token_id):
        return None


def make_config(symbols=("BTC/USDT", "ETH/USDT"), **signals) -> AppConfig:
    """Un AppConfig construido a mano: estos tests cambian de directorio de trabajo y no
    pueden depender de una ruta relativa al repositorio."""
    return AppConfig(
        runner=RunnerConfig(symbols=list(symbols)),
        risk=RiskLimits(),
        execution=PaperExecutionConfig(),
        strategies=[StrategySpec(type="crypto_momentum", id="mom", params={})],
        signals=SignalsConfig(**signals),
    )


class TestLiveUniverseBars:
    def test_es_un_mapping_y_el_regimen_no_necesita_saber_nada_mas(self):
        """El contrato completo que `regime.py` usa: iterar y `get`. Si el adaptador
        cumple eso, el regimen funciona en vivo sin tocarlo."""
        bars = {
            "BTC/USDT": build_bars([float(100 + i) for i in range(80)]),
            "ETH/USDT": build_bars([float(50 + i) for i in range(80)]),
        }
        service = FakeMarketDataService(bars)
        clock = HistoricalClock(datetime(2026, 1, 1, tzinfo=UTC))
        universe = LiveUniverseBars(service, list(bars), clock, lookback_days=180)

        assert list(universe) == ["BTC/USDT", "ETH/USDT"]
        assert len(universe) == 2
        assert universe.get("BTC/USDT") is not None
        assert universe.get("NO/EXISTE") is None  # KeyError -> None, como espera el regimen

    def test_el_regimen_calcula_sobre_el_adaptador_sin_una_linea_de_cambio(self):
        bars = {
            f"T{i}/USDT": build_bars([float(100 + i + j) for j in range(80)]) for i in range(6)
        }
        service = FakeMarketDataService(bars)
        clock = HistoricalClock(datetime(2026, 1, 1, tzinfo=UTC))
        provider = MarketRegimeProvider(
            LiveUniverseBars(service, list(bars), clock, lookback_days=180), clock
        )
        features = provider.features("T0/USDT")

        assert set(features) == {"relative_strength", "corr_to_market", "breadth", "agg_vol"}
        assert 0.0 <= features["breadth"] <= 1.0
        assert service.calls  # ha ido a buscar barras de verdad

    def test_pide_la_ventana_del_lookback_configurado(self):
        service = FakeMarketDataService({"BTC/USDT": build_bars([1.0] * 80)})
        clock = HistoricalClock(datetime(2026, 1, 1, tzinfo=UTC))
        universe = LiveUniverseBars(service, ["BTC/USDT"], clock, lookback_days=180)
        universe["BTC/USDT"]

        symbol, start, end = service.calls[-1]
        assert symbol == "BTC/USDT"
        assert (end - start).days == 180


class TestBuildRunnerAttachesBothProviders:
    def test_regimen_y_senales_llegan_a_la_estrategia_en_vivo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # que un JsonStateStore por defecto no toque el repo
        service = FakeMarketDataService({"BTC/USDT": build_bars([100.0] * 80)})
        runner = build_runner(make_config(), service, NullNotifier())
        strategy = runner.strategies[0]

        assert strategy._regime is not None, "el hueco conocido: en vivo no se adjuntaba"
        assert strategy._signals is not None

    def test_el_universo_del_regimen_en_vivo_es_el_del_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        service = FakeMarketDataService({})
        config = make_config(symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"))
        runner = build_runner(config, service, NullNotifier())

        assert list(runner.strategies[0]._regime._bars) == config.runner.symbols


class TestStartsWithoutCredentials:
    def test_apagado_por_defecto(self):
        assert load_config("config/default.toml").signals.enabled is False

    def test_sin_seccion_signals_el_config_sigue_cargando(self, tmp_path):
        path = tmp_path / "sin_signals.toml"
        path.write_text(
            "universe = ['BTC/USDT']\n\n[[strategies]]\ntype = 'crypto_momentum'\n",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.signals == SignalsConfig()

    def test_una_clave_desconocida_revienta_al_cargar_y_no_en_silencio(self, tmp_path):
        """El splat de `load_config` es estricto a proposito: una opcion mal escrita tiene
        que fallar al arrancar, no ignorarse y dejar el radar apagado sin que nadie lo
        note."""
        path = tmp_path / "mal.toml"
        path.write_text(
            "universe = ['BTC/USDT']\n\n[signals]\nenable = true\n\n"
            "[[strategies]]\ntype = 'crypto_momentum'\n",
            encoding="utf-8",
        )
        with pytest.raises(TypeError):
            load_config(path)

    def test_apagado_el_radar_no_lee_el_archivo(self, monkeypatch):
        import ai_trader.main as main

        def explode(*args, **kwargs):  # pragma: no cover - no debe llamarse
            raise AssertionError("con [signals] enabled = false no se toca el archivo")

        monkeypatch.setattr(main, "load_frames", explode)
        radar = build_signal_radar(make_config(enabled=False), HistoricalClock(datetime.now(UTC)))
        assert radar.is_empty

    def test_encendido_sin_archivo_avisa_y_arranca_igual(self, tmp_path, caplog):
        config = make_config(enabled=True, raw_root=str(tmp_path / "no_existe"))
        # El nivel se fija sobre ESTE logger y no sobre la raiz: varios estudios del repo
        # silencian `ai_trader` a ERROR al importarse, y sin esto el aviso no llegaria.
        with caplog.at_level("WARNING", logger="ai_trader.main"):
            radar = build_signal_radar(config, HistoricalClock(datetime.now(UTC)))

        assert radar.is_empty
        assert radar.features("BTC/USDT")["signal_coverage"] == 0.0
        assert any("coverage will be 0" in r.getMessage() for r in caplog.records)

    def test_sin_credenciales_las_puertas_se_saltan(self, tmp_path, monkeypatch):
        """La suma de todo: un sistema sin nada configurado opera exactamente igual que
        antes de que el radar existiera."""
        monkeypatch.chdir(tmp_path)
        service = FakeMarketDataService({"BTC/USDT": build_bars([100.0] * 80)})
        runner = build_runner(make_config(enabled=True, raw_root=str(tmp_path / "vacio")),
                              service, NullNotifier())
        strategy = runner.strategies[0]

        assert strategy._signals.is_empty
        assert not strategy._signals_active()
