from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ai_trader.app.journal import CycleJournal
from ai_trader.app.runner import RunnerConfig, SymbolCycleDiagnostics, TradingRunner
from ai_trader.app.state_store import JsonStateStore
from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.execution.router import ExecutionRouter
from ai_trader.risk.engine import RiskEngine
from ai_trader.shared.instruments import AssetClass, OutcomeToken, PredictionMarket, Venue
from ai_trader.shared.schemas import PositionStatus, Side, Signal
from ai_trader.strategies.polymarket_threshold import (
    PolymarketThresholdConfig,
    PolymarketThresholdStrategy,
)
from tests.conftest import build_bars


class FakeMarketData:
    """Datos de mercado en memoria. Ningun test toca la red."""

    def __init__(
        self,
        bars: dict[str, pd.DataFrame] | None = None,
        markets: dict[str, PredictionMarket] | None = None,
        midpoints: dict[str, float] | None = None,
    ) -> None:
        self.bars = bars or {}
        self.markets = markets or {}
        self.midpoints = midpoints or {}

    def get_daily_bars(self, symbol, start, end):
        return self.bars.get(symbol)

    def get_prediction_market(self, slug):
        return self.markets.get(slug)

    def get_prediction_midpoint(self, token_id):
        return self.midpoints.get(token_id)

    def detect_asset_class(self, symbol):
        if symbol.upper().startswith("PM::"):
            return AssetClass.PREDICTION
        return AssetClass.CRYPTO if "/" in symbol else AssetClass.STOCK


class SpyRiskEngine(RiskEngine):
    """Envuelve el motor real para contar cuantas senales lo atraviesan."""

    def __init__(self, limits):
        super().__init__(limits)
        self.evaluated = []

    def evaluate(self, signal, portfolio_state):
        self.evaluated.append(signal)
        return super().evaluate(signal, portfolio_state)


def build_runner(tmp_path, limits, market_data, strategies, journal=None, **config_overrides):
    paper_engine = PaperExecutionEngine(PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0))
    risk_engine = SpyRiskEngine(limits)

    config_kwargs = {"symbols": ["BTC/USDT"], "symbol_cooldown_hours": 0, **config_overrides}

    runner = TradingRunner(
        config=RunnerConfig(**config_kwargs),
        market_data_reader=market_data,
        strategies=strategies,
        risk_engine=risk_engine,
        execution_router=ExecutionRouter.paper(
            spot_engine=paper_engine,
            prediction_engine=PolymarketPaperExecutionEngine(paper_engine=paper_engine),
        ),
        state_store=JsonStateStore(tmp_path / "state.json"),
        journal=journal,
    )
    return runner, risk_engine


def prediction_market(yes_price: float) -> PredictionMarket:
    return PredictionMarket(
        market_id="1",
        question="Will it happen?",
        slug="test-market",
        active=True,
        closed=False,
        archived=False,
        enable_order_book=True,
        outcomes=[
            OutcomeToken(outcome="Yes", token_id="tok-yes", price=yes_price),
            OutcomeToken(outcome="No", token_id="tok-no", price=1.0 - yes_price),
        ],
    )


class TestPredictionOrdersGoThroughRisk:
    """
    El agujero mas grave del sistema anterior: las ordenes de Polymarket se contaban
    como aprobadas por riesgo sin llamar nunca al motor. No tenian limite de tamano,
    ni de exposicion, ni stop loss.
    """

    @pytest.fixture
    def strategy(self):
        return PolymarketThresholdStrategy(
            PolymarketThresholdConfig(
                slug="test-market",
                outcome="yes",
                buy_below_price=0.40,
                confidence=0.80,
            )
        )

    def test_prediction_signal_is_evaluated_by_the_risk_engine(
        self, tmp_path, limits, strategy
    ):
        market_data = FakeMarketData(
            markets={"test-market": prediction_market(0.30)},
            midpoints={"tok-yes": 0.30, "tok-no": 0.70},
        )
        runner, risk = build_runner(
            tmp_path, limits, market_data, [strategy], symbols=["PM::test-market"]
        )

        runner.run_cycle()

        assert len(risk.evaluated) == 1
        assert risk.evaluated[0].asset_class == AssetClass.PREDICTION

    def test_prediction_position_gets_a_stop_loss_from_risk(self, tmp_path, limits, strategy):
        market_data = FakeMarketData(
            markets={"test-market": prediction_market(0.30)},
            midpoints={"tok-yes": 0.30, "tok-no": 0.70},
        )
        runner, _ = build_runner(
            tmp_path, limits, market_data, [strategy], symbols=["PM::test-market"]
        )

        runner.run_cycle()
        positions = runner.get_positions()

        assert len(positions) == 1
        assert positions[0].venue == Venue.POLYMARKET
        # Antes esto era None: las posiciones de prediccion no tenian salida definida.
        assert positions[0].stop_loss is not None
        assert positions[0].take_profit is not None

    def test_risk_can_reject_a_prediction_signal(self, tmp_path, limits, strategy):
        limits.min_confidence_per_trade = 0.95  # por encima de la confianza de la estrategia
        market_data = FakeMarketData(
            markets={"test-market": prediction_market(0.30)},
            midpoints={"tok-yes": 0.30, "tok-no": 0.70},
        )
        runner, _ = build_runner(
            tmp_path, limits, market_data, [strategy], symbols=["PM::test-market"]
        )

        results = runner.run_cycle()

        assert results == []
        assert runner.get_positions() == []


class RecordingStrategy:
    """Estrategia que solo apunta cuando la llaman. No decide nada."""

    strategy_id = "recording"

    def __init__(self, events: list[str], signal_factory=None) -> None:
        self.events = events
        self._signal_factory = signal_factory

    def supports_symbol(self, symbol: str) -> bool:
        self.events.append("supports_symbol")
        return True

    def generate_signal(self, symbol, context):
        self.events.append("generate_signal")
        return None if self._signal_factory is None else self._signal_factory(symbol)


class RecordingMarketData(FakeMarketData):
    def __init__(self, events: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.events = events

    def get_daily_bars(self, symbol, start, end):
        self.events.append("load_bars")
        return super().get_daily_bars(symbol, start, end)


class TestGuardOrder:
    """
    EL ORDEN DE LAS GUARDAS DE `_process_symbol`, CONGELADO.

    Hoy es: posicion abierta -> enfriamiento -> contexto de mercado -> estrategia ->
    riesgo -> ejecucion. No estaba testeado, y es un orden con consecuencias medibles:

    - Las dos guardas de ESTADO van antes de cargar barras y antes de llamar a ninguna
      estrategia, asi que un simbolo ya abierto no cuesta ni una descarga ni un calculo.
    - El RIESGO es la ultima puerta antes de ejecutar y no se puede esquivar: toda senal,
      venga de donde venga, pasa por el. Ese agujero ya existio una vez (las ordenes de
      prediccion se contaban como aprobadas sin evaluarlas).
    - Cualquier puerta NUEVA de una estrategia -regimen, senales- vive DENTRO de
      `generate_signal`, es decir, despues del contexto y antes del riesgo. Nunca puede
      colarse por delante de las guardas de estado ni por detras del riesgo.
    """

    def _harness(self, tmp_path, limits, *, bars=None, signal=False, **overrides):
        """Un runner cuyo camino completo queda apuntado en `events`, en orden.

        Se ejercita `_process_symbol` DIRECTAMENTE: el ciclo entero mete antes el
        mantenimiento de posiciones abiertas (que tambien lee precios) y taparia
        exactamente lo que aqui se quiere ver.
        """
        events: list[str] = []
        market_data = RecordingMarketData(events, bars=bars or {})

        def make_signal(symbol: str) -> Signal:
            return Signal(
                strategy_id="recording",
                symbol=symbol,
                timeframe="1d",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                side=Side.BUY,
                confidence=0.80,
                entry_price=100.0,
            )

        strategy = RecordingStrategy(events, make_signal if signal else None)
        runner, risk = build_runner(tmp_path, limits, market_data, [strategy], **overrides)

        original_evaluate = risk.evaluate
        original_execute = runner.execution_router.execute

        def evaluate(signal, portfolio_state):
            events.append("risk")
            return original_evaluate(signal=signal, portfolio_state=portfolio_state)

        def execute(order_request, reference_price=None):
            events.append("execute")
            return original_execute(order_request, reference_price=reference_price)

        risk.evaluate = evaluate
        runner.execution_router.execute = execute
        return runner, events

    def _process(self, runner, symbol: str = "BTC/USDT"):
        return runner._process_symbol(symbol, SymbolCycleDiagnostics(symbol=symbol))

    def test_el_orden_completo_es_el_declarado(self, tmp_path, limits):
        runner, events = self._harness(
            tmp_path, limits, bars={"BTC/USDT": build_bars([100.0] * 60)}, signal=True
        )

        self._process(runner)

        assert events == ["load_bars", "supports_symbol", "generate_signal", "risk", "execute"]

    def test_una_posicion_abierta_corta_antes_de_pedir_barras(
        self, tmp_path, limits, make_position
    ):
        runner, events = self._harness(
            tmp_path, limits, bars={"BTC/USDT": build_bars([100.0] * 60)}, signal=True
        )
        runner.state.positions.append(make_position(symbol="BTC/USDT"))

        assert self._process(runner) == []
        assert events == []

    def test_el_enfriamiento_va_antes_del_contexto(self, tmp_path, limits, make_position):
        runner, events = self._harness(
            tmp_path, limits,
            bars={"BTC/USDT": build_bars([100.0] * 60)},
            signal=True,
            symbol_cooldown_hours=48,
        )
        closed = make_position(symbol="BTC/USDT")
        closed.status = PositionStatus.CLOSED
        closed.closed_at = runner.clock.now()
        runner.state.positions.append(closed)

        assert self._process(runner) == []
        assert events == []

    def test_sin_barras_no_se_llama_a_ninguna_estrategia(self, tmp_path, limits):
        runner, events = self._harness(tmp_path, limits)  # ningun simbolo devuelve barras

        assert self._process(runner) == []
        assert events == ["load_bars"]

    def test_el_riesgo_es_la_ultima_puerta_antes_de_ejecutar(self, tmp_path, limits):
        limits.min_confidence_per_trade = 0.99  # el riesgo rechaza todo
        runner, events = self._harness(
            tmp_path, limits, bars={"BTC/USDT": build_bars([100.0] * 60)}, signal=True
        )

        assert self._process(runner) == []
        assert events == ["load_bars", "supports_symbol", "generate_signal", "risk"]
        assert runner.get_positions() == []


class TestPositionLifecycle:
    def test_opening_a_position_records_the_entry_fees(self, tmp_path, limits):
        from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy

        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])

        runner.run_cycle()
        positions = runner.get_positions()

        assert len(positions) == 1
        # Antes las comisiones se calculaban y se tiraban a la basura.
        assert positions[0].entry_fees_usd > 0


class TestStatePersistence:
    def test_state_survives_a_restart(self, tmp_path, limits):
        from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy

        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})

        runner, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])
        runner.run_cycle()
        assert len(runner.get_positions()) == 1

        # Un runner nuevo apuntando al mismo estado debe reencontrar la posicion.
        reloaded, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])

        assert len(reloaded.get_positions()) == 1
        assert reloaded.get_positions()[0].symbol == "BTC/USDT"

    def test_paused_runner_does_not_trade(self, tmp_path, limits):
        from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy

        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])

        runner.pause()
        results = runner.run_cycle()

        assert results == []
        assert runner.get_positions() == []


class TestCycleJournal:
    """
    QUE EL CICLO DEJE HUELLA.

    El estado guarda la foto (que posiciones hay); el diario guarda la pelicula (que se
    decidio, con que precio y cuanto deslizamiento se cobro). Aqui se comprueba que la
    pelicula la rueda el ciclo REAL y no un formateador aparte: si el runner dejara de
    registrar un rechazo del riesgo o el slippage de un fill, estos tests caen.
    """

    def _momentum(self):
        from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy

        return CryptoMomentumStrategy()

    def test_un_ciclo_deja_exactamente_una_linea(self, tmp_path, limits):
        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(
            tmp_path, limits, market_data, [self._momentum()], journal=journal
        )

        runner.run_cycle()
        runner.run_cycle()

        records = journal.read()
        assert len(records) == 2
        assert records[0]["status"] == "ran"

    def test_la_linea_lleva_la_senal_la_decision_y_el_slippage_cobrado(self, tmp_path, limits):
        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(
            tmp_path, limits, market_data, [self._momentum()], journal=journal
        )

        runner.run_cycle()
        symbol = journal.read()[0]["symbols"][0]

        assert symbol["symbol"] == "BTC/USDT"
        assert 0.0 < symbol["signals"][0]["confidence"] <= 1.0
        assert symbol["risk"][0]["approved"] is True
        # El precio con el que se DECIDIO, para poder compararlo con el de llenado.
        assert symbol["orders"][0]["reference_price"] > 0
        fill = symbol["fills"][0]
        assert fill["filled_price"] > 0
        assert fill["fees_usd"] > 0
        # El deslizamiento REAL del modelo de microestructura, no el plano del config.
        assert fill["slippage_bps"] > 0

    def test_un_rechazo_del_riesgo_queda_registrado_con_su_motivo(self, tmp_path, limits):
        limits.min_confidence_per_trade = 0.99
        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(
            tmp_path, limits, market_data, [self._momentum()], journal=journal
        )

        runner.run_cycle()
        decision = journal.read()[0]["symbols"][0]["risk"][0]

        assert decision["approved"] is False
        assert "confidence" in decision["reason"].lower()

    def test_el_cierre_tambien_registra_su_deslizamiento(self, tmp_path, limits):
        """Salir paga deslizamiento igual que entrar. Sin este bloque, la mitad del coste
        de cada operación quedaría fuera de la medición."""
        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(
            tmp_path, limits, market_data, [self._momentum()], journal=journal
        )
        runner.run_cycle()
        assert len(runner.get_positions()) == 1

        # Un desplome por debajo del stop fuerza la salida en el ciclo siguiente.
        market_data.bars["BTC/USDT"] = build_bars(closes + [1.0])
        runner.run_cycle()

        record = journal.read()[-1]
        assert record["closed"], "la posicion tenia que haberse cerrado"
        assert record["closed"][0]["exit_fees_usd"] > 0
        exit_fill = record["exits"][0]
        assert exit_fill["symbol"] == "BTC/USDT"
        assert exit_fill["slippage_bps"] > 0
        assert exit_fill["filled_price"] > 0

    def test_sin_diario_no_se_acumulan_salidas_en_memoria(self, tmp_path, limits):
        """El buffer de salidas solo se llena si hay diario: el backtest cierra miles de
        posiciones por ventana y no puede ir dejando dicts por el camino."""
        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(tmp_path, limits, market_data, [self._momentum()])

        runner.run_cycle()
        market_data.bars["BTC/USDT"] = build_bars(closes + [1.0])
        runner.run_cycle()

        assert runner._exit_fills == []

    def test_un_ciclo_pausado_deja_linea_pero_no_toca_la_red(self, tmp_path, limits):
        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars([100.0] * 60)})
        runner, _ = build_runner(
            tmp_path, limits, market_data, [self._momentum()], journal=journal
        )
        runner.pause()

        runner.run_cycle()
        record = journal.read()[0]

        assert record["status"] == "paused"
        assert record["marked_to_market"] is False
        assert record["net_pnl_usd"] is None

    def test_sin_diario_el_ciclo_no_escribe_nada(self, tmp_path, limits, monkeypatch):
        """El defecto es NullJournal: el backtest conduce este mismo runner miles de
        veces por ventana y no puede dejar rastro en disco."""
        monkeypatch.chdir(tmp_path)
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars([100.0] * 60)})
        runner, _ = build_runner(tmp_path, limits, market_data, [self._momentum()])

        runner.run_cycle()

        assert not (tmp_path / "data" / "live").exists()

    def test_un_diario_que_falla_no_tumba_el_ciclo(self, tmp_path, limits):
        """El diario es auditoria, no operativa: un disco lleno no puede impedir que se
        abra o se cierre una posicion."""

        class BrokenJournal:
            def append(self, record):
                raise OSError("no space left on device")

        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(
            tmp_path, limits, market_data, [self._momentum()], journal=BrokenJournal()
        )

        runner.run_cycle()

        assert len(runner.get_positions()) == 1

    def test_la_watchlist_vacia_no_consulta_ningun_mercado(self, tmp_path, limits):
        """La watchlist es de segunda prioridad y por defecto no existe: con la lista
        vacia el ciclo tiene que ser identico al de antes, sin una sola llamada."""
        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars([100.0] * 60)})
        runner, _ = build_runner(
            tmp_path, limits, market_data, [_NoopStrategy()], journal=journal
        )

        runner.run_cycle()

        assert journal.read()[0]["watchlist"] == []

    def test_la_watchlist_anota_el_midpoint_sin_operar_el_mercado(self, tmp_path, limits):
        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        market_data = FakeMarketData(
            bars={"BTC/USDT": build_bars([100.0] * 60)},
            markets={"test-market": prediction_market(0.30)},
            midpoints={"tok-yes": 0.31, "tok-no": 0.69},
        )
        runner, _ = build_runner(
            tmp_path,
            limits,
            market_data,
            [_NoopStrategy()],
            journal=journal,
            prediction_watchlist=["test-market"],
        )

        runner.run_cycle()
        watched = journal.read()[0]["watchlist"]

        assert {w["outcome"] for w in watched} == {"Yes", "No"}
        assert watched[0]["midpoint"] == pytest.approx(0.31)
        # Observar no es operar: ni posicion, ni orden, ni simbolo nuevo en el universo.
        assert runner.get_positions() == []
        assert runner.config.symbols == ["BTC/USDT"]

    def test_un_mercado_de_la_watchlist_que_falla_no_corta_el_ciclo(self, tmp_path, limits):
        class ExplodingMarketData(FakeMarketData):
            def get_prediction_market(self, slug):
                raise RuntimeError("gamma caido")

        journal = CycleJournal(tmp_path / "live" / "cycles.jsonl")
        market_data = ExplodingMarketData(bars={"BTC/USDT": build_bars([100.0] * 60)})
        runner, _ = build_runner(
            tmp_path,
            limits,
            market_data,
            [_NoopStrategy()],
            journal=journal,
            prediction_watchlist=["test-market"],
        )

        runner.run_cycle()
        watched = journal.read()[0]["watchlist"]

        assert watched[0]["slug"] == "test-market"
        assert "gamma caido" in watched[0]["error"]


class TestReportsDoNotCrash:
    def test_risk_report_survives_a_missing_price(self, tmp_path, limits, make_position):
        """
        get_risk_report hacia `size * last_price` sin comprobar None, asi que reventaba
        con TypeError en cuanto un simbolo no devolvia precio.
        """
        market_data = FakeMarketData()  # ningun simbolo devuelve barras
        runner, _ = build_runner(tmp_path, limits, market_data, [_NoopStrategy()])

        runner.state.positions.append(make_position(symbol="BTC/USDT"))

        report = runner.get_risk_report()

        assert "Open positions: 1" in report
        assert "BTC/USDT" in report

    def test_reports_are_empty_but_valid_with_no_positions(self, tmp_path, limits):
        runner, _ = build_runner(tmp_path, limits, FakeMarketData(), [_NoopStrategy()])

        assert "No open positions." in runner.get_positions_report()
        assert "No closed positions yet." in runner.get_history_report()
        assert "Closed trades: 0" in runner.get_performance_report()


class _NoopStrategy:
    strategy_id = "noop"

    def supports_symbol(self, symbol: str) -> bool:
        return True

    def generate_signal(self, symbol, context):
        return None
