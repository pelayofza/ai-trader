from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

from ai_trader.app.journal import CycleJournal
from ai_trader.app.runner import TradingRunner
from ai_trader.app.state_store import JsonStateStore
from ai_trader.bots.telegram_bot import build_application
from ai_trader.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from ai_trader.data.market_data import MarketDataService
from ai_trader.execution.microstructure import BarLiquidityProvider
from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.execution.router import ExecutionRouter
from ai_trader.notifications.base import Notifier
from ai_trader.observation.regime import MarketRegimeProvider
from ai_trader.observation.signal_themes import ThemedSignalRadarProvider
from ai_trader.risk.engine import RiskEngine
from ai_trader.shared.clock import Clock, LiveClock
from ai_trader.signals.feed import load_frames
from ai_trader.strategies.registry import build_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_chat_ids(raw: str) -> frozenset[int]:
    ids = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid chat id in TELEGRAM_ALLOWED_CHAT_IDS: {chunk!r}"
            ) from exc

    if not ids:
        # El mensaje anterior mandaba usar /start para descubrir el chat id, y eso es
        # imposible en el PRIMER arranque: esta excepcion salta antes de construir la
        # aplicacion, asi que no hay bot al que escribir. Solo funciona cuando ya hay
        # algun id valido en la lista y el que pregunta es otro chat.
        raise RuntimeError(
            "TELEGRAM_ALLOWED_CHAT_IDS is empty. Without it anyone who finds the bot "
            "could pause it or trigger trades.\n"
            "To learn your chat id on a first run: send any message to your bot from "
            "Telegram, then ask the API for it (the bot does not need to be running):\n"
            "  (Invoke-RestMethod \"https://api.telegram.org/bot$env:TELEGRAM_BOT_TOKEN/"
            "getUpdates\").result.message.chat.id\n"
            "Put the number in TELEGRAM_ALLOWED_CHAT_IDS (comma-separated for several)."
        )

    return frozenset(ids)


class LiveUniverseBars(Mapping):
    """
    El universo entero visto como un `Mapping[str, DataFrame]`, en vivo.

    Existe por un hueco conocido: `MarketRegimeProvider` se construye sobre un diccionario
    `simbolo -> barras` porque en backtest ese diccionario ya esta cargado, y en vivo no
    existe. La consecuencia era que `attach_regime_provider` NO SE LLAMABA en produccion,
    asi que cualquier configuracion que el optimizador eligiera con filtros de regimen
    activos se comportaba distinto en paper que en el backtest que la selecciono. Eso es
    una discrepancia silenciosa entre lo que se mide y lo que se opera.

    El adaptador es la forma barata de cerrarlo: `regime.py` no cambia NI UNA LINEA. Solo
    necesita `for symbol in bars` y `bars.get(symbol)`, que es exactamente el contrato de
    `Mapping`, y que no haga falta tocarlo es la comprobacion de que el adaptador es
    correcto y no un parche con la forma de uno.

    Las barras las sirve `MarketDataService`, que ya cachea en disco: pedir la ventana de
    35 simbolos una vez por ciclo no es una descarga por simbolo y por dia.
    """

    def __init__(self, service: MarketDataService, symbols: Sequence[str], clock: Clock,
                 *, lookback_days: int) -> None:
        self._service = service
        self._symbols = tuple(symbols)
        self._clock = clock
        self._lookback_days = lookback_days

    def __iter__(self):
        return iter(self._symbols)

    def __len__(self) -> int:
        return len(self._symbols)

    def __getitem__(self, symbol: str):
        end = self._clock.now()
        bars = self._service.get_daily_bars(
            symbol, end - timedelta(days=self._lookback_days), end
        )
        if bars is None:
            # `Mapping.get` traduce esto a None, que es lo que el regimen ya trata como
            # "este simbolo no cuenta hoy". Un DataFrame vacio diria lo mismo y ademas
            # obligaria a que el consumidor supiera distinguirlos.
            raise KeyError(symbol)
        return bars


def build_signal_radar(config: AppConfig, clock: Clock) -> ThemedSignalRadarProvider:
    """
    El radar de senales para el proceso en vivo. Apagado -> vacio, y el sistema arranca.

    Con `[signals] enabled = false` (el defecto) no se toca el archivo: el radar sale vacio,
    la cobertura es 0 y las puertas se saltan. Encendido pero sin datos —sin credenciales,
    sin capturas todavia— pasa exactamente lo mismo, y se dice con un warning explicito en
    vez de dejar que parezca que hay senales detras de la decision.

    Publica los seis numeros de siempre MAS las quince features tematicas, y las primeras
    salen del metodo de la clase base sin tocar: las dos primitivas de precio no notan la
    diferencia y las seis tematicas encuentran su tema.
    """
    if not config.signals.enabled:
        logger.info("Signal radar disabled ([signals] enabled = false): running without it")
        return ThemedSignalRadarProvider(None, clock)

    frames = load_frames(raw_root=config.signals.raw_root or None)
    if not frames:
        logger.warning(
            "Signal radar ENABLED but the raw archive is empty (root=%s): coverage will be 0 "
            "and every signal gate will be skipped. Run `signals capture` first.",
            config.signals.raw_root or "data/signals_raw",
        )
    else:
        logger.info("Signal radar built from %s sources: %s", len(frames), sorted(frames))
    return ThemedSignalRadarProvider(frames, clock)


def build_runner(
    config: AppConfig,
    market_data_service: MarketDataService,
    notifier: Notifier,
) -> TradingRunner:
    strategies = [
        build_strategy(spec.type, spec.params, strategy_id=spec.id)
        for spec in config.strategies
    ]

    # Un unico reloj para el runner y para la liquidez: el estado de mercado con el que
    # se cobra el fill tiene que ser el del mismo instante en que se decide.
    clock = LiveClock()
    paper_engine = PaperExecutionEngine(
        config.execution,
        liquidity_provider=BarLiquidityProvider(market_data_service, clock),
    )

    # Los MISMOS dos ensambladores que inyecta el backtest, con el mismo bucle duck-typed.
    # Que aqui y en `backtest/engine.py::_build_runner` se enganchen igual es lo que hace
    # que una configuracion elegida por el optimizador signifique lo mismo en los dos
    # sitios; hasta hoy, el de regimen solo existia en backtest.
    providers = {
        "attach_regime_provider": MarketRegimeProvider(
            LiveUniverseBars(
                market_data_service,
                config.runner.symbols,
                clock,
                lookback_days=config.runner.lookback_days,
            ),
            clock,
        ),
        "attach_signal_provider": build_signal_radar(config, clock),
    }
    for strategy in strategies:
        for method, provider in providers.items():
            attach = getattr(strategy, method, None)
            if callable(attach):
                attach(provider)

    return TradingRunner(
        config=config.runner,
        market_data_reader=market_data_service,
        strategies=strategies,
        risk_engine=RiskEngine(config.risk),
        clock=clock,
        execution_router=ExecutionRouter.paper(
            spot_engine=paper_engine,
            # Comparte el motor de papel para que las comisiones configuradas
            # tambien se apliquen a los mercados de prediccion.
            prediction_engine=PolymarketPaperExecutionEngine(paper_engine=paper_engine),
        ),
        notifier=notifier,
        # Este es el UNICO sitio donde se construye el runner en vivo (el CLI headless
        # tambien pasa por aqui), asi que es donde se enchufan las dos piezas que solo
        # tienen sentido corriendo de verdad: el diario de ciclos y el aviso de que el
        # estado se recupero de una copia. En backtest ninguna de las dos se activa.
        state_store=JsonStateStore(notifier=notifier),
        journal=CycleJournal(),
    )


def main() -> None:
    load_dotenv()

    config_path = Path(os.getenv("AI_TRADER_CONFIG", DEFAULT_CONFIG_PATH))
    config = load_config(config_path)
    logger.info(
        "Loaded config | path=%s | symbols=%s | strategies=%s",
        config_path,
        len(config.runner.symbols),
        [spec.type for spec in config.strategies],
    )

    token = require_env("TELEGRAM_BOT_TOKEN")
    allowed_chat_ids = parse_chat_ids(os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""))

    market_data_service = MarketDataService()

    application = build_application(
        token=token,
        allowed_chat_ids=allowed_chat_ids,
        market_data_service=market_data_service,
        runner_factory=lambda notifier: build_runner(config, market_data_service, notifier),
    )

    logger.info("Starting Telegram bot")
    application.run_polling()


if __name__ == "__main__":
    main()
