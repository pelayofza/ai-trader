from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ai_trader.app.runner import TradingRunner
from ai_trader.bots.telegram_bot import build_application
from ai_trader.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from ai_trader.data.market_data import MarketDataService
from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.execution.router import ExecutionRouter
from ai_trader.notifications.base import Notifier
from ai_trader.risk.engine import RiskEngine
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
        raise RuntimeError(
            "TELEGRAM_ALLOWED_CHAT_IDS is empty. Without it anyone who finds the bot "
            "could pause it or trigger trades. Send /start to the bot to learn your chat id."
        )

    return frozenset(ids)


def build_runner(
    config: AppConfig,
    market_data_service: MarketDataService,
    notifier: Notifier,
) -> TradingRunner:
    strategies = [
        build_strategy(spec.type, spec.params, strategy_id=spec.id)
        for spec in config.strategies
    ]

    paper_engine = PaperExecutionEngine(config.execution)

    return TradingRunner(
        config=config.runner,
        market_data_reader=market_data_service,
        strategies=strategies,
        risk_engine=RiskEngine(config.risk),
        execution_router=ExecutionRouter.paper(
            spot_engine=paper_engine,
            # Comparte el motor de papel para que las comisiones configuradas
            # tambien se apliquen a los mercados de prediccion.
            prediction_engine=PolymarketPaperExecutionEngine(paper_engine=paper_engine),
        ),
        notifier=notifier,
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
