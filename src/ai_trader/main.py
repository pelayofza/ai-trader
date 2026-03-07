from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from ai_trader.bots.telegram_bot import build_application
from ai_trader.data.market_data import MarketDataService
from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine
from ai_trader.risk.engine import RiskEngine, RiskLimits
from ai_trader.app.runner import RunnerConfig, TradingRunner
from ai_trader.strategies import CryptoMomentumStrategy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_runner(market_data_service: MarketDataService) -> TradingRunner | None:
    """
    Temporary bootstrap.

    Returns None until at least one strategy is migrated to the new
    `generate_signal(...) -> Signal | None` contract.
    """
    strategies = [
        CryptoMomentumStrategy(),
    ]

    if not strategies:
        logger.warning("No strategies configured yet. Runner will not be created.")
        return None

    risk_engine = RiskEngine(
        RiskLimits(
            max_position_size_usd=2_000.0,
            max_open_positions=5,
            max_symbol_exposure_usd=1_000.0,
            max_total_exposure_usd=5_000.0,
            max_daily_loss_usd=150.0,
            min_confidence_per_trade=0.65,
        )
    )

    execution_engine = PaperExecutionEngine(
        PaperExecutionConfig(
            fee_rate=0.001,
            slippage_bps=5.0,
        )
    )

    runner = TradingRunner(
        config=RunnerConfig(
            symbols=[
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "BNB/USDT",
                "XRP/USDT",
                "ADA/USDT",
                "DOGE/USDT",
                "AVAX/USDT",
                "DOT/USDT",
                "MATIC/USDT",
                "LINK/USDT",
                "LTC/USDT",
                "UNI/USDT",
                "ATOM/USDT",
                "NEAR/USDT",
                "APT/USDT",
                "ARB/USDT",
                "OP/USDT",
                "INJ/USDT",
                "FIL/USDT",
                "ETC/USDT",
                "AAVE/USDT",
                "SUI/USDT",
                "SEI/USDT",
                "TIA/USDT",
            ],
            lookback_days=180,
            max_holding_days=10,
        ),
        market_data_reader=market_data_service,
        strategies=strategies,
        risk_engine=risk_engine,
        execution_engine=execution_engine,
    )
    return runner


def main() -> None:
    load_dotenv()

    telegram_bot_token = load_required_env("TELEGRAM_BOT_TOKEN")

    market_data_service = MarketDataService()
    runner = build_runner(market_data_service)

    application = build_application(
        token=telegram_bot_token,
        market_data_service=market_data_service,
        runner=runner,
    )

    logger.info("Starting Telegram bot")
    application.run_polling()


if __name__ == "__main__":
    main()