from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from telegram import Update # type: ignore
from telegram.ext import Application, CommandHandler, ContextTypes # type: ignore

from ai_trader.data.market_data import MarketDataService
from ai_trader.shared.schemas import Signal


logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunnerLike(Protocol):
    def get_status(self) -> str:
        ...

    def get_positions_report(self) -> str:
        ...

    def get_risk_report(self) -> str:
        ...

    def pause(self) -> None:
        ...

    def resume(self) -> None:
        ...

    def run_cycle(self):
        ...


@dataclass(slots=True)
class TelegramBotDependencies:
    market_data_service: MarketDataService
    runner: RunnerLike | None = None


def format_price(symbol: str, bars) -> str:
    last = bars.iloc[-1]

    close_value = None
    for candidate in ("close", "Close"):
        if candidate in bars.columns:
            close_value = float(last[candidate])
            break

    if close_value is None:
        raise KeyError(f"Could not find close column. Available columns: {list(bars.columns)}")

    timestamp = bars.index[-1]
    return (
        f"{symbol}\n"
        f"Close: {close_value:,.2f}\n"
        f"Timestamp: {timestamp}"
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await update.message.reply_text(
        "Argos online.\n"
        "Commands:\n"
        "/ping\n"
        "/status\n"
        "/price SYMBOL\n"
        "/positions\n"
        "/risk\n"
        "/pause\n"
        "/resume\n"
        "/run_cycle"
    )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await update.message.reply_text("pong")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Bot online. Runner not configured.")
        return

    await update.message.reply_text(runner.get_status())


async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        await update.message.reply_text(runner.get_positions_report())
    except Exception as exc:
        logger.exception("positions_command failed")
        await update.message.reply_text(f"Error loading positions: {exc}")


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    if not context.args:
        await update.message.reply_text("Usage: /price SYMBOL")
        return

    symbol = normalize_symbol(context.args[0])
    deps: TelegramBotDependencies = context.application.bot_data["deps"]

    try:
        end = utc_now()
        start = end - timedelta(days=180)
        bars = deps.market_data_service.get_daily_bars(symbol, start, end)

        if bars is None or bars.empty:
            await update.message.reply_text(f"No data found for {symbol}.")
            return

        await update.message.reply_text(format_price(symbol, bars))

    except Exception as exc:
        logger.exception("price_command failed for %s", symbol)
        await update.message.reply_text(f"Error loading price for {symbol}: {exc}")


async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        runner.pause()
        await update.message.reply_text("Runner paused.")
    except Exception as exc:
        logger.exception("pause_command failed")
        await update.message.reply_text(f"Error pausing runner: {exc}")


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        runner.resume()
        await update.message.reply_text("Runner resumed.")
    except Exception as exc:
        logger.exception("resume_command failed")
        await update.message.reply_text(f"Error resuming runner: {exc}")


async def run_cycle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        results = runner.run_cycle()
        await update.message.reply_text(
            f"Cycle executed. Execution results: {len(results)}"
        )
    except Exception as exc:
        logger.exception("run_cycle_command failed")
        await update.message.reply_text(f"Error running cycle: {exc}")


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        await update.message.reply_text(runner.get_risk_report())
    except Exception as exc:
        logger.exception("risk_command failed")
        await update.message.reply_text(f"Error loading risk report: {exc}")


def build_application(
    token: str,
    market_data_service: MarketDataService | None = None,
    runner: RunnerLike | None = None,
) -> Application:
    app = Application.builder().token(token).build()

    deps = TelegramBotDependencies(
        market_data_service=market_data_service or MarketDataService(),
        runner=runner,
    )
    app.bot_data["deps"] = deps

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("ping", ping_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("positions", positions_command))
    app.add_handler(CommandHandler("risk", risk_command))
    app.add_handler(CommandHandler("pause", pause_command))
    app.add_handler(CommandHandler("resume", resume_command))
    app.add_handler(CommandHandler("run_cycle", run_cycle_command))

    return app