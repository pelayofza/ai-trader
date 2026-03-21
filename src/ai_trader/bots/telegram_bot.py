from __future__ import annotations

import logging
import asyncio
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

    def get_history_report(self) -> str:
        ...

    def get_performance_report(self) -> str:
        ...

    def get_symbols_report(self) -> str:
        ...


@dataclass(slots=True)
class TelegramBotDependencies:
    market_data_service: MarketDataService
    runner: RunnerLike | None = None
    chat_id: int | None = None
    auto_cycle_enabled: bool = False
    cycle_running: bool = False


async def run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def run_cycle_in_thread(deps: TelegramBotDependencies):
    runner = deps.runner
    if runner is None:
        raise RuntimeError("Runner not configured.")

    if deps.cycle_running:
        return None

    deps.cycle_running = True
    try:
        return await asyncio.to_thread(runner.run_cycle)
    finally:
        deps.cycle_running = False


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
   
    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    deps.chat_id = update.effective_chat.id

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
        "/run_cycle\n"
        "/history\n"
        "/performance\n"
        "/symbols\n"
        "/autoon\n"
        "/autooff\n"
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
        report = await run_blocking(runner.get_positions_report)
        await update.message.reply_text(report)
    except Exception as exc:
        logger.exception("positions_command failed")
        await update.message.reply_text(f"Error loading positions: {exc}")


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()

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
        start = end - timedelta(days=7)
        bars = await run_blocking(
            deps.market_data_service.get_daily_bars,
            symbol,
            start,
            end,
        )

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
        await run_blocking(runner.pause)
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
        await run_blocking(runner.resume)
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

        if not results:
            await update.message.reply_text("Cycle executed. No new trades.")
            return

        await update.message.reply_text(build_trade_notification(runner, results))
    except Exception as exc:
        logger.exception("run_cycle_command failed")
        await update.message.reply_text(f"Error running cycle: {exc}")


def build_trade_notification(runner: RunnerLike, results) -> str:
    lines: list[str] = []
    lines.append("Trade update")
    lines.append(f"Executions: {len(results)}")

    try:
        performance = runner.get_performance_report().strip()
        if performance:
            lines.append("")
            lines.append("Performance:")
            lines.append(performance)
    except Exception as exc:
        logger.exception("build_trade_notification performance failed")
        lines.append("")
        lines.append(f"Performance unavailable: {exc}")

    try:
        history = runner.get_history_report().strip()
        if history:
            lines.append("")
            lines.append("Recent history:")
            lines.append(history)
    except Exception as exc:
        logger.exception("build_trade_notification history failed")
        lines.append("")
        lines.append(f"History unavailable: {exc}")

    return "\n".join(lines)


async def scheduled_run_cycle(context: ContextTypes.DEFAULT_TYPE) -> None:
    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner
    if runner is None:
        return
    if not deps.auto_cycle_enabled:
        return
    if deps.chat_id is None:
        return

    try:
        results = runner.run_cycle()

        if not results:
            logger.info("Scheduled cycle executed with no new trades.")
            return

        message = build_trade_notification(runner, results)
        await context.bot.send_message(
            chat_id=deps.chat_id,
            text=message,
        )
    except Exception as exc:
        logger.exception("scheduled_run_cycle failed")
        await context.bot.send_message(
            chat_id=deps.chat_id,
            text=f"Scheduled cycle failed: {exc}",
        )


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        report = await run_blocking(runner.get_risk_report)
        await update.message.reply_text(report)
    except Exception as exc:
        logger.exception("risk_command failed")
        await update.message.reply_text(f"Error loading risk report: {exc}")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        await update.message.reply_text(runner.get_history_report())
    except Exception as exc:
        logger.exception("history_command failed")
        await update.message.reply_text(f"Error loading history report: {exc}")


async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        report = await run_blocking(runner.get_performance_report)
        await update.message.reply_text(report)
    except Exception as exc:
        logger.exception("performance_command failed")
        await update.message.reply_text(f"Error loading performance report: {exc}")


async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    runner = deps.runner

    if runner is None:
        await update.message.reply_text("Runner not configured.")
        return

    try:
        report = await run_blocking(runner.get_symbols_report)
        await update.message.reply_text(report)
    except Exception as exc:
        logger.exception("symbols_command failed")
        await update.message.reply_text(f"Error loading symbols report: {exc}")


async def autoon_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    deps.auto_cycle_enabled = True
    deps.chat_id = update.effective_chat.id

    await update.message.reply_text("Automatic run_cycle enabled every 15 minutes.")


async def autooff_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    deps: TelegramBotDependencies = context.application.bot_data["deps"]
    deps.auto_cycle_enabled = False

    await update.message.reply_text("Automatic run_cycle disabled.")


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
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("performance", performance_command))
    app.add_handler(CommandHandler("symbols", symbols_command))
    app.add_handler(CommandHandler("autoon", autoon_command))
    app.add_handler(CommandHandler("autooff", autooff_command))

    if app.job_queue is not None:
        app.job_queue.run_repeating(
            scheduled_run_cycle,
            interval=900,
            first=10,
            name="scheduled_run_cycle",
        )

    return app