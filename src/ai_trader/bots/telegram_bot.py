from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from ai_trader.data.market_data import MarketDataService
from ai_trader.research.indicators.trend import trend_snapshot
from ai_trader.shared.schemas import Signal


logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunnerLike(Protocol):
    def get_status(self) -> str: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def run_cycle(self): ...


@dataclass(slots=True)
class TelegramBotDependencies:
    market_data_service: MarketDataService
    runner: RunnerLike | None = None


def format_price(symbol: str, bars) -> str:
    last = bars.iloc[-1]
    close = float(last["close"])
    timestamp = bars.index[-1]
    return (
        f"{symbol}\n"
        f"Close: {close:,.2f}\n"
        f"Timestamp: {timestamp}"
    )


def format_trend(symbol: str, snapshot) -> str:
    parts: list[str] = [f"{symbol} trend"]

    trend_value = getattr(snapshot, "trend", None)
    if trend_value is not None:
        parts.append(f"Trend: {trend_value}")

    close_value = getattr(snapshot, "close", None)
    if close_value is not None:
        parts.append(f"Close: {close_value:,.2f}")

    sma_50 = getattr(snapshot, "sma50", None)
    if sma_50 is not None:
        parts.append(f"SMA50: {sma_50:,.2f}")

    sma_200 = getattr(snapshot, "sma200", None)
    if sma_200 is not None:
        parts.append(f"SMA200: {sma_200:,.2f}")

    atr_14 = getattr(snapshot, "atr14", None)
    if atr_14 is not None:
        parts.append(f"ATR14: {atr_14:,.2f}")

    return "\n".join(parts)


def format_signal(signal: Signal) -> str:
    lines = [
        "Signal",
        f"Strategy: {signal.strategy_id}",
        f"Symbol: {signal.symbol}",
        f"Timeframe: {signal.timeframe}",
        f"Side: {signal.side.value.upper()}",
        f"Confidence: {signal.confidence:.2f}",
        f"Entry: {signal.entry_price:,.2f}",
    ]

    if signal.stop_loss is not None:
        lines.append(f"Stop loss: {signal.stop_loss:,.2f}")

    if signal.take_profit is not None:
        lines.append(f"Take profit: {signal.take_profit:,.2f}")

    if signal.reason:
        lines.append(f"Reason: {signal.reason}")

    return "\n".join(lines)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await update.message.reply_text(
        "Argos online.\n"
        "Commands:\n"
        "/ping\n"
        "/status\n"
        "/price SYMBOL\n"
        "/trend SYMBOL\n"
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


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    if not context.args:
        await update.message.reply_text("Usage: /price SYMBOL")
        return

    symbol = context.args[0].strip().upper()
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


async def trend_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    if not context.args:
        await update.message.reply_text("Usage: /trend SYMBOL")
        return

    symbol = context.args[0].strip().upper()
    deps: TelegramBotDependencies = context.application.bot_data["deps"]

    try:
        end = utc_now()
        start = end - timedelta(days=365)
        bars = deps.market_data_service.get_daily_bars(symbol, start, end)

        if bars is None or bars.empty:
            await update.message.reply_text(f"No data found for {symbol}.")
            return

        snapshot = trend_snapshot(bars)
        await update.message.reply_text(format_trend(symbol, snapshot))

    except Exception as exc:
        logger.exception("trend_command failed for %s", symbol)
        await update.message.reply_text(f"Error loading trend for {symbol}: {exc}")


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
    app.add_handler(CommandHandler("trend", trend_command))
    app.add_handler(CommandHandler("pause", pause_command))
    app.add_handler(CommandHandler("resume", resume_command))
    app.add_handler(CommandHandler("run_cycle", run_cycle_command))

    return app