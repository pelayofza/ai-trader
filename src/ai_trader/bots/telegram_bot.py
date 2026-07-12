from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from functools import wraps
from typing import Protocol

from telegram import Update  # type: ignore
from telegram.ext import Application, CommandHandler, ContextTypes  # type: ignore

from ai_trader.data.market_data import MarketDataService
from ai_trader.notifications.telegram import TelegramNotifier
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import utc_now

logger = logging.getLogger(__name__)

AUTO_CYCLE_INTERVAL_SECONDS = 900


class RunnerLike(Protocol):
    def get_status(self) -> str: ...
    def get_positions_report(self) -> str: ...
    def get_risk_report(self) -> str: ...
    def get_history_report(self) -> str: ...
    def get_performance_report(self) -> str: ...
    def get_symbols_report(self) -> str: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def run_cycle(self) -> list: ...


@dataclass(slots=True)
class TelegramBotDependencies:
    market_data_service: MarketDataService
    allowed_chat_ids: frozenset[int]
    runner: RunnerLike | None = None
    auto_cycle_enabled: bool = False
    # Cerrojo real de reentrada. Antes existia una bandera `cycle_running` en un
    # helper que nunca se llamaba, asi que un /run_cycle manual y el ciclo
    # programado podian solaparse y corromper el estado.
    cycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _deps(context: ContextTypes.DEFAULT_TYPE) -> TelegramBotDependencies:
    return context.application.bot_data["deps"]


def authorized(handler: Callable):
    """Rechaza cualquier chat que no este en la lista blanca."""

    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        allowed = _deps(context).allowed_chat_ids

        if chat_id not in allowed:
            logger.warning("Rejected Telegram command from unauthorized chat_id=%s", chat_id)
            await update.message.reply_text(
                f"Unauthorized. Add this chat id to TELEGRAM_ALLOWED_CHAT_IDS: {chat_id}"
            )
            return

        return await handler(update, context)

    return wrapper


def requires_runner(handler: Callable):
    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if _deps(context).runner is None:
            await update.message.reply_text("Runner not configured.")
            return
        return await handler(update, context)

    return wrapper


async def _reply_with(update: Update, func: Callable[[], str], label: str) -> None:
    """Ejecuta el informe fuera del event loop y responde. Nunca deja al bot mudo."""
    try:
        report = await asyncio.to_thread(func)
        await update.message.reply_text(report or f"No {label} available.")
    except Exception as exc:
        logger.exception("%s failed", label)
        await update.message.reply_text(f"Error building {label}: {exc}")


# --- comandos ---


@authorized
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "ai-trader online.\n\n"
        "/status /positions /risk /history /performance /symbols\n"
        "/price SYMBOL\n"
        "/pause /resume /run_cycle\n"
        "/autoon /autooff"
    )


@authorized
async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("pong")


@authorized
@requires_runner
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply_with(update, _deps(context).runner.get_status, "status")


@authorized
@requires_runner
async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply_with(update, _deps(context).runner.get_positions_report, "positions report")


@authorized
@requires_runner
async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply_with(update, _deps(context).runner.get_risk_report, "risk report")


@authorized
@requires_runner
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply_with(update, _deps(context).runner.get_history_report, "history report")


@authorized
@requires_runner
async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply_with(update, _deps(context).runner.get_performance_report, "performance report")


@authorized
@requires_runner
async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply_with(update, _deps(context).runner.get_symbols_report, "symbols report")


@authorized
async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /price SYMBOL")
        return

    symbol = context.args[0].strip().upper()
    service = _deps(context).market_data_service

    try:
        end = utc_now()
        bars = await asyncio.to_thread(
            service.get_daily_bars, symbol, end - timedelta(days=7), end
        )

        close = bar_schema.last_close(bars) if bars is not None else None
        if close is None:
            await update.message.reply_text(f"No data found for {symbol}.")
            return

        await update.message.reply_text(
            f"{symbol}\nClose: {close:,.4f}\nTimestamp: {bars.index[-1]}"
        )
    except Exception as exc:
        logger.exception("price_command failed for %s", symbol)
        await update.message.reply_text(f"Error loading price for {symbol}: {exc}")


@authorized
@requires_runner
async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await asyncio.to_thread(_deps(context).runner.pause)
    await update.message.reply_text("Runner paused.")


@authorized
@requires_runner
async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await asyncio.to_thread(_deps(context).runner.resume)
    await update.message.reply_text("Runner resumed.")


@authorized
@requires_runner
async def run_cycle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    deps = _deps(context)

    if deps.cycle_lock.locked():
        await update.message.reply_text("A cycle is already running. Skipping.")
        return

    await update.message.reply_text("Running cycle...")

    results = await _run_cycle(deps)
    if results is None:
        await update.message.reply_text("Cycle failed. Check logs.")
    elif not results:
        await update.message.reply_text("Cycle complete. No new trades.")
    else:
        await update.message.reply_text(f"Cycle complete. {len(results)} execution(s).")


@authorized
async def autoon_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _deps(context).auto_cycle_enabled = True
    minutes = AUTO_CYCLE_INTERVAL_SECONDS // 60
    await update.message.reply_text(f"Automatic cycle enabled (every {minutes} minutes).")


@authorized
async def autooff_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _deps(context).auto_cycle_enabled = False
    await update.message.reply_text("Automatic cycle disabled.")


# --- ciclo ---


async def _run_cycle(deps: TelegramBotDependencies) -> list | None:
    """
    Ejecuta un ciclo fuera del event loop y bajo cerrojo.

    run_cycle() recorre decenas de simbolos haciendo llamadas de red; ejecutarlo en
    el loop (como se hacia) dejaba el bot congelado durante todo el ciclo.
    """
    if deps.runner is None:
        return None

    async with deps.cycle_lock:
        try:
            return await asyncio.to_thread(deps.runner.run_cycle)
        except Exception:
            logger.exception("run_cycle failed")
            return None


async def scheduled_cycle(context: ContextTypes.DEFAULT_TYPE) -> None:
    deps = _deps(context)

    if not deps.auto_cycle_enabled or deps.runner is None:
        return

    if deps.cycle_lock.locked():
        logger.info("Scheduled cycle skipped: a cycle is already running.")
        return

    await _run_cycle(deps)


def build_application(
    token: str,
    allowed_chat_ids: frozenset[int],
    market_data_service: MarketDataService,
    runner_factory: Callable[[TelegramNotifier], RunnerLike | None],
) -> Application:
    """
    Construye la aplicacion.

    El runner se crea mediante una factoria porque necesita un TelegramNotifier, y el
    notifier necesita el bot y el event loop, que solo existen una vez construida la
    aplicacion.
    """
    app = Application.builder().token(token).build()

    deps = TelegramBotDependencies(
        market_data_service=market_data_service,
        allowed_chat_ids=allowed_chat_ids,
    )
    app.bot_data["deps"] = deps

    async def _wire_runner(application: Application) -> None:
        notifier = TelegramNotifier(
            bot=application.bot,
            chat_ids=list(allowed_chat_ids),
            loop=asyncio.get_running_loop(),
        )
        deps.runner = runner_factory(notifier)

        if deps.runner is None:
            logger.warning("No runner configured; bot will run in read-only mode.")

    app.post_init = _wire_runner

    commands = {
        "start": start_command,
        "ping": ping_command,
        "status": status_command,
        "positions": positions_command,
        "risk": risk_command,
        "history": history_command,
        "performance": performance_command,
        "symbols": symbols_command,
        "price": price_command,
        "pause": pause_command,
        "resume": resume_command,
        "run_cycle": run_cycle_command,
        "autoon": autoon_command,
        "autooff": autooff_command,
    }
    for name, handler in commands.items():
        app.add_handler(CommandHandler(name, handler))

    if app.job_queue is not None:
        app.job_queue.run_repeating(
            scheduled_cycle,
            interval=AUTO_CYCLE_INTERVAL_SECONDS,
            first=AUTO_CYCLE_INTERVAL_SECONDS,
            name="scheduled_cycle",
        )

    return app
