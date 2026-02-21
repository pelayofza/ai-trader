from datetime import datetime, timedelta, timezone

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from ai_trader.data.market_data import MarketDataService
from ai_trader.data.formatting import format_price

from ai_trader.indicators.trend import trend_snapshot
from ai_trader.data.formatting import format_trend

from ai_trader.signals.trend_follow import generate_trend_follow_proposal
from ai_trader.data.formatting import format_trade_proposal


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm online.")

async def ping(update: Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def status(update: Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("All good, and you?")

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not context.args:
        await update.message.reply_text("Usage: /price TICKER (e.g. /price AAPL)")
        return

    symbol = context.args[0].strip().upper()

    try:
        provider = MarketDataService()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=180)

        df = provider.get_daily_bars(symbol, start, end)

        if df is None or df.empty:
            await update.message.reply_text(f"No data found for {symbol}.")
            return

        message = format_price(symbol, df)
        await update.message.reply_text(message)

    except Exception as e:
        await update.message.reply_text(f"Error: {e!r}")

async def trend_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not context.args:
        await update.message.reply_text("Usage: /trend TICKER (e.g. /trend AAPL)")
        return

    symbol = context.args[0].strip().upper()

    try:
        service = MarketDataService()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365)  # 1 año para SMA50 y ATR

        df = service.get_daily_bars(symbol, start, end)

        if df is None or df.empty:
            await update.message.reply_text(f"No data found for {symbol}.")
            return

        snap = trend_snapshot(df)
        msg = format_trend(symbol, snap)
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"Error: {e!r}")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /signal TICKER STRATEGY (e.g. /signal AAPL trend)")
        return

    symbol = context.args[0].strip().upper()
    strategy = context.args[1].strip().lower()

    try:
        service = MarketDataService()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365)

        df = service.get_daily_bars(symbol, start, end)

        if df is None or df.empty:
            await update.message.reply_text(f"No data found for {symbol}.")
            return

        if strategy == "trend":
            proposal = generate_trend_follow_proposal(symbol, df)
        else:
            await update.message.reply_text("Unknown strategy. Available: trend")
            return

        await update.message.reply_text(format_trade_proposal(proposal))

    except Exception as e:
        await update.message.reply_text(f"Error: {e!r}")

def build_application(token: str) -> Application:

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("trend", trend_command))
    app.add_handler(CommandHandler("signal", signal_command))

    return app