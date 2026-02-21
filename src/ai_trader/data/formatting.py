from ai_trader.signals.models import TradeProposal

def format_price(symbol: str, df) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    close = float(last["Close"])
    prev_close = float(prev["Close"])

    change = close - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0

    high20 = float(df["High"].tail(20).max())
    low20 = float(df["Low"].tail(20).min())

    return (
        f"📈 {symbol}\n"
        f"Close: {close:.2f}\n"
        f"Day: {change:+.2f} ({change_pct:+.2f}%)\n"
        f"20D High/Low: {high20:.2f} / {low20:.2f}"
    )

def format_trend(symbol: str, snap: dict) -> str:
    if not snap.get("ok"):
        return f"📉 {symbol}\nNot ready: {snap.get('reason', 'unknown')}"

    close = snap["close"]
    sma50 = snap["sma50"]
    direction = snap["direction"]
    atr14 = snap["atr14"]
    atr_pct = snap["atr14_pct"]

    atr_line = "ATR14: n/a"
    if atr14 is not None and atr_pct is not None:
        atr_line = f"ATR14: {atr14:.2f} ({atr_pct:.2f}%)"

    emoji = "🟢" if direction == "UP" else "🔴"

    return (
        f"{emoji} {symbol} Trend\n"
        f"Close: {close:.2f}\n"
        f"SMA50: {sma50:.2f}\n"
        f"Direction: {direction}\n"
        f"{atr_line}"
    )

def format_trade_proposal(p: TradeProposal) -> str:
    if p.side == "NONE":
        return f"🟡 {p.symbol} Signal\nNo trade.\nReason: {p.reason}"

    return (
        f"🟢 {p.symbol} Signal ({p.side})\n"
        f"Entry: {p.entry:.2f}\n"
        f"Stop: {p.stop:.2f}\n"
        f"Target: {p.target:.2f}\n"
        f"R/R: {p.rr:.2f}\n"
        f"Reason: {p.reason}"
    )

