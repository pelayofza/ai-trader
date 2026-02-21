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