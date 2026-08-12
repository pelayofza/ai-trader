from __future__ import annotations

from collections.abc import Sequence

from ai_trader.app.accounting import PriceLookup, realized_pnl_usd, unrealized_pnl_usd
from ai_trader.risk.engine import RiskLimits
from ai_trader.shared.schemas import Position, PositionStatus

UNAVAILABLE = "unavailable"


def _fmt(value: float | None, spec: str = ",.2f") -> str:
    return UNAVAILABLE if value is None else format(value, spec)


class ReportBuilder:
    """
    Formateo de informes para consumo humano (hoy, Telegram).

    Vive aparte del orquestador a proposito: presentar no es orquestar, y estos
    informes eran el 40% de runner.py.
    """

    def __init__(
        self,
        positions: Sequence[Position],
        daily_realized_pnl_usd: float,
        is_paused: bool,
        limits: RiskLimits,
        price_lookup: PriceLookup,
    ) -> None:
        self.positions = list(positions)
        self.daily_realized_pnl_usd = daily_realized_pnl_usd
        self.is_paused = is_paused
        self.limits = limits
        self.price_lookup = price_lookup

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    @property
    def closed_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == PositionStatus.CLOSED]

    def status(self) -> str:
        return (
            f"paused={self.is_paused} | "
            f"open_positions={len(self.open_positions)} | "
            f"daily_realized_pnl_usd={self.daily_realized_pnl_usd:,.2f} | "
            f"closed_positions={len(self.closed_positions)}"
        )

    def positions_report(self) -> str:
        open_positions = self.open_positions
        if not open_positions:
            return "No open positions."

        lines = [f"Open positions: {len(open_positions)}"]
        total_unrealized = 0.0

        for position in open_positions:
            last_price = self.price_lookup(position)

            lines.append("")
            lines.append(f"Symbol: {position.symbol}")
            lines.append(f"Side: {position.side.value.upper()}")
            lines.append(f"Size: {position.size:.8f}")
            lines.append(f"Entry: {position.entry_price:,.4f}")
            lines.append(f"Notional at entry: {position.notional_value:,.2f} USD")
            lines.append(f"Fees paid: {position.entry_fees_usd:,.4f} USD")

            if last_price is None:
                lines.append(f"Last price: {UNAVAILABLE}")
                lines.append(f"Unrealized PnL: {UNAVAILABLE}")
            else:
                net_pnl = position.net_pnl_at(last_price)
                total_unrealized += net_pnl
                pnl_pct = net_pnl / position.notional_value * 100.0

                lines.append(f"Last price: {last_price:,.4f}")
                lines.append(f"Unrealized PnL (net of fees): {net_pnl:,.2f} USD")
                lines.append(f"Unrealized PnL %: {pnl_pct:,.2f}%")

            lines.append(f"Strategy: {position.strategy_id}")

            if position.venue is not None:
                lines.append(f"Venue: {position.venue.value}")
            if position.asset_class is not None:
                lines.append(f"Asset class: {position.asset_class.value}")
            if position.outcome:
                lines.append(f"Outcome: {position.outcome}")

            lines.append(f"Stop loss: {_fmt(position.stop_loss, ',.4f')}")
            lines.append(f"Take profit: {_fmt(position.take_profit, ',.4f')}")
            lines.append(f"Opened at: {position.opened_at.isoformat()}")

        lines.append("")
        lines.append(f"Total unrealized PnL (net of fees): {total_unrealized:,.2f} USD")

        return "\n".join(lines)

    def risk_report(self) -> str:
        limits = self.limits
        open_positions = self.open_positions

        # Se marca a mercado cuando hay precio; si no, se cae al nocional de entrada.
        # Antes esto multiplicaba size por None y reventaba con TypeError.
        exposure_by_symbol: dict[str, float] = {}
        for position in open_positions:
            last_price = self.price_lookup(position)
            exposure = (
                position.size * last_price
                if last_price is not None
                else position.notional_value
            )
            exposure_by_symbol[position.symbol] = (
                exposure_by_symbol.get(position.symbol, 0.0) + exposure
            )

        total_exposure = sum(exposure_by_symbol.values())

        lines = [
            f"Runner paused: {self.is_paused}",
            f"Open positions: {len(open_positions)} / {limits.max_open_positions}",
            (
                f"Daily realized PnL: {self.daily_realized_pnl_usd:,.2f} USD / "
                f"-{limits.max_daily_loss_usd:,.2f} USD limit"
            ),
            (
                f"Total current exposure: {total_exposure:,.2f} USD / "
                f"{limits.max_total_exposure_usd:,.2f} USD"
            ),
            f"Max position size: {limits.max_position_size_usd:,.2f} USD",
            f"Minimum confidence required: {limits.min_confidence_per_trade:.2f}",
            f"Default stop loss: {limits.default_stop_loss_pct:.2f}%",
            f"Default take profit: {limits.default_take_profit_pct:.2f}%",
            "",
            "Current exposure by symbol:",
        ]

        if not exposure_by_symbol:
            lines.append("None")
        else:
            for symbol in sorted(exposure_by_symbol):
                lines.append(
                    f"{symbol}: {exposure_by_symbol[symbol]:,.2f} USD / "
                    f"{limits.max_symbol_exposure_usd:,.2f} USD"
                )

        return "\n".join(lines)

    def history_report(self, limit: int = 10) -> str:
        closed = self.closed_positions
        if not closed:
            return "No closed positions yet."

        closed = sorted(closed, key=lambda p: p.closed_at or p.opened_at, reverse=True)

        total_realized = sum(p.realized_pnl or 0.0 for p in closed)
        wins = sum(1 for p in closed if (p.realized_pnl or 0.0) > 0)
        losses = sum(1 for p in closed if (p.realized_pnl or 0.0) < 0)
        win_rate = wins / len(closed) * 100.0

        lines = [
            f"Closed positions: {len(closed)}",
            f"Realized PnL (net of fees): {total_realized:,.2f} USD",
            f"Wins: {wins}",
            f"Losses: {losses}",
            f"Win rate: {win_rate:.2f}%",
            "",
            "Last closed positions:",
        ]

        for position in closed[:limit]:
            lines.append("")
            lines.append(f"Symbol: {position.symbol}")
            lines.append(f"Side: {position.side.value.upper()}")
            lines.append(f"Strategy: {position.strategy_id}")
            lines.append(f"Entry: {position.entry_price:,.4f}")
            lines.append(f"Exit: {_fmt(position.exit_price, ',.4f')}")
            lines.append(f"Size: {position.size:.8f}")
            lines.append(f"Fees: {position.total_fees_usd:,.4f} USD")
            lines.append(f"PnL (net): {(position.realized_pnl or 0.0):,.2f} USD")
            lines.append(f"Reason: {position.close_reason or 'unknown'}")
            lines.append(
                f"Closed at: {position.closed_at.isoformat() if position.closed_at else 'n/a'}"
            )

        return "\n".join(lines)

    def performance_report(self) -> str:
        closed = self.closed_positions
        open_positions = self.open_positions

        realized_pnl = realized_pnl_usd(self.positions)
        unrealized_pnl = unrealized_pnl_usd(self.positions, self.price_lookup)

        winners = [p for p in closed if (p.realized_pnl or 0.0) > 0]
        losers = [p for p in closed if (p.realized_pnl or 0.0) < 0]

        gross_profit = sum(p.realized_pnl or 0.0 for p in winners)
        gross_loss = abs(sum(p.realized_pnl or 0.0 for p in losers))

        win_rate = (len(winners) / len(closed) * 100.0) if closed else 0.0
        avg_win = (gross_profit / len(winners)) if winners else 0.0
        avg_loss = (-gross_loss / len(losers)) if losers else 0.0
        profit_factor = "n/a" if gross_loss == 0 else f"{gross_profit / gross_loss:.2f}"

        holding_days = [
            (p.closed_at - p.opened_at).days for p in closed if p.closed_at is not None
        ]
        avg_holding_days = sum(holding_days) / len(holding_days) if holding_days else 0.0

        total_fees = sum(p.total_fees_usd for p in self.positions)

        return "\n".join(
            [
                f"Closed trades: {len(closed)}",
                f"Open positions: {len(open_positions)}",
                f"Realized PnL: {realized_pnl:,.2f} USD",
                f"Unrealized PnL: {unrealized_pnl:,.2f} USD",
                f"Net PnL: {realized_pnl + unrealized_pnl:,.2f} USD",
                f"Total fees paid: {total_fees:,.2f} USD",
                "(all PnL figures are net of fees and slippage)",
                "",
                f"Winners: {len(winners)}",
                f"Losers: {len(losers)}",
                f"Win rate: {win_rate:.2f}%",
                f"Average win: {avg_win:,.2f} USD",
                f"Average loss: {avg_loss:,.2f} USD",
                f"Profit factor: {profit_factor}",
                f"Average holding days: {avg_holding_days:.2f}",
            ]
        )

    def symbols_report(self, universe: Sequence[str]) -> str:
        lines = ["Symbols report:"]

        for symbol in universe:
            open_positions = [p for p in self.open_positions if p.symbol == symbol]
            closed_positions = [p for p in self.closed_positions if p.symbol == symbol]

            realized_pnl = sum(p.realized_pnl or 0.0 for p in closed_positions)

            unrealized_pnl = 0.0
            current_exposure = 0.0
            for position in open_positions:
                current_exposure += position.notional_value
                last_price = self.price_lookup(position)
                if last_price is not None:
                    unrealized_pnl += position.net_pnl_at(last_price)

            lines.append("")
            lines.append(f"Symbol: {symbol}")
            lines.append(f"Open positions: {len(open_positions)}")
            lines.append(f"Closed positions: {len(closed_positions)}")
            lines.append(f"Current exposure: {current_exposure:,.2f} USD")
            lines.append(f"Realized PnL: {realized_pnl:,.2f} USD")
            lines.append(f"Unrealized PnL: {unrealized_pnl:,.2f} USD")

        return "\n".join(lines)
