from __future__ import annotations

from dataclasses import dataclass, field

from ai_trader.shared.schemas import Position, RiskDecision, Side, Signal


@dataclass(slots=True)
class RiskLimits:
    max_position_size_usd: float = 1_000.0
    max_open_positions: int = 5
    max_symbol_exposure_usd: float = 2_000.0
    max_total_exposure_usd: float = 5_000.0
    max_daily_loss_usd: float = 200.0
    min_confidence_per_trade: float = 0.65

    # Politica de salida. El riesgo es el dueno del stop-loss y del take-profit:
    # la estrategia puede proponerlos, pero nunca puede aflojarlos por encima de
    # `max_stop_distance_pct` ni saltarselos.
    default_stop_loss_pct: float = 5.0
    default_take_profit_pct: float = 10.0
    max_stop_distance_pct: float = 15.0

    def __post_init__(self) -> None:
        if self.max_position_size_usd <= 0:
            raise ValueError("max_position_size_usd must be greater than 0")
        if self.max_open_positions < 0:
            raise ValueError("max_open_positions cannot be negative")
        if self.max_symbol_exposure_usd <= 0:
            raise ValueError("max_symbol_exposure_usd must be greater than 0")
        if self.max_total_exposure_usd <= 0:
            raise ValueError("max_total_exposure_usd must be greater than 0")
        if self.max_daily_loss_usd < 0:
            raise ValueError("max_daily_loss_usd cannot be negative")
        if not 0.0 <= self.min_confidence_per_trade <= 1.0:
            raise ValueError("min_confidence_per_trade must be between 0 and 1")
        if self.default_stop_loss_pct <= 0:
            raise ValueError("default_stop_loss_pct must be greater than 0")
        if self.default_take_profit_pct <= 0:
            raise ValueError("default_take_profit_pct must be greater than 0")
        if self.max_stop_distance_pct <= 0:
            raise ValueError("max_stop_distance_pct must be greater than 0")


@dataclass(slots=True)
class PortfolioState:
    open_positions: list[Position] = field(default_factory=list)
    daily_realized_pnl_usd: float = 0.0

    @property
    def total_exposure_usd(self) -> float:
        return sum(p.notional_value for p in self.open_positions if p.is_open)

    def symbol_exposure_usd(self, symbol: str) -> float:
        return sum(
            p.notional_value
            for p in self.open_positions
            if p.is_open and p.symbol == symbol
        )

    def has_open_position_in_symbol(self, symbol: str) -> bool:
        return any(p.is_open and p.symbol == symbol for p in self.open_positions)

    def open_positions_count(self) -> int:
        return sum(1 for p in self.open_positions if p.is_open)


class RiskEngine:
    """
    Puerta unica por la que pasa toda senal, sea de cripto, de renta variable o de
    un mercado de prediccion. Decide tres cosas: si se opera, con cuanto dinero, y
    con que stop-loss y take-profit.
    """

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def evaluate(self, signal: Signal, portfolio_state: PortfolioState) -> RiskDecision:
        rejection = (
            self._check_daily_loss_limit(portfolio_state)
            or self._check_confidence(signal)
            or self._check_open_positions_limit(portfolio_state)
            or self._check_duplicate_symbol(signal, portfolio_state)
        )
        if rejection is not None:
            return rejection

        size_usd = self._compute_position_size_usd(signal)

        rejection = (
            self._check_position_size_limit(size_usd)
            or self._check_symbol_exposure_limit(signal, portfolio_state, size_usd)
            or self._check_total_exposure_limit(portfolio_state, size_usd)
        )
        if rejection is not None:
            return rejection

        stop_loss, take_profit, warnings = self._resolve_exits(signal)

        return RiskDecision(
            approved=True,
            size_usd=size_usd,
            reason="Signal approved by risk engine",
            stop_loss=stop_loss,
            take_profit=take_profit,
            warnings=warnings,
        )

    # --- politica de salida ---

    def _resolve_exits(self, signal: Signal) -> tuple[float, float, list[str]]:
        """
        La estrategia propone stop-loss y take-profit; el riesgo los valida.

        Si la estrategia no propone nada, se aplican los porcentajes por defecto.
        Si propone un stop mas lejano de lo tolerado, se recorta y se avisa.
        """
        warnings: list[str] = []
        entry = signal.entry_price
        direction = 1.0 if signal.side == Side.BUY else -1.0

        max_distance = entry * (self.limits.max_stop_distance_pct / 100.0)

        stop_loss = signal.stop_loss
        if stop_loss is None:
            stop_loss = entry - direction * entry * (self.limits.default_stop_loss_pct / 100.0)
        elif abs(entry - stop_loss) > max_distance:
            proposed = stop_loss
            stop_loss = entry - direction * max_distance
            warnings.append(
                f"Stop loss proposed by strategy ({proposed:.6f}) exceeded "
                f"max_stop_distance_pct={self.limits.max_stop_distance_pct}%; "
                f"tightened to {stop_loss:.6f}"
            )

        take_profit = signal.take_profit
        if take_profit is None:
            take_profit = entry + direction * entry * (self.limits.default_take_profit_pct / 100.0)

        return stop_loss, take_profit, warnings

    # --- comprobaciones ---

    def _check_daily_loss_limit(self, portfolio: PortfolioState) -> RiskDecision | None:
        if portfolio.daily_realized_pnl_usd <= -self.limits.max_daily_loss_usd:
            return self._reject(
                f"Daily loss limit reached: {portfolio.daily_realized_pnl_usd:.2f} USD"
            )
        return None

    def _check_confidence(self, signal: Signal) -> RiskDecision | None:
        if signal.confidence < self.limits.min_confidence_per_trade:
            return self._reject(
                f"Signal confidence below minimum threshold: {signal.confidence:.2f}"
            )
        return None

    def _check_open_positions_limit(self, portfolio: PortfolioState) -> RiskDecision | None:
        count = portfolio.open_positions_count()
        if count >= self.limits.max_open_positions:
            return self._reject(f"Maximum number of open positions reached: {count}")
        return None

    def _check_duplicate_symbol(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> RiskDecision | None:
        if portfolio.has_open_position_in_symbol(signal.symbol):
            return self._reject(f"Open position already exists for symbol {signal.symbol}")
        return None

    def _check_position_size_limit(self, size_usd: float) -> RiskDecision | None:
        if size_usd > self.limits.max_position_size_usd:
            return self._reject(
                f"Proposed position size exceeds max_position_size_usd: {size_usd:.2f} USD"
            )
        return None

    def _check_symbol_exposure_limit(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        size_usd: float,
    ) -> RiskDecision | None:
        exposure_after = portfolio.symbol_exposure_usd(signal.symbol) + size_usd
        if exposure_after > self.limits.max_symbol_exposure_usd:
            return self._reject(
                f"Symbol exposure limit exceeded for {signal.symbol}: {exposure_after:.2f} USD"
            )
        return None

    def _check_total_exposure_limit(
        self,
        portfolio: PortfolioState,
        size_usd: float,
    ) -> RiskDecision | None:
        exposure_after = portfolio.total_exposure_usd + size_usd
        if exposure_after > self.limits.max_total_exposure_usd:
            return self._reject(
                f"Total portfolio exposure limit exceeded: {exposure_after:.2f} USD"
            )
        return None

    def _compute_position_size_usd(self, signal: Signal) -> float:
        return round(self.limits.max_position_size_usd * signal.confidence, 2)

    @staticmethod
    def _reject(reason: str) -> RiskDecision:
        return RiskDecision(approved=False, size_usd=0.0, reason=reason)
