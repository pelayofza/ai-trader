from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ai_trader.shared.schemas import Position, RiskDecision, Signal


@dataclass(slots=True)
class RiskLimits:
    max_position_size_usd: float = 1_000.0
    max_open_positions: int = 5
    max_symbol_exposure_usd: float = 1_000.0
    max_total_exposure_usd: float = 5_000.0
    max_daily_loss_usd: float = 200.0
    max_confidence_per_trade: float = 1.0
    min_confidence_per_trade: float = 0.65

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
        if not 0.0 <= self.max_confidence_per_trade <= 1.0:
            raise ValueError("max_confidence_per_trade must be between 0 and 1")
        if self.min_confidence_per_trade > self.max_confidence_per_trade:
            raise ValueError("min_confidence_per_trade cannot exceed max_confidence_per_trade")


@dataclass(slots=True)
class PortfolioState:
    open_positions: list[Position] = field(default_factory=list)
    daily_realized_pnl_usd: float = 0.0

    @property
    def total_exposure_usd(self) -> float:
        return sum(position.notional_value for position in self.open_positions if position.is_open)

    def symbol_exposure_usd(self, symbol: str) -> float:
        return sum(
            position.notional_value
            for position in self.open_positions
            if position.is_open and position.symbol == symbol
        )

    def has_open_position_in_symbol(self, symbol: str) -> bool:
        return any(
            position.is_open and position.symbol == symbol
            for position in self.open_positions
        )

    def open_positions_count(self) -> int:
        return sum(1 for position in self.open_positions if position.is_open)


class RiskEngine:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def evaluate(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        warnings: list[str] = []

        decision = self._check_daily_loss_limit(portfolio_state, warnings)
        if decision is not None:
            return decision

        decision = self._check_confidence(signal, warnings)
        if decision is not None:
            return decision

        decision = self._check_open_positions_limit(portfolio_state, warnings)
        if decision is not None:
            return decision

        decision = self._check_duplicate_symbol(signal, portfolio_state, warnings)
        if decision is not None:
            return decision

        proposed_size_usd = self._compute_position_size_usd(signal)

        decision = self._check_position_size_limit(proposed_size_usd, warnings)
        if decision is not None:
            return decision

        decision = self._check_symbol_exposure_limit(signal, portfolio_state, proposed_size_usd, warnings)
        if decision is not None:
            return decision

        decision = self._check_total_exposure_limit(portfolio_state, proposed_size_usd, warnings)
        if decision is not None:
            return decision

        return RiskDecision(
            approved=True,
            size_usd=proposed_size_usd,
            reason="Signal approved by risk engine",
            warnings=warnings,
        )

    def _check_daily_loss_limit(
        self,
        portfolio_state: PortfolioState,
        warnings: list[str],
    ) -> RiskDecision | None:
        if portfolio_state.daily_realized_pnl_usd <= -self.limits.max_daily_loss_usd:
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=(
                    "Daily loss limit reached: "
                    f"{portfolio_state.daily_realized_pnl_usd:.2f} USD"
                ),
                warnings=warnings,
            )
        return None

    def _check_confidence(
        self,
        signal: Signal,
        warnings: list[str],
    ) -> RiskDecision | None:
        if signal.confidence < self.limits.min_confidence_per_trade:
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=(
                    "Signal confidence below minimum threshold: "
                    f"{signal.confidence:.2f}"
                ),
                warnings=warnings,
            )
        if signal.confidence > self.limits.max_confidence_per_trade:
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=(
                    "Signal confidence above configured maximum threshold: "
                    f"{signal.confidence:.2f}"
                ),
                warnings=warnings,
            )
        return None

    def _check_open_positions_limit(
        self,
        portfolio_state: PortfolioState,
        warnings: list[str],
    ) -> RiskDecision | None:
        current_open_positions = portfolio_state.open_positions_count()
        if current_open_positions >= self.limits.max_open_positions:
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=(
                    "Maximum number of open positions reached: "
                    f"{current_open_positions}"
                ),
                warnings=warnings,
            )
        return None

    def _check_duplicate_symbol(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        warnings: list[str],
    ) -> RiskDecision | None:
        if portfolio_state.has_open_position_in_symbol(signal.symbol):
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=f"Open position already exists for symbol {signal.symbol}",
                warnings=warnings,
            )
        return None

    def _check_position_size_limit(
        self,
        proposed_size_usd: float,
        warnings: list[str],
    ) -> RiskDecision | None:
        if proposed_size_usd > self.limits.max_position_size_usd:
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=(
                    "Proposed position size exceeds max_position_size_usd: "
                    f"{proposed_size_usd:.2f} USD"
                ),
                warnings=warnings,
            )
        return None

    def _check_symbol_exposure_limit(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        proposed_size_usd: float,
        warnings: list[str],
    ) -> RiskDecision | None:
        symbol_exposure_after_trade = (
            portfolio_state.symbol_exposure_usd(signal.symbol) + proposed_size_usd
        )
        if symbol_exposure_after_trade > self.limits.max_symbol_exposure_usd:
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=(
                    f"Symbol exposure limit exceeded for {signal.symbol}: "
                    f"{symbol_exposure_after_trade:.2f} USD"
                ),
                warnings=warnings,
            )
        return None

    def _check_total_exposure_limit(
        self,
        portfolio_state: PortfolioState,
        proposed_size_usd: float,
        warnings: list[str],
    ) -> RiskDecision | None:
        total_exposure_after_trade = portfolio_state.total_exposure_usd + proposed_size_usd
        if total_exposure_after_trade > self.limits.max_total_exposure_usd:
            return RiskDecision(
                approved=False,
                size_usd=0.0,
                reason=(
                    "Total portfolio exposure limit exceeded: "
                    f"{total_exposure_after_trade:.2f} USD"
                ),
                warnings=warnings,
            )
        return None

    def _compute_position_size_usd(self, signal: Signal) -> float:
        return round(self.limits.max_position_size_usd * signal.confidence, 2)

    @staticmethod
    def build_portfolio_state(
        open_positions: Iterable[Position],
        daily_realized_pnl_usd: float = 0.0,
    ) -> PortfolioState:
        return PortfolioState(
            open_positions=list(open_positions),
            daily_realized_pnl_usd=daily_realized_pnl_usd,
        )