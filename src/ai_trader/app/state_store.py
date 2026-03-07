from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderStatus,
    Position,
    PositionStatus,
    Side,
)


class JsonStateStore:
    def __init__(self, filepath: str = "data/runtime_state.json") -> None:
        self.path = Path(filepath)

    def load(self) -> dict:
        if not self.path.exists():
            return payload

        payload = json.loads(self.path.read_text(encoding="utf-8"))

        return {}

    def save(self, state: RunnerState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "open_positions": [
                self._serialize_position(position)
                for position in state.open_positions
            ],
            "execution_results": [
                self._serialize_execution_result(result)
                for result in state.execution_results
            ],
            "daily_realized_pnl_usd": state.daily_realized_pnl_usd,
            "is_paused": state.is_paused,
        }

        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _serialize_position(position: Position) -> dict:
        return {
            "symbol": position.symbol,
            "side": position.side.value,
            "size": position.size,
            "entry_price": position.entry_price,
            "opened_at": position.opened_at.isoformat(),
            "strategy_id": position.strategy_id,
            "status": position.status.value,
            "position_id": position.position_id,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "closed_at": position.closed_at.isoformat() if position.closed_at else None,
            "exit_price": position.exit_price,
            "realized_pnl": position.realized_pnl,
        }

    @staticmethod
    def _deserialize_position(payload: dict) -> Position:
        return Position(
            symbol=payload["symbol"],
            side=Side(payload["side"]),
            size=float(payload["size"]),
            entry_price=float(payload["entry_price"]),
            opened_at=datetime.fromisoformat(payload["opened_at"]),
            strategy_id=payload["strategy_id"],
            status=PositionStatus(payload.get("status", "open")),
            position_id=payload.get("position_id"),
            stop_loss=payload.get("stop_loss"),
            take_profit=payload.get("take_profit"),
            closed_at=(
                datetime.fromisoformat(payload["closed_at"])
                if payload.get("closed_at")
                else None
            ),
            exit_price=payload.get("exit_price"),
            realized_pnl=payload.get("realized_pnl"),
        )

    @staticmethod
    def _serialize_execution_result(result: ExecutionResult) -> dict:
        return {
            "success": result.success,
            "status": result.status.value,
            "message": result.message,
            "order_id": result.order_id,
            "filled_price": result.filled_price,
            "filled_size": result.filled_size,
            "executed_at": result.executed_at.isoformat(),
            "fees": result.fees,
            "slippage_bps": result.slippage_bps,
        }

    @staticmethod
    def _deserialize_execution_result(payload: dict) -> ExecutionResult:
        return ExecutionResult(
            success=bool(payload["success"]),
            status=OrderStatus(payload["status"]),
            message=payload["message"],
            order_id=payload.get("order_id"),
            filled_price=payload.get("filled_price"),
            filled_size=payload.get("filled_size"),
            executed_at=datetime.fromisoformat(payload["executed_at"]),
            fees=float(payload.get("fees", 0.0)),
            slippage_bps=float(payload.get("slippage_bps", 0.0)),
        )