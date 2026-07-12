from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderStatus,
    Position,
    PositionStatus,
    Side,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data") / "runtime_state.json"


class JsonStateStore:
    def __init__(self, filepath: str | Path = DEFAULT_STATE_PATH) -> None:
        self.path = Path(filepath)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.exception("Corrupt state file at %s; starting from empty state", self.path)
            return {}

        # "open_positions" es el nombre antiguo. Contenia abiertas Y cerradas, asi que
        # el nombre enganaba; se lee por compatibilidad con estados ya guardados.
        raw_positions = payload.get("positions") or payload.get("open_positions") or []

        return {
            "positions": [self._deserialize_position(item) for item in raw_positions],
            "execution_results": [
                self._deserialize_execution_result(item)
                for item in payload.get("execution_results", [])
            ],
            "daily_realized_pnl_usd": float(payload.get("daily_realized_pnl_usd", 0.0)),
            "is_paused": bool(payload.get("is_paused", False)),
        }

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        serialized = {
            "positions": [
                self._serialize_position(p) for p in payload.get("positions", [])
            ],
            "execution_results": [
                self._serialize_execution_result(r)
                for r in payload.get("execution_results", [])
            ],
            "daily_realized_pnl_usd": payload.get("daily_realized_pnl_usd", 0.0),
            "is_paused": payload.get("is_paused", False),
        }

        # Escritura atomica: se escribe a un temporal y se renombra. Antes se escribia
        # in situ, asi que un fallo a mitad dejaba el estado de trading corrupto.
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    @staticmethod
    def _serialize_position(position: Position) -> dict[str, Any]:
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
            "close_reason": position.close_reason,
            "entry_fees_usd": position.entry_fees_usd,
            "exit_fees_usd": position.exit_fees_usd,
            "venue": position.venue.value if position.venue else None,
            "asset_class": position.asset_class.value if position.asset_class else None,
            "instrument_id": position.instrument_id,
            "outcome": position.outcome,
            "metadata": position.metadata,
        }

    @staticmethod
    def _deserialize_position(payload: dict[str, Any]) -> Position:
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
            close_reason=payload.get("close_reason"),
            entry_fees_usd=float(payload.get("entry_fees_usd", 0.0)),
            exit_fees_usd=float(payload.get("exit_fees_usd", 0.0)),
            venue=Venue(payload["venue"]) if payload.get("venue") else None,
            asset_class=(
                AssetClass(payload["asset_class"]) if payload.get("asset_class") else None
            ),
            instrument_id=payload.get("instrument_id"),
            outcome=payload.get("outcome"),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _serialize_execution_result(result: ExecutionResult) -> dict[str, Any]:
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
            "venue": result.venue.value if result.venue else None,
            "asset_class": result.asset_class.value if result.asset_class else None,
            "symbol": result.symbol,
            "instrument_id": result.instrument_id,
            "outcome": result.outcome,
            "metadata": result.metadata,
        }

    @staticmethod
    def _deserialize_execution_result(payload: dict[str, Any]) -> ExecutionResult:
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
            venue=Venue(payload["venue"]) if payload.get("venue") else None,
            asset_class=(
                AssetClass(payload["asset_class"]) if payload.get("asset_class") else None
            ),
            symbol=payload.get("symbol"),
            instrument_id=payload.get("instrument_id"),
            outcome=payload.get("outcome"),
            metadata=dict(payload.get("metadata", {})),
        )
