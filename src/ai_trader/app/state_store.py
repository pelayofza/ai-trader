"""
El estado de ejecucion: la FOTO (que posiciones hay, cuanto PnL llevan, si esta pausado).

La pelicula —que se decidio y a que precio— vive en `app/journal.py`. Este fichero es lo
unico que el sistema necesita para seguir operando despues de un reinicio, y por eso su
propiedad critica no es el detalle sino la DURABILIDAD: perderlo significa que las
posiciones abiertas dejan de existir para el runner, que no las cerrara nunca y seguira
abriendo otras contra los mismos limites.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_trader.notifications.base import NullNotifier, Notifier
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

# Copias de seguridad que se conservan, de la mas reciente (.1) a la mas antigua.
# Tres cubre el caso que importa: el fallo se detecta al arrancar, y entre el ciclo que
# corrompio el fichero y el arranque siguiente puede haber pasado mas de una escritura.
BACKUP_DEPTH = 3


class InMemoryStateStore:
    """
    Estado en RAM. Comparte interfaz con JsonStateStore pero no toca disco: el runner
    lo usa en backtest para no ensuciar el estado real ni pagar E/S en cada ciclo.
    """

    def __init__(self) -> None:
        self._payload: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        return self._payload

    def save(self, payload: dict[str, Any]) -> None:
        self._payload = payload


class JsonStateStore:
    """
    Estado en disco, con copia de seguridad rotatoria y arranque tolerante a corrupcion.

    El `notifier` es opcional y por defecto no va a ninguna parte, pero en vivo es lo que
    convierte una recuperacion silenciosa en un aviso: arrancar de un backup significa
    que se han perdido los ciclos posteriores a esa copia, y eso el humano tiene que
    saberlo en el momento, no descubrirlo en el dashboard tres semanas despues.
    """

    def __init__(
        self,
        filepath: str | Path = DEFAULT_STATE_PATH,
        *,
        notifier: Notifier | None = None,
        backup_depth: int = BACKUP_DEPTH,
    ) -> None:
        self.path = Path(filepath)
        self.notifier = notifier or NullNotifier()
        self.backup_depth = backup_depth

    # --- copias de seguridad ---

    def backup_path(self, index: int) -> Path:
        """`runtime_state.json.1` es la mas reciente; `.N`, la mas antigua."""
        return self.path.with_name(f"{self.path.name}.{index}")

    def backups(self) -> list[Path]:
        """Las copias existentes, de la mas reciente a la mas antigua."""
        return [p for i in range(1, self.backup_depth + 1) if (p := self.backup_path(i)).exists()]

    def _rotate_backups(self) -> None:
        """Desplaza .1 -> .2 -> ... y COPIA el estado actual a .1.

        Se copia y no se mueve a proposito: mover dejaria un instante sin fichero
        principal, y un corte justo ahi haria que el arranque siguiente no encontrara
        estado. Copiar cuesta una escritura mas de un fichero pequeno."""
        if not self.path.exists():
            return
        for index in range(self.backup_depth - 1, 0, -1):
            source = self.backup_path(index)
            if source.exists():
                source.replace(self.backup_path(index + 1))
        shutil.copy2(self.path, self.backup_path(1))

    # --- lectura ---

    def load(self) -> dict[str, Any]:
        payload = self._read(self.path)

        if payload is None and (self.path.exists() or self.backups()):
            # Hubo estado y no se puede leer. Antes esto devolvia {} en silencio, que es
            # el peor final posible: el runner arrancaba de cero, olvidaba las posiciones
            # abiertas y volvia a abrir contra los mismos limites.
            payload = self._recover_from_backup()

        if payload is None:
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

    def _read(self, path: Path) -> dict[str, Any] | None:
        """El contenido de un fichero de estado, o None si no esta o no parsea."""
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.exception("Estado ilegible en %s", path)
            return None
        return payload if isinstance(payload, dict) else None

    def _recover_from_backup(self) -> dict[str, Any] | None:
        """Arranca de la copia mas reciente que parsee. Avisa siempre, pase lo que pase."""
        for backup in self.backups():
            payload = self._read(backup)
            if payload is None:
                continue
            message = (
                f"Estado corrupto o ausente en {self.path}: se arranca desde la copia "
                f"{backup.name}. Los ciclos posteriores a esa copia NO estan en el estado; "
                f"el diario de ciclos si los tiene."
            )
            logger.error(message)
            self.notifier.error(message)
            return payload

        message = (
            f"Estado corrupto en {self.path} y ninguna copia de seguridad utilizable: "
            f"se arranca DE CERO. Las posiciones que hubiera abiertas ya no se cerraran "
            f"solas; revisa el diario de ciclos antes de reanudar."
        )
        logger.error(message)
        self.notifier.error(message)
        return None

    # --- escritura ---

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_backups()

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
        # El fsync sobre el temporal cierra el hueco que quedaba: sin el, `replace` puede
        # publicar un fichero cuyo contenido el sistema operativo aun no ha bajado a
        # disco, y un corte de luz deja el estado a cero bytes con el nombre bueno.
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(serialized, indent=2, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
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
