from __future__ import annotations

import json

import pytest

from ai_trader.app.state_store import JsonStateStore
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import ExecutionResult, OrderStatus, PositionStatus


@pytest.fixture
def store(tmp_path) -> JsonStateStore:
    return JsonStateStore(tmp_path / "state.json")


def test_load_returns_empty_when_file_is_absent(store):
    assert store.load() == {}


def test_position_survives_a_round_trip(store, make_position):
    original = make_position(size=1.5, entry_price=250.0, entry_fees_usd=0.375)
    original.stop_loss = 237.5
    original.take_profit = 275.0
    original.venue = Venue.CCXT
    original.asset_class = AssetClass.CRYPTO

    store.save({"positions": [original], "execution_results": []})
    restored = store.load()["positions"][0]

    assert restored.symbol == original.symbol
    assert restored.size == pytest.approx(original.size)
    assert restored.entry_price == pytest.approx(original.entry_price)
    assert restored.entry_fees_usd == pytest.approx(original.entry_fees_usd)
    assert restored.stop_loss == pytest.approx(237.5)
    assert restored.opened_at == original.opened_at
    assert restored.venue == Venue.CCXT
    assert restored.asset_class == AssetClass.CRYPTO


def test_execution_result_survives_a_round_trip(store):
    original = ExecutionResult(
        success=True,
        status=OrderStatus.FILLED,
        message="filled",
        order_id="paper-1",
        filled_price=100.0,
        filled_size=2.0,
        fees=0.2,
        symbol="BTC/USDT",
    )

    store.save({"positions": [], "execution_results": [original]})
    restored = store.load()["execution_results"][0]

    assert restored.order_id == "paper-1"
    assert restored.filled_price == pytest.approx(100.0)
    assert restored.fees == pytest.approx(0.2)
    assert restored.executed_at == original.executed_at


def test_reads_the_legacy_open_positions_key(store, make_position, tmp_path):
    """El estado guardado con el esquema antiguo debe seguir cargando."""
    position = make_position()
    legacy = {
        "open_positions": [JsonStateStore._serialize_position(position)],
        "execution_results": [],
        "daily_realized_pnl_usd": -12.5,
        "is_paused": True,
    }
    store.path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = store.load()

    assert len(loaded["positions"]) == 1
    assert loaded["daily_realized_pnl_usd"] == pytest.approx(-12.5)
    assert loaded["is_paused"] is True


def test_corrupt_file_does_not_crash_the_runner(store):
    store.path.write_text("{ this is not json", encoding="utf-8")

    assert store.load() == {}


def test_save_is_atomic_and_leaves_no_temp_file(store, make_position):
    store.save({"positions": [make_position()], "execution_results": []})

    assert store.path.exists()
    assert not store.path.with_suffix(".json.tmp").exists()


def test_closed_positions_are_persisted_too(store, make_position):
    closed = make_position()
    closed.status = PositionStatus.CLOSED
    closed.realized_pnl = -3.25
    closed.close_reason = "stop_loss"

    store.save({"positions": [closed], "execution_results": []})
    restored = store.load()["positions"][0]

    assert restored.status == PositionStatus.CLOSED
    assert restored.realized_pnl == pytest.approx(-3.25)
    assert restored.close_reason == "stop_loss"
