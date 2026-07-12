"""
El PnL debe ser neto de comisiones.

El motor de ejecucion calculaba fees y slippage y los guardaba en el ExecutionResult,
pero el calculo de PnL los ignoraba por completo, asi que todos los resultados
publicados eran optimistas. Estos tests fijan el comportamiento correcto.
"""

from __future__ import annotations

import pytest

from ai_trader.shared.schemas import Side


def test_gross_pnl_ignores_fees(make_position):
    position = make_position(size=2.0, entry_price=100.0, entry_fees_usd=5.0)

    assert position.gross_pnl_at(110.0) == pytest.approx(20.0)


def test_net_pnl_subtracts_both_legs_of_fees(make_position):
    position = make_position(
        size=2.0,
        entry_price=100.0,
        entry_fees_usd=5.0,
        exit_fees_usd=3.0,
    )

    # bruto 20, menos 8 de comisiones
    assert position.net_pnl_at(110.0) == pytest.approx(12.0)


def test_fees_can_turn_a_winning_trade_into_a_loss(make_position):
    position = make_position(
        size=1.0,
        entry_price=100.0,
        entry_fees_usd=1.0,
        exit_fees_usd=1.0,
    )

    assert position.gross_pnl_at(101.5) == pytest.approx(1.5)
    assert position.net_pnl_at(101.5) == pytest.approx(-0.5)


def test_short_pnl_is_inverted(make_position):
    position = make_position(side=Side.SELL, size=2.0, entry_price=100.0, entry_fees_usd=4.0)

    assert position.gross_pnl_at(90.0) == pytest.approx(20.0)
    assert position.net_pnl_at(90.0) == pytest.approx(16.0)
    assert position.gross_pnl_at(110.0) == pytest.approx(-20.0)


def test_total_fees_sums_both_legs(make_position):
    position = make_position(entry_fees_usd=1.25, exit_fees_usd=0.75)

    assert position.total_fees_usd == pytest.approx(2.0)
