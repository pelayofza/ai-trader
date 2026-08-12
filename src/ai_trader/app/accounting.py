"""
Las dos sumas que definen el resultado de la cartera, en UN solo sitio.

Estaban escritas dos veces con el cuerpo identico: en `runner.py::_build_portfolio_state`
(que es lo que ve el motor de riesgo para dimensionar) y en
`reports.py::performance_report` (que es lo que lee un humano por Telegram). Con el diario
de ciclos habrian sido tres, y la tercera es la que se publica en el dashboard: tres
definiciones de "cuanto llevo ganado" es exactamente el fallo que la auditoria del
2026-08-12 encontro en el CVaR.

Las dos son netas de comisiones, que es la unica forma en que este repo publica un PnL
(`Position.net_pnl_at`). Y las dos tratan la falta de precio de marca igual: una posicion
que hoy no se puede valorar NO suma cero, se queda fuera de la suma. Sumar cero seria
afirmar que no se ha movido.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from ai_trader.shared.schemas import Position

# Devuelve el precio de marca de una posicion, o None si no se puede resolver.
PriceLookup = Callable[[Position], float | None]


def realized_pnl_usd(positions: Sequence[Position]) -> float:
    """PnL realizado acumulado (neto de comisiones) de todo lo ya cerrado."""
    return sum(p.realized_pnl or 0.0 for p in positions if not p.is_open)


def unrealized_pnl_usd(positions: Sequence[Position], price_lookup: PriceLookup) -> float:
    """PnL no realizado (neto de comisiones) de lo abierto, marcado a mercado.

    Las posiciones sin precio de marca disponible no entran en la suma."""
    total = 0.0
    for position in positions:
        if not position.is_open:
            continue
        mark = price_lookup(position)
        if mark is not None:
            total += position.net_pnl_at(mark)
    return total


__all__ = ["PriceLookup", "realized_pnl_usd", "unrealized_pnl_usd"]
