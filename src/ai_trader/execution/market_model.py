from __future__ import annotations

from datetime import timedelta
from typing import Protocol

from ai_trader.shared import bars as bar_schema
from ai_trader.shared.bars import Bar
from ai_trader.shared.clock import Clock
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import Position, Side, Signal

# Resultado de una salida disparada por precio: (motivo, precio de salida).
ExitEvent = tuple[str, float]


class MarketModel(Protocol):
    """
    Costura de precios del orquestador.

    Aisla las tres decisiones de precio que el runner tomaba a fuego: a que precio se
    entra, a cuanto se valora una posicion abierta, y si un stop-loss / take-profit se
    ha disparado (y a que precio se sale). El comportamiento en vivo y en backtest solo
    se diferencian aqui; el resto del runner es identico.
    """

    def entry_reference_price(self, symbol: str, signal: Signal) -> float | None:
        ...

    def mark_price(self, position: Position) -> float | None:
        ...

    def price_exit(self, position: Position) -> ExitEvent | None:
        ...


class _ReaderLike(Protocol):
    def get_daily_bars(self, symbol: str, start, end): ...
    def get_prediction_midpoint(self, token_id: str) -> float | None: ...


class CloseMarketModel:
    """
    Modelo por defecto: reproduce exactamente el comportamiento en vivo previo.

    Entra al precio de la senal (cierre de la barra de decision), valora contra el
    ultimo cierre disponible, y detecta stops cierre-contra-cierre. No mira intrabar.
    """

    def __init__(self, reader: _ReaderLike, clock: Clock) -> None:
        self.reader = reader
        self.clock = clock

    def entry_reference_price(self, symbol: str, signal: Signal) -> float | None:
        return signal.entry_price

    def mark_price(self, position: Position) -> float | None:
        if (
            position.asset_class == AssetClass.PREDICTION
            and position.venue == Venue.POLYMARKET
            and position.instrument_id
        ):
            return self.reader.get_prediction_midpoint(position.instrument_id)

        end = self.clock.now()
        bars = self.reader.get_daily_bars(position.symbol, end - timedelta(days=7), end)
        return bar_schema.last_close(bars) if bars is not None else None

    def price_exit(self, position: Position) -> ExitEvent | None:
        mark = self.mark_price(position)
        if mark is None:
            return None

        reason = _close_to_close_exit(position, mark)
        return (reason, mark) if reason is not None else None


class _BarSource(Protocol):
    def bar_on(self, symbol: str, day) -> Bar | None: ...


class IntrabarMarketModel:
    """
    Modelo realista para backtest.

    La decision se toma con barras ya cerradas (hasta ayer); la entrada se llena al
    OPEN del dia de hoy, y los stop-loss / take-profit se comprueban contra el HIGH y
    el LOW de la barra, no solo contra el cierre. Se valora a cierre del dia.
    """

    def __init__(self, source: _BarSource, clock: Clock) -> None:
        self.source = source
        self.clock = clock

    def entry_reference_price(self, symbol: str, signal: Signal) -> float | None:
        bar = self.source.bar_on(symbol, self.clock.now())
        return bar.open if bar is not None else None

    def mark_price(self, position: Position) -> float | None:
        bar = self.source.bar_on(position.symbol, self.clock.now())
        return bar.close if bar is not None else None

    def price_exit(self, position: Position) -> ExitEvent | None:
        bar = self.source.bar_on(position.symbol, self.clock.now())
        if bar is None:
            return None
        return _intrabar_exit(position, bar)


def _close_to_close_exit(position: Position, mark: float) -> str | None:
    if position.side == Side.BUY:
        if position.stop_loss is not None and mark <= position.stop_loss:
            return "stop_loss"
        if position.take_profit is not None and mark >= position.take_profit:
            return "take_profit"
    else:
        if position.stop_loss is not None and mark >= position.stop_loss:
            return "stop_loss"
        if position.take_profit is not None and mark <= position.take_profit:
            return "take_profit"
    return None


def _intrabar_exit(position: Position, bar: Bar) -> ExitEvent | None:
    """
    Detecta salida intrabar. Convencion pesimista: si en la misma barra se tocan el
    stop y el take-profit, se asume que salto primero el stop. En hueco de apertura
    que ya abre pasado el nivel, se llena al open (peor que el nivel).
    """
    sl = position.stop_loss
    tp = position.take_profit

    if position.side == Side.BUY:
        if sl is not None and bar.low <= sl:
            exit_price = min(bar.open, sl)  # hueco a la baja: peor que el stop
            return ("stop_loss", exit_price)
        if tp is not None and bar.high >= tp:
            exit_price = max(bar.open, tp)  # hueco al alza: mejor que el objetivo
            return ("take_profit", exit_price)
    else:
        if sl is not None and bar.high >= sl:
            exit_price = max(bar.open, sl)
            return ("stop_loss", exit_price)
        if tp is not None and bar.low <= tp:
            exit_price = min(bar.open, tp)
            return ("take_profit", exit_price)

    return None
