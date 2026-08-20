"""
EL SUSTRATO REAL: que simbolos entran, en que ventanas y con que geometria de folds.

Toda la maquinaria de aqui se escribio dentro del estudio de transferencia, porque fue el
primero que necesito un lado real. Eso dejaba una dependencia al reves: la capa tematica
-que corre sobre archivo de senales REAL y no toca el generador- tenia que importar de un
estudio cuyo nombre habla del sintetico. Al aparcar la linea sintetica, o se separaba o el
lado real se iba con ella.

Tres piezas:

- `crypto_universe`: los simbolos CRIPTO del universo operable. La renta variable del config
  va por otro proveedor y otra sesion de mercado; aqui no hay forma de traerla.
- `audit_real_symbols`: separa los simbolos con historico SUFICIENTE de los que no. Los que
  no llegan se DECLARAN y se omiten. No se rellenan ni se les inventa precio.
- `real_windows`: trocea el historico en sub-ventanas disjuntas ancladas AL FINAL.

Y la geometria de folds (`N_GROUPS`, `N_TEST_GROUPS`), que se declara una sola vez para que
"un fold" signifique lo mismo en todos los estudios que la usan.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from ai_trader.config import AppConfig
from ai_trader.shared.instruments import AssetClass, detect_asset_class

# CPCV: C(6,2) = 15 ventanas OOS por unidad. Los mismos numeros en todos los estudios que
# parten el historico real, para que "un fold" signifique lo mismo en todos ellos.
N_GROUPS = 6
N_TEST_GROUPS = 2


@dataclass(frozen=True, slots=True)
class SymbolAudit:
    """Por que un simbolo del universo operable entra o no en el lado real."""

    symbol: str
    n_bars: int
    first_bar: str | None
    last_bar: str | None
    reason: str

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "first_bar": self.first_bar,
            "last_bar": self.last_bar,
            "reason": self.reason,
        }


def crypto_universe(config: AppConfig) -> list[str]:
    """Los simbolos CRIPTO del universo operable. La renta variable del config va por otro
    proveedor y otra sesion de mercado; aqui no hay forma de traerla ni de compararla."""
    return sorted(
        s for s in config.runner.symbols if detect_asset_class(s) is AssetClass.CRYPTO
    )


def audit_real_symbols(
    bars: dict[str, pd.DataFrame],
    requested: Sequence[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_history_days: int,
) -> tuple[list[SymbolAudit], list[SymbolAudit]]:
    """
    Separa los simbolos con historico SUFICIENTE de los que no lo tienen.

    El umbral no es un numero a ojo: `min_history_days` = calentamiento de la estrategia
    (`lookback_days`) + un grupo de CPCV. Por debajo de eso el simbolo no puede llegar a
    operarse ni en una sola ventana OOS, asi que no aporta ranking; aporta ruido de
    cobertura. Los que no llegan se DECLARAN y se omiten: no se rellenan, no se sustituyen
    y no se les inventa precio (un par listado hace tres meses no tiene 2018).
    """
    kept: list[SymbolAudit] = []
    dropped: list[SymbolAudit] = []
    for symbol in sorted(requested):
        df = bars.get(symbol)
        if df is None or df.empty:
            dropped.append(SymbolAudit(symbol, 0, None, None, "sin barras en la cache"))
            continue
        window = df.loc[start : end - pd.Timedelta(days=1)]
        n = len(window)
        audit = SymbolAudit(
            symbol=symbol,
            n_bars=n,
            first_bar=None if n == 0 else str(window.index.min().date()),
            last_bar=None if n == 0 else str(window.index.max().date()),
            reason="",
        )
        if n < min_history_days:
            dropped.append(
                dataclasses.replace(
                    audit,
                    reason=f"{n} barras < {min_history_days} exigidas "
                    "(calentamiento + un grupo de CPCV)",
                )
            )
        else:
            kept.append(dataclasses.replace(audit, reason="historico suficiente"))
    return kept, dropped


@dataclass(frozen=True, slots=True)
class RealWindow:
    """Una sub-ventana del historico real."""

    label: str
    start: datetime
    end: datetime  # exclusivo, como los bloques de `backtest.validation`

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start.date().isoformat(),
            "end": (self.end - timedelta(days=1)).date().isoformat(),
            "days": self.days,
        }


def real_windows(start: datetime, end: datetime, window_days: int) -> list[RealWindow]:
    """
    Trocea `[start, end)` en sub-ventanas DISJUNTAS de `window_days`, ancladas al final.

    Ancladas al final y no al principio porque el resto que no completa una ventana tiene
    que caer en algun sitio, y donde menos cuesta es en la cabecera: es el tramo con menos
    simbolos vivos (media cripto de hoy no cotizaba en 2017) y el mas lejano del regimen
    que el sistema va a operar. El tramo descartado se declara en el informe.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    total = (end - start).days
    n = total // window_days
    if n < 1:
        raise ValueError(
            f"El historico real ({total} dias) no llega para una sub-ventana de "
            f"{window_days} dias"
        )
    first = end - timedelta(days=n * window_days)
    return [
        RealWindow(
            label=f"w{i + 1}",
            start=first + timedelta(days=i * window_days),
            end=first + timedelta(days=(i + 1) * window_days),
        )
        for i in range(n)
    ]
