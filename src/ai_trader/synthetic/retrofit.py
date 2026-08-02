from __future__ import annotations

from dataclasses import replace

from ai_trader.synthetic.scenarios import FactorPhase, ScenarioSpec
from ai_trader.synthetic.universe import COMMODITY, CRYPTO, EQUITY

# Retrofit DETERMINISTA: enriquece specs de ai_v1 (sin microestructura) con carga serial,
# colas, clustering y saltos DERIVADOS de la semantica de cada fase. No llama a la IA:
# reusa los specs caros ya disenados y les da la "fisica fina" que faltaba, de forma
# reproducible. El objetivo es que unas fases premien momentum (tendencia) y otras
# mean-reversion (rango), y que las crisis tengan colas/gaps realistas.

# Factores "de mercado" cuya DERIVA marca si la fase es direccional (tendencia) o plana.
_DIRECTIONAL_FACTORS = (EQUITY, CRYPTO, COMMODITY)

# Umbrales de vol de EQUITY que clasifican el estres sistemico de la fase. Se usa EQUITY
# (no CRYPTO) porque la vol bursatil es la que se dispara en las crisis; la de cripto es
# alta hasta en calma y sobre-clasificaria.
_ELEVATED_EQUITY_VOL = 0.015
_CRISIS_EQUITY_VOL = 0.025

# Dispersion por defecto del dia de los shocks entre paths (evita crashes alineados).
DEFAULT_SHOCK_JITTER_DAYS = 15


def _market_vol(phase: FactorPhase) -> float:
    return phase.vol.get(EQUITY, 0.0)


def _market_drift(phase: FactorPhase) -> float:
    return max((abs(phase.drift.get(f, 0.0)) for f in _DIRECTIONAL_FACTORS), default=0.0)


def _idio_ar_for(phase: FactorPhase) -> float:
    """Signo/fuerza de la autocorrelacion idiosincratica segun el caracter de la fase.

    Fase direccional (deriva grande frente a su vol) -> tendencia (+): edge de momentum.
    Fase plana (poca deriva) -> reversion (-): edge de mean-reversion. Intermedia -> 0.
    """
    equity_vol = _market_vol(phase)
    trend_ratio = _market_drift(phase) / (equity_vol + 1e-9)
    if trend_ratio > 0.15:
        return 0.25
    if trend_ratio < 0.05:
        return -0.30
    return 0.0


def _tail_cluster_jump_for(phase: FactorPhase) -> tuple[float, float, float, float]:
    """(tail_dof, vol_persistence, jump_intensity, jump_scale) segun el estres de la fase.
    El clustering es ubicuo (base 0.85); las colas y los saltos crecen con la crisis."""
    equity_vol = _market_vol(phase)
    if equity_vol >= _CRISIS_EQUITY_VOL:
        return 5.0, 0.92, 0.04, 5.0
    if equity_vol >= _ELEVATED_EQUITY_VOL:
        return 8.0, 0.88, 0.015, 4.0
    return 0.0, 0.85, 0.0, 0.0


def enrich_phase(phase: FactorPhase) -> FactorPhase:
    idio_ar = _idio_ar_for(phase)
    tail_dof, persistence, jump_intensity, jump_scale = _tail_cluster_jump_for(phase)
    return phase.with_microstructure(
        idio_ar=idio_ar,
        tail_dof=tail_dof,
        vol_persistence=persistence,
        jump_intensity=jump_intensity,
        jump_scale=jump_scale,
    )


def enrich_spec(
    spec: ScenarioSpec, *, shock_jitter_days: int = DEFAULT_SHOCK_JITTER_DAYS
) -> ScenarioSpec:
    """Devuelve el spec con microestructura derivada. Solo toca los campos de
    microestructura (drift/vol/phases/shocks siguen siendo los disenados por la IA)."""
    phases = tuple(enrich_phase(p) for p in spec.phases)
    shocks = tuple(
        s if s.jitter_days else replace(s, jitter_days=shock_jitter_days) for s in spec.shocks
    )
    return replace(spec, phases=phases, shocks=shocks)
