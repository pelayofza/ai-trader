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

# --- constantes calibradas contra el informe de fidelidad ---------------------------
#
# Los tres numeros de abajo NO son opinion: salen de iterar el propio harness
# (`ai_trader.synthetic.fidelity_study --library ai_v3`) hasta que la mediana real de
# cripto cae dentro del [p10, p90] del ensemble sintetico en curtosis, clustering y
# exceedances. El estudio publica esos umbrales y falla si dejan de cumplirse.

# Grados de libertad de la t-Student por regimen. Menos dof = mas cola. La calma NO es
# gaussiana: es el cambio que mueve la curtosis de 0,37 a ~4, porque las fases tranquilas
# ocupan la mayor parte de un horizonte de 730 dias.
CRISIS_TAIL_DOF = 4.0
ELEVATED_TAIL_DOF = 4.5
CALM_TAIL_DOF = 5.0

# Reaccion a la noticia del GARCH (el alpha). Con el 0.15 por defecto del motor el
# clustering medido se quedaba en ~0,09 contra 0,13-0,31 del mercado real. Sube la
# autocorrelacion de |r| a lag 1 SIN tocar la persistencia total ni el nivel de
# volatilidad (los dos pesos siguen sumando p).
NEWS_SHARE = 0.25

# Subida fraccional de las cargas factoriales por regimen (B6). Con betas congeladas la
# correlacion entre activos es una constante del universo y NO puede dispararse en las
# caidas, que es el hecho de mercado mas caro de ignorar: media 0,49 contra 0,65 real.
# Subir mas las betas seguiria acercando la correlacion a 0,65, pero a costa de inflar la
# volatilidad total (la beta entra al cuadrado en la varianza), que HOY ya esta en su
# sitio. Este es el punto donde la correlacion mejora sin estropear el nivel de riesgo.
CRISIS_BETA_STRESS = 1.00
ELEVATED_BETA_STRESS = 0.45


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
    El clustering es ubicuo (base 0.85); los saltos crecen con la crisis.

    Las COLAS tambien son ubicuas, y esa es la correccion medida: la version anterior daba
    tail_dof=0 -gaussiana EXACTA- a toda fase tranquila, que es la mayor parte de un
    horizonte de 730 dias, y por eso la curtosis sintetica salia ~0,4 contra 3,3-17,8 del
    cripto real. El cripto real tiene curtosis de 4 para arriba TAMBIEN en ventanas
    tranquilas: la calma no es gaussiana, solo es menos leptocurtica que el panico."""
    equity_vol = _market_vol(phase)
    if equity_vol >= _CRISIS_EQUITY_VOL:
        return CRISIS_TAIL_DOF, 0.92, 0.04, 5.0
    if equity_vol >= _ELEVATED_EQUITY_VOL:
        return ELEVATED_TAIL_DOF, 0.88, 0.015, 4.0
    return CALM_TAIL_DOF, 0.85, 0.0, 0.0


def _beta_stress_for(phase: FactorPhase) -> float:
    """Cuanto suben las betas en esta fase, con la MISMA semantica de fase que las colas
    y los saltos: el estres sistemico lo marca la vol de EQUITY."""
    equity_vol = _market_vol(phase)
    if equity_vol >= _CRISIS_EQUITY_VOL:
        return CRISIS_BETA_STRESS
    if equity_vol >= _ELEVATED_EQUITY_VOL:
        return ELEVATED_BETA_STRESS
    return 0.0


def enrich_phase(phase: FactorPhase) -> FactorPhase:
    idio_ar = _idio_ar_for(phase)
    tail_dof, persistence, jump_intensity, jump_scale = _tail_cluster_jump_for(phase)
    return phase.with_microstructure(
        idio_ar=idio_ar,
        tail_dof=tail_dof,
        vol_persistence=persistence,
        vol_news=NEWS_SHARE,
        jump_intensity=jump_intensity,
        jump_scale=jump_scale,
        beta_stress=_beta_stress_for(phase),
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
