from __future__ import annotations

import json
import logging
import os
import re
from typing import Protocol

from ai_trader.synthetic.scenarios import FactorPhase, ScenarioSpec
from ai_trader.synthetic.universe import (
    FACTOR_DESCRIPTIONS,
    SyntheticUniverse,
    universe_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"


class ScenarioDesigner(Protocol):
    """
    Disena escenarios macro. Es la unica pieza con IA del generador; se inyecta para
    poder testear todo el pipeline con un disenador determinista sin tocar red.
    """

    def design(
        self,
        universe: SyntheticUniverse,
        n_scenarios: int,
        horizon_days: int,
    ) -> list[ScenarioSpec]:
        ...


# --- validacion / normalizacion comun --------------------------------------


def normalize_spec(
    spec: ScenarioSpec, universe: SyntheticUniverse, horizon_days: int
) -> ScenarioSpec:
    """
    Deja un ScenarioSpec en forma canonica y segura para el motor:
    - descarta factores y simbolos desconocidos (la IA a veces inventa),
    - ajusta la suma de fases exactamente a horizon_days,
    - garantiza al menos una fase valida.
    """
    valid_factors = set(universe.factors)
    valid_symbols = {s.strip().upper() for s in universe.symbols}

    phases = [
        FactorPhase(
            length_days=max(1, p.length_days),
            drift={f: v for f, v in p.drift.items() if f in valid_factors},
            vol={f: abs(v) for f, v in p.vol.items() if f in valid_factors},
        )
        for p in spec.phases
        if p.length_days > 0
    ]
    if not phases:
        phases = [FactorPhase(length_days=horizon_days, drift={}, vol={})]

    phases = _fit_horizon(phases, horizon_days)

    shocks = tuple(
        s for s in spec.shocks if s.factor in valid_factors and 0 <= s.day < horizon_days
    )
    tilts = {
        s.strip().upper(): float(v)
        for s, v in spec.asset_tilts.items()
        if s.strip().upper() in valid_symbols
    }

    return ScenarioSpec(
        id=spec.id,
        name=spec.name,
        narrative=spec.narrative,
        phases=tuple(phases),
        shocks=shocks,
        asset_tilts=tilts,
    )


def _fit_horizon(phases: list[FactorPhase], horizon_days: int) -> list[FactorPhase]:
    """Escala/recorta las fases para que sumen exactamente horizon_days."""
    total = sum(p.length_days for p in phases)
    if total == horizon_days:
        return phases

    # Reparte proporcionalmente y corrige el redondeo en la ultima fase.
    scaled: list[FactorPhase] = []
    running = 0
    for i, p in enumerate(phases):
        if i == len(phases) - 1:
            length = horizon_days - running
        else:
            length = max(1, round(p.length_days * horizon_days / total))
        length = max(1, length)
        running += length
        scaled.append(FactorPhase(length_days=length, drift=p.drift, vol=p.vol))

    # Si el redondeo se paso, recorta desde el final.
    overflow = sum(p.length_days for p in scaled) - horizon_days
    while overflow > 0:
        last = scaled[-1]
        take = min(overflow, last.length_days - 1)
        scaled[-1] = FactorPhase(last.length_days - take, last.drift, last.vol)
        overflow -= take
        if take == 0:
            break
    return scaled


def parse_scenarios(
    payload: str,
    universe: SyntheticUniverse,
    horizon_days: int,
) -> list[ScenarioSpec]:
    """Extrae y valida la lista de escenarios de la respuesta JSON de la IA."""
    data = _extract_json(payload)
    raw = data.get("scenarios", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON list of scenarios")

    specs: list[ScenarioSpec] = []
    for i, item in enumerate(raw):
        try:
            spec = ScenarioSpec.from_dict({"id": item.get("id", f"sc_{i:03d}"), **item})
            specs.append(normalize_spec(spec, universe, horizon_days))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed scenario %s: %s", i, exc)
    if not specs:
        raise ValueError("No valid scenarios parsed from AI response")
    return specs


def _extract_json(payload: str) -> dict | list:
    """Tolera vallas de codigo ```json ... ``` y texto alrededor del JSON."""
    text = payload.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Ultimo recurso: primer bloque {...} o [...] equilibrado.
        for opener, closer in (("[", "]"), ("{", "}")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
        raise


# --- disenador con IA real -------------------------------------------------


SYSTEM_PROMPT = """You are a quantitative macro strategist designing synthetic market \
regimes to stress-test trading strategies. You think in terms of a small set of macro \
FACTORS and how assets load on them. You output STRICT JSON only, no prose."""


def build_prompt(universe: SyntheticUniverse, n_scenarios: int, horizon_days: int) -> str:
    factor_lines = "\n".join(f"- {f}: {FACTOR_DESCRIPTIONS.get(f, '')}" for f in universe.factors)
    universe_json = json.dumps(universe_summary(universe), indent=2)
    return f"""Design {n_scenarios} DIVERSE macro scenarios, each {horizon_days} \
trading days long.

The market is modeled with these macro factors (daily log-return units, e.g. 0.01 = 1%/day):
{factor_lines}

Assets and their factor loadings (fixed, you do NOT change these):
{universe_json}

Each scenario must be a distinct macro narrative. Cover a WIDE range across the set:
monetary policy (hawkish/dovish surprises), geopolitical shocks (war, sanctions, oil \
embargo), liquidity crises, risk-on melt-ups, stagflation, crypto-specific booms and \
winters, sector rotations, sovereign debt stress, currency crises, black-swan crashes, \
and calm low-vol drifts. Make them realistic and heterogeneous, NOT variations of one idea.

For each scenario provide:
- "id": short slug (e.g. "hawkish_fed_shock")
- "name": human title
- "narrative": 1-2 sentences describing the macro story
- "phases": list of regime segments; each has "length_days" and per-factor "drift" and \
"vol" maps (daily log-return units). Phases let a scenario evolve (e.g. calm -> panic -> \
recovery). Lengths should sum to about {horizon_days}.
- "shocks": optional list of discrete one-day jumps: {{"day": int, "factor": str, \
"magnitude": float}} (e.g. a -0.09 EQUITY crash day).
- "asset_tilts": optional map symbol -> extra daily drift, for idiosyncratic responses \
the common factors miss (e.g. oil embargo lifts XOM beyond the COMMODITY factor).

Guidance on magnitudes: typical daily factor vol 0.004-0.02 in calm regimes, 0.02-0.06 \
in crises; drifts usually |0.0005-0.003|/day; shocks |0.03-0.12|.

Return ONLY a JSON object: {{"scenarios": [ ... ]}}."""


class ClaudeScenarioDesigner:
    """Disena escenarios llamando a Claude. Import de `anthropic` perezoso: el modulo
    se importa sin la dependencia; solo hace falta al generar de verdad."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        max_tokens: int = 16_000,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def design(
        self, universe: SyntheticUniverse, n_scenarios: int, horizon_days: int
    ) -> list[ScenarioSpec]:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot run the AI scenario designer"
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "The 'anthropic' package is required for ClaudeScenarioDesigner. "
                "Install it with `poetry add anthropic`."
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = build_prompt(universe, n_scenarios, horizon_days)

        logger.info("Requesting %s scenarios from %s", n_scenarios, self.model)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        specs = parse_scenarios(text, universe, horizon_days)
        logger.info("Parsed %s scenarios from AI response", len(specs))
        return specs


# --- disenador determinista (offline) --------------------------------------
#
# Reproduce un conjunto de arquetipos macro SIN IA. Sirve para (a) desarrollar y
# testear todo el pipeline sin red ni coste, y (b) tener un baseline reproducible.
# No sustituye la diversidad de la IA, pero cumple el mismo contrato ScenarioDesigner.

_ARCHETYPES: tuple[dict, ...] = (
    {
        "id": "calm_bull",
        "name": "Deriva alcista tranquila",
        "narrative": "Expansion ordenada, vol baja, apetito de riesgo sostenido.",
        "phases": [{"length_days": 1.0, "drift": {"EQUITY": 0.0010, "CRYPTO": 0.0015},
                    "vol": {"EQUITY": 0.007, "RATES": 0.004, "USD": 0.004,
                            "COMMODITY": 0.008, "CRYPTO": 0.02}}],
        "shocks": [],
        "asset_tilts": {},
    },
    {
        "id": "risk_off_crash",
        "name": "Panico risk-off",
        "narrative": "Calma, luego crash de renta variable y cripto con huida a bonos y oro.",
        "phases": [
            {"length_days": 0.4, "drift": {"EQUITY": 0.0005},
             "vol": {"EQUITY": 0.008, "CRYPTO": 0.02}},
            {"length_days": 0.2, "drift": {"EQUITY": -0.004, "CRYPTO": -0.006, "RATES": -0.002},
             "vol": {"EQUITY": 0.03, "CRYPTO": 0.05, "RATES": 0.012, "COMMODITY": 0.02}},
            {"length_days": 0.4, "drift": {"EQUITY": 0.0008, "CRYPTO": 0.001},
             "vol": {"EQUITY": 0.015, "CRYPTO": 0.03}},
        ],
        "shocks": [{"day": 0, "factor": "EQUITY", "magnitude": -0.07}],  # dia recolocado luego
        "asset_tilts": {"GLD": 0.0008, "TLT": 0.0009},
    },
    {
        "id": "hawkish_rates",
        "name": "Shock de tipos hawkish",
        "narrative": "Sorpresa de tipos al alza: bonos y growth sufren, dolar se fortalece.",
        "phases": [{"length_days": 1.0,
                    "drift": {"RATES": 0.0020, "USD": 0.0010, "EQUITY": -0.0008, "CRYPTO": -0.001},
                    "vol": {"EQUITY": 0.012, "RATES": 0.015, "USD": 0.008,
                            "COMMODITY": 0.012, "CRYPTO": 0.03}}],
        "shocks": [],
        "asset_tilts": {"TLT": -0.0006, "NVDA": -0.0006},
    },
    {
        "id": "crypto_winter",
        "name": "Invierno cripto",
        "narrative": "Desapalancamiento prolongado del ecosistema cripto con bolsa lateral.",
        "phases": [{"length_days": 1.0,
                    "drift": {"CRYPTO": -0.0035, "EQUITY": 0.0002},
                    "vol": {"EQUITY": 0.009, "CRYPTO": 0.045, "USD": 0.006}}],
        "shocks": [],
        "asset_tilts": {"SOL/USDT": -0.0015, "ETH/USDT": -0.0008},
    },
    {
        "id": "commodity_supply_shock",
        "name": "Shock de oferta en materias primas",
        "narrative": "Embargo energetico: materias primas disparadas, inflacion, bolsa presionada.",
        "phases": [{"length_days": 1.0,
                    "drift": {"COMMODITY": 0.0030, "EQUITY": -0.0006, "RATES": 0.0008},
                    "vol": {"EQUITY": 0.014, "COMMODITY": 0.03, "RATES": 0.01, "CRYPTO": 0.03}}],
        "shocks": [{"day": 0, "factor": "COMMODITY", "magnitude": 0.06}],
        "asset_tilts": {"XOM": 0.0015, "GLD": 0.0006},
    },
    {
        "id": "melt_up",
        "name": "Melt-up de liquidez",
        "narrative": "Inyeccion de liquidez dovish: risk-on generalizado y euforia cripto.",
        "phases": [{"length_days": 1.0,
                    "drift": {"EQUITY": 0.0018, "CRYPTO": 0.0035, "RATES": -0.0010},
                    "vol": {"EQUITY": 0.01, "CRYPTO": 0.035, "RATES": 0.006, "COMMODITY": 0.012}}],
        "shocks": [],
        "asset_tilts": {},
    },
    {
        "id": "stagflation",
        "name": "Estanflacion",
        "narrative": "Crecimiento debil con inflacion pegajosa: bolsa lateral-bajista, oro fuerte.",
        "phases": [{"length_days": 1.0,
                    "drift": {"EQUITY": -0.0005, "COMMODITY": 0.0012, "RATES": 0.0006},
                    "vol": {"EQUITY": 0.013, "COMMODITY": 0.02, "RATES": 0.011, "CRYPTO": 0.03}}],
        "shocks": [],
        "asset_tilts": {"GLD": 0.0007},
    },
    {
        "id": "choppy_range",
        "name": "Rango sin tendencia",
        "narrative": "Mercado lateral y ruidoso, sin direccion clara en ningun factor.",
        "phases": [{"length_days": 1.0, "drift": {},
                    "vol": {"EQUITY": 0.012, "RATES": 0.009, "USD": 0.008,
                            "COMMODITY": 0.014, "CRYPTO": 0.035}}],
        "shocks": [],
        "asset_tilts": {},
    },
)


class TemplateScenarioDesigner:
    """Disenador determinista basado en arquetipos. Cumple ScenarioDesigner sin IA."""

    def design(
        self, universe: SyntheticUniverse, n_scenarios: int, horizon_days: int
    ) -> list[ScenarioSpec]:
        specs: list[ScenarioSpec] = []
        for i in range(n_scenarios):
            base = _ARCHETYPES[i % len(_ARCHETYPES)]
            # Reparte length_days (fracciones) sobre el horizonte y recoloca shocks.
            phases = [
                {
                    "length_days": max(1, round(p["length_days"] * horizon_days)),
                    "drift": p.get("drift", {}),
                    "vol": p.get("vol", {}),
                }
                for p in base["phases"]
            ]
            shocks = [
                {"day": min(horizon_days - 1, round(0.4 * horizon_days)),
                 "factor": s["factor"], "magnitude": s["magnitude"]}
                for s in base.get("shocks", [])
            ]
            suffix = "" if i < len(_ARCHETYPES) else f"_{i // len(_ARCHETYPES)}"
            spec = ScenarioSpec.from_dict(
                {
                    "id": f"{base['id']}{suffix}",
                    "name": base["name"],
                    "narrative": base["narrative"],
                    "phases": phases,
                    "shocks": shocks,
                    "asset_tilts": base.get("asset_tilts", {}),
                }
            )
            specs.append(normalize_spec(spec, universe, horizon_days))
        return specs
