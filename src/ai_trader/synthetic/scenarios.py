from __future__ import annotations

from dataclasses import dataclass, field

# Estos son los objetos que la IA produce (el "que pasa" de cada escenario) y que el
# motor numerico consume (el "como se ve en velas"). Son la frontera entre las dos
# piezas: 100% serializables a JSON para poder auditarlos y no re-ejecutar la IA.


@dataclass(slots=True, frozen=True)
class FactorPhase:
    """
    Un tramo temporal del escenario con un regimen de factores estable.

    Encadenando fases se modela una narrativa (p.ej. calma -> panico -> recuperacion):
    cada factor tiene una deriva diaria (drift) y una volatilidad diaria (vol) durante
    la fase. Valores en unidades de LOG-RETORNO DIARIO del factor (0.01 = 1%/dia).
    """

    length_days: int
    drift: dict[str, float] = field(default_factory=dict)
    vol: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"length_days": self.length_days, "drift": dict(self.drift), "vol": dict(self.vol)}

    @classmethod
    def from_dict(cls, data: dict) -> FactorPhase:
        return cls(
            length_days=int(data["length_days"]),
            drift={str(k): float(v) for k, v in (data.get("drift") or {}).items()},
            vol={str(k): float(v) for k, v in (data.get("vol") or {}).items()},
        )


@dataclass(slots=True, frozen=True)
class FactorShock:
    """
    Un salto discreto sobre un factor en un dia concreto (indice desde el inicio del
    escenario). Modela eventos abruptos: un anuncio de la Fed, un default, un ataque.
    `magnitude` es un retorno diario adicional puntual sobre ese factor (0.08 = +8%).
    """

    day: int
    factor: str
    magnitude: float

    def to_dict(self) -> dict:
        return {"day": self.day, "factor": self.factor, "magnitude": self.magnitude}

    @classmethod
    def from_dict(cls, data: dict) -> FactorShock:
        return cls(
            day=int(data["day"]),
            factor=str(data["factor"]),
            magnitude=float(data["magnitude"]),
        )


@dataclass(slots=True, frozen=True)
class ScenarioSpec:
    """
    La descripcion completa de un escenario macro. Es lo que la IA disena.

    - phases: la evolucion del regimen de factores en el tiempo.
    - shocks: eventos puntuales superpuestos.
    - asset_tilts: deriva diaria EXTRA por simbolo, para respuestas idiosincraticas
      que los factores comunes no capturan (p.ej. "embargo de petroleo -> XOM sube
      aunque la bolsa caiga"). Honra el "la IA analiza la respuesta de CADA activo".
    """

    id: str
    name: str
    narrative: str
    phases: tuple[FactorPhase, ...]
    shocks: tuple[FactorShock, ...] = ()
    asset_tilts: dict[str, float] = field(default_factory=dict)

    @property
    def horizon_days(self) -> int:
        return sum(p.length_days for p in self.phases)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "narrative": self.narrative,
            "phases": [p.to_dict() for p in self.phases],
            "shocks": [s.to_dict() for s in self.shocks],
            "asset_tilts": dict(self.asset_tilts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScenarioSpec:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            narrative=str(data.get("narrative", "")),
            phases=tuple(FactorPhase.from_dict(p) for p in data.get("phases", [])),
            shocks=tuple(FactorShock.from_dict(s) for s in data.get("shocks", [])),
            asset_tilts={str(k): float(v) for k, v in (data.get("asset_tilts") or {}).items()},
        )
