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

    Ademas del regimen de factores, la fase declara su MICROESTRUCTURA ESTADISTICA
    (campos opcionales, defaults neutros = mundo gaussiano iid de siempre). Son la
    "fisica fina" que el motor implementa y que da caracter a cada regimen:

    - `idio_ar`: autocorrelacion AR(1) del componente idiosincratico por activo.
      >0 = tendencia (edge de momentum); <0 = reversion a la media (edge de mean-rev).
      Es el mecanismo que hace que unas fases premien una primitiva y otras la otra.
    - `tail_dof`: grados de libertad de la t-Student de las innovaciones (0 = gaussiano).
      Bajo (3-6) = colas gruesas, tipico de crisis.
    - `vol_persistence`: persistencia tipo GARCH de la volatilidad (0 = sin clustering).
    - `jump_intensity`: probabilidad diaria de un salto en el hueco de apertura (0 = ninguno).
    - `jump_scale`: tamano del salto en unidades de vol diaria del activo.
    """

    length_days: int
    drift: dict[str, float] = field(default_factory=dict)
    vol: dict[str, float] = field(default_factory=dict)
    idio_ar: float = 0.0
    tail_dof: float = 0.0
    vol_persistence: float = 0.0
    jump_intensity: float = 0.0
    jump_scale: float = 0.0

    def to_dict(self) -> dict:
        out: dict = {
            "length_days": self.length_days,
            "drift": dict(self.drift),
            "vol": dict(self.vol),
        }
        # Solo se serializan los campos de microestructura si son NO neutros, para que
        # los spec.json existentes (ai_v1) no cambien y el diff sea legible.
        for name in ("idio_ar", "tail_dof", "vol_persistence", "jump_intensity", "jump_scale"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> FactorPhase:
        return cls(
            length_days=int(data["length_days"]),
            drift={str(k): float(v) for k, v in (data.get("drift") or {}).items()},
            vol={str(k): float(v) for k, v in (data.get("vol") or {}).items()},
            idio_ar=float(data.get("idio_ar", 0.0)),
            tail_dof=float(data.get("tail_dof", 0.0)),
            vol_persistence=float(data.get("vol_persistence", 0.0)),
            jump_intensity=float(data.get("jump_intensity", 0.0)),
            jump_scale=float(data.get("jump_scale", 0.0)),
        )

    def with_microstructure(
        self,
        *,
        idio_ar: float | None = None,
        tail_dof: float | None = None,
        vol_persistence: float | None = None,
        jump_intensity: float | None = None,
        jump_scale: float | None = None,
    ) -> FactorPhase:
        """Copia la fase sustituyendo solo los campos de microestructura indicados.
        Lo usa el retrofit determinista para enriquecer specs de ai_v1 sin tocar drift/vol."""
        return FactorPhase(
            length_days=self.length_days,
            drift=dict(self.drift),
            vol=dict(self.vol),
            idio_ar=self.idio_ar if idio_ar is None else idio_ar,
            tail_dof=self.tail_dof if tail_dof is None else tail_dof,
            vol_persistence=self.vol_persistence if vol_persistence is None else vol_persistence,
            jump_intensity=self.jump_intensity if jump_intensity is None else jump_intensity,
            jump_scale=self.jump_scale if jump_scale is None else jump_scale,
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
