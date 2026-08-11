from __future__ import annotations

from dataclasses import dataclass, field

# Estos son los objetos que la IA produce (el "que pasa" de cada escenario) y que el
# motor numerico consume (el "como se ve en velas"). Son la frontera entre las dos
# piezas: 100% serializables a JSON para poder auditarlos y no re-ejecutar la IA.

# Campos de microestructura de una fase, en un solo sitio: es la lista que decide que se
# serializa (solo lo NO neutro) y la que el motor expande a timelines por dia. Anadir un
# campo nuevo aqui y en FactorPhase es todo lo que hace falta.
MICROSTRUCTURE_FIELDS: tuple[str, ...] = (
    "idio_ar",
    "tail_dof",
    "vol_persistence",
    "vol_news",
    "jump_intensity",
    "jump_scale",
    "beta_stress",
)


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
    - `vol_news`: fraccion de esa persistencia que reacciona a la NOTICIA del dia anterior
      (el alpha del GARCH; el resto es inercia). Sube el clustering medible a lag 1 sin
      tocar la persistencia total. 0 = el reparto por defecto del motor, que es el que
      genero ai_v2: el campo es un OVERRIDE, igual que `tail_dof`=0 significa "gaussiano".
    - `jump_intensity`: probabilidad diaria de un salto en el hueco de apertura (0 = ninguno).
    - `jump_scale`: tamano del salto en unidades de vol diaria del activo.
    - `beta_stress`: subida FRACCIONAL de las cargas factoriales durante la fase
      (0.4 = betas un 40% mayores). Es lo unico que permite que la correlacion entre
      activos se dispare en las caidas: con betas congeladas la correlacion de un modelo
      de factores es una constante del universo, no del regimen.
    """

    length_days: int
    drift: dict[str, float] = field(default_factory=dict)
    vol: dict[str, float] = field(default_factory=dict)
    idio_ar: float = 0.0
    tail_dof: float = 0.0
    vol_persistence: float = 0.0
    vol_news: float = 0.0
    jump_intensity: float = 0.0
    jump_scale: float = 0.0
    beta_stress: float = 0.0

    def to_dict(self) -> dict:
        out: dict = {
            "length_days": self.length_days,
            "drift": dict(self.drift),
            "vol": dict(self.vol),
        }
        # Solo se serializan los campos de microestructura si son NO neutros, para que
        # los spec.json existentes (ai_v1) no cambien y el diff sea legible.
        for name in MICROSTRUCTURE_FIELDS:
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
            **{name: float(data.get(name, 0.0)) for name in MICROSTRUCTURE_FIELDS},
        )

    def with_microstructure(self, **overrides: float) -> FactorPhase:
        """Copia la fase sustituyendo solo los campos de microestructura indicados.
        Lo usa el retrofit determinista para enriquecer specs de ai_v1 sin tocar drift/vol."""
        unknown = set(overrides) - set(MICROSTRUCTURE_FIELDS)
        if unknown:
            raise TypeError(f"Unknown microstructure field(s): {sorted(unknown)}")
        return FactorPhase(
            length_days=self.length_days,
            drift=dict(self.drift),
            vol=dict(self.vol),
            **{
                name: float(overrides.get(name, getattr(self, name)))
                for name in MICROSTRUCTURE_FIELDS
            },
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
    # Dispersion del dia del shock ENTRE paths del ensemble: cada path lo recoloca en
    # [day-jitter, day+jitter] de forma determinista por semilla. 0 = dia fijo (como
    # antes). Evita que un crash caiga SIEMPRE el mismo dia en los 30 paths.
    jitter_days: int = 0

    def to_dict(self) -> dict:
        out = {"day": self.day, "factor": self.factor, "magnitude": self.magnitude}
        if self.jitter_days:
            out["jitter_days"] = self.jitter_days
        return out

    @classmethod
    def from_dict(cls, data: dict) -> FactorShock:
        return cls(
            day=int(data["day"]),
            factor=str(data["factor"]),
            magnitude=float(data["magnitude"]),
            jitter_days=int(data.get("jitter_days", 0)),
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
