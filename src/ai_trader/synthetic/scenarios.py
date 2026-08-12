from __future__ import annotations

from dataclasses import dataclass, field, fields

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


# Parametros de un canal de observacion, en un solo sitio: la lista que decide que se
# serializa (solo lo que NO es el default neutro) y la que el emisor recorre. Es el mismo
# patron que MICROSTRUCTURE_FIELDS y por la misma razon: anadir un mando aqui y en
# SignalChannel es todo lo que hace falta, y los spec.json ya escritos no cambian.
SIGNAL_CHANNEL_FIELDS: tuple[str, ...] = (
    "rho",
    "lead_days",
    "noise_ar",
    "informative_share",
    "coverage",
    "corr_group",
)

# Tope de |phi| del ruido, el mismo que el AR(1) del motor (`engine._MAX_AR`): por encima
# la serie deja de ser estacionaria y sqrt(1-phi^2) deja de ser la escala correcta.
MAX_NOISE_AR = 0.98


@dataclass(slots=True, frozen=True)
class SignalChannel:
    """
    Un CANAL DE OBSERVACION: como se ve, desde fuera, algo que el mundo ya ha decidido.

        senal_t = rho * z(retorno_t->t+h) + sqrt(1 - rho^2) * ruido_t

    No es una senal: es el canal por el que una senal se observa. Esa distincion es todo
    el argumento de coste de esta pieza. Simular "el sentimiento de Twitter" —su nivel, su
    estacionalidad, su distribucion— pide un generador aprendido con millones de
    parametros, y con el llegaria la circularidad: el mundo lo habriamos ajustado nosotros.
    Simular el CANAL cuesta cinco numeros interpretables, y lo que se publica al final no
    son "los mejores parametros" sino el rho a partir del cual la estrategia deja de
    perder dinero, que es una propiedad del DISENO y no de ningun historico.

    LOS CINCO MANDOS, Y QUE PREGUNTA CONTESTA CADA UNO
    - `rho`: capacidad predictiva REAL del canal (su IC). Es el eje que se barre.
    - `lead_days` (h): cuantos dias se adelanta. El retorno que la senal anticipa es el de
      t -> t+h, asi que h=1 es "sabe lo de manana" y h=10, "sabe lo de las dos semanas".
    - `noise_ar` (phi): autocorrelacion AR(1) del ruido. Sin ella el canal es confeti y no
      una serie: nada de lo que se mide sobre senales reales —persistencia, rachas— existe,
      y la puerta se enfrentaria a un problema mas facil que el de verdad.
    - `informative_share`: fraccion de dias en los que el canal SI lleva informacion. Los
      demas son falsos positivos: picos indistinguibles de los buenos que no anticipan
      nada. Son lo unico que le da COSTE a la puerta; sin ellos, cerrar ante una senal
      nunca renuncia a nada y el canal gana por construccion.
    - `coverage`: fraccion de dias con observacion. El resto no son ceros, son AUSENCIAS
      (el radar las ve como falta de cobertura y se salta la puerta).
    - `corr_group`: que canales son LA MISMA APUESTA. Los que comparten grupo comparten el
      sorteo del ruido y de las mascaras; los de grupos distintos son independientes. No es
      decoracion: por la ley fundamental del gestor activo, cinco canales de rho=0,03 poco
      correlacionados valen mas que uno de 0,08, y sin este mando no se distingue
      multiplicar apuestas de repetir la misma en voz mas alta.

    TODO ESTA PARAMETRIZADO EN POSITIVO, y es una regla dura: **0 = MENOS edge, nunca
    mas**. `informative_share` y `coverage` van en positivo justo para que un default
    olvidado degrade a "sin senal" y no a "senal perfecta" —que es como se escribiria
    `false_positive_rate = 0`, el mismo hecho con el signo que miente—. Con los defaults de
    esta clase el canal emite ruido puro cero dias al ano: no hay forma de conseguir edge
    por descuido. `corr_group` sigue la misma regla y por eso su default ("") significa
    "todos en la MISMA apuesta": la independencia hay que declararla para reclamarla.

    La equivalencia con el nombre negativo, para que nadie tenga que deducirla:
    `false_positive_rate = 1 - informative_share`.
    """

    name: str
    rho: float = 0.0
    lead_days: int = 1
    noise_ar: float = 0.0
    informative_share: float = 0.0
    coverage: float = 0.0
    corr_group: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("SignalChannel needs a name")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1] (got {self.rho})")
        if self.lead_days < 1:
            raise ValueError(f"lead_days must be >= 1 (got {self.lead_days})")
        if not -MAX_NOISE_AR <= self.noise_ar <= MAX_NOISE_AR:
            raise ValueError(
                f"noise_ar must be in [-{MAX_NOISE_AR}, {MAX_NOISE_AR}] (got {self.noise_ar})"
            )
        if not 0.0 <= self.informative_share <= 1.0:
            raise ValueError(
                f"informative_share must be in [0, 1] (got {self.informative_share})"
            )
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError(f"coverage must be in [0, 1] (got {self.coverage})")

    @property
    def false_positive_rate(self) -> float:
        """El mismo mando con el nombre negativo, para poder citarlo sin reescribirlo."""
        return 1.0 - self.informative_share

    @property
    def expected_ic(self) -> float:
        """
        IC que este canal DEBERIA entregar, que no es `rho` salvo con todo al maximo.

        Un dia no informativo emite ruido de la misma varianza (el emisor iguala varianzas
        a proposito, para que un falso positivo no se distinga por el tamano), asi que la
        correlacion contra el futuro se diluye en proporcion a los dias informativos. Es la
        cifra contra la que el estudio contrasta el IC medido: si no coinciden, el canal no
        entrega lo que declara y el barrido esta midiendo otra cosa.
        """
        return self.rho * self.informative_share

    @property
    def group(self) -> str:
        """Grupo de correlacion efectivo. Vacio es un grupo como cualquier otro -el
        compartido- y no 'sin grupo': ver la regla del default en el docstring."""
        return self.corr_group

    def to_dict(self) -> dict:
        out: dict = {"name": self.name}
        # Solo lo que se aparta del default neutro, igual que la microestructura de fase:
        # asi un canal apagado no ensucia el spec y el diff dice lo que se toco.
        for field_name in SIGNAL_CHANNEL_FIELDS:
            value = getattr(self, field_name)
            if value != _CHANNEL_DEFAULTS[field_name]:
                out[field_name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> SignalChannel:
        return cls(
            name=str(data["name"]),
            **{
                field_name: type(_CHANNEL_DEFAULTS[field_name])(data[field_name])
                for field_name in SIGNAL_CHANNEL_FIELDS
                if field_name in data
            },
        )


_CHANNEL_DEFAULTS: dict = {
    f.name: f.default for f in fields(SignalChannel) if f.name in SIGNAL_CHANNEL_FIELDS
}


@dataclass(slots=True, frozen=True)
class ScenarioSpec:
    """
    La descripcion completa de un escenario macro. Es lo que la IA disena.

    - phases: la evolucion del regimen de factores en el tiempo.
    - shocks: eventos puntuales superpuestos.
    - asset_tilts: deriva diaria EXTRA por simbolo, para respuestas idiosincraticas
      que los factores comunes no capturan (p.ej. "embargo de petroleo -> XOM sube
      aunque la bolsa caiga"). Honra el "la IA analiza la respuesta de CADA activo".
    - signals: canales de OBSERVACION del escenario. Vacio (el default) = mundo sin
      senales externas, que es como se genero todo lo publicado. No los produce el motor
      de paths sino un pase aparte (`synthetic/signal_channel.py`), por una razon que no
      es de gusto: al emitirse DESPUES, sobre barras ya cerradas, la no interferencia con
      la secuencia RNG del motor no es una promesa que haya que auditar leyendo codigo,
      es imposible por construccion.
    """

    id: str
    name: str
    narrative: str
    phases: tuple[FactorPhase, ...]
    shocks: tuple[FactorShock, ...] = ()
    asset_tilts: dict[str, float] = field(default_factory=dict)
    signals: tuple[SignalChannel, ...] = ()

    @property
    def horizon_days(self) -> int:
        return sum(p.length_days for p in self.phases)

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "name": self.name,
            "narrative": self.narrative,
            "phases": [p.to_dict() for p in self.phases],
            "shocks": [s.to_dict() for s in self.shocks],
            "asset_tilts": dict(self.asset_tilts),
        }
        # La clave solo aparece si hay canales, de modo que los spec.json de ai_v1/v2/v3
        # no cambian ni un byte y las librerias publicadas se siguen regenerando iguales.
        if self.signals:
            out["signals"] = [c.to_dict() for c in self.signals]
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ScenarioSpec:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            narrative=str(data.get("narrative", "")),
            phases=tuple(FactorPhase.from_dict(p) for p in data.get("phases", [])),
            shocks=tuple(FactorShock.from_dict(s) for s in data.get("shocks", [])),
            asset_tilts={str(k): float(v) for k, v in (data.get("asset_tilts") or {}).items()},
            signals=tuple(SignalChannel.from_dict(c) for c in data.get("signals", [])),
        )
