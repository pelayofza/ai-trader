"""
Geometria de la validacion temporal: donde empieza y acaba cada ventana.

Este modulo NO corre backtests ni conoce el motor: solo produce y audita
particiones del calendario. Se separa a proposito, porque una fuga temporal es un
error de GEOMETRIA (dias que aparecen en los dos lados del corte, o demasiado
juntos) y comprobarlo no deberia exigir ejecutar nada.

Tres esquemas, de menos a mas exigente:

- `single_split_folds`  : el corte 70/30 historico. Se conserva como REFERENCIA, para
                          poder medir en que se diferencia de los otros dos.
- `walk_forward_folds`  : varias ventanas OOS consecutivas. Cada fold entrena con
                          el pasado y se puntua en el tramo siguiente.
- `cpcv_folds`          : Combinatorial Purged Cross-Validation. Parte el rango en
                          N grupos y prueba TODAS las combinaciones de k grupos de
                          test, asi que un mismo tramo se evalua acompanado de
                          contextos distintos y salen C(N, k) ventanas en vez de N.

Y dos defensas, que es lo que distingue esto de "partir por la mitad varias veces":

- PURGA: se borran del train los `purge_days` inmediatamente ANTERIORES a cada
  bloque de test. Una decision tomada el ultimo dia de train puede seguir viva
  varios dias (el runner mantiene posiciones hasta `max_holding_days`), asi que
  sin purga el final del train y el principio del test comparten el desenlace de
  las mismas operaciones.
- EMBARGO: se borran del train los `embargo_days` inmediatamente POSTERIORES a
  cada bloque de test. Los retornos estan serialmente correlacionados: entrenar
  con el dia siguiente al test es entrenar con su eco. Solo muerde cuando hay
  train DESPUES del test, es decir en CPCV; en walk-forward anclado el train
  siempre precede al test y el embargo no tiene nada que recortar.

Convencion: los bloques son SEMIABIERTOS `[start, end)`. La particion antigua no
lo era y por eso el dia del corte caia en las dos ventanas a la vez (un dia de
solape que nadie habia declarado).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations

import pandas as pd

logger = logging.getLogger(__name__)

SCHEME_SINGLE = "single_split"
SCHEME_WALK_FORWARD = "walk_forward"
SCHEME_CPCV = "cpcv"
SCHEMES = (SCHEME_SINGLE, SCHEME_WALK_FORWARD, SCHEME_CPCV)

DEFAULT_SPLIT_RATIO = 0.7
DEFAULT_N_FOLDS = 4  # ventanas OOS del walk-forward
DEFAULT_N_GROUPS = 6  # grupos en que CPCV parte el rango
DEFAULT_N_TEST_GROUPS = 2  # grupos de test por combinacion -> C(6,2) = 15 folds

# Embargo por defecto, en fraccion del rango total. 1% es la regla practica de
# Lopez de Prado; se traduce a dias enteros con un minimo de 1 para que el embargo
# nunca sea decorativo.
DEFAULT_EMBARGO_PCT = 0.01

# Un bloque necesita al menos dos dias con barra para que el motor pueda calcular
# un retorno. Los trozos de train que la purga deja por debajo se descartan.
MIN_BLOCK_DAYS = 2


def _utc_midnight(value: datetime) -> datetime:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.normalize().to_pydatetime().replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True, order=True)
class Block:
    """Tramo de calendario SEMIABIERTO `[start, end)`, en medianoche UTC."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _utc_midnight(self.start))
        object.__setattr__(self, "end", _utc_midnight(self.end))
        if self.end <= self.start:
            raise ValueError(f"Block end must be after start (got {self.start} -> {self.end})")

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    @property
    def last_day(self) -> datetime:
        """Ultimo dia INCLUIDO. Es lo que espera el motor, que recorre [start, end]."""
        return self.end - timedelta(days=1)

    def overlaps(self, other: Block) -> bool:
        return self.start < other.end and other.start < self.end

    def overlap_days(self, other: Block) -> int:
        lo, hi = max(self.start, other.start), min(self.end, other.end)
        return max(0, (hi - lo).days)

    def as_dict(self) -> dict:
        return {
            "start": self.start.date().isoformat(),
            "end": self.last_day.date().isoformat(),
            "days": self.days,
        }

    def __str__(self) -> str:
        return f"[{self.start.date()}, {self.last_day.date()}]"


@dataclass(frozen=True, slots=True)
class LeakageCheck:
    """
    Auditoria de fuga temporal de UN fold. Es la comprobacion que da sentido a la
    purga y al embargo: sin ella, declararlos seria una promesa sin verificar.
    """

    overlap_days: int
    purge_gap_days: int | None  # menor hueco train -> test (None si no hay train antes)
    embargo_gap_days: int | None  # menor hueco test -> train (None si no hay train despues)
    required_purge_days: int
    required_embargo_days: int

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def reasons(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.overlap_days > 0:
            out.append(f"{self.overlap_days} dia(s) presentes en train Y en test")
        if self.purge_gap_days is not None and self.purge_gap_days < self.required_purge_days:
            out.append(
                f"purga insuficiente: {self.purge_gap_days}d < {self.required_purge_days}d"
            )
        if (
            self.embargo_gap_days is not None
            and self.embargo_gap_days < self.required_embargo_days
        ):
            out.append(
                f"embargo insuficiente: {self.embargo_gap_days}d < {self.required_embargo_days}d"
            )
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "overlap_days": self.overlap_days,
            "purge_gap_days": self.purge_gap_days,
            "embargo_gap_days": self.embargo_gap_days,
            "required_purge_days": self.required_purge_days,
            "required_embargo_days": self.required_embargo_days,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class Fold:
    """Un fold: los bloques de train y los de test, ya purgados y embargados."""

    label: str
    scheme: str
    train: tuple[Block, ...]
    test: tuple[Block, ...]
    purge_days: int
    embargo_days: int

    def __post_init__(self) -> None:
        # El train PUEDE quedarse vacio (una purga agresiva se lo come, y el fold sigue
        # siendo una ventana OOS valida). El test no: un fold sin ventana que puntuar no
        # es un fold, y descubrirlo a mitad de ejecucion cuesta minutos de backtest.
        if not self.test:
            raise ValueError(f"Fold '{self.label}' has no test block")

    @property
    def test_days(self) -> int:
        return sum(b.days for b in self.test)

    @property
    def train_days(self) -> int:
        return sum(b.days for b in self.train)

    @property
    def test_start(self) -> datetime:
        return self.test[0].start

    @property
    def test_end(self) -> datetime:
        return self.test[-1].end

    def leakage(self) -> LeakageCheck:
        overlap = sum(r.overlap_days(t) for r in self.train for t in self.test)

        purge_gaps = [
            (t.start - r.end).days for r in self.train for t in self.test if r.end <= t.start
        ]
        embargo_gaps = [
            (r.start - t.end).days for r in self.train for t in self.test if r.start >= t.end
        ]
        return LeakageCheck(
            overlap_days=overlap,
            purge_gap_days=min(purge_gaps) if purge_gaps else None,
            embargo_gap_days=min(embargo_gaps) if embargo_gaps else None,
            required_purge_days=self.purge_days,
            required_embargo_days=self.embargo_days,
        )

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "scheme": self.scheme,
            "train": [b.as_dict() for b in self.train],
            "test": [b.as_dict() for b in self.test],
            "train_days": self.train_days,
            "test_days": self.test_days,
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
            "leakage": self.leakage().as_dict(),
        }


# ------------------------------------------------------------ algebra de bloques ----


def merge_adjacent(blocks: list[Block] | tuple[Block, ...]) -> tuple[Block, ...]:
    """Funde bloques contiguos o solapados en los minimos tramos contiguos."""
    out: list[Block] = []
    for block in sorted(blocks):
        if out and block.start <= out[-1].end:
            out[-1] = Block(out[-1].start, max(out[-1].end, block.end))
        else:
            out.append(block)
    return tuple(out)


def subtract(block: Block, holes: tuple[Block, ...]) -> list[Block]:
    """Lo que queda de `block` tras quitarle los tramos `holes`."""
    pieces = [block]
    for hole in holes:
        remaining: list[Block] = []
        for piece in pieces:
            if not piece.overlaps(hole):
                remaining.append(piece)
                continue
            if hole.start > piece.start:
                remaining.append(Block(piece.start, hole.start))
            if hole.end < piece.end:
                remaining.append(Block(hole.end, piece.end))
        pieces = remaining
    return pieces


def purge_and_embargo(
    train: tuple[Block, ...],
    test: tuple[Block, ...],
    *,
    purge_days: int,
    embargo_days: int,
    min_block_days: int = MIN_BLOCK_DAYS,
) -> tuple[Block, ...]:
    """
    Recorta del train todo lo que toque la zona prohibida de cada bloque de test:
    `[test.start - purge_days, test.end + embargo_days)`.

    Con purge=embargo=0 sigue eliminando el solape puro, que es el minimo
    innegociable. Los fragmentos que quedan por debajo de `min_block_days` se
    descartan enteros: un bloque de un dia no produce ni un retorno.
    """
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be non-negative")

    holes = merge_adjacent(
        [
            Block(t.start - timedelta(days=purge_days), t.end + timedelta(days=embargo_days))
            for t in test
        ]
    )
    kept: list[Block] = []
    for block in train:
        kept.extend(p for p in subtract(block, holes) if p.days >= min_block_days)
    return merge_adjacent(kept)


def partition(start: datetime, end: datetime, n_groups: int) -> tuple[Block, ...]:
    """Parte `[start, end)` en `n_groups` tramos contiguos de longitud casi igual.

    El resto de la division se reparte entre los PRIMEROS grupos, de forma
    determinista: la misma peticion da siempre la misma rejilla."""
    start, end = _utc_midnight(start), _utc_midnight(end)
    if end <= start:
        raise ValueError("end must be after start")
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")

    total = (end - start).days
    if total < n_groups * MIN_BLOCK_DAYS:
        raise ValueError(
            f"Range of {total} days cannot be split into {n_groups} groups of "
            f"at least {MIN_BLOCK_DAYS} days"
        )

    size, extra = divmod(total, n_groups)
    blocks: list[Block] = []
    cursor = start
    for i in range(n_groups):
        width = size + (1 if i < extra else 0)
        blocks.append(Block(cursor, cursor + timedelta(days=width)))
        cursor += timedelta(days=width)
    return tuple(blocks)


def resolve_embargo_days(start: datetime, end: datetime, *, pct: float = DEFAULT_EMBARGO_PCT) -> int:
    """Embargo por defecto: `pct` del rango, redondeado, con minimo de 1 dia."""
    total = (_utc_midnight(end) - _utc_midnight(start)).days
    return max(1, round(pct * total))


# ------------------------------------------------------------------ esquemas --------


def single_split_folds(
    start: datetime,
    end: datetime,
    *,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    split_date: datetime | None = None,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> tuple[Fold, ...]:
    """El corte unico historico, expresado como un fold. Se conserva para poder
    COMPARARLO con los esquemas multiventana, no porque siga siendo suficiente."""
    cutoff = resolve_split_cutoff(start, end, split_ratio=split_ratio, split_date=split_date)
    test = (Block(cutoff, _utc_midnight(end) + timedelta(days=1)),)
    train = purge_and_embargo(
        (Block(start, cutoff),), test, purge_days=purge_days, embargo_days=embargo_days
    )
    return (
        Fold(
            label="single",
            scheme=SCHEME_SINGLE,
            train=train,
            test=test,
            purge_days=purge_days,
            embargo_days=embargo_days,
        ),
    )


def walk_forward_folds(
    start: datetime,
    end: datetime,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    purge_days: int = 0,
    embargo_days: int | None = None,
    anchored: bool = True,
    min_block_days: int = MIN_BLOCK_DAYS,
) -> tuple[Fold, ...]:
    """
    `n_folds` ventanas OOS consecutivas. El rango se parte en `n_folds + 1` grupos:
    el primero solo puede entrenar, y cada uno de los siguientes es el test de un
    fold cuyo train es todo lo anterior (anclado) o solo el grupo previo (rodante).

    Anclado por defecto porque es lo que hace un operador real: no tira el pasado.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    embargo = resolve_embargo_days(start, end) if embargo_days is None else embargo_days

    groups = partition(start, end, n_folds + 1)
    folds: list[Fold] = []
    for i in range(1, len(groups)):
        test = (groups[i],)
        raw_train = (
            (Block(groups[0].start, groups[i].start),) if anchored else (groups[i - 1],)
        )
        train = purge_and_embargo(
            raw_train, test, purge_days=purge_days, embargo_days=embargo,
            min_block_days=min_block_days,
        )
        if not train:
            logger.warning(
                "walk-forward fold %d se queda sin train tras purgar %dd; se conserva "
                "solo como ventana OOS", i, purge_days,
            )
        folds.append(
            Fold(
                label=f"wf{i}",
                scheme=SCHEME_WALK_FORWARD,
                train=train,
                test=test,
                purge_days=purge_days,
                embargo_days=embargo,
            )
        )
    return tuple(folds)


def cpcv_folds(
    start: datetime,
    end: datetime,
    *,
    n_groups: int = DEFAULT_N_GROUPS,
    n_test_groups: int = DEFAULT_N_TEST_GROUPS,
    purge_days: int = 0,
    embargo_days: int | None = None,
    min_block_days: int = MIN_BLOCK_DAYS,
) -> tuple[Fold, ...]:
    """
    Combinatorial Purged Cross-Validation: `C(n_groups, n_test_groups)` folds.

    Se parte el rango en `n_groups` tramos y se prueban TODAS las combinaciones de
    `n_test_groups` como test; el resto es train, purgado y embargado alrededor de
    cada tramo de test. Los grupos de test contiguos se funden en un unico bloque,
    para que el fold vea un tramo continuo y no dos artificialmente cortados.

    Frente al walk-forward aporta dos cosas: mas ventanas con el mismo historico
    (15 en vez de 4 con la configuracion por defecto) y tramos de test evaluados en
    compania de contextos de train distintos, que es lo que revela si un resultado
    depende del trozo concreto de historia con el que se acompano.
    """
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be >= 1 and < n_groups")
    embargo = resolve_embargo_days(start, end) if embargo_days is None else embargo_days

    groups = partition(start, end, n_groups)
    folds: list[Fold] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test = merge_adjacent([groups[i] for i in combo])
        rest = merge_adjacent([groups[i] for i in range(n_groups) if i not in combo])
        train = purge_and_embargo(
            rest, test, purge_days=purge_days, embargo_days=embargo,
            min_block_days=min_block_days,
        )
        if not train:
            logger.warning(
                "CPCV fold %s se queda sin train tras purgar %dd/embargar %dd",
                combo, purge_days, embargo,
            )
        folds.append(
            Fold(
                label="cpcv" + "-".join(str(i) for i in combo),
                scheme=SCHEME_CPCV,
                train=train,
                test=test,
                purge_days=purge_days,
                embargo_days=embargo,
            )
        )
    return tuple(folds)


def build_folds(
    start: datetime,
    end: datetime,
    *,
    scheme: str = SCHEME_WALK_FORWARD,
    n_folds: int = DEFAULT_N_FOLDS,
    n_groups: int = DEFAULT_N_GROUPS,
    n_test_groups: int = DEFAULT_N_TEST_GROUPS,
    purge_days: int = 0,
    embargo_days: int | None = None,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    anchored: bool = True,
) -> tuple[Fold, ...]:
    """Punto de entrada unico por nombre de esquema (lo que consumen CLI y scoring)."""
    if scheme == SCHEME_SINGLE:
        return single_split_folds(
            start, end, split_ratio=split_ratio,
            purge_days=purge_days, embargo_days=embargo_days or 0,
        )
    if scheme == SCHEME_WALK_FORWARD:
        return walk_forward_folds(
            start, end, n_folds=n_folds, purge_days=purge_days,
            embargo_days=embargo_days, anchored=anchored,
        )
    if scheme == SCHEME_CPCV:
        return cpcv_folds(
            start, end, n_groups=n_groups, n_test_groups=n_test_groups,
            purge_days=purge_days, embargo_days=embargo_days,
        )
    raise ValueError(f"Unknown validation scheme '{scheme}'; expected one of {SCHEMES}")


# ------------------------------------------------------------------ auditoria -------


def assert_no_leakage(folds: tuple[Fold, ...] | list[Fold]) -> None:
    """
    Falla si CUALQUIER fold tiene fuga temporal. El motor la llama ANTES de gastar
    computo: un plan de validacion con fuga no merece que se corra.
    """
    offenders = [(f, f.leakage()) for f in folds]
    broken = [(f, chk) for f, chk in offenders if not chk.ok]
    if not broken:
        return
    detail = "; ".join(f"{f.label}: {', '.join(chk.reasons)}" for f, chk in broken)
    raise ValueError(f"Temporal leakage in {len(broken)}/{len(offenders)} fold(s) -> {detail}")


def coverage(folds: tuple[Fold, ...] | list[Fold]) -> dict:
    """Cuanto calendario cubren los tests y cuantas veces se pisa cada dia. Un dia
    evaluado en varios folds NO es fuga (los folds son evaluaciones alternativas del
    mismo pasado), pero si dice cuanta informacion independiente hay de verdad."""
    spans = merge_adjacent([b for f in folds for b in f.test])
    covered = sum(b.days for b in spans)
    evaluated = sum(f.test_days for f in folds)
    return {
        "n_folds": len(folds),
        "calendar_days_covered": covered,
        "evaluated_days": evaluated,
        "reuse_factor": round(evaluated / covered, 2) if covered else 0.0,
    }


# ----------------------------------------------------------------- corte unico ------


def resolve_split_cutoff(
    start: datetime,
    end: datetime,
    *,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    split_date: datetime | None = None,
) -> datetime:
    """
    Frontera train/test del corte UNICO (esquema historico).

    Vive aqui, junto al resto de la geometria, y se re-exporta desde el motor para
    no romper a quien ya la importaba: los baselines tienen que puntuarse en
    EXACTAMENTE la misma ventana out-of-sample que la estrategia, y la unica forma
    de garantizarlo es que ambos deriven el corte del mismo sitio.
    """
    if split_date is not None:
        if not start < split_date < end:
            raise ValueError("split_date must fall strictly between start and end")
        return split_date

    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1")

    span = end - start
    return start + timedelta(days=int(span.days * split_ratio))
