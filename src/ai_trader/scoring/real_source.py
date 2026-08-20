"""
EL SUSTRATO QUE DECIDE: muestras sacadas del historico REAL, no de un mundo generado.

Es la implementacion de `scoring.optimize.SampleSource` sobre mercado. Sustituye a la
libreria sintetica como sustrato por defecto del optimizador, y el motivo esta MEDIDO y
publicado: el Spearman entre el ranking real y el sintetico es -0,04 sobre 16
configuraciones (IC95% por bloques [-0,44, +0,49], p = 0,89) y NEGATIVO (-0,67) sobre las
nueve que operan de verdad en los dos mundos. La regla de decision estaba escrita en el
codigo antes de mirar (`RHO_ACCEPT = 0.30`), asi que el sintetico deja de ser criterio de
seleccion. Ver `data/transfer/report_ai_v3.json`.

COMO SE MUESTREA, Y POR QUE ASI
--------------------------------
- Una UNIDAD es una sub-ventana de calendario (`real_substrate.real_windows`), que es
  tambien la unidad de hold-out. `DEFAULT_WINDOW_DAYS` = 544 dias no es un numero redondo:
  es exactamente el que usa el estudio de transferencia publicado, de modo que las cifras
  de los dos sitios se puedan comparar sin traducir nada.
- Una MUESTRA es un fold CPCV dentro de esa ventana: C(6,2) = 15 ventanas OOS por unidad,
  con purga y embargo. Sin esto el sustrato real daria cinco muestras en total y el CVaR de
  la recompensa seria el minimo de cinco numeros.
- El hold-out es TEMPORAL: las `n_validation` ventanas mas RECIENTES se reservan y el CEM
  no las ve nunca. Sortearlas -que es lo que hace el lado sintetico, donde los escenarios
  no tienen orden- permitiria entrenar en 2024 y validar en 2019, que es fuga temporal
  disfrazada de hold-out.

LO QUE ESTE SUSTRATO NO ARREGLA, Y HAY QUE DECIR
------------------------------------------------
El historico real es UN camino con pocos bloques independientes: cinco sub-ventanas de las
que cuatro entrenan. Rankear ahi tiene su propio problema de sobreajuste -- exactamente el
problema que el sintetico venia a resolver y no resolvio. No se tapa: `describe()` publica
el numero de unidades efectivas a cada lado, y `run_optimization` sigue reportando PBO y
DSR sobre este lado igual que sobre el otro.

EL COSTE, QUE NO ES EL MISMO QUE EL DEL SINTETICO
--------------------------------------------------
Cada unidad son 15 backtests con purga sobre los 24 pares que superan el minimo de
historico, y esta MEDIDO: 121 s por (configuracion, unidad) con la cache caliente. Las
cuatro unidades de train son ~8 minutos POR CANDIDATA, asi que una corrida de CEM con
poblacion 20 y 10 iteraciones son ~27 horas. Esta escrito aqui para que nadie lance una
optimizacion completa esperando lo que costaba sobre parquet precalculado.

Los baselines, en cambio, cuestan 1 s por unidad y se cachean: van por `baseline_fold_scores`,
que construye los folds y puntua las carteras pasivas SIN correr ninguna estrategia.

`signals` enchufa el ARCHIVO REAL de senales a las estrategias (lo que devuelve
`signals/feed.py::load_frames`). Por defecto va APAGADO y no por descuido: el estudio de la
capa tematica midio que armar el radar multiplica el coste por 7,9, y encenderlo aqui sin
decirlo convertiria una corrida de horas en una de dias.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.backtest.metrics import DEFAULT_HEADLINE_WEIGHTS, HeadlineWeights
from ai_trader.backtest.validation import SCHEME_CPCV
from ai_trader.config import DEFAULT_CONFIG_PATH, AppConfig, StrategySpec, load_config
from ai_trader.data.real_history import (
    DEFAULT_EXCHANGE,
    DEFAULT_REAL_END,
    DEFAULT_REAL_START,
    build_service,
    fetch_real_bars,
)
from ai_trader.scoring.multiwindow import (
    baseline_fold_scores,
    resolve_purge_days,
    validate_multiwindow,
)
from ai_trader.scoring.real_substrate import (
    N_GROUPS,
    N_TEST_GROUPS,
    RealWindow,
    SymbolAudit,
    audit_real_symbols,
    crypto_universe,
    real_windows,
)
from ai_trader.scoring.sample_eval import SampleEvaluation

logger = logging.getLogger(__name__)

# El mismo troceo que el estudio de transferencia publicado: 2017-09 -> 2026-01 da cinco
# sub-ventanas disjuntas ancladas al final, y una cabecera descartada que se declara.
DEFAULT_WINDOW_DAYS = 544

# Cuantas de las ventanas MAS RECIENTES se reservan. Una, y no dos, porque con cinco
# ventanas reservar dos deja tres para entrenar: el hold-out se comeria el sustrato.
DEFAULT_N_VALIDATION_WINDOWS = 1


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """
    Particion de sub-ventanas en train y validation POR ORDEN CRONOLOGICO.

    No tiene semilla, y esa ausencia es el contenido: no hay nada que sortear. Las ultimas
    ventanas son la validacion porque son el futuro respecto de las demas, que es la unica
    forma de hold-out que significa algo cuando las unidades tienen orden temporal.
    """

    train: tuple[str, ...]
    validation: tuple[str, ...]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_validation(self) -> int:
        return len(self.validation)

    @property
    def seed(self) -> None:
        """Compatibilidad con `ScenarioSplit` al serializar: aqui no hay sorteo."""
        return None


def split_windows_by_time(
    windows: Sequence[RealWindow], *, n_validation: int = DEFAULT_N_VALIDATION_WINDOWS
) -> TemporalSplit:
    """Reserva las `n_validation` sub-ventanas mas recientes como hold-out."""
    if len(windows) < 2:
        raise ValueError("hacen falta al menos 2 sub-ventanas para partir")
    n_val = min(max(int(n_validation), 1), len(windows) - 1)
    labels = [w.label for w in windows]
    return TemporalSplit(train=tuple(labels[:-n_val]), validation=tuple(labels[-n_val:]))


class RealWindowSource:
    """Muestras = folds CPCV dentro de sub-ventanas del historico real.

    Las barras se cargan UNA vez para todo el rango y las comparten todas las ventanas: el
    motor recorta por fechas y el calentamiento de cada sub-ventana sale de su propia
    historia previa, por eso la descarga empieza antes de la primera ventana."""

    def __init__(
        self,
        base_config: AppConfig,
        bars: dict[str, pd.DataFrame],
        windows: Sequence[RealWindow],
        split: TemporalSplit,
        *,
        symbols: tuple[str, ...],
        omitted: tuple[SymbolAudit, ...] = (),
        head_discarded_days: int = 0,
        purge_days: int | None = None,
        starting_equity: float = DEFAULT_STARTING_EQUITY,
        headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
        signals: dict | None = None,
    ) -> None:
        self._config = base_config
        self._bars = bars
        self._windows = {w.label: w for w in windows}
        self.split = split
        self._symbols = symbols
        self._omitted = omitted
        self._head_discarded_days = head_discarded_days
        self._purge_days = (
            resolve_purge_days(base_config) if purge_days is None else purge_days
        )
        self._starting_equity = starting_equity
        self._headline_weights = headline_weights
        self._signals = signals
        self._baseline_cache: dict[str, dict[str, list[float]]] = {}
        self._n_folds: int | None = None

    # ------------------------------------------------------------------ construccion ---

    @classmethod
    def build(
        cls,
        *,
        config_path=DEFAULT_CONFIG_PATH,
        base_config: AppConfig | None = None,
        start: str = DEFAULT_REAL_START,
        end: str = DEFAULT_REAL_END,
        window_days: int = DEFAULT_WINDOW_DAYS,
        n_validation_windows: int = DEFAULT_N_VALIDATION_WINDOWS,
        exchange: str = DEFAULT_EXCHANGE,
        offline: bool = True,
        starting_equity: float = DEFAULT_STARTING_EQUITY,
        headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
        signals: dict | None = None,
    ) -> RealWindowSource:
        """
        Descarga (o lee de cache) el historico, audita el universo y trocea la ventana.

        `offline=True` por defecto: el sustrato del ranking no debe depender de que el
        exchange conteste hoy lo mismo que ayer. La primera vez hay que poblar la cache con
        `offline=False`; a partir de ahi el sustrato es reproducible sin red.
        """
        base_config = base_config or load_config(config_path)
        real_start = pd.Timestamp(start, tz="UTC")
        real_end = pd.Timestamp(end, tz="UTC")

        # El minimo de historico exigible: calentamiento + un grupo de CPCV. Por debajo de
        # eso un simbolo no llega a operarse ni en una ventana OOS.
        min_history = base_config.runner.lookback_days + window_days // N_GROUPS

        requested = crypto_universe(base_config)
        logger.info(
            "Barras reales (%s, %s -> %s) para %d pares cripto%s",
            exchange, start, end, len(requested),
            " [offline: solo cache]" if offline else "",
        )
        bars = fetch_real_bars(
            requested,
            real_start.to_pydatetime(),
            real_end.to_pydatetime(),
            build_service(exchange, offline=offline),
        )
        kept, dropped = audit_real_symbols(
            bars, requested, start=real_start, end=real_end, min_history_days=min_history
        )
        if not kept:
            raise ValueError("Ningun simbolo real supera el minimo de historico")

        symbols = tuple(a.symbol for a in kept)
        windows = real_windows(
            real_start.to_pydatetime(), real_end.to_pydatetime(), window_days
        )
        head = (windows[0].start - real_start.to_pydatetime()).days
        split = split_windows_by_time(windows, n_validation=n_validation_windows)
        logger.info(
            "Sustrato real: %d sub-ventanas de %d dias (%d train, %d validation) | "
            "universo %d/%d pares | cabecera descartada %d dias",
            len(windows), window_days, split.n_train, split.n_validation,
            len(symbols), len(requested), head,
        )
        return cls(
            base_config, bars, windows, split,
            symbols=symbols,
            omitted=tuple(dropped),
            head_discarded_days=head,
            starting_equity=starting_equity,
            headline_weights=headline_weights,
            signals=signals,
        )

    # ------------------------------------------------------------------ el contrato ----

    @property
    def train_units(self) -> tuple[str, ...]:
        return self.split.train

    @property
    def validation_units(self) -> tuple[str, ...]:
        return self.split.validation

    def evaluations(
        self, spec: StrategySpec, units: tuple[str, ...]
    ) -> list[SampleEvaluation]:
        out: list[SampleEvaluation] = []
        for label in units:
            result = self._validate(spec, label)
            if result is None:
                # Una unidad caida penaliza como una muestra fallida del lado sintetico, y
                # no desaparece: si desapareciera, esa configuracion se estaria puntuando
                # sobre menos unidades que las demas y el ranking no seria un ranking.
                out.extend(SampleEvaluation.failure() for _ in range(self._folds_per_unit()))
                continue
            out.extend(_as_sample(f) for f in result.folds)
        return out

    def baseline_scores(self, units: tuple[str, ...]) -> dict[str, list[float]]:
        """Scores de los baselines fold a fold, en el mismo orden que `evaluations`.

        Un baseline solo entra si esta disponible en TODAS las unidades: comparar contra una
        serie con huecos seria comparar contra otra cosa. Se cachean por unidad porque no
        dependen de la estrategia candidata."""
        per_unit: list[dict[str, list[float]]] = []
        for label in units:
            cached = self._baseline_cache.get(label)
            if cached is None:
                window = self._windows[label]
                try:
                    cached = baseline_fold_scores(
                        self._config, self._bars, window.start, window.end,
                        scheme=SCHEME_CPCV,
                        n_groups=N_GROUPS,
                        n_test_groups=N_TEST_GROUPS,
                        purge_days=self._purge_days,
                        starting_equity=self._starting_equity,
                        headline_weights=self._headline_weights,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Baselines no evaluables en %s: %s", label, exc)
                    cached = {}
                self._baseline_cache[label] = cached
            per_unit.append(cached)

        if not per_unit:
            return {}
        complete = set.intersection(*(set(d) for d in per_unit))
        return {
            name: [v for d in per_unit for v in d[name]] for name in sorted(complete)
        }

    def describe(self) -> dict:
        first = min(w.start for w in self._windows.values())
        last = max(w.end for w in self._windows.values())
        return {
            "substrate": "real",
            "exchange_window": {
                "start": first.date().isoformat(),
                "end": (last - timedelta(days=1)).date().isoformat(),
            },
            "n_units": len(self._windows),
            "n_train_units": self.split.n_train,
            "n_validation_units": self.split.n_validation,
            "holdout": "temporal (las ventanas mas recientes)",
            "folds_per_unit": self._n_folds,
            "scheme": SCHEME_CPCV,
            "n_groups": N_GROUPS,
            "n_test_groups": N_TEST_GROUPS,
            "purge_days": self._purge_days,
            "symbols": list(self._symbols),
            "symbols_omitted": [a.as_dict() for a in self._omitted],
            "head_discarded_days": self._head_discarded_days,
            "signals_armed": self._signals is not None,
        }

    # ------------------------------------------------------------------ interno --------

    def _folds_per_unit(self) -> int:
        """Cuantas muestras produce una unidad. Se aprende de la primera validacion que
        sale bien; si ninguna ha salido aun, el valor teorico de C(n_groups, n_test)."""
        if self._n_folds is not None:
            return self._n_folds
        n, k = N_GROUPS, N_TEST_GROUPS
        total = 1
        for i in range(k):
            total = total * (n - i) // (i + 1)
        return total

    def _validate(self, spec: StrategySpec, label: str):
        window = self._windows[label]
        try:
            result = validate_multiwindow(
                self._config, spec, self._bars, window.start, window.end,
                scheme=SCHEME_CPCV,
                n_groups=N_GROUPS,
                n_test_groups=N_TEST_GROUPS,
                purge_days=self._purge_days,
                starting_equity=self._starting_equity,
                headline_weights=self._headline_weights,
                with_baselines=False,
                # El corte unico no interviene y cuesta un backtest entero por unidad: se
                # apaga a proposito, no por descuido.
                compare_single_split=False,
                block_cache={},
                baseline_cache={},
                signals=self._signals,
            )
        except Exception as exc:  # noqa: BLE001 - una unidad mala no tumba la optimizacion
            logger.warning("Unidad real fallida (%s): %s", label, exc)
            return None
        self._n_folds = len(result.folds)
        return result


def _as_sample(fold) -> SampleEvaluation:
    """Un fold de CPCV ES una muestra out-of-sample: mismos campos, mismo significado."""
    return SampleEvaluation(
        score=fold.score,
        sharpe=fold.sharpe,
        turnover=fold.turnover,
        max_drawdown_pct=fold.max_drawdown_pct,
        num_trades=fold.num_trades,
        oos_observations=fold.oos_observations,
        returns_skew=fold.returns_skew,
        returns_kurtosis=fold.returns_kurtosis,
    )
