"""
El estudio de divergencia live-vs-backtest.

El problema de testear esto es que la medicion REAL no se puede correr todavia: hacen
falta meses de diario y hoy hay dos dias. Asi que los tests hacen lo unico que sustituye
a esperar: fabrican un diario cuyo desvio se conoce EXACTAMENTE —se inyecta un arrastre
de referencia y un exceso de coste concretos— y exigen que el estudio los recupere.

El diario "en vivo" no se escribe a mano: se genera corriendo el motor de backtest de
verdad con un `MemoryJournal` y despues se PERTURBA. Asi lo que se parea son lineas con
la forma real que escribe el runner, no una maqueta que podria haber derivado del esquema.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.app.journal import exit_record, order_record
from ai_trader.app.runner import RunnerConfig
from ai_trader.backtest import divergence_study as ds
from ai_trader.backtest.engine import BacktestEngine
from ai_trader.config import AppConfig, StrategySpec
from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine, fill_price
from ai_trader.risk.engine import RiskLimits
from ai_trader.shared.schemas import ExecutionResult, OrderStatus, Position, Side

SYMBOL = "BTC/USDT"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------- utilidades ----


def trending_df(n_days: int, start_price: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Rampa alcista determinista: cada cierre marca maximo nuevo y el momentum entra."""
    index = pd.DatetimeIndex(
        [START - timedelta(days=200) + timedelta(days=i) for i in range(n_days)],
        name="timestamp",
    )
    closes = np.array([start_price + i * step for i in range(n_days)])
    opens = np.array([start_price, *closes[:-1]])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.1,
            "low": np.minimum(opens, closes) - 0.1,
            "close": closes,
            "volume": 5_000.0,
        },
        index=index,
    )


def make_config() -> AppConfig:
    return AppConfig(
        runner=RunnerConfig(
            symbols=[SYMBOL],
            lookback_days=120,
            max_holding_days=5,
            symbol_cooldown_hours=0,
            max_trades_per_cycle=5,
        ),
        risk=RiskLimits(
            min_confidence_per_trade=0.50,
            risk_fraction_per_trade=0.10,
            max_symbol_exposure_usd=1_000_000.0,
            max_total_exposure_usd=1_000_000.0,
            max_daily_loss_usd=1_000_000.0,
        ),
        execution=PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0),
        strategies=[
            StrategySpec(
                type="crypto_momentum",
                id="mom",
                params={"min_bars": 30, "require_breakout": True, "min_atr_pct": 0.1},
            )
        ],
    )


def run_journal(config: AppConfig, bars: dict, start: datetime, end: datetime) -> list[dict]:
    """Un diario real, generado conduciendo el motor con un `MemoryJournal`."""
    return ds.resimulate(config, bars, start, end)


def perturb(
    records: list[dict],
    *,
    reference_drift: float = 0.0,
    extra_slippage_bps: float = 0.0,
    fill_hour: int = 19,
) -> list[dict]:
    """
    Convierte un diario re-simulado en uno con pinta de VIVO, con un desvio conocido.

    Tres cosas, que son las tres que de verdad separan a los dos mundos:
    - la referencia se decide con otro precio (en vivo, un cierre diario ya viejo),
    - el coste cobrado se desvia del modelado,
    - el fill ocurre a media tarde y no a medianoche.
    """
    out = copy.deepcopy(records)
    for record in out:
        day = datetime.fromisoformat(record["timestamp"])
        executed = day.replace(hour=fill_hour, minute=0, second=0, microsecond=0)
        for symbol_block in record.get("symbols", []):
            for order, fill in zip(symbol_block["orders"], symbol_block["fills"]):
                order["reference_price"] = round(
                    order["reference_price"] * (1.0 + reference_drift), 8
                )
                order["decided_at"] = (executed - timedelta(seconds=6)).isoformat()
                if fill.get("filled_price") is None:
                    continue
                fill["slippage_bps"] = round(fill["slippage_bps"] + extra_slippage_bps, 6)
                fill["filled_price"] = fill_price(
                    order["reference_price"], order["side"], fill["slippage_bps"]
                )
                fill["executed_at"] = executed.isoformat()
        for exit_fill in record.get("exits", []):
            exit_fill["executed_at"] = executed.isoformat()
            exit_fill["decided_at"] = (executed - timedelta(seconds=6)).isoformat()
    return out


@pytest.fixture(scope="module")
def world() -> dict:
    """Un mundo entero: barras, config, ventana y el diario re-simulado sin perturbar."""
    config = make_config()
    bars = {SYMBOL: trending_df(400)}
    start, end = START, START + timedelta(days=59)
    return {
        "config": config,
        "bars": bars,
        "start": start,
        "end": end,
        "records": run_journal(config, bars, start, end),
    }


def plan_for(world: dict, **over) -> dict:
    plan = {
        "config_path": "config/default.toml",
        "journal_path": "data/live/cycles.jsonl",
        "offline": True,
        "starting_equity": 10_000.0,
        "symbols": [SYMBOL],
        "cycle_interval_seconds": 900,
        "reference_cost_bps": 15.0,
        "thresholds": {},
    }
    plan.update(over)
    return plan


def hourly_df(levels: dict[str, float], drift_per_day: float) -> pd.DataFrame:
    """
    Barras 1H deterministas, una serie por dia: el dia arranca a las 00:00 en el precio
    con el que ESE dia se decidio y sube linealmente `drift_per_day` (en fraccion) hasta
    las 23:00. Anclarlas al precio de cada dia es lo que deja el desplazamiento conocido:
    con un nivel fijo se estaria midiendo la tendencia de la rampa, no la latencia.
    """
    stamps: list[pd.Timestamp] = []
    closes: list[float] = []
    for day, level in sorted(levels.items()):
        base = pd.Timestamp(day, tz="UTC")
        for hour in range(24):
            stamps.append(base + pd.Timedelta(hours=hour))
            closes.append(level * (1.0 + drift_per_day * hour / 23.0))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=pd.DatetimeIndex(stamps, name="timestamp"),
    )


def analyze(world: dict, records: list[dict], *, hourly=None, **kwargs) -> dict:
    """
    `analyze` con las series del mundo ya cargadas, sin tocar disco ni red.

    `hourly={}` por defecto A PROPOSITO: sin eso el estudio leeria la cache 1H REAL del
    disco y tasaria la latencia de unos precios inventados contra el bitcoin de verdad.
    Los tests que si quieren medir esa pierna pasan sus propias barras.
    """
    return ds.analyze(
        records,
        world["config"],
        plan_for(world),
        offline=True,
        bars=world["bars"],
        hourly={} if hourly is None else hourly,
        **kwargs,
    )


# ------------------------------------------------------------------ la potencia ------


class TestPotencia:
    """Un mes de calendario, o no hay cifra. Es la regla que impide que el estudio
    publique una divergencia que tendria el mismo aspecto que la buena."""

    def test_un_diario_corto_no_publica_cifra_y_dice_por_que(self, world):
        corto = [r for r in world["records"] if r["timestamp"] < (START + timedelta(days=5)).isoformat()]

        report = analyze(world, corto)

        assert report["status"] == ds.STATUS_NO_POWER
        assert report["power"]["sufficient"] is False
        assert report["power"]["missing_days"] > 0
        # Ni una sola cifra de divergencia: no es que salgan a cero, es que no salen.
        assert report["decisions"] is None
        assert report["fill_price"] is None
        assert report["cost"] is None
        assert report["latency"] is None
        assert report["verdict"] is None
        assert any("dias" in reason for reason in report["power"]["reasons"])

    def test_un_diario_disperso_no_cuela_por_tener_span(self, world):
        """Dos dias al principio y dos al final dan 59 dias de span y no han observado
        nada. Las dos condiciones existen justo para eso."""
        primeros = world["records"][:2]
        ultimos = world["records"][-2:]

        power = ds.check_power(ds.journal_span(primeros + ultimos))

        assert power["sufficient"] is False
        assert power["span_days"] >= ds.MIN_JOURNAL_DAYS
        assert power["n_days_with_cycles"] == 4

    def test_con_calendario_suficiente_si_publica(self, world):
        report = analyze(world, perturb(world["records"]))

        assert report["status"] == ds.STATUS_MEASURED
        assert report["verdict"] is not None


# --------------------------------------------------- (1)(2) la descomposicion --------


class TestDescomposicion:
    """La diferencia de precio se parte en tres, y los tres SUMAN. Si no cerrara, una
    pierna estaria absorbiendo el error de otra sin decirlo."""

    def test_las_tres_piernas_suman_el_total(self, world):
        report = analyze(world, perturb(world["records"], reference_drift=0.004))
        price = report["fill_price"]

        assert price["n_repriced"] > 0
        assert price["decomposition_ok"] is True
        # El residuo no es cero, y el motivo esta declarado: los precios se archivan
        # redondeados a 8 decimales. Lo que se exige es que se quede en ese orden de
        # magnitud y no en el de una divergencia con significado.
        assert price["decomposition_residual_max"] <= ds.DECOMPOSITION_TOLERANCE_BPS
        assert price["decomposition_residual_max"] < 1e-5

    def test_recupera_el_arrastre_de_referencia_inyectado(self, world):
        # 40 pb de arrastre en la referencia, en compras: en vivo se paga 40 pb mas.
        report = analyze(world, perturb(world["records"], reference_drift=0.004))

        referencia = report["fill_price"]["components"]["reference_bps"]["median"]
        assert referencia == pytest.approx(40.0, abs=0.5)
        # Y el coste no se contamina: el deslizamiento no se toco.
        assert report["fill_price"]["components"]["cost_bps"]["median"] == pytest.approx(0.0, abs=1e-6)

    def test_recupera_el_exceso_de_coste_inyectado(self, world):
        report = analyze(world, perturb(world["records"], extra_slippage_bps=7.5))

        assert report["fill_price"]["components"]["cost_bps"]["median"] == pytest.approx(7.5, abs=1e-6)
        # Sin arrastre, la pierna de referencia se queda en cero.
        assert report["fill_price"]["components"]["reference_bps"]["median"] == pytest.approx(0.0, abs=1e-6)

    def test_el_censo_de_entradas_distingue_por_que_se_cae_cada_una(self, world):
        """Una orden sin llenar es una decision que no se ejecuto; una sin barra del dia
        es un simbolo que el backtest no habria podido operar. Contarlas juntas escondria
        la unica de las dos que es un problema de datos."""
        price = analyze(world, perturb(world["records"]))["fill_price"]

        assert price["n_entries"] == price["n_unfilled"] + price["n_without_bar"] + price["n_repriced"]
        assert price["n_repriced"] > 0

    def test_sin_desvio_ninguno_la_divergencia_es_cero(self, world):
        """El control. Un diario que no se ha tocado tiene que dar cero: si diera algo,
        el estudio estaria midiendo su propia aritmetica."""
        report = analyze(world, perturb(world["records"]))

        assert report["fill_price"]["total_bps"]["median"] == pytest.approx(0.0, abs=1e-6)
        assert report["fill_price"]["total_bps"]["max_abs"] == pytest.approx(0.0, abs=1e-6)

    def test_el_coste_se_compara_contra_el_modelo_no_contra_una_constante(self, world):
        report = analyze(world, perturb(world["records"], extra_slippage_bps=7.5))
        cost = report["cost"]

        assert cost["n_fills"] > 0
        assert cost["slippage_bps"]["abs_gap_median"] == pytest.approx(7.5, abs=1e-6)
        # El modelado NO es el plano del config: depende del simbolo y del tamano.
        assert cost["slippage_bps"]["modeled"]["median"] != world["config"].execution.slippage_bps


# --------------------------------------------- (3) decisiones que no se tomaron ------


class TestDecisiones:
    """La pierna que detecta lo que el PnL esconde."""

    def test_las_decisiones_se_deduplican_por_dia(self, world):
        """En vivo la misma senal sobre la misma barra puede repetirse en varios ciclos
        del dia. Contarla 96 veces contra la unica del backtest no seria una divergencia,
        seria una diferencia de cadencia."""
        repetido = []
        for record in world["records"][:40]:
            for _ in range(4):
                repetido.append(copy.deepcopy(record))

        keys = ds.decision_keys(repetido)
        una_vez = ds.decision_keys(world["records"][:40])

        assert keys["signals"] == una_vez["signals"]

    def test_un_mundo_al_que_le_faltan_senales_falla_la_regla(self, world):
        """Si en vivo se generan la mitad de las senales, el problema no es el coste."""
        mutilado = copy.deepcopy(perturb(world["records"]))
        for i, record in enumerate(mutilado):
            if i % 2 == 0:
                for symbol_block in record.get("symbols", []):
                    symbol_block["signals"] = []
                    symbol_block["risk"] = []

        report = analyze(world, mutilado)
        regla = report["verdict"]["rules"]["decisions"]

        assert regla["ok"] is False
        assert regla["value"] < ds.DECISION_COVERAGE_MIN
        assert "DATOS" in regla["text"]

    def test_los_recuentos_en_bruto_se_publican_pero_marcados_como_no_comparables(self, world):
        report = analyze(world, perturb(world["records"]))
        raw = report["decisions"]["raw_counts"]

        assert raw["live"]["n_cycles"] > 0
        assert raw["resim"]["n_cycles"] > 0
        assert "NO son comparables" in raw["note"]

    def test_publica_ejemplos_de_las_decisiones_descuadradas(self, world):
        mutilado = copy.deepcopy(perturb(world["records"]))
        for record in mutilado[:20]:
            for symbol_block in record.get("symbols", []):
                symbol_block["signals"] = []

        report = analyze(world, mutilado)
        stage = report["decisions"]["stages"]["signals"]

        assert stage["only_resim"] > 0
        assert stage["examples_only_resim"]
        assert set(stage["examples_only_resim"][0]) == {"day", "symbol", "strategy_id"}


# ------------------------------------------------------------------- latencia --------


class TestLatencia:
    """El hueco entre decidir y llenar, en tiempo y en dinero."""

    def test_mide_el_hueco_decision_fill_de_cada_orden(self, world):
        report = analyze(world, perturb(world["records"]))
        gap = report["latency"]["decision_to_fill_seconds"]

        assert gap["n"] > 0
        assert gap["median"] == pytest.approx(6.0)
        assert gap["coverage"] == 1.0

    def test_declara_la_cobertura_cuando_faltan_sellos_antiguos(self, world):
        """Las lineas archivadas antes de que el diario sellara `decided_at` no se
        rellenan: se dice cuantas son."""
        sin_sello = copy.deepcopy(perturb(world["records"]))
        for record in sin_sello[: len(sin_sello) // 2]:
            for symbol_block in record.get("symbols", []):
                for order in symbol_block["orders"]:
                    order.pop("decided_at", None)
            for exit_fill in record.get("exits", []):
                exit_fill.pop("decided_at", None)

        report = analyze(world, sin_sello)
        gap = report["latency"]["decision_to_fill_seconds"]

        assert 0.0 < gap["coverage"] < 1.0

    def test_la_antiguedad_de_la_referencia_es_la_hora_del_fill(self, world):
        """En vivo se decide con el ultimo cierre diario: a las 19:00 UTC ese cierre
        lleva 19 horas puesto, y eso es exactamente lo que un mercado real cobraria."""
        report = analyze(world, perturb(world["records"], fill_hour=19))

        assert report["latency"]["reference_age_hours"]["median"] == pytest.approx(19.0)

    def test_sin_barras_horarias_la_latencia_se_publica_en_tiempo_y_no_en_pb(self, world):
        """Degradar es declararlo, no inventar un cero: un cero ahi afirmaria que el
        mercado no se movio entre la decision y el fill."""
        report = analyze(world, perturb(world["records"]), hourly={})
        drift = report["latency"]["drift_bps"]

        assert drift["n"] == 0
        assert "1H" in drift["reason"]
        assert report["latency"]["reference_age_hours"]["n"] > 0

    def test_con_barras_horarias_tasa_el_desplazamiento_en_pb(self, world):
        """El mercado sube un 1% entre el cierre con el que se decidio y el fill de las
        19:00. Comprando, eso es 100 pb pagados de mas que el backtest no ve."""
        registros = perturb(world["records"], fill_hour=19)
        entradas = [e for e in ds.executions(registros) if e.kind == ds.KIND_ENTRY]
        levels = {e.day: e.reference_price for e in entradas}
        hourly = {SYMBOL: hourly_df(levels, drift_per_day=0.01)}

        report = analyze(world, registros, hourly=hourly)
        drift = report["latency"]["drift_bps"]

        # A las 19:00 lleva recorridas 19 de las 23 horas del tramo: 1% * 19/23 = 82,6 pb.
        assert drift["n"] > 0
        assert drift["median"] == pytest.approx(100.0 * 19.0 / 23.0, abs=0.5)
        assert drift["share_of_reference_cost"] > ds.LATENCY_MAX_COST_SHARE
        assert report["verdict"]["rules"]["latency"]["ok"] is False


# ------------------------------------------------------------------ el informe -------


class TestInforme:
    def test_es_determinista(self, world):
        registros = perturb(world["records"], reference_drift=0.002)

        uno = ds._strip_volatile(analyze(world, registros))
        otro = ds._strip_volatile(analyze(world, registros))

        assert uno == otro

    def test_es_serializable_a_json(self, world):
        report = analyze(world, perturb(world["records"]))

        recargado = json.loads(json.dumps(report))

        assert recargado["status"] == ds.STATUS_MEASURED

    def test_se_imprime_sin_romperse_en_los_dos_estados(self, world, capsys):
        """La salida por consola es la que lee quien lanza el estudio a mano. Se ejercita
        en los DOS caminos porque el de sin-potencia tiene la mitad de los campos a None
        y un formateo descuidado ahi reventaria justo el dia que se estrena."""
        ds._print_report(analyze(world, perturb(world["records"])))
        medido = capsys.readouterr().out

        ds._print_report(analyze(world, world["records"][:3]))
        sin_potencia = capsys.readouterr().out

        assert "DECISIONES" in medido.upper()
        assert "veredicto global" in medido
        assert "SIN POTENCIA" in sin_potencia
        assert "veredicto" not in sin_potencia

    def test_declara_el_techo_de_lo_que_puede_medir(self, world):
        """Mientras la ejecucion sea de papel, la pierna de coste no mide modelo contra
        mercado. Decirlo es parte del resultado."""
        report = analyze(world, perturb(world["records"]))

        assert "PAPEL" in report["ceiling"]

    def test_declara_las_asimetrias_que_no_son_de_ejecucion(self, world):
        """Son la primera explicación que hay que descartar al leer una divergencia de
        recuentos: sin declararlas, el arranque en plano parece un fallo de datos."""
        report = analyze(world, perturb(world["records"]))
        asymmetries = report["resimulation"]["asymmetries"]

        assert len(asymmetries) == 3
        assert any(ds.WINDOW_END_REASON in a for a in asymmetries)
        assert any("SIN posiciones abiertas" in a for a in asymmetries)

    def test_el_veredicto_puede_fallar(self, world):
        """Un estudio que no puede salir mal no es evidencia."""
        report = analyze(world, perturb(world["records"], extra_slippage_bps=40.0))

        assert report["verdict"]["rules"]["cost"]["ok"] is False
        assert report["verdict"]["ok"] is not True

    def test_sin_fills_suficientes_la_regla_de_coste_no_aprueba_por_defecto(self, world):
        """Sin datos no se aprueba: se marca `null`, que es distinto de aprobar."""
        pocos = ds.verdict(
            {"coverage": 1.0},
            {"n_fills": 3, "slippage_bps": {"abs_gap_median": 0.1}},
            {"drift_bps": {"share_of_reference_cost": 0.2}},
            n_fills=3,
        )

        assert pocos["rules"]["cost"]["ok"] is None
        assert pocos["ok"] is None
        assert "sin potencia" in pocos["rules"]["cost"]["text"]


# --------------------------------------------------------------- las costuras --------


class TestCosturas:
    """Las piezas que se anadieron a modulos existentes para hacer medible esto."""

    def test_el_diario_en_memoria_emite_el_mismo_esquema_que_el_de_disco(self, world):
        record = world["records"][0]

        assert {"timestamp", "status", "symbols", "opened", "closed", "exits"} <= set(record)

    def test_la_re_simulacion_deja_huella_solo_si_se_le_pide(self):
        bars = {SYMBOL: trending_df(400)}
        engine = BacktestEngine.from_bars(make_config(), bars, starting_equity=10_000.0)

        # Sin diario, el motor no construye ni una linea: es el camino caliente.
        assert engine._journal is None

    def test_la_orden_sella_cuando_se_decidio(self):
        decided = datetime(2026, 3, 1, 19, 0, tzinfo=timezone.utc)

        record = order_record("mom", "buy", 1.5, 100.0, decided_at=decided)

        assert record["decided_at"] == decided.isoformat()
        assert record["reference_price"] == 100.0

    def test_la_salida_archiva_su_referencia_y_su_instante(self):
        position = Position(
            symbol=SYMBOL,
            side=Side.BUY,
            size=1.0,
            entry_price=100.0,
            opened_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
            strategy_id="mom",
        )
        result = ExecutionResult(
            success=True,
            status=OrderStatus.FILLED,
            message="",
            filled_price=110.0,
            filled_size=1.0,
            slippage_bps=3.0,
        )
        decided = datetime(2026, 3, 5, 19, 0, tzinfo=timezone.utc)

        record = exit_record(position, result, "take_profit", reference_price=110.5, decided_at=decided)

        assert record["reference_price"] == 110.5
        assert record["decided_at"] == decided.isoformat()
        assert record["close_reason"] == "take_profit"
        assert record["strategy_id"] == "mom"

    def test_el_precio_de_llenado_extraido_es_el_que_cobra_el_motor(self):
        """La funcion que re-tasa una orden archivada y la que cobra el motor son LA
        MISMA linea. Si divergieran, el estudio mediria una diferencia de formula."""
        engine = PaperExecutionEngine()

        for side in ("buy", "sell"):
            for bps in (0.0, 3.5, 137.25):
                assert engine._apply_slippage(1234.5678, side, bps) == fill_price(
                    1234.5678, side, bps
                )

    def test_las_ejecuciones_se_leen_del_diario_con_sus_dos_instantes(self, world):
        execs = ds.executions(perturb(world["records"]))
        entries = [e for e in execs if e.kind == ds.KIND_ENTRY]

        assert entries
        assert all(e.decided_at is not None and e.executed_at is not None for e in entries)
        assert all(e.latency_seconds == pytest.approx(6.0) for e in entries)

    def test_las_ordenes_y_sus_fills_estan_alineados(self, world):
        assert ds.order_fill_mismatches(world["records"]) == 0
