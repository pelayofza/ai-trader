"""
EL DIARIO DE CICLOS: la pelicula que el estado no guarda.

Lo que se fija aqui son las tres propiedades que hacen que un archivo sirva dentro de
seis meses, no dentro de seis minutos:

1. APPEND-ONLY DE VERDAD. Reiniciar el proceso no reescribe ni recorta nada: lo de ayer
   sigue byte a byte donde estaba. Es la propiedad que convierte el fichero en evidencia
   y no en una cache.
2. ROTACION SIN PERDIDA. Al cambiar de mes o al superar el tamano, el fichero en curso
   pasa a shard y se abre uno nuevo con el MISMO nombre; leer sigue devolviendo la
   historia entera y en orden.
3. UNA LINEA ROTA NO INVALIDA EL FICHERO. Un corte de luz a mitad de escritura deja una
   cola ilegible, y perder esa linea no puede costar el mes entero.

Y una cuarta que es de contenido, no de fichero: la linea tiene que llevar el
`slippage_bps` REALMENTE cobrado y el motivo del rechazo del riesgo. Sin eso el diario no
sirve para lo unico que justifica escribirlo, que es medir la divergencia live-vs-backtest.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ai_trader.app.journal import (
    STATUS_PAUSED,
    STATUS_RAN,
    CycleJournal,
    NullJournal,
    cycle_record,
    fill_record,
    journal_summary,
    max_drawdown_usd,
    pnl_curve,
    risk_record,
    signal_record,
)
from ai_trader.shared.schemas import ExecutionResult, OrderStatus, RiskDecision

UTC = timezone.utc


def make_record(*, month: str = "2026-08", day: int = 1, net_pnl: float = 0.0, **extra) -> dict:
    """Una linea minima pero VALIDA del diario (pasa por el constructor real)."""
    year, mm = month.split("-")
    record = cycle_record(
        timestamp=datetime(int(year), int(mm), day, 12, 0, tzinfo=UTC),
        status=STATUS_RAN,
        symbols=[],
        opened=[],
        closed=[],
        exits=[],
        equity_usd=None,
        realized_pnl_usd=net_pnl,
        unrealized_pnl_usd=0.0,
        exposure_usd=0.0,
        open_positions=0,
        max_open_positions=5,
        daily_realized_pnl_usd=0.0,
    )
    record.update(extra)
    return record


@pytest.fixture
def journal(tmp_path) -> CycleJournal:
    return CycleJournal(tmp_path / "live" / "cycles.jsonl")


class TestAppendOnly:
    def test_crea_el_directorio_y_escribe_una_linea_por_ciclo(self, journal):
        journal.append(make_record(day=1))
        journal.append(make_record(day=2))

        lines = journal.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["timestamp"].startswith("2026-08-01")

    def test_un_reinicio_no_pierde_lo_ya_escrito(self, tmp_path):
        """La prueba de que es append y no reescritura: un CycleJournal nuevo sobre el
        mismo fichero -que es lo que pasa al reiniciar el proceso- continua el archivo."""
        first = CycleJournal(tmp_path / "cycles.jsonl")
        first.append(make_record(day=1))

        second = CycleJournal(tmp_path / "cycles.jsonl")
        second.append(make_record(day=2))

        assert [r["timestamp"][:10] for r in second.read()] == ["2026-08-01", "2026-08-02"]

    def test_el_contenido_sobrevive_al_viaje_de_ida_y_vuelta(self, journal):
        record = make_record(day=3, net_pnl=12.5)
        journal.append(record)

        assert journal.read() == [record]

    def test_una_linea_rota_se_salta_y_el_resto_se_lee(self, journal):
        journal.append(make_record(day=1))
        # Una escritura interrumpida: media linea, sin salto final.
        with open(journal.path, "a", encoding="utf-8") as handle:
            handle.write('{"timestamp": "2026-08-02T12:00')
        journal.append(make_record(day=3))

        recovered = journal.read()
        assert [r["timestamp"][:10] for r in recovered] == ["2026-08-01", "2026-08-03"]


class TestRotacion:
    def test_rota_al_cambiar_de_mes_y_el_fichero_en_curso_conserva_el_nombre(self, journal):
        journal.append(make_record(month="2026-08", day=30))
        journal.append(make_record(month="2026-09", day=1))

        assert (journal.path.parent / "cycles-2026-08.001.jsonl").exists()
        assert [r["timestamp"][:7] for r in json.loads(
            "[" + ",".join(journal.path.read_text(encoding="utf-8").splitlines()) + "]"
        )] == ["2026-09"]

    def test_rota_por_tamano_dentro_del_mismo_mes(self, tmp_path):
        journal = CycleJournal(tmp_path / "cycles.jsonl", max_bytes=200)
        for day in range(1, 6):
            journal.append(make_record(day=day))

        shards = sorted(p.name for p in tmp_path.glob("cycles-*.jsonl"))
        assert shards, "con 200 bytes de tope tenia que haber rotado"
        assert all(name.startswith("cycles-2026-08.") for name in shards)

    def test_la_rotacion_no_pierde_ni_un_ciclo_y_los_devuelve_en_orden(self, tmp_path):
        journal = CycleJournal(tmp_path / "cycles.jsonl", max_bytes=200)
        for day in range(1, 10):
            journal.append(make_record(day=day))
        journal.append(make_record(month="2026-09", day=1))

        days = [r["timestamp"][:10] for r in journal.read()]
        assert days == [f"2026-08-0{d}" for d in range(1, 10)] + ["2026-09-01"]

    def test_el_orden_alfabetico_de_los_shards_es_el_cronologico(self, tmp_path):
        """La secuencia va en el nombre justamente para esto: si un mes se parte en
        varios shards, `sorted()` tiene que bastar para leerlos en orden."""
        journal = CycleJournal(tmp_path / "cycles.jsonl", max_bytes=200)
        for day in range(1, 12):
            journal.append(make_record(day=day))

        names = [p.name for p in journal.shards()]
        assert names == sorted(names[:-1]) + [journal.path.name]

    def test_un_reinicio_a_mitad_de_mes_no_provoca_una_rotacion_espuria(self, tmp_path):
        """El mes del fichero en curso se relee de su primera linea; si se perdiera, el
        primer ciclo tras cada reinicio abriria shard nuevo."""
        CycleJournal(tmp_path / "cycles.jsonl").append(make_record(day=1))
        CycleJournal(tmp_path / "cycles.jsonl").append(make_record(day=2))

        assert list(tmp_path.glob("cycles-*.jsonl")) == []


class TestContenidoDeLaLinea:
    def test_el_fill_lleva_el_slippage_realmente_cobrado(self):
        result = ExecutionResult(
            success=True,
            status=OrderStatus.FILLED,
            message="filled",
            order_id="paper-1",
            filled_price=100.25,
            filled_size=2.0,
            fees=0.2,
            slippage_bps=17.5,
        )

        record = fill_record(result)

        assert record["slippage_bps"] == pytest.approx(17.5)
        assert record["fees_usd"] == pytest.approx(0.2)
        assert record["filled_price"] == pytest.approx(100.25)

    def test_el_rechazo_del_riesgo_conserva_el_motivo_con_su_cifra(self, make_signal):
        signal = make_signal(confidence=0.70)
        decision = RiskDecision(
            approved=False, size_usd=0.0, reason="Signal confidence below minimum: 0.42"
        )

        record = risk_record(signal, decision)

        assert record["approved"] is False
        assert record["reason"] == "Signal confidence below minimum: 0.42"

    def test_la_senal_lleva_su_confianza(self, make_signal):
        assert signal_record(make_signal(confidence=0.83))["confidence"] == pytest.approx(0.83)

    def test_un_ciclo_pausado_deja_linea_pero_no_se_marca_a_mercado(self):
        record = cycle_record(
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            status=STATUS_PAUSED,
            symbols=[],
            opened=[],
            closed=[],
            exits=[],
            equity_usd=None,
            realized_pnl_usd=10.0,
            unrealized_pnl_usd=None,
            exposure_usd=0.0,
            open_positions=1,
            max_open_positions=5,
            daily_realized_pnl_usd=0.0,
        )

        assert record["marked_to_market"] is False
        assert record["net_pnl_usd"] is None, "un cero afirmaria que nada se ha movido"


class TestDerivadas:
    def test_la_curva_ignora_los_ciclos_sin_marcar(self):
        marcado = make_record(day=1, net_pnl=5.0)
        pausado = {**make_record(day=2), "net_pnl_usd": None, "status": STATUS_PAUSED}

        assert len(pnl_curve([marcado, pausado])) == 1

    def test_el_drawdown_en_usd_mide_desde_el_pico(self):
        curve = [{"net_pnl_usd": v} for v in (0.0, 40.0, 10.0, 25.0, -5.0)]

        assert max_drawdown_usd(curve) == pytest.approx(45.0)

    def test_el_resumen_de_un_diario_vacio_no_revienta(self):
        summary = journal_summary([])

        assert summary["n_cycles"] == 0
        assert summary["curve"] == []
        assert summary["max_drawdown_pct"] is None

    def test_el_resumen_agrupa_los_rechazos_por_familia(self):
        record = make_record(day=1)
        record["symbols"] = [
            {
                "symbol": "BTC/USDT",
                "signals": [{"confidence": 0.9}],
                "risk": [
                    {"approved": False, "reason": "Signal confidence below minimum: 0.42"},
                    {"approved": False, "reason": "Signal confidence below minimum: 0.51"},
                    {"approved": True, "reason": "Signal approved by risk engine"},
                ],
                "fills": [{"success": True, "fees_usd": 0.5, "slippage_bps": 12.0}],
            }
        ]

        summary = journal_summary([record])

        assert summary["rejections"] == {"Signal confidence below minimum": 2}
        assert summary["n_approved"] == 1
        assert summary["n_fills"] == 1
        assert summary["fees_usd"] == pytest.approx(0.5)
        assert summary["slippage_bps_mean"] == pytest.approx(12.0)

    def test_las_comisiones_de_salida_no_se_cuentan_dos_veces(self):
        """El fill de entrada ya trae su comision; del cierre solo se suma la de salida."""
        record = make_record(day=1)
        record["symbols"] = [
            {"symbol": "BTC/USDT", "fills": [{"success": True, "fees_usd": 1.0}]}
        ]
        record["closed"] = [{"entry_fees_usd": 1.0, "exit_fees_usd": 0.75, "fees_usd": 1.75}]

        assert journal_summary([record])["fees_usd"] == pytest.approx(1.75)

    def test_el_deslizamiento_medio_incluye_las_dos_patas(self):
        """Salir paga deslizamiento igual que entrar: una media que solo mire las
        entradas subestima el coste de ejecucion a la mitad."""
        record = make_record(day=1)
        record["symbols"] = [
            {"symbol": "BTC/USDT", "fills": [{"success": True, "slippage_bps": 10.0}]}
        ]
        record["exits"] = [{"success": True, "slippage_bps": 20.0, "symbol": "BTC/USDT"}]

        summary = journal_summary([record])

        assert summary["n_fills"] == 1
        assert summary["n_exit_fills"] == 1
        assert summary["slippage_bps_mean"] == pytest.approx(15.0)


class TestVistaDelDashboard:
    """
    El colector de la vista de paper trading (`dashboard/build_dashboard.py`).

    Con datos y sin ellos: la vista tiene que decir "sin ciclos registrados" en vez de
    romperse, y con datos tiene que leer cada cosa de su fuente -las abiertas del ultimo
    ciclo del diario (que es lo unico que trae precio de marca) y las cerradas del estado
    persistido (que es el registro autoritativo)-. Y sin tocar la red: si generar el
    dashboard necesitara precios en vivo, dejaria de ser reproducible.
    """

    @staticmethod
    def _tree(tmp_path):
        import shutil

        from golden_support import REPO_ROOT

        (tmp_path / "config").mkdir()
        shutil.copy(REPO_ROOT / "config" / "default.toml", tmp_path / "config" / "default.toml")
        (tmp_path / "data").mkdir()
        return tmp_path

    def _collect(self, tmp_path, monkeypatch):
        import dashboard.build_dashboard as bd

        monkeypatch.setattr(bd, "ROOT", tmp_path)
        return bd.collect_paper()

    def test_sin_nada_en_disco_devuelve_cero_ciclos_y_no_revienta(self, tmp_path, monkeypatch):
        paper = self._collect(self._tree(tmp_path), monkeypatch)

        assert paper["summary"]["n_cycles"] == 0
        assert paper["curve"] == []
        assert paper["closed_positions"] == []
        # Los limites se leen igual: la vista es util antes de que haya un solo ciclo.
        assert paper["limits"]["max_open_positions"] > 0

    def test_con_diario_lee_curva_abiertas_y_cerradas_de_su_fuente(
        self, tmp_path, monkeypatch, make_position
    ):
        from ai_trader.app.state_store import JsonStateStore
        from ai_trader.shared.schemas import PositionStatus

        root = self._tree(tmp_path)

        journal = CycleJournal(root / "data" / "live" / "cycles.jsonl")
        journal.append(make_record(day=1, net_pnl=0.0))
        last = make_record(day=2, net_pnl=25.0)
        last["opened"] = [
            {"symbol": "BTC/USDT", "side": "buy", "strategy_id": "mom", "size": 0.01,
             "entry_price": 100.0, "mark_price": 110.0, "unrealized_pnl_usd": 0.1,
             "opened_at": "2026-08-02T12:00:00+00:00", "stop_loss": 95.0,
             "take_profit": 120.0, "fees_usd": 0.001}
        ]
        last["open_positions"] = 1
        journal.append(last)

        closed = make_position(symbol="ETH/USDT")
        closed.status = PositionStatus.CLOSED
        closed.exit_price = 120.0
        closed.realized_pnl = 18.5
        closed.close_reason = "take_profit"
        JsonStateStore(root / "data" / "runtime_state.json").save({"positions": [closed]})

        paper = self._collect(root, monkeypatch)

        assert paper["summary"]["n_cycles"] == 2
        assert [p["net"] for p in paper["curve"]] == [0.0, 25.0]
        # Abiertas: del ultimo ciclo, porque el estado no guarda precio de marca.
        assert paper["open_positions"][0]["symbol"] == "BTC/USDT"
        # Cerradas: del estado, que es el registro autoritativo aunque falte el diario.
        assert paper["closed_positions"][0]["symbol"] == "ETH/USDT"
        assert paper["closed_positions"][0]["realized_pnl_usd"] == pytest.approx(18.5)

    def test_la_curva_se_recorta_para_el_grafico_pero_las_cifras_no(
        self, tmp_path, monkeypatch
    ):
        """Con un ciclo cada 15 minutos son ~35.000 puntos al ano. El grafico se recorta;
        el drawdown y el PnL siguen saliendo de la curva entera."""
        root = self._tree(tmp_path)
        journal = CycleJournal(root / "data" / "live" / "cycles.jsonl", max_bytes=10**9)
        for i in range(900):
            journal.append(make_record(day=1 + i % 28, net_pnl=float(i)))

        paper = self._collect(root, monkeypatch)

        assert paper["summary"]["n_cycles"] == 900
        assert len(paper["curve"]) == 400
        assert paper["curve"][0]["net"] == pytest.approx(0.0)
        assert paper["curve"][-1]["net"] == pytest.approx(899.0), "el ultimo punto no se pierde"
        # La caida maxima se mide sobre los 900, no sobre los 400 dibujados.
        assert paper["summary"]["max_drawdown_usd"] == pytest.approx(0.0)


class TestNullJournal:
    def test_no_escribe_nada_y_no_estorba(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        journal = NullJournal()

        assert journal.append(make_record()) is None
        assert journal.read() == []
        assert not (tmp_path / "data").exists()
