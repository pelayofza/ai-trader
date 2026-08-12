"""
Caracterizacion del CLI: `src/ai_trader/cli.py`, hoy al 0% de cobertura.

Es el punto de entrada por el que se opera la herramienta y no habia ni un test que lo
tocase: 365 sentencias sin red. Estos casos NO afirman que la salida sea correcta —
afirman que es LA MISMA de hoy. Esa es toda su utilidad de cara al refactor.

Ninguno toca la red. Los que escriben en `data/` van redirigidos a tmp_path, para que
ejecutar la verificacion no ensucie el arbol.
"""
from __future__ import annotations

import json

import pytest

from golden_support import REPO_ROOT, assert_golden, run_cli

# Muestra sintetica fija: misma libreria, escenario y camino en cada ejecucion. Se ha
# comprobado que dos pasadas dan un JSON identico byte a byte.
SAMPLE = "ai_v1:hawkish_fed_shock:0"

# `data/synthetic/` esta en el .gitignore. A diferencia del archivo de senales, que crece
# con cada captura, la libreria es un artefacto ESTATICO: una vez generada no cambia, y
# por eso si se puede congelar su backtest. Pero en una maquina limpia no existe, y ahi lo
# honesto es saltar con instrucciones en vez de fallar como si fuese una regresion.
_LIBRARY = REPO_ROOT / "data" / "synthetic" / "ai_v1"
needs_library = pytest.mark.skipif(
    not _LIBRARY.exists(),
    reason=f"falta {_LIBRARY}; generala con: ai-trader synth generate --library ai_v1",
)


# --------------------------------------------------------------------------- catalogo


def test_signals_catalog_json() -> None:
    assert_golden("cli_signals_catalog.json", run_cli("signals", "catalog", "--json"))


def test_signals_catalog_text() -> None:
    assert_golden("cli_signals_catalog.txt", run_cli("signals", "catalog"))


def test_synth_list() -> None:
    assert_golden("cli_synth_list.txt", run_cli("synth", "list"))


# ------------------------------------------------------------------------ backtest e2e
# El pipeline end-to-end completo: config -> barras sinteticas -> estrategias -> riesgo
# -> ejecucion -> metricas -> score. Es el camino que mas codigo atraviesa.


@pytest.mark.slow
@needs_library
def test_backtest_single_split_json() -> None:
    assert_golden(
        "cli_backtest_single_split.json",
        run_cli("backtest", "--synthetic", SAMPLE, "--json"),
    )


@pytest.mark.slow
@needs_library
def test_backtest_single_split_text() -> None:
    """El formato impreso, no solo las cifras: es lo que se lee por terminal."""
    assert_golden(
        "cli_backtest_single_split.txt",
        run_cli("backtest", "--synthetic", SAMPLE),
    )


@pytest.mark.slow
@needs_library
def test_backtest_walk_forward_json() -> None:
    assert_golden(
        "cli_backtest_walk_forward.json",
        run_cli("backtest", "--synthetic", SAMPLE, "--validation", "walk_forward", "--json"),
    )


@pytest.mark.slow
@needs_library
def test_backtest_cpcv_json() -> None:
    assert_golden(
        "cli_backtest_cpcv.json",
        run_cli("backtest", "--synthetic", SAMPLE, "--validation", "cpcv", "--json"),
    )


@pytest.mark.slow
@needs_library
def test_backtest_multiwindow_text() -> None:
    """La tabla multiventana con actividad y gate: mucha logica de formato en una salida."""
    assert_golden(
        "cli_backtest_walk_forward.txt",
        run_cli("backtest", "--synthetic", SAMPLE, "--validation", "walk_forward"),
    )


# ------------------------------------------------------------------- caminos de error
# Baratos y valiosos: fijan los codigos de salida y los mensajes que hoy se emiten.


def test_backtest_without_dates_returns_2() -> None:
    out = run_cli("backtest", expect_code=2)
    assert_golden("cli_backtest_no_dates.txt", out)


def test_backtest_malformed_synthetic_spec() -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli("backtest", "--synthetic", "solo_dos:partes")
    assert "LIBRARY:SCENARIO:PATH" in str(excinfo.value)


def test_unknown_command_exits_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli("no-existe")
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- señales
#
# Estos cuatro leen `data/signals_raw/`, que es APPEND-ONLY POR DISENO (store.py:22) y
# esta en el .gitignore: cada `signals capture` le anade lineas. Congelar su salida
# entera daria un golden que se rompe solo, y un test que falla por razones ajenas al
# refactor no es una red -- ensena a ignorar el rojo.
#
# Asi que se parte en dos: lo que sale del CODIGO y del catalogo (deterministico) va a
# golden; lo que sale del ARCHIVO (recuentos, primer y ultimo dia) se comprueba como
# invariante. La cobertura del CLI es la misma y los falsos positivos desaparecen.


def _project(rows, keys):
    """Proyeccion estable de una lista de dicts: solo los campos que fija el catalogo."""
    return json.dumps(
        [{k: row.get(k) for k in keys} for row in rows], indent=2, ensure_ascii=False
    )


@pytest.mark.slow
def test_signals_audit_entity_mapping() -> None:
    """El mapeo simbolo -> entidad sale del config y de las reglas, no del archivo."""
    report = json.loads(run_cli("signals", "audit", "--json"))
    entities = report["entities"]

    assert_golden("cli_signals_audit_entities.json", json.dumps(entities, indent=2,
                                                                ensure_ascii=False))
    assert_golden(
        "cli_signals_audit_sources.json",
        _project(report["archive"]["sources"],
                 ("source_key", "tier", "cadence", "pit", "history_from", "backtestable")),
    )

    archive = report["archive"]
    assert archive["n_sources"] == 17, "el catalogo declara 17 fuentes"
    assert 0 <= archive["n_with_archive"] <= archive["n_sources"]
    assert archive["records"] >= 0


@pytest.mark.slow
def test_signals_depth_from_archive(monkeypatch, tmp_path) -> None:
    """`depth` publica un registro en data/; aqui se desvia a tmp para no ensuciar."""
    import ai_trader.signals.depth as depth

    written: list = []

    def _fake_write_ledger(report, path=None, *, merge=True):
        target = tmp_path / "history_depth.json"
        target.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        written.append(target)
        return target

    monkeypatch.setattr(depth, "write_ledger", _fake_write_ledger)
    report = json.loads(run_cli("signals", "depth", "--from-archive", "--json"))

    assert written, "measure_depth deberia haber publicado el registro"
    assert_golden(
        "cli_signals_depth_sources.json",
        _project(report["sources"], ("source_key", "method", "connected", "error")),
    )

    assert report["n_sources"] == 17
    for row in report["sources"]:
        assert row["days"] >= 0
        # Un tramo medido tiene los dos extremos o ninguno; nunca medio.
        assert (row["first_day"] is None) == (row["last_day"] is None)


@pytest.mark.slow
def test_signals_events_pool(monkeypatch, tmp_path) -> None:
    """`events` publica el recuento en data/; se desvia igual que depth."""
    import ai_trader.signals.events as events

    def _fake_write_pool_report(report, path=None):
        target = tmp_path / "event_pool.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return target

    monkeypatch.setattr(events, "write_pool_report", _fake_write_pool_report)
    report = json.loads(run_cli("signals", "events", "--json"))

    # `spec` son las constantes de normalizacion del pool: tope de dias al evento,
    # ventana posterior, recorte de magnitud. Puro codigo.
    assert_golden("cli_signals_events_spec.json", json.dumps(report["spec"], indent=2,
                                                             ensure_ascii=False))
    assert_golden(
        "cli_signals_events_sources.json",
        _project(
            [{"source_key": k, **v} for k, v in sorted(report["sources"].items())],
            ("source_key", "tier", "cadence", "announced", "magnitude"),
        ),
    )

    assert report["n_event_sources"] == len(report["sources"])
    assert report["pooled_events_total"] == sum(
        row["pooled_events"] for row in report["sources"].values()
    ), "el total pooled debe ser la suma por fuente"


@pytest.mark.veryslow
def test_signals_features_normalization_spec() -> None:
    """Deriva el archivo crudo entero: ~90 s. Fuera de la vuelta rapida a proposito."""
    report = json.loads(run_cli("signals", "features", "--json"))

    assert_golden("cli_signals_features_spec.json", json.dumps(report["spec"], indent=2,
                                                               ensure_ascii=False))
    coverage = report["coverage"]
    assert 0.0 <= coverage["self_pct"] <= 100.0
    assert 0.0 <= coverage["cross_pct"] <= 100.0
    assert coverage["rows"] >= 0 and coverage["columns"] >= 0
