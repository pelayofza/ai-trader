"""
Caracterizacion de los dos generadores de artefactos: `dashboard/` y `docs/`.

Ninguno de los dos estaba bajo test —ni siquiera entraban en la medicion de cobertura,
porque viven fuera del paquete `ai_trader`— y entre los dos suman ~2.000 lineas que leen
casi todos los informes publicados del repo. Un refactor que cambie una constante, un
formato o el orden de una lista se ve aqui y en ningun otro sitio.

La referencia NO se duplica en `tests/golden/`: el artefacto commiteado (`dashboard/
index.html`, `docs/metodologia.html`) ES la referencia. Son 750 KB que ya estan
versionados y que el flujo normal regenera; copiarlos seria tener dos verdades.

Los dos son caros (el dashboard corre el ranking de muestra entero), asi que van marcados
`veryslow`: no entran en la vuelta rapida, si en la verificacion completa. Y el coste hay que
vigilarlo: al pasar de dos familias a ocho, el ranking de muestra y el barrido de la demo se
multiplicaron por cuatro y el build se fue a ~25 min, es decir, la verificacion completa de
~12 a ~40. Se recorto bajando `RANK_N_PATHS` a 1 y el barrido de la demo a dos escenarios, que
es lo correcto porque las dos vistas son ILUSTRATIVAS —la evidencia vive en `data/`— y esto se
paga en cada vuelta.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from golden_support import REPO_ROOT, scrub

pytestmark = pytest.mark.veryslow


def _assert_matches_committed(generated: Path, committed: Path) -> None:
    from golden_support import _diff_message

    actual = scrub(generated.read_text(encoding="utf-8"))
    expected = scrub(committed.read_text(encoding="utf-8"))
    if expected != actual:
        rel = committed.relative_to(REPO_ROOT).as_posix()
        pytest.fail(
            _diff_message(rel, expected, actual)
            + f"\n\nEl artefacto regenerado ya no coincide con {rel}. "
            f"Si el cambio es deliberado, regeneralo y commitealo."
        )


def test_dashboard_reproduces_committed_html(tmp_path, monkeypatch) -> None:
    import dashboard.build_dashboard as bd

    out = tmp_path / "index.html"
    monkeypatch.setattr(bd, "OUT_HTML", out)
    bd.build()

    assert out.exists(), "build() no escribio el dashboard"
    _assert_matches_committed(out, REPO_ROOT / "dashboard" / "index.html")


def test_docs_reproduces_committed_html(tmp_path, monkeypatch) -> None:
    import docs.build_docs as bdo

    out = tmp_path / "metodologia.html"
    monkeypatch.setattr(bdo, "OUT", out)
    bdo.build()

    assert out.exists(), "build() no escribio la metodologia"
    _assert_matches_committed(out, REPO_ROOT / "docs" / "metodologia.html")


def test_builders_do_not_write_into_the_repo(tmp_path, monkeypatch) -> None:
    """Que generar los artefactos no ensucie el arbol por un camino lateral.

    Importa para la verificacion misma: si `build()` tocase `data/` de paso, cada pasada
    dejaria el repo modificado y no se podria distinguir un cambio del refactor de un
    efecto secundario de haberlo comprobado."""
    import subprocess

    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout

    import docs.build_docs as bdo

    monkeypatch.setattr(bdo, "OUT", tmp_path / "m.html")
    bdo.build()

    after = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout
    assert before == after, f"build() modifico el arbol:\nantes:\n{before}\ndespues:\n{after}"
