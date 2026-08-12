"""
Soporte para los tests de CARACTERIZACION.

Un test de caracterizacion no dice lo que el codigo DEBERIA hacer: congela lo que hace
HOY, byte a byte, para que un refactor que cambie el comportamiento no pueda pasar en
silencio. Los ficheros de referencia viven en `tests/golden/`.

Regenerar la referencia (solo cuando el cambio de salida es DELIBERADO y revisado):

    $env:AI_TRADER_REGOLD="1"; .venv\\Scripts\\python.exe -m pytest tests/test_characterization_cli.py

Y despues `git diff tests/golden/` para ver exactamente que cambio. Si ese diff no se
puede explicar linea por linea, el refactor no era equivalente.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
REGOLD = os.getenv("AI_TRADER_REGOLD") == "1"


@lru_cache(maxsize=None)
def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - entorno sin git
        return ""


_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:\+00:00|Z)?")

# Un hash corto de git lleva letras hex: sustituirlo por literal no puede colisionar con
# una cifra del informe. El CONTADOR de commits, en cambio, es un entero de dos digitos
# ("81") y sustituirlo por literal corrompia las series de precios (`81.32` ->
# `<NCOMMITS>.32`), que es justo lo contrario de lo que debe hacer un scrubber: enmascarar
# datos reales. Por eso todo lo que no sea el hash va anclado a su clave o a su etiqueta.
_KEYED = (
    (re.compile(r'("commit_count":\s*")\d+(")'), r"\1<NCOMMITS>\2"),
    (re.compile(r'("commit":\s*")[0-9a-f]{7,40}(")'), r"\1<COMMIT>\2"),
    (re.compile(r'("(?:generated_at|date)":\s*")\d{4}-\d{2}-\d{2}(")'), r"\1<GITDATE>\2"),
    # El HTML de docs lleva la metadata en prosa: "(81 commits)2026-08-11". Va por patron
    # y no por el valor de HEAD porque el artefacto COMMITEADO —que es la referencia— se
    # genero en un commit anterior y lleva el hash y la fecha de aquel.
    (re.compile(r"\(\d+ commits\)\d{4}-\d{2}-\d{2}"), "(<NCOMMITS> commits)<GITDATE>"),
    (re.compile(r"\(\d+ commits\)"), "(<NCOMMITS> commits)"),
    (re.compile(r'(<span class="mono">)[0-9a-f]{7,40}(</span>)'), r"\1<COMMIT>\2"),
    # `_count_tests()` cuenta los tests del repo: anadir un test cambiaria el artefacto
    # sin que el comportamiento cambie.
    (re.compile(r"<b>\d+ tests</b>"), "<b><NTESTS> tests</b>"),
    # El bloque `paper` del dashboard es ESTADO LOCAL DE LA MAQUINA, no evidencia
    # publicada: sale de `data/live/cycles.jsonl` y de `data/runtime_state.json`, que
    # estan fuera de git a proposito (crecen cada 15 minutos en la maquina que opera).
    # A diferencia del resto de vistas —que leen informes commiteados y por eso son
    # reproducibles en cualquier clon— este bloque cambia solo con que el bot haya
    # corrido un ciclo mas. Enmascararlo es lo que mantiene la caracterizacion util para
    # las otras ~2.000 lineas del generador; lo que el bloque contenga se prueba en
    # tests/test_journal.py, contra el diario y no contra el HTML.
    (re.compile(r'"paper": \{.*?\}, "roadmap":', re.S), '"paper": <LIVE>, "roadmap":'),
)


def scrub(text: str) -> str:
    """Neutraliza lo que varia sin que el comportamiento cambie.

    Solo cinco cosas: la marca de tiempo, la metadata del commit, el recuento de tests,
    la ruta absoluta del repo (distinta en cada maquina) y el bloque de paper trading en
    vivo del dashboard (estado local, no evidencia commiteada). Cualquier otra diferencia
    se considera un cambio de comportamiento y debe romper el test."""
    text = text.replace("\r\n", "\n")

    root = str(REPO_ROOT)
    for variant in (root, root.replace("\\", "/"), root.replace("\\", "\\\\")):
        text = text.replace(variant, "<ROOT>")

    for pattern, repl in _KEYED:
        text = pattern.sub(repl, text)

    # Solo si el hash lleva alguna letra. Uno de cada veintisiete hashes cortos sale con
    # los siete caracteres numericos, y entonces este replace literal seria el mismo error
    # que el del contador de commits: se comeria cualquier cifra que lo contuviera. Cuando
    # eso pasa, los patrones anclados de arriba ya lo han cubierto.
    short_sha = _git("rev-parse", "--short", "HEAD")
    if short_sha and not short_sha.isdigit():
        text = text.replace(short_sha, "<COMMIT>")

    gitdate = _git("log", "-1", "--format=%cd", "--date=short")
    if gitdate:
        # Anclada a la etiqueta que la envuelve en el HTML de docs, no suelta: una fecha
        # corta suelta puede aparecer dentro de una serie temporal legitima.
        text = text.replace(f"commits){gitdate}", "commits)<GITDATE>")

    return _TIMESTAMP.sub("<TS>", text)


def assert_golden(name: str, actual: str) -> None:
    """Compara contra `tests/golden/<name>`, o la crea si aun no existe."""
    path = GOLDEN_DIR / name
    actual = scrub(actual)

    if REGOLD or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(actual)
        if not REGOLD:
            pytest.fail(
                f"No habia referencia para {name}: se ha creado con la salida actual. "
                f"Revisala ('git diff tests/golden/') y vuelve a ejecutar."
            )
        return

    expected = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if expected == actual:
        return

    pytest.fail(_diff_message(name, expected, actual))


def _first_divergence(expected: str, actual: str, width: int = 110) -> str:
    """El primer caracter en el que dejan de coincidir, con su contexto.

    El dashboard mete 600 KB de JSON en UNA linea: un diff por lineas ahi no dice nada,
    y volcarla entera tampoco. Lo util es el punto exacto donde se separan."""
    n = min(len(expected), len(actual))
    i = next((k for k in range(n) if expected[k] != actual[k]), n)
    lo = max(0, i - width)
    return (
        f"  primer cambio en el caracter {i}:\n"
        f"    referencia: ...{expected[lo:i + width]!r}\n"
        f"    actual:     ...{actual[lo:i + width]!r}"
    )


def _diff_message(name: str, expected: str, actual: str, context: int = 3) -> str:
    import difflib

    exp_lines, act_lines = expected.splitlines(), actual.splitlines()
    diff = [
        line if len(line) <= 220 else f"{line[:200]}... (+{len(line) - 200} chars)"
        for line in difflib.unified_diff(
            exp_lines, act_lines, fromfile=f"golden/{name}", tofile="actual",
            n=context, lineterm="",
        )
    ]
    head = "\n".join(diff[:30])
    extra = f"\n... (+{len(diff) - 30} lineas de diff)" if len(diff) > 30 else ""
    return (
        f"CARACTERIZACION ROTA: {name}\n"
        f"  referencia {len(exp_lines)} lineas / {len(expected)} chars\n"
        f"  actual     {len(act_lines)} lineas / {len(actual)} chars\n"
        f"{_first_divergence(expected, actual)}\n"
        f"El comportamiento ha cambiado. Si el cambio es DELIBERADO, regenera con "
        f"AI_TRADER_REGOLD=1 y revisa el diff.\n\n{head}{extra}"
    )


def run_cli(*argv: str, expect_code: int = 0) -> str:
    """Ejecuta el CLI EN PROCESO y devuelve su stdout.

    En proceso y no por subproceso por dos razones: cuenta para la cobertura de
    `cli.py` (que hoy es 0%) y evita pagar el arranque del interprete en cada caso."""
    from ai_trader.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(list(argv))

    assert code == expect_code, (
        f"`{' '.join(argv)}` devolvio {code}, se esperaba {expect_code}.\n"
        f"stdout:\n{buf.getvalue()[:2000]}"
    )
    return buf.getvalue()
