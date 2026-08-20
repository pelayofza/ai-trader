"""
LA REGLA DE `ai_trader/research/`, comprobada en vez de prometida.

La linea sintetica esta aparcada (ver `ai_trader/research/__init__.py`). Aparcar algo solo
significa algo si se puede comprobar que lo demas no depende de ello: un docstring que diga
"esto ya no se usa" envejece en la primera semana, y el dia que alguien vuelva a importar de
ahi no se enterara nadie.

Estos tests son esa comprobacion. Si fallan, o el aislamiento se rompio o se decidio
romperlo -- y en el segundo caso hay que actualizar aqui, a la vista, no en silencio.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "ai_trader"

# Los paquetes del sistema que opera y del que puntua. `cli.py` no entra: tiene el
# subcomando `synth`, que tambien esta archivado y que importa de forma perezosa.
LIVE_PACKAGES = (
    "app", "backtest", "bots", "data", "execution", "notifications",
    "observation", "risk", "scoring", "shared", "signals", "strategies",
)


def _python_files(package: str) -> list[Path]:
    return [p for p in (SRC / package).rglob("*.py") if "__pycache__" not in p.parts]


class TestNothingLiveImportsResearch:
    @pytest.mark.parametrize("package", LIVE_PACKAGES)
    def test_no_live_package_imports_research(self, package: str) -> None:
        offenders = [
            f"{p.relative_to(SRC).as_posix()}:{n}"
            for p in _python_files(package)
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if line.lstrip().startswith(("from ai_trader.research", "import ai_trader.research"))
        ]
        assert not offenders, (
            "La linea sintetica esta aparcada y el codigo vivo no debe importarla: "
            + ", ".join(offenders)
        )

    def test_the_cli_is_the_only_module_outside_research_that_imports_it(self) -> None:
        """Y solo para `synth`, de forma perezosa. Es la unica excepcion declarada."""
        importers = {
            p.relative_to(SRC).as_posix()
            for package in (*LIVE_PACKAGES, ".")
            for p in (
                _python_files(package) if package != "." else sorted(SRC.glob("*.py"))
            )
            if "from ai_trader.research" in p.read_text(encoding="utf-8")
        }
        assert importers == {"cli.py"}


class TestTheSystemRunsWithoutResearch:
    """Lo anterior mira el texto; esto lo ejecuta: con `ai_trader.research` bloqueado,
    todo lo que opera y puntua tiene que seguir importando."""

    @pytest.fixture
    def research_blocked(self, monkeypatch):
        class _Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name == "ai_trader.research" or name.startswith("ai_trader.research."):
                    raise ImportError(f"bloqueado a proposito: {name}")
                return None

        for name in list(sys.modules):
            if name.startswith("ai_trader"):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
        yield
        for name in list(sys.modules):
            if name.startswith("ai_trader"):
                del sys.modules[name]

    @pytest.mark.parametrize(
        "module",
        [
            "ai_trader.main",
            "ai_trader.app.runner",
            "ai_trader.backtest.engine",
            "ai_trader.scoring.optimize",
            "ai_trader.scoring.real_source",
            "ai_trader.scoring.theme_study",
            "ai_trader.strategies.registry",
        ],
    )
    def test_module_imports_with_research_blocked(self, research_blocked, module) -> None:
        assert importlib.import_module(module) is not None

    def test_the_cli_still_starts_with_research_blocked(self, research_blocked) -> None:
        """El CLI que opera no puede dejar de arrancar porque falte la carpeta archivada.
        `--help` construye TODOS los subparsers, incluido `synth derive`, que es el unico
        que necesitaba algo de ahi: sin la carpeta se queda sin valores para --enricher, y
        eso es todo lo que pasa."""
        from ai_trader.cli import main

        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])

        assert exit_info.value.code == 0
