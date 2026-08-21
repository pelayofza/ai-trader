<#
.SYNOPSIS
    Verificacion unica de ai-trader. Si esto pasa en verde, el comportamiento observable
    no ha cambiado.

.DESCRIPTION
    Cuatro cosas, en orden de coste creciente:

      1. ruff        - linting sobre src, tests, dashboard y docs
      2. contrato    - lo que ruff NO mira: los ficheros de la tarea diaria externa
      3. pytest      - la suite entera, incluidos los tests de CARACTERIZACION que
                       comparan la salida real contra tests/golden/
      4. arbol limpio- que verificar no haya dejado ficheros modificados por el camino

    El paso 4 no es cosmetico: varios comandos publican informes en data/, y si la propia
    verificacion ensucia el arbol deja de poder distinguirse un cambio del refactor de un
    efecto secundario de haberlo comprobado.

    EL PASO 2 Y POR QUE NO LO CUBRE NINGUNO DE LOS OTROS
    ---------------------------------------------------
    `config/*.json` y `tools/*.py` alimentan la captura diaria por activo que corre FUERA
    de este repo (Claude Cowork, 08:00 Europe/Madrid). Los dos quedan fuera de las redes
    normales, y cada uno por un motivo distinto:

      * `tools/` esta excluido de ruff a proposito (ver pyproject.toml): no sigue el estilo
        del repo y no tiene por que. Pero excluirlo del linter lo excluyo tambien de
        cualquier comprobacion, y `validar_respuestas_v2.py` se escribio en una sesion SIN
        Python -- es decir, sin haberse ejecutado nunca. `py_compile` es el minimo: no dice
        que este bien, dice que al menos parsea.

      * `config/` son datos, no codigo, asi que ruff no los abre. Un JSON mal cerrado o un
        `id_opcion` repetido no rompe nada aqui y rompe la ejecucion de manana en un
        sandbox donde nadie esta mirando -- y el dia perdido no se recupera, porque la
        captura es point-in-time.

    La comprobacion de fondo vive en `tests/test_ai_reports_contract.py` y por tanto
    tambien corre en el paso 3. Repetirla aqui cuesta menos de un segundo y es lo mismo que
    hace ruff: fallar pronto en vez de a los veinte minutos.

.PARAMETER Fast
    Salta los casos `veryslow` (el dashboard tarda ~4,5 min el solo). Para la vuelta
    corta mientras se itera. El gate de verdad es la pasada completa.

.PARAMETER Regold
    Regenera los ficheros de referencia en vez de compararlos. Solo cuando el cambio de
    salida es DELIBERADO: despues hay que revisar `git diff tests/golden/` linea a linea.

.EXAMPLE
    .\scripts\verify.ps1
    .\scripts\verify.ps1 -Fast
#>
[CmdletBinding()]
param(
    [switch]$Fast,
    [switch]$Regold
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No existe $python" -ForegroundColor Red
    Write-Host "Crea el entorno con: poetry install" -ForegroundColor Yellow
    exit 1
}

$started = Get-Date
$failures = @()

function Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        $script:failures += $Name
        Write-Host "FALLO: $Name" -ForegroundColor Red
    } else {
        Write-Host "OK: $Name" -ForegroundColor Green
    }
}

# --- estado del arbol ANTES, para el paso 4 -------------------------------------------
$treeBefore = (& git status --porcelain) -join "`n"

# --- 1. linting -----------------------------------------------------------------------
Step "ruff" { & $python -m ruff check . }

# --- 2. el contrato de la captura diaria externa --------------------------------------
# No toca red ni lee la captura del dia: solo `config/` y `tools/`, que SI estan
# versionados. Que no haya ejecuciones en `data/signals_raw/` no es un fallo -- esa carpeta
# esta en el .gitignore y un clon recien hecho no tiene ninguna.
Step "contrato" {
    & $python -c @'
import sys
from pathlib import Path

root = Path.cwd()
fallos = []

# 2a. Los scripts del agente parsean. Excluidos de ruff a proposito; sin esto no los mira
#     nadie, y el validador v2 se escribio sin poder ejecutarse ni una vez.
#     Se usa `compile` de serie y no `py_compile`: el segundo ESCRIBE el .pyc, y aunque
#     __pycache__ este en el .gitignore, un paso de verificacion que deja ficheros por el
#     camino es justo lo que el paso 4 existe para detectar.
scripts = sorted((root / "tools").glob("*.py"))
for script in scripts:
    try:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
    except SyntaxError as exc:
        fallos.append("%s no compila: linea %s, %s" % (script.name, exc.lineno, exc.msg))

# 2b. El contrato del cuestionario y del universo es coherente.
sys.path.insert(0, str(root / "src"))
from ai_trader.signals.ai_reports import contract_problems, load_contract

fallos.extend(contract_problems(root))

if fallos:
    print("\n".join("  - " + f for f in fallos))
    raise SystemExit(1)

c = load_contract(root)
print("OK: %d preguntas (%d suman) sobre %d activos; %d scripts del agente compilan"
      % (c["n_questions"], c["n_sumable"], c["universe"]["n_assets"], len(scripts)))
'@
}

# --- 3. la suite ----------------------------------------------------------------------
if ($Regold) { $env:AI_TRADER_REGOLD = "1" } else { Remove-Item Env:\AI_TRADER_REGOLD -ErrorAction SilentlyContinue }

$pytestArgs = @("-m", "pytest", "-q", "--tb=short")
if ($Fast) {
    $pytestArgs += @("-m", "not veryslow")
    Write-Host "(modo -Fast: se saltan los casos veryslow)" -ForegroundColor DarkGray
}
Step "pytest" { & $python @pytestArgs }

if ($Regold) { Remove-Item Env:\AI_TRADER_REGOLD -ErrorAction SilentlyContinue }

# --- 4. la verificacion no debe ensuciar el arbol -------------------------------------
Write-Host ""
Write-Host "=== arbol limpio ===" -ForegroundColor Cyan
$treeAfter = (& git status --porcelain) -join "`n"
if ($treeBefore -ne $treeAfter) {
    # Con -Regold es lo esperado: los golden se han reescrito a proposito.
    if ($Regold) {
        Write-Host "OK: el arbol cambio porque -Regold reescribio las referencias" -ForegroundColor Green
        Write-Host "    revisa 'git diff tests/golden/' antes de commitear" -ForegroundColor Yellow
    } else {
        $failures += "arbol limpio"
        Write-Host "FALLO: verificar dejo el arbol modificado" -ForegroundColor Red
        Write-Host "antes:"  -ForegroundColor DarkGray; Write-Host $treeBefore
        Write-Host "despues:" -ForegroundColor DarkGray; Write-Host $treeAfter
    }
} else {
    Write-Host "OK: arbol limpio" -ForegroundColor Green
}

# --- resumen --------------------------------------------------------------------------
$elapsed = (Get-Date) - $started
Write-Host ""
Write-Host ("=" * 62)
if ($failures.Count -eq 0) {
    Write-Host ("VERDE  ({0:mm\:ss})" -f $elapsed) -ForegroundColor Green
    if ($Fast) { Write-Host "Parcial: faltan los casos veryslow." -ForegroundColor Yellow }
    exit 0
}
Write-Host ("ROJO  ({0:mm\:ss})  fallo: {1}" -f $elapsed, ($failures -join ", ")) -ForegroundColor Red
exit 1
