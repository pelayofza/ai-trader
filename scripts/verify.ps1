<#
.SYNOPSIS
    Verificacion unica de ai-trader. Si esto pasa en verde, el comportamiento observable
    no ha cambiado.

.DESCRIPTION
    Tres cosas, en orden de coste creciente:

      1. ruff        - linting sobre src, tests, dashboard y docs
      2. pytest      - la suite entera, incluidos los tests de CARACTERIZACION que
                       comparan la salida real contra tests/golden/
      3. arbol limpio- que verificar no haya dejado ficheros modificados por el camino

    El paso 3 no es cosmetico: varios comandos publican informes en data/, y si la propia
    verificacion ensucia el arbol deja de poder distinguirse un cambio del refactor de un
    efecto secundario de haberlo comprobado.

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

# --- estado del arbol ANTES, para el paso 3 -------------------------------------------
$treeBefore = (& git status --porcelain) -join "`n"

# --- 1. linting -----------------------------------------------------------------------
Step "ruff" { & $python -m ruff check . }

# --- 2. la suite ----------------------------------------------------------------------
if ($Regold) { $env:AI_TRADER_REGOLD = "1" } else { Remove-Item Env:\AI_TRADER_REGOLD -ErrorAction SilentlyContinue }

$pytestArgs = @("-m", "pytest", "-q", "--tb=short")
if ($Fast) {
    $pytestArgs += @("-m", "not veryslow")
    Write-Host "(modo -Fast: se saltan los casos veryslow)" -ForegroundColor DarkGray
}
Step "pytest" { & $python @pytestArgs }

if ($Regold) { Remove-Item Env:\AI_TRADER_REGOLD -ErrorAction SilentlyContinue }

# --- 3. la verificacion no debe ensuciar el arbol -------------------------------------
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
