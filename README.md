# ai-trader

Sistema de trading algorítmico en **paper trading**, con vocación de cubrir tres clases de activo:
renta variable, cripto y mercados de predicción (Polymarket).

Estado actual: cripto (vía CCXT/Binance) y Polymarket funcionando en paper. Stocks (vía Alpaca)
tiene proveedor de datos pero ninguna estrategia.

## Arquitectura

El flujo es siempre el mismo, venga la señal de donde venga:

```
estrategia  ->  Signal  ->  riesgo  ->  OrderRequest  ->  router  ->  motor de ejecución
```

Ninguna orden llega al mercado sin pasar por el motor de riesgo, y el riesgo —no la
estrategia— es quien fija el stop-loss y el take-profit definitivos.

```
config/default.toml     Universo, límites de riesgo, comisiones y estrategias activas.
config.py               Carga el TOML y lo valida.
cli.py                  Entrada headless: `run-cycle`, `report`. No necesita Telegram.
main.py                 Entrada con bot de Telegram.

app/runner.py           Orquestador. Solo orquesta.
app/reports.py          Formateo de informes para humanos.
app/state_store.py      Persistencia atómica del estado.

strategies/registry.py  Registro tipo -> constructor. Una estrategia es {tipo, params}.
risk/engine.py          Puerta única: aprueba, dimensiona y asigna stop-loss/take-profit.
execution/router.py     Enruta cada orden al motor de su clase de activo.
execution/paper.py      Simulación de fills: slippage y comisiones.
notifications/          Canal hacia el humano (Telegram) desacoplado del núcleo.
data/                   Proveedores (Alpaca, CCXT, Polymarket) + caché parquet.
shared/                 Vocabulario común + `clock.py`, la costura para el backtest.
```

## Puesta en marcha

Requiere Python 3.11+ y [Poetry](https://python-poetry.org/).

```powershell
poetry install
Copy-Item .env.example .env   # y rellena los valores

poetry run ai-trader run-cycle    # un ciclo, headless
poetry run ai-trader report       # informes, sin operar
poetry run ai-trader-bot          # con bot de Telegram

# backtest con split train/test out-of-sample
poetry run ai-trader backtest --start 2025-12-20 --end 2026-06-01 --capital 10000
```

## Backtest

Reproduce las estrategias configuradas sobre histórico, conduciendo el **mismo runner**
que opera en vivo con un reloj simulado y datos con anti look-ahead. La decisión se toma
con la barra ya cerrada, la entrada se llena al open del día siguiente y los stop-loss /
take-profit se comprueban intrabar contra high/low. Dimensiona por fracción del equity
(compounding real) y separa train (in-sample) de test (out-of-sample); la métrica
cabecera es el **Sharpe out-of-sample penalizado**, `Sharpe − λ·turnover − κ·maxDD`
(`backtest/metrics.py`), pensada para el scoring de estrategias por RL. Sustituyó al
Calmar, que premiaba la inactividad y disparaba la varianza del estimador al meter el
drawdown en el denominador.

Los pesos λ y κ están **medidos, no supuestos**: se barrieron en rejilla sobre cientos
de backtests reales de la librería sintética `ai_v2`, midiendo por combinación la
correlación de rangos in-sample/out-of-sample y el gap train-validation, más una
auditoría de que la penalización por rotación no duplica los costes que la curva de
equity ya paga. La evidencia se publica en `data/calibration/` y se reproduce con:

```powershell
.venv\Scripts\python.exe -m ai_trader.scoring.weight_study   # el estudio completo (horas)
.venv\Scripts\python.exe -m ai_trader.scoring.weight_study --analyze-only  # re-analiza
```

Los mercados de predicción (Polymarket) quedan fuera del backtest: no hay histórico
OHLCV, solo midpoint vivo.

Variables de entorno (ver `.env.example`):

| Variable | Obligatoria | Para qué |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | sí | Bot de Telegram |
| `TELEGRAM_ALLOWED_CHAT_IDS` | sí | Lista blanca de chats autorizados |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | solo si operas stocks | Datos de renta variable |

## Comandos de Telegram

`/status` `/positions` `/risk` `/history` `/performance` `/symbols` `/price SYMBOL`
`/pause` `/resume` `/run_cycle` `/autoon` `/autooff`

## Desarrollo

```powershell
poetry run pytest      # tests
poetry run ruff check .  # linter
```

## Notas

- `data/runtime_state.json` es **estado de ejecución mutable**, no fuente. No se versiona.
- `.cache/bars/` es caché de barras en parquet; se puede borrar sin consecuencias, se regenera.

## Mover la herramienta a otro ordenador

Copia la carpeta `ai-trader` sin `.venv/` ni `venv/`, instala Python y Poetry, y ejecuta
`poetry install` en el destino.
