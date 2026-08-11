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
execution/paper.py      Simulación de fills: cuánto se llena y a qué precio.
execution/microstructure.py  Coste de ejecución: spread por símbolo, volatilidad, impacto.
backtest/engine.py      Conduce el runner real sobre histórico; corre planes de folds.
backtest/validation.py  Geometría temporal: walk-forward, CPCV, purga, embargo, auditoría.
scoring/multiwindow.py  Agrega las ventanas de una muestra en una distribución robusta.
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

# validación multiventana con purga y embargo (walk_forward | cpcv)
poetry run ai-trader backtest --start 2025-12-20 --end 2026-06-01 --validation cpcv
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

### Validación temporal multiventana

Un backtest se puede partir en train y test de muchas formas, y la forma elegida **cambia
la respuesta**. Además del corte único 70/30 —que se conserva como referencia— cada
muestra se puede evaluar en **varias** ventanas out-of-sample disjuntas, y sus headline
scores se agregan con el mismo CVaR@25% que rankea el resto del sistema:

- `walk_forward`: N ventanas consecutivas; cada fold entrena con el pasado y se puntúa en
  el tramo siguiente.
- `cpcv`: Combinatorial Purged Cross-Validation — N grupos y todas las combinaciones de k
  como test, así que salen C(N,k) ventanas y cada tramo se evalúa acompañado de contextos
  distintos.

Entre train y test se abren dos huecos: **purga** (`max_holding_days`, exactamente lo que
puede seguir viva una posición abierta el último día de train) y **embargo** (1% del
rango, contra el eco serial de los retornos). La geometría vive en
`backtest/validation.py`, separada del motor a propósito: una fuga temporal es un error de
geometría y comprobarlo no debería exigir ejecutar nada. `assert_no_leakage` audita el
plan **antes** de gastar cómputo, y hay tests que comprueban día a día que ningún día cae
en los dos lados.

Lo que cambia al partir el tiempo de otra forma está **medido**, no supuesto, y el
resultado no confirmó la sospecha de partida: el corte único **no** sobre-estima
sistemáticamente (su diferencia con la mediana de las ventanas es ≈0, pero va de −3,4 a
+4,7 — es *arbitrario*, no optimista). Lo que sí sostiene la evidencia: puntúa +1,35 por
encima del **CVaR** de las ventanas, porque el CVaR de un solo número es ese número; y que
mover la ventana mueve el resultado **3,3 veces más que cambiar de estrategia** (rango
entre ventanas 5,10 vs 1,53 entre configuraciones), con la configuración ganadora
cambiando en 4 de 8 muestras. No hace falta que el corte único esté sesgado para que sea
una mala regla de decisión: basta con que sea arbitrario. La evidencia se publica en
`data/validation/` y se reproduce con:

```powershell
.venv\Scripts\python.exe -m ai_trader.scoring.validation_study --workers 7
.venv\Scripts\python.exe -m ai_trader.scoring.validation_study --analyze-only
```

Aviso honesto: dentro de un backtest no se ajusta nada (la configuración entra fija y cada
ventana construye reloj, estado y estrategias nuevos), así que purgar y embargar **no**
cambian ninguna cifra out-of-sample —hay un test que lo fija como invariante—. Lo que
aporta valor hoy es tener varias ventanas en vez de una; la purga es la geometría correcta
para cuando algo *sí* se ajuste sobre el train.

**Anualización por clase de activo.** Sharpe, Sortino y volatilidad se anualizan por
`√N`, donde `N` es el número de observaciones al año, y eso depende del mercado: cripto
cotiza 24/7 (365 barras) y la renta variable solo en sesión (252). El factor lo fija el
**universo del config** —252 si la cartera es exclusivamente bursátil, 365 en cuanto hay
un activo 24/7, porque el backtest recorre la unión de días con barra— y se aplica igual
a la estrategia y a sus baselines: comparar dos Sharpe anualizados con escalas distintas
no significaría nada. Las métricas lo reportan como `periods_per_year`. El **CAGR** es
aparte: vive en tiempo de calendario y divide siempre por 365 días naturales, sea cual
sea el activo.

Los mercados de predicción (Polymarket) quedan fuera del backtest: no hay histórico
OHLCV, solo midpoint vivo.

### Fidelidad del sustrato sintético

Que la librería sintética tenga colas gruesas, agrupamiento de volatilidad y estructura
serial no dice que los tenga **en la magnitud del mercado**. Esa pregunta se responde
midiendo: `synthetic/fidelity.py` calcula los *stylized facts* (autocorrelación de
retornos, autocorrelación de |retorno| a lags 1-10, exceedances más allá de 3σ, curtosis
en exceso y correlaciones cruzadas par a par) y `synthetic/fidelity_study.py` los compara
contra el histórico diario real de Binance vía CCXT, cacheado en disco. El histórico real
se trocea en ventanas del mismo tamaño que un camino sintético, porque esos estimadores
están sesgados en muestras cortas y comparar longitudes distintas compararía el sesgo.

Se reportan tres ejes por métrica: **nivel** (ratio sintético/real), **ordenación**
(correlación de rangos de Spearman sobre la sección cruzada de activos, o de pares) y
**cobertura** (qué fracción de los valores reales cae dentro del [p10, p90] del ensemble).
Además, el estudio es un **test de aceptación**, no un vistazo: contrasta cada medición con
umbrales declarados en `synthetic/fidelity.py` (cobertura ≥ 60% por métrica y mediana real
dentro de la banda del ensemble en curtosis, clustering y exceedances) y **devuelve 1** si
no se cumplen, de modo que una regresión del generador rompe el comando. El informe se
escribe igualmente: no cumplir también es un resultado. La evidencia se publica en
`data/fidelity/` y la vista *Fidelidad* del dashboard:

```powershell
.venv\Scripts\python.exe -m ai_trader.synthetic.fidelity_study --library ai_v3            # descarga + mide
.venv\Scripts\python.exe -m ai_trader.synthetic.fidelity_study --library ai_v3 --offline  # solo caché
.venv\Scripts\python.exe -m ai_trader.synthetic.fidelity_study --library ai_v3 --verify-determinism
```

Se publican **dos** informes medidos con el mismo harness y la misma ventana real, porque
sin el "antes" una corrección medida no se distingue de una afirmación:

| | `ai_v2` | `ai_v3` | real |
|---|---|---|---|
| Curtosis en exceso | 0,37 | **3,40** | 4,19 |
| Exceedances > 3σ | 0,55% | **1,27%** | 1,46% |
| Clustering (autocorr. \|r\|, lag 1) | 0,092 | **0,196** | 0,190 |
| Correlación cruzada (par a par) | 0,489 | **0,542** | 0,653 |
| Volatilidad anualizada | 97,2% | 102,4% | 98,7% |
| Cobertura media p10–p90 | 35% | **98%** | — |
| Veredicto | no cumple | **acepta** | |

`ai_v3` se deriva de los mismos `spec.json` que `ai_v1` con el retrofit determinista
(`synthetic/retrofit.py`), sin llamar a la IA. Tres cambios en la física, calibrados
iterando este mismo harness como función objetivo: colas de Student **también en las fases
de calma** (antes eran gaussianas exactas, y son la mayor parte del horizonte), reparto
news/inercia del GARCH movido al *spec* (`vol_news`) y **cargas factoriales que suben en el
pánico** (`beta_stress`) — sin esto último, la correlación de un modelo de factores es una
constante del universo y no puede dispararse en las caídas. Los dos campos nuevos son
neutros por defecto y hay tests que congelan con un hash que `ai_v1` y `ai_v2` se regeneran
byte a byte desde sus `spec.json`.

Dos límites declarados y medidos: el generador cubre la **mediana** del mercado, no los años
de manía (el p90 de la curtosis real va de 30 a 90 en DOGE y XRP, y perseguirlo rompería el
nivel de volatilidad), y la **ordenación** entre activos —qué activo tiene más cola— sigue
siendo floja o negativa. Documentación completa en §2.8.

### Costes de ejecución

El deslizamiento no es una constante. Cada fill paga **medio spread del símbolo**
(tabla explícita en `execution/microstructure.py`: de 0,4 pb en un índice amplio a
20-25 pb en un altcoin de segunda fila), más un término por la **volatilidad reciente**
del activo, más **impacto de mercado** por la ley de raíz cuadrada sobre la fracción del
volumen de la barra que consume la orden. Además, ninguna orden puede consumir más de
una fracción del volumen de la barra (`max_participation`, 10% por defecto): lo que
exceda ese techo **no se llena**. Las salidas están exentas del techo —entrar es
opcional, salir no— y pagan el impacto de todo el tamaño.

La liquidez se estima con la mediana del volumen de las últimas 20 barras **ya
cerradas** (mismo corte anti look-ahead que la estrategia), no con la barra de hoy ni
con su pico. El generador sintético escala el volumen de cada activo a su volumen
típico negociado (`adv_usd` en `synthetic/universe.py`), de modo que la columna
`volume` es un eje de liquidez real y no un adorno.

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
- `data/calibration/` y `data/fidelity/` **sí** se versionan: son la evidencia publicada de los
  estudios (pesos del headline y fidelidad sintético-vs-real) que consumen dashboard y documentación.
- **El universo sintético y el operado no son el mismo, a propósito.** `config/default.toml` es lo
  que se opera en vivo y solo lleva símbolos vivos; `config/synthetic.toml` tiene que coincidir
  símbolo a símbolo con `DEFAULT_UNIVERSE` del generador o habría activos sin barras. De ahí que
  `MATIC/USDT` esté fuera del primero (Binance lo deslistó, ahora es POL) y dentro del segundo,
  donde no cotiza contra ningún exchange y retirarlo desincronizaría la evidencia ya publicada
  sobre 35 activos. El motivo completo está en `src/ai_trader/synthetic/universe.py`.
- **El diseño de escenarios con IA no es reproducible y no puede serlo:** los modelos actuales
  retiraron los parámetros de muestreo, así que no hay palanca de determinismo. Se mitiga guardando
  el `spec.json` de cada escenario, que es la única salida cara; todo lo posterior se regenera
  determinísticamente desde él.

## Mover la herramienta a otro ordenador

Copia la carpeta `ai-trader` sin `.venv/` ni `venv/`, instala Python y Poetry, y ejecuta
`poetry install` en el destino.
