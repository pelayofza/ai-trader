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
app/accounting.py       PnL realizado y no realizado. Una sola definición.
app/state_store.py      La FOTO: estado atómico, con copia rotatoria y arranque
                        tolerante a corrupción.
app/journal.py          La PELÍCULA: diario append-only de ciclos en data/live/.

strategies/registry.py  Registro tipo -> constructor. Una estrategia es {tipo, params}.
risk/engine.py          Puerta única: aprueba, dimensiona y asigna stop-loss/take-profit.
execution/router.py     Enruta cada orden al motor de su clase de activo.
execution/paper.py      Simulación de fills: cuánto se llena y a qué precio.
execution/microstructure.py  Coste de ejecución: spread por símbolo, volatilidad, impacto.
backtest/engine.py      Conduce el runner real sobre histórico; corre planes de folds.
backtest/validation.py  Geometría temporal: walk-forward, CPCV, purga, embargo, auditoría.
data/real_history.py    Barras reales: caché offline y ventana histórica CERRADA.
scoring/multiwindow.py  Agrega las ventanas de una muestra en una distribución robusta.
scoring/real_substrate.py  Sub-ventanas, universo cripto y auditoría de símbolos.
scoring/real_source.py  EL SUSTRATO QUE DECIDE: folds CPCV sobre histórico real, hold-out
                        temporal. Es el sustrato por defecto del optimizador.
scoring/optimize.py     CEM sobre los params de una primitiva. No sabe de dónde salen sus
                        muestras: se las da un SampleSource.
scoring/theme_study.py  ¿Aporta la capa de señal, con el archivo REAL enchufado?

research/               INVESTIGACIÓN APARCADA. El generador sintético y los seis estudios
                        que lo usaban de sustrato. No se trabaja aquí salvo petición
                        explícita; ver "Investigación archivada" al final.
signals/                Ingesta de señales externas: catálogo, puerto, 30 adaptadores,
                        archivo, captura, sonda de profundidad, normalización y
                        codificación de eventos.
signals/ai_reports.py   La SEGUNDA vía de captura: contrato y última ejecución del reporte
                        diario por activo que escribe un agente externo. Sólo lee.
observation/signal_radar.py  Las 30 fuentes -> seis features: tono, intensidad y
                        cobertura, por activo y de mercado. Nunca bloquea sin datos.
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
take-profit se comprueban intrabar contra high/low (lo que esa convención ve y lo que no,
medido: *Dentro de la barra diaria*). Dimensiona por fracción del equity
(compounding real) y separa train (in-sample) de test (out-of-sample); la métrica
cabecera es el **Sharpe out-of-sample penalizado**, `Sharpe − λ·turnover − κ·maxDD`
(`backtest/metrics.py`), pensada para el scoring de estrategias por RL. Sustituyó al
Calmar, que premiaba la inactividad y disparaba la varianza del estimador al meter el
drawdown en el denominador.

Los pesos λ y κ están **medidos, no supuestos** — con una salvedad que hay que decir: se
midieron sobre la librería sintética `ai_v2`, que hoy está aparcada. Se conservan porque lo que
mostró aquel barrido fue que **la superficie es plana**: dentro del rango probado, los pesos no
cambian qué configuración gana. Re-medirlos sobre real está en el roadmap, y no es urgente por
ese motivo. Se barrieron en rejilla sobre cientos de backtests, midiendo por combinación la
correlación de rangos in-sample/out-of-sample y el gap train-validation, más una
auditoría de que la penalización por rotación no duplica los costes que la curva de
equity ya paga. La evidencia se publica en `data/calibration/` y se reproduce con:

```powershell
.venv\Scripts\python.exe -m ai_trader.research.weight_study   # el estudio completo (horas)
.venv\Scripts\python.exe -m ai_trader.research.weight_study --analyze-only  # re-analiza
```

### El sustrato que decide: sub-ventanas del histórico real

El ranking que elige estrategias sale del **mercado**, no de un mundo generado
(`scoring/real_source.py`). Fue sintético hasta que el estudio de transferencia midió que los
dos ranking no se parecen (ρ = −0,04; ver *Investigación archivada*): un juez del que se sabe
que no transfiere no puede seguir eligiendo.

Cómo se muestrea, y por qué así:

- Una **unidad** es una sub-ventana de calendario de 544 días — el mismo troceo del estudio de
  transferencia publicado, para que las cifras de los dos sitios se comparen sin traducir nada.
  Sobre la caché actual son 5 ventanas entre 2018-07-22 y 2025-12-31, con 24 pares cripto.
- Una **muestra** es un fold CPCV dentro de esa ventana: C(6,2) = 15 ventanas OOS con purga y
  embargo. Sin esto el sustrato real daría cinco muestras en total y el CVaR de la recompensa
  sería el mínimo de cinco números.
- El **hold-out es temporal**: la ventana más reciente se reserva y el CEM no la ve nunca.
  Sortearla —que es lo que hace un hold-out de escenarios, donde no hay orden— permitiría
  entrenar en 2024 y validar en 2019, que es fuga temporal disfrazada.
- Los símbolos sin histórico suficiente se **declaran y se omiten**, nunca se rellenan.

Lo que este sustrato **no** arregla, y se dice en vez de taparse: el histórico real es un único
camino con cuatro unidades de entrenamiento. Rankear ahí tiene su propio sobreajuste — que es
exactamente el problema que el sintético venía a resolver y no resolvió. Por eso `describe()`
publica el número de unidades efectivas y el resultado sigue llevando PBO y DSR al lado.

**Coste, medido:** 121 s por (configuración, unidad) con la caché caliente, o sea ~8 min por
candidata sobre las cuatro unidades de train. Una corrida completa de CEM se mide en horas. Los
baselines, en cambio, cuestan 1 s por unidad: van por `multiwindow.baseline_fold_scores`, que
construye los folds y puntúa las carteras pasivas sin correr ninguna estrategia.

Para volver a puntuar sobre una librería generada hay que pedirlo explícitamente:

```python
from ai_trader.research.synthetic_source import SyntheticSampleSource
run_optimization("crypto_momentum", source=SyntheticSampleSource.build())
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
.venv\Scripts\python.exe -m ai_trader.research.validation_study --workers 7
.venv\Scripts\python.exe -m ai_trader.research.validation_study --analyze-only
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

### Un ranking que exige operar (suelo de actividad)

El estudio anterior destapó una propiedad degenerada del ranking, y esta es la respuesta.
El headline de una ventana en la que la estrategia **no abre ninguna posición** es **0
exacto**: la curva es una recta, luego Sharpe 0, rotación 0 y caída 0. No es una nota mala ni
buena, es la *ausencia* de nota — y sin embargo entra en la distribución como un número más.
Como la recompensa es el CVaR@25 % (la media del peor cuartil), en un periodo donde casi todo
lo que se juega pierde **ese cero gana**: la ganadora del ranking real era `mean_reversion#07`,
con el **93 % de sus ventanas vacías**, un CVaR de 0,0000 exacto y el gate **aprobado** (los
pasivos estaban en −1,42).

La respuesta **no** es penalizar la baja rotación: eso ya lo hace λ y el estudio de pesos midió
que penalizar *no* estabiliza nada. Lo que faltaba era un **requisito de elegibilidad**
(`scoring/activity.py`), que es otra cosa: una configuración inelegible conserva su recompensa
publicada intacta y sigue apareciendo en todas las tablas; lo único que pierde es **competir** y
**aprobar el gate**. No perder es legítimo; ganar un ranking de estrategias sin haber jugado, no.

Ahora la actividad viaja **pegada** a la recompensa —en `RewardStats`, en
`MultiWindowValidation`, en el gate y en todo lo que publique un ranking— con dos cifras:
operaciones por ventana OOS y **fracción de ventanas vacías**. Y para ser *rankeable* hay que
superar un suelo de dos condiciones:

| Condición | Valor | De dónde sale |
|---|---|---|
| Ventanas vacías | ≤ **25 %** | **Derivada.** Es α, la fracción de cola que *es* la recompensa (CVaR@25 %). Por encima de ella, el cuartil que fija el CVaR puede estar hecho de ceros estructurales. |
| Operaciones en la ventana mediana | ≥ **3** | **Medida** (`research/activity_study.py`). Regla declarada antes de mirar: el valor de la rejilla {1, 2, 3, 5, 8, 13, 21} que reproduce con menos desacuerdos la condición derivada; a igualdad, el mayor. |

El umbral **3** gana con 1 desacuerdo (frente a 6 del 1, 3 del 2, 3 del 5 y 4 del 8). Y la
elección del número casi no importa, que es la mejor noticia posible sobre un umbral: con 1, 2,
3 o 5 operaciones por ventana sale **el mismo conjunto rankeable** en el lado real (9 de 16),
porque a esa escala quien excluye es la condición derivada. La aritmética que sostiene todo esto
no se razona, se comprueba: de las 1 200 ventanas reales, **307 están vacías y las 307 puntúan 0
exacto**; ninguna ventana con operaciones puntúa 0. Y el **12,2 %** de las ventanas que *fijan*
la recompensa estaban vacías.

Lo que cambia, medido sobre esas mismas 16 configuraciones:

| | sin suelo | con suelo |
|---|---|---|
| Ganadora del ranking real | `mean_reversion#07` (0,07 ops/ventana) | **`crypto_momentum#00`** (32,6 ops/ventana) |
| Configuraciones rankeables | 16 | **9** |
| Aprueban el gate de baselines | 7 | **1** |

Las 6 que pierden la aprobación abren entre 0,1 y 2,9 operaciones por ventana: aprobaban por no
jugar. **La elección cambia, y por eso se publican las dos listas** (`rankings.real` y
`rankings.real_rankable` en `data/transfer/report_ai_v3.json`).

Y el contraste que delimita el alcance: al re-correr el **estudio de validación multiventana**
—donde las cuatro configuraciones operan de sobra— el suelo no mueve **ni un veredicto** (7 y 6
aprobadas, idénticas). El requisito no recorta aprobaciones en general; muerde exactamente donde
la inactividad estaba ganando.

Un control merece mención aparte porque parecía *la* métrica obvia y habría elegido justo al
revés: la **reproducibilidad** del ranking entre mitades del histórico sale **más alta** con las
inactivas dentro (0,39) que sin ellas (0,28; subconjuntos aleatorios del mismo tamaño, 0,36).
Se entiende en cuanto se mide — una configuración que no opera puntúa 0 en todos los bloques y
su puesto no se mueve jamás. Es estabilidad de cementerio, así que se publica como control y no
como criterio.

```powershell
.venv\Scripts\python.exe -m ai_trader.research.activity_study    # evidencia -> data/activity/
```

### Seis primitivas cuya tesis vive en las señales

Había treinta fuentes cableadas y **cero estrategias cuya tesis viviera en ellas**: el caudal
llegaba a la decisión colapsado en seis números que dos primitivas de precio usaban solo como
puerta. Y con seis números el problema es aritmético — cinco estrategias que lean `signal_tone`
no son cinco apuestas, son una repetida cinco veces.

El radar publica ahora **quince números más**: una terna de tono, intensidad y cobertura por cada
uno de cinco **temas** (`liquidation`, `vol_surface`, `macro`, `attention`, `flow`). Los seis de
siempre no cambian ni un dígito, y eso no es una promesa sino una propiedad estructural — el
proveedor temático es una subclase y `features()` llama a `super()`; hay un test que lo compara
byte a byte con `float.hex()`.

| familia | tema | núcleo de precio | qué aporta la señal |
|---|---|---|---|
| `liquidation_cascade` | liquidation | barra de capitulación (rango ≫ ATR, cierre en el extremo) | dónde queda combustible: distancia al cúmulo × notional, apalancamiento, funding |
| `vol_term_structure` | vol_surface | compresión de vol realizada + rotura de Donchian | `−z(skew_25d)`: qué está pagando la protección |
| `event_calendar_drift` | macro | deriva de N días confirmada por una ventana corta | **solo intensidad**: el tema no tiene dirección |
| `attention_ignition` | attention | volumen ÷ mediana móvil + cierre en máximos (**solo largo**) | Upbit, gap App Store Corea−EE.UU., Naver, prima P2P |
| `flow_persistence` | flow | retroceso dentro de tendencia persistente | el único tema con tono de calidad: 11 de 12 fuentes con polaridad |
| `signal_composite` | los cinco | mínimo: piso de ATR + giro de la media corta | **es la tesis**: el agregado de los cinco temas |

Tres decisiones que conviene leer antes que la tabla:

- **La capa de señal es inerte por construcción.** Con los parámetros por defecto no puede
  cambiar elegibilidad, ni lado, ni tamaño, **con ningún radar** —vacío o lleno—. Los cuatro
  mandos están en el borde exacto de su rango, y `_signals_active()` devuelve `False`, así que la
  puerta ni se consulta.
- **Los umbrales de señal no entran en `search_space`**, y el argumento no es la inercia:
  dieciséis de las treinta fuentes empezaron a existir el día que arrancó la captura, así que
  «cobertura de un tema» está correlacionada casi uno a uno con **la fecha**. Un piso sorteable
  dejaría al optimizador elegir en qué tramo de historia se le permite operar.
- **La sexta familia existe por la ley fundamental del gestor activo, y su puerta se pasó antes
  de gastar CPU.** Medido sobre `ai_v4` sin un solo backtest: el tono compuesto correlaciona
  **0,0390** con el retorno futuro contra **0,0186** del mejor tema solo — ×2,10, cuando √K
  predice ×1,55. Si no hubiera agregado, la sexta familia no aportaría nada sobre las otras cinco
  y no habría motivo para pagar su coste en el `n_trials` del DSR.

Y un hallazgo que no estaba previsto: **el radar consume IC**. Lo declarado no llega entero a la
puerta —0,0390 medido contra 0,0744 declarado— entre la tolerancia a datos rancios y la
re-normalización causal. `expected_ic` es una **cota superior**, no una predicción.

#### `ai_v4`: el mismo mundo, que además se deja observar

`ai_v4` es `ai_v3` más cinco canales de observación, uno por tema, emitidos con la fórmula de
siempre `señal_t = ρ·z(r_t→t+h) + √(1−ρ²)·ruido_t`. **Las velas son las mismas**: verificado con
SHA sobre doce muestras, porque el motor no lee `spec.signals` y la emisión es un pase aparte.

```powershell
.venv\Scripts\python.exe -m ai_trader.cli synth derive --from ai_v3 --to ai_v4 --enricher v4
.venv\Scripts\python.exe -m ai_trader.research.fidelity_study --library ai_v4 --offline
```

| canal | ρ | h | φ | informative | coverage | IC esperado |
|---|---|---|---|---|---|---|
| `liquidation` | 0,16 | 2 | 0,55 | 0,25 | 0,50 ⚠ | 0,040 |
| `vol_surface` | 0,12 | 10 | 0,60 | 0,25 | 0,75 | 0,030 |
| `macro` | 0,10 | 5 | 0,10 | 0,10 | 0,85 | 0,010 |
| `attention` | 0,10 | 2 | 0,50 | 0,25 | 0,50 ⚠ | 0,025 |
| `flow` | 0,12 | 10 | 0,45 | 0,40 | 0,85 | 0,048 |

Agregado (√Σ IC²) = **0,0744**. La libreria pasa su estudio de fidelidad con **19
comprobaciones** —las 9 de siempre más 2 por canal— y los cinco canales certifican: el IC medido
cae dentro de la tolerancia y la fuga al pasado queda entre 0,060 y 0,079 contra un límite de
0,100.

⚠ **El suelo de `coverage = 0,50` es una concesión al estimador, no al mundo.** La cobertura real
de dos temas es 0 y 1 fuentes backtesteables; declararla literalmente hace que la librería **no
pueda pasar su propio estudio de fidelidad**, y no porque el mundo esté mal simulado sino porque
el estimador se queda sin muestra. Medido sobre 36 series: con `coverage` 0,20 son cero las series
certificables (hacen falta 200 observaciones), con 0,25 son tres, y con 0,30 la fuga al pasado sale
0,097 contra un límite de 0,100 **sin que se fugue nada** —sigue 1,5/√n—. El sesgo va en la
dirección segura: declarar más cobertura de la que hay hace el mundo *más* favorable a la señal,
así que el break-even que se publique es una cota optimista.

#### Lo que salió al medirlo: el veredicto no se mueve, y el mundo no aporta nada

```powershell
.venv\Scripts\python.exe -m ai_trader.research.transfer_study --library ai_v4 --offline `
    --workers 7 --configs-per-family 8 --verify-determinism 4          # -> data/transfer/report_ai_v4.json
.venv\Scripts\python.exe -m ai_trader.research.transfer_study --library ai_v3 --offline `
    --workers 7 --configs-per-family 8 --out-dir data\transfer\control_8f   # el CONTROL de rejilla
.venv\Scripts\python.exe -m ai_trader.research.activity_study --library ai_v4
```

**ρ = +0,038**, IC95% por bloques `[−0,117, +0,198]`, p de permutación 0,76 → **sin
transferencia**, igual que el `−0,038` publicado con dos familias. Cuadruplicar los candidatos
no convierte al sintético en criterio de selección. Determinismo: 4 comprobaciones, 0
discrepancias.

**El control de rejilla devuelve el informe IDÉNTICO campo a campo** —las 64 configuraciones,
cada score y cada intervalo— salvo `plan.library_id`. Así que la descomposición no deja
residuo: **el efecto del mundo es exactamente cero y todo el cambio es de rejilla.**

Y ese cero era predecible, lo cual es justo lo que lo hace útil como control: `ai_v4` tiene
barras byte a byte iguales a `ai_v3`, y en este estudio la capa de señal está inerte en las
ocho familias, así que los canales declarados no se emiten ni se consultan. **Si el control
hubiera dado algo distinto de cero habría significado una fuga** — algún camino colando
`spec.signals` en un estudio que no debe verlo. Es un test de falsación del diseño aditivo, y
lo pasa.

Con eso, las tres cosas que cambian son atribuibles a la rejilla y no al sustrato:

| | 2 familias (`ai_v3`) | 8 familias |
|---|---|---|
| Spearman(recompensa, operaciones) en el **real** | −0,84 | **+0,004** |
| Top-4 del sintético en la mitad buena del real | 1 de 4 | **4 de 4** (azar 2,0; p=0,057) |
| ρ entre las que operan en ambos mundos | −0,67 (sobre 9) | −0,245 (sobre 43) |

La primera fila es la que más cambia lo que se creía: **aquel −0,84 no describía al mercado de
2018-2025, describía a dos primitivas que apenas operaban**. Con ocho familias, el real deja de
premiar no operar.

El **suelo de actividad se corrobora solo**: re-derivado sobre las 64 configuraciones sale
`T = 3` operaciones por ventana, el mismo valor publicado. No se adopta nada nuevo — se publica
que dos rejillas distintas eligen el mismo número. La elección del ganador sí cambia con el
suelo (`flow_persistence#00` → `event_calendar_drift#02`), que es para lo que el suelo está.

#### La limitación, medida — y más pequeña de lo que se anunció

Esta sección decía que la capa de señal no se podía evaluar hacia atrás en dos de las seis
familias, y repartía los temas usando el flag `backtestable` del catálogo. **Al medirlo, ese
reparto falla en los dos sentidos**, y la limitación real es más pequeña y más precisa.

La evaluabilidad ahora **se mide** sondeando el radar sobre el archivo (`measured_themes`), no se
deriva del catálogo:

| tema | cobertura máx. | sondas legibles | medido | declaraba el catálogo |
|---|---|---|---|---|
| `macro` | 0,833 | 100,0 % | evaluable | evaluable |
| `flow` | 0,583 | 91,1 % | evaluable | evaluable |
| `vol_surface` | 0,333 | 4,2 % | **evaluable** | ciego ← **no coinciden** |
| `attention` | 0,286 | 19,3 % | evaluable | evaluable |
| `liquidation` | 0,167 | 0,0 % | ciego | ciego |

`vol_surface` **se lee**: `deribit_volatility` publica desde **2021-03-24** y el tema supera el
0,25 que exige la puerta. El catálogo lo daba por no backtestable solo porque su profundidad
*medida* aún no llega a los 365 días de `depth.MIN_MEASURED_DAYS`. La frase «sus fuentes
empezaron a existir el día que arrancó la captura» era cierta para `liquidation` y **falsa** para
`vol_surface`. En sentido contrario, `cex_listings` es backtestable y está en `attention`, pero es
un calendario de listados y BTC no se lista: sobre ese símbolo no aporta lectura.

**El único tema que de verdad no llega es `liquidation`** (0,167 máximo, legible el 0,0 % de las
sondas). Su familia es la única declarada no evaluable, y ahora con ese número al lado.

#### Lo que se midió al final

La comparación pareada que esta sección prometía para «cuando haya datos» **ya está medida** sobre
cinco de las seis familias (`theme_study`, informe en `data/themes/`). Misma familia, misma
ventana, mismas barras; lo único que cambia es el umbral de la puerta y si el archivo llega al
motor:

| familia | ciego → armado | intervalo por bloques | movió en | veredicto |
|---|---|---|---|---|
| `signal_composite` | 0,319 → **0,630** | [+0,126, +0,505] | 20/20 | **la capa ayuda** |
| `flow_persistence` | −0,057 → **+0,090** | [+0,013, +0,292] | 14/20 | **la capa ayuda** |
| `event_calendar_drift` | 0,160 → 0,096 | [−0,420, +0,303] | 20/20 | indistinguible |
| `attention_ignition` | −0,014 → −0,014 | [0, 0] | 0/20 | sin potencia |
| `vol_term_structure` | 0,123 → 0,123 | [0, 0] | 0/20 | sin potencia |
| `liquidation_cascade` | — | — | — | no evaluable |

Es la primera evidencia del proyecto en la que una señal externa mueve una decisión sobre mercado
real, y hay que leerla con cuatro reservas que no son menores: **no hay corrección por
comparaciones múltiples** —el intervalo de `flow_persistence` empieza en +0,013 y no sobreviviría
a una—; los 20 pares son 4 configuraciones × 5 ventanas y las de una misma familia están
correlacionadas, así que el **N efectivo es menor que 20** y el intervalo por bloques es
**optimista**; el compuesto, ciego, es un seguidor de tendencia corriente, de modo que *que su
capa ayude* es casi su definición y lo medido es la magnitud, no la dirección; y las cinco
ventanas no comparten universo (de 8 símbolos en la más antigua a 24 en la última).

**Lo que queda abierto es distinto de lo que se anunciaba: medir no es tener potencia.** Dos
familias salieron `sin_potencia` porque la puerta no llegó a atar en ninguna pareja — no porque
el tema fuera ciego, sino por la intersección de tres cosas raras: que el tema sea legible ese día
(`attention` lo es el 19,3 % de las sondas, `vol_surface` el 4,2 %), que el núcleo quiera entrar, y
que el tono cruce el umbral. Eso no lo arregla que una fuente cumpla 365 días: lo arregla **más
archivo**.

Criterio para repetirlo, que no cambia. El estudio de transferencia trocea en sub-ventanas de
**544 días** y cada estrategia necesita **180 de calentamiento**: son **724 días de señal
capturada** para correr UNA sub-ventana con vista. Primera fecha posible para Hyperliquid:
**2028-08-06**. Las cinco sub-ventanas de la geometría publicada pedirían ~2.900 días, ocho años —
esas cifras **no se van a poder reproducir con vista**, y decir lo contrario sería prometer un
calendario que no depende de nadie.

### Dentro de la barra diaria: sesiones y la ventana ciega

La convención de arriba —decidir con la barra cerrada, llenar al open del día siguiente,
comprobar stops contra high/low— trata las **24 horas** de la vela como un bloque opaco.
Durante mucho tiempo se justificó solo por prudencia; ahora está **medida**, abriendo la
vela diaria con barras **1H** de 24 pares cripto sobre una ventana cerrada
(2020-01-01 → 2026-01-01, 43 196 días-símbolo). El día UTC se parte en tres sesiones cuyos
cortes **no son redondos**: cada frontera es la hora de una apertura de mercado real en su
versión más temprana del año (asiática 00–07, europea 07–13, estadounidense 13–24 UTC).

**La cifra que decide: el hueco entre el cierre que la estrategia ve y el open al que se
llena vale 0,07 % del rango del día en mediana (0,55 pb; p99 3,2 %).** En un mercado 24/7
la vela de las 00:00 UTC empieza donde terminó la de ayer: **la ventana ciega no tiene
ancho** y la convención de llenar al open no introduce sesgo. Es un resultado, no una
ausencia de resultado, y desplaza la pregunta.

Limitaciones que sí quedan declaradas:

- **Latencia de ejecución.** Llenar «al open» solo es exacto si la orden sale en ese
  instante. Una hora tarde, el precio de llenado ya se ha desplazado **57,9 pb** (9,2 % del
  rango del día) y se ha gastado el 22,6 % de ese rango. Frente al coste de entrada que el
  motor sí cobra (15 pb de referencia: comisión + deslizamiento plano), llegar una hora
  tarde cuesta **3,9×**. No sesga el precio modelado, pero pone un **techo a la puntualidad**
  con la que el ciclo real debe ejecutar para que el backtest siga describiéndolo.
- **Asimetría entre sesiones.** La sesión estadounidense concentra el 48,9 % de la varianza
  realizada en el 45,8 % del reloj (intensidad 1,07) y **fija el mínimo del día el 47,7 %**
  de las veces. Ahí es donde muerde la convención pesimista de los stops: el motor no sabe
  en qué orden se tocaron stop y objetivo, y ahora al menos se sabe *dónde* cae la ambigüedad.

Sobre la tendencia temporal, la hipótesis declarada antes de medir era que el peso de la
sesión estadounidense crece tras enero de 2024 (ETF al contado de bitcoin). **La dirección
se sostiene y el mecanismo no:** la cuota sube +7,64 puntos entre el antes y el después del
corte, en 10 de 10 pares de la cohorte equilibrada (test de signos exacto, p = 0,002), pero
la serie año a año enseña que ya venía subiendo desde 2020 (Spearman año-cuota = 0,80 *solo*
en los años previos) y el escalón que cruza el corte es **−2,29 puntos**. Lo que el contraste
pre/post mide es una deriva de varios años partida por la mitad, no un efecto de enero de
2024.

**El motor no se ha tocado:** primero se mide y se declara. Evidencia en `data/sessions/` y
en §3.3 de la documentación; se reproduce con:

```powershell
.venv\Scripts\python.exe -m ai_trader.backtest.session_study            # descarga 1H y mide
.venv\Scripts\python.exe -m ai_trader.backtest.session_study --offline  # solo caché
```

### Divergencia live-vs-backtest: la medición cableada, y por qué todavía no da cifra

Es la cifra que justifica el capítulo 3 entero —cuánto se aparta lo ejecutado de lo que el
motor predecía— y **la única del proyecto que no se puede acelerar con cómputo**: consume
calendario. El estudio ya está entero, y lo que hace hoy es **negarse a publicar**.

Coge la ventana que cubre `data/live/cycles.jsonl`, corre el **mismo periodo** con
`backtest/engine.py` sobre las barras reales de esos días y compara **decisión a decisión**
por `(día UTC, símbolo, estrategia)`. No un Sharpe contra otro: dos curvas distintas pueden
dar el mismo Sharpe, y entonces el número no dice *dónde* está la diferencia. La
re-simulación no reimplementa nada — se le engancha un diario en memoria al mismo motor, así
que emite exactamente el mismo esquema de línea que el vivo.

La diferencia de precio se reparte en **tres sumandos que suman**: `referencia` (decidir con
un cierre diario que en el instante del *fill* ya es viejo), `coste` (deslizamiento cobrado
contra el modelado) y el término `cruzado` de segundo orden, publicado en vez de repartido
para que la descomposición cierre y se pueda auditar. Aparte van el embudo de **decisiones**
—si en vivo se generan la mitad de las señales, el problema no es el coste sino los datos— y
la **latencia**, en tiempo (`decided_at → executed_at`, que el diario ahora sella) y en
puntos básicos contra barras 1H reales.

Tres reglas declaradas que **pueden fallar**: cobertura de decisiones ≥ 0,80; desvío mediano
del coste ≤ 5 pb; desplazamiento por latencia ≤ 1× el coste de referencia. Y una puerta
previa: **con menos de 30 días de diario no re-simula ni publica**, porque una divergencia
medida sobre cuatro días tendría el mismo aspecto que la buena. Hoy el diario cubre horas,
así que el informe dice cuántos días faltan y nada más.

**Techo declarado:** mientras la ejecución sea de papel, el `filled_price` lo produce el
mismo motor que el backtest, así que la pierna de coste mide *contexto en vivo contra
contexto re-simulado*, no modelo contra mercado. La de referencia/latencia no tiene ese techo.

```powershell
.venv\Scripts\python.exe -m ai_trader.backtest.divergence_study --offline  # solo caché
.venv\Scripts\python.exe -m ai_trader.backtest.divergence_study --verify-determinism
```

### Señales externas: 30 declaradas, 30 conectadas, 14 backtesteables, y ya en la decisión

Hasta hoy las estrategias solo veían **precio y volumen**. `signals/` es el sitio donde se enchufan
las fuentes externas y está **entero conectado**: 30 fuentes que producen 72 features, **30 con
adaptador**, **21 con profundidad medida** y **14 backtesteables**. Y desde el 12/08/2026 llegan al
espacio de observación y a la decisión, **en backtest y en vivo**.

**Todas por la misma vía**, y **ninguna** al espacio de búsqueda del optimizador. El campo `tier`
describe la naturaleza de la fuente (`A` oferta mecánica, `B` efecto estadístico) y **ya no enruta**:
lo que decide la codificación es la **cadencia**. La bifurcación anterior —lo mecánico como *veto*,
lo continuo como feature— se retiró porque su defensa, «muestras de decenas», era un número que
nadie había medido; al medirlo resultó falso por un factor de diez (**321 ajustes** de dificultad en
la ventana de la sonda —463 desde 2009—, **621 hacks** fechados desde 2016, el calendario del FOMC
desde 2017: **917 eventos** *pooled* en `data/signals/event_pool.json`). Donde la muestra sí es
corta, la razón también está medida: el endpoint de *unlocks* de DefiLlama responde **402 Payment
Required** y beaconcha.in **401 Unauthorized**. Tampoco hay veto: ninguna señal bloquea, y la guarda
operativa que eso deja abierta —sancionado, deslistado, mercado detenido— está declarada aparte en
el roadmap.

Que las tres cifras no coincidan es el contenido, no un hueco. Cada fuente declara `history_from`
—la primera fecha con dato **comprobado por nosotros**— y `pit` (*forward_capture* /
*archive_revisable* / *derived_from_price* / *chain_immutable*). Esa comprobación es una operación
explícita: la sonda descarga la serie, la deriva con el adaptador real y escribe lo que encuentra en
`data/signals/history_depth.json`; el catálogo solo puede declarar lo que ese registro respalda **y
con al menos un año de ventana medida** (la prima P2P, la dispersión de *funding* y la lista OFAC
tienen medición de un día, que es el día que arrancó la captura) — y un test lo verifica.

| fuente | desde (medido) | qué aporta |
|---|---|---|
| `defillama_fees` | 2011-01-31 | comisiones, ingresos y TVL por protocolo o cadena → P/F y P/S |
| `defillama_volumes` | 2017-08-17 | volumen DEX (DefiLlama) contra CEX (CCXT): dónde se forma el precio |
| `defillama_stablecoins` | 2017-11-29 | oferta por cadena: pólvora seca y rotación entre ecosistemas |
| `wikipedia_pageviews` | 2015-07-01 | atención **por idioma**: descomposición geográfica barata |
| `cftc_cot` | 2018-04-13 | posicionamiento CME por categoría, fechado el día que se **publica** |
| `etf_flows` | 2024-01-11 | flujos de ETF spot **por emisor** (TFTC, CC BY 4.0) |
| `github_activity` | 2009-08-23 | commits diarios y contribuidores únicos (asimétrica: la fecha
la alcanza *contributors*; *commits* solo trae 52 semanas por captura) |
| `btc_difficulty` | 2014-08-19 | 321 ajustes en la ventana de la sonda; cabeceras de bloque, no revisables |
| `defillama_hacks` | 2016-06-17 | 621 incidentes fechados: la segunda muestra más grande de las de evento |
| `macro_calendar` | 2017-02-01 | FOMC e IPC: fechas exactas, publicadas con meses de antelación |
| `deribit_volatility` | 2021-03-24 | DVOL, *skew* de 25 delta y estructura temporal (el delta se calcula aquí) |
| `cex_listings` | 2018-08-02 | altas, bajas y avisos de vigilancia de Upbit: 523 eventos sobre 343 tokens |
| `federal_register` | 2015-11-02 | actividad regulatoria fechada, publicada antes que su noticia |
| `sec_edgar_fts` | 2025-06-20 | menciones en *filings* de la SEC, con la pata 13F/13G aparte |

**El lote de alta fricción (13/08/2026) añadió doce fuentes y una codificación.** Lo que las
mantenía sin conectar no era el precio —nueve de las doce son gratis y sin credencial— sino el
trabajo: reconstruir el estado de Hyperliquid cuenta a cuenta, calcular el delta porque el libro de
Deribit no publica griegas, parsear títulos en coreano. Trajeron el **mapa de precios**, la tercera
codificación: un mapa de liquidación no es un evento fechado (no hay fecha futura que anticipar) ni
un nivel comparable con su historia (que un clúster esté al 4 % es un hecho absoluto), y las otras
dos lo leerían mal **sin dar error**. Y trajeron el campo que faltaba, `typical_adv_usd`: el perp
mediano de Hyperliquid mueve **307 000 $/día** y el mercado KRW mediano de Upbit, **248 000 $** —el
«efecto Upbit» es real *y* vive donde no cabe tamaño, y las dos cosas hay que saberlas antes de
escalar, no después—. Se mide con `signals/liquidity.py` y hay un test que exige que lo declarado
esté respaldado por el registro, igual que con `history_from`.

**La fuente número 30 no tiene proveedor: se compone, y lo que midió no es lo que se esperaba.**
`dat_mnav` (13/08/2026) es el índice de estrés de vendedores forzados a partir de las tesorerías
cotizadas. Una tesorería cotizada es una empresa cuyo balance **es** un tesoro de cripto; por encima
de 1× de mNAV emitir acciones es acretivo y por **debajo** diluye, así que la vía barata para
levantar caja pasa a ser **vender el tesoro** — aritmética, no sentimiento. Lo publicable no es el
mNAV de nadie sino la **distribución** por activo subyacente: la fracción bajo 1× es oferta futura
estructural. No hay API (bitcointreasuries, mnav.io, Artemis son cuadros de mando), así que la serie
se compone de tres patas: tenencias y acciones del **XBRL de la SEC** (`CryptoAssetNumberOfUnits`,
`CryptoAssetFairValue`, `Assets`, `EntityCommonStockSharesOutstanding`) y los cierres de la **acción
y del activo de la misma sesión** —no de CCXT: mezclar un cierre cripto de medianoche UTC con uno de
bolsa de las 21:00 mete nueve horas de desfase dentro de un cociente—.

De **138 declarantes** de cripto en el registro XBRL quedan **3 compañías** publicables, en 3 activos
distintos, así que **hoy no hay ninguna distribución que publicar** y la fuente produce 0 filas. Eso
es la medición, no un pendiente, y el desglose está en `data/signals/dat_cohort.json` (`signals dat`):
25 trusts y 3 brokers fuera por SIC, **61 que declaran el valor razonable en dólares y no el número de
unidades**, 26 cuyo tesoro no llega al 50 % del balance (mineras), 6 multiclase, 12 que no nombran su
activo y 4 cuyo activo no se opera. Los tres filtros de cohorte **no miran el mNAV** a propósito:
definirla con él truncaría justo la cola que se publica.

Lo que más costó fue **identificar qué activo tiene cada una**, y la primera versión estaba mal:
identificar por *precio implícito* (valor razonable ÷ unidades) y quedarse con el único activo del
universo que cuadrase daba **dos falsos positivos de ocho** —TON Strategy Co salía NEAR, Hyperion
DeFi salía LTC—, porque «el único que cuadra» solo significa algo con el conjunto de candidatos
completo y hay miles de tokens frente a veinticuatro. Hoy **identifica un nombre** (la etiqueta de
unidad o la razón social) y el precio implícito solo **verifica**, que es lo que atrapa a CleanSpark
declarando 1 719 000 unidades `Bitcoin` a 58,53 $ —son 1 719, con un error de escala de mil en el
propio *filing*—. El **retraso** de publicación no se supone: cada hecho trae su fecha de referencia
y su fecha de publicación, la fila se fecha en la de publicación y la mediana medida (**49 días**) se
declara en el campo nuevo `disclosure_lag_days`, con un test que la compara contra el registro. El
hueco mayor está declarado y no se parchea: las APIs XBRL solo exponen hechos **sin dimensiones**, así
que una compañía con varias clases de acción no publica recuento de ordinarias y **la mayor tesorería
que existe queda fuera**.

**El radar convierte esas 72 columnas en seis números** (`observation/signal_radar.py`, con la forma
exacta del proveedor de régimen): **tono**, **intensidad** y **cobertura**, por activo y de mercado.
El tono suma solo las features con polaridad declarada y razonada una a una; la intensidad no tiene
signo y es el único eje en el que *momentum* y reversión a la media quieren cosas opuestas —piso en
la primera como confirmación, techo en la segunda como filtro de catástrofe—; en el tono las dos
ponen un piso, porque el modo de fallo característico de la reversión es comprar una caída de −3σ
que es el primer día de un reprecio permanente.

**Una puerta de señales nunca bloquea por falta de datos.** «No hay *unlock*» y «no sé de *unlocks*»
se escriben con el mismo cero: la **cobertura** es la feature que los distingue, y por debajo del
25 % de las fuentes del bloque la puerta **no se evalúa** (falla abierta). Ese umbral es una
constante, no un parámetro: si fuera sorteable, el optimizador podría subirlo hasta convertir el
radar en un filtro de *disponibilidad de datos*. Los umbrales por defecto están en el **borde exacto**
del recorte de las z, así que no existe lectura posible que los active — la inercia está demostrada
por el rango, no confiada.

**Los eventos no pasan por la misma normalización** (`signals/events.py`): una z contra una serie
que es 99 % ceros no significa nada. Se codifican como proximidad al próximo evento **acotada a 30
días** (así «no hay nada a la vista» es un 0 finito y no un infinito disfrazado), estela del último
acotada a 10, magnitud sobre una escala declarada por fuente y recortada a ±4 —el mismo tope que las
z—, y una marca de si esa entidad *tiene* calendario. Lo que no se anuncia (un *hack*, una sanción)
tiene la mirada hacia adelante apagada **por código**.

**La compuerta del cableado:** con las puertas en su valor neutro, `validate_multiwindow` devuelve
los scores **idénticos** a los publicados en `data/transfer/units_ai_v3.json` (cinco unidades
reproducidas, quince ventanas OOS cada una), y hay un test que lo comprueba. En el mismo movimiento
se cerró un hueco anterior: `main.py` **no adjuntaba** el proveedor de régimen en producción, así
que cualquier configuración elegida con filtros de régimen activos se comportaba distinto en paper
que en el backtest que la seleccionó; un `Mapping` perezoso sobre `MarketDataService` lo arregla sin
tocar una línea de `observation/regime.py`.

Lo que ese cableado **no** cerraba —limitar los grados de libertad *reduce* el riesgo de sobreajuste,
pero no lo *mide*— lo cierra el apartado siguiente: el **break-even de ρ**, con ρ = 0 como grupo de
control. Lo que sigue abierto es el otro lado de la pregunta: cuánto ρ tiene cada una de estas
treinta fuentes, que se mide en el sustrato real y no aquí.

**Toda feature se publica normalizada, con dos varas** (`signals/normalize.py`): `<feature>_z` es la
z contra la propia historia de la entidad (expansiva y **causal**, mínimo 20 observaciones) y
`<feature>_x` la z contra la sección cruzada del día (mínimo 5 entidades). Centro mediana, escala
IQR/1.349 —con media y sigma, un día extremo apaga la señal justo cuando empieza a pasar algo—,
recorte declarado a **±4** y huecos a **NaN**, nunca a 0: un 0 diría «normal, en la media», que es
una afirmación que no se ha observado. La primera no existe para un listado nuevo; la segunda sí,
desde el primer día, y por eso hacen falta las dos.

**La captura arranca antes que los adaptadores.** 11 de las 29 fuentes son *forward capture*: nadie
publica el pasado de una cola de staking, de un libro P2P ni de la lista SDN de hace tres meses.
Para ésas la profundidad histórica no depende de escribir mejor código después, sino del calendario.
Cada día sin capturar es profundidad que no se recupera.

Cuatro decisiones que se pagan caras si se toman al revés:

- **El puerto tiene dos capas.** `fetch_raw` toca la red y devuelve el payload intacto;
  `daily_from_raw` es una función **pura** que lo traduce. La parte que se equivoca es el mapeo, y
  separadas corregirlo **re-deriva** en vez de re-descargar. La mitad frágil se testea sin red.
- **El crudo va en `data/`, no en `.cache/`.** Append-only en
  `data/signals_raw/<fuente>/<entidad>/<YYYY-MM>.jsonl.gz` con su `fetched_at`. No se re-deriva: se
  **guarda** —el mismo principio que el `spec.json` de cada escenario sintético—. Una revisión del
  proveedor añade una línea y no borra la anterior, que es la única forma de medir cuánto revisa.
  Lo derivado vive en `.cache/signals/` y es desechable.
- **El esquema diario es propio** (`shared/signals.py`): `(entidad, día) -> features + observed`.
  No reutiliza `normalize_bars`, que deduplica con `keep='last'` y se comería tres emisores
  publicando el mismo día. `observed` cuenta las observaciones detrás de cada celda: distingue
  «vale 0» de «no hay dato».
- **La entidad se deriva, y la tabla de overrides arranca vacía** (`shared/entities.py`). La regla
  (`XYZ/USDT -> XYZ`) funciona el día que se añade un listado nuevo; cada `EntityRef` lleva su
  procedencia (`rule` / `override` / `unmapped`) para que el cruce sea auditable. Sobre el universo
  configurado: 24 símbolos → 24 entidades, **100 % por regla, 0 overrides**.

Lo que la medición destapó y no estaba en ningún folleto: el **COT se conoce tres días después de su
fecha** (se archiva por día de publicación; fecharlo el martes metería tres días de futuro en
cualquier cruce), el sello de *funding* de CCXT es el **próximo cobro** y habría fechado
observaciones en el futuro, un único *slug* con 400 se llevaba por delante las otras 23 series de su
fuente, Wikimedia contesta 403 al User-Agent genérico y 429 en cuanto hay prisa, TFTC publica los
ETF de BTC pero no los de ETH, y FRED ya no sirve oro. Ninguno de esos huecos se rellena con un
sustituto que significaría otra cosa: se declara y aparece como cobertura cero.

El radar está **apagado por defecto** (`[signals] enabled = false` en `config/default.toml`). Sin
credenciales, sin archivo o con medio catálogo caído el sistema arranca igual: radar vacío, cobertura
0, puertas saltadas y un aviso explícito. En esa sección no hay ningún umbral, a propósito.

```powershell
.venv\Scripts\python.exe -m ai_trader.cli signals catalog   # las 29 fuentes declaradas
.venv\Scripts\python.exe -m ai_trader.cli signals capture   # archiva lo que devuelvan hoy
.venv\Scripts\python.exe -m ai_trader.cli signals depth     # MIDE la profundidad y compara con lo declarado
.venv\Scripts\python.exe -m ai_trader.cli signals features  # panel normalizado desde el archivo (sin red)
.venv\Scripts\python.exe -m ai_trader.cli signals events    # eventos pooled por fuente + radar de un símbolo
.venv\Scripts\python.exe -m ai_trader.cli signals adv       # MIDE el ADV de las entidades donde vive cada señal
.venv\Scripts\python.exe -m ai_trader.cli signals audit     # cobertura de entidades y archivo
```

### La segunda vía de captura: el reporte diario por activo

Todo lo anterior va contra APIs y devuelve **números**. Hay una segunda vía, de otra naturaleza y
sin una línea de código en común: un **agente externo** (Claude Cowork) corre todas las mañanas a
las **08:00 Europe/Madrid**, lee fuentes públicas de la web y devuelve **categorías** — 37 preguntas
por activo sobre los 24 del universo, más un reporte HTML y la captura numérica de la que salen las
dos cosas. Escribe en `data/signals_raw/ai_reports/{FECHA}/`, que está en el `.gitignore`.

El contrato vive en `config/` y **manda sobre el prompt**: `assets.json` (el universo, y los dos
campos que gobiernan qué preguntas admiten `no_aplica`), `cuestionario_cripto_v2.json`,
`esquema_etiquetas.json`, `plantilla_respuestas_v2.json`, `plantilla_reporte.html` e
`INSTRUCCIONES_AGENTE.md`. Para añadir o quitar un activo se edita el JSON y no se toca ningún
prompt. `tools/validar_respuestas_v2.py` valida un fichero de respuestas contra ese contrato.

Tres decisiones que son el contenido del pipeline:

- **La hora de corte se declara antes de buscar nada** (06:00Z por defecto), y se descarta toda
  fuente publicada después *aunque contenga el dato que se busca*. Un artículo de las 19:00 puede
  llevar el cierre del día: usarlo da un backtest espectacular, un live plano y un error que **no se
  detecta a posteriori**. Cada respuesta lleva `fuente_ts` para que el filtro se pueda volver a
  aplicar en el entrenamiento.
- **Primero se mide, después se narra.** Las dos salidas se derivan de la misma captura numérica
  (`_medidas/medidas_{TICKER}.json`), escrita antes que una sola frase del HTML. En la versión
  anterior el cuestionario se respondía leyendo el reporte recién escrito: eso no medía el mercado,
  medía al redactor — una pregunta acababa puntuando **verbosidad** y otra premiaba que el reporte
  **omitiese** los eventos macro.
- **El ancla es lo único de lo que depende que el día se pueda recuperar.** Precio a la hora de
  corte, como número, contrastado con ≥2 fuentes (si discrepan se guarda el rango, nunca un
  promedio). Sin él no hay forma de calcular a posteriori qué pasó después. El esquema de etiquetas
  a T+14 está declarado y **todos sus campos se escriben a `null`**: el proceso que los rellena es
  cálculo numérico sobre mercado y se dejó fuera a propósito.

Está **capturado, no conectado**: ni una línea del paquete lee todavía ese archivo — no hay
adaptador, no hay feature en el radar y nada llega al motor. Se captura desde ya por el mismo motivo
que las fuentes *forward capture*: el pasado no se puede descargar. `signals/ai_reports.py` sólo lee
el contrato y la última ejecución para el dashboard y la metodología, y
`tests/test_ai_reports_contract.py` comprueba el contrato en cada verificación, porque un JSON mal
cerrado aquí no rompe nada y rompe la ejecución de mañana en un sandbox donde nadie está mirando.

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
típico negociado (`adv_usd` en `research/synthetic/universe.py`), de modo que la columna
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

## Operación continua (paper trading en vivo)

Arrancar el bot no es el objetivo: el objetivo es que **el tiempo empiece a contar**. La
divergencia entre lo ejecutado y lo que el backtest predecía es la única medición del
proyecto que no se puede acelerar con cómputo, y necesita meses de calendario. Cada
semana que el proceso no corre es una semana perdida al final.

### 1. Qué hace falta antes de arrancar

```powershell
Copy-Item .env.example .env    # y rellena TELEGRAM_BOT_TOKEN y TELEGRAM_ALLOWED_CHAT_IDS
$env:AI_TRADER_CONFIG = "config/default.toml"   # universo de 24 pares cripto
```

| Variable | Obligatoria | Para qué |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | sí | Token del bot (BotFather) |
| `TELEGRAM_ALLOWED_CHAT_IDS` | sí | Lista blanca. Sin ella, cualquiera que encuentre el bot puede pausarlo o disparar un ciclo |
| `AI_TRADER_CONFIG` | no (por defecto `config/default.toml`) | Universo, límites de riesgo, comisiones y estrategias |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | solo si operas stocks | Datos de renta variable |

**Cómo descubrir tu chat id la primera vez.** No sirve mandarle `/start` al bot: sin
`TELEGRAM_ALLOWED_CHAT_IDS` el arranque falla *antes* de construir la aplicación, así que
no hay bot al que escribir. Se le pregunta a la API, que no necesita el bot corriendo:

```powershell
# 1. Manda cualquier mensaje a tu bot desde Telegram.
# 2. Pregunta quién le ha escrito:
$t = (Select-String -Path .env -Pattern '^TELEGRAM_BOT_TOKEN=').Line.Split('=',2)[1].Trim()
(Invoke-RestMethod "https://api.telegram.org/bot$t/getUpdates").result.message.chat |
  Select-Object -Property id, type, username -Unique
```

Pega el `id` en `TELEGRAM_ALLOWED_CHAT_IDS` (separados por comas si son varios). Telegram
solo guarda los mensajes 24 h: si sale vacío, vuelve a escribirle y repite.

Con la lista ya rellena, `/start` desde **otro** chat sí devuelve su id en el mensaje de
rechazo, que es para lo que sirve esa vía.

### 2. Arrancar

```powershell
.venv\Scripts\python.exe -m ai_trader.main    # equivalente a `poetry run ai-trader-bot`
```

El ciclo automático **no arranca solo**: hay que encenderlo con `/autoon` desde Telegram.
A partir de ahí corre cada `AUTO_CYCLE_INTERVAL_SECONDS` (900 s, en
`src/ai_trader/bots/telegram_bot.py`), con cerrojo de reentrada: un `/run_cycle` manual y
el programado nunca se solapan.

**Correr cada 15 minutos no es avisar cada 15 minutos.** El ciclo mantiene su ritmo, pero
hacia Telegram solo salen tres cosas: cada **apertura y cada cierre de posición** en el
acto, los **errores**, y un **latido periódico** —el resumen de ciclo— como mucho una vez
cada `ROUTINE_NOTICE_INTERVAL_SECONDS` (24 h, en `src/ai_trader/app/runner.py`). Lo que se
repetía idéntico ciclo tras ciclo (señal generada, rechazo de riesgo, mercado de predicción
no encontrado) se queda en el log; sus cuentas viajan dentro del resumen (`signals=`,
`risk_rejected=`). Un runner **pausado** sigue latiendo una vez al día a propósito: sin ese
mensaje, detenido y caído se ven igual desde fuera.

### 3. Que sobreviva a un reinicio de la máquina

Tarea programada de Windows, con arranque al inicio y reintento si el proceso muere.
`RU` = el usuario con el que corre; `-WindowStyle Hidden` evita la consola:

```powershell
$exe    = "$PWD\.venv\Scripts\pythonw.exe"
$action = New-ScheduledTaskAction -Execute $exe -Argument "-m ai_trader.main" -WorkingDirectory $PWD
$daily  = New-ScheduledTaskTrigger -AtStartup
$policy = New-ScheduledTaskSettingsSet -RestartInterval (New-TimeSpan -Minutes 5) `
            -RestartCount 999 -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "ai-trader" -Action $action -Trigger $daily `
  -Settings $policy -RunLevel Limited
Start-ScheduledTask -TaskName "ai-trader"
```

`-ExecutionTimeLimit 0` es imprescindible: por defecto Windows mata la tarea a los tres
días. `-MultipleInstances IgnoreNew` evita dos procesos escribiendo el mismo estado.

Comprobación y parada:

```powershell
Get-ScheduledTaskInfo -TaskName "ai-trader"   # LastRunTime / LastTaskResult
Stop-ScheduledTask   -TaskName "ai-trader"
```

La alternativa es [NSSM](https://nssm.cc/) si prefieres un servicio de Windows de verdad;
la tarea programada basta y no necesita permisos de administrador.

### 4. Pausar, reanudar y comprobar que sigue vivo

| Quiero | Cómo |
|---|---|
| Parar de abrir posiciones (sin matar el proceso) | `/pause` — se persiste en el estado, así que **sobrevive al reinicio** |
| Reanudar | `/resume` |
| Apagar solo el ciclo automático | `/autooff` (el bot sigue respondiendo) |
| Saber si está vivo | `/ping` → `pong`, o `/status` |
| Saber si el **ciclo** está vivo | La última línea de `data/live/cycles.jsonl`: si su marca de tiempo tiene más de ~20 minutos, el ciclo no está corriendo aunque el bot conteste |

Un runner pausado **también** escribe su línea en el diario. Es lo que distingue «vivo y
parado» de «caído», que desde fuera son idénticos.

```powershell
Get-Content data\live\cycles.jsonl -Tail 1 | ConvertFrom-Json | Select-Object timestamp,status,open_positions,net_pnl_usd
```

### 5. Qué escribe en disco, y qué hay que copiar

| Ruta | Qué es | ¿Git? | ¿Se puede perder? |
|---|---|---|---|
| `data/live/cycles.jsonl` | **Diario de ciclos.** El fichero en curso; siempre este nombre | no | **No.** No se regenera: es la evidencia de lo que se decidió y a qué precio |
| `data/live/cycles-YYYY-MM.NNN.jsonl` | Shards ya cerrados (rotación por mes o al superar 8 MB) | no | No |
| `data/runtime_state.json` | Estado de ejecución: posiciones, PnL del día, pausa | no | Sí, con coste: se pierden las posiciones abiertas |
| `data/runtime_state.json.1` … `.3` | Copias rotatorias, la `.1` es la más reciente | no | Sí |

El diario está **fuera de git a propósito** —crece cada 15 minutos en la máquina que
opera y sería un conflicto de *merge* permanente— pero es un activo del proyecto: hay que
copiarlo a otro disco periódicamente, porque es lo único que no se puede volver a
generar. Un `robocopy data\live <destino>\live /MIR` en una tarea semanal basta.

Escritura resistente a corte de luz, en las dos rutas: el diario es **append-only** con
`fsync` por línea (nunca se reescribe el fichero entero), y el estado se escribe a un
temporal, se sincroniza y se renombra. Si el estado aparece corrupto al arrancar, el
sistema **arranca desde la copia más reciente que parsee y avisa por Telegram**; antes se
arrancaba de cero en silencio, que es el peor final posible: el runner olvidaba las
posiciones abiertas y no las cerraba nunca.

### 6. Observar mercados de predicción sin operarlos (opcional)

```toml
[runner]
prediction_watchlist = ["quien-gana-x", "otro-slug"]
```

Vacía por defecto. Con la lista puesta, cada ciclo anota en el diario el *midpoint* vivo
del CLOB de cada resultado, **sin generar señal, sin pasar por riesgo y sin abrir
posición**. No es universo: no añade tickers ni cambia lo que se opera. Es la única forma
de construir el histórico de Polymarket, que hoy no se puede descargar ni comprar y que
es lo que impide *backtestear* mercados de predicción.

## Desarrollo

```powershell
poetry run pytest      # tests
poetry run ruff check .  # linter
```

## Notas

- `data/runtime_state.json` es **estado de ejecución mutable**, no fuente. No se versiona.
  Sus copias rotatorias (`.1` … `.3`) tampoco. Ver «Operación continua».
- `data/live/` es el **diario de ciclos** del paper trading en vivo: append-only, no se
  versiona y **no se regenera**. Ver «Operación continua» para la copia de seguridad.
- `.cache/bars/` es caché de barras en parquet; se puede borrar sin consecuencias, se regenera.
  El timeframe va en el nombre del fichero (`CRYPTO__BTC_USDT_1D.parquet` /
  `..._1H.parquet`), así que las barras horarias del estudio de sesiones **no pisan** el
  histórico diario que consumen el backtest y el resto de estudios.
- `data/calibration/`, `data/fidelity/`, `data/validation/`, `data/transfer/` y
  `data/sessions/` **sí** se versionan: son la evidencia publicada de los estudios (pesos del
  headline, fidelidad sintético-vs-real, partición temporal, transferencia de ranking y
  descomposición por sesión horaria) que consumen dashboard y documentación.
- **El universo sintético y el operado no son el mismo, a propósito.** `config/default.toml` es lo
  que se opera en vivo y solo lleva símbolos vivos; `config/synthetic.toml` tiene que coincidir
  símbolo a símbolo con `DEFAULT_UNIVERSE` del generador o habría activos sin barras. De ahí que
  `MATIC/USDT` esté fuera del primero (Binance lo deslistó, ahora es POL) y dentro del segundo,
  donde no cotiza contra ningún exchange y retirarlo desincronizaría la evidencia ya publicada
  sobre 35 activos. El motivo completo está en `src/ai_trader/research/synthetic/universe.py`.
- **El diseño de escenarios con IA no es reproducible y no puede serlo:** los modelos actuales
  retiraron los parámetros de muestreo, así que no hay palanca de determinismo. Se mitiga guardando
  el `spec.json` de cada escenario, que es la única salida cara; todo lo posterior se regenera
  determinísticamente desde él.

## Mover la herramienta a otro ordenador

Copia la carpeta `ai-trader` sin `.venv/` ni `venv/`, instala Python y Poetry, y ejecuta
`poetry install` en el destino.

## Investigación archivada: el mundo sintético

> **Esta línea está aparcada.** Durante meses la apuesta fue generar mundos sintéticos con los
> que rankear estrategias: un mundo generado da distribuciones en vez de un único camino
> histórico, y cubre regímenes que la historia no dio. Se llevó hasta el final y se midió.
>
> **Fidelidad: conseguida.** `ai_v3` acepta los nueve umbrales de hechos estilizados contra
> Binance (cobertura 35% → 98%).
> **Transferencia: fallida.** ρ de Spearman entre el ranking real y el sintético = **−0,04**
> sobre 16 configuraciones, y **−0,67** entre las nueve que operan de verdad. La regla de
> aceptación estaba escrita en el código *antes* de mirar (`RHO_ACCEPT = 0.30`).
>
> Fidelidad no es transferencia: un mundo puede tener las colas, el agrupamiento de volatilidad
> y la estructura de correlaciones del mercado y aun así ordenar las estrategias al revés. El
> sustrato que decide es ahora el histórico real (`scoring/real_source.py`).
>
> **Nada de esto se borra** — un resultado negativo caro es el que no hay que repetir. El código
> vive en `src/ai_trader/research/` y no se mantiene; los comandos siguen funcionando con la
> ruta nueva. Lo que sigue son las cifras tal y como se midieron.

### Fidelidad del sustrato sintético

Que la librería sintética tenga colas gruesas, agrupamiento de volatilidad y estructura
serial no dice que los tenga **en la magnitud del mercado**. Esa pregunta se responde
midiendo: `research/synthetic/fidelity.py` calcula los *stylized facts* (autocorrelación de
retornos, autocorrelación de |retorno| a lags 1-10, exceedances más allá de 3σ, curtosis
en exceso y correlaciones cruzadas par a par) y `research/fidelity_study.py` los compara
contra el histórico diario real de Binance vía CCXT, cacheado en disco. El histórico real
se trocea en ventanas del mismo tamaño que un camino sintético, porque esos estimadores
están sesgados en muestras cortas y comparar longitudes distintas compararía el sesgo.

Se reportan tres ejes por métrica: **nivel** (ratio sintético/real), **ordenación**
(correlación de rangos de Spearman sobre la sección cruzada de activos, o de pares) y
**cobertura** (qué fracción de los valores reales cae dentro del [p10, p90] del ensemble).
Además, el estudio es un **test de aceptación**, no un vistazo: contrasta cada medición con
umbrales declarados en `research/synthetic/fidelity.py` (cobertura ≥ 60% por métrica y mediana real
dentro de la banda del ensemble en curtosis, clustering y exceedances) y **devuelve 1** si
no se cumplen, de modo que una regresión del generador rompe el comando. El informe se
escribe igualmente: no cumplir también es un resultado. La evidencia se publica en
`data/fidelity/` y la vista *Fidelidad* del dashboard:

```powershell
.venv\Scripts\python.exe -m ai_trader.research.fidelity_study --library ai_v3            # descarga + mide
.venv\Scripts\python.exe -m ai_trader.research.fidelity_study --library ai_v3 --offline  # solo caché
.venv\Scripts\python.exe -m ai_trader.research.fidelity_study --library ai_v3 --verify-determinism
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
(`research/synthetic/retrofit.py`), sin llamar a la IA. Tres cambios en la física, calibrados
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
### Transferencia de ranking real-vs-sintético

Que el generador se **parezca** al mercado no dice que **ordene** las estrategias como él. Un
generador puede clavar las colas y ordenar al revés, y ninguna métrica de fidelidad lo
detectaría. `research/transfer_study.py` mide justo eso: las 16 configuraciones de las dos primitivas de precio (la
rejilla del estudio de pesos, hipercubo latino con semilla fija) se puntúan **dos veces** y se
compara el orden de los dos rankings.

Todo el diseño persigue que entre los dos lados **lo único que cambie sea el mundo del que
salen los precios**: el mismo config, el mismo universo (los 11 pares que existen a la vez en
Binance y en la librería), el mismo CPCV de 15 ventanas con purga de 10 días y embargo de 5, y
la misma **longitud de ventana** — un camino sintético dura 544 días, así que el histórico real
se trocea en 5 sub-ventanas disjuntas de ese tamaño en vez de evaluarse de una pieza. Los
scores de cada configuración (sub-ventana × fold en el real, muestra × fold en el sintético)
van a **una sola distribución** y se toma su CVaR@25%: el CVaR de CVaR compondría dos
conservadurismos y dispararía la varianza del estimador.

La regla de decisión estaba escrita en el código **antes** de mirar el resultado
(`RHO_ACCEPT = 0.30`). El resultado:

| | valor | |
|---|---|---|
| Spearman de los dos rankings | **−0,04** | IC95% por bloques [−0,44, +0,49] |
| p (permutación, 20 000 barajados) | 0,89 | |
| Top-4 del sintético en la mitad buena del real | 1/4 | por azar saldrían 2,0 |
| Desacuerdos ≥ 8 puestos | 4 | 3 de ellos **sobrevalorados** por el sintético |
| ρ sobre las 9 que operan de verdad en ambos | **−0,67** | IC95% [−0,88, +0,23], p = 0,06 |

**Veredicto: no hay transferencia.** No se diseñan estrategias contra el sintético: el ranking
lo fija el histórico real. Y la lectura de 2026-08-20 va más lejos que la de entonces: el
sintético no se queda como banco de estrés ni como veto — **la línea entera se aparca**. Un
mecanismo que no ordena tampoco vetaría con criterio, y mantener vivo un sustrato que no decide
cuesta atención sin comprar nada.

Antes de creerse un ρ ≈ 0 hay que descartar que el problema sea que **ninguno** de los dos
lados estaba rankeando estrategias, y el estudio lo comprueba: el headline de una configuración
que no opera es **cero exacto** (curva plana → Sharpe 0, rotación 0, caída 0) y en un periodo
donde casi todo lo que arriesga pierde, ese cero gana. La correlación entre recompensa y
actividad lo confirma — **−0,84** en el lado real frente a −0,09 en el sintético: el mercado de
2018-2025 premió no operar y el mundo sintético no. Al quitar las inactivas el acuerdo no
aparece; se vuelve **negativo**. Es un subconjunto *post-hoc* de 9 puntos y su intervalo cruza
el cero, así que es señal fuerte y no prueba, pero descarta que el ρ nulo sea un artefacto.
Ese hallazgo abrió el apartado siguiente: hoy el ranking se publica con y sin **suelo de
actividad**, y con él la ganadora del lado real ya no es la que no operaba.

Se reproduce con (usa la caché de barras ya descargada; verifica determinismo re-corriendo
unidades en procesos nuevos):

```powershell
.venv\Scripts\python.exe -m ai_trader.research.transfer_study --library ai_v3 --offline --workers 7 --verify-determinism 4
.venv\Scripts\python.exe -m ai_trader.research.transfer_study --library ai_v3 --analyze-only   # re-analiza sin backtestear
```

Límites declarados en el propio informe, no en una nota al pie: el histórico real es **un solo
camino** (5 bloques, sin ensemble — de ahí el bootstrap por bloques y no iid), hay **sesgo de
supervivencia** en el lado real que juega *en contra* de la hipótesis que se quería validar, 13
pares del universo operable se omiten por histórico insuficiente (se declaran, no se rellenan),
y 16 configuraciones de las dos primitivas de precio distinguen "ordena como el mercado" de "no ordena", no
0,35 de 0,45. Evidencia completa en `data/transfer/report_ai_v3.json` y documentación en §2.9.
### El break-even del IC: desde qué capacidad predictiva paga una señal

El radar de señales (más abajo) mete treinta fuentes en la decisión, y hasta aquí su única
defensa contra el sobreajuste era **negativa**: ninguna feature entra en `search_space`, así que el
optimizador no puede ajustar umbrales contra el resultado. Eso limita los grados de libertad pero
**no mide nada**, y era el hueco que la propia evolución del radar dejó escrito. Esto es la
medición, y contesta una sola pregunta: **¿a partir de qué ρ una señal externa hace que la
estrategia bata al baseline después de costes?**

Sobre el mundo sintético —cuyo futuro ya está escrito— no se simula la señal sino el **canal de
observación**:

```
señal_t = ρ · z(retorno_t→t+h) + √(1−ρ²) · ruido_t
```

Simular «el sentimiento de Twitter» —su nivel, su estacionalidad, su reacción a un hack— pediría un
generador aprendido de datos, y con él vuelve la circularidad: el mundo contra el que se mide lo
habríamos ajustado nosotros. El canal cuesta **cinco números interpretables**, y lo que se publica
no son «los mejores parámetros» sino un umbral: una propiedad del **diseño** —de esta estrategia,
con estos costes, con esta puerta— y no de ningún histórico. No se puede sobreajustar a datos que
nunca entraron.

Cinco celdas sobre **las mismas barras** (cambiar ρ no mueve ni una vela), las 16 configuraciones de las dos primitivas
publicadas, CPCV de 15 ventanas, 8 muestras: 640 unidades de backtest. La configuración se elige en
los escenarios de *train* y se puntúa en los de *validación*, que no participaron en la elección.

| celda | IC declarado | IC medido | elegida | recompensa OOS | baseline | margen |
|---|---|---|---|---|---|---|
| sin canal ni puerta | — | — | `mean_reversion#06` | +0,447 | +0,586 | −0,140 |
| ρ = 0 **(control)** | 0,000 | +0,004 | `crypto_momentum#07` | −0,578 | +0,586 | −1,164 |
| ρ = 0,05 | 0,050 | +0,054 | `crypto_momentum#07` | −0,464 | +0,586 | −1,050 |
| ρ = 0,10 | 0,100 | +0,106 | `mean_reversion#06` | +0,314 | +0,586 | −0,273 |
| ρ = 0,20 | 0,200 | +0,207 | `mean_reversion#06` | +0,568 | +0,586 | **−0,018** |

**El break-even está por encima de ρ = 0,20**: no se alcanza en la rejilla, aunque en el extremo se
queda a 0,018 puntos. Y eso ya contesta una pregunta que valía una evolución entera, porque un IC
diario **sostenido** de 0,20 es enorme: la referencia habitual para datos alternativos está un orden
de magnitud por debajo. Esa referencia es **literatura, no una medición de este repositorio** —el ρ
de nuestras treinta fuentes está sin medir, y medirlo es trabajo del sustrato real—, así que la
comparación se ofrece como escala, no como conclusión. Lo que sí es medición es lo demás, y la
lectura útil no es «hacen falta señales mejores» sino que el cuello de botella es el **uso**: una
puerta que cierra o abre tira toda la información salvo un bit.

Tres controles hacen legible ese número, y los tres salieron como tenían que salir:

- **ρ = 0 es el grupo de control**, y salió **limpio**: sin información la estrategia no bate al
  baseline (−1,164). Si lo hubiera batido, el barrido se publicaría *anulado y sin break-even*,
  porque lo medido no habría sido capacidad predictiva sino el AR(1) del ruido o el simple hecho de
  operar menos. Es el test de falsación que no existía en ninguna parte del repositorio.
- **La celda «sin canal» es el cero del eje, no una rival.** La puerta sólo puede *quitar* entradas,
  así que la curva de ρ arranca por debajo de ella y sube. Cuánto cuesta ese filtro **depende del
  régimen** y por eso el informe lo publica en los dos lados del hold-out: −1,02 puntos en
  validación (mercado subiendo) y **+0,09 en train** (mercado cayendo, donde filtrar al azar reduce
  exposición y por tanto ayuda). Lo que no depende del régimen —y es lo que sostiene el
  break-even— es la monotonía en ρ. Por eso el valor de la señal se lee **siempre contra ρ = 0** y
  nunca contra la celda sin puerta.
- **El canal entrega lo que declara**: IC medido 0,004 / 0,054 / 0,106 / 0,207 frente a 0 / 0,05 /
  0,10 / 0,20 declarados, y correlación con retornos **ya realizados** de 0,057 — ruido, como debe
  ser en una señal que mira hacia delante. Son umbrales de aceptación que pueden fallar, dentro del
  mismo veredicto binario del estudio de fidelidad.

Y el valor de la información es **monótono** en ρ, que es la comprobación de que todo el
instrumento mide lo que dice: +0,11 · +0,89 · +1,15 sobre el control, para ρ = 0,05 · 0,10 · 0,20.

La lectura más limpia es la de **una misma configuración** a través de las celdas, porque quita el
ruido de que la elegida cambie — y es la que descarta la explicación alternativa («puntúa mejor
porque opera menos»):

| `mean_reversion#06` | sin canal | ρ = 0 | ρ = 0,05 | ρ = 0,10 | ρ = 0,20 |
|---|---|---|---|---|---|
| recompensa OOS | +0,447 | +0,071 | +0,090 | +0,314 | +0,568 |
| operaciones/ventana | 16,9 | 11,8 | 11,9 | 12,1 | 11,6 |

La puerta corta **las mismas ~30 % de entradas en las cuatro celdas**. Lo único que cambia entre
ρ = 0 y ρ = 0,20 es *cuáles* corta, y eso vale medio punto de recompensa. Ahí también se ve el otro
umbral, el que sí cae dentro de la rejilla: entre ρ = 0,10 y ρ = 0,20 la señal deja de salir peor
que ignorarla, es decir, **paga el filtro**; batir además al baseline pide más, porque la
configuración sin puerta ya estaba −0,140 por debajo.

La celda «sin canal» reproduce **128 unidades de `data/transfer/units_ai_v3.json` score a score**:
es la prueba de que la costura del canal —la factoría de proveedores del motor, la tabla de
polaridad inyectable, el catálogo de fuentes simuladas— no movió nada del sistema. Y las señales
llegan a la estrategia por el **mismo contrato que en vivo** (`attach_signal_provider` +
`signal_gate_reason`, mismo recorte anti-*look-ahead*): si entraran por un camino paralelo, el
barrido mediría una estrategia que no es la que opera.

```powershell
.venv\Scripts\python.exe -m ai_trader.research.signal_study --workers 7 --verify-determinism
.venv\Scripts\python.exe -m ai_trader.research.signal_study --analyze-only   # re-analiza sin backtestear
```

Límites declarados, todos en el propio informe: **ninguna de las 16 configuraciones bate al baseline
en validación en ninguna celda** —las dos ventanas de validación tocaron un tramo alcista donde la
cartera equiponderada es muy difícil de batir—, así que lo medido es *cuánto acerca la señal*, no
*cuánto gana*; aquí **no entra un solo dato real** (el sintético es el sustrato de selección y el
real el de verificación, nunca el mismo dato haciendo las dos cosas); con `informative_share` y
`coverage` al máximo lo publicado es la cota **optimista**, porque bajarlos sólo puede empeorarlo;
es **un** canal, así que la *breadth* del grupo de correlación queda declarada y sin medir; y se
barre una sola geometría de adelanto (h = 1), que es la más favorable a la señal. Evidencia completa
en `data/signal_channel/` y documentación en §4.12.

#### Y con ocho familias: nada se mueve

El barrido se repitió sobre las **64 configuraciones** de las ocho familias (2.560 unidades,
16,4 h). El bloque de break-even sale **idéntico campo a campo** al de arriba: mismo veredicto,
mismos márgenes, misma puerta costando 1,025 puntos. Las 48 candidatas nuevas están en el desglose
por configuración y **no mueven ni un margen**.

Y no es que no compitieran, que fue lo primero que sospeché y es falso: ninguna fue descartada, y
las temáticas **operan más** que las publicadas — `event_calendar_drift` abre ~108 operaciones por
ventana frente a las 48 de momentum y las 7 de reversión. Compitieron en igualdad y **ninguna gana
una sola celda**.

Lo único que responde a la señal es el compuesto: `signal_composite#06` no aparece en el top-10 de
la celda ciega y sube al **puesto 2** en la celda de ρ = 0,20 (0,323 con 73,5 operaciones por
ventana). Su posición se mueve con la fuerza de la señal, que es lo que su diseño predice — y aun
así no basta. De hecho **ninguna de las 64 bate al baseline en ninguna celda**, y en la celda ciega
los puestos 2 a 5 los ocupan configuraciones con **0,00 operaciones por ventana**: un cero por no
operar le gana a cualquiera que opere y pierda, que es exactamente para lo que existe el suelo de
actividad.

Comprobaciones: determinismo limpio, control ρ = 0 limpio, baselines idénticos entre celdas, y la
celda ciega reproduce **una a una** las 512 unidades que el estudio de transferencia calculó días
antes. Esa última importa por un motivo extra: esta corrida se **pausó a mitad y se reanudó** desde
su punto de guardado, así que la reproducción exacta de evidencia previa e independiente es la
validación más fuerte del mecanismo de pausa.

#### Los otros dos estudios sobre la misma rejilla

**Validación temporal** (16 configuraciones, 2.560 folds auditados sin fuga). El optimismo del
corte único frente a la **cola** sale **+1,327**, contra el +1,355 publicado con cuatro
configuraciones: que no se mueva al cuadruplicar la rejilla es la corroboración más fuerte que ese
estudio podía dar. Lo que sí crece es la arbitrariedad de la elección: el ganador del corte único
deja de serlo en **6 de 8** escenarios con walk-forward (antes 4) y en **7 de 8** con CPCV
(antes 5). Comprobado que no son empates: hay 20 filas de 128 con todos los folds a cero, y
excluyéndolas los vuelcos salen **idénticos** y el hueco contra la cola sube a **+1,72**.

**Pesos del headline** (64 configuraciones, 34 activas, 11,3 h). Aquí **se cae una conclusión
publicada**. Sobre el subconjunto activo el mejor punto es **λ = 4, κ = 4** con rank IC **0,194**
frente a 0,146 sin penalizar: ganancia **+0,0475 ± 0,0189**. Lo publicado con dos familias decía
que penalizar *no estabiliza*, y allí todas las penalizaciones empeoraban el rank IC.

Tres reservas, y la primera es seria: el óptimo cae en la **esquina** de la rejilla probada, así que
no está acotado — lo medido es «más penalización es mejor que menos dentro de lo probado», no que el
óptimo sea 4. Segunda: penalizar **cambia la elección** hacia un candidato con Sharpe de validación
*menor* (1,49 frente a 1,72), o sea que los pesos que más estabilizan el orden no eligen mejor.
Tercera: **no se adopta**; mover λ cambiaría retroactivamente quién es rankeable en informes ya
publicados. Una coherencia que antes no existía: el λ implícito de los costes es **5,96** sobre las
activas, casi el mismo 6,27 publicado — con dos familias el óptimo empírico era 0 y el implícito
6,3, y se contradecían; con ocho, el empírico se acerca al que los costes ya imponen.

Y un efecto secundario que conviene no esconder: **DSR y PBO no son aditivos**. El Sharpe
deflactado se calcula sobre la distribución del propio conjunto probado —`deflated_sharpe_ratio`
usa `n_trials = len(trials)` y la dispersión de esos Sharpe para estimar el máximo esperado por
azar—, así que **ampliar la rejilla cambia el DSR de las configuraciones que ya estaban**, sin que
nadie haya tocado esas estrategias: pasar de 16 candidatos a 64 endurece el listón para todos. O se
publican dos conjuntos separados —y entonces no hay un DSR del sistema, hay dos— o se publica el
número que sale y se explica. Se hace lo segundo, porque separarlos sería usar la partición para no
pagar el descuento. Afecta al camino del optimizador (`run_optimization`), que es el único que
calcula DSR y PBO; los estudios de arriba publican recompensa y márgenes.
