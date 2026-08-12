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
scoring/multiwindow.py  Agrega las ventanas de una muestra en una distribución robusta.
scoring/transfer_study.py  ¿Ordena el mundo sintético las estrategias como el real?
signals/                Ingesta de señales externas: catálogo, puerto, 17 adaptadores,
                        archivo, captura, sonda de profundidad, normalización y
                        codificación de eventos.
observation/signal_radar.py  Las 17 fuentes -> seis features: tono, intensidad y
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

### Transferencia de ranking real-vs-sintético

Que el generador se **parezca** al mercado no dice que **ordene** las estrategias como él. Un
generador puede clavar las colas y ordenar al revés, y ninguna métrica de fidelidad lo
detectaría. `scoring/transfer_study.py` mide justo eso: las mismas 16 configuraciones (la
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
lo fija el histórico real y el sintético se queda como banco de estrés y veto, nunca como
criterio de selección.

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
.venv\Scripts\python.exe -m ai_trader.scoring.transfer_study --offline --workers 7 --verify-determinism 4
.venv\Scripts\python.exe -m ai_trader.scoring.transfer_study --analyze-only   # re-analiza sin backtestear
```

Límites declarados en el propio informe, no en una nota al pie: el histórico real es **un solo
camino** (5 bloques, sin ensemble — de ahí el bootstrap por bloques y no iid), hay **sesgo de
supervivencia** en el lado real que juega *en contra* de la hipótesis que se quería validar, 13
pares del universo operable se omiten por histórico insuficiente (se declaran, no se rellenan),
y 16 configuraciones de dos familias distinguen "ordena como el mercado" de "no ordena", no
0,35 de 0,45. Evidencia completa en `data/transfer/` y documentación en §2.9.

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
| Operaciones en la ventana mediana | ≥ **3** | **Medida** (`scoring/activity_study.py`). Regla declarada antes de mirar: el valor de la rejilla {1, 2, 3, 5, 8, 13, 21} que reproduce con menos desacuerdos la condición derivada; a igualdad, el mayor. |

El umbral **3** gana con 1 desacuerdo (frente a 6 del 1, 3 del 2, 3 del 5 y 4 del 8). Y la
elección del número casi no importa, que es la mejor noticia posible sobre un umbral: con 1, 2,
3 o 5 operaciones por ventana sale **el mismo conjunto rankeable** en el lado real (9 de 16),
porque a esa escala quien excluye es la condición derivada. La aritmética que sostiene todo esto
no se razona, se comprueba: de las 1 200 ventanas reales, **307 están vacías y las 307 puntúan 0
exacto**; ninguna ventana con operaciones puntúa 0. Y el **12,2 %** de las ventanas que *fijan*
la recompensa estaban vacías.

Lo que cambia, medido sobre las mismas 16 configuraciones:

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
.venv\Scripts\python.exe -m ai_trader.scoring.activity_study    # evidencia -> data/activity/
```

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

### Señales externas: 17 declaradas, 17 conectadas, 10 backtesteables, y ya en la decisión

Hasta hoy las estrategias solo veían **precio y volumen**. `signals/` es el sitio donde se enchufan
las fuentes externas y está **entero conectado**: 17 fuentes que producen 40 features, **17 con
adaptador**, **13 con profundidad medida** y **10 backtesteables**. Y desde el 12/08/2026 llegan al
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

**El radar convierte esas 40 columnas en seis números** (`observation/signal_radar.py`, con la forma
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

Lo que **no** cierra, y se escribe en vez de callarse: sin el *break-even* de ρ (evolución «canal de
observación sintético») no hay test de falsación. Limitar los grados de libertad reduce el riesgo de
sobreajuste; no lo mide.

**Toda feature se publica normalizada, con dos varas** (`signals/normalize.py`): `<feature>_z` es la
z contra la propia historia de la entidad (expansiva y **causal**, mínimo 20 observaciones) y
`<feature>_x` la z contra la sección cruzada del día (mínimo 5 entidades). Centro mediana, escala
IQR/1.349 —con media y sigma, un día extremo apaga la señal justo cuando empieza a pasar algo—,
recorte declarado a **±4** y huecos a **NaN**, nunca a 0: un 0 diría «normal, en la media», que es
una afirmación que no se ha observado. La primera no existe para un listado nuevo; la segunda sí,
desde el primer día, y por eso hacen falta las dos.

**La captura arranca antes que los adaptadores.** 5 de las 17 fuentes son *forward capture*: nadie
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
.venv\Scripts\python.exe -m ai_trader.cli signals catalog   # las 17 fuentes declaradas
.venv\Scripts\python.exe -m ai_trader.cli signals capture   # archiva lo que devuelvan hoy
.venv\Scripts\python.exe -m ai_trader.cli signals depth     # MIDE la profundidad y compara con lo declarado
.venv\Scripts\python.exe -m ai_trader.cli signals features  # panel normalizado desde el archivo (sin red)
.venv\Scripts\python.exe -m ai_trader.cli signals events    # eventos pooled por fuente + radar de un símbolo
.venv\Scripts\python.exe -m ai_trader.cli signals audit     # cobertura de entidades y archivo
```

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
  sobre 35 activos. El motivo completo está en `src/ai_trader/synthetic/universe.py`.
- **El diseño de escenarios con IA no es reproducible y no puede serlo:** los modelos actuales
  retiraron los parámetros de muestreo, así que no hay palanca de determinismo. Se mitiga guardando
  el `spec.json` de cada escenario, que es la única salida cara; todo lo posterior se regenera
  determinísticamente desde él.

## Mover la herramienta a otro ordenador

Copia la carpeta `ai-trader` sin `.venv/` ni `venv/`, instala Python y Poetry, y ejecuta
`poetry install` en el destino.
