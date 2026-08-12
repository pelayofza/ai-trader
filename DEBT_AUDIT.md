# Auditoría de deuda técnica — Fase 1 (solo lectura)

Base auditada: commit `6700058` en `refactor/safe-cleanup`.
Ningún fichero del repo se ha modificado para producir este documento.

Complementa a [REFACTOR.md](REFACTOR.md), que describe la red de seguridad y el
perímetro. Aquí van el inventario y los hallazgos con su evidencia.

---

## 0. Rectificaciones a REFACTOR.md

Dos cifras del documento de la Fase 0 eran incorrectas. Se corrigen aquí:

| Afirmación en REFACTOR.md | Real | Cómo se detectó |
|---|---|---|
| «`build()` de `build_dashboard.py`: 1.262 líneas (1290–2552)» | **`build()` tiene 51 líneas** (1290–1340). Lo que ocupa el rango hasta la 2549 es `ROADMAP`, una **constante de datos de 1.049 líneas** | Recuento por AST de nodos de nivel superior. El grep original (`^def`) no veía las asignaciones a nivel de módulo, así que atribuyó a la función todo el hueco hasta el siguiente `def` |
| «22.058 líneas en `src/`» | **26.158 líneas** totales; 22.058 son las **no vacías** | `Measure-Object -Line` de PowerShell no cuenta las líneas en blanco. Contrastado con `splitlines()` de Python (26.158) y `wc -l` (26.154; la diferencia son 4 ficheros sin salto final) |

La primera cambia el diagnóstico: no hay ninguna función monstruosa en el dashboard.

---

## 1. Inventario

### Mapa de módulos

| Paquete | LOC | Ficheros |
|---|---:|---:|
| `tests/` | 10.640 | 29 |
| `src/ai_trader/scoring/` | 6.632 | 16 |
| `dashboard/` | 5.082 | 3 |
| `src/ai_trader/signals/` | 3.244 | 10 |
| `src/ai_trader/synthetic/` | 3.002 | 10 |
| `docs/` | 2.874 | 3 |
| `src/ai_trader/backtest/` | 2.825 | 5 |
| `src/ai_trader/signals/adapters/` | 2.794 | 12 |
| `src/ai_trader/app/` | 1.131 | 4 |
| `src/ai_trader/shared/` | 1.032 | 7 |
| `src/ai_trader/observation/` | 1.014 | 4 |
| `src/ai_trader/` (raíz) | 994 | 4 |
| `src/ai_trader/execution/` | 870 | 6 |
| `src/ai_trader/strategies/` | 716 | 5 |
| `src/ai_trader/data/providers/` | 689 | 7 |
| `src/ai_trader/data/` | 564 | 5 |
| `src/ai_trader/bots/` | 321 | 2 |
| `src/ai_trader/risk/` | 241 | 2 |
| `src/ai_trader/notifications/` | 89 | 3 |
| **TOTAL** | **44.754** | **137** |

Los 12 ficheros mayores:

```
2553  dashboard/build_dashboard.py      (1.049 de ellas son la constante ROADMAP)
2529  dashboard/template.py
2223  docs/template.py
1867  src/ai_trader/scoring/transfer_study.py
1331  src/ai_trader/backtest/session_study.py
1103  tests/test_synthetic.py
 868  tests/test_transfer.py
 834  tests/test_signal_adapters.py
 814  src/ai_trader/signals/catalog.py
 753  src/ai_trader/scoring/weight_calibration.py
 741  tests/test_signals.py
 688  src/ai_trader/cli.py
```

### Grafo de dependencias internas

Construido con AST resolviendo también `from paquete import submodulo` (sin esa
resolución, los 12 adaptadores parecen huérfanos y no lo son).

- **Módulos huérfanos: 0.** Ningún módulo de `ai_trader` deja de ser importado.
- **Ciclos de import: 1**, y está gestionado a propósito (ver H5).
- Fan-in (los más importados): `shared.instruments` (30), `shared.clock` (28),
  `shared.schemas` (23), `config` (21), `shared.bars` (21), `signals.source` (21),
  `shared.signals` (21), `backtest.metrics` (17).
- Fan-out (los que más importan): `dashboard.build_dashboard` (42),
  `docs.build_docs` (31), `ai_trader.cli` (23), `backtest.engine` (18), `main` (15).

El fan-out de los dos builders es la cifra que explica su fragilidad: entre los dos
tocan 73 módulos internos y hasta la Fase 0 no los cubría ningún test.

### Puntos de entrada reales: 10, no 4

| Entry point | Cómo se invoca | ¿Bajo red? |
|---|---|---|
| `ai_trader.cli:main` | `ai-trader` (pyproject) | Sí — caracterizado, 63 % |
| `ai_trader.main:main` | `ai-trader-bot` (pyproject) | Parcial — `tests/test_live_wiring.py` |
| `dashboard.build_dashboard:build` | `python -m dashboard.build_dashboard` | Sí — golden del HTML |
| `docs.build_docs:build` | `python -m docs.build_docs` | Sí — golden del HTML |
| `scoring.transfer_study` | `python -m …` | **No** |
| `scoring.validation_study` | `python -m …` | **No** |
| `scoring.weight_study` | `python -m …` | **No** |
| `scoring.activity_study` | `python -m …` | **No** |
| `synthetic.fidelity_study` | `python -m …` | **No** |
| `backtest.session_study` | `python -m …` | **No** |

Los seis `*_study` son ejecutables con `if __name__ == "__main__"` y son los que
**producen los informes de `data/` que luego leen el dashboard y la documentación**. Son
el hueco de cobertura más grande que queda (ver H6).

### Invocación dinámica

El único mecanismo dinámico del repo es el registro de adaptadores de señales:
`signals/capture.py:177` → `signals/adapters/__init__.py:77 register_all()` → el
`register()` de cada uno de los 10 módulos → `signals/source.py:208 register_adapter()`.
Verificado: **los 12 adaptadores están alcanzados**, ninguno es código muerto.

---

## 2. Lo que se buscó y NO está

Vale la pena registrarlo, porque descarta dos de las tres sospechas del encargo.

| Sospecha | Resultado | Método |
|---|---|---|
| Código muerto / features antiguas | **0 hallazgos** | AST sobre las 671 definiciones de nivel superior + recuento de menciones en todo el repo (`.py`, `.toml`, `.md`). Ninguna aparece una sola vez. Con el listón en dos menciones solo salen los colectores de los builders, que es la forma normal de un generador |
| Claves de config muertas | **0 hallazgos** | Las 31 claves de `config/default.toml` aplanadas y contrastadas contra todo el código. `universe` se lee en `config.py:63` y es obligatoria |
| Módulos huérfanos | **0 hallazgos** | Grafo de imports por AST |
| Imports muertos | **0 hallazgos** | `ruff check .` limpio |

**Consecuencia:** no hay nada que marcar con `# TODO(deuda): ¿uso actual?` ni que llevar a
un `DEBT_BACKLOG.md` por la regla de «scope antiguo reutilizable». Esa regla no tiene
ningún caso al que aplicarse en esta base.

La deuda de este repo **no es código huérfano**: es duplicación de cálculo y ficheros
grandes por acumulación de contenido.

---

## 3. Hallazgos

### H1 — `_atr()` duplicado byte a byte entre las dos estrategias

- **Dónde:** `src/ai_trader/strategies/momentum_crypto.py:26` y
  `src/ai_trader/strategies/mean_reversion.py:31`
- **Qué es:** la misma función de 12 líneas (Average True Range), idéntica hasta el
  espaciado en blanco. Similitud estructural AST **1.00**.
- **Por qué es deuda:** dos copias del mismo indicador. Corregir el ATR en una y no en la
  otra haría que dos estrategias que dicen usar el mismo indicador usaran indicadores
  distintos, y eso no lo detecta ningún test: cada estrategia tiene los suyos.
- **Evidencia:** comparación AST normalizada de las 478 funciones de ≥12 líneas del repo;
  es el único par con ratio 1.00. Lectura directa de ambos: mismas llamadas a
  `bar_schema`, mismo `tr1/tr2/tr3`, mismo `ewm(alpha=1/window, adjust=False)`.
  Ningún otro módulo define `_atr` (`grep -rn "_atr" src/` → solo estos dos y sus usos).
- **Riesgo de tocarlo:** **Bajo.** Función pura, sin estado, y las dos estrategias corren
  dentro del backtest sintético que está congelado byte a byte en `tests/golden/`.
- **Confianza en la evidencia:** **Alta.**
- **Acción propuesta:** extraer a `src/ai_trader/shared/indicators.py` (módulo nuevo) o a
  `shared/bars.py`, e importar desde ambas. Verificar con `verify.ps1`: los golden del
  backtest deben salir idénticos.
- **Cubo: ÁMBAR.** Cumple todos los requisitos de VERDE salvo uno: es cálculo numérico
  usado en backtesting, y tu regla lo excluye de VERDE por definición. La decisión que te
  toca es **dónde** vive la función compartida.

### H2 — El cálculo de *stylized facts* está TRIPLICADO

- **Dónde:**
  - `src/ai_trader/synthetic/fidelity.py:154-235` — la versión buena: `log_returns()`,
    `autocorrelation()`, `series_facts()`, cubierta por `tests/test_fidelity.py` (533 líneas)
  - `dashboard/build_dashboard.py:226-268` (`survey()`, dentro de `stylized_facts`)
  - `docs/build_docs.py:86-125` (`_stylized()`)
- **Qué es:** los dos builders reimplementan a mano, con numpy inline, un cálculo que ya
  existe testeado en el paquete: log-retornos, autocorrelación de `r`, autocorrelación de
  `|r|` y exceedances más allá de 3σ. Similitud AST entre las dos copias: **0.89**.
- **Por qué es deuda:** son las cifras de fidelidad que se publican en el dashboard y en la
  documentación. Con tres implementaciones, una corrección en `fidelity.py` —el que tiene
  tests— no llega a los dos artefactos que la gente lee.
- **Evidencia de que el cambio sería equivalente:** verificado **empíricamente**, no por
  inspección. Se ejecutaron ambas implementaciones sobre las mismas 48 series de la
  librería `ai_v3`:

  | Métrica | max\|diferencia\| | media |
  |---|---|---|
  | autocorrelación de `r` | 1,94e-16 | 2,63e-17 |
  | autocorrelación de `\|r\|` | 2,78e-16 | 6,39e-17 |
  | exceedances 3σ | 3,47e-18 | 7,95e-19 |

  Es el epsilon de `float64`: algebraicamente idénticas (el factor N/N−1 se cancela entre
  numerador y denominador). Y como los builders redondean a 3 decimales, la sustitución
  daría **las mismas cifras publicadas**.
- **La diferencia que sí existe, y que hay que decidir:** los contratos de borde no
  coinciden. `series_facts()` exige `MIN_OBSERVATIONS = 200` y los builders aceptan desde
  `len(r) < 10`; `log_returns()` filtra precios no positivos y los builders no. Sobre los
  datos de hoy no cambia nada (**0 series descartadas** de 48, porque los paths son de 730
  días), pero con una librería de paths cortos el resultado **sí** diferiría.
- **Riesgo de tocarlo:** **Medio.** Cambio pequeño en líneas, pero toca cifras publicadas
  y cruza de `dashboard/`+`docs/` hacia el paquete.
- **Confianza en la evidencia:** **Alta** (verificación numérica ejecutada, no razonada).
- **Acción propuesta:** que los dos builders llamen a `fidelity.series_facts()` y se queden
  solo con su proyección de claves (que sí es legítimamente distinta: `ac_spread`/`n_trend`
  en el dashboard, `spread`/`trend` en docs). Los dos golden de HTML son el criterio de
  aceptación: si no cambian ni un byte, la sustitución fue equivalente.
- **Cubo: ÁMBAR.** Cálculo estadístico → nunca VERDE. Además hay una decisión tuya: si al
  unificar se adopta `MIN_OBSERVATIONS = 200` (cambia el contrato para librerías futuras de
  paths cortos) o se parametriza el umbral para conservar el comportamiento actual.

### H3 — `_git()` duplicado en los dos builders

- **Dónde:** `dashboard/build_dashboard.py:122` y `docs/build_docs.py:68`
- **Qué es:** misma función de 6 líneas (ejecutar git y devolver stdout, "" si falla). Solo
  difieren el nombre del parámetro (`*args` / `*a`) y el ajuste de línea.
- **Por qué es deuda:** duplicación literal entre dos módulos que ya comparten mucho.
- **Evidencia:** lectura directa de ambas. `grep -rn "def _git" .` → exactamente dos
  definiciones, ninguna otra en el repo (`tests/golden_support.py` tiene la suya, pero es
  del andamiaje de tests y con `lru_cache`, no es la misma pieza).
- **Riesgo de tocarlo:** **Bajo.** Los dos artefactos están bajo golden.
- **Confianza en la evidencia:** **Alta.**
- **Acción propuesta:** un módulo compartido. **Aquí está la pregunta de fondo**, y es la
  misma que en H1 y H2: `dashboard/` y `docs/` son dos paquetes hermanos, fuera de
  `src/ai_trader/`, sin ningún sitio común donde poner código. Hoy no existe esa carpeta.
- **Cubo: ÁMBAR.** No es mecánico ni local: exige **crear una ubicación nueva**, que es una
  decisión de arquitectura tuya, no mía.

### H4 — `ROADMAP`: 1.049 líneas de contenido editorial dentro del generador

- **Dónde:** `dashboard/build_dashboard.py:1501-2549`
- **Qué es:** una constante de datos (el catálogo de evoluciones pendientes con sus
  prompts) que ocupa el **41 % del fichero**. El código real del builder son ~1.400 líneas.
- **Por qué es deuda:** contenido y código conviven en el mismo fichero; cada edición del
  roadmap produce un diff en el módulo que genera el dashboard.
- **Evidencia:** recuento por AST de nodos de nivel superior (`Assign ROADMAP`, 1.049 loc).
  Se consume en un solo sitio: `collect_roadmap()` (`build_dashboard.py:1277`).
- **Riesgo de tocarlo:** **Bajo** si se mueve a un módulo Python hermano (el golden del
  HTML lo cubre entero). **Alto** si se convierte a JSON/YAML.
- **Confianza en la evidencia:** **Alta.**
- **Acción propuesta:** mover tal cual a `dashboard/roadmap.py` e importarlo. Movimiento
  puro, sin reformatear, para que el diff sea legible.
- **Cubo: ÁMBAR.** Como movimiento a un módulo Python es de riesgo bajo, pero es tu
  contenido y tú decides si quieres esa separación. **Si se propusiera pasarlo a JSON sería
  ROJO**: crearía un formato de fichero nuevo, que es frontera pública por tu regla.

### H5 — Ciclo de imports `scoring.activity` ↔ `scoring.aggregate`

- **Dónde:** `src/ai_trader/scoring/activity.py:76` (importa `DEFAULT_CVAR_ALPHA`) y
  `src/ai_trader/scoring/aggregate.py:10,86`
- **Qué es:** el único ciclo del grafo. **Ya está gestionado a propósito**: `aggregate.py`
  usa `TYPE_CHECKING` para el tipo y un import diferido dentro de la función para
  `measure_activity`, con el motivo escrito en un comentario (`aggregate.py:83-85`).
- **Por qué podría no ser deuda:** está documentado, es deliberado y funciona. Romperlo
  exigiría mover `DEFAULT_CVAR_ALPHA` a un módulo neutro, y esa constante define la
  recompensa (CVaR@25 %) que consumen 13 módulos.
- **Riesgo de tocarlo:** **Alto.** Toca la definición de la recompensa.
- **Confianza en que sea deuda:** **Baja** — hay un argumento explícito de por qué está así.
- **Acción propuesta:** ninguna. Queda registrado para que nadie lo "arregle" sin leer el
  comentario.
- **Cubo: ROJO.** Confianza Baja → ROJO automático por tu regla. Coincide con el criterio.

### H6 — Seis entry points sin ninguna red

- **Dónde:** `scoring/transfer_study.py`, `scoring/validation_study.py`,
  `scoring/weight_study.py`, `scoring/activity_study.py`, `synthetic/fidelity_study.py`,
  `backtest/session_study.py`
- **Qué es:** seis programas ejecutables (`python -m …`) que generan los informes de
  `data/` que consumen el dashboard y la documentación. Cobertura: 21 % (`activity_study`),
  32 % (`weight_study`), 53 % (`validation_study`), 70 % (`transfer_study`),
  73 % (`session_study`).
- **Por qué es deuda:** no es deuda de código, es **deuda de verificación**. Son la fuente
  de las cifras publicadas y nadie comprueba que sigan produciendo lo mismo.
- **Evidencia:** `grep -rln "if __name__" src/` + medición de cobertura de la Fase 0.
- **Riesgo de tocarlos:** **Alto.** Cálculo pesado, poco cubierto y caro de ejecutar.
- **Confianza:** **Alta** en el diagnóstico; irrelevante para la acción, porque la acción no
  es tocarlos.
- **Acción propuesta:** **no refactorizar nada aquí.** Si en algún momento hay que hacerlo,
  antes toca caracterizar su informe de salida (los JSON de `data/` ya publicados sirven de
  golden natural, igual que se hizo con los HTML).
- **Cubo: ROJO.**

### H7 — Pares con similitud alta que NO son deuda

Se registran para que no se vuelvan a levantar en la próxima pasada.

| Par | Ratio | Por qué no es deuda |
|---|---|---|
| `build_dashboard.py:224 stylized_facts` / `:226 survey` | 0,97 | `survey` está **anidada dentro** de `stylized_facts`; el detector las cuenta dos veces |
| `telegram_bot.py:52 authorized` / `:56 wrapper` | 0,95 | Igual: `wrapper` es el closure del decorador |
| `validation_study.py:146 _init_worker` / `weight_study.py:103 _init_worker` | 0,85 | Mismo patrón de arranque de worker, pero con parámetros distintos (`purge_days` vs `specs`+`split_ratio`). Unificarlos daría un helper con firma variádica: más difícil de leer que las dos copias. **Confianza Media → ROJO** |
| `mean_reversion.py:92` / `momentum_crypto.py:77` `__post_init__` | 0,90 | Validación de parámetros propios de cada estrategia. Comparten forma, no contenido. **Confianza Media → ROJO** |
| `backtest_source.py:140 trading_days` / `baselines.py:288 _calendar` | 0,89 | No verificado en profundidad. **Confianza Media → ROJO** |
| `defillama.py:102 _stablecoin_row` / `fred.py:139 _macro_row` | 0,89 | Adaptadores de proveedores distintos; comparten la forma del contrato `daily_from_raw`, que es justo el patrón que hace barata la fuente N+1. **Confianza Baja → ROJO** |

### H8 — Bloques de plantilla muy largos

- **Dónde:** `docs/template.py:106 _signals_block()` (234 loc), `:882 _fidelity_block()`
  (153), `:1037 _transfer_block()` (143), `:747 _validation_block()` (133),
  `:436 _sessions_block()` (132)
- **Qué es:** funciones que emiten HTML por concatenación. 13 funciones del repo superan
  las 120 líneas; cinco de ellas están aquí.
- **Por qué podría ser deuda:** tamaño. **Por qué podría no serlo:** son plantillas
  lineales; trocearlas suele empeorar la legibilidad, porque el HTML deja de leerse en
  orden.
- **Riesgo:** **Bajo** (golden del HTML), pero el diff sería enorme para cero cambio
  funcional.
- **Confianza en que sea deuda:** **Media** — es una cuestión de gusto, no un defecto.
- **Cubo: ROJO.** Confianza Media → ROJO por tu regla.

### H9 — `main()` de `cli.py`: 185 líneas

- **Dónde:** `src/ai_trader/cli.py:500`
- **Qué es:** la definición entera del parser de argparse (7 subcomandos, ~40 argumentos).
- **Por qué no se toca:** **es la frontera pública del CLI.** Cualquier reorganización
  arriesga cambiar un `help`, un `default` o un `choices`, y eso es contrato con quien usa
  la herramienta.
- **Cubo: ROJO.** Frontera pública → ROJO por tu regla, sin matices.

### H10 — `pytest-cov` no está declarado como dependencia

- **Dónde:** `pyproject.toml`, `[tool.poetry.group.dev.dependencies]`
- **Qué es:** en la Fase 0 se instaló a mano en el venv para medir cobertura. No está
  declarado, así que la medición no es reproducible en otra máquina.
- **Evidencia:** `grep -n "pytest-cov" pyproject.toml` → sin resultados; el paquete está
  instalado en `.venv`.
- **Riesgo de tocarlo:** **Bajo.** Añadir una dependencia de *dev* no altera el
  comportamiento de la herramienta. Nota: `poetry` falla en silencio en este entorno, así
  que el `poetry.lock` habría que regenerarlo aparte o dejarlo pendiente.
- **Confianza:** **Alta.**
- **Acción propuesta:** añadir `pytest-cov = "^6.0"` al grupo de dev.
- **Cubo: VERDE.** Mecánico, local, reversible con una línea, y verificable
  (`verify.ps1` debe seguir en verde). No toca código, ni cálculo, ni frontera de runtime.

---

## 4. Resumen por cubos

| Cubo | Hallazgos |
|---|---|
| **VERDE** (ejecutable sin consultar) | H10 |
| **ÁMBAR** (requiere tu criterio) | H1, H2, H3, H4 |
| **ROJO** (no se toca) | H5, H6, H7 (6 pares), H8, H9 |

Ganancia potencial si se aprobara todo el ámbar: ~1.130 líneas movidas o eliminadas
(1.049 de `ROADMAP` + ~80 de duplicación real). En código *ejecutable*, la deuda
eliminable son **unas 80 líneas** — el resto es mover contenido de sitio.

## 5. La decisión que desbloquea tres de los cuatro ámbar

H1, H2 y H3 son el mismo problema: **hay lógica compartida entre `dashboard/`, `docs/` y
`src/ai_trader/`, y hoy no existe ningún sitio donde ponerla.** Las opciones:

1. **Que los builders importen de `ai_trader`** — resuelve H2 de forma natural
   (`fidelity.series_facts` ya está ahí y testeado) y H1 (`shared/indicators.py`), pero no
   H3: `_git` no pinta nada dentro del paquete de trading.
2. **Crear un paquete hermano** (`reporting/`, `artifacts/`) del que dependan los dos
   builders — resuelve los tres, pero añade un paquete nuevo al proyecto.
3. **Solo H2** (que es donde está el riesgo real de divergencia de cifras) y dejar H1 y H3.

## 6. Nota sobre las reglas de clasificación

Aplicadas a este repo, tus reglas dejan **un solo VERDE**, y quiero que quede claro por qué
antes de que parezca que la auditoría no encontró nada.

La regla «cualquier cosa que toque cálculo numérico, estadística, aleatoriedad,
entrenamiento, backtesting o métricas → nunca VERDE» cubre `scoring/` (6.632 loc),
`synthetic/` (3.002), `backtest/` (2.825) y `strategies/` (716): **13.175 de las 26.158
líneas de `src/`**, el 50 %. Sumando la regla de frontera pública (`cli.py`, `config.py`,
los formatos de `data/`) y la de confianza Media→ROJO, lo que queda elegible para VERDE es
casi solo la configuración del proyecto.

No sugiero relajarlas: son coherentes con lo que pediste, y la alternativa —tocar el motor
de backtest con una red que no cubre los seis `*_study`— es exactamente el riesgo que
querías evitar. Pero sí implica que **el grueso del valor de esta fase está en los cuatro
ámbar, y los cuatro necesitan una decisión tuya para avanzar.**
