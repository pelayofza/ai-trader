# Backlog de deuda: lo que NO se toco, y por que

Estado tras la Fase 2. Complementa a [DEBT_AUDIT.md](DEBT_AUDIT.md) (inventario y
hallazgos) y a [REFACTOR.md](REFACTOR.md) (red de seguridad y perimetro).

Ordenado por **ratio valor / riesgo**: arriba lo que mas compensa por lo que cuesta.
Cada entrada dice **que evidencia falta** para poder decidirla.

---

## 0. Correccion a DEBT_AUDIT.md: la duplicacion estaba INFRAESTIMADA

La auditoria concluyo «unas 80 lineas de deuda eliminable en codigo ejecutable» y solo
identifico dos duplicados exactos (`_atr` y `_git`). **Esa cifra estaba mal por un fallo
de metodo mio**: el detector filtraba funciones de **≥12 lineas**, y ese umbral arbitrario
oculto la mayor parte del problema, que esta en funciones cortas.

Repetido el barrido **sin umbral**, comparando el AST del cuerpo (sin docstrings):

| | Auditoria (Fase 1) | Real |
|---|---|---|
| Grupos de cuerpo identico | 2 | **21** |
| Lineas eliminables | ~80 | **~135** |

Lo que sigue incorpora ya la cifra corregida.

---

## 1. B1 — `load_*_report`: seis copias literales del mismo cargador

**El mejor ratio del backlog.** Valor medio, riesgo muy bajo, y no toca cálculo.

- `src/ai_trader/backtest/session_study.py:1183` `load_sessions_report`
- `src/ai_trader/scoring/activity_study.py:92` `load_activity_report`
- `src/ai_trader/scoring/transfer_study.py:169` `load_transfer_report`
- `src/ai_trader/scoring/validation_study.py:132` `load_validation_report`
- `src/ai_trader/scoring/weight_calibration.py:68` `load_calibration_report`
- `src/ai_trader/synthetic/fidelity.py:598` `load_fidelity_report`

**Evidencia:** los seis cuerpos son idénticos byte a byte tras normalizar el AST. Los seis
hacen lo mismo: `Path(x)`, `if not exists: return None`, `json.loads(read_text)`. Solo
cambian el nombre y el docstring. ~35 líneas eliminables.

**Riesgo: Bajo.** No hay cálculo, no hay estado, y los seis se ejercitan desde el dashboard
y la documentación, que están bajo golden.

**Confianza: Alta.**

**Qué falta para decidir:** dónde vive el helper común. Mismo bloqueo que B2 y B6 — hoy no
hay un módulo compartido natural. Candidato: `shared/reports.py`. Nótese que los seis
nombres son públicos y los importan `dashboard/` y `docs/`, así que habría que conservarlos
como alias que delegan (igual que se hizo con `_atr`).

## 2. B2 — *Stylized facts* triplicado (era H2)

`synthetic/fidelity.py:154-235` (con tests) frente a `dashboard/build_dashboard.py:226-268`
y `docs/build_docs.py:86-125`, que lo reimplementan inline.

**Evidencia ya reunida, no hace falta más:** ambas implementaciones ejecutadas sobre las
mismas 48 series de `ai_v3` difieren en ≤2,8e-16 (epsilon de float64). Con el redondeo a 3
decimales que aplican los builders, la sustitución daría cifras idénticas.

**Valor: Alto** — son las cifras de fidelidad publicadas; hoy una corrección en el módulo
con tests no llega a los dos artefactos que se leen.
**Riesgo: Medio.** **Confianza: Alta.**

**Qué falta para decidir:** (a) dónde vive el código compartido, y (b) si al unificar se
adopta `MIN_OBSERVATIONS = 200` de `fidelity.py` —que cambia el contrato para librerías
futuras de paths cortos— o se parametriza el umbral para conservar el comportamiento de
hoy. Sobre los datos actuales no cambia nada (0 series descartadas de 48).

## 3. B3 — `_cvar` triplicado: la recompensa del sistema

- `src/ai_trader/scoring/aggregate.py:103-104` — la definición canónica, dentro de
  `aggregate_reward`
- `src/ai_trader/scoring/activity_study.py:104` `_cvar`
- `src/ai_trader/scoring/transfer_study.py:654` `_cvar`

**Evidencia:** las tres calculan `k = max(1, ceil(alpha * n))` y
`sort(arr)[:k].mean()`, idéntico. El docstring de `transfer_study` **lo admite
explícitamente**: «Replica exactamente la aritmética de `aggregate_reward` […] pero sin
construir el dataclass: el bootstrap la llama decenas de miles de veces».

**Esto NO es un descuido, es una decisión consciente y justificada por rendimiento.** Pero
el riesgo de divergencia es real: cambiar la definición de CVaR en `aggregate.py` dejaría a
los dos estudios calculando la recompensa vieja.

**Valor: Alto** (es *la* métrica del sistema). **Riesgo: Alto** (tocarla afecta a lo que
rankea). **Confianza: Alta** en el diagnóstico.

**Qué falta para decidir:** medir si extraer un `cvar()` puro y llamarlo desde los tres
sitios tiene coste de rendimiento apreciable en el bootstrap. Si lo tiene, la alternativa
barata es **no unificar** y añadir un test que compare las tres implementaciones sobre las
mismas entradas, de modo que la divergencia se detecte aunque las copias sigan existiendo.

## 4. B4 — `ROADMAP`: 1.049 líneas de contenido dentro del generador (era H4)

`dashboard/build_dashboard.py:1501-2549`. El 41 % del fichero es una constante de datos que
se consume en un solo sitio (`collect_roadmap()`, línea 1277).

**Valor: Medio** (higiene: separar contenido de código). **Riesgo: Bajo** si se mueve a un
módulo Python hermano; el golden del HTML lo cubre entero. **Confianza: Alta.**

**Qué falta para decidir:** solo tu preferencia. Mover a `dashboard/roadmap.py` es un
movimiento puro. **Pasarlo a JSON/YAML sería ROJO**: crearía un formato de fichero nuevo.

## 5. B5 — Duplicados exactos pequeños

Todos con cuerpo idéntico verificado por AST:

| Función | Copias | Dónde |
|---|---|---|
| `load_capture_report` / `load_ledger` / `load_pool_report` | 3 | `signals/capture.py:252`, `signals/depth.py:289`, `signals/events.py:433` |
| `build_specs` | 2 | `scoring/transfer_study.py:181`, `scoring/weight_study.py:157` |
| `_worker_bars` | 2 | `scoring/validation_study.py:162`, `scoring/weight_study.py:127` |
| `to_utc` / `_to_utc` / `_utc` | 3 | `data/intraday.py:65`, `scoring/baselines.py:306`, `synthetic/fidelity_study.py:102` |
| `_sanitize` | 2 | `observation/features.py:58`, `observation/regime.py:29` |
| `crypto_universe` | 2 | `backtest/session_study.py:764`, `scoring/transfer_study.py:219` |
| `_sma` | 2 | `strategies/mean_reversion.py:26`, `strategies/momentum_crypto.py:26` |
| `open_positions` | 2 | `app/reports.py:41`, `app/runner.py:106` |
| `records` | 2 | `signals/audit.py:238`, `signals/capture.py:125` |
| `days` | 2 | `backtest/validation.py:88`, `scoring/transfer_study.py:285` |

**`_sma` merece una nota:** es el gemelo de `_atr`, que sí se unificó en la Fase 2. No se
tocó porque la aprobación fue para `_atr` en concreto, y porque `_sma` no aparecía en la
auditoría (lo ocultaba el umbral de 12 líneas). Es el candidato más inmediato: mismo
patrón, mismo riesgo, misma prueba de equivalencia ya ensayada.

**Riesgo: Bajo** los que no son cálculo; **`_sma`, `to_utc` y `_sanitize` son ÁMBAR** por
tocar series numéricas. **Confianza: Alta** en que los cuerpos son idénticos.

## 6. B6 — `_git()` duplicado en los dos builders (era H3)

`dashboard/build_dashboard.py:122` y `docs/build_docs.py:68`. Seis líneas. Valor bajo,
riesgo bajo. **Bloqueado por lo mismo que B1 y B2:** no hay sitio común, y `_git` no pinta
nada dentro del paquete de trading.

## 7. B7 — `pytest-cov` sin declarar (era H10, y estaba mal clasificado)

Se clasificó **VERDE** en la auditoría. Al ir a ejecutarlo apareció evidencia que lo
desmiente y se descartó:

- **`poetry.lock` existe y está trackeado** (281 KB). Declarar la dependencia sin
  regenerarlo deja `poetry install` inconsistente: sería una regresión, no una mejora.
- Regenerar el lock con `poetry lock` puede resolver versiones nuevas de **todo** el árbol,
  mucho más de lo aprobado, y `poetry` falla en silencio en este entorno.
- La versión propuesta en la auditoría (`^6.0`) era **incorrecta**: la instalada es 7.1.0.

**Reclasificado a ÁMBAR.** Qué falta: comprobar en un entorno donde poetry funcione que
`poetry lock` no mueve nada más, y commitear `pyproject.toml` + `poetry.lock` juntos.

## 8. B8 — Seis entry points sin red (era H6)

`transfer_study`, `validation_study`, `weight_study`, `activity_study`, `fidelity_study`,
`session_study`. Cobertura entre el 21 % y el 73 %. Producen los informes de `data/` que
consumen el dashboard y la documentación.

**Valor: Alto** — es deuda de *verificación*, no de código: son la fuente de las cifras
publicadas y nadie comprueba que sigan produciendo lo mismo. **Riesgo de tocarlos: Alto.**

**Qué falta:** no refactorizar nada aquí. El paso previo es caracterizarlos, y los JSON ya
publicados en `data/` sirven de golden natural (igual que se hizo con los HTML). Ojo: son
caros de ejecutar, así que irían marcados `veryslow`.

## 9. B9 — Fixture fijo para el archivo de señales

Hoy `signals audit|depth|events|features` solo tienen congelada la parte que sale del
código y del catálogo; los recuentos del archivo se comprueban como invariantes, porque
`data/signals_raw/` es append-only y está gitignoreado.

**Qué falta:** un archivo pequeño y fijo en `tests/fixtures/signals_raw/` y monkeypatch de
`store.DEFAULT_RAW_ROOT` (que se lee en tiempo de llamada, así que el patch funciona). Con
eso esos cuatro comandos se podrían congelar enteros.

## 10. B10 — Bloques de plantilla largos (era H8)

`docs/template.py`: `_signals_block()` 234 loc, `_fidelity_block()` 153, `_transfer_block()`
143, `_validation_block()` 133, `_sessions_block()` 132.

**Valor: Bajo.** Son plantillas lineales; trocearlas suele empeorar la legibilidad porque el
HTML deja de leerse en orden. **Confianza en que sea deuda: Media** — es cuestión de gusto.

---

## No tocar (registrado para que nadie lo "arregle")

| Qué | Por qué |
|---|---|
| Ciclo `scoring.activity` ↔ `scoring.aggregate` | Gestionado a propósito con `TYPE_CHECKING` + import diferido, con el motivo escrito en `aggregate.py:83-85`. Romperlo exige mover `DEFAULT_CVAR_ALPHA`, que define la recompensa y lo consumen 13 módulos |
| `main()` de `cli.py` (185 loc) | Es la frontera pública del CLI: 7 subcomandos y ~40 argumentos. Reorganizarlo arriesga cambiar un `help`, un `default` o un `choices` |
| `stylized_facts`/`survey`, `authorized`/`wrapper` | Falsos positivos del detector: la segunda está **anidada dentro** de la primera |
| `_init_worker` (validation vs weight) | Mismo patrón, parámetros distintos (`purge_days` vs `specs`+`split_ratio`). Unificarlos daría una firma variádica menos legible que las dos copias |
| `__post_init__` de las dos estrategias | Validan parámetros propios de cada una: comparten forma, no contenido |
| `_stablecoin_row` / `_macro_row` | Comparten la forma del contrato `daily_from_raw`, que es justo el patrón que hace barata la fuente N+1 |

---

## Nota de método

Dos fallos de mi instrumentación salieron a la luz haciendo este trabajo, y los dos
producían **subestimación**, nunca falsos positivos:

1. El grep `^def` de la Fase 0 atribuyó a `build()` 1.262 líneas que en realidad eran una
   constante de datos (`build()` tiene 51).
2. El umbral de ≥12 líneas del detector de duplicados de la Fase 1 ocultó 19 de los 21
   grupos idénticos.

Conviene tenerlo presente al leer cualquier cifra de estos documentos: están medidas, pero
la medición tiene supuestos, y aquí dos de ellos fallaron.
