# Reduccion de deuda tecnica — modo ultraconservador

Rama: `refactor/safe-cleanup`. Punto de retorno: commit `33af4f6` (radar de senales,
810 tests en verde).

**Regla que manda sobre todo lo demas:** ante la duda, no se toca. Si no se puede
demostrar con evidencia que algo no se usa y que el cambio es equivalente, se queda y se
anota en el backlog del final.

---

## 1. La red de seguridad

Un unico comando:

```powershell
.\scripts\verify.ps1          # completo: ruff + suite entera + arbol limpio
.\scripts\verify.ps1 -Fast    # salta los `veryslow` (vuelta corta al iterar)
.\scripts\verify.ps1 -Regold  # regenera las referencias (solo si el cambio es deliberado)
```

Hace tres cosas: **ruff**, **pytest** (incluida la caracterizacion contra
`tests/golden/`) y una comprobacion de que **verificar no ha ensuciado el arbol** —
varios comandos publican informes en `data/`, y si la propia verificacion los modifica
deja de poder distinguirse un cambio del refactor de un efecto de haberlo comprobado.

Estado en el momento de cerrar la Fase 0:

```
=== ruff ===          OK
=== pytest ===        837 passed  (11:37)
=== arbol limpio ===  OK
VERDE  (11:39)
```

### Que habia antes

| | Antes | Despues |
|---|---|---|
| Tests | 810 | **837** |
| Linting | ruff, limpio | igual |
| Script de verificacion | **no habia** | `scripts/verify.ps1` |
| Cobertura global | 76 % (2.392 de 9.925 sentencias sin cubrir) | — |
| `src/ai_trader/cli.py` | **0 %** — 365 sentencias, el punto de entrada principal | **63 %** |
| `dashboard/` y `docs/` | **fuera de la medicion**: ~2.000 lineas | los dos `build()` caracterizados |

### Que se ha anadido

Tests de **caracterizacion**: no dicen lo que el codigo deberia hacer, congelan lo que
hace HOY.

- `tests/test_characterization_cli.py` — el CLI de punta a punta, en proceso.
  Los cuatro backtests sinteticos (corte unico, walk-forward, CPCV, y las dos salidas de
  texto) recorren el pipeline completo: config → barras → estrategias → riesgo →
  ejecucion → metricas → score. Se comprobo que dos pasadas dan un JSON identico byte a
  byte antes de congelarlo.
- `tests/test_characterization_artifacts.py` — los dos generadores de HTML.
- `tests/test_golden_support.py` — tests del propio andamiaje (ver mas abajo por que).
- `tests/golden/` — las referencias.

### Dos decisiones que conviene conocer

**El scrubber tenia un agujero.** La primera version sustituia la metadata volatil por su
literal. El contador de commits vale `"81"`, asi que reescribia las series de precios del
dashboard: `81.32` → `<NCOMMITS>.32`. Un scrubber que se come datos reales convierte el
golden en una red con agujeros justo donde importa. Ahora todo va anclado a su clave
(`"commit_count": "..."`), y `tests/test_golden_support.py` fija ese caso para que no
vuelva.

**No todo se puede congelar, y forzarlo seria peor.** `data/signals_raw/` es append-only
por diseno y esta en el `.gitignore`: cada `signals capture` le anade lineas. Un golden
de la salida entera de `signals audit|depth|events|features` se rompe solo, y un test que
falla por razones ajenas al refactor no es una red — ensena a ignorar el rojo. Para esos
cuatro se parte la salida en dos: lo que viene del **codigo y del catalogo** (el mapeo
simbolo→entidad, el `spec` de normalizacion, tier/cadencia/pit de cada fuente) va a
golden; lo que viene del **archivo** (recuentos, primer y ultimo dia) se comprueba como
invariante. La cobertura del CLI es la misma y los falsos positivos desaparecen.

**Lo que sigue sin cubrirse del CLI, y por que.** De 0 % se pasa a **63 %**; las 135
sentencias restantes son los subcomandos que necesitan red o credenciales —`run-cycle`,
`signals capture`, `synth generate --ai`— y no se pueden ejercitar offline. Caen dentro
del perimetro "sin red" del apartado 2: se pueden leer, no se pueden refactorizar.

`data/synthetic/` tambien esta gitignoreado, pero es distinto: es un artefacto **estatico**
—una vez generado no cambia— y por eso su backtest si se congela. En una maquina sin esa
libreria los casos se **saltan** con instrucciones, en vez de fallar como si fuesen una
regresion.

---

## 2. Perimetro: donde se puede tocar y donde no

La cobertura no es una nota, es el mapa de lo que se puede refactorizar con seguridad.

### Con red — se puede tocar

`backtest/`, `strategies/`, `risk/`, `shared/`, `synthetic/`, el nucleo de `signals/`,
`execution/` salvo Polymarket, y ahora `cli.py`, `dashboard/` y `docs/`.

### Sin red — NO se toca (salvo que se construya la red antes)

Todo lo que necesita red o credenciales para ejercitarse:

| Modulo | Cobertura |
|---|---|
| `data/providers/polymarket_gamma.py` | 22 % |
| `data/market_data.py` | 31 % |
| `data/providers/alpaca.py` | 37 % |
| `data/providers/ccxt_crypto.py` | 37 % |
| `notifications/telegram.py` | 41 % |
| `bots/telegram_bot.py` | 44 % |
| `data/providers/polymarket_clob.py` | 47 % |
| `data/providers/http.py` | 50 % |

### Zona intermedia — estudios caros

`scoring/activity_study.py` (21 %), `scoring/weight_study.py` (32 %),
`scoring/validation_study.py` (53 %), `scoring/transfer_study.py` (70 %),
`backtest/session_study.py` (73 %). Son los generadores de los informes publicados: caros
de ejecutar y poco cubiertos. Tocarlos exige caracterizar antes su informe de salida.

---

## 3. Una de las tres sospechas no se sostiene

El encargo apuntaba a tres cosas: duplicacion, **codigo muerto** y restos de features
antiguas. La segunda se ha medido y no aparece.

Se recorrieron con AST las **671 definiciones de nivel superior** del proyecto
(`src/`, `dashboard/`, `docs/`) y se conto cuantas veces se menciona cada nombre en todo
el repo. **Ninguna aparece una sola vez**, es decir, no hay ni una funcion o clase
huerfana. Bajando el liston a dos menciones solo salen los colectores de los dos
builders — definidos y llamados una vez desde `build()`, que es la forma normal de un
generador, no codigo muerto. `ruff` tampoco reporta imports muertos.

La sonda no es infalible (el `getattr` dinamico, los registries y los entry points la
enganan, y una mencion puede ser un comentario), pero es suficiente para dejar de buscar
ahi: **la deuda de esta base no es codigo huerfano, es duplicacion entre los dos builders
y funciones demasiado grandes.**

---

## 4. Backlog

Candidatos **observados, no confirmados**. Ninguno se toca sin evidencia de que no se usa
y de que el cambio es equivalente.

| # | Candidato | Evidencia hasta ahora | Estado |
|---|---|---|---|
| 1 | Colectores paralelos entre `dashboard/build_dashboard.py` y `docs/build_docs.py` | Ambos definen `_git` y pares `_fidelity`/`collect_fidelity`, `_transfer`, `_activity`, `_validation`, `_sessions`, `_market`, `_trade`, `_signals`. **Ojo:** al leerlos, la duplicacion es MENOR de lo que sugieren los nombres. `_fidelity` y `collect_fidelity` comparten la carga del informe y el aviso, pero proyectan campos distintos (`{synth, coverage}` frente a `{synth_median, coverage_pct, ratio, rank_corr}`) porque uno alimenta prosa y el otro un grafico. Extraible con seguridad: **la carga**, no la proyeccion | Acotado; pendiente de recorrer los 8 pares restantes |
| 1b | `_market()` (docs) vs `collect_market()` (dashboard) | Aqui la duplicacion **si** es real: leen las mismas cuatro fuentes (`CACHE_DIR`, `CCXTCryptoConfig`, `config.runner.symbols`, `DEFAULT_UNIVERSE`) y comparten 7 de 8 campos. docs anade `n_crypto`; dashboard anade `timeout_ms` y `providers`. Un colector comun que devuelva el superconjunto es viable | Mejor candidato de los vistos |
| 2 | `build()` de `build_dashboard.py` | 1.262 lineas en una sola funcion (1290–2552) | Medido |
| 3 | Caracterizar `signals audit\|depth\|events` sobre un archivo fijo | Hoy solo se congela la parte que viene del codigo; con un fixture en `tests/fixtures/signals_raw/` se podria congelar todo | Anotado |
| 4 | `pytest-cov` no esta en `pyproject.toml` | Se instalo a mano en el venv para medir | Decidir si se fija como dependencia de dev |

---

## 5. Nota sobre el tamano

El encargo hablaba de ~10.000 lineas. Lo medido es **22.058 en `src/`** (102 ficheros) mas
**7.924 en `tests/`**, sin contar los ~2.000 de `dashboard/` y `docs/`. No cambia el
metodo, pero si el alcance de lo que queda por revisar.
