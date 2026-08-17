# Dónde se quedó esto (pausa del 2026-08-16, 01:10)

Árbol limpio, todo commiteado. Se paró para apagar el ordenador, no por un fallo.

## Lo que falta, en orden

### 1. `signal_study` — ya REANUDA, así que una pausa cuesta una unidad

```powershell
.venv\Scripts\python.exe -m ai_trader.scoring.signal_study --library ai_v4 `
    --workers 7 --configs-per-family 8 --verify-determinism
```

Escribe cada unidad a `data\signal_channel\progress_ai_v4.jsonl` según sale, y al arrancar
salta las que ya estén: se puede parar cuando haga falta y el mismo comando continúa. Con
`--no-resume` empieza de cero.

Reanudar es **exacto y no una aproximación**: cada unidad es independiente y determinista, y
las filas se ordenan al final. Comprobado por el camino real de `run_units` sobre un plan de 8
unidades — de un tirón contra correr 4, pausar y reanudar: **idénticas fila a fila**.

Coste total desde cero: **~15 h de reloj** (medido: 271 min para el 30% en el intento anterior).

### 2. `theme_study` para `vol_term_structure` — 40 unidades, ~1,6 h

```powershell
.venv\Scripts\python.exe -m ai_trader.scoring.theme_study --offline --workers 7 `
    --families vol_term_structure --out data\themes\report_vol_term_structure.json
```

Esta familia se excluyó del informe publicado con un motivo **falso** (*"sus fuentes empezaron
a existir el día que arrancó la captura"*): `deribit_volatility` publica desde **2021-03-24** y
el tema alcanza 0,333 de cobertura medida. Correrla sola es legítimo porque el estudio compara
cada familia **consigo misma**, así que ninguna cifra depende de quién más corra.

Espera poca potencia: el tema es legible sólo en el **4,2%** de las sondas, así que lo más
probable es `sin_potencia` — pero medido, que es la diferencia.

### 3. Artefactos, cuando estén 1 y 2

Dashboard, metodología y README con las cifras nuevas, y regold de `dashboard/index.html` y
`docs/metodologia.html` (`$env:AI_TRADER_REGOLD="1"`). Luego `.\scripts\verify.ps1`.

## Lo que ya está medido y commiteado

| estudio | resultado |
|---|---|
| temático (4 familias) | la capa **ayuda** en `signal_composite` y `flow_persistence` |
| validación (16 configs) | corrobora: +1,327 contra la cola; la elección cambia en 6/8 y 7/8 |
| pesos (64 configs) | **revierte** lo publicado: (λ=4, κ=4) estabiliza, +0,0475 ± 0,0189 |
| transferencia + control | ρ = +0,038 sin transferencia; residuo del control, cero |

## Lo que hay que decidir, y no es un arreglo de paso

Una fuente de eventos cuenta como **cubierta** aunque su calendario esté vacío
(`signal_radar.py:445`). En vivo es defendible; hacia atrás significa que en 2019 el tema
`macro` publica cobertura 0,833 con **cuatro de sus cinco fuentes diciendo exactamente nada**,
y que `deribit_expiries` cuenta como cubierta en 2019 con datos que empiezan en 2026. No se
filtra información futura de precios —los valores son cero—, pero la **elegibilidad** sí
depende de fuentes que no existían.

Cambiarlo movería las seis features publicadas y la evidencia congelada. No se ha tocado.

## Un patrón que conviene no repetir

Cinco de cinco estimaciones de coste se quedaron cortas (el temático 384 min contra 210
proyectados; los pesos 680 contra 360). El error siempre es el mismo: medir una unidad aislada
y proyectar sin la contención real de siete workers sobre cuatro núcleos físicos. **Usar el ETA
que imprime la propia corrida**, no la proyección.
