# Instrucciones — generador del reporte diario de señales (ai-trader) · **v2**

> **Qué cambia respecto a la v1 y por qué.** En v1 el agente escribía el reporte y después
> respondía el cuestionario *«usando exclusivamente el reporte que acababa de generar»*. Eso no
> medía el mercado: medía al redactor. El cuestionario entrevistaba al generador, y varias
> preguntas acababan siendo funciones deterministas de la prosa (P30 era la conclusión del
> propio reporte; P28 contaba ítems de una lista que escribía el mismo agente, o sea verbosidad;
> P29 premiaba con +1 que el reporte **omitiese** los eventos macro).
>
> **v2 invierte la causalidad: primero se mide, después se narra.** Las dos salidas —el reporte y
> el cuestionario— se derivan de la MISMA captura numérica, hecha antes de escribir una sola
> frase. Y el cuestionario ya **no** está limitado a lo que el reporte cuente: puede usar
> cualquier fuente fechada y anterior a la hora de corte, declarándolo.

---

## 0 · Lo primero de todo: fija la HORA DE CORTE

Antes de buscar nada, declara `hora_corte_utc` en ISO-8601 y **escríbela en todos los ficheros**.
Por defecto, para la tarea programada: **08:00 Europe/Madrid = `06:00Z`** del día en curso.

A partir de ahí rige la **regla point-in-time**, que es la más importante de este documento:

> **Se descarta toda fuente publicada después de la hora de corte, aunque el buscador la
> devuelva primero y aunque contenga el dato que buscas.**

Por qué: un artículo publicado a las 19:00 puede contener el cierre del día. Usarlo para
«predecir» ese día es contaminación de look-ahead. Produce un backtest espectacular y un live
plano, **no se detecta a posteriori** y no se puede reparar después. Por eso cada fuente lleva
`fecha_publicacion` y cada respuesta lleva `fuente_ts`: para que el filtro se pueda volver a
aplicar en el entrenamiento aunque hoy se cuele algo.

Si un dato **sólo** existe en fuentes posteriores al corte: es `sin_datos`. No es negociable.

## Rutas y ficheros de entrada

| Qué | Dónde |
|---|---|
| Universo de activos (**manda**) | `config/assets.json` |
| Cuestionario | `config/cuestionario_cripto_v2.json` |
| Ejemplo comentado de respuestas | `config/plantilla_respuestas_v2.json` |
| Esquema de etiquetas | `config/esquema_etiquetas.json` |
| Plantilla HTML (CSS exacto) | `config/plantilla_reporte.html` |
| Contexto global compartido | `config/contexto_global.md` |
| Renderizador (si hay Python) | `tools/render_reporte.py` |
| Validador v2 (si hay Python) | `tools/validar_respuestas_v2.py` |
| Carpeta de salida | `data/signals_raw/ai_reports/{FECHA}/` |
| Capturas numéricas intermedias | `data/signals_raw/ai_reports/{FECHA}/_medidas/` |

**La lista de activos NO está en este documento.** Está en `config/assets.json` y ese fichero
manda: para añadir o quitar un activo se edita ahí y no se toca ningún prompt. `assets.json`
gobierna además dos decisiones del cuestionario:

- `producto_cotizado`: `"no"` habilita `no_aplica` en P13/P14. `"desconocido"` **no** lo habilita
  — obliga a `sin_datos`.
- `desbloqueos_programados`: `"no"` habilita `no_aplica` en P36.

Si el fichero no existe, **para y dilo**. No reconstruyas la lista de memoria.

## Si no hay Python en la sesión

`poetry run` está roto en este proyecto y, en algunos dispositivos, el sandbox Linux entero no
arranca (`Workspace unavailable`). Comprueba si tienes shell **antes de planificar**:

- **Con Python:** usa `.venv\Scripts\python.exe` (nunca `poetry run`) para `render_reporte.py` y
  `validar_respuestas_v2.py`.
- **Sin Python:** escribe el HTML a mano replicando `config/plantilla_reporte.html`, y **valida a
  mano** siguiendo la checklist de la sección 6. Dilo en la primera parte del `_log.txt`.

---

## 1 · MEDIR (antes de escribir nada)

Para cada ticker, captura `{FECHA}/_medidas/medidas_{TICKER}.json`. **Este fichero se escribe
antes que el HTML.** Es lo que hace verificable que se midió primero: sin él, nada distingue
«medí y luego narré» de «narré y luego rellené los números para que cuadraran».

```json
{
  "ticker": "DOGE", "fecha": "2026-08-21", "hora_corte_utc": "2026-08-21T06:00:00Z",
  "ancla": {
    "precio_usd": 0.0751, "ts_utc": "2026-08-21T06:00:00Z",
    "fuente": "CoinDesk", "url": "https://...", "exchange_ref": "binance", "par": "DOGEUSDT",
    "precio_rango_usd": [0.0748, 0.0755], "n_fuentes_contrastadas": 2
  },
  "metricas": {
    "ret_24h_pct":            {"v": 7.67,  "u": "%",     "fuente": "CoinDesk", "url": "https://...", "ts": "2026-08-21T05:40:00Z"},
    "ret_7d_pct":             {"v": null,  "u": "%",     "motivo_null": "sin_datos"},
    "exceso_7d_pp":           {"v": null,  "u": "pp",    "motivo_null": "sin_datos"},
    "dist_max_90d_pct":       {"v": -12.4, "u": "%",     "fuente": "...", "url": "...", "ts": "..."},
    "vol_24h_usd":            {"v": null,  "u": "USD",   "motivo_null": "sin_datos"},
    "vol_24h_sobre_media_30d":{"v": null,  "u": "ratio", "motivo_null": "sin_datos"},
    "cap_usd":                {"v": null,  "u": "USD",   "motivo_null": "sin_datos"},
    "ranking":                {"v": null,  "u": "puesto","motivo_null": "sin_datos"},
    "btc_ret_24h_pct":        {"v": 7.67,  "u": "%",     "fuente": "...", "url": "...", "ts": "..."},
    "cap_total_ret_24h_pct":  {"v": 4.10,  "u": "%",     "fuente": "...", "url": "...", "ts": "..."},
    "fng_hoy":                {"v": 72,    "u": "idx",   "fuente": "...", "url": "...", "ts": "..."},
    "fng_delta_7d":           {"v": 43,    "u": "idx",   "fuente": "...", "url": "...", "ts": "..."},
    "netflow_exchange_usd_24h":     {"v": null, "u": "USD", "motivo_null": "sin_datos"},
    "flujo_producto_cotizado_usd":  {"v": null, "u": "USD", "motivo_null": "no_aplica"},
    "oi_usd":                 {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "oi_sobre_media_30d":     {"v": null, "u": "ratio", "motivo_null": "sin_datos"},
    "long_short_ratio_top_traders": {"v": null, "u": "ratio", "motivo_null": "sin_datos"},
    "liq_neta_usd_24h":       {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "soporte_clave_usd":      {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "resistencia_clave_usd":  {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "pos_en_rango_pct":       {"v": null, "u": "%",     "motivo_null": "sin_datos"},
    "sma50_usd":              {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "sma100_usd":             {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "sma200_usd":             {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "n_medias_por_debajo_del_precio": {"v": null, "u": "entero", "motivo_null": "sin_datos"},
    "rsi_14_diario":          {"v": null, "u": "idx",   "motivo_null": "sin_datos"},
    "retorno_mediano_mes_historico_pct": {"v": null, "u": "%", "motivo_null": "sin_datos"},
    "n_eventos_macro_7d":     {"v": 1,    "u": "entero","fuente": "...", "url": "...", "ts": "..."},
    "funding_8h_pct":         {"v": null, "u": "%/8h",  "motivo_null": "sin_datos"},
    "vol_realizada_14d_anual_pct": {"v": null, "u": "%", "motivo_null": "sin_datos"},
    "iv_1m_pct":              {"v": null, "u": "%",     "motivo_null": "no_aplica"},
    "profundidad_1pct_usd":   {"v": null, "u": "USD",   "motivo_null": "sin_datos"},
    "spread_bps":             {"v": null, "u": "bps",   "motivo_null": "sin_datos"},
    "beta_btc_30d":           {"v": null, "u": "beta",  "motivo_null": "sin_datos"},
    "desbloqueo_14d_pct_float": {"v": 0.0, "u": "%",    "fuente": "...", "url": "...", "ts": "..."},
    "emision_neta_anual_pct": {"v": null, "u": "%",     "motivo_null": "sin_datos"}
  },
  "banderas_riesgo": {
    "_comentario": "Checklist CERRADO de P28. true/false sólo si está verificado; null si no se pudo comprobar.",
    "desbloqueo_1pct_14d": false, "regulatorio_adverso_pendiente": null,
    "funding_extremo": null, "oi_extremo": null, "incidente_seguridad_7d": false,
    "bajo_soporte_clave": null, "macro_alto_impacto_7d": true, "profundidad_baja": null
  }
}
```

Reglas de la captura:

1. **`motivo_null` sólo puede ser `"sin_datos"` o `"no_aplica"`.** `sin_datos` = la métrica existe
   para este activo pero no se encontró. `no_aplica` = no existe (sin producto cotizado, sin
   derivados, sin opciones, oferta ya circulante). Confundirlos mete un cero falso donde había un
   hueco, y son cosas opuestas.
2. **Nunca estimes un número que no has encontrado.** Preferir el hueco a la estimación no es
   pulcritud: un valor inventado es indistinguible de uno medido una vez está en el fichero.
3. **Contrasta cada precio con ≥2 fuentes.** Si discrepan, `precio_rango_usd`, nunca un promedio.
   Y mira la **hora de corte** de cada una: en una sesión direccional, dos cifras separadas por
   varios puntos porcentuales suelen ser cronología, no error.
4. **No uses las APIs de CoinGecko ni de Binance** (bloqueadas por robots.txt) ni intentes rodear
   el bloqueo con curl o Python.
5. Las métricas duras nuevas (`funding_8h_pct`, `vol_realizada_14d_anual_pct`,
   `profundidad_1pct_usd`, `spread_bps`, `beta_btc_30d`, `iv_1m_pct`) son **best-effort desde
   web**: Coinglass, TradingView, páginas de derivados de los exchanges. Se espera que falten a
   menudo, sobre todo fuera de los grandes. **Eso está previsto y es preferible al invento**, pero
   anótalo: los huecos no serán aleatorios (faltarán más en small caps y en días volátiles), así
   que sesgan el dataset justo donde más importa. Dilo en el log.
6. `beta_btc_30d` se puede **estimar** con la correlación de retornos diarios de 30 sesiones si no
   hay cifra publicada. Es la única estimación permitida, y hay que declararla en `derivacion`.

## 2 · NARRAR: `reporte_{TICKER}.html`

Se escribe **a partir de `medidas_{TICKER}.json`**, no al revés. Ruta:
`data/signals_raw/ai_reports/{FECHA}/reporte_{TICKER}.html`.

HTML autocontenido, CSS inline de `config/plantilla_reporte.html`, sin dependencias externas.
Cabecera con franja de 6 señales y `lede`; **13 secciones** numeradas con estos títulos exactos:

1 Snapshot de mercado · 2 Fundamentales de red y tokenomics · 3 Contexto de mercado global ·
4 Actividad on-chain y whales · 5 Flujos institucionales y ETFs · 6 Derivados y posicionamiento ·
7 Noticias y catalizadores (48h) · 8 Marco regulatorio · 9 Análisis técnico ·
10 Sentimiento y estacionalidad · 11 Calendario y eventos próximos · 12 Riesgos principales ·
13 Resumen ejecutivo

Requisitos de contenido:

- **Sección 1:** ≥6 KPIs y una tabla comparando ≥2 fuentes de precio **con su hora de corte por
  fila**. El ancla numérica aparece explícita.
- **Sección 6:** incluye funding, volatilidad realizada, profundidad/spread y beta si se
  capturaron. Si no, dilo: «no disponible en las fuentes consultadas».
- **Sección 7:** `<ul class='tl'>` con tono alcista/bajista/neutral por noticia, y **la fecha de
  publicación de cada una**. Las posteriores al corte no entran.
- **Sección 9:** obligatoria la tabla de escenarios
  `Escenario | Disparador | Zona objetivo | Probabilidad cualitativa`, filas Alcista/Base/Bajista.
- **Sección 11:** desbloqueos de tokens con fecha y % del float, y calendario macro de 7 días.
- **Sección 12:** lista de riesgos en prosa **libre**. Ojo: en v2 **P28 ya no cuenta estos
  ítems**, cuenta las banderas tipificadas de `medidas_*.json`. Escribe los riesgos que haya, sin
  pensar en la puntuación.
- **Sección 13:** cierra con el sesgo global del día. Es P30, que en v2 es **benchmark, no
  feature**: ya no contamina el score.
- **Fuentes:** ≥6 URLs reales abiertas de verdad, **cada una con su fecha de publicación**.
- El reporte es **DESCRIPTIVO**. Prohibidas las recomendaciones de inversión.

## 3 · RESPONDER: `respuestas_{TICKER}.json`

37 preguntas en total, repartidas así: **`respuestas` lleva 36 entradas — `P01`..`P37` excepto
`P30` — y `P30` va sola en el bloque `benchmark_llm`.** El validador espera exactamente eso e
imprime «36 respuestas». Forma exacta en `config/plantilla_respuestas_v2.json`.

- Las preguntas con `origen_esperado: "medicion"` se responden **aplicando la `derivacion` del
  cuestionario al número de `medidas_*.json`**, mecánicamente. Si el número existe, la opción está
  determinada: no hay juicio que ejercer y no se mira el HTML.
- Las de `origen_esperado: "reporte"` son juicio narrativo y salen del HTML.
- Las de `origen_esperado: "cualquiera"` salen de donde haya dato. **Puedes usar fuentes que no
  estén en el reporte**, siempre que estén fechadas y sean anteriores al corte. Declara
  `origen: "fuente_directa"` y rellena `fuente_ts`. Esto es lo que rompe la circularidad.
- Cada respuesta lleva `valor_crudo` + `unidad` siempre que la pregunta tenga `metrica_cruda` y
  `estado = "medido"`. La categoría **no sustituye** al número: se guardan los dos, porque
  «más de +5%» agrupa un +5% con un +40% y en cripto esa diferencia lo es todo.
- `estado` ∈ {`medido`, `sin_datos`, `no_aplica`} y `disponible` ∈ {0,1} son la máscara. Ambos
  `sin_datos` y `no_aplica` llevan `valor: null`.
- **P30 va en el bloque `benchmark_llm`**, no en `respuestas`, y no suma.
- `puntuacion_agregada` suma **sólo** las preguntas con `sumable: true` (29 de 37) y lleva
  `usar_en_entrenamiento: false`. Es para leer el día de un vistazo, no una feature.
- Rellena `cobertura`: un activo con 8 de 29 sumables disponibles no es comparable con uno que
  tenga 29 de 29, y sin ese dato el ranking del día miente.

## 4 · Artefactos del día (además de los 2 por activo)

- **`etiquetas_{FECHA}.json`** — una entrada por ticker con `ancla_precio_usd` y `ancla_ts_utc`
  copiados, y **todos los campos de resultado a `null`**. Esquema en
  `config/esquema_etiquetas.json`. El proceso que los rellena a T+14 todavía no existe; el
  fichero se escribe igualmente, porque lo que fija el contrato y salva el día es el ancla.
- **`_resumen.json`** — tabla del día con, por ticker: media, cobertura, sesgo P30 y el
  **rango percentil por sección transversal** de la media entre los activos del día. Para
  largo/corto lo que importa es el ranking relativo entre activos de la misma fecha, no el score
  absoluto de uno.
- **`_log.txt`** — ver sección 6.

## 5 · Reglas de reejecución

- **No sobrescribas nada de fechas anteriores.** Nunca.
- Si ya existe un fichero de hoy con ese nombre, sufijo `_v2`, `_v3`… El sufijo se aplica a
  reporte, respuestas y medidas **a la vez**, para que el trío siga emparejado.
- Si un activo falla, registra el error en el log y **continúa con el siguiente**.

## 6 · Validar y registrar

Con Python: `.venv\Scripts\python.exe tools/validar_respuestas_v2.py <fichero> --fix` y luego sin
`--fix` hasta que imprima `OK`.

Sin Python, comprueba a mano, por fichero:

- [ ] 36 entradas en `respuestas` (`P01`..`P37` sin `P30`), en orden, más `benchmark_llm.P30`.
- [ ] Cada `id_opcion` existe en su pregunta; cada `valor` coincide con el del cuestionario.
- [ ] `estado` coherente con `id_opcion`; `disponible` coherente con `estado`.
- [ ] `valor_crudo` presente en toda pregunta con `metrica_cruda` y `estado = medido`.
- [ ] `no_aplica` sólo donde `assets.json` lo respalda.
- [ ] `fuente_ts` ≤ `hora_corte_utc` en todas las respuestas que lo lleven.
- [ ] `ancla.precio_usd` es un número, con `ts_utc` y `n_fuentes_contrastadas ≥ 2`.
- [ ] Agregada recalculada dos veces: suma sólo de `sumable: true`, media a 4 decimales,
      interpretación por los umbrales del cuestionario.
- [ ] HTML: 13 secciones en orden, tabla de escenarios, ≥6 fuentes con fecha.

El **`_log.txt`** empieza con `MODO: disco` o `MODO: nube`, y con una línea que diga si hubo
Python o no. Después: hora de ejecución, **hora de corte**, activos procesados, fuentes
principales, datos no obtenidos por activo, discrepancias reportadas como rango, fuentes
descartadas por ser posteriores al corte, incidencias, y la tabla resumen (ticker, sesgo,
media, cobertura).

## 7 · Devolución

Un bloque compacto por activo, sin repetir el contenido del reporte:

`TICKER | precio ancla | 24h | sesgo P30 | media | cobertura sumables | nº fuentes | datos no obtenidos`

Y una línea final `INCIDENCIAS:`.

---

## Lo que este pipeline NO promete

El cuestionario no mide el mercado: mide lo que un conjunto de fuentes públicas decía del mercado
a una hora concreta. v2 arregla la circularidad más gruesa y hace el dato reconstruible, pero
quedan tres cosas que ningún cambio de esquema resuelve y que conviene tener presentes antes de
entrenar nada con esto:

1. **Acuerdo entre anotadores desconocido.** Con temperatura no nula y búsqueda web no
   determinista, dos ejecuciones del mismo día dan respuestas distintas. Pasa el mismo HTML por el
   cuestionario cinco veces y mide el acuerdo (alfa de Krippendorff) por pregunta: ese número es
   el techo de señal de cada columna. Espera que las de juicio narrativo (P08, P19, P24, P26)
   salgan mal y las duras (P01, P05) muy bien.
2. **N efectivo minúsculo.** Horizonte de 14 días con observación diaria significa ventanas
   solapadas: 250 días por activo son ~18 periodos independientes. Contra eso hay 29 variables
   sumables fuertemente correlacionadas y uno o dos regímenes de mercado en todo el año.
3. **El baseline barato.** Antes de dar por buenas 37 preguntas de LLM, entrena con sólo
   `[ret_24h, ret_7d, funding, vol_realizada]` —exactos, gratis y point-in-time. Si el cuestionario
   no bate eso de forma robusta, el pipeline no se paga.
