# Instrucciones — generador del reporte diario de señales (ai-trader)

Fecha de ejecución: **2026-08-20**. Zona horaria del usuario: Europe/Madrid (UTC+2).

## Rutas
- Contexto global compartido (LÉELO PRIMERO): `/root/ai-trader/config/contexto_global.md`
- Cuestionario: `/root/ai-trader/config/cuestionario_cripto_v1.json`
- Renderizador: `/root/ai-trader/tools/render_reporte.py`
- Validador: `/root/ai-trader/tools/validar_respuestas.py`
- Carpeta de salida: `/root/ai-trader/data/signals_raw/ai_reports/2026-08-20/`
- Carpeta de datos intermedios: `/root/ai-trader/data/signals_raw/ai_reports/2026-08-20/_datos/`

## Reglas de fuentes (IMPORTANTE)
1. Las **portadas de agregadores** (coinmarketcap.com, cryptorank.io, coingecko.com) devuelven en este
   entorno un snapshot **cacheado y obsoleto** (BTC ~$69.800). **No las uses como fuente de precio.**
   Las APIs de CoinGecko y Binance están bloqueadas por robots.txt: no intentes rodearlo con curl/python.
2. Usa **WebSearch + WebFetch sobre artículos fechados** de 18–20 ago 2026. Fuentes que han funcionado:
   fortune.com/article/price-of-bitcoin-08-19-2026/, finance.yahoo.com, altcoinbuzz.io,
   forbes.com/financial-services/top-10-cryptocurrencies-2/, investingnews.com, cointelegraph.com,
   coindesk.com, theblock.co, ambcrypto.com, bitcoinist.com, u.today, cryptonews.com, beincrypto.com,
   fxstreet.com, tradingview.com/news, bitget/bybit/okx/kraken price pages.
3. **Contrasta al menos 2 fuentes por dato de precio.** Si discrepan, repórtalo como **rango** en el HTML
   (p. ej. «$0,074–0,077 según fuente»). Nunca inventes una cifra intermedia sin decir que es un rango.
4. Si un dato no se encuentra tras búsqueda razonable, **dilo explícitamente en el HTML**
   («no disponible en las fuentes consultadas»). Eso permite responder `sin_datos` en el cuestionario.
   **Prefiere admitir la ausencia antes que estimar.** No inventes cifras on-chain, de whales ni de ETFs.
5. Prohibido incluir recomendaciones de inversión. El reporte es **descriptivo**.

## Investigación mínima por activo
Precio y variación 24h/7d · rango 7d · distancia a máximos (52 semanas / ATH) · volumen 24h y comparación
con la media reciente · capitalización · ranking · dominancia si aplica · oferta circulante/máxima y
emisión/inflación · actividad de red (TVL, transacciones, direcciones activas, staking) · contexto global
(usa el fichero compartido) · on-chain y whales (acumulación/distribución, grandes wallets, flujos de
exchange) · flujos institucionales (ETF/ETP/trust del activo, o indica que no existe) · derivados (interés
abierto, funding, posicionamiento, liquidaciones) · noticias y catalizadores de las últimas 48 h con
etiqueta alcista/bajista/neutral · regulación · análisis técnico (soportes, resistencias, medias 50/100/200,
RSI, patrones) con **tabla de escenarios** · sentimiento social · estacionalidad del mes · riesgos ·
calendario próximo (desbloqueos de tokens, upgrades, votaciones, macro).

## Paso 1 — `_datos/datos_{TICKER}.json`
Esquema exacto (las secciones deben ser 1..13, en este orden y con estos títulos):

```json
{
  "activo": "Dogecoin", "ticker": "DOGE", "par": "DOGE/USDT",
  "fecha": "2026-08-20", "hora_corte": "≈10:00 CEST (datos anclados a 19–20 ago 2026)",
  "senales": [
    {"label": "Sesgo diario", "valor": "Neutral", "tono": "neutral"},
    {"label": "Momentum 24h", "valor": "+1,2%", "tono": "alcista"},
    {"label": "Tendencia 7d", "valor": "Bajista", "tono": "bajista"},
    {"label": "Derivados", "valor": "OI moderado", "tono": "neutral"},
    {"label": "Flujo institucional", "valor": "No aplica", "tono": "neutral"},
    {"label": "Riesgo", "valor": "Medio", "tono": "aviso"}
  ],
  "resumen_cabecera": "2–4 frases con lo esencial del día.",
  "secciones": [
    {"n": 1, "titulo": "Snapshot de mercado",
     "kpis": [{"label": "Precio", "valor": "$0,0751", "nota": "rango $0,0748–0,0755"}],
     "bloques": [
       {"tipo": "parrafo", "texto": "Texto con **negrita** admitida."},
       {"tipo": "tabla", "titulo": "opcional", "headers": ["Métrica", "Valor", "Fuente"],
        "rows": [["Volumen 24h", "$1,2 B", "AltcoinBuzz"]]},
       {"tipo": "lista", "items": [{"tono": "alcista", "texto": "..."},
                                   {"tono": "bajista", "texto": "..."},
                                   {"tono": "neutral", "texto": "..."}]},
       {"tipo": "nota", "texto": "Aclaración o límite del dato."}
     ]}
  ],
  "fuentes": [{"titulo": "Fortune — Bitcoin price 19 ago 2026", "url": "https://..."}]
}
```

Títulos obligatorios de las 13 secciones, en este orden:
1 Snapshot de mercado · 2 Fundamentales de red y tokenomics · 3 Contexto de mercado global ·
4 Actividad on-chain y whales · 5 Flujos institucionales y ETFs · 6 Derivados y posicionamiento ·
7 Noticias y catalizadores (48h) · 8 Marco regulatorio · 9 Análisis técnico ·
10 Sentimiento y estacionalidad · 11 Calendario y eventos próximos · 12 Riesgos principales ·
13 Resumen ejecutivo

Requisitos de contenido:
- Sección 1: al menos 6 KPIs (precio, 24h, 7d, volumen 24h, capitalización, ranking) y una tabla comparando
  al menos 2 fuentes de precio.
- Sección 7: usa `lista` con `tono` alcista/bajista/neutral para cada noticia.
- Sección 9: **obligatoria** una tabla de escenarios con columnas
  `["Escenario", "Disparador", "Zona objetivo", "Probabilidad cualitativa"]` y filas Alcista/Base/Bajista.
- Sección 12: usa `lista`; **el número de ítems debe ser el número real de riesgos** (P28 lo cuenta).
- Sección 13: cierra con el sesgo global del día (muy bajista / bajista / neutral / alcista / muy alcista)
  y una frase que justifique. Debe ser coherente con las secciones anteriores.
- `fuentes`: mínimo 6 URLs reales que hayas abierto de verdad. No inventes URLs.

Renderiza:
```
python3 /root/ai-trader/tools/render_reporte.py \
  /root/ai-trader/data/signals_raw/ai_reports/2026-08-20/_datos/datos_{TICKER}.json \
  /root/ai-trader/data/signals_raw/ai_reports/2026-08-20/reporte_{TICKER}.html
```

## Paso 2 — `respuestas_{TICKER}.json`
Rellena las 30 preguntas **usando exclusivamente el reporte que acabas de generar** (si un dato no está
en el reporte, la respuesta es `sin_datos`; si la métrica no existe para el activo y hay opción
`no_aplica`, usa `no_aplica`). Una sola opción por pregunta.

```json
{
  "schema_version": "1.0",
  "cuestionario": "cuestionario_cripto_v1",
  "activo": "Dogecoin", "ticker": "DOGE", "fecha": "2026-08-20",
  "reporte_fuente": "reporte_DOGE.html",
  "respuestas": {
    "P01": {"id_opcion": "subida_leve", "valor": 1,
            "evidencia": "Sección 1 · Snapshot: variación 24h +1,8%"}
  },
  "puntuacion_agregada": {"suma_valores": 0, "preguntas_respondidas": 0,
                          "preguntas_sin_datos": 0, "media": 0, "interpretacion": "neutral"}
}
```
- `evidencia` debe citar **sección y dato concreto** del reporte (mínimo 8 caracteres). Es auditable.
- Guárdalo en la carpeta de salida (no en `_datos/`).
- Calcula la agregada y **verifícala** con:
```
python3 /root/ai-trader/tools/validar_respuestas.py \
  /root/ai-trader/data/signals_raw/ai_reports/2026-08-20/respuestas_{TICKER}.json --fix
python3 /root/ai-trader/tools/validar_respuestas.py \
  /root/ai-trader/data/signals_raw/ai_reports/2026-08-20/respuestas_{TICKER}.json
```
El segundo comando debe imprimir `OK`. Si imprime `FALLO`, corrige y repite hasta que pase.

## Reglas de reejecución
- No sobrescribas archivos de fechas anteriores.
- Si ya existe `reporte_{TICKER}.html` de hoy, crea `reporte_{TICKER}_v2.html` (y `_v3`…) y lo mismo
  para las respuestas.

## Devolución
Al terminar, responde SOLO con un bloque compacto por activo (sin repetir el contenido del reporte):
`TICKER | precio | 24h | sesgo global | media cuestionario | nº fuentes | datos no obtenidos (lista corta)`
