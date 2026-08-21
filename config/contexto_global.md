# Contexto global de mercado — 2026-08-21 · **CORTE 06:00Z (08:00 Europe/Madrid)**

> **ESTE FICHERO ES POINT-IN-TIME.** Todo lo que contiene procede de fuentes **publicadas antes de
> las 06:00Z del 21-ago-2026**. Es la base de la sección 3 de todos los reportes de hoy y del
> bloque de contexto de `medidas_{TICKER}.json`.
>
> **No es el estado del mercado a última hora del día, y eso es deliberado.** Una ejecución
> anterior de hoy, con corte ≈10:45Z, vio BTC en $79.241 y $606 M de entradas en ETFs. Esos datos
> **no existían todavía a las 06:00Z** y por tanto no pueden usarse: meterlos sería look-ahead.
> Si encuentras una cifra más alta que las de aquí, comprueba la hora de publicación antes de
> usarla. Casi seguro es posterior al corte.

## Fuentes descartadas por ser posteriores al corte (no las uses hoy)
| Fuente | Publicada | Qué contenía |
|---|---|---|
| CoinDesk live *«Bitcoin, ether ETFs pull in $800 million»* | **06:43Z** | flujos ETF del 20-ago (+$606 M BTC, +$221 M ETH), BTC en $77.010 |
| Cualquier titular de BTC por encima de ≈$75.600 | posterior | la subida a $77.000–79.200 ocurre después del corte |

Si un dato **sólo** aparece en fuentes posteriores a 06:00Z, la respuesta correcta es `sin_datos`.
No es negociable y no se compensa con conocimiento previo.

## Titular del día a las 06:00Z
Tercera sesión consecutiva al alza. El movimiento arrancó el **miércoles 19-ago** con un short
squeeze de ≈$3.000 M tras dos catalizadores simultáneos: la intervención del Tesoro de EE. UU. en
el mercado de bonos y la reunión de Trump con ejecutivos cripto en la Casa Blanca. A las 06:00Z
del viernes el rally **sigue vivo pero desacelerando**: BTC gana ≈+7,5% en 24 h frente al +8,1%
del día anterior, y ETH se enfría con fuerza, de +18,4% a +3,8%.

## Snapshot global (corte 06:00Z)
| Métrica | Valor | Fuente y hora |
|---|---|---|
| BTC | **$74.411 – $75.527** | CoinGabbar 01:30Z ($74.606, +7,6%); Blockhead 04:20Z (máx. $75.527, +7,7%) |
| ETH | **$2.337 – $2.343** | CoinGabbar 01:30Z ($2.343,51, +3,8%); Blockhead 04:20Z ($2.337, +5,0%) |
| ETH 7 días | **≈+25%** | Blockhead 04:20Z. Es el líder de la semana |
| Cap. total cripto | **$2,56 T** (+4,1% 24h) | CoinGabbar 01:30Z. Venía de $2,45 T |
| Volumen 24h global | **$128,7 B** | CoinGabbar 01:30Z |
| Dominancia BTC | **53% – 57,8%** | discrepancia real: Blockhead vía CoinGecko 53%; CoinGabbar 57,8%. **Rango, no promedio** |
| Dominancia ETH | ≈11% | CoinGabbar 01:30Z |
| Altcoin Season Index | **≈30** | bajando desde 36 y 43. **NO es altseason** |
| Cap. stablecoins | $289 B (+0,2%) | CoinGabbar 01:30Z |
| Cap. DeFi | $66,9 B (+4,4%), dominancia 2,6% | CoinGabbar 01:30Z |
| Volumen BTC 24h | ≈$55 B | CoinGabbar / Blockhead |
| Volumen ETH 24h | ≈$24,9 – 33 B | CoinGabbar $24,89 B; Blockhead ≈$33 B |
| Cap. BTC | ≈$1,48 – 1,5 T | ambas fuentes |

**Rotación:** el mercado sube en bloque, pero la beta la lidera BTC. Con la dominancia al alza y
el ASI en 30, las altcoins que suben mucho lo hacen **por arrastre y por cierre de cortos**, no
por demanda propia. Dilo así en los reportes salvo que el activo tenga catalizador verificado.

## Sentimiento
- **Fear & Greed: 72 → CODICIA** (CoinGabbar, corte 01:30Z). Ayer **62**, hace una semana **29
  (Miedo)**, hace un mes **33**.
- `fng_hoy = 72`, `fng_delta_7d = +43` → P06 `codicia`, P07 `mejora_fuerte`.
- Advertencia: 72 roza el territorio eufórico. El propio agregador señala que la codicia extrema
  precede históricamente a las correcciones. El mercado sigue en bear estructural: BTC ≈−40%
  desde el ATH de **$126.198,07 del 6-oct-2025**.

## Derivados (lo conocido a las 06:00Z)
- Liquidaciones del episodio 19–20 ago: **≈$3.000 M, ~92% cortos**, frente a $263,5 M en largos;
  más de $1.000 M barridos en una sola hora. Mayor evento desde 2021.
- **El interés abierto NO se reconstruyó** tras la purga. Es el dato más informativo del día: el
  rally es **cobertura de cortos más flujo spot**, no apalancamiento largo nuevo.
- **Funding contenido**: BTC **0,0101%**, ETH **0,0103%** por 8 h. Muy bajo para la magnitud del
  movimiento → no hay todavía sobrecalentamiento de largos.
- Estos son los únicos valores de funding verificados hoy. Para el resto de activos, búscalos; si
  no aparecen antes del corte, `sin_datos`.

## Flujos institucionales (lo publicado antes del corte)
- **ETFs spot BTC: +$517 M el 19-ago** — mayor entrada diaria desde principios de mayo.
- **ETFs spot ETH: +$189 M el 19-ago** — la mayor desde octubre de 2025.
- **Los flujos del 20-ago no eran públicos a las 06:00Z.** Si no encuentras una fuente publicada
  antes del corte que los recoja, es `sin_datos`, no la cifra que veas en un artículo de las 06:43Z.
- Qué activos tienen producto cotizado está en **`config/assets.json`**, ya verificado. Recuerda:
  `producto_cotizado: "no"` habilita `no_aplica` en P13/P14; `"desconocido"` obliga a `sin_datos`.

## Catalizadores macro y de política (todos anteriores al corte)
- **Casa Blanca, 19-ago:** Trump reúne a Coinbase, Ripple, Robinhood, Kraken e ICE y pide al
  Congreso una «versión justa» de la **CLARITY Act**. Voto de procedimiento en el Senado **en
  septiembre**.
- **Tesoro de EE. UU.:** duplica las recompras de bonos largos de $2 B a **≥$4 B por operación**,
  del **9-sep al 4-nov**. El 30 años cae de 5,337% a ≈5,20%. Bessent sugiere más intervención.
- **CFTC (Michael Selig):** instruye a su equipo para redactar normas cripto **aunque la CLARITY
  Act fracase**, con la autoridad existente.
- **SEC (Paul Atkins):** propuesta «Regulation Crypto Assets» (exención hasta $5 M en 4 años,
  captación hasta $75 M anuales con cuentas auditadas).
- **Fed:** tipos **3,50%–3,75%** sin cambios; actas del 19-ago con **tres disidencias a favor de
  subir 25 pb**. **Jackson Hole a finales de agosto** con Kevin Warsh.
- Riesgo latente: rendimientos globales de bonos en máximos históricos para la vida de BTC, carry
  trade japonés, y el conflicto de Irán sin resolver. El rally se apoya en **dos patas** —yields
  bajos y momentum de la CLARITY Act—: ceder una desmonta el movimiento.

## Calendario macro para P29 (próximos 7 días naturales: 21 – 28 ago)
- **Jackson Hole, finales de agosto**, con comparecencia del presidente de la Fed (Kevin Warsh).
  → cuenta como **evento de alto impacto**. Verifica la fecha exacta antes de puntuar.
- Si no encuentras un segundo evento fechado (IPC, nóminas, FOMC, vencimiento trimestral de
  opciones, BCE o BoJ) dentro de la ventana, la respuesta es **`uno`**, no `varios`.
- **P29 exige citar el evento con su fecha en la evidencia.** Sin fecha verificada → `sin_datos`.
  Ya no existe la opción «no se mencionan»: v1 premiaba la omisión y eso se ha eliminado.

## Noticias sectoriales del 21-ago con impacto por activo (publicadas antes del corte)
- **ARB** — Arbitrum activa **Elara / ArbOS 61** el 20-ago 17:00Z en One y Nova: screening de
  cumplimiento opcional y desactivado por defecto, priority fees, límite Stylus a 96 KB.
  **Anchored** estrena 10 acciones tokenizadas sobre Uniswap vía Arbitrum el **24-ago**.
- **OP** — **gobernanza negativa**: el voto decisivo de un equipo financiado por Optimism desvía
  **$49 M en tokens OP** fuera de los airdrops de usuarios (CoinDesk, 20-ago).
- **UNI** — Hayden Adams: miles de pools activas, **8 con más de $1.000 M de volumen mensual**.
- **XRP** — **Ripple respalda un fondo de crédito RLUSD** en su mejor semana en meses.
- **ETH** — la Ethereum Foundation lanza el **Better Codes Open Challenge** post-cuántico ($1 M).
- **BTC** — el salto sobre $71.000 activó un **golden cross**. Strategy no compró ni vendió BTC la
  semana pasada: levantó $334 M vendiendo acciones MSTR.
- **LTC** — Binance retira el par **LTC/BNB** a las 03:00Z de hoy. No es deslistado del activo.
- **Riesgo idiosincrático vivo:** MANTRA (OM, fuera del universo) se desploma 18% tras un exploit
  con parada de cadena; BounceBit pausa nodos; XCAD Network cierra.
- **Binance** lanza **Agent OS**, plataforma de agentes IA conectados a sus herramientas.

## Estacionalidad
**Agosto** es históricamente uno de los meses más flojos para BTC y para el conjunto (retorno
mediano ligeramente negativo); septiembre el peor; octubre el más fuerte. La sesión actual va
**contra** la estacionalidad, lo que refuerza la lectura de que es un evento de liquidez y
política, no de calendario. Para P27, si no hay estacionalidad publicada del activo concreto, usa
la del mercado y **decláralo** en la evidencia con `origen: "fuente_directa"`.

## Reglas de fuentes (vigentes)
- **Regla point-in-time por encima de todo**: nada publicado después de 06:00Z.
- Contrasta cada precio con **≥2 fuentes**. Si discrepan, **rango**, nunca promedio inventado.
- Atiende a la hora de corte de cada fuente: en sesión direccional, dos cifras muy distintas
  suelen ser **cronología, no error**.
- No uses las APIs de CoinGecko ni de Binance (bloqueadas por robots.txt) ni rodees el bloqueo.
- Datos on-chain, whales, derivados, funding, profundidad de libro y beta **suelen no existir**
  fuera de los grandes. Dilo explícitamente: `motivo_null: "sin_datos"`. **Nunca los estimes.**

## Fuentes del contexto global (todas publicadas antes de 06:00Z del 21-ago)
1. CoinGabbar — *Crypto News Today: Bitcoin Nears $75K, ETH Rises As Fear Index Hits 72*, datos con corte 01:30Z — https://www.coingabbar.com/en/crypto-currency-news/crypto-news-today-bitcoin-ethereum-price-up-fear-index-72
2. Blockhead — *Bitcoin Tops $75,000, Ether Jumps 5% as Rally Continues*, 04:20Z — https://www.blockhead.co/2026/08/21/bitcoin-tops-75-000-ether-jumps-5-as-rally-continues/
3. CoinDesk — *Bitcoin breaks out of six-week range, tops $71,000 as $3 billion in shorts get wiped out*, 20-ago — https://www.coindesk.com/markets/2026/08/20/bitcoin-breaks-out-of-six-week-range-tops-usd71-000-as-usd3-billion-in-shorts-get-wiped-out
4. The Block — *Spot bitcoin ETFs report $517 million in net inflows*, 20-ago — https://www.theblock.co/news/markets/2026-08-20-us-bitcoin-etf-517-million-inflows-412291
5. CoinDesk — *Live updates: Bitcoin extends gains as Bessent suggests more Treasury intervention*, 20-ago — https://www.coindesk.com/tech/2026/08/20/live-updates-bitcoin-etfs-draw-usd517-million-ether-pulls-usd189-million-in-biggest-inflows-in-months
6. CoinDesk — *U.S. CFTC chief puts staff on notice to create crypto regulations if Clarity Act fails*, 20-ago — https://www.coindesk.com/policy/2026/08/20/u-s-cftc-chief-puts-staff-on-notice-to-create-crypto-regulations-if-clarity-act-fails
7. CoinDesk — *Optimism-funded team's deciding vote shifts $49 million in OP tokens away from users*, 20-ago — https://www.coindesk.com/web3/2026/08/20/optimism-funded-team-s-deciding-vote-shifts-usd49-million-in-op-tokens-away-from-users
8. FXStreet — *Top altcoins price forecast: Ripple, Solana, Cardano*, 20-ago 05:03Z — https://www.fxstreet.com/cryptocurrencies/news/top-altcoins-price-forecast-ripple-rallies-above-1-solana-eyes-85-cardano-eases-gains-202608200503
