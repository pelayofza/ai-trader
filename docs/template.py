"""Render de la documentacion funcional: HTML autocontenido, print-optimizado (-> PDF)."""
from __future__ import annotations

CSS = r"""
:root{
  --ink:#141414; --ink2:#3f3f3d; --muted:#6c6a64; --line:#d9d8d1; --soft:#f4f3ee;
  --surface:#ffffff; --accent:#1c5cab; --good:#0a7a0a; --bad:#b02a2a; --warn:#9a6a00;
}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;background:#eceae3;color:var(--ink);
  font-family:"Iowan Old Style",Georgia,"Times New Roman",serif;font-size:16px;line-height:1.62}
.page{max-width:820px;margin:26px auto;background:var(--surface);padding:54px 64px;
  box-shadow:0 1px 20px rgba(0,0,0,.10)}
h1,h2,h3{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.25;color:var(--ink)}
h1{font-size:30px;margin:0 0 6px}
h2{font-size:22px;margin:40px 0 10px;padding-top:8px;border-top:2px solid var(--ink)}
h3{font-size:17px;margin:24px 0 6px}
h4{font-family:system-ui,sans-serif;font-size:15px;margin:16px 0 4px;color:var(--ink2)}
p{margin:0 0 12px}
ul,ol{margin:0 0 12px;padding-left:22px}
li{margin:4px 0}
a{color:var(--accent);text-decoration:none}
.sub{color:var(--muted);font-size:15px;font-family:system-ui,sans-serif}
.meta{color:var(--muted);font-size:13px;font-family:system-ui,sans-serif;margin-top:14px;
  border-top:1px solid var(--line);padding-top:10px}
.lead{font-size:17px;color:var(--ink2)}
.why{background:var(--soft);border-left:3px solid var(--accent);padding:10px 16px;margin:12px 0;
  font-size:15px}
.why b{font-family:system-ui,sans-serif}
.note{background:#fbf6e8;border:1px solid #e6d6a8;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:14.5px}
.formula{background:#0e0e0e;color:#eee;font-family:ui-monospace,Consolas,monospace;
  padding:12px 16px;border-radius:6px;font-size:14px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:12px 0;font-family:system-ui,sans-serif;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--soft);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:13px}
.toc{font-family:system-ui,sans-serif;font-size:14.5px;background:var(--soft);border:1px solid var(--line);
  border-radius:8px;padding:16px 22px;margin:20px 0}
.toc ol{margin:0;padding-left:22px}
.toc a{color:var(--ink2)}
.tag{font-family:system-ui,sans-serif;font-size:12px;color:var(--muted)}
.ok{color:var(--good);font-weight:600}.pend{color:var(--warn);font-weight:600}
figure{margin:14px 0}
.cap{font-family:system-ui,sans-serif;font-size:12.5px;color:var(--muted);margin-top:4px}
@media(prefers-color-scheme:dark){
  body{background:#111;color:#e9e8e2}
  .page{background:#191919;box-shadow:none}
  :root{--ink:#f0efe9;--ink2:#cfcec6;--muted:#9a988f;--line:#333;--soft:#232320;--surface:#191919;--accent:#7fb0ee}
  th{color:#9a988f}
}
@media print{
  @page{size:A4;margin:17mm}
  body{background:#fff;color:#000;font-size:10.6pt}
  .page{box-shadow:none;margin:0;max-width:none;padding:0}
  h2{break-before:page;border-top:none}
  h1,h2,h3,h4{break-after:avoid}
  table,figure,.why,.formula,.note{break-inside:avoid}
  .noprint{display:none}
  a{color:#000}
}
.noprint{font-family:system-ui,sans-serif}
.printbtn{position:fixed;top:16px;right:16px;background:var(--accent);color:#fff;border:0;
  border-radius:8px;padding:9px 15px;font-size:14px;cursor:pointer}
"""


def _rows(pairs, headers):
    h = "".join(f"<th{' class=n' if i else ''}>{c}</th>" for i, c in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(
            f"<td{' class=n mono' if i else ' class=mono'}>{c}</td>" for i, c in enumerate(r)
        ) + "</tr>"
        for r in pairs
    )
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _n(value, decimals=2):
    """Numero con coma decimal (el documento esta en castellano)."""
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def _pc(value, decimals=2):
    """Fraccion -> porcentaje con coma decimal."""
    return "—" if value is None else _n(100.0 * float(value), decimals) + " %"


def _ratio(numerator, denominator):
    if not numerator or not denominator:
        return None
    return float(numerator) / float(denominator)


PIT_LABEL = {
    "forward_capture": "solo hacia adelante",
    "archive_revisable": "backfill revisable",
    "derived_from_price": "derivada del precio",
    "chain_immutable": "registro inmutable",
}


def _dat_section(dat, table):
    """
    La fuente COMPUESTA: el mNAV de las tesorerias cotizadas.

    Es la unica del catalogo sin proveedor —no hay API de mNAV— y la unica cuyo resultado
    principal, hoy, es que NO se puede componer la distribucion que pretendia publicar. Eso
    se escribe con el mismo detalle que se habria escrito un exito: la metodologia no
    documenta solo lo que salio.
    """
    if not dat.get("companies_examined"):
        return ""
    policy = dat.get("policy") or {}
    assets = dat.get("assets") or {}
    lag = dat.get("median_lag_days")
    return f"""
<h4>Tesorerías cotizadas: la única fuente <i>compuesta</i>, y lo que midió</h4>
<p>Una tesorería cotizada es una empresa cuyo balance <b>es</b> un tesoro de cripto. Por encima de 1× de
mNAV la máquina va hacia adelante —emitir acciones es acretivo, se emite y se compra más— y por
<b>debajo</b> va en reversa, y es aritmética y no sentimiento: emitir diluye, así que la vía barata para
levantar caja pasa a ser <b>vender el tesoro</b>. Lo interesante no es el mNAV de ninguna compañía sino la
<b>distribución</b> por activo subyacente: la fracción por debajo de 1× es oferta futura estructural sobre
ese activo, y la distancia de la mediana a esa frontera dice cuánto falta para que la cola engorde. No hay
API —los cuatro sitios que lo publican son cuadros de mando— así que la serie se <b>compone</b> de tres
patas: tenencias y acciones del XBRL de la SEC, más el precio de la acción y el del activo <i>del mismo
cierre de sesión</i>, porque el mNAV es un cociente y mezclar un cierre de cripto (medianoche UTC) con uno
de bolsa (21:00 UTC) mete nueve horas de desfase dentro de él.</p>

<div class="why"><b>Y el resultado de componerla no es el que se esperaba.</b> De
<b>{dat.get('companies_examined', 0)} declarantes</b> de cripto en el registro XBRL quedan
<b>{dat.get('companies', 0)} compañías</b> publicables, repartidas en {len(assets)} activos distintos. Con
una cohorte mínima de {policy.get('min_cohort', 3)} para que la palabra «distribución» signifique algo,
<b>hoy no hay ninguna distribución que publicar</b> y la fuente produce {dat.get('rows', 0)} filas. Eso es
la medición, no un pendiente: el adaptador existe para que el hueco esté <b>fechado y desglosado</b> en vez
de leerse como «nadie lo ha intentado».</div>

<p>La cohorte se define con <b>tres filtros y ninguno mira el mNAV</b>, porque definirla con el propio mNAV
truncaría justo la cola que se publica: fuera los SIC {" y ".join(policy.get("excluded_sic", []))} —los
trusts al contado, que crean y redimen <i>al</i> NAV y por tanto tienen el mNAV clavado en 1 por arbitraje,
y los brokers que custodian cripto de sus clientes—; dentro solo si el tesoro pasa del
<b>{_pc(policy.get('treasury_min_asset_share'), 0)}</b> del activo total del balance, que es lo que separa
una tesorería de una minera con algo de cripto; y solo si se sabe <b>qué activo</b> tiene.</p>

<p>Lo tercero es lo que más cuesta, y la primera versión estaba mal. Identificar por el <b>precio
implícito</b> —valor razonable dividido entre unidades— y quedarse con el único activo del universo que
cuadrase producía <b>dos falsos positivos de ocho</b>: TON Strategy Co (Toncoin, ~1,60 $) salía NEAR e
Hyperion DeFi (HYPE) salía LTC. El fallo no es la tolerancia: es que «el único que cuadra» solo significa
algo si el conjunto de candidatos está <b>completo</b>, y hay miles de tokens frente a veinticuatro. Hoy
<b>identifica un nombre</b> —la etiqueta de unidad o la razón social del emisor— y el precio implícito solo
<b>verifica</b>, con tolerancia ×{_n(policy.get('unit_price_tolerance', 0), 2)}. Eso atrapa los dos errores
que ninguna otra regla ve y que no darían ningún error por sí solos: CleanSpark declarando 1.719.000
unidades etiquetadas <span class="mono">Bitcoin</span> que valen 58,53 $ cada una —son 1.719, con un error
de escala de mil en el propio <i>filing</i>— y Bit Digital con un valor razonable que cubre toda su cartera
y un cociente que no es el precio de nada.</p>

{table}

<div class="why"><b>El hueco mayor está declarado y no se parchea.</b> Las APIs XBRL de la SEC solo exponen
hechos <b>sin dimensiones</b>. Una compañía con varias clases de acción etiqueta su recuento por clase, así
que el hecho no existe sin dimensión y <b>la tesorería más grande que existe queda fuera</b> —comprobado:
su <span class="mono">companyfacts</span> tiene un solo <i>tag</i> en el espacio <span class="mono">dei</span>—.
No se sustituye por la media ponderada del periodo, que sí está: es una <b>media</b>, y estas compañías
emiten acciones contra el mercado todos los días, así que subestimaría el recuento justo en las más activas
y las empujaría hacia la cola inferior, que es la parte de la distribución que la señal mide.</div>

<p>El retraso de publicación no se supone: cada hecho trae la fecha a la que se refiere y la fecha en que
se publicó, la fila se fecha en <b>la de publicación</b> —igual que el COT se fecha el viernes y no el
martes— y el desfase realizado es de <b>{_n(lag, 0) if lag is not None else '—'} días</b> de mediana, medido
y declarado en el catálogo. Sin eso, un <i>backtest</i> usaría el 30 de junio una tenencia que no se publicó
hasta agosto, y no daría ningún error. El <b>N</b> de esta fuente tampoco lo dan los eventos de una compañía
—cada una publica cuatro veces al año— sino el <i>pooling</i> sobre la cohorte:
<b>{dat.get('pooled_observations', 0)} observaciones de compañía</b>, que va publicado al lado del número de
compañías porque doscientas observaciones de tres y doscientas de cuarenta sostienen inferencias
distintas.</p>
"""


def _signals_block(s):
    """Seccion 2.2: la plataforma de ingesta de senales externas.

    Va en el capitulo de DATOS, detras de la captura de precio y delante del generador
    sintetico, porque es exactamente eso: el segundo sustrato de datos del sistema. Y va
    con las dos cifras juntas por delante —cuantas conectadas y cuantas con profundidad
    MEDIDA— porque la diferencia entre las dos es el contenido: tener adaptador no es
    tener historia, y solo lo segundo permite backtestear."""
    if not s:
        return ""

    m, e, n = s["summary"], s["entities"], s["normalization"]
    radar, events = s.get("radar", {}), s.get("events", {})
    spec = events.get("spec", {})
    liquidity, maps = s.get("liquidity", {}), s.get("price_maps", {})
    dat = s.get("dat", {})
    rows = [
        (key, enc, PIT_LABEL.get(pit, pit), nf, conn, hist, measured, days, adv)
        for key, enc, pit, nf, conn, hist, measured, days, adv in s["rows"]
    ]
    table = _rows(
        rows,
        ["Fuente", "Codificación", "Point-in-time", "Features", "Adaptador", "Declarada",
         "MEDIDA", "Días", "ADV típico"],
    )
    pool_table = _rows(
        events.get("rows", []),
        ["Fuente de evento", "Eventos pooled", "Entidades", "Anunciado", "Primer día",
         "Último día"],
    ) if events.get("rows") else ""
    map_table = _rows(
        maps.get("rows", []),
        ["Mapa de precios", "Fotos", "Entidades", "Primer día", "Último día"],
    ) if maps.get("rows") else ""
    adv_table = _rows(
        liquidity.get("rows", []),
        ["Venue", "Entidades", "Con volumen", "Mediana 24 h", "Decil inferior", "Máximo"],
    ) if liquidity.get("rows") else ""
    dat_table = _rows(
        dat.get("rows", []),
        ["Motivo por el que la compañía no entra", "Compañías"],
    ) if dat.get("rows") else ""
    return f"""
<h3>2.2 · Señales externas: {m['n_sources']} fuentes, una sola vía hasta la decisión</h3>
<p>La captura de §2.1 produce <b>precio y volumen</b>, y durante toda la vida del proyecto eso fue todo lo
que el sistema podía ver: su único canal de contexto era el bloque de régimen (§4.2), construido sobre las
propias barras. Éste es el segundo sustrato de datos, y el único que trae información de fuera del precio.
El catálogo declara {m['n_sources']} fuentes que producen {m['n_features']} features; hoy
<b>{s['n_connected']} tienen adaptador</b> —el catálogo entero, evento incluido—,
<b>{s['n_measured']} tienen profundidad medida</b> y <b>{m['n_backtestable']} pueden entrar en un
backtest</b>. Que las tres cifras no coincidan es el punto: tener adaptador no es tener historia, y tener
historia solo cuenta si se ha comprobado. Y todas ellas llegan ya al espacio de observación y a la
decisión, en <i>backtest</i> y en vivo.</p>

<div class="why"><b>Por qué el esqueleto antes que las fuentes.</b> El valor no está en conectar
<i>una</i> fuente, está en que la fuente N+1 cueste casi cero: un adaptador, una entrada de catálogo y un
test. Las diecisiete comparten puerto, archivo, esquema, normalización y sonda. Y hay un motivo para
arrancar la captura antes que nada: <b>{m['by_pit']['forward_capture']} de las {m['n_sources']}</b> fuentes
son <i>forward capture</i> —nadie publica su pasado—, así que su profundidad histórica no depende de
escribir mejor código después sino del calendario. Cada día sin capturar es profundidad que no se
recupera.</div>

<p>Dos campos del catálogo sostienen la honestidad del resto. <span class="mono">history_from</span> es la
primera fecha con dato <b>comprobado por nosotros</b>, no la que anuncia el proveedor. Esa comprobación es
una operación explícita —<span class="mono">ai&#8209;trader signals depth</span>— que descarga la serie, la
deriva con el adaptador real y escribe lo que encuentra en
<span class="mono">data/signals/history_depth.json</span> (última medición: {s['depth_measured_at']}); el
catálogo solo puede declarar lo que ese registro respalda, y un test lo verifica.
<span class="mono">pit</span> dice de qué naturaleza es esa historia —capturada en vivo, backfill revisable
o derivada del precio—. Juntos contestan la única pregunta que hay que responder antes de puntuar nada:
qué fuentes pueden participar en un backtest y cuáles solo en vivo.</p>

{table}

<h4>Toda feature se publica normalizada, con dos varas de medir</h4>
<p>Las fuentes producen magnitudes que no se pueden ni sumar ni comparar: miles de millones de oferta de
<i>stablecoins</i>, puntos básicos de dispersión de <i>funding</i>, visitas, primas porcentuales, commits.
Y la comparación entre activos es peor: que BTC tenga 23.000 visitas y SEI 40 no dice nada sobre cuál está
recibiendo atención <b>inusual</b>. Por eso nada sale de la ingesta en crudo, y sale con
<b>dos</b> normalizaciones y no con una: <span class="mono">&lt;feature&gt;{n['suffix_self']}</span> es la
z contra la <b>propia historia</b> de la entidad —ventana expansiva y causal, hasta el día <i>t</i>
incluido, con un mínimo de {n['min_history']} observaciones— y
<span class="mono">&lt;feature&gt;{n['suffix_cross']}</span> es la z contra la <b>sección cruzada</b> del
día, con un mínimo de {n['min_cross_section']} entidades. La primera no existe para un listado nuevo; la
segunda sí, desde el primer día. El centro es la mediana y la escala <span class="mono">{n['scale']}</span>
—con media y desviación típica, un solo día extremo infla la escala y apaga la señal justo cuando empieza
a pasar algo—, todo recortado a <b>±{n['clip']}</b>, un umbral declarado y no escondido. Lo que no se puede
calcular queda en <b>NaN</b>: un 0 diría «normal, en la media», que es una afirmación sobre el mundo que no
se ha observado.</p>

<h4>Lo que la medición destapó, y que ningún folleto decía</h4>
<ul>
<li><b>El COT se conoce tres días después de su fecha.</b> La foto es del martes y se publica el viernes.
La serie se archiva con el día de <b>publicación</b>: fecharla el martes metería tres días de futuro en
cualquier cruce, y el recorte por día de observación (§4.2) no puede verlo, porque para él el martes es
pasado.</li>
<li><b>El sello de <i>funding</i> de CCXT es el próximo cobro, no la observación.</b> Se descubrió porque
la sonda devolvió una serie que empezaba <i>mañana</i>. El sello sigue sirviendo para agrupar venues dentro
de la misma ventana —que es para lo que vale—, y el día sale del momento de la observación.</li>
<li><b>Un único <i>slug</i> que devolvía 400 se llevó por delante las otras 23 series de su fuente.</b> El
principio de «una fuente caída no tumba la captura» faltaba un nivel más abajo: una llamada rechazada no
puede tumbar a las demás de su fuente.</li>
<li><b>Hay huecos que son medición, no olvido.</b> TFTC publica la serie de ETF de BTC pero no la de ETH, y
FRED discontinuó las series de oro de la LBMA. Ninguno de los dos se rellena con un sustituto que
significaría otra cosa: el hueco se declara y aparece como cobertura cero.</li>
</ul>

<p><b>Una sola vía, y el <span class="mono">tier</span> pasó a describir en vez de enrutar.</b> Durante una
fase el diseño fueron <i>dos puertas</i>: el tier <b>A</b> (desbloqueos, colas de staking, dificultad,
hacks, sanciones) entraría como <b>elegibilidad</b> —una guarda que veta operar— y solo el <b>B</b>
(sentimiento, flujos de ETF, macro, actividad de desarrollo, dispersión de <i>funding</i>) como features.
La bifurcación se retiró, y conviene decir por qué con precisión: la defensa que la justificaba —«muestras
de <i>decenas</i>; sobre catorce observaciones un CEM libre construye una estrategia preciosa y falsa»—
descansaba en un número <b>que nadie había medido</b>. Al medirlo con la misma sonda que corrigió los
<span class="mono">history_from</span> del folleto, resultó falso por un factor de diez.</p>

{pool_table}

<p>Son <b>{events.get('pooled_total', 0)} eventos</b> en total, publicados en
<span class="mono">data/signals/event_pool.json</span>. Y donde la muestra <i>sí</i> es corta, la razón es
también una medición y no una propiedad de la fuente: el endpoint de desbloqueos de DefiLlama pasó a
responder <b>402 Payment Required</b> y el de colas de staking, <b>401 Unauthorized</b>. La unidad de esa
cuenta es el <b>evento normalizado</b> —% del <i>float</i>, % del ADV— y no el token, que es lo que permite
ponerlos todos en la misma distribución: catorce desbloqueos por token son cientos a través del universo.</p>

<p>Así que <b>todas las señales fluyen al motor por la misma vía</b> —features al espacio de observación,
en <i>backtest</i> y en vivo— y lo que determina la <b>codificación</b> no es el tier sino la
<b>cadencia</b>: {radar.get('n_event', 0)} fuentes de evento, {radar.get('n_continuous', 0)} continuas y
{radar.get('n_price_map', 0)} mapas de precios. La distinción no es cosmética:
<span class="mono">staking_queue</span> es mecánica (tier A) y publica un <b>nivel</b> diario, así que
enrutar por el tier le habría puesto un días-al-evento a una cola de validadores.</p>

<h4>Por qué los eventos no pasan por la misma normalización</h4>
<p>Una z contra una serie que es <b>99 % ceros</b> no significa nada: su mediana es 0, su rango
intercuartílico es 0 y el resultado es o NaN o un número enorme. Peor: rellenar los huecos con ceros y
normalizar produce una feature que <i>parece</i> razonable, no falla ningún test de forma y no significa
nada. Las de evento se codifican aparte, y la diferencia vive en el código: proximidad al próximo evento
<b>acotada a {_n(spec.get('days_ahead_cap', 0), 0)} días</b>, estela del último acotada a
{_n(spec.get('days_active_cap', 0), 0)}, magnitud dividida por una escala declarada por fuente y recortada
a <b>±{_n(spec.get('magnitude_clip', 0), 0)}</b> —el mismo tope que las z, para que ninguna domine a la
otra por construcción—, y una marca de si esa entidad <i>tiene</i> calendario en esa fuente.</p>

<div class="why"><b>El tope no es un detalle de implementación.</b> «Días hasta el próximo desbloqueo» es
infinito cuando no hay ninguno, y el infinito no entra en un vector de observación. Las dos salidas
perezosas mienten de formas distintas: un 0 diría «es hoy» y un 9999 aplastaría la escala de todo lo demás.
Con un tope declarado y la cuenta invertida, «no hay nada a la vista» es un 0 que significa exactamente
eso. Y el tope hace un segundo trabajo que no se ve: es lo que mantiene honesto usar el calendario de
<i>hoy</i> para fechar el pasado, porque las reuniones del FOMC de 2019 también estaban publicadas en 2019
—a treinta días vista la respuesta es la que se habría tenido entonces; a dos años, no—. Lo que no se
anuncia (un <i>hack</i>, una sanción) tiene la mirada hacia adelante apagada <b>por código</b>: solo queda
su estela.</div>

<h4>La tercera codificación: el mapa de precios</h4>
<p>El lote de fuentes de alta fricción trajo un objeto que ninguna de las dos codificaciones anteriores
sabe leer: un <b>mapa de liquidación</b>. Se observa todos los días —luego no es un evento fechado: no hay
ninguna fecha futura que anticipar— y lo que dice no es un nivel sino una <b>distancia en precio</b>, más
el notional acumulado hasta ella. Las dos codificaciones anteriores lo leerían mal, cada una a su manera, y
—esto es lo que obliga a añadir la tercera— <b>ninguna de las dos daría error</b>: con las dos varas de la
normalización, la feature contestaría «¿es hoy la distancia alta <i>para este activo</i>?», que no es la
pregunta, porque que un clúster esté al 4 % es un hecho absoluto y no un percentil de su historia; con la
codificación de evento, la proximidad contaría días hasta una fecha que no existe.</p>
<p>La proximidad se mide entonces en la unidad en la que el hecho vive —porcentaje de precio— con el mismo
patrón que ya estaba: tope declarado de
<b>{_n((maps.get('spec') or {}).get('distance_cap_pct', 0), 0)} %</b>, cuenta invertida para que «no hay
nada cerca» sea un 0 que significa eso, y magnitud normalizada por su escala y recortada al mismo tope que
todo lo demás. El <b>signo lo pone el lado</b>: un clúster por debajo son largos que revientan vendiendo, y
por encima, cortos que revientan comprando. Y no hay estela sino <b>caducidad</b>: un calendario viejo
sigue siendo cierto, pero una foto del libro de hace dos semanas no describe ningún libro, así que pasados
{_n((maps.get('spec') or {}).get('stale_days', 0), 0)} días deja de contar como cobertura en vez de seguir
pareciendo fresca. Qué fuente va por aquí lo declara el catálogo campo a campo, y no la cadencia: es la
excepción, y es explícita para que se vea.</p>

{map_table}

<div class="why"><b>El hueco del mapa está declarado.</b> Solo el <b>75,7 %</b> de las posiciones
muestreadas trae precio de liquidación —en las de margen cruzado con holgura el venue lo devuelve nulo—,
así que el mapa está incompleto por abajo y el notional publicado es una <b>cota inferior</b>. No se
estima: estimarlo exigiría replicar el motor de margen del venue. Y la muestra son las 200 cuentas mayores
del <i>leaderboard</i>, que cubren el <b>21,7 %</b> del interés abierto: el sesgo hacia las cuentas grandes
es deliberado —una cuenta de 763 dólares no mueve el precio al ser liquidada— y significa que
«distribución del apalancamiento» aquí quiere decir «distribución en la cola de arriba».</div>

{_dat_section(dat, dat_table)}

<h4>El ADV: por qué una señal genuina puede no servir para nada</h4>
<p>Una señal puede ser real, tener historia, superar el break-even de IC de §7.5 y no servir para nada, y
la razón casi nunca es estadística: es que <b>vive en activos donde no cabe tamaño</b>. Ese fallo no lo
detecta ninguna métrica de las anteriores —el Sharpe de un backtest sin impacto de mercado no sabe cuánto
volumen tenía el activo— y se descubre tarde, cuando ya se ha escalado. Por eso el catálogo declara desde
el lote caro un campo más, <span class="mono">typical_adv_usd</span>: el volumen diario <b>mediano</b>, en
dólares, de las entidades donde esa señal existe. Se mide con
<span class="mono">signals/liquidity.py</span>, se apunta en
<span class="mono">data/signals/entity_adv.json</span> y un test exige que lo declarado esté respaldado por
el registro, exactamente igual que con <span class="mono">history_from</span>.</p>

{adv_table}

<p>La mediana y no la media, y sobre las entidades que <i>negocian</i>: en Hyperliquid la media diría 12,7
millones de dólares porque BTC mueve 1.700, y el perpetuo del medio mueve trescientos mil. La fuente más
estrecha del catálogo vive en entidades de
<b>{_n(liquidity.get('thinnest_adv_usd', 0), 0)} dólares al día</b>
(<span class="mono">{liquidity.get('thinnest_source', '')}</span>). El «efecto Upbit» es de los eventos más
limpios que existen en cripto <i>y</i> vive en mercados cuya mediana mueve un cuarto de millón: las dos
cosas son verdad a la vez, y tenerlas juntas delante es la única forma de no confundir una señal buena con
una señal escalable. La tolerancia del test es de un <b>orden de magnitud</b>, y es deliberado: entre
174.000 y 253.000 no hay ninguna decisión distinta; entre 174.000 y 174 millones están todas.</p>

<h4>El radar: de {m['n_features']} columnas crudas a seis números</h4>
<p>{m['n_features']} columnas no son un espacio de observación: son {m['n_features']} grados de libertad
esperando a que alguien los pondere. El radar (<span class="mono">observation/signal_radar.py</span>, con
la forma exacta del proveedor de régimen de §4.2: <span class="mono">features(symbol)</span>, memo por el
«ahora» del reloj y el mismo recorte anti-<i>look-ahead</i>) las reduce a <b>tres ejes</b> publicados por
<b>dos bloques</b>: <span class="mono">{' · '.join(radar.get('features', [])[:3])}</span> del activo y
<span class="mono">{' · '.join(radar.get('features', [])[3:])}</span> del mercado, iguales para todos los
símbolos ese día. El <b>tono</b> suma solo las {radar.get('n_polarity', 0)} features con polaridad
declarada y razonada una a una; lo que no está en esa tabla no es un olvido, es una feature cuya dirección
exigiría una hipótesis que nadie ha medido. La <b>intensidad</b> no tiene signo, y es el único eje en el
que momentum y reversión a la media quieren cosas opuestas: la primera la usa como confirmación (piso) y
la segunda como filtro de catástrofe (techo). En el tono las dos ponen un <b>piso</b>, porque el modo de
fallo característico de la reversión es comprar una caída de −3 σ que resulta ser el primer día de un
reprecio permanente.</p>

<div class="note"><b>La trampa de los tres estados, y el invariante que la desactiva.</b> La convención del
espacio de observación es «feature no disponible = 0.0 neutro» (§4.2), y con eventos es peor que peligrosa:
<i>no hay desbloqueo</i> y <i>no sé de desbloqueos</i> se escriben con el mismo cero. La <b>cobertura</b>
es la feature que los distingue, y con ella el invariante central: <b>una puerta de señales nunca bloquea
por falta de datos</b>. Por debajo del {radar.get('min_coverage_pct', 0)} % de las fuentes del bloque, la
puerta <b>no se evalúa</b> —falla abierta—. Ese umbral es una <b>constante, no un parámetro</b>: si fuera
sorteable, un sorteo del optimizador podría subirlo hasta convertir el radar en un filtro de
<i>disponibilidad de datos</i>, que rankearía por qué fuentes tenían cobertura en cada tramo de historia en
vez de por lo que dicen.</div>

<div class="why"><b>Qué sustituye a la puerta, y qué queda sin cubrir.</b> Dos límites que se pueden
comprobar y un test que falta. <b>Primero:</b> ninguna feature —de ningún tier— entra en el espacio de
búsqueda del optimizador, y los umbrales de las puertas son constantes declaradas en código, no parámetros
sorteables; el CEM solo reconstruye <span class="mono">strategies</span>, así que el límite es estructural y
la huella de las 16 configuraciones publicadas no se mueve. <b>Segundo:</b> el tamaño de muestra se
<i>mide</i> con la sonda en vez de declararse, y se amplía por <i>pooling</i> del evento normalizado —la
unidad de observación es el evento comparable (% del <i>float</i>, % del ADV), no el token, así que catorce
desbloqueos por token son cientos a través del universo—. <b>Y tercero,</b> el que faltaba: limitar los grados de
libertad <i>reduce</i> el riesgo de sobreajuste pero no lo <i>mide</i>, y el instrumento que sí lo mide ya
existe (§4.12). Es el <b>IC de break-even</b>: barrer la capacidad predictiva ρ de un canal de observación
sintético, con ρ = 0 como grupo de control, de modo que la pregunta que se le hace al histórico real sea
una sola cifra binaria. Lo que sigue faltando, y conviene no confundirlo, es el <b>otro lado</b> de esa
pregunta: cuánto ρ tiene realmente cada una de estas diecisiete fuentes. Eso se mide en el sustrato real,
con la profundidad que la captura vaya comprando día a día.</div>

<div class="note"><b>No hay veto.</b> Con la unificación se decidió que ninguna señal bloquee: no existe
guarda de elegibilidad por señales ni lista negra, y toda señal actúa como feature. La consecuencia se
declara en vez de silenciarse: nada impide hoy abrir posición en un activo <b>sancionado, deslistado o con
el mercado detenido</b>. Esa guarda es <i>operativa</i> y no de alfa —vetar un activo sancionado no es una
estrategia, es una restricción administrativa—, así que vive aparte y su riesgo, mientras el dinero sea de
papel y el universo se configure a mano, es teórico.</div>

<h4>Las cuatro decisiones de diseño, y qué se paga si se toman al revés</h4>
<ul>
<li><b>El puerto tiene dos capas.</b> <span class="mono">fetch_raw</span> toca la red y devuelve el payload
intacto; <span class="mono">daily_from_raw</span> es una función <b>pura</b> que lo traduce. La parte que
se equivoca es siempre el mapeo, y separadas corregirlo <b>re-deriva</b> sobre lo ya archivado en vez de
re-descargar — que con una fuente cuyo pasado no existe no es una molestia, es información perdida. Como
efecto secundario, la mitad frágil se testea sin red y en milisegundos.</li>
<li><b>El crudo va en <span class="mono">data/</span> y no en <span class="mono">.cache/</span>.</b>
Append-only, en <span class="mono">data/signals_raw/&lt;fuente&gt;/&lt;entidad&gt;/&lt;mes&gt;.jsonl.gz</span>,
con su <span class="mono">fetched_at</span>. No se re-deriva: se <b>guarda</b>. Es el mismo principio por el
que se guarda el <span class="mono">spec.json</span> de cada escenario (§2.6). Nada se sobrescribe: cuando
una fuente revisable reescribe el pasado, la línea vieja sigue ahí, y comparar las dos es la única forma de
medir cuánto revisa. Lo derivado vive en <span class="mono">.cache/signals/</span> y es desechable.</li>
<li><b>El esquema diario es propio.</b> <span class="mono">(entidad, día) → features + observed</span>, y
<b>no</b> reutiliza el de barras: aquel deduplica por índice con <span class="mono">keep='last'</span>, que
para barras es correcto y aquí destruiría varias observaciones legítimas del mismo día. La agregación vive
en un solo sitio y cada feature declara <i>cómo</i> se agrega — sumar un estado o quedarse con el último de
un flujo son errores que no se ven en la gráfica. La columna <span class="mono">observed</span> cuenta las
observaciones crudas detrás de cada celda: es lo que distingue «vale 0» de «no hay dato», y la convención
del sistema («feature no disponible = 0.0 neutro», §4.2) no lo distingue por sí sola.</li>
<li><b>La entidad se deriva, y la tabla arranca vacía.</b> Las fuentes externas no hablan en símbolos de
mercado. La clave común sale de una regla (<span class="mono">XYZ/USDT → XYZ</span>) que funciona el día que
se añade un listado nuevo sin que nadie edite nada; la tabla de <i>overrides</i> existe para cuando la regla
acierta la forma y falla el fondo, y hoy tiene <b>{e['n_overrides']} entradas</b>. Cada
<span class="mono">EntityRef</span> lleva su procedencia (regla / override / sin resolver) para que el cruce
sea auditable. Sobre el universo configurado: {e['n_symbols']} símbolos → {e['n_entities']} entidades,
cobertura <b>{_n(e['coverage_pct'], 1)} %</b> ({e['by_source']['rule']} por regla,
{e['by_source']['override']} por override, {e['by_source']['unmapped']} sin resolver).</li>
</ul>

<h4>El cableado, en los dos sitios, y su compuerta</h4>
<p>El catálogo declara, la captura archiva ({s['records']} registros hoy), la sonda mide, la normalización
publica y el radar lo lleva a la decisión. Las estrategias reciben el proveedor por el mismo bucle
<i>duck-typed</i> que ya usaban para el régimen, y ese bucle es idéntico en el motor de <i>backtest</i> y
en el proceso en vivo: tener dos comportamientos según dónde corre el sistema es un defecto, no una fase.
De paso se cerró un hueco anterior —el proveedor de régimen <b>no se adjuntaba en producción</b>, así que
cualquier configuración elegida con filtros de régimen activos se comportaba distinto en <i>paper</i> que
en el <i>backtest</i> que la seleccionó— con un <span class="mono">Mapping</span> perezoso sobre el
servicio de datos que no obligó a cambiar una sola línea del proveedor de régimen.</p>

<div class="note"><b>La compuerta se cumplió.</b> Con las puertas de señal en su valor neutro, la
validación multiventana devuelve los scores <b>idénticos</b> a los ya publicados —cinco unidades
reproducidas, quince ventanas OOS cada una, hasta el último decimal— y hay un test que lo comprueba contra
el fichero publicado. Adjuntar el radar no movió un solo número, que es la única forma de que la evidencia
anterior siga hablando del sistema que corre. Los <i>defaults</i> de las puertas, además, no son
permisivos: son <b>imposibles de activar</b>, porque el tono vive en ±{n['clip']} y la intensidad en
[0, {n['clip']}] por el recorte de las z, y los umbrales por defecto están en el borde exacto de ese
intervalo. Es la mejora concreta sobre <span class="mono">min_relative_strength = −1.0</span>, que se
documentó como «sin filtro» sin serlo: la fuerza relativa no está acotada y un −1.0 es alcanzable.</div>
"""


def _activity_block(a):
    """Seccion 4.10: el suelo de actividad, sus dos condiciones y de donde sale cada una.

    Va pegada al gate (§4.9) y no en una seccion suelta porque es la segunda mitad del
    mismo veredicto: sin ella, "batir a los baselines" se puede conseguir sin operar."""
    if not a:
        return (
            "<div class=\"note\"><b>Suelo de actividad sin evidencia publicada.</b> El informe "
            "(<span class=\"mono\">data/activity/</span>) no está en este árbol, así que este "
            "documento no puede citar de dónde sale el umbral. Regenéralo con "
            "<span class=\"mono\">python -m ai_trader.scoring.activity_study</span>.</div>"
        )
    f, dec, m = a["floor"], a["decision"], a["mechanism"]["sides"]["real"]
    g, rep, band = a["gate"]["real"], a["reproducibility"]["sides"]["real"], a["band"]["real"]
    rows = "".join(
        f"<tr><td class=mono>{r['config_id']}</td>"
        f"<td class='n mono'>{_n(r['reward'], 3)}</td>"
        f"<td class='n mono'>{_n(r['trades_per_window'], 2)}</td>"
        f"<td class='n mono'>{_n(r['zero_window_pct'], 0)} %</td>"
        f"<td>{'sí' if r['rankable'] else '<b>no</b>'}</td></tr>"
        for r in a["rows"]
    )
    sweep = "".join(
        f"<tr><td class='n mono'>{_n(s['threshold'], 0)}"
        f"{' ←' if s['threshold'] == dec['chosen'] else ''}</td>"
        f"<td class='n mono'>{s['sides']['real']['n_eligible']}</td>"
        f"<td class='n mono'>{s['disagreements']}</td>"
        f"<td class=mono>{s['sides']['real']['winner'] or '—'}</td>"
        f"<td class='n mono'>{len(s['sides']['real']['approved'])}</td></tr>"
        for s in a["sweep"]
    )
    lost = ", ".join(f"<span class='mono'>{c}</span>" for c in g["lost"]) or "ninguna"
    return f"""
<h3>4.10 · El suelo de actividad: quién puede ganar el ranking</h3>
<p>El headline de una ventana en la que la estrategia <b>no abre ninguna posición</b> es
<b>0 exacto</b>: la curva es una recta, luego Sharpe 0, rotación 0 y caída 0. No es una nota mala ni
buena, es la <i>ausencia</i> de nota — y sin embargo entra en la distribución como un número más. Como
la recompensa es el CVaR (la media del peor cuartil) y un cero le gana a cualquier cosa que arriesgue y
pierda, en un periodo donde casi todo lo que se juega pierde <b>el ranking se ordena por inactividad</b>.
Medido sobre el material del §4.11: Spearman(recompensa, operaciones por ventana) =
<b>{_n(m["spearman_reward_activity"], 2)}</b> en el lado real, y
{_n(m["cvar_tail_empty_pct"], 1)} % de las ventanas que <i>fijan</i> la recompensa estaban vacías.</p>
<div class="why"><b>Por qué un requisito y no una penalización.</b> Penalizar la baja rotación ya existe
—es λ (§4.5)— y el estudio de pesos midió que <b>no estabiliza nada</b>: el rank IC es máximo sin
penalizar. Restar más puntos por operar poco sería cobrar dos veces la misma factura y degradar el
ranking. Lo que faltaba no era un descuento sino una <b>condición de entrada</b>: no perder es legítimo,
lo que no lo es es ganar un ranking de estrategias sin haber jugado. Por eso la recompensa de una
configuración inelegible <b>no se toca</b> —se calcula, se publica y se compara igual—; lo único que
pierde es competir y aprobar el gate.</div>
<p>Una configuración es <b>rankeable</b> si cumple las dos condiciones. Una está derivada y la otra
medida:</p>
<table><thead><tr><th>Condición</th><th class=n>valor</th><th>de dónde sale</th></tr></thead><tbody>
<tr><td>Ventanas vacías</td><td class='n mono'>≤ {_n(f["max_zero_window_pct"], 0)} %</td>
  <td><b>Derivada.</b> Es α, la fracción de cola que <i>es</i> la recompensa (CVaR@{_n(f["max_zero_window_pct"], 0)}%).
  Por encima de ella, el cuartil que promedia el CVaR puede estar hecho de ceros estructurales.</td></tr>
<tr><td>Operaciones en la ventana mediana</td>
  <td class='n mono'>≥ {_n(f["min_median_trades_per_window"], 0)}</td>
  <td><b>Medida.</b> Regla declarada antes de mirar: el valor de la rejilla
  {{{", ".join(_n(x, 0) for x in dec["grid"])}}} que reproduce con menos desacuerdos la condición
  derivada; a igualdad, el mayor. Resultado: {_n(dec["chosen"], 0)}, con {dec["disagreements"]}
  desacuerdo{"" if dec["disagreements"] == 1 else "s"}.</td></tr>
</tbody></table>
<p>El barrido completo, publicado entero porque un umbral que solo se sostiene enseñando su resultado y
escondiendo los de al lado no se sostiene:</p>
<table><thead><tr><th class=n>umbral</th><th class=n>rankeables</th><th class=n>desacuerdos</th>
<th>ganador del ranking real</th><th class=n>aprueban el gate</th></tr></thead><tbody>{sweep}</tbody></table>
<p>Y la elección del número casi no importa, que es la mejor noticia posible sobre un umbral: con
{"/".join(_n(x, 0) for x in band["same_partition"])} operaciones por ventana sale <b>exactamente el
mismo conjunto rankeable</b> en el lado real ({band["n_rankable"]} de {len(a["rows"])}), porque a esa
escala quien excluye es la condición derivada. En esta rejilla la condición medida solo muerde una vez
—una configuración del mundo sintético que se queda fuera con el 25,0 % justo de ventanas vacías— y se
mantiene porque cubre un modo de fallo que estas configuraciones no contienen: operar poquísimo pero con
regularidad, sin dejar ni una ventana vacía. Lo que está en discusión es la partición, no el decimal.</p>
<div class="note"><b>El control que parecía obvio y habría elegido al revés.</b> La <i>reproducibilidad</i>
del ranking entre mitades del histórico sale <b>más alta</b> con las inactivas dentro
({_n(rep["all_configs"], 3)}) que sin ellas ({_n(rep["rankable_only"], 3)}; subconjuntos aleatorios del
mismo tamaño: {_n(rep["random_same_size_mean"], 3)}). Se entiende en cuanto se mide: una configuración
que no opera puntúa 0 en todos los bloques y su puesto no se mueve jamás. Es estabilidad de cementerio,
así que la reproducibilidad se publica como <b>control</b> y no como criterio.</div>
<p>Efecto sobre el gate en el material del §4.11: de <b>{len(g["approved_without_floor"])}</b>
configuraciones aprobadas se pasa a <b>{len(g["approved_with_floor"])}</b>. Pierden la aprobación
{g["n_lost"]}: {lost} — todas ellas por no operar, no por puntuar peor.</p>
<div class="note"><b>Dónde NO cambia nada, que delimita el alcance.</b> El estudio de validación
multiventana (§4.8) se re-corrió entero con el suelo puesto y sus veredictos salieron
<b>idénticos</b> (7 aprobadas en walk-forward, 6 en CPCV, sobre 32 unidades): allí las cuatro
configuraciones operan de sobra. El requisito no recorta aprobaciones en general — muerde
exactamente donde la inactividad estaba ganando.</div>
<table><thead><tr><th>Configuración</th><th class=n>recompensa real</th><th class=n>ops/ventana</th>
<th class=n>vacías</th><th>rankeable</th></tr></thead><tbody>{rows}</tbody></table>
<p class="meta">Evidencia: <span class="mono">data/activity/report_{a["library"]}.json</span> ·
código: <span class="mono">scoring/activity.py</span> y
<span class="mono">scoring/activity_study.py</span>.</p>"""


def _divergence_block(d):
    """
    Seccion 5.4: cuanto se aparta lo ejecutado de lo que el motor predecia.

    Es la unica seccion del documento cuyo estado NORMAL durante meses es "sin cifra", y
    por eso los dos caminos estan escritos con el mismo cuidado. El de sin-potencia no es
    un hueco: es el estudio negandose a publicar, con cuantos dias faltan y por que, que
    es una afirmacion comprobable donde antes habia una promesa.

    LA SECCION VA MARCADA COMO VOLATIL, y es el unico sitio del documento que lo esta.
    Todo lo demas sale de informes commiteados y por eso se regenera igual en cualquier
    clon; esto sale de `data/live/`, que esta fuera de git a proposito porque crece cada
    quince minutos en la maquina que opera. Sin la marca, la caracterizacion de
    `docs/metodologia.html` fallaria con solo dejar el bot corriendo una noche. Lo que la
    marca oculta se prueba en `tests/test_divergence.py`, contra el informe y no contra
    el HTML.
    """
    return "<!--LIVE-->" + _divergence_body(d) + "<!--/LIVE-->"


def _divergence_body(d):
    if not d:
        return (
            "<h3>5.4 · Divergencia live-vs-backtest</h3>"
            "<div class=\"note\"><b>No medida en esta copia.</b> Se mide con "
            "<span class=\"mono\">python -m ai_trader.backtest.divergence_study</span> y se publica en "
            "<span class=\"mono\">data/live/divergence.json</span>. El material —el diario de ciclos— "
            "ya se está guardando desde el primer ciclo.</div>"
        )

    journal, power = d["journal"], d["power"]
    head = (
        "<h3>5.4 · Divergencia live-vs-backtest</h3>"
        "<p>La cifra que justifica el capítulo 3 entero. Se coge la ventana de calendario que cubre el "
        "diario, se corre <b>el mismo periodo</b> con el motor de <i>backtest</i> sobre las barras reales "
        "de esos días, y se comparan las dos ejecuciones <b>decisión a decisión</b>, no un Sharpe contra "
        "otro: dos curvas distintas pueden dar el mismo Sharpe, y entonces el número no dice dónde está "
        "la diferencia. La unidad de pareo es <span class=\"mono\">(día UTC, símbolo, estrategia)</span>, "
        "que es lo más fino en lo que los dos mundos son comparables — en vivo el <i>runner</i> despierta "
        "decenas de veces al día y en <i>backtest</i> una vez, pero los dos deciden sobre la misma barra "
        "diaria ya cerrada.</p>"
        "<p>La re-simulación no reimplementa nada: se le engancha un diario en memoria al mismo motor, de "
        "modo que emite <b>exactamente el mismo esquema de línea</b> que el paper trading en vivo.</p>"
    )

    if not d["measured"]:
        reasons = "".join(f"<li>{r}</li>" for r in power.get("reasons", []))
        return head + (
            "<div class=\"why\"><b>Hoy no hay cifra, y eso es el resultado.</b> El diario cubre "
            f"<b>{journal['span_days']} días</b> de calendario ({journal['n_days']} con ciclos) y la regla "
            f"declarada pide <b>{power['required_days']}</b>: faltan <b>{power['missing_days']}</b>. El "
            "estudio <b>se niega a re-simular y a publicar</b> en vez de sacar una divergencia medida "
            "sobre cuatro días, que tendría exactamente el mismo aspecto que la buena."
            f"<ul>{reasons}</ul>"
            "Las dos condiciones —<i>span</i> de calendario y días con ciclos— no son redundantes: un "
            "proceso que corrió dos días, se apagó cinco semanas y volvió tiene <i>span</i> de sobra y no "
            "ha observado nada.</div>"
            "<p>Cuando haya calendario, lo que se publicará es la diferencia de precio de llenado "
            "<b>repartida en tres sumandos que suman</b> —referencia, coste y término cruzado—, el embudo "
            "de decisiones de los dos mundos, y la latencia en tiempo y en puntos básicos. Que las piernas "
            "sumen es lo que impide que una absorba en silencio el error de otra.</p>"
            f"<p class=\"tag\">Informe: <span class=\"mono\">data/live/divergence.json</span> · "
            f"{d['generated_at']}</p>"
        )

    comps, verdict = d["components"], d["verdict"]
    rules = verdict["rules"]

    def bps(block, key="median"):
        value = (block or {}).get(key)
        return "—" if value is None else f"{value:+.2f}"

    stage_rows = "".join(
        f"<tr><td>{k}</td><td class='n mono'>{r['live']}</td>"
        f"<td class='n mono'>{r['resim']}</td><td class='n mono'><b>{r['both']}</b></td>"
        f"<td class='n mono'>{r['only_live']}</td><td class='n mono'>{r['only_resim']}</td></tr>"
        for k, r in d["stages"].items()
    )
    verdict_rows = "".join(
        f"<li><b>{rules[k]['rule']}</b> → {rules[k]['text']}</li>"
        for k in ("decisions", "cost", "latency")
    )
    return head + (
        "<table><thead><tr><th>Pierna</th><th class=n>mediana (pb)</th><th class=n>p90</th>"
        "<th>Qué es</th></tr></thead><tbody>"
        f"<tr><td><b>total</b></td><td class='n mono'><b>{bps(d['total_bps'])}</b></td>"
        f"<td class='n mono'>{bps(d['total_bps'], 'p90')}</td>"
        "<td>lo que se pagó de más frente a lo que el modelo predecía</td></tr>"
        f"<tr><td>referencia</td><td class='n mono'>{bps(comps['reference_bps'])}</td>"
        f"<td class='n mono'>{bps(comps['reference_bps'], 'p90')}</td>"
        "<td>decidir con un cierre diario que en el instante del <i>fill</i> ya es viejo</td></tr>"
        f"<tr><td>coste</td><td class='n mono'>{bps(comps['cost_bps'])}</td>"
        f"<td class='n mono'>{bps(comps['cost_bps'], 'p90')}</td>"
        "<td>deslizamiento cobrado contra el modelado</td></tr>"
        f"<tr><td>cruzado</td><td class='n mono'>{bps(comps['cross_bps'])}</td>"
        f"<td class='n mono'>{bps(comps['cross_bps'], 'p90')}</td>"
        "<td>término de segundo orden, publicado para que la suma cierre</td></tr>"
        "</tbody></table>"
        f"<p class=\"tag\">{d['n_repriced']} entradas re-tasadas; la descomposición "
        f"{'cierra' if d['decomposition_ok'] else '<b>NO cierra</b>'}.</p>"
        "<h4>Decisiones que no se tomaron</h4>"
        "<p>El recuento de decisiones detecta divergencias que el PnL esconde: si en vivo se generan la "
        "mitad de las señales, el problema no es el coste, <b>son los datos</b>.</p>"
        "<table><thead><tr><th>Etapa</th><th class=n>vivo</th><th class=n>re-simulado</th>"
        "<th class=n>ambos</th><th class=n>sólo vivo</th><th class=n>sólo resim.</th></tr></thead>"
        f"<tbody>{stage_rows}</tbody></table>"
        f"<h4>Las tres reglas declaradas</h4><ul>{verdict_rows}</ul>"
        f"<div class=\"why\"><b>El techo de lo que esto mide hoy.</b> {d['ceiling']}</div>"
        f"<p class=\"tag\">Informe: <span class=\"mono\">data/live/divergence.json</span> · "
        f"{journal['span_days']} días de diario · {d['generated_at']}</p>"
    )


def _sessions_block(s):
    """Seccion 3.5: la limitacion MEDIDA de la convencion de llenado (§3.4).

    Va inmediatamente detras de la convencion que sostiene -o corrige-, porque separarlas
    dejaria esa convencion justificada solo por prudencia, que es como estaba antes de
    medirla."""
    if not s:
        return (
            "<div class=\"note\"><b>Limitación no cuantificada.</b> La fracción de formación de "
            "precio que cae en la ventana entre el cierre con el que se decide y el open al que se "
            "llena <b>no está medida</b> en esta copia del repositorio. Se mide con "
            "<span class=\"mono\">python -m ai_trader.backtest.session_study</span> y se publica en "
            "<span class=\"mono\">data/sessions/report.json</span>.</div>"
        )

    gap, trend, us = s["gap"], s["trend"], s["us"]
    lat = s["latency_1h"] or {}
    by_key = {x["key"]: x for x in s["sessions"]}
    us_session = by_key[s["us_key"]]

    session_rows = "".join(
        f"<tr><td><b>{x['label']}</b></td>"
        f"<td class='n mono'>{x['start_hour']:02d}–{x['end_hour']:02d}</td>"
        f"<td class='n mono'>{_pc(x['clock_share'], 1)}</td>"
        f"<td class='n mono'>{_pc(s['overall'][x['key']]['abs_return'], 1)}</td>"
        f"<td class='n mono'>{_pc(s['overall'][x['key']]['variance'], 1)}</td>"
        f"<td class='n mono'><b>{_n(s['overall'][x['key']]['variance_intensity'], 2)}</b></td>"
        f"<td class='n mono'>{_pc(s['overall'][x['key']]['sets_low'], 1)}</td></tr>"
        for x in s["sessions"]
    )
    latency_rows = "".join(
        f"<tr><td class=mono>+{r['hours']} h</td>"
        f"<td class='n mono'>{_pc(r['slip_share_of_range_median'], 2)}</td>"
        f"<td class='n mono'>{_n(r['slip_bps_median'], 1)}</td>"
        f"<td class='n mono'>{_pc(r['path_share_of_range_median'], 1)}</td></tr>"
        for r in s["latency_rows"]
    )

    material = s["verdicts"]["gap"]["material"]
    limitation = (
        f"<div class=\"note\"><b>Limitación declarada: el hueco entre el cierre visto y el open "
        f"llenado.</b> Medido sobre {s['n_symbols']} pares y "
        f"{_n(s['n_days'], 0)} días-símbolo de barras 1H de {s['exchange']} "
        f"({s['window']['start']} → {s['window']['end']}), ese hueco vale "
        f"<b>{_pc(gap['share_of_range']['median'], 3)}</b> del rango del día en mediana "
        f"({_n(gap['bps']['median'], 2)} pb; p99 {_pc(gap['share_of_range']['p99'], 2)}), frente al "
        f"umbral declarado del {_pc(s['thresholds']['gap_material_share'], 0)}. "
        + (
            "<b>Está por encima del umbral:</b> el motor ignora un tramo de formación de precio "
            "que importa, y la convención de llenar al open deja de ser conservadora."
            if material
            else
            "En un mercado 24/7 la vela de las 00:00 UTC empieza donde terminó la de ayer: la "
            "ventana ciega <b>no tiene ancho</b> y la convención de llenar al open no introduce "
            "sesgo. Es un resultado, no una ausencia de resultado — y desplaza la pregunta."
        )
        + " <b>Lo que sí queda sin modelar es la latencia:</b> llenar «al open» solo es exacto si la "
        f"orden sale en ese instante. Con una hora de retraso el precio de llenado ya se ha "
        f"desplazado <b>{_pc(lat.get('slip_share_of_range_median'), 2)}</b> del rango del día "
        f"({_n(lat.get('slip_bps_median'), 1)} pb) y se ha gastado "
        f"{_pc(lat.get('path_share_of_range_median'), 1)} de ese rango. Esa fracción de rango parece "
        "modesta, pero es el denominador equivocado: frente al coste de entrada que el motor "
        f"<i>sí</i> cobra ({_n(s['reference_cost_bps'], 1)} pb de referencia, §3.6), llegar una hora "
        f"tarde cuesta <b>{_n(_ratio(lat.get('slip_bps_median'), s['reference_cost_bps']), 1)}×</b>. "
        "El backtest supone ejecución instantánea a las 00:00 UTC: la suposición no sesga el precio "
        "—el hueco es cero— pero pone un <b>techo a la puntualidad</b> con la que el ciclo real tiene "
        "que ejecutar para que el backtest siga describiéndolo.</div>"
    )

    direction = {
        "us_crece": "<b>se sostiene</b>",
        "us_decrece": "<b>sale al revés</b>",
    }.get(trend["verdict"], "<b>no se sostiene</b>")
    sign = "+" if (trend["mean_delta_variance"] or 0) >= 0 else ""
    years = trend["pre_split_years"]
    mechanism = (
        f" Su <b>mecanismo, en cambio, no</b>: la cuota ya venía subiendo antes del corte "
        f"(Spearman año-cuota = {_n(trend['pre_split_spearman'], 2)} sobre "
        f"{years[0]}–{years[-1]}, por encima del umbral declarado de "
        f"{_n(trend['pre_trend_rho_threshold'], 2)}) y el escalón que cruza el corte es "
        f"{_pc(trend['step_at_split'], 2)}. Es una <b>deriva de varios años</b> que el corte parte "
        "por la mitad, no un salto atribuible a lo que pasó en enero de 2024. Leer el contraste "
        "pre/post como efecto del mecanismo sería atribuirle crédito ajeno."
        if trend["shape"] == "deriva_previa" and years
        else (
            f" Y su <b>mecanismo encaja</b>: antes del corte no había pendiente (Spearman = "
            f"{_n(trend['pre_split_spearman'], 2)}) y el escalón en el corte vale "
            f"{_pc(trend['step_at_split'], 2)}."
            if trend["shape"] == "salto_en_el_corte"
            else " La <b>forma</b> del cambio queda sin contrastar con esta evidencia."
        )
    )
    trend_text = (
        f"La <b>dirección</b> de la hipótesis declarada antes de medir —que el peso de la sesión "
        f"estadounidense crece tras enero de 2024, por los ETF al contado— {direction}: la cuota "
        f"estadounidense de varianza cambia {sign}{_pc(trend['mean_delta_variance'], 2)} entre el "
        f"antes y el después de {trend['split']}, en {trend['n_positive']} de {trend['n_symbols']} "
        f"pares de la cohorte equilibrada (test de signos exacto, p = "
        f"{_n(trend['sign_test_p'], 4)}; umbral declarado {_pc(trend['threshold'], 0)}).{mechanism}"
    )

    return f"""
<h3>3.5 · Dónde se forma el precio dentro de la barra diaria</h3>
{limitation}
<p>La medición sale de abrir la vela diaria con barras <b>1H</b> y repartir el día UTC en tres
sesiones. Los cortes <b>no son redondos</b>: cada frontera es la hora de una apertura de mercado real,
tomada en su versión más temprana del año (los mercados de referencia cambian de hora UTC dos veces al
año; la rejilla de velas del exchange, no). Se elige la más temprana para que una sesión nunca contenga
la apertura de la siguiente.</p>
<table><thead><tr><th>Sesión</th><th class="n">UTC</th><th class="n">reloj</th>
<th class="n">|retorno|</th><th class="n">varianza</th><th class="n">intensidad</th>
<th class="n">fija el mínimo</th></tr></thead><tbody>{session_rows}</tbody></table>
<p><b>Intensidad</b> = cuota de varianza ÷ fracción de reloj: 1,00 es neutro. Es la única lectura
comparable, porque las tres sesiones no duran lo mismo ({', '.join(f"{x['hours']} h" for x in s['sessions'])}).
La sesión estadounidense concentra <b>{_pc(us['variance'], 1)}</b> de la varianza realizada en el
{_pc(us_session['clock_share'], 1)} del reloj (intensidad {_n(us['variance_intensity'], 2)}).</p>
<div class="why"><b>Por qué importa la columna «fija el mínimo».</b> La convención pesimista de arriba
—si en la misma barra se tocan stop y objetivo, gana el stop— se apoya en no saber en qué orden
ocurrieron. Esta columna dice dónde cae el extremo del día, y por tanto en qué tramo muerde esa
ambigüedad. No la resuelve: la localiza, que es el paso previo a poder resolverla con datos 1H si
algún día se decide modelar el camino intrabar.</div>
<p>Coste de la latencia de ejecución, en fracción del rango del día:</p>
<table><thead><tr><th>Retraso sobre el open</th><th class="n">desplazamiento / rango</th>
<th class="n">pb</th><th class="n">rango ya gastado</th></tr></thead>
<tbody>{latency_rows}</tbody></table>
<p>{trend_text} La tendencia se mide sobre una <b>cohorte equilibrada</b> ({s['n_cohort']} pares
presentes en todos los años): con el universo completo, la serie mediría qué se listó cuándo.</p>
<div class="note"><b>El motor no se ha tocado.</b> Este estudio mide y declara; no cambia ni una línea
de <span class="mono">execution/market_model.py</span>. Cambiar la convención antes de tener la cifra
habría sido sustituir una suposición por otra. Evidencia completa en
<span class="mono">data/sessions/report.json</span> · reproducible con
<span class="mono">python -m ai_trader.backtest.session_study --offline</span> ({s['generated_at']}).</div>"""


def _ic_grid(c):
    """Tabla λ×κ del rank IC: la evidencia cruda del barrido, para poder auditarla sin
    abrir el JSON."""
    head = "".join(f"<th class=n>κ={_n(k, 1)}</th>" for k in c["kappas"])
    rows = []
    for lam in c["lambdas"]:
        cells = []
        for kap in c["kappas"]:
            point = c["grid"].get((lam, kap))
            chosen = lam == c["lambda"] and kap == c["kappa"]
            value = _n(point, 3) if point is not None else "—"
            cells.append(f"<td class='n mono'>{'<b>' + value + '</b>' if chosen else value}</td>")
        rows.append(f"<tr><td class=mono>λ={_n(lam, 2)}</td>{''.join(cells)}</tr>")
    return (
        f"<table><thead><tr><th>&nbsp;</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _calibration_block(c):
    """Seccion 4.5: la calibracion de λ y κ, con la evidencia que la sostiene."""
    if not c:
        return (
            "<div class=\"note\"><b>Limitación declarada.</b> El informe de calibración "
            "(<span class=\"mono\">data/calibration/</span>) no está disponible en este árbol, así "
            "que este documento no puede citar las cifras que fijan λ y κ. Regenéralo con "
            "<span class=\"mono\">python -m ai_trader.scoring.weight_study</span>.</div>"
        )
    return f"""
<h3>4.5 · De dónde salen λ y κ (calibración medida)</h3>
<p>λ y κ no son una preferencia estética: son una <b>regla de selección</b>, y una regla de selección se
juzga por lo que elige. Se han barrido en rejilla
(λ ∈ {{{", ".join(_n(x, 2) for x in c["lambdas"])}}} × κ ∈ {{{", ".join(_n(x, 1) for x in c["kappas"])}}})
sobre <b>{c["n_backtests"]} backtests reales</b> de la librería <span class="mono">{c["library"]}</span>:
{c["n_configs"]} configuraciones (hipercubo latino sobre el espacio de búsqueda de las familias
que ese informe declara)
× {c["n_samples"]} muestras.</p>
<p>El truco que hace asequible el barrido: los <b>componentes</b> del score (Sharpe, turnover, maxDD)
<i>no dependen de los pesos</i>. Se corre el backtest una vez por (configuración, muestra), se guardan
los componentes crudos y la rejilla entera se evalúa después en memoria — {c["n_backtests"]} backtests,
no {c["n_backtests"]} × {len(c["lambdas"]) * len(c["kappas"])}.</p>
<div class="why"><b>Las dos cifras que deciden, y por qué son adimensionales.</b> Subir λ o κ encoge y
desplaza la escala del score, así que comparar <i>gaps</i> en crudo premiaría a los pesos grandes por
deflactar la unidad, no por elegir mejor. Por eso el criterio son dos magnitudes invariantes a la escala:
<ul>
<li><b>Rank IC</b> (eje temporal): correlación de rangos de Spearman entre el ranking de las
{c["n_configs"]} configuraciones <b>dentro</b> de muestra (ventana train) y <b>fuera</b> (ventana test),
promediada sobre las muestras. Responde: "si elijo con lo que veo, ¿sobrevive la elección?".</li>
<li><b>Gap train-validation normalizado</b> (eje de escenarios): cuánto se desinfla la configuración
elegida al pasar de los {c["n_train"]} arquetipos de train a los {c["n_validation"]} reservados, dividido
por la dispersión entre configuraciones. Responde: "¿cuánto de lo que elegí era del escenario y no de la
estrategia?".</li>
</ul>
Las combinaciones se comparan <b>pareadas</b> sobre las mismas muestras: la varianza común se cancela y
diferencias pequeñas se vuelven medibles, cosa que restar dos medias con sus errores sueltos no logra.</div>
<p><b>Rank IC medio por combinación</b> (más alto = elección más estable; en negrita, los pesos fijados):</p>
{_ic_grid(c)}
<h4>Resultado 1 · Los pesos no cambian la decisión</h4>
<p>En los {len(c["lambdas"]) * len(c["kappas"])} puntos de la rejilla — desde no penalizar nada hasta
penalizar ocho veces más que la configuración anterior — gana <b>siempre la misma configuración</b>
({c["n_winners"]} ganadora distinta en total), y lo mismo ocurre al repetir el barrido sólo con las
{c["n_active"]} configuraciones que operan de verdad. En el rango medido λ y κ <b>no arbitran nada</b>:
quien decide el ranking es el Sharpe.</p>
<h4>Resultado 2 · Penalizar no estabiliza; degrada un poco</h4>
<p>Se esperaba un punto dulce —algo de penalización actuando como regularizador—. No lo hay: el rank IC
es <b>máximo sin penalizar</b> ({_n(c["ic_neutral"], 3)} ± {_n(c["ic_neutral_se"], 3)}) y baja de forma
<b>monótona</b> al subir cualquiera de los dos pesos, hasta {_n(c["worst"]["rank_ic_mean"], 3)} en la
esquina (λ={_n(c["lambdas"][-1], 0)}, κ={_n(c["kappas"][-1], 0)}). El gap train-validation normalizado se
mueve en la misma dirección ({_n(c["gap_neutral"], 2)} sin penalizar →
{_n(c["prev"]["selection_gap_norm"], 2)} con los pesos anteriores). Los pesos que traía la herramienta
(λ = 0,5; κ = 1,0) costaban <b>{_n(abs(c["prev"]["rank_ic_gain"]), 3)} ± {_n(c["prev"]["rank_ic_gain_se"], 3)}</b>
de rank IC frente a no penalizar: un 17% del nivel de la señal, y significativo.</p>
<div class="why"><b>Por qué tiene sentido que el término de drawdown sea el que más cuesta.</b> Es la misma
objeción que retiró al Calmar: el máximo drawdown es el estadístico <b>más ruidoso</b> de una curva de
equity, porque depende de un único par de puntos de un único camino. Meterlo en el denominador disparaba la
varianza del estimador; meterlo como sumando la sube menos, pero la sube. Lo que el estudio añade es que
ese ruido <b>no compra nada</b>: ni cambia la configuración elegida ni mejora la supervivencia del ranking.</div>
<h4>Resultado 3 · La rotación ya se paga dentro del Sharpe</h4>
<div class="why">Ésta era la pregunta con más riesgo —¿se está cobrando dos veces la misma rotación?— y la
respuesta resultó ser la contraria de la temida. La curva de equity ya paga
<span class="mono">fee_rate + slippage</span> = {_n(c["cost_rate"] * 100, 3)}% de cada notional rotado, en
las dos patas. Convertido a la unidad de λ (puntos de Sharpe por unidad de turnover):
<p style="text-align:center" class="mono">λ<sub>implícito</sub> = cost_rate × 365 / σ<sub>anual</sub>
= {_n(c["cost_rate"], 4)} × 365 / {_n(c["median_vol"], 3)} ≈ <b>{_n(c["implied_lambda"], 1)}</b></p>
<b>Control de que la cadena es real y no una hipótesis:</b> las comisiones efectivamente cobradas divididas
por el notional reconstruido desde el turnover dan {_n(c["measured_fee_rate"] * 100, 4)}%, que reproduce
exactamente el <span class="mono">fee_rate</span> configurado. La aritmética se apoya en lo cobrado, no en
una suposición.
<p>Con un turnover mediano de {_n(c["median_turnover"], 3)}, la fricción ya se come
{_n(c["sharpe_drag"], 3)} puntos de Sharpe <i>dentro</i> del propio Sharpe. La penalización explícita
λ = {_n(c["lambda"], 2)} añade sobre eso un <b>{_n(c["share_pct"], 0)}%</b> (IQR del λ implícito:
{_n(c["implied_p25"], 1)}–{_n(c["implied_p75"], 1)}: depende de la volatilidad de cada configuración, no es
una constante, y por eso ninguna λ fija puede "tarifar" los costes). Lejos de duplicar la factura, λ era
—y sigue siendo— un <b>margen de seguridad</b> pequeño sobre el modelo de costes.</p>
<p><b>Vigencia tras el modelo de costes por microestructura (§3.6).</b> Este estudio se midió con el
deslizamiento plano que entonces cobraba el motor. El modelo actual cobra <b>más</b> fricción, y sobre
todo la reparte por símbolo y por tamaño: eso <b>refuerza</b> la conclusión —la rotación ya se paga
dentro del Sharpe, y λ es un margen aún más pequeño en términos relativos—, no la invierte. Re-medir la
rejilla con los costes nuevos es trabajo pendiente de la línea A, no un supuesto de esta.</p></div>
<h4>Los pesos que quedan fijados, y por qué no son los del máximo</h4>
<ul>
<li><b>λ = {_n(c["lambda"], 2)}</b> — el menor valor <b>no nulo</b> de la rejilla. La evidencia preferiría
λ = 0, pero el headline tiene una propiedad comprometida: <i>misma curva de equity y más rotación tiene que
puntuar peor</i>. Con λ = 0 esa puerta se reabre. Su precio medido es
{_n(abs(c["gain"]), 3)} ± {_n(c["gain_se"], 3)} de rank IC (indistinguible de cero en el subconjunto
activo), y equivale a un {_n(c["share_pct"], 0)}% del coste ya cobrado.</li>
<li><b>κ = {_n(c["kappa"], 1)}</b> — ninguna propiedad del diseño lo exige, es el término que más
estabilidad cuesta por unidad y su input es el más ruidoso. El mecanismo sigue en la fórmula: quien quiera
aversión explícita al drawdown pasa <span class="mono">kappa_maxdd</span>, pero ya no se cobra por defecto
sin que nadie lo haya pedido.</li>
</ul>
<div class="note"><b>Límites de este estudio.</b> Un único corte temporal 70/30 por muestra, un solo camino
por escenario y {c["n_configs"]} configuraciones, sobre un sustrato de rotación baja (turnover mediano
{_n(c["median_turnover"], 3)}). Es suficiente para descartar que penalizar fuerte ayude y para mostrar que
los pesos no cambian la decisión; <b>no</b> para afinar decimales, ni para extrapolar a un régimen de costes
más duros. Cuando la línea C (costes que muerden) aterrice, este estudio hay que repetirlo: el harness ya
está, y re-analizar cuesta segundos porque los componentes están cacheados.</div>"""


def _fidelity_table(rows, before_label=None):
    """Tabla de la comparacion, metrica a metrica: nivel, ordenacion y cobertura.

    Si hay libreria anterior, sus cifras van en su propia columna: el lector tiene que
    poder ver el antes y el despues sin cambiar de pagina."""
    prev_cols = f"<th class=n>{before_label}</th>" if before_label else ""
    head = (f"<tr><th>Stylized fact</th><th class=n>real</th>{prev_cols}"
            "<th class=n>sintético</th>"
            "<th class=n>ratio</th><th class=n>rank corr</th><th class=n>cobertura</th></tr>")
    body = []
    for row in rows:
        tag = ""
        if row["is_cross"]:
            tag = " <i>(pares)</i>"
        elif not row["is_target"]:
            tag = " <i>(contexto)</i>"
        ratio = "—" if row["ratio"] is None else f"{_n(row['ratio'], 2)}×"
        prev = row.get("before")
        prev_cell = ""
        was = ""
        if before_label:
            prev_cell = (
                f"<td class='n mono'>{_n(prev['synth'], row['decimals']) if prev else '—'}</td>"
            )
            was = f" <i>({_n(prev['coverage'], 0)}%)</i>" if prev else ""
        body.append(
            f"<tr><td>{row['label']}{tag}</td>"
            f"<td class='n mono'>{_n(row['real'], row['decimals'])}</td>"
            f"{prev_cell}"
            f"<td class='n mono'>{_n(row['synth'], row['decimals'])}</td>"
            f"<td class='n mono'>{ratio}</td>"
            f"<td class='n mono'>{_n(row['rank_corr'], 2)}</td>"
            f"<td class='n mono'>{_n(row['coverage'], 0)}%{was}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def _acceptance_table(acceptance):
    """Los umbrales que el estudio puede FALLAR, con lo que se midio en cada uno."""
    head = ("<tr><th>Umbral</th><th>Qué exige</th><th class=n>medido</th>"
            "<th class=n>veredicto</th></tr>")
    body = []
    for check in acceptance["checks"]:
        if check["kind"] == "coverage":
            demands = f"cobertura ≥ {_n(check['threshold'], 0)}%"
            measured = f"{_n(check['value'], 0)}%"
        else:
            band = check["band"]
            span = "—" if band is None else f"[{_n(band[0], 3)}, {_n(band[1], 3)}]"
            demands = f"mediana real dentro de {span}"
            measured = _n(check["value"], 3)
        verdict = "cumple" if check["passed"] else "<b>falla</b>"
        body.append(
            f"<tr><td>{check['label']}</td><td>{demands}</td>"
            f"<td class='n mono'>{measured}</td><td class=n>{verdict}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def _validation_block(v):
    """Seccion 4.8: como se parte el tiempo, y cuanto cambia la respuesta segun como."""
    if not v:
        return (
            "<h3>4.8 · Partición temporal: de un corte a una distribución de ventanas</h3>"
            "<p>Cada muestra se evalúa en <b>varias</b> ventanas out-of-sample disjuntas "
            "(walk-forward o CPCV) con purga y embargo entre entrenamiento y test, y sus "
            "puntuaciones se agregan con el mismo CVaR@25% que rankea el resto del sistema. "
            "El corte único 70/30 se conserva sólo como referencia comparativa.</p>"
            "<div class=\"note\"><b>Limitación declarada.</b> El informe del estudio "
            "(<span class=\"mono\">data/validation/</span>) no está disponible en este árbol, "
            "así que este documento no puede citar cuánto optimismo llevaba el corte único. "
            "Regenéralo con "
            "<span class=\"mono\">python -m ai_trader.scoring.validation_study</span>.</div>"
        )
    o, tail, std = v["opt_wf"], v["opt_tail"], v["std"]
    return f"""
<h3>4.8 · Partición temporal: de un corte a una distribución de ventanas</h3>
<p>Un backtest se puede partir en entrenamiento y test de muchas formas, y <b>la forma elegida cambia la
respuesta</b>. El sistema partía cada muestra con un único corte temporal 70/30. Eso tenía tres problemas
distintos, y sólo el primero es evidente:</p>
<ul>
<li><b>Un número no tiene cola.</b> La unidad de evaluación de este proyecto es la distribución y su
estadístico es el CVaR (§4.6) — pero el CVaR de una lista de un elemento <b>es</b> ese elemento. El
estadístico robusto existía sin tener sobre qué ser robusto dentro de la muestra.</li>
<li><b>La dispersión temporal era inobservable.</b> Con una sola ventana no hay forma de saber cuánto
depende el resultado del tramo de historia que tocó. No es que fuera pequeña: es que no se medía.</li>
<li><b>El corte no era limpio.</b> Las dos ventanas se pedían como intervalos cerrados, así que el día
del corte caía en <b>ambas</b>; y una posición abierta el último día de entrenamiento sigue viva dentro
del test hasta <span class="mono">max_holding_days</span>, de modo que el desenlace de las mismas
operaciones se contaba a los dos lados.</li>
</ul>
<h4>Los tres esquemas</h4>
<table><thead><tr><th>Esquema</th><th class="n">Ventanas OOS</th><th>Cómo</th></tr></thead><tbody>
<tr><td><span class="mono">single_split</span></td><td class="n">1</td><td>El corte 70/30 histórico. Se
conserva como <b>referencia</b>, para poder medir en qué se diferencia de los otros dos.</td></tr>
<tr><td><span class="mono">walk_forward</span></td><td class="n">{v["n_folds_wf"]}</td><td>El rango se
parte en {v["n_folds_wf"] + 1} grupos; cada uno a partir del segundo es el test de un fold que entrena
con todo lo anterior. Es lo que hace un operador real: avanza y no tira el pasado.</td></tr>
<tr><td><span class="mono">cpcv</span></td><td class="n">{v["n_folds_cpcv"]}</td><td>Combinatorial Purged
Cross-Validation: {v["n_groups"]} grupos y <b>todas</b> las combinaciones de {v["n_test_groups"]} como
test. Más ventanas con la misma historia, y cada tramo evaluado acompañado de contextos distintos.</td></tr>
</tbody></table>
<h4>Purga y embargo</h4>
<p>Entre el entrenamiento y cada tramo de test se abren dos huecos, y cada uno tapa una vía distinta:</p>
<ul>
<li><b>Purga</b> ({v["purge"]} días): se borran del entrenamiento los días inmediatamente
<b>anteriores</b> al test. El número no se elige a ojo — es exactamente
<span class="mono">max_holding_days</span>, cuánto puede seguir viva una posición abierta el último día
de entrenamiento.</li>
<li><b>Embargo</b> ({v["embargo"]} días, el 1% del rango): se borran los días inmediatamente
<b>posteriores</b>. Los retornos están serialmente correlacionados, así que entrenar con el día siguiente
al test es entrenar con su eco. Sólo muerde en CPCV, donde hay entrenamiento <i>después</i> del test.</li>
</ul>
<div class="why"><b>Lo que la purga no hace, dicho claro.</b> Dentro de un backtest no se ajusta nada: la
configuración entra fija y el motor construye reloj, estado y estrategias <b>nuevos</b> por ventana, de
modo que el entrenamiento de un fold no influye en su test. Purgar y embargar por tanto <b>no mejoran ni
empeoran</b> ninguna cifra out-of-sample, y hay un test que fija ese invariante (§4.15). Sirven para otras
dos cosas, ambas reales: que la referencia in-sample que se reporta no esté contaminada por operaciones
que siguen vivas dentro del test, y que la geometría ya sea correcta cuando algo <b>sí</b> se ajuste
sobre el entrenamiento — el optimizador del bucle exterior, o una política aprendida. Presentarlo de otro
modo sería vender como mejora de resultados lo que es una garantía estructural.</div>
<h4>Qué cambia al partir el tiempo de otra forma (medido)</h4>
<p>Se corrieron los tres esquemas sobre exactamente las mismas barras y la misma ventana, para
{v["n_units"]} unidades ({v["n_configs"]} configuraciones × {v["n_samples"]} muestras de
<span class="mono">{v["library"]}</span>), y se comparó <b>pareado</b>:</p>
<table><thead><tr><th>Magnitud</th><th class="n">Mediana</th><th class="n">IQR</th><th class="n">Mín … máx</th></tr></thead><tbody>
<tr><td>Diferencia del 70/30 frente a la <b>mediana</b> walk-forward</td>
<td class="n mono">{_n(o["median"], 3)}</td><td class="n mono">{_n(o["p25"], 2)} … {_n(o["p75"], 2)}</td>
<td class="n mono">{_n(o["min"], 1)} … {_n(o["max"], 1)}</td></tr>
<tr><td>Diferencia del 70/30 frente a la mediana CPCV</td>
<td class="n mono">{_n(v["opt_cpcv"]["median"], 3)}</td>
<td class="n mono">{_n(v["opt_cpcv"]["p25"], 2)} … {_n(v["opt_cpcv"]["p75"], 2)}</td>
<td class="n mono">{_n(v["opt_cpcv"]["min"], 1)} … {_n(v["opt_cpcv"]["max"], 1)}</td></tr>
<tr><td>Diferencia frente a la <b>cola</b> (CVaR walk-forward), que es lo que se rankea</td>
<td class="n mono">{_n(tail["median"], 3)}</td>
<td class="n mono">{_n(tail["p25"], 2)} … {_n(tail["p75"], 2)}</td>
<td class="n mono">{_n(tail["min"], 1)} … {_n(tail["max"], 1)}</td></tr>
<tr><td>Dispersión entre ventanas de una misma muestra (desviación típica)</td>
<td class="n mono">{_n(std["median"], 3)}</td><td class="n mono">{_n(std["p25"], 2)} … {_n(std["p75"], 2)}</td>
<td class="n mono">{_n(std["min"], 1)} … {_n(std["max"], 1)}</td></tr>
<tr><td>Rango entre la mejor y la peor ventana</td>
<td class="n mono">{_n(v["range"]["median"], 2)}</td>
<td class="n mono">{_n(v["range"]["p25"], 2)} … {_n(v["range"]["p75"], 2)}</td>
<td class="n mono">{_n(v["range"]["min"], 1)} … {_n(v["range"]["max"], 1)}</td></tr>
</tbody></table>
<div class="why"><b>El resultado no confirma la hipótesis de partida, y se reporta como salió.</b> La
sospecha que motivó este trabajo era que el corte único <i>sobre-estimaba sistemáticamente</i> la
robustez. La medición no lo sostiene: frente a la mediana de las ventanas honestas la diferencia es
<b>{_n(o["median"], 3)}</b> —indistinguible de cero— y el rango va de {_n(o["min"], 2)} a
+{_n(o["max"], 2)}. El corte único no está sesgado al alza: es <b>arbitrario</b>, y regala o castiga
según dónde caiga el corte. Lo que sí sostiene la evidencia son otras tres cosas, y la tercera es la que
decide.</div>
<h4>1 · La cola no existía</h4>
<p>Frente al CVaR@25% de las ventanas, el corte único puntúa <b>{_n(tail["median"], 3)}</b> más alto
(IQR {_n(tail["p25"], 2)} … {_n(tail["p75"], 2)}), y positivo en más de tres de cada cuatro unidades. No
es un sesgo del corte: es que <b>el CVaR de un solo número es ese número</b>. El sistema rankea por la
cola mala (§4.6) y, con una sola ventana, no había cola que promediar — el estadístico robusto se
degradaba silenciosamente a "el resultado del último 30% del rango". Ésta es la brecha estructural, y es
la que justifica el cambio.</p>
<h4>2 · El ruido temporal es mayor que la señal que se quiere medir</h4>
<p>Entre ventanas de una misma muestra el headline se mueve una desviación típica de
<b>{_n(std["median"], 3)}</b>, con un rango mediano de <b>{_n(v["range"]["median"], 2)}</b> puntos entre
la mejor y la peor. Puesto en contexto: dentro de una misma muestra, lo que separa a la mejor de la peor
<b>configuración</b> es <b>{_n(v["svn"]["config_spread_walk_forward"]["median"], 2)}</b>. Es decir, mover
la ventana mueve el resultado <b>{_n(v["svn"]["ratio"], 1)} veces más</b> que cambiar de estrategia.</p>
<div class="why">Ése es el argumento entero, en una frase: si el ruido temporal es tres veces la señal que
se quiere medir, elegir con <b>una</b> ventana es elegir mayoritariamente por el tramo de historia que
tocó. No hace falta que el corte único esté sesgado para que sea una mala regla de decisión — basta con
que sea arbitrario, que es justo lo que resultó ser.</div>
<p>CPCV estima la dispersión más baja ({_n(v["std_cpcv"], 2)}) porque sus ventanas solapan y cada una
cubre más calendario, así que promedian más historia — no porque el mundo sea más estable.</p>
<h4>3 · La elección cambia</h4>
<p>Lo que decide no es el nivel sino el <b>orden</b>. El acuerdo de rangos (Spearman) entre ordenar las
configuraciones por el corte único y ordenarlas por la recompensa multiventana tiene mediana
<b>{_n(v["rank"]["walk_forward"]["median"], 2)}</b> pero media
<b>{_n(v["rank"]["walk_forward"]["mean"], 2)}</b> y baja hasta
<b>{_n(v["rank"]["walk_forward"]["min"], 1)}</b> (orden exactamente invertido). La configuración ganadora
cambia en <b>{v["flips"]["walk_forward"]}/{v["flips"]["n_samples"]}</b> muestras
({_n(v["flips"]["walk_forward_pct"], 0)}%) con walk-forward y en
<b>{v["flips"]["cpcv"]}/{v["flips"]["n_samples"]}</b> ({_n(v["flips"]["cpcv_pct"], 0)}%) con CPCV.</p>
<div class="note"><b>Límites declarados, y son serios.</b> {v["n_units"]} unidades sobre
{v["n_configs"]} configuraciones y {v["n_samples"]} muestras de una sola librería, un camino por
escenario. El acuerdo de rangos y los cambios de elección se miden sobre <b>{v["flips"]["n_samples"]}
muestras</b>: "{v["flips"]["walk_forward"]} de {v["flips"]["n_samples"]}" es una señal, no una tasa — el
intervalo de confianza de esa proporción cubre medio rango, y afirmar "la mitad de las veces" como cifra
sería sobre-leer el dato. Las ventanas de un mismo esquema <b>comparten historia</b> (en CPCV cada tramo
entra en varios folds), así que la dispersión medida no son observaciones independientes y no debe leerse
como un error estándar. Y el camino que usa el optimizador (§4.13) sigue puntuando cada muestra con el
corte único: cablear ahí el esquema multiventana es la evolución pendiente. Los
{v["leakage"]["folds_audited"]} folds ejecutados pasaron la auditoría de fuga temporal. Evidencia completa
en <span class="mono">data/validation/report_{v["library"]}.json</span> ({v["generated_at"]}); un backtest
suelto se corre con <span class="mono">ai-trader backtest --validation cpcv</span>.</div>"""


def _fidelity_block(f):
    """Seccion 2.10: ¿se parece este mundo al real? Con la evidencia que lo responde."""
    if not f:
        return (
            "<h3>2.10 · Fidelidad contra el mercado real</h3>"
            "<div class=\"note\"><b>Limitación declarada.</b> El informe de fidelidad "
            "(<span class=\"mono\">data/fidelity/</span>) no está disponible en este árbol, así "
            "que este documento no puede citar las cifras que comparan el mundo sintético con el "
            "histórico real. Regenéralo con "
            "<span class=\"mono\">python -m ai_trader.synthetic.fidelity_study</span>.</div>"
        )
    kurt, exc = f["kurtosis"], f["exceed"]
    clus, ac, vol, cross = f["clustering"], f["autocorr"], f["vol"], f["cross"]
    b, acc = f.get("before"), f.get("acceptance")

    def ratio(row):
        return "—" if row["ratio"] is None else f"{_n(row['ratio'], 2)}×"

    def was(key, field="synth", decimals=2):
        row = (b or {}).get("by_key", {}).get(key)
        return "—" if row is None else _n(row[field], decimals)

    return f"""
<h3>2.10 · Fidelidad contra el mercado real: el test que el generador puede fallar</h3>
<p>Las secciones anteriores comparan el mundo sintético <b>consigo mismo</b>: una librería tiene colas
y agrupamiento donde otra no los tenía. Eso no dice que los tenga <b>en la magnitud del mercado</b>.
Esta sección responde esa pregunta contra el histórico diario real de
<span class="mono">{f["exchange"]}</span> ({f["start"]} → {f["end"]}, {f["n_symbols"]} criptomonedas,
{f["n_pairs"]} pares), midiendo exactamente las mismas magnitudes sobre los dos mundos.</p>
{f'''<p>Se publican <b>dos</b> librerías medidas con el mismo harness y la misma ventana real:
<span class="mono">{b["library"]}</span>, el generador cuyo hueco se midió, y
<span class="mono">{f["library"]}</span>, el que lo cierra. El "antes" no es decoración: sin control,
una corrección medida no se distingue de una afirmación. La cobertura media pasa de
<b>{_n(b["coverage_mean_pct"], 0)}%</b> a <b>{_n(f["coverage_mean_pct"], 0)}%</b>.</p>''' if b else ""}
<div class="why"><b>Dos decisiones metodológicas que hacen que la comparación signifique algo.</b>
<ul>
<li><b>Misma longitud de muestra.</b> El histórico real se trocea en ventanas de {f["window_days"]}
días — el mismo horizonte que un camino sintético — y se compara mediana contra mediana. La
autocorrelación y sobre todo la curtosis son estimadores <b>sesgados en muestras cortas</b>: medir el
real sobre ocho años seguidos y el sintético sobre dos compararía el sesgo, no el mundo. Las ventanas
avanzan {f["step_days"]} días, así que <b>solapan</b>: dan una tendencia central mejor, no
estimaciones independientes, y se declara por eso.</li>
<li><b>Ventana histórica cerrada.</b> El periodo descargado es una constante del estudio, no "hasta
hoy": si el final se moviera con la fecha de ejecución, dos regeneraciones no serían comparables y el
informe no sería reproducible. Los datos se cachean en disco (la misma caché que usa el sistema en
vivo), de modo que el estudio se puede repetir sin red.</li>
</ul></div>
<p>Y tres ejes de lectura, deliberadamente distintos, porque un generador puede fallar en uno y
acertar en otro:</p>
<ul>
<li><b>Nivel</b> (columna <i>ratio</i>): ¿la magnitud es la del mercado? 1,00 sería clavarlo.</li>
<li><b>Ordenación</b> (<i>rank corr</i>): correlación de rangos de Spearman entre la sección cruzada
real y la sintética. Es <b>invariante a la escala</b>, así que responde a algo que el nivel no puede:
si el mundo sintético sabe <i>qué</i> activo tiene más cola o <i>qué</i> par está más acoplado, aunque
el nivel absoluto esté mal calibrado.</li>
<li><b>Cobertura</b>: qué fracción de los valores reales cae dentro del rango [p10, p90] que el
ensemble sintético produce para ese mismo activo. Un generador honesto no tiene que acertar el número
real: tiene que poder <b>producirlo</b> como una realización plausible.</li>
</ul>
{_fidelity_table(f["rows"], b["library"] if b else None)}
{f'''<h4>El test de aceptación: umbrales que el estudio puede fallar</h4>
<p>Un estudio que no puede salir mal no es evidencia. El harness contrasta cada medición con umbrales
declarados en el código (<span class="mono">synthetic/fidelity.py</span>) y <b>devuelve error</b> si no
se cumplen, de modo que una regresión en el generador rompe el comando en vez de pasar desapercibida.
Son dos familias: <b>cobertura</b> —el valor real de cada activo cae dentro del [p10, p90] del ensemble
en al menos {_n(acc["min_coverage_pct"], 0)}% de los activos— y <b>mediana de mercado</b> —la mediana
real de la sección cruzada cae dentro de la banda del ensemble entero— en los tres hechos que la
corrección ataca.</p>
{_acceptance_table(acc)}''' if acc else ""}
<h4>Qué se cambió, y qué movió cada cosa</h4>
<p>Tres cambios en la física del generador, ninguno en los escenarios: la librería realista se deriva de
los mismos <span class="mono">spec.json</span> con un retrofit determinista, sin volver a llamar a la IA.
Las constantes salen de iterar este mismo harness como función objetivo, no de una intuición.</p>
<ol>
<li><b>La calma no es gaussiana.</b> La versión anterior daba <span class="mono">tail_dof = 0</span>
—normal <i>exacta</i>— a toda fase tranquila, que es la mayor parte de un horizonte de
{f["window_days"]} días; las colas gruesas sólo existían en las crisis. Ahora toda fase tiene cola de
Student (calma 5, elevada 4,5, crisis 4), porque el cripto real tiene curtosis de 4 para arriba también
en ventanas tranquilas: la cola gruesa no es una propiedad de las crisis, es una propiedad del proceso.
Curtosis {was("excess_kurtosis")} → <b>{_n(kurt["synth_median"], 2)}</b>, exceedances
{was("exceed_3sigma_pct")}% → <b>{_n(exc["synth_median"], 2)}%</b>.</li>
<li><b>Más reacción a la noticia en el GARCH.</b> El reparto entre noticia de ayer e inercia estaba
clavado a 0,15 / 0,85 en el motor; ahora es un campo del <i>spec</i>
(<span class="mono">vol_news</span>) calibrado a 0,25. Los dos pesos siguen sumando la persistencia
total, así que la varianza incondicional sigue siendo 1: sube el agrupamiento <b>medible</b>, no el
nivel de riesgo. Clustering {was("ac_abs1", decimals=3)} → <b>{_n(clus["synth_median"], 3)}</b>
(real {_n(clus["real_median"], 3)}).</li>
<li><b>Cargas que suben en el pánico.</b> Con betas congeladas, la correlación entre activos de un
modelo de factores es una constante del universo: <b>no puede</b> dispararse en las caídas, que es el
hecho de mercado más caro de ignorar. Cada fase declara ahora un <span class="mono">beta_stress</span>
y las betas del día se escalan con él — y la volatilidad diaria que calibra mechas y huecos usa la beta
<b>efectiva</b>, no la congelada, o el rango intradía dejaría de casar con la vol de ese día. La
covarianza sigue siendo <span class="mono">B_t Σ B_t' + D</span> con D diagonal positiva, así que sigue
siendo definida positiva por construcción. Correlación cruzada {was("cross_corr", decimals=3)} →
<b>{_n(cross["synth_median"], 3)}</b> (real {_n(cross["real_median"], 3)}).</li>
</ol>
<div class="why"><b>Por qué esto no rompe nada de lo ya publicado.</b> Los dos campos nuevos son
neutros por defecto y las rutas antiguas se conservan bit a bit: la vía gaussiana <i>exacta</i> —con el
mismo consumo de RNG— sigue siendo la ruta por defecto cuando ninguna fase pide colas, y el reparto del
GARCH vive en el <i>spec</i>, no en una constante del motor. Las librerías anteriores se regeneran desde
sus <span class="mono">spec.json</span> byte a byte, y hay tests que lo congelan con un hash.</div>
<h4>Resultado 1 · Las colas y el agrupamiento ya están en la magnitud del mercado</h4>
<p>Curtosis en exceso <b>{_n(kurt["synth_median"], 2)}</b> frente a {_n(kurt["real_median"], 2)} real,
con {_n(kurt["coverage_pct"], 0)}% de cobertura{f" (antes {was('excess_kurtosis', 'coverage', 0)}%)" if b else ""};
exceedances más allá de 3σ {_n(exc["synth_median"], 2)}% frente a {_n(exc["real_median"], 2)}%
(recordatorio: bajo una normal serían 0,27%); agrupamiento de volatilidad
{_n(clus["synth_median"], 3)} frente a {_n(clus["real_median"], 3)}. Y el <b>nivel de riesgo</b> sigue
donde estaba: volatilidad anualizada {_n(vol["synth_median"], 0)}% sintética frente a
{_n(vol["real_median"], 0)}% real ({ratio(vol)}). Eso último es lo que hace que esto sea una corrección
y no un cambio de escala: se podían haber engordado las colas subiendo la volatilidad, y no es lo que
pasó.</p>
<h4>Resultado 2 · El co-movimiento mejora pero no llega, y es una decisión</h4>
<p>La correlación cruzada media es {_n(cross["synth_median"], 3)} frente a
{_n(cross["real_median"], 3)} real, con {_n(cross["coverage_pct"], 0)}% de cobertura sobre los
{cross["n"]} pares y una correlación de rangos de {_n(cross["rank_corr"], 2)}: el modelo de factores
<b>ordena</b> los pares como la realidad, con un nivel algo más débil. Cerrar el resto exigiría subir
más las betas de estrés, y la beta entra <b>al cuadrado</b> en la varianza: se compraría acoplamiento
inflando la volatilidad total, que hoy está en su sitio. Se prefiere el nivel de riesgo correcto con
algo menos de co-movimiento que lo contrario.</p>
<h4>Resultado 3 · Límite declarado: la mediana del mercado, no los años de manía</h4>
<div class="why">El p90 de la curtosis real de cripto va de 30 a 90 (DOGE y XRP en sus años de manía).
<b>Ni con <span class="mono">dof = 4</span> se reproduce eso</b>, y perseguirlo rompería el nivel de
volatilidad, que es la propiedad que sostiene todo lo demás. Por eso el umbral de aceptación está sobre
la <b>mediana</b> de la sección cruzada y no sobre su cola: lo que este generador promete es el cripto de
un año cualquiera, no la historia de sus outliers.
<p><b>Consecuencia honesta:</b> lo que se mida sobre este sustrato ya no subestima la pérdida de cola
del mercado típico —los drawdowns, los huecos que se saltan un stop y el peor cuartil que puntúa el CVaR
(§4) están ahora en el orden correcto— pero sigue siendo optimista para un escenario de manía. Ésa es la
frontera del generador, y se declara en vez de disimularse.</p>
<p>Hay un segundo límite medido, y va en dirección contraria a la mejora: la <b>ordenación</b> de las
colas y el agrupamiento entre activos (<i>rank corr</i> en la tabla) es floja e incluso negativa. El
mundo sintético ya produce la magnitud correcta, pero no sabe <i>qué</i> activo tiene más cola: en el
mercado real son los más ruidosos (DOGE, XRP) y en el generador el ruido idiosincrático —al que se le
aplica un AR(1) por régimen— <b>blanquea</b> justo esa estructura. Es el siguiente hilo del que tirar, y
la cobertura y la mediana no lo ocultan: por eso las tres lecturas se publican por separado.</p></div>
<h4>Resultado 4 · La autocorrelación se lee al revés que las demás</h4>
<p>Autocorrelación real {_n(ac["real_median"], 3)} frente a {_n(ac["synth_median"], 3)} sintética. Aquí
un ratio lejos de 1 <b>no es un defecto: es el diseño</b>. El mercado no regala estructura serial —si la
regalara sería dinero gratis— mientras que el generador la fija a propósito, con signo según el régimen
(§2.8), para que la reversión a la media y el momentum tengan algo real que capturar. Lo que sí es una
advertencia es la magnitud: el <i>edge</i> sintético es <b>más limpio</b> que cualquier cosa que el
mercado ofrezca, así que un rendimiento medido aquí no se traslada a un rendimiento allí.</p>
<div class="note"><b>Límites de este estudio.</b> Solo criptomonedas: la renta variable del universo va
por otro proveedor y otra sesión de mercado, así que {f["n_symbols"]} activos y {f["n_pairs"]} pares es
todo el ancho disponible; con {f["n_symbols"]} puntos, una correlación de rangos distingue "ordena como
el mercado" de "no ordena", pero no permite comparar dos valores parecidos. Las
{f["n_real_windows"]} ventanas reales solapan y proceden de <b>un único camino de la historia</b> (los
ciclos de 2018-2025), mientras que el lado sintético son {f["n_synthetic_samples"]} mundos distintos:
la comparación es entre "lo que pasó" y "lo que podría pasar", y esa asimetría no se puede eliminar,
solo declarar.{" Sin contraparte real: " + ", ".join(f["missing"]) + "." if f["missing"] else ""}</div>
<p class="tag">Evidencia completa: <span class="mono">data/fidelity/report_{f["library"]}.json</span> ·
reproducible con <span class="mono">python -m ai_trader.synthetic.fidelity_study</span>
({f["n_scenarios"]} escenarios × {f["n_paths"]} caminos; {f["generated_at"]}).</p>"""


def _signal_channel_block(s):
    """Seccion 4.12: el break-even del IC.

    Va detras de la transferencia y no en el capitulo de datos porque lo que publica es una
    propiedad de la ESTRATEGIA —a partir de que capacidad predictiva una senal paga sus
    costes— y no del generador. Es, ademas, el unico test de FALSACION del radar: sin el,
    una feature de muestra corta puede estar sobreajustada y el sistema no tiene forma de
    saberlo."""
    if not s:
        return (
            "<h3>4.12 · Break-even del IC: cuándo paga una señal</h3>"
            "<div class=\"note\"><b>Limitación declarada.</b> El informe del canal sintético "
            "(<span class=\"mono\">data/signal_channel/</span>) no está disponible en este árbol, "
            "así que este documento <b>no puede afirmar</b> desde qué capacidad predictiva una "
            "señal externa paga sus costes. Mientras no exista, el radar de señales entra en la "
            "decisión sin ningún test de falsación. Genéralo con "
            "<span class=\"mono\">python -m ai_trader.scoring.signal_study</span>.</div>"
        )

    be = s["break_even"]
    lead = (be["by_lead"] or [{}])[0]
    ch, crit, rep = s["channel"], s["criterion"], s["reproduction"]
    on = [r for r in s["rows"] if r["arm"] != "off"]
    top = on[-1] if on else {}
    reached = lead.get("break_even_rho") is not None
    answer = (
        f"ρ = {_n(lead.get('break_even_rho'), 2)}"
        if reached
        else f"por encima de {_n(lead.get('break_even_above'), 2)}"
    )

    rows = "".join(
        f"<tr><td class=mono>{r['cell_id']}"
        f"{' <span class=tag>(control)</span>' if r['arm'] == 'off' else ''}</td>"
        f"<td class='n mono'>{'—' if r['expected_ic'] is None else _n(r['expected_ic'], 3)}</td>"
        f"<td class='n mono'>{'—' if r['measured_ic'] is None else _n(r['measured_ic'], 3)}</td>"
        f"<td class=mono>{r['selected'] or '—'}</td>"
        f"<td class='n mono'>{_n(r['reward'], 3)}</td>"
        f"<td class='n mono'>{_n(r['baseline'], 3)}</td>"
        f"<td class='n mono'>{_n(r['margin'], 3)}</td>"
        f"<td class='n mono'>{r['n_beating']}/{s['n_configs']}</td>"
        f"<td>{'<b>sí</b>' if r['beats'] else 'no'}</td></tr>"
        for r in s["rows"]
    )

    # La curva de UNA configuracion fija a traves de las celdas: quita el ruido de que la
    # elegida cambie de celda, y es lo que separa "opera menos" de "opera mejor".
    gc = s.get("gate_cost") or {}
    curve_id = gc.get("off_selected")
    curve = (gc.get("by_config") or {}).get(curve_id) or {}
    curve_html = ""
    if curve:
        cells = [r["cell_id"] for r in s["rows"]]
        curve_html = (
            "<table><thead><tr><th>" + curve_id + "</th>"
            + "".join(f"<th class=n>{c}</th>" for c in cells)
            + "</tr></thead><tbody>"
            + "<tr><td><b>recompensa OOS</b></td>"
            + "".join(f"<td class='n mono'>{_n(curve['rewards'].get(c), 3)}</td>" for c in cells)
            + "</tr><tr><td>operaciones/ventana</td>"
            + "".join(
                f"<td class='n mono'>{_n(curve['trades_per_window'].get(c), 1)}</td>"
                for c in cells
            )
            + "</tr></tbody></table>"
            "<p>La puerta corta prácticamente las <b>mismas</b> entradas en todas las celdas con "
            "canal: lo único que cambia entre ρ = 0 y el extremo de la rejilla es <i>cuáles</i>, y "
            "eso vale medio punto de recompensa. Es lo que descarta la explicación alternativa —que "
            "puntúe mejor por operar menos—.</p>"
        )

    control = (
        "El grupo de control salió <b>limpio</b>: con ρ = 0 la estrategia <b>no</b> bate al "
        "baseline, así que lo que se mide en las demás celdas es información y no el ruido "
        "persistente del canal ni el efecto de operar menos."
        if be["control_clean"]
        else "El grupo de control salió <b>sucio</b>: con ρ = 0 la estrategia ya bate al baseline, "
        "así que el barrido <b>no publica break-even</b>. Lo que estaría midiendo es la puerta, "
        "no la señal."
    )
    return f"""
<h3>4.12 · Break-even del IC: desde qué capacidad predictiva paga una señal</h3>
<p>El radar del §2.2 mete diecisiete fuentes en la decisión, y hasta aquí su única defensa contra el
sobreajuste era <b>negativa</b>: ninguna feature entra en el espacio de búsqueda, así que el optimizador
no puede ajustar umbrales contra el resultado. Eso limita los grados de libertad pero <b>no mide
nada</b>. Esta sección es la medición, y contesta una sola pregunta: <b>¿a partir de qué capacidad
predictiva ρ una señal externa hace que la estrategia bata al baseline después de costes?</b></p>
<p>Sobre el mundo sintético, cuyo futuro ya está escrito, se simula el <b>canal de observación</b> —no la
señal—:</p>
<p class="mono" style="text-align:center">señal_t = ρ · z(retorno_t→t+h) + √(1−ρ²) · ruido_t</p>
<div class="why"><b>Por qué el canal y no la señal.</b> Simular «el sentimiento de Twitter» —su nivel, su
estacionalidad, su reacción a un hack— pide un generador aprendido de datos, y con él vuelve la
circularidad que este proyecto evita desde el principio: el mundo contra el que se mide lo habríamos
ajustado nosotros. El canal cuesta <b>cinco números interpretables</b>, y lo que se publica no son «los
mejores parámetros» sino un <b>umbral</b>: una propiedad del <i>diseño</i> —de esta estrategia, con estos
costes, con esta puerta— y no de ningún histórico. No se puede sobreajustar a datos que nunca
entraron.</div>
<table><thead><tr><th>celda</th><th class=n>IC declarado</th><th class=n>IC medido</th>
<th>elegida en train</th><th class=n>recompensa OOS</th><th class=n>baseline</th><th class=n>margen</th>
<th class=n>configs que baten</th><th>¿bate?</th></tr></thead><tbody>{rows}</tbody></table>
<p><b>Respuesta: el break-even está {answer}</b> con un adelanto de {lead.get('lead_days', 1)} día.
{control}</p>
<div class="why"><b>Cómo se lee, declarado antes de correr.</b> El criterio vive en el código
(<span class="mono">signal_study.CRITERION</span>) y viaja dentro del informe; uno elegido a la vista del
resultado no sería un criterio.
<ul>
<li><b>Selección:</b> {crit["seleccion"]}.</li>
<li><b>Lectura:</b> {crit["lectura"]} ({", ".join(s["split"]["validation"])}).</li>
<li><b>Batir:</b> {crit["batir"]}.</li>
<li><b>Control:</b> {crit["control"]}.</li>
<li><b>Delta:</b> {crit["delta"]}.</li>
</ul></div>
<div class="note"><b>ρ = 0 es el test de falsación que no existía.</b> Una puerta que consulta una señal
<i>sin información</i> sigue haciendo una cosa: <b>operar menos</b>. Y como la recompensa es el CVaR@25%
y un cero le gana a cualquier cosa que arriesgue y pierda (§4.10), operar menos puede parecer talento.
Por eso el valor de la información se lee siempre contra ρ = 0 —misma puerta, sin información— y nunca
contra la celda sin puerta.</div>
<div class="why"><b>«Sin canal» es el cero del eje, no una celda rival.</b> La puerta sólo puede
<i>quitar</i> entradas, así que la curva de ρ arranca por debajo de ella y sube. Cuánto cuesta ese
filtro <b>depende del régimen</b>, y por eso el informe lo publica en los dos lados del hold-out:
{_n(gc.get("delta_validation"), 2)} puntos en validación —mercado subiendo— y
{_n(gc.get("delta_train"), 2)} en train —mercado cayendo, donde filtrar al azar reduce exposición y por
tanto ayuda—. Leer sólo un lado convertiría una propiedad del tramo en una constante del sistema. Lo
que <b>no</b> depende del régimen es la monotonía en ρ, y es lo que sostiene el break-even.{curve_html}
</div>
<div class="why"><b>0 = menos edge, nunca más.</b> Los mandos del canal van en positivo
(<span class="mono">informative_share</span>, <span class="mono">coverage</span>) y no en negativo
(<span class="mono">false_positive_rate</span>), para que un default olvidado degrade a «sin señal» y no
a «señal perfecta». Un canal recién construido no emite nada. En el barrido los dos están al máximo
(<span class="mono">{_n(ch["informative_share"], 2)}</span> y
<span class="mono">{_n(ch["coverage"], 2)}</span>) para que ρ <i>sea</i> el IC entregado —medido:
{_n(top.get('expected_ic'), 3)} declarado → {_n(top.get('measured_ic'), 3)} medido en la celda más
informativa—, así que lo publicado es la cota <b>optimista</b>: bajarlos sólo puede empeorar el
break-even.</div>
<div class="note"><b>Tres controles que hacen legible el resultado.</b> (1) La emisión ocurre
<b>después</b> de generar las velas, en un pase aparte y con su propio generador aleatorio: que no
interfiera con el motor no es una promesa auditable a mano, es imposible por construcción. (2) Las
estrategias reciben las señales por el <b>mismo contrato que en vivo</b>
(<span class="mono">attach_signal_provider</span> + <span class="mono">signal_gate_reason</span>, mismo
recorte anti-<i>look-ahead</i>), así que lo que se barre es lo que opera. (3) La celda sin canal
reproduce {rep.get("compared", 0)} unidades del §4.11
<b>{"score a score" if rep.get("identical") else "con discrepancias"}</b>, que es la prueba de que la
costura del canal no movió nada del sistema.</div>
<div class="note"><b>Separación por función, no por ventana.</b> Aquí no entra un solo dato real, y es
deliberado: el sintético es el sustrato de <b>selección</b> (barrer ρ, rankear, decidir la puerta) y el
real el de <b>verificación</b> (medir el ρ de una señal de verdad y estudiar la transferencia). Nunca el
mismo dato haciendo las dos cosas. Al mercado real le queda por tanto una sola pregunta, y es binaria:
<i>el ρ que mide mi señal, ¿está por encima de este umbral y con margen?</i></div>"""


# Etiqueta legible de cada familia. Antes era un ternario dentro de la tabla —"reversión si
# es mean_reversion, si no momentum"— que con dos familias funcionaba y con tres habria
# etiquetado en SILENCIO como "momentum" a todo lo demas. Un informe que miente sin fallar es
# peor que uno que revienta, asi que el fallback es ahora el nombre crudo de la familia.
FAMILY_LABELS: dict[str, str] = {
    "crypto_momentum": "momentum",
    "mean_reversion": "reversión",
    "liquidation_cascade": "liquidación",
    "vol_term_structure": "volatilidad",
    "event_calendar_drift": "calendario",
    "attention_ignition": "atención",
    "flow_persistence": "flujo",
    "signal_composite": "compuesta",
}


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


def _families_phrase(families) -> str:
    """Como se nombra en prosa la rejilla que un informe DECLARA.

    Se deriva del informe y no se escribe a mano: es la unica forma de que la frase siga
    siendo cierta cuando se publiquen dos informes con rejillas distintas al lado.
    """
    names = list(families or [])
    if not names:
        return "las familias declaradas"
    if len(names) <= 2:
        return "las dos primitivas de precio"
    return f"las {len(names)} familias ({', '.join(_family_label(n) for n in names)})"


VERDICT_LABEL = {
    "la_capa_ayuda": "la capa ayuda",
    "la_capa_resta": "la capa resta",
    "indistinguible": "indistinguible",
    "sin_potencia": "sin potencia",
}


def _themes_real_block(t) -> str:
    """Seccion 4.16: la capa de senal encendida sobre ARCHIVO REAL, comparacion pareada."""
    if not t:
        return ('<h3>4.16 · La capa temática contra señal real</h3>\n'
                '<p class="muted">Sin informe publicado. Se genera con '
                '<span class="mono">python -m ai_trader.scoring.theme_study --offline</span>.</p>')

    filas = []
    for fam in t["families"]:
        d = fam["paired_difference"]
        banda = ("sin muestra" if d["lo"] is None
                 else f"[{_n(d['lo'], 3)}, {_n(d['hi'], 3)}]")
        filas.append(
            f"<tr><td class=\"mono\">{fam['family']}</td>"
            f"<td class=\"mono\">{fam.get('gate_param', '—')}</td>"
            f"<td class=\"n\">{_n(fam['blind_mean'], 3)}</td>"
            f"<td class=\"n\">{_n(fam['armed_mean'], 3)}</td>"
            f"<td class=\"n mono\">{banda}</td>"
            f"<td class=\"n\">{fam['n_windows_where_the_layer_moved']}/{fam['n_pairs']}</td>"
            f"<td><b>{VERDICT_LABEL.get(fam['verdict'], fam['verdict'])}</b></td></tr>"
        )
    for skip in t["families_skipped"]:
        filas.append(
            f"<tr><td class=\"mono\">{skip['family']}</td>"
            f"<td colspan=\"5\">{skip['reason']}</td>"
            f"<td><b>no evaluable</b></td></tr>"
        )

    cob = []
    for name, m in sorted((t["themes"].get("measured") or {}).items()):
        ev = name in (t["themes"].get("evaluable") or [])
        de = name in (t["themes"].get("declared_evaluable") or [])
        dis = name in (t["themes"].get("disagreement") or [])
        cob.append(
            f"<tr><td class=\"mono\">{name}</td>"
            f"<td class=\"n\">{_n(m['max_coverage'], 3)}</td>"
            f"<td class=\"n\">{_n(100 * m['readable_share'], 1)}%</td>"
            f"<td>{'evaluable' if ev else 'ciego'}</td>"
            f"<td>{'evaluable' if de else 'ciego'}"
            f"{' <b>← no coinciden</b>' if dis else ''}</td></tr>"
        )

    ventanas = "".join(
        f"<tr><td class=\"mono\">{w['label']}</td><td>{w['start']}</td><td>{w['end']}</td>"
        f"<td class=\"n\">{len(w.get('symbols', []))}</td></tr>"
        for w in t["windows"]
    )

    return f"""
<h3>4.16 · La capa temática contra señal real</h3>
<p>Todo lo demás que este documento mide sobre señales lo mide en un canal <b>sintético</b>, donde la
capacidad predictiva se fija por construcción. Este estudio no: enciende la capa temática sobre el
<b>archivo real capturado</b> y compara cada familia <b>consigo misma</b>. La diferencia es pareada
—misma configuración, misma ventana, mismas barras— y lo único que cambia entre las dos piernas es el
umbral de la puerta y si el archivo llega al motor. Por eso mide la capa y no la elección de estrategia.
{t['n_failed_units']} unidades fallidas.</p>
<table><thead><tr><th>familia</th><th>puerta</th><th class="n">ciego</th><th class="n">armado</th>
<th class="n">intervalo por bloques</th><th class="n">movió en</th><th>veredicto</th></tr></thead>
<tbody>{''.join(filas)}</tbody></table>
<p class="muted">«Movió en» cuenta las parejas en las que armar la puerta cambió algo. Por debajo de
{t['min_paired_windows']} el veredicto es <b>sin potencia</b>: no se publica una diferencia que el
ruido explica, exactamente como en el estudio de divergencia.</p>

<div class="why"><b>Las cuatro reservas van aquí, no en una nota al pie.</b> Se contrastan
{len(t['families'])} familias y <b>no hay corrección por comparaciones múltiples</b>: el intervalo de
<span class="mono">flow_persistence</span> empieza en +0,013 y no sobreviviría a una. Los pares son
4 configuraciones × 5 ventanas y las configuraciones de una misma familia están correlacionadas, así que
el <b>N efectivo es menor que 20</b> y el intervalo por bloques resulta <b>optimista</b>. El compuesto,
ciego, es un seguidor de tendencia corriente —su núcleo de precio es deliberadamente mínimo—, de modo
que <i>que su capa ayude</i> es casi su definición: lo medido es la magnitud, no la dirección. Y las
cinco ventanas no comparten universo, de 8 símbolos en la más antigua a {t['n_symbols']} en la última,
así que las parejas no pesan igual.</div>

<h4>Qué temas se pueden leer hacia atrás, medido y no declarado</h4>
<p>La evaluabilidad se <b>mide sondeando el radar sobre el archivo</b> en vez de derivarse del flag
<span class="mono">backtestable</span> del catálogo. No es una distinción teórica: el catálogo se
equivoca en los dos sentidos, y las dos veces con consecuencias.</p>
<table><thead><tr><th>tema</th><th class="n">cobertura máxima</th><th class="n">sondas legibles</th>
<th>medido</th><th>declarado</th></tr></thead><tbody>{''.join(cob)}</tbody></table>
<div class="note"><b>El desacuerdo es el dato.</b> <span class="mono">vol_surface</span> figura como
ciego en el catálogo y <b>se lee</b>: <span class="mono">deribit_volatility</span> publica desde
<b>2021-03-24</b> y el tema alcanza 0,333 de cobertura, por encima del 0,25 que exige la puerta. El
catálogo lo da por no backtestable solo porque su profundidad <i>medida</i> aún no llega a los 365 días
que exige <span class="mono">depth.MIN_MEASURED_DAYS</span>. En sentido contrario,
<span class="mono">cex_listings</span> es backtestable y pertenece a <span class="mono">attention</span>,
pero es un calendario de listados y BTC no se lista: sobre ese símbolo no produce lectura. Publicar la
derivación del catálogo como si fuera una medición es lo que llevó a excluir una familia con un motivo
que resultó falso.</div>

<h4>Las ventanas y su universo</h4>
<p>El universo de cada sub-ventana se resuelve <b>una vez</b> antes de repartir el trabajo, con el mismo
criterio de histórico que §4.11, y viaja declarado dentro de la tarea. Que en la ventana más antigua
falten los pares jóvenes es correcto; lo que no valdría es que lo decidiera la disponibilidad del
proveedor dentro de cada proceso, porque entonces las dos piernas de una comparación pareada podrían
correr sobre universos distintos sin que nada avisara.</p>
<table><thead><tr><th>ventana</th><th>desde</th><th>hasta</th><th class="n">símbolos</th></tr></thead>
<tbody>{ventanas}</tbody></table>
"""


def _extended_grid_block(e) -> str:
    """Seccion 4.17: que cambia al pasar de 16 candidatos a 64."""
    if not e:
        return ""
    sig, val, w = e["signal"], e["validation"], e["weights"]
    igual = ("<b>idéntico campo a campo</b> al del informe congelado"
             if sig["identical_to_frozen"] else "distinto del congelado")
    return f"""
<h3>4.17 · Qué cambia al pasar de 16 candidatos a 64</h3>
<p>Los tres estudios que dependen de la rejilla se repitieron con las ocho familias. Ninguno sustituye
al congelado: se publican al lado, porque lo medido con dos primitivas sigue siendo cierto sobre lo que
midió y borrarlo haría imposible ver qué se mueve al ampliar.</p>

<h4>Break-even del IC: no se mueve nada</h4>
<p>Con {sig['n_configs']} configuraciones, el bloque de break-even sale {igual}. El veredicto sigue
siendo <b>{sig['verdict'].replace('_', ' ')}</b> y la puerta binaria sigue costando
<b>{_n(sig['gate_cost'], 3)}</b> puntos de recompensa por sí sola. Las 48 candidatas nuevas están en el
desglose por configuración y <b>no mueven ni un margen</b>.</p>
<div class="why"><b>Y no es que no compitieran.</b> Ninguna fue descartada, y las temáticas
<i>operan más</i> que las publicadas: <span class="mono">event_calendar_drift</span> abre unas 108
operaciones por ventana frente a las 48 de momentum y las 7 de reversión. Compitieron en igualdad y
ninguna gana una sola celda. Lo único que responde a la señal es el compuesto, que no aparece en el
top-10 de la celda ciega y sube al <b>puesto 2</b> en la celda de ρ = 0,20: su posición se mueve con la
fuerza de la señal, que es exactamente lo que su diseño predice. No basta para batir al baseline —de
hecho <b>ninguna de las 64 lo bate en ninguna celda</b>—. Comprobaciones: determinismo limpio, control
ρ = 0 limpio, y la celda ciega reproduce {'' if sig['reproduction'] else 'NO reproduce '}las 512
unidades del estudio de transferencia una a una.</div>

<h4>Validación temporal: la conclusión aguanta, la arbitrariedad crece</h4>
<p>Con {val['n_configs']} configuraciones, el optimismo del corte único frente a la <b>cola</b> sale
<b>{_n(val['vs_tail'], 3)}</b>, contra el <b>{_n(val['frozen_vs_tail'], 3)}</b> publicado con cuatro.
Que ese número no se mueva al cuadruplicar la rejilla es la corroboración más fuerte que este estudio
podía dar. Lo que sí crece es la arbitrariedad de la elección: el ganador del corte único deja de serlo
en <b>{val['flips']['walk_forward']} de {val['flips']['n_samples']}</b> escenarios con walk-forward
(antes {val['frozen_flips']['walk_forward']}) y en <b>{val['flips']['cpcv']} de
{val['flips']['n_samples']}</b> con CPCV (antes {val['frozen_flips']['cpcv']}).
{val['folds_audited']:,} folds auditados sin fuga.</p>
<div class="why"><b>Comprobado que no son empates.</b> Hay 20 filas de 128 con todos los folds a cero
—configuraciones que no abren una sola operación— y un vuelco entre estrategias que no operan no es un
vuelco. Rehecho el cálculo excluyéndolas, los vuelcos salen <b>idénticos</b> y el hueco contra la cola
sube a +1,72: el efecto es real y, restringido a lo que de verdad opera, mayor.</div>

<h4>Pesos del headline: aquí sí se cae una conclusión publicada</h4>
<p>Sobre el subconjunto activo ({w['n_active']} de {w['n_configs']} configuraciones con actividad
suficiente), el mejor punto es <b>λ = {_n(w['best'][0], 2)}, κ = {_n(w['best'][1], 2)}</b> con un rank IC
de <b>{_n(w['best_ic'], 4)}</b> frente a {_n(w['base_ic'], 4)} sin penalizar: una ganancia de
<b>{_n(w['gain'], 4)} ± {_n(w['gain_se'], 4)}</b>. Lo publicado con dos familias decía que penalizar
<i>no estabiliza</i>, y allí todas las penalizaciones empeoraban el rank IC (su óptimo era
λ = {_n(w['frozen_best'][0], 2)}, κ = {_n(w['frozen_best'][1], 2)}).</p>
<div class="note"><b>Tres reservas, y la primera es seria.</b> El óptimo cae en la <b>esquina</b> de la
rejilla probada, así que no está acotado: lo medido es «más penalización es mejor que menos dentro de lo
probado», no que el óptimo sea 4. Segunda: penalizar <b>cambia la elección</b> hacia un candidato con
Sharpe de validación <i>menor</i> (1,49 frente a 1,72), de modo que los pesos que más estabilizan el
orden no eligen mejor. Tercera: <b>no se adopta</b>. Mover λ cambiaría retroactivamente quién es
rankeable en informes ya publicados, y eso es su propia evolución con su propio coste.</div>
<p>Una coherencia que antes no existía: la auditoría de costes da un λ implícito de
<b>{_n(w['implied_lambda'], 2)}</b> sobre las activas, prácticamente el mismo
{_n(w['frozen_implied_lambda'], 2)} publicado. Con dos familias el óptimo empírico era 0 y el implícito
6,3 —se contradecían—; con ocho, el óptimo empírico se acerca al que los costes ya imponen.</p>
"""


def _transfer_extended_block(t) -> str:
    """Seccion 4.11-bis: la misma pregunta con OCHO familias, y el control que la atribuye."""
    if not t:
        return ""
    residue = t.get("control_residue")
    control = (
        "<p><b>El control de rejilla devuelve el informe idéntico campo a campo</b> —las "
        f"{t['n_configs']} configuraciones, cada score y cada intervalo— salvo el identificador "
        "de librería: <b>el efecto del mundo es exactamente cero</b> y todo el cambio es de "
        "rejilla.</p>"
        if residue == 0
        else (
            f"<p>El control de rejilla difiere en {residue} bloques del informe extendido, así "
            "que parte del cambio SÍ es del sustrato y hay que leerlo con cuidado.</p>"
            if residue is not None
            else "<p class=muted>Control de rejilla no disponible: la comparación con el "
            "estudio congelado <b>no es atribuible</b>.</p>"
        )
    )
    return f"""
<h3>4.11-bis · La misma pregunta con ocho familias</h3>
<p>El estudio de §4.11 habla de las dos primitivas de precio. Repetido sobre
<span class="mono">{t['library']}</span> con las <b>{t['n_families']} familias</b>
({t['n_configs']} configuraciones), el Spearman sale <b>{_n(t['spearman'], 3)}</b> contra un
umbral de {_n(t['threshold'], 2)}: <b>sigue sin transferir</b>. Cuadruplicar los candidatos y
darle al mundo un canal de observación no convierte al sintético en criterio de selección.</p>
{control}
<div class="why"><b>Por qué ese cero es el resultado y no un trámite.</b> Era predecible —las
velas de las dos librerías son idénticas byte a byte y en este estudio la capa de señal está
inerte en las ocho familias, así que los canales declarados no se emiten ni se consultan—, y eso
es justo lo que lo hace útil: <b>si el control hubiera dado algo distinto de cero, significaría
una fuga</b>, algún camino colando los canales en un estudio que no debe verlos. Es un test de
falsación del diseño aditivo, y lo pasa. Corolario práctico: cualquier estudio que no inyecte un
proveedor de señales da lo mismo sobre las dos librerías, así que este control no hay que
repetirlo.</div>
<p>Con el residuo en cero, lo que cambia es atribuible a la <b>rejilla</b> y no al sustrato. Y lo
que más cambia es la fila que sostenía una conclusión anterior: el Spearman entre recompensa y
número de operaciones en el lado <b>real</b> pasa de <b>−0,84</b> con dos familias a
<b>+0,004</b> con ocho. Aquel −0,84 no describía al mercado de 2018-2025: describía a dos
primitivas que apenas operaban.</p>
<p>El suelo de actividad de §4.10, re-derivado sobre las {t['n_configs']} configuraciones, elige
<b>{_n(t['activity_floor'], 0)} operaciones por ventana</b> — el mismo valor publicado. No se
adopta nada nuevo: se publica que dos rejillas distintas eligen el mismo número.</p>
"""


def _themed_families_block(families) -> str:
    """
    Las seis tematicas, una tarjeta cada una. Se ITERA sobre lo que el generador publica y no
    se escriben seis bloques: con seis marcas literales por familia, anadir la septima obliga
    a tocar dos ficheros que se desincronizan a la primera.
    """
    if not families:
        return "<p class=muted>Sin familias temáticas declaradas.</p>"
    out = []
    for fam in families:
        params = _rows(fam["params"], ["Parámetro", "Valor"])
        out.append(
            f"<h4>{fam['name']} <span class=tag>tema: {fam['theme']}</span></h4>"
            f"<p>{fam['idea']}</p>"
            f"<details><summary>Parámetros por defecto de "
            f"<span class=mono>{fam['id']}</span></summary>{params}</details>"
        )
    return "\n".join(out)


def _themed_spaces_block(families) -> str:
    """El espacio de busqueda de cada tematica, en §4.13, junto al de las dos de precio."""
    if not families:
        return ""
    out = []
    for fam in families:
        rows = _rows(fam["space"], ["Parámetro", "Rango CEM"])
        out.append(f"<h4>{fam['name']}</h4>{rows}")
    return "\n".join(out)


def _transfer_block(t):
    """Seccion 4.11: ¿ordena el mundo sintetico las estrategias como el real?

    Es la pregunta que §2.10 NO responde, y va en el capitulo de ESTRATEGIAS y no en el de
    datos porque lo que compara son dos ORDENACIONES DE ESTRATEGIAS. La fidelidad mide
    parecido estadistico; esto mide si el orden se traslada, que es lo unico que el
    producto le pide al generador."""
    if not t:
        return (
            "<h3>4.11 · Transferencia de ranking</h3>"
            "<div class=\"note\"><b>Limitación declarada.</b> El informe de transferencia "
            "(<span class=\"mono\">data/transfer/</span>) no está disponible en este árbol, así que "
            "este documento <b>no puede afirmar</b> que el mundo sintético ordene las estrategias "
            "como el mercado. Mientras no exista, la regla vigente es la conservadora: el sintético "
            "no se usa como criterio de selección. Genéralo con "
            "<span class=\"mono\">python -m ai_trader.scoring.transfer_study --offline</span>.</div>"
        )

    v, b, cb = t["verdict"], t["boot"], t["boot_configs"]
    k, ds, val, act = t["top_k"], t["discrepancies"], t["validation"], t["activity"]
    transfers = v["transfers"]
    inverted = v["key"] == "ordenacion_invertida"

    rows = "".join(
        f"<tr><td class=mono>{r['config_id']}"
        f"{'' if r['active'] else ' <span class=tag>(apenas opera)</span>'}</td>"
        f"<td>{_family_label(r['family'])}</td>"
        f"<td class='n mono'>{_n(r['reward_real'], 2)}</td>"
        f"<td class='n mono'>{_n(r['reward_synthetic'], 2)}</td>"
        f"<td class='n mono'>{_n(r['trades_real'], 1)}</td>"
        f"<td class='n mono'>{r['rank_real']}</td>"
        f"<td class='n mono'>{r['rank_synthetic']}</td>"
        f"<td class='n mono'>{'+' if r['delta'] > 0 else ''}{r['delta']}</td></tr>"
        for r in t["rows"]
    )

    ba, pa = act["bootstrap_active"], act["permutation_active"]
    if act["spearman_active"] is None:
        activo = ""
    elif act["spearman_active"] <= -v["threshold"]:
        activo = f"""<p>Y al quitar la inactividad el resultado <b>empeora</b>, no mejora. Sobre las
{act["n_active"]} configuraciones que operan de verdad en los dos mundos —el único subconjunto donde la
pregunta original tiene sentido— el acuerdo no es nulo sino <b>negativo</b>:
ρ = {_n(act["spearman_active"], 3)}. Entre estrategias que actúan, el mundo sintético tiende a
<b>invertir</b> el orden del mercado.
{f'''Con la cautela obligatoria: es un subconjunto <i>post-hoc</i> de {act["n_active"]} puntos y su
intervalo por bloques [{_n(ba["lo"], 2)}, {_n(ba["hi"], 2)}] cruza el cero
(p = {_n(pa["p_value"], 3)}), así que es una señal fuerte y no una prueba. Lo que sí queda descartado es
que el ρ ≈ 0 de la tabla sea un artefacto de la inactividad: al quitarla, el acuerdo no aparece.'''
if ba and not ba["excludes_zero"] else ""}</p>"""
    else:
        activo = f"""<p>Restringido a las {act["n_active"]} configuraciones que operan de verdad en los
dos mundos, el acuerdo es ρ = {_n(act["spearman_active"], 3)}: el veredicto de la tabla no cambia al
quitar las que apenas abren posiciones.</p>"""
    caveats = "".join(
        f"<li><b>{c['title']}.</b> {c['text']}</li>" for c in t["caveats"]
    )

    if transfers:
        veredicto = f"""<div class="note"><b>Veredicto: el orden transfiere.</b> ρ = {_n(t["rho"], 3)}
≥ {_n(v["threshold"], 2)}, así que el generador es un <b>pre-cribado legítimo</b> y el flujo definitivo
del sistema queda fijado: <b>{v["flow"]}</b>. Lo que esto autoriza es a <i>filtrar</i> candidatas barato
en el mundo sintético; lo que no autoriza es a que el ranking que decide salga de ahí.</div>"""
    elif inverted:
        veredicto = f"""<div class="note"><b>Veredicto: el orden se invierte.</b> ρ = {_n(t["rho"], 3)}
≤ −{_n(v["threshold"], 2)}. El mundo sintético no es que no informe: es que ordena <b>al revés</b> que
el mercado, de modo que seguirlo sería peor que elegir al azar. {v["flow"]}.</div>"""
    else:
        veredicto = f"""<div class="note"><b>Veredicto: no hay transferencia medible.</b>
ρ = {_n(t["rho"], 3)}, por debajo del umbral de {_n(v["threshold"], 2)} declarado antes de mirar el
resultado. La consecuencia operativa, y este documento la asume: <b>{v["flow"]}</b>. La librería
sintética sigue siendo válida para lo que sí está medido —fidelidad de los hechos estilizados (§2.10),
banco de estrés y regresión determinista— pero <b>no</b> como juez que ordena candidatas.</div>"""

    return f"""
<h3>4.11 · Ordenación en datos reales y sintéticos: ¿ordena el sintético como el mercado?</h3>
<p>§2.10 responde si el mundo sintético <b>se parece</b> al mercado. No responde lo único que el producto
le pide de verdad: que si una configuración es mejor que otra ahí, <b>tienda a serlo también aquí</b>.
No son la misma pregunta —un generador puede clavar las colas y ordenar al revés, y ninguna métrica de
fidelidad lo detectaría—, así que se mide aparte y con su propio umbral.</p>
<p>El diseño persigue una sola cosa: que entre los dos lados <b>lo único que cambie sea el mundo del que
salen los precios</b>. Las mismas <b>{t["n_configs"]}</b> configuraciones (la rejilla del estudio de
pesos de §4.5: hipercubo latino con semilla {t["study_seed"]} sobre
{_families_phrase(t.get("families"))}), el mismo
config, el mismo universo de <b>{len(t["symbols"])}</b> pares —los que existen a la vez en el mercado y
en la librería—, el mismo esquema CPCV con {val["n_folds"]} ventanas OOS, la misma purga de
{val["purge_days"]} días y el mismo embargo de {val["embargo_days"]}. Y la misma <b>longitud de
ventana</b>: un camino sintético dura {val["window_days"]} días, así que el histórico real
({t["real_start"]} → {t["real_end"]}) se trocea en {t["n_sub_windows"]} sub-ventanas disjuntas de ese
tamaño en vez de evaluarse de una sola pieza. Comparar un Sharpe estimado sobre ocho años con otro
estimado sobre dieciocho meses compararía precisiones, no mundos.</p>
<p>Cada configuración produce muchos scores —sub-ventana × fold en el real, muestra × fold en el
sintético— y todos van a <b>una sola distribución</b>, de la que se toma el CVaR al
{_n(val["cvar_alpha"] * 100, 0)}%. Tomar el CVaR de cada unidad y luego el CVaR de esos CVaR compondría
dos conservadurismos y dispararía la varianza del estimador.</p>
{veredicto}
<table><thead><tr><th>Lectura</th><th class=n>valor</th><th>qué dice</th></tr></thead><tbody>
<tr><td>Spearman de los dos rankings</td><td class='n mono'>{_n(t["rho"], 3)}</td>
  <td>acuerdo global del orden sobre {t["n_configs"]} configuraciones</td></tr>
<tr><td>IC{_n(b["ci_pct"], 0)}% · bootstrap por bloques</td>
  <td class='n mono'>[{_n(b["lo"], 2)}, {_n(b["hi"], 2)}]</td>
  <td>{b["n_blocks_real"]} bloques reales, {b["n_blocks_synthetic"]} sintéticos ·
  {"excluye" if b["excludes_zero"] else "<b>incluye</b>"} el cero</td></tr>
<tr><td>IC{_n(cb["ci_pct"], 0)}% · remuestreando configuraciones</td>
  <td class='n mono'>[{_n(cb["lo"], 2)}, {_n(cb["hi"], 2)}]</td>
  <td>otra pregunta: ¿saldría lo mismo con otras {t["n_configs"]} configuraciones del hipercubo?</td></tr>
<tr><td>p (permutación)</td><td class='n mono'>{_n(t["permutation"]["p_value"], 4)}</td>
  <td>probabilidad de un acuerdo así de fuerte por azar</td></tr>
<tr><td>Top-{k["k"]} del sintético en la mitad buena del real</td>
  <td class='n mono'>{k["hits"]}/{k["k"]}</td>
  <td>la lectura operativa del pre-cribado · por azar {_n(k["expected_by_chance"], 1)},
  p = {_n(k["p_value"], 3)}</td></tr>
<tr><td>Desacuerdos ≥ {ds["threshold"]} puestos</td><td class='n mono'>{ds["n_large"]}</td>
  <td>{ds["n_overrated_by_synthetic"]} sobrevaloradas y {ds["n_underrated_by_synthetic"]}
  infravaloradas por el sintético. Sobrevalorar es el error caro: asciende basura a la fase real</td></tr>
</tbody></table>
<p>El <b>bootstrap es por bloques</b>, no iid, y esa elección es la que hace honesto el intervalo: el
histórico real es un único camino, de modo que sus {b["n_blocks_real"]} sub-ventanas son su única fuente
de dispersión y los {val["n_folds"]} folds de cada una comparten calendario. Remuestrear scores sueltos
fingiría una independencia que no existe y devolvería un intervalo demasiado estrecho.</p>
<h4>El control que hay que hacer antes de creerse un ρ ≈ 0</h4>
<p>Un ρ nulo admite dos lecturas muy distintas: «el sintético no transfiere» o «ninguno de los dos lados
estaba rankeando estrategias». Hay una razón concreta para sospechar la segunda: el headline de una
configuración que <b>no opera</b> es <b>cero exacto</b> —curva plana, Sharpe 0, rotación 0, caída 0— y
en un periodo donde casi todo lo que arriesga pierde, ese cero gana; el CVaR, que puntúa por la cola
mala, lo premia doblemente. La correlación entre recompensa y actividad lo mide:
<b>{_n(act["reward_vs_activity_real"], 3)}</b> en el lado real —cuanto más opera una configuración,
peor puntúa— frente a {_n(act["reward_vs_activity_synthetic"], 3)} en el sintético, que no castiga
operar. Esa asimetría es en sí misma un hallazgo sobre el generador: produce oportunidad donde el
mercado de 2018-2025 no la dio.</p>
{activo}
<h4>Configuración a configuración</h4>
<table><thead><tr><th>Configuración</th><th>familia</th><th class=n>CVaR real</th>
<th class=n>CVaR sintético</th><th class=n>ops/ventana real</th><th class=n>puesto real</th>
<th class=n>puesto sintético</th><th class=n>Δ</th></tr></thead><tbody>{rows}</tbody></table>
<p class="tag">Δ = puesto real − puesto sintético. Positivo = el sintético la coloca mejor de lo que
merece. {t["leakage"]["folds_audited"]} folds auditados,
{"sin fuga temporal" if t["leakage"]["clean"] else "<b>CON FUGA</b>"}.</p>
<div class="note"><b>Lo que esta cifra no es.</b><ul>{caveats}</ul></div>
<p class="tag">Evidencia completa: <span class="mono">data/transfer/report_{t["library"]}.json</span> ·
reproducible con <span class="mono">python -m ai_trader.scoring.transfer_study --offline</span>
({t["n_sub_windows"]} sub-ventanas reales + {t["n_samples"]} muestras de
<span class="mono">{t["library"]}</span>{" <b>(librería de reserva: la pedida no existía)</b>"
if t["is_fallback"] else ""}; {t["generated_at"]}).</p>"""


def _market_block(m):
    """Seccion 2.1: la captura de datos REALES.

    Es la seccion que faltaba. La documentacion empezaba por el generador sintetico, y
    eso dejaba sin describir el sustrato del que sale TODA la evidencia externa del
    proyecto -fidelidad, transferencia, sesiones- y el unico que se usa en vivo."""
    if not m:
        return "<h3>2.1 · Captura de datos reales</h3><p>Sin datos de configuración.</p>"
    return f"""
<h3>2.1 · Captura de datos reales</h3>
<p>El sistema opera sobre <b>velas diarias</b> (OHLCV) de {m["n_symbols"]} pares de criptomoneda contra
<i>stablecoin</i>, leídas de <span class="mono">{m["exchange"]}</span> a través de un adaptador CCXT que
pagina en lotes de {m["batch"]} barras hasta cubrir el rango pedido. Cada ciclo de decisión pide
{m["lookback_days"]} días de historia por símbolo: lo justo para que las medias largas, el canal de
Donchian y las ventanas de volatilidad estén calientes el primer día.</p>
<table><thead><tr><th>Clase de activo</th><th>Proveedor</th><th>Qué se usa hoy</th></tr></thead><tbody>
<tr><td>Criptomonedas</td><td class=mono>ccxt · {m["exchange"]}</td><td><b>El universo operado.</b> Cotiza
24/7, así que hay una barra por día natural.</td></tr>
<tr><td>Renta variable</td><td class=mono>alpaca</td><td>Proveedor implementado y <b>sin estrategia
detrás</b>: la clase de activo está aparcada a propósito (§6).</td></tr>
<tr><td>Mercados de predicción</td><td class=mono>polymarket (gamma / CLOB)</td><td>Precio vivo y libro,
pero <b>no hay OHLCV histórico</b>: no se puede backtestear, sólo capturar hacia adelante.</td></tr>
</tbody></table>
<div class="why"><b>Tres decisiones que hacen que el dato real sea auditable.</b>
<ul>
<li><b>Caché en disco, no memoria.</b> Las barras se guardan en
<span class="mono">{m["cache_dir"]}/&lt;símbolo&gt;_&lt;timeframe&gt;.parquet</span> y se reutilizan. El
estudio de fidelidad y el de transferencia se pueden repetir <b>sin red</b>, que es lo que hace que dos
regeneraciones den lo mismo aunque el exchange esté caído o haya cambiado el pasado.</li>
<li><b>Ventanas históricas cerradas, no «hasta hoy».</b> Cada estudio declara su rango como constante. Si
el final se moviera con la fecha de ejecución, dos ejecuciones del mismo estudio no serían comparables y
la palabra «reproducible» no significaría nada.</li>
<li><b>El corte anti-<i>look-ahead</i> vive en la fuente de datos</b>, no en la estrategia (§3.2). Da
igual quién pida las barras —una estrategia, el bloque de régimen, el radar de señales—: nadie puede ver
la barra de hoy, porque la capa que las sirve no la devuelve.</li>
</ul></div>
<div class="note"><b>El universo operado y el sintético no son el mismo, y la asimetría es deliberada.</b>
Aquí sólo hay cripto: es donde está toda la evidencia empírica del repositorio. El universo del generador
(§2.5) tiene {m["n_synthetic_assets"]} activos incluyendo renta variable y refugios, porque los factores
compartidos son lo que hace que un escenario de tipos signifique algo para cripto. La regla que mantiene
los dos ficheros consistentes: <b>éste tiene que ser operable hoy</b> —un símbolo deslistado falla en
silencio en cada ciclo, que fue el caso de MATIC/USDT— y el sintético tiene que coincidir con el universo
declarado en código.</div>"""


def _risk_gate_block(t):
    """Seccion 3.3: el motor de riesgo, con los limites REALES del config operado.

    Publicar los numeros y no solo el mecanismo es lo que convierte 'hay control de
    riesgo' en algo que se puede auditar."""
    if not t:
        return ""
    return f"""
<h3>3.3 · El motor de riesgo: la puerta única, con sus números</h3>
<p>Una señal no es una orden. Toda señal —venga de la estrategia que venga— pasa por el motor de riesgo,
que decide si se ejecuta, con cuánto tamaño y con qué salidas. No hay ninguna vía que lo esquive: es la
única función que convierte señal en orden.</p>
<table><thead><tr><th>Límite</th><th class=n>valor</th><th>qué impide</th></tr></thead><tbody>
<tr><td>Confianza mínima por operación</td><td class='n mono'>{_n(t["min_confidence"], 2)}</td>
<td>Que una señal débil abra posición sólo porque no había nada mejor.</td></tr>
<tr><td>Tamaño máximo por posición</td><td class='n mono'>{_n(t["max_position_size_usd"], 0)} $</td>
<td>Que un trade concentre la cuenta. Con equity conocido manda además la fracción de riesgo
({_pc(t["risk_fraction_per_trade"], 0)} del capital), de modo que el tamaño <b>compone</b>.</td></tr>
<tr><td>Exposición máxima por símbolo / total</td>
<td class='n mono'>{_n(t["max_symbol_exposure_usd"], 0)} $ / {_n(t["max_total_exposure_usd"], 0)} $</td>
<td>Que varias señales del mismo activo, o del mismo día, se acumulen en una apuesta única.</td></tr>
<tr><td>Posiciones abiertas simultáneas</td><td class='n mono'>{t["max_open_positions"]}</td>
<td>Que la cartera se convierta en un índice por goteo.</td></tr>
<tr><td>Pérdida diaria máxima</td><td class='n mono'>{_n(t["max_daily_loss_usd"], 0)} $</td>
<td>Que un mal día siga abriendo posiciones nuevas.</td></tr>
<tr><td>Stop / objetivo por defecto</td>
<td class='n mono'>−{_n(t["default_stop_loss_pct"], 0)} % / +{_n(t["default_take_profit_pct"], 0)} %</td>
<td>Que una posición quede sin salida definida. La estrategia puede proponer los suyos, pero
<b>nunca más lejos</b> de {_n(t["max_stop_distance_pct"], 0)} %.</td></tr>
<tr><td>Vida máxima de una posición</td><td class='n mono'>{t["max_holding_days"]} días</td>
<td>Que una posición sin desenlace se quede indefinidamente ocupando exposición. Es también el número que
fija la <b>purga</b> de la validación temporal (§4.8): no es una coincidencia, es la misma magnitud.</td></tr>
<tr><td>Enfriamiento por símbolo / operaciones por ciclo</td>
<td class='n mono'>{t["cooldown_hours"]} h / {t["max_trades_per_cycle"]}</td>
<td>Que el sistema reabra lo que acaba de cerrar, y que un ciclo raro dispare una ráfaga.</td></tr>
</tbody></table>
<div class="why"><b>Por qué el riesgo es dueño del stop y no la estrategia.</b> Una estrategia optimiza su
propia señal; el stop es una decisión de <b>cartera</b>, no de señal. Si cada estrategia pudiera fijar el
suyo, el optimizador aprendería a ensancharlo hasta que dejara de doler — que es exactamente cómo se
fabrica una curva de equity bonita con una cola desastrosa. Por eso el riesgo puede <b>recortar</b> lo que
propone la estrategia, nunca al revés.</div>"""


def _pnl_block(t):
    """Seccion 3.9: como se registra el resultado de un trade.

    Es la mitad del ciclo que nunca estuvo escrita: el documento explicaba como se
    decide y como se llena, y daba por hecha la contabilidad."""
    if not t:
        return ""
    return f"""
<h3>3.9 · Cómo se registra el resultado: PnL, equity y rotación</h3>
<p>Una posición se cierra por una de tres causas —<span class="mono">stop_loss</span>,
<span class="mono">take_profit</span> o <span class="mono">max_holding_days</span>— y en las tres el
cierre <b>pasa por el motor de ejecución</b>, igual que la entrada: se le cobran su deslizamiento y su
comisión. Cerrar «a precio de mercado» sin pagar la salida es el atajo clásico que convierte una
estrategia mediocre en una rentable sobre el papel.</p>
<div class="formula">PnL_neto = (precio_salida − precio_entrada) × tamaño × dirección − comisiones(entrada + salida)</div>
<p>La comisión es <span class="mono">fee_rate = {_pc(t["fee_rate"], 1)}</span> del <i>notional</i>
llenado, en <b>cada</b> pata, y sigue al tamaño <b>realmente ejecutado</b>, no al pedido: si la orden se
llenó a medias por el techo de capacidad (§3.7), se cobra la mitad. El PnL registrado en cada posición
cerrada es siempre el neto.</p>
<p>Con eso se construyen las tres magnitudes que el resto del documento usa:</p>
<table><thead><tr><th>Magnitud</th><th>Cómo se calcula</th><th>Dónde se usa</th></tr></thead><tbody>
<tr><td><b>Equity</b></td><td>Capital inicial ({_n(t["starting_equity"], 0)} $ en backtest) + PnL neto
realizado acumulado + PnL no realizado de lo abierto, valorado al cierre del día.</td>
<td>Es la serie de la que salen Sharpe, drawdown y CAGR (§4.4).</td></tr>
<tr><td><b>PnL diario realizado</b></td><td>Suma de los cierres del día. Se reinicia cada día.</td>
<td>Alimenta el límite de pérdida diaria del motor de riesgo (§3.3).</td></tr>
<tr><td><b>Rotación (<i>turnover</i>)</b></td><td><i>Notional</i> movido por día en unidades del capital
inicial, contando <b>las dos patas</b> de cada operación.</td>
<td>Es el término que penaliza la métrica de cabecera (§4.4), y lo que permite comprobar que las
comisiones cobradas cuadran con el <span class="mono">fee_rate</span> configurado (§4.5).</td></tr>
</tbody></table>
<div class="why"><b>Por qué el equity se marca a mercado todos los días y no sólo al cerrar.</b> Si sólo
contara el PnL realizado, la curva sería una escalera plana entre cierres y el drawdown mediría cuándo se
decidió cerrar, no cuánto se sufrió. Marcar a diario es lo que hace que el máximo drawdown y la
volatilidad describan la experiencia real de tener la posición abierta.</div>
<div class="note"><b>La foto y la película se guardan por separado, y a propósito.</b> El estado persistido
(<span class="mono">data/runtime_state.json</span>) tiene la <i>foto</i> —posiciones abiertas y cerradas,
PnL realizado— y es todo lo que hace falta para seguir operando tras un reinicio. La <i>película</i> —qué
se decidió cada ciclo, con qué precio de referencia, qué rechazó el riesgo y cuánto deslizamiento se cobró
de verdad— va a un diario append-only aparte (<span class="mono">data/live/cycles.jsonl</span>, §5.1),
porque son dos cosas con vidas distintas: la foto se sobrescribe cada ciclo y la película no se sobrescribe
nunca. Ese diario es el material con el que se medirá la divergencia entre lo ejecutado en vivo y lo que
predice el <i>backtest</i> (§5, §6).</div>"""


def _factor_table(factors):
    body = "".join(f"<tr><td class=mono>{f}</td><td>{d}</td></tr>" for f, d in factors)
    return ("<table><thead><tr><th>Factor</th><th>Interpretación</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def render_html(f: dict) -> str:
    def g(lib, k, default="—"):
        v = f.get(lib)
        return v[k] if v else default

    def sf(key, k, default="—"):
        v = f.get(key)
        return v[k] if v else default

    rep = {
        "COMMIT": f.get("commit", ""),
        "COMMITN": f.get("commit_count", ""),
        "DATE": f.get("date", ""),
        "NTESTS": f.get("n_tests", "—"),
        "NASSETS": f.get("n_assets", 35),
        "NASSETS_MINUS_ONE": int(f.get("n_assets", 35)) - 1,
        "NCRYPTO": f.get("n_crypto", 12),
        "NCRYPTOLIVE": (f.get("market") or {}).get("n_symbols", "—"),
        "NEQUITY": f.get("n_equity", 20),
        "NMACRO": f.get("n_macro", 3),
        "NOWN": f.get("n_own", "—"),
        "NREGIME": f.get("n_regime", "—"),
        "V2SCEN": g("ai_v2", "scenarios"),
        "V2PATHS": g("ai_v2", "paths"),
        "V2SAMPLES": g("ai_v2", "samples"),
        "V2HORIZON": g("ai_v2", "horizon"),
        "SFV1SPREAD": sf("sf_v1", "spread"),
        "SFV2SPREAD": sf("sf_v2", "spread"),
        "SFV2REV": sf("sf_v2", "revert"),
        "SFV2TREND": sf("sf_v2", "trend"),
        "SFV2TOTAL": sf("sf_v2", "total"),
        "SFV1CLUS": sf("sf_v1", "clustering"),
        "SFV2CLUS": sf("sf_v2", "clustering"),
        "SFV1EXC": sf("sf_v1", "exceed"),
        "SFV2EXC": sf("sf_v2", "exceed"),
        "FACTOR_TABLE": _factor_table(f.get("factors", [])),
        "MOM_PARAMS": _rows(f.get("mom_params", []), ["Parámetro", "Valor"]),
        "MR_PARAMS": _rows(f.get("mr_params", []), ["Parámetro", "Valor"]),
        "SPACE_MOM": _rows(f.get("space_mom", []), ["Parámetro", "Rango CEM"]),
        "SPACE_MR": _rows(f.get("space_mr", []), ["Parámetro", "Rango CEM"]),
        "THEMED_FAMILIES": _themed_families_block(f.get("themed", [])),
        "THEMED_SPACES": _themed_spaces_block(f.get("themed", [])),
        "MARKET": _market_block(f.get("market")),
        "RISKGATE": _risk_gate_block(f.get("trade")),
        "PNL": _pnl_block(f.get("trade")),
        "CALIBRATION": _calibration_block(f.get("calibration")),
        "FIDELITY": _fidelity_block(f.get("fidelity")),
        "TRANSFER": _transfer_block(f.get("transfer")),
        "TRANSFER_EXTENDED": _transfer_extended_block(f.get("transfer_extended")),
        "THEMES_REAL": _themes_real_block(f.get("themes_real")),
        "EXTENDED_GRID": _extended_grid_block(f.get("extended_grid")),
        "SIGNALCHANNEL": _signal_channel_block(f.get("signal_channel")),
        "VALIDATION": _validation_block(f.get("validation")),
        "SESSIONS": _sessions_block(f.get("sessions")),
        "DIVERGENCE": _divergence_block(f.get("divergence")),
        "ACTIVITY": _activity_block(f.get("activity")),
        "SIGNALS": _signals_block(f.get("signals")),
        "LAMBDA": _n((f.get("calibration") or {}).get("lambda", 0.5), 2),
        "KAPPA": _n((f.get("calibration") or {}).get("kappa", 1.0), 1),
    }
    html = BODY
    for k, v in rep.items():
        html = html.replace("%%" + k + "%%", str(v))
    return (
        "<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>AI-Trader · Documentación funcional</title><style>" + CSS + "</style></head><body>"
        "<button class=\"printbtn noprint\" onclick=\"window.print()\">Descargar PDF (imprimir)</button>"
        "<div class=\"page\">" + html + "</div></body></html>"
    )


# ------------------------------------------------------------------ prose ----------
BODY = r"""
<h1>AI-Trader — Documentación funcional</h1>
<div class="sub">Cómo funciona la herramienta, con el detalle necesario para replicarla y auditarla.
Cada decisión de diseño va justificada.</div>
<div class="meta">Documento vivo · generado desde el commit <span class="mono">%%COMMIT%%</span>
(%%COMMITN%% commits)%%DATE%% · se regenera con <span class="mono">python -m docs.build_docs</span>.
Las cifras marcadas son extraídas del repositorio en el momento de generar este documento.</div>

<div class="toc noprint">
<b>Contenido</b>
<ol>
<li><a href="#s1">Resumen ejecutivo</a></li>
<li><a href="#s2">Datos</a> — captura real, señales externas, generación sintética y sus tests</li>
<li><a href="#s3">Trade</a> — cómo se ejecuta una operación, qué cuesta y cómo se contabiliza</li>
<li><a href="#s4">Estrategias</a> — recompensa, ordenación y validación</li>
<li><a href="#s5">Resultados</a></li>
<li><a href="#s6">Limitaciones y evoluciones</a></li>
</ol>
<b>Anexos</b>
<ol type="A">
<li><a href="#a1">Reproducibilidad</a></li>
<li><a href="#a2">Glosario</a></li>
</ol>
</div>

<h2 id="s1">1 · Resumen ejecutivo</h2>
<p class="lead">AI-Trader es una herramienta de inversión cuantitativa que hoy opera en <b>paper
trading</b> sobre criptomonedas. Junta cinco piezas que normalmente viven separadas: la captura de datos
de mercado, una plataforma de señales externas, un generador de mercados sintéticos deterministas, un
motor de ejecución con costes y capacidad realistas, y un aparato de evaluación que decide qué estrategia
merece operar. Este capítulo recorre la cadena entera; los cinco siguientes la detallan.</p>

<h3>1.1 · El recorrido, de punta a punta</h3>
<p>Una sola operación atraviesa trece etapas, y cada una tiene su propia sección en este documento. El
orden importa: nada de lo que viene después puede arreglar un error cometido antes.</p>
<table><thead><tr><th class="n">#</th><th>Etapa</th><th>Qué ocurre</th><th>Detalle</th></tr></thead><tbody>
<tr><td class="n">1</td><td><b>Captura de precio</b></td><td>Velas diarias de %%NCRYPTOLIVE%% pares
cripto, cacheadas en disco para que cualquier estudio se pueda repetir sin red.</td><td>§2.1</td></tr>
<tr><td class="n">2</td><td><b>Señales externas</b></td><td>Diecisiete fuentes fuera del precio —flujos de
ETF, macro, cadena, atención, oferta desbloqueada— normalizadas y con su profundidad histórica
<i>medida</i>, no declarada.</td><td>§2.2</td></tr>
<tr><td class="n">3</td><td><b>Mundos sintéticos</b></td><td>Una IA diseña la física de cada escenario
macro y un motor determinista la convierte en velas con colas gruesas, agrupamiento de volatilidad y
estructura serial.</td><td>§2.3–2.8</td></tr>
<tr><td class="n">4</td><td><b>Observación</b></td><td>Lo que la política ve en el momento de decidir:
mercado propio, contexto cross-sectional y el radar de señales. Sólo datos hasta el cierre de
ayer.</td><td>§4.2</td></tr>
<tr><td class="n">5</td><td><b>Señal</b></td><td>Ocho primitivas paramétricas proponen entrada,
confianza y salidas: dos de régimen opuesto que solo miran precio —momentum y reversión— y seis
temáticas con núcleo de precio y capa de señal.</td><td>§4.1, §4.1-bis</td></tr>
<tr><td class="n">6</td><td><b>Riesgo</b></td><td>Puerta única: tamaño, exposición, confianza mínima,
pérdida diaria y propiedad del stop. Ninguna señal la esquiva.</td><td>§3.3</td></tr>
<tr><td class="n">7</td><td><b>Ejecución</b></td><td>Se llena al <i>open</i> del día siguiente, pagando
medio spread, volatilidad e impacto por tamaño, con techo de capacidad y fills parciales.</td>
<td>§3.4–3.8</td></tr>
<tr><td class="n">8</td><td><b>Contabilidad</b></td><td>PnL neto de comisiones en las dos patas, equity
marcado a mercado a diario y rotación medida.</td><td>§3.9</td></tr>
<tr><td class="n">9</td><td><b>Puntuación</b></td><td>Cada muestra da un <i>headline</i> fuera de muestra
(Sharpe penalizado) y la distribución se agrega por su cola: CVaR@25%.</td><td>§4.3–4.6</td></tr>
<tr><td class="n">10</td><td><b>Validación</b></td><td>Arquetipos macro enteros reservados como hold-out,
y varias ventanas temporales por muestra con purga y embargo.</td><td>§4.7–4.8</td></tr>
<tr><td class="n">11</td><td><b>Veredicto</b></td><td>Aprueba quien bate al mejor rival pasivo <i>y</i>
supera el suelo de actividad. Y se comprueba si el orden sintético se parece al real.</td>
<td>§4.9–4.11</td></tr>
<tr><td class="n">12</td><td><b>Búsqueda</b></td><td>Optimización de caja negra (CEM) sobre los
parámetros, con el resultado descontado por el número de intentos (DSR/PBO).</td><td>§4.13–4.14</td></tr>
<tr><td class="n">13</td><td><b>En vivo</b></td><td>El mismo camino, con reloj real y dinero de papel.
Es la etapa que aún no ha producido resultados.</td><td>§5</td></tr>
</tbody></table>
<div class="why"><b>La propiedad que sostiene todo lo anterior:</b> las etapas 4 a 8 son <b>el mismo
código</b> en <i>backtest</i> y en vivo. El motor de <i>backtest</i> no reimplementa la lógica de trading:
inyecta un reloj simulado y una fuente de datos con anti-<i>look-ahead</i>, y hace avanzar el mismo
orquestador que operaría con dinero (§3.1). Lo que se optimiza es, literalmente, lo que se ejecutaría.</div>

<h3>1.2 · Qué es y qué no es (todavía)</h3>
<ul>
<li><b>Es:</b> un banco de pruebas honesto para diseñar, evaluar y optimizar estrategias contra un
universo de %%NASSETS%% activos y una batería de regímenes macro sintéticos, ejecutando exactamente el
mismo camino que se usaría en vivo.</li>
<li><b>No es (aún):</b> no mueve dinero real; no hay estrategia de renta variable ni de mercados de
predicción; la «inteligencia» de optimización está en fase de <b>caja negra</b> (CEM sobre parámetros), no
de aprendizaje por refuerzo por gradiente de política; y no hay todavía un historial en vivo con el que
medir la divergencia frente al <i>backtest</i>.</li>
</ul>

<h3>1.3 · Los cuatro resultados que condicionan lo demás</h3>
<p>Este proyecto publica mediciones, no impresiones, y cuatro de ellas explican por qué el sistema está
construido como está. Dos salieron como se esperaba y dos no; las cuatro se reportan igual.</p>
<table><thead><tr><th>Pregunta</th><th>Respuesta medida</th><th>Consecuencia</th></tr></thead><tbody>
<tr><td>¿Se parece el mundo sintético al mercado?</td><td class="ok">Sí, y con umbrales que el estudio
puede fallar: cobertura del 98 %, colas y agrupamiento en la magnitud real.</td><td>El generador vale
como banco de estrés y de regresión (§2.10).</td></tr>
<tr><td>¿<b>Ordena</b> las estrategias como el mercado?</td><td class="pend"><b>No.</b> ρ ≈ 0 sobre las 16
configuraciones, y negativo entre las que operan de verdad.</td><td>El sintético deja de ser criterio de
selección: el ranking que decide sale del histórico real (§4.11).</td></tr>
<tr><td>¿Sobre-estimaba el corte único 70/30?</td><td class="pend">No sobre-estima: es <b>arbitrario</b>.
Mover la ventana mueve el resultado varias veces más que cambiar de estrategia.</td><td>Elegir con una
sola ventana es elegir por el tramo de historia que tocó (§4.8).</td></tr>
<tr><td>¿Qué estaba ganando el ranking real?</td><td class="pend">La <b>inactividad</b>:
ρ(recompensa, operaciones) = −0,84. Una curva plana puntúa cero exacto y en un periodo bajista ese cero
gana.</td><td>Rankear exige operar: hay un suelo de actividad medido (§4.10).</td></tr>
</tbody></table>

<h3>1.4 · Principios rectores</h3>
<p>Todo el diseño se somete a cuatro principios. No son eslóganes: cada uno se traduce en decisiones
concretas de código verificadas por tests.</p>
<div class="why"><b>Honestidad estadística.</b> La unidad de evaluación es la <b>distribución</b>
sobre muchas muestras, nunca un único camino. <i>Por qué:</i> un solo path es ruido; publicar solo
la media esconde la cola. Se reporta la dispersión y se optimiza un estadístico consciente de la cola
(§4.3–4.6).</div>
<div class="why"><b>Anti-look-ahead por diseño.</b> En cada decisión solo se usan datos hasta el
cierre del día anterior. <i>Por qué:</i> el sesgo de futuro es la causa número uno de backtests
engañosos; aquí es imposible por construcción, no por disciplina (§3.2).</div>
<div class="why"><b>Determinismo total.</b> La terna (especificación, universo, semilla) produce las
mismas velas byte a byte; toda la cadena va sembrada. <i>Por qué:</i> reproducibilidad y
auditabilidad — cualquier resultado se puede regenerar y verificar (anexo A).</div>
<div class="why"><b>No romper el contrato en vivo.</b> Lo que se optimiza se ejecuta por el mismo
camino estrategia → riesgo → ejecución. <i>Por qué:</i> elimina la divergencia backtest/live, que es
la fuente clásica de sorpresas en producción (§3.1).</div>

<h2 id="s2">2 · Datos</h2>
<p class="lead">La calidad de todo lo demás depende de la calidad del sustrato de datos, así que la
auditoría empieza aquí y no por el código. Hay tres sustratos y conviene no confundirlos: el
<b>mercado real</b> (§2.1), las <b>señales externas</b> (§2.2) y los <b>mundos sintéticos</b> (§2.3–2.8).
Los dos últimos existen porque el primero es escaso: un único camino de la historia, y sólo el que
ocurrió.</p>

%%MARKET%%

%%SIGNALS%%

<h3>2.3 · Por qué además datos sintéticos</h3>
<p>El desarrollo y buena parte de la evaluación se hacen sobre mercados <b>sintéticos</b>. Las razones, y
sus contrapartidas, son explícitas:</p>
<ul>
<li><b>Independencia temporal por diseño.</b> Al no solaparse con ningún periodo histórico real, se
elimina el <i>data-snooping</i> sobre eventos ya conocidos: no se puede, ni sin querer, ajustar a la
crisis de 2008 o al COVID.</li>
<li><b>Control del régimen.</b> Podemos generar a voluntad crisis, rangos laterales, tendencias,
shocks de tipos o inviernos cripto, y en la proporción que necesitemos para estresar una estrategia.</li>
<li><b>Volumen ilimitado y hold-out honesto.</b> Generamos tantas muestras como haga falta para una
estadística robusta, y podemos reservar <b>arquetipos macro enteros</b> como validación (§4.7).</li>
</ul>
<div class="note"><b>Las dos contrapartidas, y la segunda resultó mayor que la primera.</b> La conocida:
un mercado sintético sólo es útil si se parece al real en sus propiedades estadísticas, y por eso existen
la validación interna de §2.9 y sobre todo la externa contra el histórico real de §2.10. La que no se
había previsto: parecerse no basta. Lo que el sistema le pide al generador es que <b>ordene</b> las
estrategias como el mercado, se midió aparte (§4.11) y la respuesta fue que <b>no</b>. Las tres razones de
arriba siguen valiendo para lo que el sintético hace bien —cubrir regímenes que la historia no dio,
estresar, dar hold-out de arquetipos y regresión determinista— pero ya no para decidir qué configuración
es mejor.</div>

<h3>2.4 · El modelo de factores</h3>
<p>El retorno diario de cada activo se construye con un modelo de factores:</p>
<div class="formula">r_i(t) = tilt_i + Σ_k β_ik · f_k(t) + idio_i · ε_i(t)</div>
<p>Las correlaciones entre activos <b>emergen</b> de que comparten exposición (β) a un puñado de
factores macro comunes f_k; no se imponen con una matriz.</p>
<div class="why"><b>Por qué un modelo de factores y no una matriz de correlación N×N.</b> La
covarianza resultante es <b>siempre definida positiva por construcción</b> — no existe matriz que
pueda degenerar, el problema clásico de las correlaciones impuestas. Además, el diseñador (la IA) solo
tiene que razonar sobre "qué hacen cinco factores" en cada escenario, no sobre cientos de
correlaciones sueltas.</div>
<p>Los cinco factores y su interpretación:</p>
%%FACTOR_TABLE%%
<p>Las <b>betas son fijas</b> por activo (estructura estable que no cambia entre escenarios); los
<b>tilts</b> son deriva extra por símbolo que la IA añade por escenario para respuestas
idiosincráticas que los factores comunes no capturan (p.ej. un embargo de petróleo que levanta a una
petrolera por encima del factor de materias primas).</p>

<h3>2.5 · El universo sintético</h3>
<p>%%NASSETS%% activos: %%NCRYPTO%% criptomonedas, %%NEQUITY%% de renta variable (índices y nombres
por sector) y %%NMACRO%% de macro/refugio (oro, bonos largos, dólar). La mezcla es deliberada para que
las correlaciones cruzadas tengan sentido: las financieras cargan tipos en positivo, la energía carga
materias primas, y los refugios cargan la renta variable en negativo. Cada activo declara además su
<b>liquidez</b> (volumen típico negociado en dólares al día), que abarca más de dos órdenes de magnitud
entre un índice amplio y un altcoin: es el eje que hace que ejecutar cueste distinto en cada mercado
(§3.6). El universo se guarda de forma autocontenida en el manifiesto de cada librería (precio inicial,
betas, volatilidad idiosincrática y liquidez), de modo que se pueda reconstruir el universo exacto y
regenerar datos idénticos aunque el código cambie.</p>
<div class="note"><b>Por qué el universo sintético y el operado no son el mismo (el caso MATIC/USDT).</b>
Binance deslistó ese par (el token migró a POL), así que está <b>retirado</b> del universo que se opera
en vivo (§2.1): allí un símbolo muerto falla en silencio en cada ciclo. En el universo sintético se
<b>mantiene a propósito</b>, y esa asimetría es la decisión, no un descuido: aquí no cotiza nada contra un
exchange — el símbolo es solo la etiqueta de un perfil de cargas factoriales y las velas las genera el
motor —, tiene contraparte real en la ventana con la que se mide la fidelidad (2017-2026, y el informe
publicado lo lista sin símbolos ausentes), y retirarlo cambiaría el universo de %%NASSETS%% a
%%NASSETS_MINUS_ONE%% activos, desincronizando toda la evidencia ya publicada sobre %%NASSETS%%: la
calibración de pesos y el propio estudio de fidelidad habría que re-medirlos. <b>Límite declarado:</b> si
el estudio de fidelidad se vuelve a correr sobre una ventana reciente, MATIC/USDT no tendrá datos reales y
aparecerá como ausente; ese es el momento de renombrarlo a POL/USDT y republicar las mediciones, no
antes.</div>

<h3>2.6 · El diseñador de escenarios (la única pieza con IA)</h3>
<p>Un modelo de lenguaje (Claude) diseña la "física" de cada escenario macro y la emite como
<b>JSON estricto</b>: fases temporales (cada una con deriva y volatilidad por factor), shocks
discretos y tilts por activo. Ese objeto — el <i>ScenarioSpec</i> — es 100% serializable y auditable.</p>
<div class="why"><b>Por qué separar la especificación (IA) de la calibración (código).</b> La IA aporta
<b>diversidad y narrativa macro plausible</b> (política monetaria, guerras, crisis de liquidez,
euforias); el código aporta <b>determinismo, validación y reproducibilidad</b>. La frontera es un JSON
que se puede leer y auditar sin ejecutar la IA.</div>
<p>El <i>spec.json</i> de cada escenario se guarda en disco: <b>es lo único insustituible</b>. Los
caminos Monte Carlo se regeneran a partir de él sin volver a llamar a la IA (operación
"resynthesize"), lo que permite ampliar el número de caminos o reconstruir los datos borrados de forma
determinista.</p>
<div class="note"><b>Reproducibilidad de la IA: no la hay, y no puede haberla.</b> Dos llamadas con el
mismo prompt y el mismo modelo devuelven escenarios distintos. Antes esto se atribuía a
<span class="mono">temperature=1.0</span>; la realidad es más fuerte: los modelos actuales
<b>retiraron los parámetros de muestreo</b> — enviar <span class="mono">temperature</span>,
<span class="mono">top_p</span> o <span class="mono">top_k</span> devuelve un error 400 —, así que ya no
existe ninguna palanca con la que forzar determinismo. Rehacer una librería con IA produce <i>siempre</i>
una librería nueva. <b>Mitigación: no se re-deriva, se guarda.</b> El <i>spec.json</i> es la salida cara e
insustituible y se persiste en disco; todo lo que va detrás (caminos, velas, backtests, métricas) es
determinista dado el spec y la semilla. La reproducibilidad del proyecto descansa en el artefacto
guardado, no en la llamada a la IA. <b>Límite declarado:</b> el manifiesto registra la <i>clase</i> del
diseñador, no el identificador del modelo; dos librerías generadas con modelos distintos son
indistinguibles por el manifiesto.</div>

<h3>2.7 · Del spec a las velas (el motor numérico)</h3>
<p>El motor convierte un spec en velas OHLCV multi-activo de forma <b>determinista</b> (vía
<span class="mono">numpy.default_rng(seed)</span>). Las fases se expanden a series diarias de deriva y
volatilidad; los shocks se suman como deriva puntual en su día. Las velas son válidas por construcción
(máximo ≥ cuerpo ≥ mínimo, precios positivos). La apertura de cada día parte del cierre anterior más un
hueco nocturno; las mechas se escalan por la volatilidad de ese día.</p>

<h3>2.8 · La microestructura estadística: por qué un generador ingenuo miente</h3>
<p>Este es el corazón de la auditoría de datos. Un generador ingenuo — ruido gaussiano independiente,
volatilidad constante — <b>miente sistemáticamente en la dirección de hacer las estrategias parecer
mejores y más seguras de lo que son</b>. Y un optimizador es exactamente la herramienta que
encontrará y explotará ese sesgo. Por eso cada mecanismo de realismo se añadió con una justificación
concreta:</p>
<ul>
<li><b>Volatilidad por fase, no promediada.</b> El rango intradía debe <b>ensancharse en las fases de
pánico</b>. Si se promedia al horizonte, el ATR (rango medio verdadero) no reacciona al régimen, y con
él mienten el filtro de volatilidad y el dimensionado del stop de la estrategia. <i>(Era un bug; se
corrigió.)</i></li>
<li><b>Colas gruesas (t-Student, con varianza ajustada).</b> Los mercados reales tienen colas más
gruesas que la normal. Sin ellas, la pérdida de cola queda sistemáticamente subestimada — justo en los
escenarios de crisis que el generador existe para producir.</li>
<li><b>Agrupamiento de volatilidad (tipo GARCH).</b> La volatilidad se agrupa: a un día agitado le
sigue otro agitado. Sin agrupamiento, la estructura temporal del riesgo es irreal.</li>
<li><b>Estructura serial (autocorrelación AR(1) idiosincrática, con signo por régimen).</b> Sin
autocorrelación, los retornos son independientes y <b>la reversión a la media es rentable
imposible por construcción</b>, no por falta de <i>edge</i>. Con el signo dependiente de la fase, unos
regímenes <b>tienden</b> (favorecen al momentum) y otros <b>revierten</b> (favorecen a la reversión),
de modo que el evaluador tiene algo real que distinguir. Es el mecanismo más importante: sin él, un
optimizador solo aprendería a clasificar fases.</li>
<li><b>Saltos en el hueco de apertura.</b> Con huecos gaussianos, un stop se ejecuta <b>siempre</b>
cerca de su nivel. Con saltos, un hueco puede <b>saltarse el stop</b> — la pérdida de cola de las
crisis deja de estar subestimada.</li>
<li><b>Dispersión del día del shock entre caminos.</b> Para que un crash no caiga siempre el mismo día
en todos los caminos del ensemble (diversidad realista).</li>
</ul>
<div class="why"><b>Invariantes de estas extensiones.</b> Todas están <b>ajustadas en varianza</b>
(añadir cola o estructura serial no cambia la volatilidad total, solo su forma), preservan la covarianza
definida positiva (sigue emergiendo de factores compartidos) y tienen <b>valores neutros por
defecto</b>: con la microestructura desactivada, el mundo vuelve a ser el gaussiano independiente
original, byte a byte. Eso permitió introducirlas sin invalidar la librería previa.</div>

<h3>2.9 · Primer test de viabilidad: del mundo que miente (ai_v1) al mundo realista (ai_v2)</h3>
<p>La microestructura se asigna a los escenarios existentes mediante un <b>retrofit determinista</b>:
una función que deriva el carácter de cada fase de su propia semántica (una fase direccional → tendencia;
una fase plana → reversión; una fase de crisis, medida por la volatilidad de la renta variable → colas,
saltos y agrupamiento). Se reusa así el diseño caro de la IA sin volver a llamarla, y se produce una
nueva librería (<span class="mono">ai_v2</span>) conservando la anterior para comparar.</p>
<p>La librería de referencia del <i>scoring</i>, <span class="mono">ai_v2</span>, contiene
<b>%%V2SCEN%% escenarios × %%V2PATHS%% caminos = %%V2SAMPLES%% muestras</b>, con horizonte de
%%V2HORIZON%% días. La tabla compara sus <i>stylized facts</i> con los de la librería anterior
(independiente e ingenua):</p>
<table><thead><tr><th>Propiedad estadística</th><th class="n">ai_v1 (iid)</th><th class="n">ai_v2</th><th>Lectura</th></tr></thead>
<tbody>
<tr><td>Dispersión de la autocorrelación entre escenarios</td><td class="n mono">%%SFV1SPREAD%%</td><td class="n mono">%%SFV2SPREAD%%</td><td>Diversidad de régimen serial</td></tr>
<tr><td>Escenarios que revierten / tienden (de %%SFV2TOTAL%%)</td><td class="n mono">0 / 0</td><td class="n mono">%%SFV2REV%% / %%SFV2TREND%%</td><td>La reversión deja de ser imposible</td></tr>
<tr><td>Agrupamiento de volatilidad (autocorr. de |retorno|)</td><td class="n mono">%%SFV1CLUS%%</td><td class="n mono">%%SFV2CLUS%%</td><td>La volatilidad se agrupa</td></tr>
<tr><td>Exceso de días más allá de 3σ (%)</td><td class="n mono">%%SFV1EXC%%</td><td class="n mono">%%SFV2EXC%%</td><td>Colas más gruesas</td></tr>
</tbody></table>
<p>Interpretación: el mundo dejó de mentir en la dirección optimista. En <span class="mono">ai_v1</span>
todos los escenarios eran indistinguibles del ruido; en <span class="mono">ai_v2</span> hay regímenes
que tienden y regímenes que revierten, la volatilidad se agrupa y las colas engordan.</p>
<div class="why"><b>Sustrato por defecto de la evaluación.</b> El harness de puntuación y optimización
(§4) corre sobre <span class="mono">ai_v2</span> por defecto — es una constante única
(<span class="mono">DEFAULT_LIBRARY_ID</span>) y hay un test que la fija. La librería anterior
(<span class="mono">ai_v1</span>) se conserva únicamente como referencia comparativa: para evaluar sobre
ella hay que pedirla <b>explícitamente</b>. Importa porque optimizar contra ruido independiente premia
justo los sesgos optimistas que el retrofit vino a corregir.</div>
<div class="note"><b>Esta tabla compara el mundo sintético consigo mismo.</b> Dice que ai_v2 dejó de
ser ruido independiente, no que sus colas o su agrupamiento tengan el tamaño de los del mercado. Esa es
otra pregunta, y se responde con datos reales en §2.10.</div>

%%FIDELITY%%

<h3>2.11 · Qué garantizan los tests sobre los datos</h3>
<p>La base cuenta con <b>%%NTESTS%% tests</b> automatizados y <i>linting</i> (ruff) sobre todo el código.
Más allá del número importa <b>qué</b> se garantiza, así que la lista va repartida por capítulos: aquí los
invariantes del sustrato, en §3.10 los de una operación y en §4.15 los del ranking.</p>
<ul>
<li><b>Determinismo:</b> la misma semilla produce las mismas velas, byte a byte.</li>
<li><b>Validez de las velas:</b> máximo ≥ cuerpo ≥ mínimo y precios positivos, en todos los regímenes,
incluidos los que activan saltos.</li>
<li><b>Estructura de correlación:</b> activos que cargan el mismo factor correlacionan positivamente; un
shock mueve al activo que lo carga.</li>
<li><b>Microestructura ajustada en varianza:</b> añadir colas, agrupamiento o autocorrelación no cambia
la volatilidad total; la autocorrelación tiene el signo esperado (negativo → reversión, positivo →
tendencia).</li>
<li><b>Neutralidad de los valores por defecto:</b> con la microestructura desactivada, el generador
reproduce exactamente el mundo previo.</li>
<li><b>El medidor de fidelidad mide lo que dice:</b> recupera el signo y el tamaño de una autocorrelación
conocida, distingue una serie con colas t-Student de una gaussiana y una con volatilidad agrupada de una
plana, devuelve NaN (no 0) cuando la serie es degenerada, y su correlación de rangos da +1 y −1 en
órdenes idénticos e invertidos. El estudio completo es determinista y declara los símbolos sin
contraparte real en vez de rellenarlos.</li>
<li><b>Las señales no pueden mentir sobre su profundidad:</b> un test compara el
<span class="mono">history_from</span> declarado en el catálogo con el registro de mediciones de la sonda
y falla si alguien declara una fecha que el registro no respalda.</li>
<li><b>Puente al backtest:</b> el output del generador es directamente consumible por el motor de
backtest, y el ciclo completo (estrategia → riesgo → ejecución) funciona sobre datos sintéticos.</li>
</ul>

<h2 id="s3">3 · Trade: cómo se ejecuta una operación</h2>
<p class="lead">Este capítulo sigue <b>una sola operación</b> desde que existe una barra cerrada hasta que
su resultado entra en la curva de equity: quién decide, quién autoriza, a qué precio se llena, qué cuesta
y cómo se apunta. Todo lo que el capítulo 4 puntúa sale de aquí, así que cualquier optimismo escondido en
estas ocho secciones contamina todas las cifras del documento.</p>

<h3>3.1 · Se conduce el sistema real, no una réplica</h3>
<p>El backtest <b>no reimplementa</b> la lógica de trading. Inyecta un reloj simulado, una fuente de
datos con anti-look-ahead, un modelo de mercado intrabar y un estado en memoria, y hace avanzar el
<b>mismo orquestador que operaría en vivo</b> un día a la vez.</p>
<div class="why"><b>Por qué.</b> Elimina de raíz toda una clase de divergencia backtest/live —la causa
número uno de sorpresas en producción—. Lo que se testea es, literalmente, lo que se ejecutaría. La única
costura que cambia es el <b>modelo de mercado</b>: en vivo se entra al precio de la señal y se valora
contra el último cierre; en backtest se entra al <i>open</i> siguiente y los stops se comprueban contra el
máximo y el mínimo de la barra (§3.4).</div>

<h3>3.2 · Anti-look-ahead</h3>
<p>La fuente de datos solo devuelve barras <b>estrictamente anteriores</b> al día del reloj: la decisión
se toma con el cierre de ayer, nunca con el de hoy (que aún no ha cerrado desde el punto de vista de la
decisión). El bloque de observación cross-sectional (§4.2) y el radar de señales (§2.2) usan exactamente
el mismo corte temporal, para que ninguna feature adelante información. El corte vive en la <b>capa que
sirve los datos</b>, no en cada consumidor: así no depende de que nadie se despiste.</p>

%%RISKGATE%%

<h3>3.4 · El llenado y la convención pesimista</h3>
<p>La ejecución es deliberadamente conservadora:</p>
<ul>
<li>La decisión se toma con la barra <b>cerrada</b>; la <b>entrada</b> se ejecuta al <b>open del día
siguiente</b> (no al cierre con el que se decidió).</li>
<li>El stop-loss y el take-profit se evalúan contra el <b>máximo/mínimo</b> de la barra.</li>
<li><b>Convención pesimista explícita:</b> si en la misma barra se tocan stop y take-profit, gana el
<b>stop</b>; si la apertura abre pasado el nivel del stop (hueco), se llena al <b>open</b>, peor.</li>
</ul>
<div class="why"><b>Por qué pesimista.</b> Un backtest optimista es peligroso. Ante la ambigüedad
intrabar, preferimos <b>subestimar</b> el resultado: si una estrategia parece buena aquí, tiene margen.</div>
<p>Toda esa convención trata las <b>24 horas</b> de la vela como un bloque opaco, y durante mucho tiempo
se justificó solo por prudencia. Prudente no es lo mismo que <i>medido</i>: la pregunta de cuánta
formación de precio cae en la ventana que el motor no ve tiene una respuesta numérica, y es la siguiente
sección.</p>

%%SESSIONS%%

<h3>3.6 · Costes de ejecución: spread, volatilidad e impacto</h3>
<p>Cada ejecución paga comisiones y deslizamiento (<i>slippage</i>). El deslizamiento <b>no es una
constante</b>: se calcula orden a orden como suma de tres términos, cada uno responsable de una
pregunta distinta:</p>
<table><thead><tr><th>Término</th><th>Qué mide</th><th>De qué depende</th></tr></thead><tbody>
<tr><td><b>Medio spread</b></td><td>Lo que cuesta cruzar el libro por existir, aunque la orden sea
minúscula</td><td>Solo del <b>símbolo</b> (tabla explícita: de 0,4 pb en un índice amplio a 20-25 pb en
un altcoin de segunda fila)</td></tr>
<tr><td><b>Volatilidad</b></td><td>El ensanchamiento de la horquilla cuando el activo se mueve</td>
<td>De la <b>volatilidad reciente</b> del activo (20 barras cerradas)</td></tr>
<tr><td><b>Impacto</b></td><td>Mover el precio con el propio tamaño</td><td>De la <b>fracción del
volumen de la barra</b> que consume la orden, por la <b>ley de raíz cuadrada</b></td></tr>
</tbody></table>
<p>La ley de raíz cuadrada (Almgren-Chriss, y la evidencia empírica de ejecución) dice que el coste de
impacto crece con la <b>raíz</b> de la participación, no linealmente: cuadruplicar el tamaño duplica el
impacto. El resultado se limita a un techo declarado (500 pb por defecto): más allá, extrapolar el
modelo no significaría nada, y es preferible un tope explícito a un número inventado.</p>
<div class="why"><b>Por qué importa.</b> Un deslizamiento plano de 5 pb dice que mover un millón de
dólares en un altcoin ilíquido cuesta lo mismo que moverlos en BTC. Eso no es una simplificación
conservadora: es un <b>subsidio</b> a las estrategias que rotan mucho y operan lo pequeño e ilíquido,
justo donde un backtest tiende a encontrar señales espurias.</div>

<h3>3.7 · Techo de capacidad y fills parciales</h3>
<p>Ninguna orden puede consumir más de una <b>fracción del volumen de la barra</b> (10% por defecto).
Lo que exceda ese techo <b>no se llena</b>: la orden queda <i>parcialmente llenada</i> y la posición se
abre por el tamaño realmente ejecutado. Si la barra no da ni para eso, la orden se <b>rechaza</b>.</p>
<p>Las <b>salidas</b> están exentas del techo, por asimetría deliberada: entrar es opcional —si no hay
liquidez, no entras— pero salir no lo es. Un cierre se llena entero y <b>paga el impacto de todo el
tamaño</b>, que con la ley de raíz cuadrada puede ser varias veces el de la entrada.</p>
<div class="why"><b>Por qué asimétrico.</b> Si un cierre se llenara parcialmente, la posición quedaría
cerrada en el estado del sistema pero pagada solo a medias: una contabilidad optimista escondida en un
detalle de ejecución. Se prefiere cobrar de más al salir que dejar el agujero.</div>

<h3>3.8 · El volumen como proxy de liquidez</h3>
<p>La columna <span class="mono">volume</span> de las velas, que antes era decorativa (ninguna pieza la
consultaba), es hoy el <b>eje de liquidez</b> del sistema: el generador la escala al volumen típico
negociado de cada activo, y el motor de ejecución la lee para el impacto y la capacidad. La liquidez se
estima con la <b>mediana</b> de las últimas 20 barras <b>ya cerradas</b> —nunca la de hoy— por dos
razones: el mismo corte anti-look-ahead que usa la estrategia, y porque el volumen se dispara los días
de movimiento fuerte, de modo que tomar ese pico como capacidad regalaría profundidad justo en los días
en que de verdad escasea.</p>
<div class="note"><b>Limitación.</b> Los baselines pasivos (§4.9) siguen pagando un coste plano de
referencia, no el modelo completo: solo hacen dos operaciones sobre los activos más líquidos, y darles
el coste barato endurece el listón que la estrategia debe superar, que es la dirección prudente del
error.</div>
<div class="note"><b>Nota de escala.</b> Con capitales de cinco cifras el término que domina es el
spread: una orden de unos cientos de dólares no mueve el precio de ningún activo del universo, y el
techo de capacidad no llega a morder. Los términos de impacto y capacidad son los que hacen que el
resultado <b>deje de escalar</b> cuando el capital crece, que es exactamente la pregunta que un
backtest con costes planos no puede responder.</div>

%%PNL%%

<h3>3.10 · Qué garantizan los tests sobre una operación</h3>
<ul>
<li><b>Anti-look-ahead:</b> tests dedicados verifican que la barra de hoy queda excluida y la de ayer
incluida, tanto en la fuente de datos como en el ensamblador de régimen. En la validación multiventana se
comprueba además a nivel de ejecución: borrar todas las barras <b>posteriores</b> a una ventana no cambia
ni un decimal de su curva de equity.</li>
<li><b>Contabilidad honesta:</b> el PnL es neto de comisiones; las salidas (stop / take-profit / tiempo)
pasan por el motor de ejecución para pagar costes reales, y las comisiones siguen al tamaño
<b>llenado</b>, no al pedido.</li>
<li><b>Los costes discriminan:</b> un altcoin ilíquido paga más deslizamiento que BTC por el mismo
notional; el coste crece con el tamaño de forma cóncava (4× tamaño → 2× impacto) y con la volatilidad;
sin dato de liquidez no se inventa impacto. Una orden por encima del techo de capacidad se llena
<b>parcialmente</b>.</li>
<li><b>El riesgo es puerta única:</b> ninguna orden se construye sin decisión de riesgo, y el stop
propuesto por una estrategia no puede quedar más lejos del máximo declarado.</li>
<li><b>El modelo de mercado se comporta como se documenta:</b> con stop y objetivo tocados en la misma
barra gana el stop, y un hueco por debajo del stop se llena al open.</li>
</ul>

<h2 id="s4">4 · Estrategias</h2>
<p class="lead">Aquí se decide <b>qué</b> se opera. El capítulo va en el orden en que se construye una
decisión de selección: las estrategias que existen, lo que ven, cómo se puntúa una muestra, cómo se
agrega la distribución, cómo se parte el tiempo, contra quién compite, qué se exige para poder competir,
si el orden sintético se parece al real, cómo se busca y cuánto hay que descontar por haber buscado.</p>

<h3>4.1 · Dos primitivas de regímenes opuestos</h3>
<p>El sistema arranca con dos estrategias paramétricas, puras y <b>sin IA</b>, elegidas como regímenes
<b>opuestos</b>:</p>
<ul>
<li><b>Momentum / seguimiento de tendencia:</b> compra fuerza — cruce de medias al alza más ruptura de
máximos recientes, con filtro de volatilidad. Gana cuando el precio tiene inercia.</li>
<li><b>Reversión a la media:</b> compra debilidad estirada — precio varias desviaciones por debajo de su
media, apostando a que revierte. Gana en rangos sin tendencia.</li>
</ul>
<div class="why"><b>Por qué opuestas, y por qué paramétricas primero.</b> Opuestas, para que el evaluador
y el optimizador tengan algo <b>real</b> que aprender según el escenario (recuérdese que el generador se
construyó precisamente para que unos regímenes premien a una y otros a la otra). Paramétricas y sin IA,
porque son interpretables, baratas de evaluar y optimizables por caja negra antes de dar el salto —más
caro y arriesgado— al aprendizaje por refuerzo por gradiente.</div>
<p>Cada estrategia se registra como <span class="mono">{tipo, parámetros}</span>, de modo que generar una
variante no requiere escribir código. Parámetros por defecto:</p>
<h4>Momentum</h4>
%%MOM_PARAMS%%
<h4>Reversión a la media</h4>
%%MR_PARAMS%%

<h3>4.1-bis · Las seis familias temáticas</h3>
<p>Las dos primitivas de arriba miran precio y volumen, y usan las señales externas solo como
<b>puerta</b>. Eso bastaba mientras la señal confirmaba o desmentía lo que el precio ya había dicho;
deja de bastar en cuanto la señal quiere ser <b>la tesis</b>. Y hay un motivo aritmético: el radar
colapsaba treinta fuentes en seis números, así que cinco estrategias que leyeran
<span class="mono">signal_tone</span> no serían cinco apuestas sino una repetida cinco veces.</p>
<p>Por eso el radar publica ahora <b>quince números más</b> —una terna de tono, intensidad y cobertura
por cada uno de los cinco temas— y encima de ellos viven seis familias nuevas. Todas tienen la misma
forma: un <b>núcleo de precio</b> que corre solo y una <b>capa de señal</b> que modula lado, tamaño y
elegibilidad. La capa es <b>inerte por construcción</b> con los parámetros por defecto: no puede
cambiar nada con ningún radar, vacío o lleno.</p>
%%THEMED_FAMILIES%%
<div class="why"><b>Los umbrales de señal no entran en el espacio de búsqueda, y el argumento no es la
inercia.</b> Dieciséis de las treinta fuentes empezaron a existir el día que arrancó la captura, así
que «cobertura de un tema» está correlacionada casi uno a uno con <b>la fecha</b>. Un piso de tono
sorteable dejaría al optimizador elegir implícitamente <i>en qué tramo de historia se le permite
operar</i> a la estrategia, y rankearía por disponibilidad de datos en vez de por criterio. Es el mismo
fallo que el mínimo de cobertura existe para impedir, y aquí sería peor porque la señal es la primitiva.
La consecuencia se acepta y se escribe: <b>la capa se afirma, no se optimiza</b>. Si está mal, el
backtest dirá «esta familia no añade nada sobre su núcleo», que es exactamente la medición que se
busca.</div>
<div class="note"><b>Cuatro de las seis se rankean CIEGAS, y dos no pueden dejar de serlo.</b> En el
estudio de transferencia la capa está inerte en las ocho familias —ninguna dimensión sorteable toca un
umbral de señal—, así que lo que se ordena es el núcleo de precio con el filtro abierto de par en par.
Eso es una decisión de diseño y se puede levantar. Lo que no se puede levantar hoy es la profundidad:
de los cinco temas, solo <b>macro</b> (3 fuentes con historia medida), <b>attention</b> (2) y
<b>flow</b> (8) alcanzan cobertura en un backtest histórico; <b>liquidation</b> (1 de 4) y
<b>vol_surface</b> (1 de 2) se quedan por debajo del mínimo y sus dos primitivas se miden como núcleo
ciego en todo el histórico. El criterio para repetir el análisis está declarado en §6.2, con fechas.</div>

<h3>4.2 · El espacio de observación</h3>
<p>Se define de forma <b>explícita</b> y con <b>orden estable</b> lo que la política ve en cada momento de
decisión — el contrato de entrada para el aprendizaje futuro. El cierre de un solo activo no basta.
Contiene %%NOWN%% features del <b>mercado del propio activo</b> (retornos a varios horizontes, tendencia,
RSI/MACD, volatilidad ATR y realizada, posición en el canal de Donchian, drawdown), %%NREGIME%%
features <b>cross-sectional / de régimen</b> (fuerza relativa al mercado, correlación móvil, amplitud del
mercado, volatilidad agregada), un <i>one-hot</i> de clase de activo y los seis números del <b>radar de
señales</b> (§2.2).</p>
<div class="why"><b>Por qué explícito y por qué cross-sectional.</b> Una política necesita situar al
activo en su contexto: no es lo mismo una sobreventa cuando todo el mercado cae (cuchillo) que cuando el
activo es un rezagado en un mercado sano. Los bloques de contexto se inyectan como colaboradores
<b>opcionales</b>: si no están presentes, la observación degrada a un solo activo y el contrato en vivo no
se rompe.</div>
<div class="note"><b>Cautela documentada.</b> El volumen sintético es un proxy simplista; las features
que dependen de él están marcadas como tales y deben interpretarse con cuidado.</div>

<h3>4.3 · La unidad de evaluación es la distribución</h3>
<p>Una estrategia no se juzga por un camino, sino por la <b>distribución de su resultado</b> sobre las
%%V2SAMPLES%% muestras (escenarios × caminos). Puntuar sobre un solo camino es medir ruido; se ha
verificado que un único path no es informativo.</p>
<p>Esa dispersión es <b>entre mundos</b>. Hay una segunda, ortogonal: la que hay <b>dentro de un mundo</b>,
entre tramos de su propia historia. La validación multiventana (§4.8) la mide, y resulta no ser pequeña.
Conviene no confundirlas: la primera pregunta "¿funciona en escenarios distintos?", la segunda "¿funciona
en momentos distintos del mismo escenario?".</p>
<div class="note"><b>Estado.</b> Lo que hoy alimenta al optimizador (§4.13) es la primera dispersión: cada
muestra sigue aportando <b>un</b> número, obtenido con el corte único. Componer las dos agregaciones —el
CVaR entre ventanas y el CVaR entre muestras— no es automático: encadenar dos veces la cola puede ser
excesivamente conservador, y esa decisión se tomará midiendo, no razonando (§6).</div>

<h3>4.4 · La métrica de cabecera: Sharpe penalizado</h3>
<p>La puntuación de una muestra es su <b>headline score fuera de muestra</b>:</p>
<p style="text-align:center"><span class="mono"><b>Sharpe<sub>OOS</sub> − λ·turnover − κ·maxDD</b></span>
&nbsp;&nbsp;(λ sobre la rotación diaria, κ sobre el drawdown en fracción; ambos <b>medidos</b>, no
supuestos: λ = %%LAMBDA%% y κ = %%KAPPA%% — ver §4.5)</p>
<p>El <b>turnover</b> es el <i>notional</i> rotado por día en unidades del capital inicial, contando las
dos patas de cada operación (§3.9). Un turnover de 0,20 significa "cada día rota el 20% de la cartera",
así que mide <i>churn</i> de verdad — tamaño por frecuencia — y no solo el número de operaciones.</p>
<div class="why"><b>Por qué no el Calmar (la métrica que esto sustituye).</b> Durante una fase la
cabecera fue el Calmar out-of-sample (retorno anualizado dividido por el máximo drawdown). Era una mala
elección por tres razones, y se documentan porque explican el diseño del sustituto:
<ul>
<li><b>El máximo drawdown es un extremo, no un momento.</b> Es el estadístico más ruidoso de una curva de
equity: depende de un único par de puntos de un único camino. Ponerlo en el <b>denominador</b> multiplica
la varianza del estimador.</li>
<li><b>Premiaba la inactividad.</b> Una estrategia que opera dos veces en dos años tiene un drawdown
minúsculo y un Calmar altísimo, imbatible por cualquier estrategia real. Un optimizador convergería a "no
operar casi nunca" — un óptimo degenerado y perfectamente alcanzable.</li>
<li><b>Degeneraba sin drawdown.</b> Una estrategia sin caída alguna recibía, por convención, la peor
puntuación posible: cero.</li>
</ul>
El headline actual cierra las tres puertas. El drawdown, si se cobra, entra como <b>penalización suave y
aditiva</b> (ni denominador ni descarte binario), el Sharpe no se puede inflar dejando de operar (una curva
plana puntúa 0, no infinito) y la rotación tiene un precio explícito, de modo que el óptimo degenerado
desaparece <i>por construcción</i>, no por vigilancia. Nótese que las dos puertas que de verdad cerraban el
óptimo degenerado son el Sharpe y el precio de la rotación, <b>no</b> el término de drawdown: por eso la
calibración de §4.5 puede dejarlo en cero sin reabrir ninguna de ellas.</div>
<div class="note"><b>Anualizar: dos unidades que no son la misma.</b> El Sharpe de la cabecera está
anualizado, y anualizar mal lo desplaza sistemáticamente. Hay que separar dos casos:
<ul>
<li><b>Tiempo de calendario (el CAGR).</b> Un año natural tiene 365 días para todo el mundo: una acción
que renta un 10% entre enero y diciembre ha rentado un 10% anual, haya cotizado 252 sesiones o 365. El
CAGR divide <i>siempre</i> por días naturales, sin distinguir clase de activo.</li>
<li><b>Número de observaciones (Sharpe, Sortino, volatilidad).</b> Se anualizan por
<span class="mono">√N</span>, donde N es cuántos retornos entran en un año — y ahí sí depende del
mercado: cripto cotiza 24/7 (una barra por día natural, N&nbsp;=&nbsp;365) y la renta variable solo en
sesión (N&nbsp;=&nbsp;252).</li>
</ul>
Una constante global de 365 aplicada también a la renta variable <b>inflaba su Sharpe y su volatilidad
un 20%</b> (<span class="mono">√(365/252) = 1,204</span>). Ahora el factor lo fija el <i>universo</i>:
252 si la cartera es exclusivamente bursátil, 365 en cuanto hay un activo 24/7 — porque el backtest
recorre la <b>unión</b> de días con barra, así que basta un cripto para que haya un punto cada día
natural. El factor se resuelve una sola vez, se aplica por igual a la estrategia y a sus baselines
(comparar dos Sharpe anualizados con escalas distintas no significaría nada) y se <b>reporta</b> en las
métricas como <span class="mono">periods_per_year</span>, para que las cifras publicadas sean
interpretables sin mirar el código. El universo sintético mezcla cripto y renta variable, así que todas
las cifras de este documento están anualizadas por 365.</div>

%%CALIBRATION%%

<h3>4.6 · El estadístico de recompensa: CVaR al 25%</h3>
<p>Los headline scores de todas las muestras se agregan con el <b>CVaR@25%</b> — la media del peor
cuartil (<i>Expected Shortfall</i>). Ese número <b>es</b> la recompensa que optimiza el buscador y el
criterio con el que se ordena el ranking.</p>
<div class="why"><b>Por qué CVaR y no la media, ni «media − λ·desviación».</b> El CVaR premia la
<b>robustez</b> y castiga explícitamente la <b>cola mala</b>: optimizar la media del peor 25% empuja
hacia políticas que no se hunden en los escenarios adversos. Frente a «media − λ·desviación», que fue el
criterio anterior, tiene dos ventajas concretas: no castiga la varianza <b>al alza</b> (una política que
a veces sorprende bien no compra con eso su cola mala) y no depende de un λ arbitrario. Es además más
estable que un único cuantil (P25). Se reportan igualmente media, desviación y P25 para no ocultar la
forma de la distribución.</div>

<h3>4.7 · Hold-out de escenarios enteros</h3>
<p>Los escenarios se reparten en entrenamiento y validación de forma determinista, reservando
<b>arquetipos macro completos</b> como hold-out (aproximadamente 22 train / 8 validación).</p>
<div class="why"><b>Por qué escenarios enteros y no caminos sueltos.</b> Si un camino de validación
perteneciera a un escenario que también está en entrenamiento, sería <b>fuga de arquetipo</b>: el modelo
ya habría visto esa "física" y el hueco de sobreajuste quedaría enmascarado. Partir por escenario lo
impide.</div>

%%VALIDATION%%

<h3>4.9 · Baselines: el listón que hay que superar para «aprobar»</h3>
<p>Una estrategia no funciona porque su puntuación sea positiva: funciona si bate a lo que consigue
cualquiera <b>sin hacer nada</b>. Sobre cada muestra se construyen tres alternativas pasivas:</p>
<table><thead><tr><th>Baseline</th><th>Qué hace</th></tr></thead><tbody>
<tr><td>Comprar y mantener BTC</td><td>Compra BTC el primer día de la ventana y liquida el último.</td></tr>
<tr><td>Cartera equiponderada</td><td>Reparte el capital a partes iguales entre todo el universo, sin rebalancear.</td></tr>
<tr><td>Comprar y mantener SPY</td><td>Lo mismo con el índice de renta variable.</td></tr>
</tbody></table>
<p>Los tres se puntúan con el <b>mismo headline score</b>, sobre la <b>misma ventana out-of-sample</b>
(la frontera train/test sale de una única función compartida con el motor de backtest) y pagando las
<b>mismas comisiones y slippage</b> en ambas patas. El veredicto <b>aprueba / no aprueba</b> compara
recompensas agregadas: el CVaR@25% de la estrategia contra el del <b>mejor</b> baseline, sobre las
muestras de validación — batir baselines en escenarios que el optimizador ya vio no probaría nada.</p>
<p>Y exige una segunda condición, añadida después de medir que la primera sola no bastaba: la estrategia
tiene que ser <b>rankeable</b>, es decir, superar el suelo de actividad de §4.10. Una curva plana puntúa
0 exacto y bate a los pasivos en cualquier periodo bajista sin haber abierto una posición; eso no es
batirlos, es no haber jugado.</p>
<div class="why"><b>Por qué en la cola y no en la media.</b> El gate se juega con el mismo estadístico
con el que se optimiza. Una estrategia con mejor media pero peor cuartil malo <b>no</b> aprueba: si la
recompensa es la cola, el listón también tiene que serlo. El porcentaje de muestras en las que gana se
reporta como información, pero no decide.</div>
<div class="note"><b>Sin rival no hay aprobado.</b> Si un baseline no se puede construir (por ejemplo,
SPY en un universo solo cripto), no se sustituye ni se rellena: se declara como ausente. Y si no hay
ningún baseline disponible, el veredicto es <b>no aprueba</b> — un filtro que no se puede evaluar no es
un filtro superado. Lo mismo vale para la actividad: si no se ha medido, no se da por buena.</div>

%%ACTIVITY%%

%%TRANSFER%%

%%TRANSFER_EXTENDED%%

%%THEMES_REAL%%

%%EXTENDED_GRID%%

%%SIGNALCHANNEL%%

<h3>4.13 · El optimizador: Cross-Entropy Method</h3>
<p>La mejora de las primitivas se plantea, en esta fase, como <b>optimización de caja negra</b> de sus
parámetros mediante el método de entropía cruzada (CEM): se muestrean parámetros de una gaussiana, se
conserva el cuartil élite y se reajusta la gaussiana hacia él, iterando. Es determinista con semilla.</p>
<div class="why"><b>Por qué caja negra primero.</b> Es "RL-lite" honesto que encaja con el backtest
existente sin nueva teoría, y deja la interfaz limpia para enchufar después aprendizaje por refuerzo por
gradiente por-paso si se decide extender el contrato de la estrategia.</div>
<p>Rangos de búsqueda sobre los que opera el CEM:</p>
<h4>Momentum</h4>
%%SPACE_MOM%%
<h4>Reversión a la media</h4>
%%SPACE_MR%%
%%THEMED_SPACES%%
<div class="note"><b>Lo que NO entra en el espacio de búsqueda.</b> Ninguna feature de señal, ningún
umbral de puerta y ningún parámetro del radar (§2.2). Son constantes declaradas en código, y hay un test
que lo demuestra: añadir una dimensión <b>sustituye</b> el espacio publicado en vez de ampliarlo. Es lo
que mantiene fija la huella de las 16 configuraciones sobre las que se ha medido todo lo demás — que hoy
son <b>las 16 primeras de 64</b>: las seis familias temáticas se añadieron <b>al final</b> de la lista de
familias, y como el hipercubo latino se siembra con <span class="mono">semilla + índice de familia</span>,
eso preserva byte a byte las configuraciones publicadas. Insertar una familia en medio las habría
<b>sustituido en silencio</b> por otras dieciséis con los mismos nombres; hay un test que lo congela y una
guarda que se niega a sobrescribir un informe cuya rejilla no coincide con la que se va a escribir.</div>

<h3>4.14 · Descuento del sobreajuste por múltiples pruebas</h3>
<p>Buscar sobre un espacio de parámetros garantiza encontrar algo que brilla <i>aunque no haya nada que
encontrar</i>: con suficientes intentos sobre puro ruido, el mejor Sharpe esperado crece solo. La
herramienta pone número a ese efecto por dos vías independientes.</p>
<p><b>PBO (<i>Probability of Backtest Overfitting</i>, por CSCV).</b> Con la matriz muestras ×
configuraciones que el propio buscador genera, las muestras se parten en bloques; en cada combinación de
mitades se elige la configuración ganadora <i>dentro</i> de muestra y se mira su rango <i>fuera</i> de
muestra. El PBO es la fracción de particiones en las que la ganadora cae por debajo de la mediana. Un
50% significa que elegir por backtest equivale a tirar una moneda.</p>
<p><b>DSR (<i>Deflated Sharpe Ratio</i>).</b> Dado el Sharpe del ganador y la dispersión de los Sharpe de
<b>todos</b> los intentos, se calcula el máximo esperado bajo la hipótesis nula (no hay señal) y se
devuelve la probabilidad de que el Sharpe verdadero sea mayor que cero una vez descontado ese umbral,
corrigiendo además por asimetría y colas gruesas de los retornos.</p>
<div class="why"><b>Por qué las dos.</b> Responden a preguntas distintas. El DSR pregunta "¿es creíble
<i>este</i> Sharpe habiendo probado N configuraciones?"; el PBO pregunta "¿el procedimiento de
<i>elegir</i> por backtest acierta?". La segunda es la que importa cuando lo que se publica no es un
número sino un método.</div>

<h3>4.15 · Qué garantizan los tests sobre el ranking</h3>
<ul>
<li><b>La métrica de cabecera no tiene óptimo degenerado:</b> una curva plana puntúa 0 (no infinito), el
drawdown penaliza de forma aditiva y no en el denominador, y rotar más con la misma curva de equity
puntúa peor.</li>
<li><b>El ranking compite por la cola:</b> una distribución con mejor media pero peor cuartil malo pierde,
y la varianza al alza no se castiga.</li>
<li><b>El filtro de baselines no se puede esquivar:</b> sin ningún baseline disponible el veredicto es
"no aprueba", y los baselines pagan las mismas comisiones que la estrategia.</li>
<li><b>No hay fuga temporal entre folds:</b> para cada esquema (walk-forward anclado y rodante, CPCV,
corte único) se comprueba <b>día a día</b> —sobre fechas explícitas, no sobre los intervalos que
construyeron el fold— que ningún día aparece a la vez en entrenamiento y test, y que los huecos respetan
la purga y el embargo declarados. El auditor se ejercita también en negativo: folds con solape y folds
con purga insuficiente <b>deben</b> hacer fallar la ejecución antes de gastar cómputo. Y se fija el
invariante que delimita qué hace la purga: purgar cambia la ventana in-sample y <b>no</b> la
out-of-sample; dos folds con el mismo test y entrenamientos distintos dan idéntico resultado OOS.</li>
<li><b>El descuento por múltiples pruebas discrimina:</b> un caso de habilidad genuina da PBO ≈ 0 y uno
de sobreajuste puro da PBO ≈ 1; probar más configuraciones sube el listón del DSR.</li>
<li><b>El espacio de búsqueda no crece por la puerta de atrás:</b> el CEM sólo reconstruye estrategias, y
ninguna feature de observación o de señal puede convertirse en parámetro sorteable.</li>
</ul>

<h2 id="s5">5 · Resultados</h2>
<p class="lead">Un <i>backtest</i> es una medición sobre historia; un resultado es lo que hace el sistema
cuando corre. Este capítulo se llena solo, desde dos ficheros que escribe el propio <i>runner</i> y desde
ningún otro sitio, y lo que aún no tiene son <b>meses de calendario</b>, que es lo único del proyecto que
no se puede comprimir con cómputo.</p>
<h3>5.1 · El diario de ciclos</h3>
<p>Hasta ahora el sistema guardaba la <b>foto</b> —qué posiciones hay y cuánto PnL llevan— y no la
<b>película</b>. La foto es todo lo que hace falta para seguir operando tras un reinicio, y es exactamente
lo que no sirve para auditar: no dice qué se decidió, con qué precio de referencia, por qué el riesgo
rechazó una señal ni cuánto deslizamiento se cobró de verdad. Desde ahora cada ciclo escribe una línea en
<span class="mono">data/live/cycles.jsonl</span> con todo eso.</p>
<p>Tres propiedades, y ninguna es cosmética. <b>Append-only</b>: nunca se reescribe el fichero, se añade
una línea; un fallo a mitad de ciclo deja una cola ilegible que el lector cuenta y salta, no un archivo de
meses truncado. <b>Con <span class="mono">fsync</span> por línea</b>: el proceso corre durante meses en una
máquina doméstica, donde el corte de luz no es un caso hipotético sino <i>el</i> caso. Y <b>rotación</b>
por mes o al superar los 8&nbsp;MB, con la secuencia en el nombre del fichero para que el orden alfabético
de los <i>shards</i> sea el cronológico. Es el mismo principio que gobierna el archivo crudo de señales
(§2.2): lo que no se capture hoy no existirá, porque nadie publica el pasado de un libro de órdenes.</p>
<table><thead><tr><th>Bloque</th><th>Qué se publica</th><th>De dónde sale</th></tr></thead><tbody>
<tr><td><b>Diario de ciclos</b></td><td>Símbolos evaluados, señales con su confianza, decisión del riesgo
con su motivo, orden enviada con el precio de referencia, y el llenado con comisión y
<span class="mono">slippage_bps</span> realmente cobrado.</td><td>Registro append-only del <i>runner</i>
(§3.9). <b>Hecho.</b></td></tr>
<tr><td><b>Curva y posiciones</b></td><td>PnL neto marcado a mercado por ciclo; posiciones abiertas con su
precio de marca y cerradas con PnL neto de comisiones.</td><td>Diario + estado persistido (§3.9).
<b>Hecho.</b></td></tr>
<tr><td><b>Riesgo desplegado</b></td><td>Exposición frente al límite, número de posiciones frente al
máximo y caída máxima desde el pico.</td><td>Motor de riesgo (§3.3), anotado en cada línea.
<b>Hecho.</b></td></tr>
<tr><td><b>Divergencia live-vs-backtest</b></td><td>La cifra que justifica todo el capítulo 3: cuánto se
aparta lo ejecutado de lo que el motor predecía, separando precio de llenado, coste y latencia.</td>
<td>Diario de ciclos contra el mismo periodo re-simulado (§5.4). <b>Medición cableada; la cifra necesita
meses de calendario y el estudio se niega a publicarla antes.</b></td></tr>
</tbody></table>
<p>Cada línea lleva además los dos instantes que hacen medible la latencia: <b>cuándo se decidió</b> la
orden —justo después de que el riesgo aprobara y con el precio de referencia ya fijado— y cuándo se
llenó. El sello del ciclo no sirve para eso, porque la línea se escribe al <i>terminar</i> el ciclo y es
posterior a todos sus <i>fills</i>; restarlos daría un hueco negativo. Es información que no se puede
reconstruir después, que es el mismo motivo por el que el diario existe.</p>
<h3>5.2 · Por qué es PnL y no <i>equity</i> de cuenta</h3>
<p>En vivo el sistema opera <b>sin capital declarado</b>: el motor de riesgo dimensiona en absoluto
—tamaño máximo por posición, exposición máxima— y no como fracción de un patrimonio (§3.3). Declarar un
capital sólo para poder dibujar una curva de <i>equity</i> <b>cambiaría el dimensionado de todas las
órdenes</b>, así que no se hace: se publica la curva de PnL neto, que es la misma salvo una constante
aditiva, y la caída máxima en dólares, porque un porcentaje necesitaría una base que aquí no existe. Los
ciclos <b>pausados</b> dejan línea pero no entran en la curva: no se marcan a mercado, y meterlos con un
cero aplanaría justo el tramo en que el sistema estuvo parado.</p>
<h3>5.3 · Durabilidad</h3>
<p>Perder el estado no es perder un fichero: es que el <i>runner</i> olvide las posiciones abiertas, no las
cierre nunca y siga abriendo otras contra los mismos límites. Antes bastaba un fichero corrupto para que
arrancara de cero <b>en silencio</b>. Ahora cada escritura deja copia rotatoria (tres niveles), el estado se
escribe a un temporal que se sincroniza antes de renombrarse, y si al arrancar no parsea se recupera de la
copia más reciente que sí lo haga <b>avisando por Telegram</b>. Si no hay ninguna utilizable, arranca de
cero pero diciéndolo.</p>

%%DIVERGENCE%%

<div class="why"><b>Por qué esto no se puede acelerar.</b> Es la única parte del proyecto que consume
<b>tiempo de calendario</b> y no cómputo: la divergencia entre lo ejecutado y lo simulado necesita meses
de operaciones para ser medible. Por eso el paper trading corre en paralelo a todo lo demás (§6): cada
semana sin correr es una semana perdida al final.</div>

<h2 id="s6">6 · Limitaciones y evoluciones</h2>
<p class="lead">Declarar los huecos es parte de la honestidad de la herramienta. Este capítulo tiene tres
partes: lo que está cerrado y medido, lo que sigue abierto y en qué orden, y los límites que no se van a
cerrar porque son estructurales.</p>

<h3>6.1 · Lo que está cerrado, y qué significa «cerrado»</h3>
<p>Cerrado significa <b>medido y publicado</b>, no «salió bien». Dos de estas líneas devolvieron el
resultado que no se quería, y están igual de cerradas: la pregunta tiene respuesta.</p>
<table><thead><tr><th>Área</th><th>Qué quedó establecido</th></tr></thead><tbody>
<tr><td>Realismo del generador</td><td>Colas, agrupamiento, estructura serial y saltos, con valores
neutros por defecto: el sustrato dejó de mentir en la dirección optimista (§2.8).</td></tr>
<tr><td>Fidelidad contra el mercado real</td><td>Cobertura del 98 % y nueve umbrales que el estudio puede
fallar. La ordenación entre activos sigue floja, y se publica (§2.10).</td></tr>
<tr><td>Transferencia de ranking</td><td><b>Resultado negativo y aceptado:</b> el mundo sintético pasa
todos los umbrales de fidelidad y aun así no ordena las estrategias como el mercado (§4.11).</td></tr>
<tr><td>Métrica y ranking</td><td>Headline penalizado, CVaR@25%, baselines pasivos con las mismas
comisiones, DSR y PBO: el óptimo degenerado del Calmar desapareció (§4.4–4.14).</td></tr>
<tr><td>Pesos λ y κ</td><td>Medidos por barrido en rejilla sobre cientos de backtests: la superficie es
plana, penalizar no estabiliza y los costes ya están dentro del Sharpe (§4.5).</td></tr>
<tr><td>Suelo de actividad</td><td>Rankear exige operar. El umbral es medido, el barrido se publica
entero y el gate pasa de siete aprobadas a una (§4.10).</td></tr>
<tr><td>Costes que muerden</td><td>Deslizamiento por símbolo, volatilidad y tamaño con ley de raíz
cuadrada, techo de capacidad y fills parciales (§3.6–3.8).</td></tr>
<tr><td>Ventana ciega del backtest</td><td>Medida por sesión horaria: el hueco cierre→open es
<b>cero</b> (0,55 pb), pero llegar una hora tarde cuesta varias veces el coste modelado (§3.5).</td></tr>
<tr><td>Validación temporal multiventana</td><td>Walk-forward y CPCV con purga y embargo, auditados
contra fuga. El corte único no estaba sesgado: estaba <b>arbitrario</b> (§4.8).</td></tr>
<tr><td>Plataforma de señales externas</td><td>Diecisiete fuentes con adaptador, profundidad medida por
sonda y cableado al espacio de observación en backtest y en vivo, con la compuerta cumplida: los scores
publicados no se movieron (§2.2).</td></tr>
<tr><td>Limpieza de consistencia</td><td>Anualización por clase de activo, no-reproducibilidad del
diseñador documentada, y el universo operado separado del sintético con el motivo escrito
(§2.1, §2.5, §4.4).</td></tr>
</tbody></table>

<h3>6.1-bis · Lo que no se podía medir hacia atrás, y qué se midió al final</h3>
<p>Cuando se diseñaron las seis familias temáticas, esta sección decía que la capa de señal no se podía
evaluar hacia atrás y declaraba la fecha en que se podría. <b>Esa afirmación era demasiado pesimista y ya
está medida</b>: §4.16 publica la comparación pareada —misma familia, misma ventana, mismas barras, la
puerta abierta y cerrada— sobre <b>cinco</b> de las seis. Lo que queda como limitación es más pequeño y
más preciso que lo que se anunció, y conviene separar las tres cosas que se confundían.</p>

<p><b>Primero: la evaluabilidad se mide, no se declara.</b> La versión anterior de este documento repartía
las familias entre «con archivo real» y «sin él» usando el flag <span class="mono">backtestable</span> del
catálogo. Al sondear el radar sobre el archivo, ese reparto falla en los dos sentidos.
<span class="mono">vol_surface</span> figuraba como ciego y <b>se lee</b>: alcanza 0,333 de cobertura
porque <span class="mono">deribit_volatility</span> publica desde <b>2021-03-24</b>, y el catálogo solo lo
daba por no backtestable porque su profundidad <i>medida</i> aún no llega a los 365 días que exige
<span class="mono">depth.MIN_MEASURED_DAYS</span>. La frase «sus fuentes empezaron a existir el día que
arrancó la captura» era cierta para <span class="mono">liquidation</span> y <b>falsa</b> para
<span class="mono">vol_surface</span>. En sentido contrario, <span class="mono">cex_listings</span> es
backtestable y pertenece a <span class="mono">attention</span>, pero es un calendario de listados y BTC no
se lista: sobre ese símbolo no aporta lectura ninguna.</p>

<p><b>Segundo: el único tema que de verdad no llega es <span class="mono">liquidation</span></b>, con una
cobertura máxima medida de <b>0,167</b> frente al 0,25 que exige la puerta, y legible en el <b>0,0 %</b>
de las sondas. Su familia —<span class="mono">liquidation_cascade</span>— es la única que se declara no
evaluable, y ahora con ese número al lado en vez de con una frase. Sus fuentes (Hyperliquid, el 2026-08-13)
sí empezaron a existir el día de la captura, y <span class="mono">lending_health</span> ni eso, porque su
reloj no ha arrancado por falta de credencial.</p>

<p><b>Tercero, y es lo que sí queda abierto: medir no es tener potencia.</b> De las cinco medidas, dos
—<span class="mono">attention_ignition</span> y <span class="mono">vol_term_structure</span>— salieron
<b>sin potencia</b>: la puerta no llegó a atar en ninguna de las 20 parejas, así que armado y ciego dan el
mismo número exacto. El motivo no es que el tema sea ciego, sino la intersección de tres cosas raras: que
el tema sea legible ese día (<span class="mono">attention</span> lo es en el 19,3 % de las sondas,
<span class="mono">vol_surface</span> en el 4,2 %), que el núcleo de precio quiera entrar, y que el tono
cruce el umbral. Eso no se arregla esperando a que una fuente cumpla 365 días: se arregla con
<b>más archivo</b>, y el calendario de esa espera es el que ya estaba escrito.</p>

<p><b>El criterio para repetirlo, que no cambia.</b> La transferencia trocea el histórico en sub-ventanas
de <b>544 días</b> —la longitud de un camino sintético, para que los dos lados comparen mundos y no
precisiones de estimador— y cada estrategia necesita además <b>180 días</b> de calentamiento: son
<b>724 días de señal capturada</b> para correr UNA sub-ventana con vista. Para Hyperliquid la primera
fecha posible es el <b>2028-08-06</b>. Las cinco sub-ventanas de la geometría publicada pedirían unos
2.900 días, ocho años: <b>esas cifras no se van a poder reproducir con vista</b>, y decir lo contrario
sería prometer un calendario que no depende de nadie.</p>

<div class="note"><b>Lo que sí se puede afirmar hoy, y no es poco.</b> En dos familias
—<span class="mono">flow_persistence</span> y <span class="mono">signal_composite</span>— el intervalo por
bloques de la diferencia pareada <b>excluye el cero</b>: la capa ayuda. Con las reservas de §4.16, que son
serias y están escritas allí: sin corrección por comparaciones múltiples, con N efectivo menor que el
nominal, y con el compuesto en la posición incómoda de que su tesis <i>entera</i> vive en la capa, de modo
que lo medido en su caso es la magnitud y no la dirección. Es la primera evidencia del proyecto en la que
una señal externa mueve una decisión sobre mercado real, y es deliberadamente modesta.</div>

<h3>6.2 · Lo que queda abierto, por criticidad</h3>
<p>El orden responde a una asimetría de coste: una estrategia añadida hoy se re-evalúa gratis cuando el
juez mejore; un juez malo contamina todo lo que puntúe mientras siga malo. Por eso el sustrato y el juez
van delante de la cosecha, y el paper trading se lanza en paralelo porque es lo único que compra tiempo de
calendario.</p>
<table><thead><tr><th class="n">#</th><th>Trabajo pendiente</th><th>Por qué está donde está</th></tr></thead><tbody>
<tr><td class="n">1</td><td><b>Divergencia live-vs-backtest</b> sobre el diario de ciclos</td><td>El diario
ya se escribe y la vista ya lo lee (§5): lo que falta es calendario. Sigue el primero porque es lo único
que no se puede comprimir después.</td></tr>
<tr><td class="n">2</td><td><b>Canal de observación sintético: barrido de ρ y break-even del IC</b></td>
<td>Es el <b>test de falsación</b> que el radar de señales dejó abierto. Hoy limitar los grados de
libertad <i>reduce</i> el riesgo de sobreajuste pero no lo <i>mide</i>: hasta que exista una cifra de
capacidad predictiva mínima —con ρ = 0 como control— una feature de muestra corta puede estar
sobreajustada y el sistema no puede saberlo.</td></tr>
<tr><td class="n">3–5</td><td>Que el <b>generador emita las señales</b> y re-medir la transferencia de
forma pareada; lote caro de fuentes (apalancamiento observable, opciones, atención legal); índice de
estrés de vendedores forzados</td><td>La transferencia de §4.11 se midió con estrategias que sólo ven
precio y volumen, sobre un mundo cuyo único <i>edge</i> es un AR(1) colocado por régimen. Ampliar el
espacio de inputs <b>en los dos lados</b> es lo que convierte «no transfiere» en una conclusión y no en
un artefacto del instrumento.</td></tr>
<tr><td class="n">6–11</td><td><b>Rigor del juez:</b> CPCV en dos etapas dentro del optimizador;
re-correr la validación con el ensemble completo; alinear los bloques del PBO con fronteras de escenario;
reportar el recuento de muestras fallidas junto al reward; declarar que el DSR asume intentos
independientes; arreglar la <i>ordenación</i> de colas y agrupamiento entre activos</td><td>Seis arreglos
sobre cifras que se publican como garantía. El CEM sigue puntuando con el corte único que su propio
estudio desacredita; los bloques del PBO pueden partir un escenario en dos (la misma fuga de arquetipo que
el hold-out evita); la penalización por fallo domina la cola del CVaR; el CEM no produce los intentos
independientes que el DSR supone.</td></tr>
<tr><td class="n">12</td><td><b>Contingencia:</b> mover el sustrato primario del ranking al histórico
real</td><td>Es la consecuencia directa de §4.11 si los frentes 2–5 no cambian el resultado. Está
<b>bloqueada</b> a propósito: es un cambio caro y sería prematuro hacerlo antes de saber si el problema
era el generador o el espacio de inputs con el que se le preguntó.</td></tr>
<tr><td class="n">13–14</td><td>Optimización CEM completa; presupuesto de latencia de ejecución</td>
<td>La primera está bloqueada por el juez: gastar cómputo caro en un ganador elegido por un juez que no
transfiere es tirar el cómputo. La segunda sale de §3.5 — el hueco es cero, pero la puntualidad tiene un
precio medido.</td></tr>
<tr><td class="n">15–18</td><td>Nuevas estrategias cripto; re-medir λ y κ con los costes nuevos; anotar el
modelo de IA en el manifiesto; guarda operativa por símbolo (sanciones, deslistado, halt)</td>
<td><b>No prioritarias, y por motivos distintos.</b> Añadir candidatos a un juez en el que aún no se
confía multiplica el problema de múltiples pruebas; la superficie de pesos ya salió plana sobre 480
backtests; y la guarda operativa cubre un riesgo que, mientras el dinero sea de papel y el universo se
configure a mano, sigue siendo teórico.</td></tr>
<tr><td class="n">19–20</td><td>Renta variable y mercados de predicción</td><td><b>Segundo plano
explícito.</b> Toda la evidencia empírica del repositorio es cripto; la pata de renta variable del
generador no tiene ni un dato real detrás y el universo de megacaps arrastra sesgo de supervivencia.
Polymarket no tiene histórico que backtestear: el paper trading en vivo es lo que empezará a
generarlo.</td></tr>
</tbody></table>
<p class="tag">El dashboard mantiene el detalle accionable de cada evolución —evidencia medida, criterio
de aceptación y prompt reproducible— con este mismo orden.</p>

<h3>6.3 · Los límites que no se van a cerrar</h3>
<p>No todo hueco es trabajo pendiente. Estos cuatro son propiedades del enfoque, y se declaran para que
nadie los lea como promesas:</p>
<ul>
<li><b>El diseño con IA no es reproducible</b>, y ya no puede serlo: los modelos actuales retiraron los
parámetros de muestreo. Lo que se guarda es el artefacto, no la llamada (§2.6).</li>
<li><b>El histórico real es un único camino.</b> Todas las ventanas reales comparten los mismos ciclos de
2018-2025. Comparar «lo que pasó» con «lo que podría pasar» es una asimetría que sólo se puede declarar,
no eliminar (§2.10).</li>
<li><b>El generador produce el cripto de un año cualquiera, no sus años de manía.</b> El umbral de
aceptación está sobre la mediana; perseguir el p90 de curtosis rompería el nivel de volatilidad
(§2.10).</li>
<li><b>Foco cripto.</b> Los <i>stocks</i> no salen del universo <b>sintético</b> — GLD, TLT y UUP son lo
que hace que los escenarios de tipos y de dólar signifiquen algo para cripto vía factores compartidos —,
pero se puntúa y se opera sólo cripto (§2.1).</li>
</ul>

<h2 id="a1">A · Reproducibilidad</h2>
<p>Toda la cadena es determinista y regenerable. Notas operativas:</p>
<ul>
<li><b>Generar una librería sintética</b> (llama a la IA una vez): diseña los escenarios y sintetiza el
ensemble Monte Carlo, guardando el manifiesto autocontenido y los <i>spec.json</i>.</li>
<li><b>Ampliar o reconstruir</b> caminos sin IA: "resynthesize" regenera a partir de los
<i>spec.json</i> guardados con las mismas semillas.</li>
<li><b>Derivar la librería realista</b> (<span class="mono">ai_v2</span>): el retrofit determinista
enriquece los escenarios con microestructura y regenera, conservando la librería anterior.</li>
<li><b>Capturar datos reales:</b> las barras se cachean en disco, de modo que los estudios que las
consumen (fidelidad, transferencia, sesiones) se repiten sin red y dan lo mismo.</li>
<li><b>Capturar señales:</b> <span class="mono">ai-trader signals capture</span> archiva el crudo
append-only y <span class="mono">ai-trader signals depth</span> mide la profundidad histórica real de
cada fuente.</li>
<li><b>Evaluar y optimizar:</b> el backtest corre sobre las muestras de la librería realista
(<span class="mono">ai_v2</span>, el valor por defecto); el CEM optimiza el CVaR@25% del headline sobre el
conjunto de entrenamiento y reporta la validación, el veredicto frente a los baselines y el descuento
por múltiples pruebas.</li>
<li><b>Artefactos vivos:</b> el dashboard y esta documentación se regeneran desde los datos del repo
(<span class="mono">python -m dashboard.build_dashboard</span> y
<span class="mono">python -m docs.build_docs</span>).</li>
</ul>
<div class="note"><b>Nota de entorno.</b> En la máquina de desarrollo se invoca el intérprete del
entorno virtual directamente para tests y regeneración. Los datos sintéticos generados no se versionan
(se regeneran de forma determinista desde los <i>spec.json</i> y las semillas).</div>

<h2 id="a2">B · Glosario</h2>
<ul>
<li><b>Factor / beta / idiosincrático:</b> fuente de riesgo común; sensibilidad de un activo a ella; la
parte del retorno no explicada por los factores.</li>
<li><b>ATR:</b> rango medio verdadero — medida de volatilidad usada para dimensionar stops.</li>
<li><b>Canal de Donchian:</b> banda entre el máximo y el mínimo de N días; su ruptura señala momentum.</li>
<li><b>z-score:</b> distancia del precio a su media en desviaciones típicas; base de la reversión.</li>
<li><b>Calmar / Sharpe / Sortino:</b> ratios de rentabilidad ajustada por, respectivamente, máximo
drawdown, volatilidad total y volatilidad a la baja.</li>
<li><b>CVaR@25% (Expected Shortfall):</b> media del peor 25% de los resultados; mide la cola.</li>
<li><b>Turnover (rotación):</b> <i>notional</i> negociado por día en unidades del capital inicial,
contando entrada y salida; 0,20 significa que cada día rota el 20% de la cartera.</li>
<li><b>Fill parcial:</b> orden llenada por menos tamaño del pedido porque el techo de capacidad de la
barra no daba para más.</li>
<li><b>Baseline:</b> alternativa pasiva de referencia (comprar y mantener) que una estrategia debe batir
para justificar su existencia.</li>
<li><b>Rankeable:</b> configuración que supera el suelo de actividad y por tanto puede competir en el
ranking y aprobar el gate.</li>
<li><b>DSR (Deflated Sharpe Ratio):</b> Sharpe descontado por el número de configuraciones probadas;
responde a "¿es creíble este resultado habiendo buscado tanto?".</li>
<li><b>PBO (Probability of Backtest Overfitting):</b> con qué frecuencia la configuración ganadora dentro
de muestra queda por debajo de la mediana fuera de muestra.</li>
<li><b>Hold-out:</b> datos reservados que el modelo no ve durante el ajuste, para medir sobreajuste.</li>
<li><b>Walk-forward:</b> validación que avanza en el tiempo — se entrena con el pasado y se puntúa en el
tramo siguiente, repitiendo. Da varias ventanas out-of-sample consecutivas y disjuntas.</li>
<li><b>CPCV (Combinatorial Purged Cross-Validation):</b> se parte el rango en N grupos y se prueban todas
las combinaciones de k como test. Da C(N,k) ventanas con la misma historia, y evalúa cada tramo
acompañado de contextos de entrenamiento distintos.</li>
<li><b>Purga:</b> días de entrenamiento eliminados justo <i>antes</i> del test, para que ninguna
operación abierta al final del entrenamiento tenga su desenlace dentro del test.</li>
<li><b>Embargo:</b> días de entrenamiento eliminados justo <i>después</i> del test, para que la
correlación serial de los retornos no filtre el test en el entrenamiento posterior.</li>
<li><b>CEM:</b> método de entropía cruzada — optimización de caja negra por muestreo y élite.</li>
<li><b>Stylized facts:</b> propiedades estadísticas robustas de los mercados reales (colas gruesas,
agrupamiento de volatilidad, autocorrelación) que un buen simulador debe reproducir.</li>
<li><b>Point-in-time (PIT):</b> propiedad de una serie de señal que garantiza que su valor en el día
<i>t</i> es el que se conocía ese día, y no una revisión posterior.</li>
<li><b>Forward capture:</b> fuente cuyo pasado nadie publica: su profundidad histórica sólo crece
capturando, un día por día real.</li>
</ul>
<p class="meta">Fin del documento · AI-Trader · commit <span class="mono">%%COMMIT%%</span>.</p>
"""
