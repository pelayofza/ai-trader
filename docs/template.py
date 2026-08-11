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
    """Seccion 5.2b: la calibracion de λ y κ, con la evidencia que la sostiene."""
    if not c:
        return (
            "<div class=\"note\"><b>Limitación declarada.</b> El informe de calibración "
            "(<span class=\"mono\">data/calibration/</span>) no está disponible en este árbol, así "
            "que este documento no puede citar las cifras que fijan λ y κ. Regenéralo con "
            "<span class=\"mono\">python -m ai_trader.scoring.weight_study</span>.</div>"
        )
    return f"""
<h3>5.2b · De dónde salen λ y κ (calibración medida)</h3>
<p>λ y κ no son una preferencia estética: son una <b>regla de selección</b>, y una regla de selección se
juzga por lo que elige. Se han barrido en rejilla
(λ ∈ {{{", ".join(_n(x, 2) for x in c["lambdas"])}}} × κ ∈ {{{", ".join(_n(x, 1) for x in c["kappas"])}}})
sobre <b>{c["n_backtests"]} backtests reales</b> de la librería <span class="mono">{c["library"]}</span>:
{c["n_configs"]} configuraciones (hipercubo latino sobre el espacio de búsqueda de las dos primitivas)
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
<p><b>Vigencia tras el modelo de costes por microestructura (§3.4).</b> Este estudio se midió con el
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
    """Seccion 3.7: como se parte el tiempo, y cuanto cambia la respuesta segun como."""
    if not v:
        return (
            "<h3>3.7 · Partición temporal: de un corte a una distribución de ventanas</h3>"
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
<h3>3.7 · Partición temporal: de un corte a una distribución de ventanas</h3>
<p>Un backtest se puede partir en entrenamiento y test de muchas formas, y <b>la forma elegida cambia la
respuesta</b>. El sistema partía cada muestra con un único corte temporal 70/30. Eso tenía tres problemas
distintos, y sólo el primero es evidente:</p>
<ul>
<li><b>Un número no tiene cola.</b> La unidad de evaluación de este proyecto es la distribución y su
estadístico es el CVaR (§5.3) — pero el CVaR de una lista de un elemento <b>es</b> ese elemento. El
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
empeoran</b> ninguna cifra out-of-sample, y hay un test que fija ese invariante (§6). Sirven para otras
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
cola mala (§5.3) y, con una sola ventana, no había cola que promediar — el estadístico robusto se
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
como un error estándar. Y el camino que usa el optimizador (§5.5) sigue puntuando cada muestra con el
corte único: cablear ahí el esquema multiventana es la evolución pendiente. Los
{v["leakage"]["folds_audited"]} folds ejecutados pasaron la auditoría de fuga temporal. Evidencia completa
en <span class="mono">data/validation/report_{v["library"]}.json</span> ({v["generated_at"]}); un backtest
suelto se corre con <span class="mono">ai-trader backtest --validation cpcv</span>.</div>"""


def _fidelity_block(f):
    """Seccion 2.8: ¿se parece este mundo al real? Con la evidencia que lo responde."""
    if not f:
        return (
            "<h3>2.8 · Fidelidad contra el mercado real</h3>"
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
<h3>2.8 · Fidelidad contra el mercado real (validación externa)</h3>
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
(§5) están ahora en el orden correcto— pero sigue siendo optimista para un escenario de manía. Ésa es la
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
(§2.6), para que la reversión a la media y el momentum tengan algo real que capturar. Lo que sí es una
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


def _transfer_block(t):
    """Seccion 2.9: ¿ordena el mundo sintetico las estrategias como el real?

    Es la pregunta que §2.8 NO responde. La fidelidad mide parecido estadistico; esto mide
    si el orden se traslada, que es lo unico que el producto le pide al generador."""
    if not t:
        return (
            "<h3>2.9 · Transferencia de ranking</h3>"
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
        f"<td>{'reversión' if r['family'] == 'mean_reversion' else 'momentum'}</td>"
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
sintética sigue siendo válida para lo que sí está medido —fidelidad de los hechos estilizados (§2.8),
banco de estrés y regresión determinista— pero <b>no</b> como juez que ordena candidatas.</div>"""

    return f"""
<h3>2.9 · Transferencia de ranking: ¿ordena el sintético como el mercado?</h3>
<p>§2.8 responde si el mundo sintético <b>se parece</b> al mercado. No responde lo único que el producto
le pide de verdad: que si una configuración es mejor que otra ahí, <b>tienda a serlo también aquí</b>.
No son la misma pregunta —un generador puede clavar las colas y ordenar al revés, y ninguna métrica de
fidelidad lo detectaría—, así que se mide aparte y con su propio umbral.</p>
<p>El diseño persigue una sola cosa: que entre los dos lados <b>lo único que cambie sea el mundo del que
salen los precios</b>. Las mismas <b>{t["n_configs"]}</b> configuraciones (la rejilla del estudio de
pesos de §5.2b: hipercubo latino con semilla {t["study_seed"]} sobre las dos primitivas), el mismo
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
        "CALIBRATION": _calibration_block(f.get("calibration")),
        "FIDELITY": _fidelity_block(f.get("fidelity")),
        "TRANSFER": _transfer_block(f.get("transfer")),
        "VALIDATION": _validation_block(f.get("validation")),
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
<li><a href="#s1">Propósito, alcance y principios</a></li>
<li><a href="#s2">Los datos, primero</a></li>
<li><a href="#s3">Metodología de backtest</a></li>
<li><a href="#s4">Estrategias y espacio de observación</a></li>
<li><a href="#s5">Evaluación y scoring</a></li>
<li><a href="#s6">Tests y garantías</a></li>
<li><a href="#s7">Limitaciones conocidas y hoja de ruta</a></li>
<li><a href="#s8">Reproducibilidad</a></li>
<li><a href="#s9">Glosario</a></li>
</ol>
</div>

<h2 id="s1">1 · Propósito, alcance y principios</h2>
<p class="lead">AI-Trader es una herramienta de inversión cuantitativa (criptomonedas, renta variable
y mercados de predicción tipo Polymarket) que hoy opera en <b>paper trading</b>. Combina un motor
de riesgo, un motor de ejecución en papel, un backtest de alta fidelidad y un <b>generador de
mercados sintéticos deterministas</b> sobre el que se desarrollan y estresan estrategias. Ese
generador reproduce las propiedades estadísticas del mercado (§2.8), pero <b>no</b> su ordenación de
estrategias (§2.9): el ranking que decide sale del histórico real.</p>

<h3>Qué es y qué no es (todavía)</h3>
<ul>
<li><b>Es:</b> un banco de pruebas honesto para diseñar, evaluar y optimizar estrategias contra un
universo de %%NASSETS%% activos y una batería de regímenes macro sintéticos, ejecutando exactamente
el mismo camino que se usaría en vivo.</li>
<li><b>No es (aún):</b> no mueve dinero real; no hay todavía una estrategia optimizada de renta
variable o de mercados de predicción; y la "inteligencia" de optimización está en fase de
<b>optimización de caja negra</b> (CEM sobre parámetros), no de aprendizaje por refuerzo por
gradiente de política.</li>
</ul>

<h3>Principios rectores</h3>
<p>Todo el diseño se somete a cuatro principios. No son eslóganes: cada uno se traduce en decisiones
concretas de código verificadas por tests.</p>
<div class="why"><b>Honestidad estadística.</b> La unidad de evaluación es la <b>distribución</b>
sobre muchas muestras, nunca un único camino. <i>Por qué:</i> un solo path es ruido; publicar solo
la media esconde la cola. Se reporta la dispersión y se optimiza un estadístico consciente de la cola
(§5).</div>
<div class="why"><b>Anti-look-ahead por diseño.</b> En cada decisión solo se usan datos hasta el
cierre del día anterior. <i>Por qué:</i> el sesgo de futuro es la causa número uno de backtests
engañosos; aquí es imposible por construcción, no por disciplina (§3.2).</div>
<div class="why"><b>Determinismo total.</b> La terna (especificación, universo, semilla) produce las
mismas velas byte a byte; toda la cadena va sembrada. <i>Por qué:</i> reproducibilidad y
auditabilidad — cualquier resultado se puede regenerar y verificar.</div>
<div class="why"><b>No romper el contrato en vivo.</b> Lo que se optimiza se ejecuta por el mismo
camino estrategia → riesgo → ejecución. <i>Por qué:</i> elimina la divergencia backtest/live, que es
la fuente clásica de sorpresas en producción (§3.1).</div>

<h2 id="s2">2 · Los datos, primero</h2>
<p class="lead">La calidad de todo lo demás depende de la calidad del sustrato de datos. Por eso la
auditoría empieza aquí, no por el código.</p>

<h3>2.1 · Por qué datos sintéticos</h3>
<p>La herramienta se desarrolla y evalúa sobre mercados <b>sintéticos</b>, no sobre el histórico real.
Las razones, y sus contrapartidas, son explícitas:</p>
<ul>
<li><b>Independencia temporal por diseño.</b> Al no solaparse con ningún periodo histórico real, se
elimina el <i>data-snooping</i> sobre eventos ya conocidos: no se puede, ni sin querer, ajustar a la
crisis de 2008 o al COVID.</li>
<li><b>Control del régimen.</b> Podemos generar a voluntad crisis, rangos laterales, tendencias,
shocks de tipos o inviernos cripto, y en la proporción que necesitemos para estresar una estrategia.</li>
<li><b>Volumen ilimitado y hold-out honesto.</b> Generamos tantas muestras como haga falta para una
estadística robusta, y podemos reservar <b>arquetipos macro enteros</b> como validación (§5.3).</li>
</ul>
<div class="note"><b>Contrapartida reconocida.</b> Un mercado sintético solo es útil si se parece al
real en sus propiedades estadísticas (<i>stylized facts</i>). De ahí la validación interna de §2.7 y,
sobre todo, la validación <b>externa contra el histórico real</b> de §2.8 — que es la que dice en qué
se parece y en qué no.</div>
<div class="note"><b>Y una contrapartida que resultó ser mayor de lo previsto.</b> Parecerse no basta:
lo que el sistema le pide al generador es que <b>ordene</b> las estrategias como el mercado. Eso se
midió aparte (§2.9) y la respuesta fue que <b>no</b>. Por eso las tres razones de arriba siguen siendo
válidas para lo que el sintético hace bien —cubrir regímenes que la historia no dio, estresar, dar
hold-out de arquetipos y regresión determinista— pero <b>ya no</b> para decidir qué configuración es
mejor. El ranking que decide tiene que salir del histórico real.</div>

<h3>2.2 · El modelo de factores</h3>
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

<h3>2.3 · El universo</h3>
<p>%%NASSETS%% activos: %%NCRYPTO%% criptomonedas, %%NEQUITY%% de renta variable (índices y nombres
por sector) y %%NMACRO%% de macro/refugio (oro, bonos largos, dólar). La mezcla es deliberada para que
las correlaciones cruzadas tengan sentido: las financieras cargan tipos en positivo, la energía carga
materias primas, y los refugios cargan la renta variable en negativo. Cada activo declara además su
<b>liquidez</b> (volumen típico negociado en dólares al día), que abarca más de dos órdenes de magnitud
entre un índice amplio y un altcoin: es el eje que hace que ejecutar cueste distinto en cada mercado
(§3.4). El universo se guarda de forma autocontenida en el manifiesto de cada librería (precio inicial,
betas, volatilidad idiosincrática y liquidez), de modo que se pueda reconstruir el universo exacto y
regenerar datos idénticos aunque el código cambie.</p>
<div class="note"><b>Por qué el universo sintético y el operado no son el mismo (el caso MATIC/USDT).</b>
Binance deslistó ese par (el token migró a POL), así que está <b>retirado</b> del universo que se opera
en vivo (<span class="mono">config/default.toml</span>): allí un símbolo muerto falla en silencio en cada
ciclo. En el universo sintético se <b>mantiene a propósito</b>, y esa asimetría es la decisión, no un
descuido: aquí no cotiza nada contra un exchange — el símbolo es solo la etiqueta de un perfil de cargas
factoriales y las velas las genera el motor —, tiene contraparte real en la ventana con la que se mide
la fidelidad (2017-2026, y el informe publicado lo lista sin símbolos ausentes), y retirarlo cambiaría el
universo de %%NASSETS%% a %%NASSETS_MINUS_ONE%% activos, desincronizando toda la evidencia ya publicada
sobre %%NASSETS%%: la calibración de pesos y el propio estudio de fidelidad habría que re-medirlos.
<b>Límite declarado:</b> si el estudio de fidelidad se vuelve a correr sobre una ventana reciente,
MATIC/USDT no tendrá datos reales y aparecerá como ausente; ese es el momento de renombrarlo a POL/USDT
y republicar las mediciones, no antes.</div>

<h3>2.4 · El diseñador de escenarios (la única pieza con IA)</h3>
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
(<span class="mono">claude-opus-4-8</span>, familia Opus&nbsp;4.7+) <b>retiraron los parámetros de
muestreo</b> — enviar <span class="mono">temperature</span>, <span class="mono">top_p</span> o
<span class="mono">top_k</span> devuelve un error 400 —, así que ya no existe ninguna palanca con la que
forzar determinismo. Rehacer una librería con IA produce <i>siempre</i> una librería nueva.
<br><br><b>Mitigación: no se re-deriva, se guarda.</b> El <i>spec.json</i> es la salida cara e
insustituible y se persiste en disco; todo lo que va detrás (caminos, velas, backtests, métricas) es
determinista dado el spec y la semilla. La reproducibilidad del proyecto descansa en el artefacto
guardado, no en la llamada a la IA. <b>Límite declarado:</b> el manifiesto registra la <i>clase</i> del
diseñador, no el identificador del modelo; dos librerías generadas con modelos distintos son
indistinguibles por el manifiesto.</div>

<h3>2.5 · Del spec a las velas (el motor numérico)</h3>
<p>El motor convierte un spec en velas OHLCV multi-activo de forma <b>determinista</b> (vía
<span class="mono">numpy.default_rng(seed)</span>). Las fases se expanden a series diarias de deriva y
volatilidad; los shocks se suman como deriva puntual en su día. Las velas son válidas por construcción
(máximo ≥ cuerpo ≥ mínimo, precios positivos). La apertura de cada día parte del cierre anterior más un
hueco nocturno; las mechas se escalan por la volatilidad de ese día.</p>

<h3>2.6 · La microestructura estadística: por qué un generador ingenuo miente</h3>
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

<h3>2.7 · Validación: del mundo que miente (ai_v1) al mundo realista (ai_v2)</h3>
<p>La microestructura se asigna a los escenarios existentes mediante un <b>retrofit determinista</b>:
una función que deriva el carácter de cada fase de su propia semántica (una fase direccional → tendencia;
una fase plana → reversión; una fase de crisis, medida por la volatilidad de la renta variable → colas,
saltos y agrupamiento). Se reusa así el diseño caro de la IA sin volver a llamarla, y se produce una
nueva librería (<span class="mono">ai_v2</span>) conservando la anterior para comparar.</p>
<p>La librería actual, <span class="mono">ai_v2</span>, contiene <b>%%V2SCEN%% escenarios × %%V2PATHS%%
caminos = %%V2SAMPLES%% muestras</b>, con horizonte de %%V2HORIZON%% días. La tabla compara sus
<i>stylized facts</i> con los de la librería anterior (independiente e ingenua):</p>
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
(§5) corre sobre <span class="mono">ai_v2</span> por defecto — es una constante única
(<span class="mono">DEFAULT_LIBRARY_ID</span>) y hay un test que la fija. La librería anterior
(<span class="mono">ai_v1</span>) se conserva únicamente como referencia comparativa: para evaluar sobre
ella hay que pedirla <b>explícitamente</b>. Importa porque optimizar contra ruido independiente premia
justo los sesgos optimistas que el retrofit vino a corregir.</div>
<div class="note"><b>Esta tabla compara el mundo sintético consigo mismo.</b> Dice que ai_v2 dejó de
ser ruido independiente, no que sus colas o su agrupamiento tengan el tamaño de los del mercado. Esa es
otra pregunta, y se responde con datos reales en §2.8.</div>

%%FIDELITY%%

%%TRANSFER%%

<h2 id="s3">3 · Metodología de backtest</h2>

<h3>3.1 · Se conduce el sistema real, no una réplica</h3>
<p>El backtest <b>no reimplementa</b> la lógica de trading. Inyecta un reloj simulado, una fuente de
datos con anti-look-ahead, un modelo de mercado intrabar y un estado en memoria, y hace avanzar el
<b>mismo orquestador que operaría en vivo</b> un día a la vez.</p>
<div class="why"><b>Por qué.</b> Elimina de raíz toda una clase de divergencia backtest/live —la causa
número uno de sorpresas en producción—. Lo que se testea es, literalmente, lo que se ejecutaría.</div>

<h3>3.2 · Anti-look-ahead</h3>
<p>La fuente de datos solo devuelve barras <b>estrictamente anteriores</b> al día del reloj: la decisión
se toma con el cierre de ayer, nunca con el de hoy (que aún no ha cerrado desde el punto de vista de la
decisión). El bloque de observación cross-sectional (§4.3) usa exactamente el mismo corte temporal, para
que ninguna feature de mercado adelante información.</p>

<h3>3.3 · El modelo de mercado intrabar y su convención pesimista</h3>
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

<h3>3.4 · Costes de ejecución: spread, volatilidad e impacto</h3>
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

<h3>3.5 · Techo de capacidad y fills parciales</h3>
<p>Ninguna orden puede consumir más de una <b>fracción del volumen de la barra</b> (10% por defecto).
Lo que exceda ese techo <b>no se llena</b>: la orden queda <i>parcialmente llenada</i> y la posición se
abre por el tamaño realmente ejecutado. Si la barra no da ni para eso, la orden se <b>rechaza</b>.</p>
<p>Las <b>salidas</b> están exentas del techo, por asimetría deliberada: entrar es opcional —si no hay
liquidez, no entras— pero salir no lo es. Un cierre se llena entero y <b>paga el impacto de todo el
tamaño</b>, que con la ley de raíz cuadrada puede ser varias veces el de la entrada.</p>
<div class="why"><b>Por qué asimétrico.</b> Si un cierre se llenara parcialmente, la posición quedaría
cerrada en el estado del sistema pero pagada solo a medias: una contabilidad optimista escondida en un
detalle de ejecución. Se prefiere cobrar de más al salir que dejar el agujero.</div>

<h3>3.6 · El volumen como proxy de liquidez</h3>
<p>La columna <span class="mono">volume</span> de las velas, que antes era decorativa (ninguna pieza la
consultaba), es hoy el <b>eje de liquidez</b> del sistema: el generador la escala al volumen típico
negociado de cada activo, y el motor de ejecución la lee para el impacto y la capacidad. La liquidez se
estima con la <b>mediana</b> de las últimas 20 barras <b>ya cerradas</b> —nunca la de hoy— por dos
razones: el mismo corte anti-look-ahead que usa la estrategia, y porque el volumen se dispara los días
de movimiento fuerte, de modo que tomar ese pico como capacidad regalaría profundidad justo en los días
en que de verdad escasea.</p>
<div class="note"><b>Limitación.</b> Los baselines pasivos (§5.6) siguen pagando un coste plano de
referencia, no el modelo completo: solo hacen dos operaciones sobre los activos más líquidos, y darles
el coste barato endurece el listón que la estrategia debe superar, que es la dirección prudente del
error.</div>
<div class="note"><b>Nota de escala.</b> Con capitales de cinco cifras el término que domina es el
spread: una orden de unos cientos de dólares no mueve el precio de ningún activo del universo, y el
techo de capacidad no llega a morder. Los términos de impacto y capacidad son los que hacen que el
resultado <b>deje de escalar</b> cuando el capital crece, que es exactamente la pregunta que un
backtest con costes planos no puede responder.</div>

%%VALIDATION%%

<h3>3.8 · El motor de riesgo como puerta única</h3>
<p>Toda señal, venga de donde venga, pasa por el motor de riesgo antes de convertirse en orden. El riesgo
es <b>dueño</b> del stop y el take-profit (puede recortar lo que propone la estrategia), y aplica los
límites de exposición por símbolo y total, la confianza mínima por operación y el límite de pérdida
diaria. <i>Por qué puerta única:</i> ninguna orden puede esquivar el control de riesgo.</p>

<h2 id="s4">4 · Estrategias y espacio de observación</h2>

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

<h3>4.2 · El espacio de observación</h3>
<p>Se define de forma <b>explícita</b> y con <b>orden estable</b> lo que la política ve en cada momento de
decisión — el contrato de entrada para el aprendizaje futuro. El cierre de un solo activo no basta.
Contiene %%NOWN%% features del <b>mercado del propio activo</b> (retornos a varios horizontes, tendencia,
RSI/MACD, volatilidad ATR y realizada, posición en el canal de Donchian, drawdown) y %%NREGIME%%
features <b>cross-sectional / de régimen</b> (fuerza relativa al mercado, correlación móvil, amplitud del
mercado, volatilidad agregada), más un <i>one-hot</i> de clase de activo.</p>
<div class="why"><b>Por qué explícito y por qué cross-sectional.</b> Una política necesita situar al
activo en su contexto: no es lo mismo una sobreventa cuando todo el mercado cae (cuchillo) que cuando el
activo es un rezagado en un mercado sano. El bloque de régimen se inyecta como colaborador
<b>opcional</b>: si no está presente, la observación degrada a un solo activo y el contrato en vivo no se
rompe.</div>
<div class="note"><b>Cautela documentada.</b> El volumen sintético es un proxy simplista; las features
que dependen de él están marcadas como tales y deben interpretarse con cuidado.</div>

<h2 id="s5">5 · Evaluación y scoring</h2>

<h3>5.1 · La unidad de evaluación es la distribución</h3>
<p>Una estrategia no se juzga por un camino, sino por la <b>distribución de su resultado</b> sobre las
%%V2SAMPLES%% muestras (escenarios × caminos). Puntuar sobre un solo camino es medir ruido; se ha
verificado que un único path no es informativo.</p>
<p>Esa dispersión es <b>entre mundos</b>. Hay una segunda, ortogonal y hasta ahora invisible: la que hay
<b>dentro de un mundo</b>, entre tramos de su propia historia. La validación multiventana (§3.7) la mide,
y resulta no ser pequeña. Conviene no confundirlas: la primera pregunta "¿funciona en escenarios
distintos?", la segunda "¿funciona en momentos distintos del mismo escenario?".</p>
<div class="note"><b>Estado.</b> Lo que hoy alimenta al optimizador (§5.5) es la primera dispersión: cada
muestra sigue aportando <b>un</b> número, obtenido con el corte único. Componer las dos agregaciones —el
CVaR entre ventanas y el CVaR entre muestras— no es automático: encadenar dos veces la cola puede ser
excesivamente conservador, y esa decisión se tomará midiendo, no razonando (§7).</div>

<h3>5.2 · La métrica de cabecera: Sharpe penalizado</h3>
<p>La puntuación de una muestra es su <b>headline score fuera de muestra</b>:</p>
<p style="text-align:center"><span class="mono"><b>Sharpe<sub>OOS</sub> − λ·turnover − κ·maxDD</b></span>
&nbsp;&nbsp;(λ sobre la rotación diaria, κ sobre el drawdown en fracción; ambos <b>medidos</b>, no
supuestos: λ = %%LAMBDA%% y κ = %%KAPPA%% — ver §5.2b)</p>
<p>El <b>turnover</b> es una métrica nueva del motor: el <i>notional</i> rotado por día en unidades del
capital inicial, contando las dos patas de cada operación. Un turnover de 0,20 significa "cada día rota
el 20% de la cartera", así que mide <i>churn</i> de verdad — tamaño por frecuencia — y no solo el número
de operaciones.</p>
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
calibración de §5.2b puede dejarlo en cero sin reabrir ninguna de ellas.</div>
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

<h3>5.3 · El estadístico de recompensa: CVaR al 25%</h3>
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

<h3>5.4 · Hold-out de escenarios enteros</h3>
<p>Los escenarios se reparten en entrenamiento y validación de forma determinista, reservando
<b>arquetipos macro completos</b> como hold-out (aproximadamente 22 train / 8 validación).</p>
<div class="why"><b>Por qué escenarios enteros y no caminos sueltos.</b> Si un camino de validación
perteneciera a un escenario que también está en entrenamiento, sería <b>fuga de arquetipo</b>: el modelo
ya habría visto esa "física" y el hueco de sobreajuste quedaría enmascarado. Partir por escenario lo
impide.</div>

<h3>5.5 · El optimizador: Cross-Entropy Method</h3>
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

<h3>5.6 · Baselines: el listón que hay que superar para «aprobar»</h3>
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
<div class="why"><b>Por qué en la cola y no en la media.</b> El gate se juega con el mismo estadístico
con el que se optimiza. Una estrategia con mejor media pero peor cuartil malo <b>no</b> aprueba: si la
recompensa es la cola, el listón también tiene que serlo. El porcentaje de muestras en las que gana se
reporta como información, pero no decide.</div>
<div class="note"><b>Sin rival no hay aprobado.</b> Si un baseline no se puede construir (por ejemplo,
SPY en un universo solo cripto), no se sustituye ni se rellena: se declara como ausente. Y si no hay
ningún baseline disponible, el veredicto es <b>no aprueba</b> — un filtro que no se puede evaluar no es
un filtro superado.</div>

<h3>5.7 · Descuento del sobreajuste por múltiples pruebas</h3>
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

<h2 id="s6">6 · Tests y garantías</h2>
<p>La base cuenta con <b>%%NTESTS%% tests</b> automatizados y <i>linting</i> (ruff) sobre todo el
código. Más allá del número, importa <b>qué</b> se garantiza. Los tests se agrupan por invariante:</p>
<ul>
<li><b>Determinismo:</b> la misma semilla produce las mismas velas y las mismas curvas de equity.</li>
<li><b>Validez de las velas:</b> máximo ≥ cuerpo ≥ mínimo y precios positivos, en todos los regímenes,
incluidos los que activan saltos.</li>
<li><b>Anti-look-ahead:</b> tests dedicados verifican que la barra de hoy queda excluida y la de ayer
incluida, tanto en la fuente de datos como en el ensamblador de régimen. En la validación multiventana
se comprueba además a nivel de ejecución: borrar todas las barras <b>posteriores</b> a una ventana no
cambia ni un decimal de su curva de equity.</li>
<li><b>No hay fuga temporal entre folds:</b> para cada esquema (walk-forward anclado y rodante, CPCV,
corte único) se comprueba <b>día a día</b> —sobre fechas explícitas, no sobre los intervalos que
construyeron el fold— que ningún día aparece a la vez en entrenamiento y test, y que los huecos
respetan la purga y el embargo declarados. El auditor se ejercita también en negativo: folds con solape
y folds con purga insuficiente <b>deben</b> hacer fallar la ejecución antes de gastar cómputo. Se fija
además el invariante que delimita qué hace la purga: purgar cambia la ventana in-sample y <b>no</b>
la out-of-sample; y dos folds con el mismo test y entrenamientos distintos dan idéntico resultado OOS,
que es lo que legitima reutilizar tramos ya corridos entre folds.</li>
<li><b>Estructura de correlación:</b> activos que cargan el mismo factor correlacionan positivamente; un
shock mueve al activo que lo carga.</li>
<li><b>Microestructura ajustada en varianza:</b> añadir colas, agrupamiento o autocorrelación no cambia
la volatilidad total; la autocorrelación tiene el signo esperado (negativo → reversión, positivo →
tendencia).</li>
<li><b>Neutralidad de los valores por defecto:</b> con la microestructura desactivada, el generador
reproduce exactamente el mundo previo.</li>
<li><b>Puente al backtest:</b> el output del generador es directamente consumible por el motor de
backtest, y el ciclo completo (estrategia → riesgo → ejecución) funciona sobre datos sintéticos.</li>
<li><b>Contabilidad honesta:</b> el PnL es neto de comisiones; las salidas (stop/take-profit/tiempo)
pasan por el motor de ejecución para pagar costes reales.</li>
<li><b>Los costes discriminan:</b> un altcoin ilíquido paga más deslizamiento que BTC por el mismo
notional; el coste crece con el tamaño de forma cóncava (4× tamaño → 2× impacto) y con la volatilidad;
sin dato de liquidez no se inventa impacto. Una orden por encima del techo de capacidad se llena
<b>parcialmente</b> y las comisiones siguen al tamaño llenado, no al pedido.</li>
<li><b>La métrica de cabecera no tiene óptimo degenerado:</b> hay tests que fijan que una curva plana
puntúa 0 (no infinito), que el drawdown penaliza de forma aditiva y no en el denominador, y que rotar
más con la misma curva de equity puntúa peor.</li>
<li><b>El ranking compite por la cola:</b> una distribución con mejor media pero peor cuartil malo pierde,
y la varianza al alza no se castiga.</li>
<li><b>El filtro de baselines no se puede esquivar:</b> sin ningún baseline disponible el veredicto es
"no aprueba", y los baselines pagan las mismas comisiones que la estrategia.</li>
<li><b>El medidor de fidelidad mide lo que dice:</b> recupera el signo y el tamaño de una
autocorrelación conocida, distingue una serie con colas t-Student de una gaussiana y una con
volatilidad agrupada de una plana, devuelve NaN (no 0) cuando la serie es degenerada, y su
correlación de rangos da +1 y −1 en órdenes idénticos e invertidos. El estudio completo es
determinista y declara los símbolos sin contraparte real en vez de rellenarlos.</li>
<li><b>El descuento por múltiples pruebas discrimina:</b> un caso de habilidad genuina da PBO ≈ 0 y uno
de sobreajuste puro da PBO ≈ 1; probar más configuraciones sube el listón del DSR.</li>
</ul>

<h2 id="s7">7 · Limitaciones conocidas y hoja de ruta</h2>
<p>Declarar los huecos es parte de la honestidad de la herramienta. Estado actual:</p>
<table><thead><tr><th>Área</th><th>Estado</th><th>Por qué importa</th></tr></thead><tbody>
<tr><td>Realismo del generador (colas, agrupamiento, estructura serial, saltos)</td><td class="ok">Hecho</td><td>El sustrato dejó de mentir en la dirección optimista.</td></tr>
<tr><td>Cableado del sustrato realista en el scoring (<span class="mono">ai_v2</span> por defecto)</td><td class="ok">Hecho</td><td>Lo que se optimiza se mide contra el mundo realista, no contra ruido iid.</td></tr>
<tr><td>Métrica y ranking honestos (headline penalizado, CVaR@25%, baselines, DSR/PBO)</td><td class="ok">Hecho</td><td>El óptimo degenerado del Calmar desapareció, y ganar exige batir a no hacer nada.</td></tr>
<tr><td>Calibración con evidencia de los pesos λ (turnover) y κ (drawdown)</td><td class="ok">Hecho</td><td>Medidos por barrido en rejilla (§5.2b): la superficie es plana y λ no duplica los costes ya pagados.</td></tr>
<tr><td>Costes que muerden (slippage por símbolo/vol/tamaño; fills parciales)</td><td class="ok">Hecho</td><td>La fricción dejó de ser una constante: el coste distingue BTC de un altcoin ilíquido y el tamaño tiene techo (§3.4-3.6).</td></tr>
<tr><td>Validación temporal multiventana (walk-forward y CPCV con purga y embargo)</td><td class="ok">Hecho</td><td>Implementada, auditada contra fuga temporal y <b>medida</b>: el corte único no estaba sesgado al alza, pero degradaba el CVaR a un solo número y escondía una dispersión entre ventanas capaz de cambiar qué configuración gana (§3.7).</td></tr>
<tr><td>Validación sintético-vs-real (correlación de rangos con CCXT)</td><td class="ok">Hecho</td><td>Medida contra ocho años de histórico real: el nivel de riesgo y el orden de las correlaciones cruzadas se sostienen (§2.8).</td></tr>
<tr><td>Transferencia de ranking real-vs-sintético</td><td class="ok">Hecho</td><td><b>Medida, y con resultado negativo:</b> el mundo sintético pasa todos los umbrales de fidelidad y aun así <b>no ordena</b> las estrategias como el mercado (§2.9). Que esté "hecho" significa que la pregunta está respondida, no que la respuesta fuera la deseada — y esa respuesta es la que reordena el trabajo pendiente.</td></tr>
<tr><td>Limpieza de consistencia (universo, anualización, reproducibilidad del diseñador)</td><td class="ok">Hecho</td><td>La anualización distingue clase de activo (252 sesiones / 365 días naturales) y se reporta; la no-reproducibilidad del diseñador está documentada y su petición ya no envía parámetros de muestreo retirados; MATIC/USDT queda fuera del universo operado y dentro del sintético, con el motivo escrito.</td></tr>
</tbody></table>

<h3>7.1 · Lo que queda abierto, por criticidad</h3>
<p>El orden no es de gusto: responde a una asimetría de coste. Una estrategia añadida hoy se re-evalúa
gratis cuando el juez mejore; un juez malo contamina todo lo que puntúe mientras siga malo. Por eso el
sustrato y el juez van delante de la cosecha. El paper trading en vivo se lanza en paralelo porque es lo
único que compra tiempo de calendario, que no se puede comprimir después.</p>
<table><thead><tr><th>#</th><th>Trabajo pendiente</th><th>Por qué está donde está</th></tr></thead><tbody>
<tr><td>1</td><td>Mover el sustrato primario del ranking al histórico <b>real</b> (el sintético, a capa de estrés y veto)</td><td>Es la consecuencia directa de §2.9, y no admite interpretación porque el umbral estaba declarado antes de mirar. El generador pasa todos los umbrales de fidelidad y aun así no ordena; sin embargo el CEM, <span class="mono">sample_eval</span> y todo el <i>scoring</i> siguen puntuando exclusivamente sobre librerías sintéticas. Mientras eso no cambie, el sistema elige con un juez del que ya se sabe que no transfiere.</td></tr>
<tr><td>2</td><td>Una configuración que no opera no puede ganar el ranking</td><td>Lo destapó el propio estudio de transferencia: en el lado real, Spearman(recompensa, operaciones) = −0,84. Una curva plana puntúa headline <b>cero exacto</b>, y en un periodo donde casi todo lo que arriesga pierde, ese cero gana y hasta aprueba el <i>gate</i>. No es un error de cálculo —no perder es legítimo— pero sí una regla de selección degenerada para un sistema cuyo propósito es operar, y contamina cualquier ranking que se mida encima.</td></tr>
<tr><td>3</td><td>Cablear el CPCV en el optimizador, en dos etapas</td><td>El CEM sigue puntuando con el corte único que el propio estudio desacredita. No está sesgado, está arbitrario — y arbitrario es letal para un optimizador que escala el relieve del paisaje. Criba barata + finalistas con CPCV completo, agregando por <i>pooling</i> de todos los folds, no CVaR de CVaR.</td></tr>
<tr><td>4</td><td>Paper trading corriendo en vivo, con diario de ciclos</td><td>Cero desarrollo pendiente y en paralelo a todo lo demás: la divergencia live-vs-backtest necesita meses para ser medible, así que cada semana sin correr es una semana perdida al final.</td></tr>
<tr><td>5–9</td><td>Rigor del juez: re-correr la validación con el ensemble completo; alinear los bloques del PBO con fronteras de escenario; reportar el recuento de muestras fallidas junto al reward; declarar que el DSR asume intentos independientes; arreglar la <i>ordenación</i> de colas y agrupamiento entre activos</td><td>Cinco arreglos baratos sobre cifras que se publican como garantía. El estudio de validación corrió con un camino por escenario; los bloques del PBO pueden partir un escenario en dos (la misma fuga de arquetipo que el hold-out evita); la penalización por fallo domina la cola del CVaR; el CEM no produce los intentos independientes que el DSR supone; y el generador acierta la magnitud de las colas pero no <i>qué</i> activo las tiene (§2.8), que es un candidato natural a explicar por qué tampoco acierta qué estrategia gana.</td></tr>
<tr><td>10</td><td>Optimización CEM completa sobre el sustrato corregido</td><td>Bloqueada a propósito, y ahora con más motivo: correrla antes de arreglar el sustrato del ranking es gastar cómputo caro en un ganador elegido por un juez que no transfiere.</td></tr>
<tr><td>11</td><td>Nuevas estrategias cripto</td><td><b>No priorizada, deliberadamente.</b> Solo 6 de 32 filas aprueban el gate bajo CPCV, y eso admite dos lecturas — estrategias flojas o juez ruidoso. Hasta saber cuál, cada candidato nuevo multiplica el problema de múltiples pruebas y sube el listón del DSR de todos los demás.</td></tr>
<tr><td>12–13</td><td>Re-medición de λ y κ con los costes nuevos; modelo de IA anotado en el manifiesto</td><td>Impacto bajo con evidencia detrás: la superficie de pesos ya salió plana sobre 480 backtests y el modelo de costes nuevo solo refuerza la conclusión.</td></tr>
<tr><td>14–15</td><td>Renta variable y mercados de predicción</td><td><b>Segundo plano explícito.</b> Toda la evidencia empírica del repo es cripto; la pata de renta variable del generador no tiene ni un dato real detrás, no existe estrategia de acciones, y el universo de megacaps arrastra sesgo de supervivencia estructural. Polymarket no tiene histórico que backtestear: el paper trading en vivo es lo que empezará a generarlo.</td></tr>
</tbody></table>
<div class="note"><b>Foco: cripto.</b> Los <i>stocks</i> no salen del universo <b>sintético</b> — GLD, TLT y UUP
son lo que hace que los escenarios de tipos y de dólar signifiquen algo para cripto vía los factores
compartidos. La regla es: se genera con los 35 activos, se puntúa y se opera solo cripto.</div>
<p class="tag">El dashboard de la herramienta mantiene el detalle accionable de cada mejora, con su
evidencia medida y un prompt reproducible para abordarla.</p>

<h2 id="s8">8 · Reproducibilidad</h2>
<p>Toda la cadena es determinista y regenerable. Notas operativas:</p>
<ul>
<li><b>Generar una librería sintética</b> (llama a la IA una vez): diseña los escenarios y sintetiza el
ensemble Monte Carlo, guardando el manifiesto autocontenido y los <i>spec.json</i>.</li>
<li><b>Ampliar o reconstruir</b> caminos sin IA: "resynthesize" regenera a partir de los
<i>spec.json</i> guardados con las mismas semillas.</li>
<li><b>Derivar la librería realista</b> (<span class="mono">ai_v2</span>): el retrofit determinista
enriquece los escenarios con microestructura y regenera, conservando la librería anterior.</li>
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

<h2 id="s9">9 · Glosario</h2>
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
<li><b>Baseline:</b> alternativa pasiva de referencia (comprar y mantener) que una estrategia debe batir
para justificar su existencia.</li>
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
</ul>
<p class="meta">Fin del documento · AI-Trader · commit <span class="mono">%%COMMIT%%</span>.</p>
"""
