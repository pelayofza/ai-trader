#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renderiza reporte_{TICKER}.html a partir de datos_{TICKER}.json.
Uso: python3 render_reporte.py datos_DOGE.json /ruta/salida/reporte_DOGE.html
HTML autocontenido, CSS inline, sin dependencias externas."""
import json, sys, html, os

TONOS = {
    "alcista":  ("#0f3d2e", "#3ddc97", "Alcista"),
    "bajista":  ("#3d1620", "#ff6b81", "Bajista"),
    "neutral":  ("#2a2f3a", "#c9d1d9", "Neutral"),
    "aviso":    ("#3d3216", "#ffc857", "Atención"),
}

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#c9d1d9;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 72px}
header.rep{border-bottom:1px solid #21262d;padding-bottom:20px;margin-bottom:8px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#7d8590;margin:0 0 6px}
h1{font-size:30px;line-height:1.2;margin:0 0 4px;color:#f0f6fc;font-weight:650}
h1 .tk{color:#58a6ff}
.sub{color:#7d8590;font-size:13px;margin:0}
.strip{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 4px}
.sig{border:1px solid;border-radius:8px;padding:8px 12px;min-width:132px;flex:1 1 auto}
.sig .l{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#7d8590;display:block;margin-bottom:2px}
.sig .v{font-size:14px;font-weight:650}
.lede{background:#161b22;border:1px solid #21262d;border-left:3px solid #58a6ff;
 border-radius:6px;padding:14px 16px;margin:18px 0 0;color:#adbac7;font-size:14px}
section{margin-top:34px;scroll-margin-top:16px}
h2{font-size:19px;color:#f0f6fc;margin:0 0 14px;padding-bottom:8px;border-bottom:1px solid #21262d;font-weight:600}
h2 .num{display:inline-block;min-width:26px;height:26px;line-height:26px;text-align:center;
 background:#1f6feb;color:#fff;border-radius:6px;font-size:12px;margin-right:10px;vertical-align:2px;font-weight:700}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:11px 13px}
.kpi .k{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:#7d8590;display:block}
.kpi .val{font-size:17px;color:#f0f6fc;font-weight:650;display:block;margin:2px 0 1px}
.kpi .n{font-size:11px;color:#7d8590;display:block}
p.par{margin:0 0 12px}
table{width:100%;border-collapse:collapse;margin:6px 0 16px;font-size:13.5px}
th{background:#161b22;color:#adbac7;text-align:left;font-weight:600;font-size:11px;
 letter-spacing:.06em;text-transform:uppercase;padding:9px 11px;border-bottom:1px solid #30363d}
td{padding:9px 11px;border-bottom:1px solid #21262d;vertical-align:top}
tr:last-child td{border-bottom:none}
ul.tl{list-style:none;padding:0;margin:0 0 14px}
ul.tl li{position:relative;padding:8px 12px 8px 14px;margin-bottom:6px;background:#161b22;
 border:1px solid #21262d;border-left-width:3px;border-radius:5px;font-size:14px}
.tag{display:inline-block;font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;
 padding:2px 7px;border-radius:4px;margin-right:8px;font-weight:700;vertical-align:1px}
.nota{background:#161b22;border:1px dashed #30363d;border-radius:6px;padding:11px 13px;
 font-size:13px;color:#8b949e;margin:0 0 14px}
.src{font-size:12.5px;color:#7d8590}
.src a{color:#58a6ff;text-decoration:none;word-break:break-word}
.src li{margin-bottom:5px}
footer.rep{margin-top:44px;padding-top:16px;border-top:1px solid #21262d;font-size:11.5px;color:#6e7681}
.disc{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 14px;
 font-size:12px;color:#8b949e;margin-top:18px}
@media(max-width:600px){.wrap{padding:20px 14px 48px}h1{font-size:23px}.sig{min-width:112px}}
"""

def esc(t):
    return html.escape(str(t), quote=False)

def rich(t):
    """Escapa y permite **negrita** y `código`."""
    s = esc(t)
    out, bold = [], False
    i = 0
    while i < len(s):
        if s.startswith("**", i):
            out.append("</strong>" if bold else "<strong style='color:#f0f6fc'>")
            bold = not bold; i += 2
        elif s[i] == "`":
            j = s.find("`", i + 1)
            if j == -1:
                out.append(s[i]); i += 1
            else:
                out.append("<code style=\"background:#21262d;padding:1px 5px;border-radius:4px;"
                           "font-size:12.5px\">" + s[i+1:j] + "</code>")
                i = j + 1
        else:
            out.append(s[i]); i += 1
    if bold:
        out.append("</strong>")
    return "".join(out)

def bloque(b):
    t = b.get("tipo", "parrafo")
    if t == "parrafo":
        return "<p class='par'>%s</p>" % rich(b.get("texto", ""))
    if t == "nota":
        return "<div class='nota'>%s</div>" % rich(b.get("texto", ""))
    if t == "tabla":
        h = "".join("<th>%s</th>" % esc(x) for x in b.get("headers", []))
        rows = ""
        for r in b.get("rows", []):
            rows += "<tr>" + "".join("<td>%s</td>" % rich(c) for c in r) + "</tr>"
        cap = ""
        if b.get("titulo"):
            cap = "<p class='par' style='font-size:13px;color:#adbac7;margin-bottom:4px'><strong style='color:#f0f6fc'>%s</strong></p>" % esc(b["titulo"])
        return cap + "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (h, rows)
    if t == "lista":
        li = ""
        for it in b.get("items", []):
            if isinstance(it, str):
                it = {"tono": "neutral", "texto": it}
            bg, fg, lbl = TONOS.get(it.get("tono", "neutral"), TONOS["neutral"])
            tag = "<span class='tag' style='background:%s;color:%s'>%s</span>" % (bg, fg, esc(it.get("etiqueta", lbl)))
            li += "<li style='border-left-color:%s'>%s%s</li>" % (fg, tag, rich(it.get("texto", "")))
        return "<ul class='tl'>%s</ul>" % li
    return ""

def render(d):
    tk = esc(d.get("ticker", "?"))
    parts = ["<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Reporte diario %s · %s</title>" % (tk, esc(d.get("fecha", ""))),
             "<style>%s</style></head><body><div class='wrap'>" % CSS]
    parts.append("<header class='rep'>")
    parts.append("<p class='eyebrow'>Reporte diario de señales · %s</p>" % esc(d.get("fecha", "")))
    parts.append("<h1>%s <span class='tk'>%s</span></h1>" % (esc(d.get("activo", "")), esc(d.get("par", tk))))
    parts.append("<p class='sub'>Generado el %s · corte de datos %s · documento descriptivo, no es asesoramiento de inversión</p>"
                 % (esc(d.get("fecha", "")), esc(d.get("hora_corte", "—"))))
    if d.get("senales"):
        parts.append("<div class='strip'>")
        for s in d["senales"]:
            bg, fg, _ = TONOS.get(s.get("tono", "neutral"), TONOS["neutral"])
            parts.append("<div class='sig' style='background:%s;border-color:%s'>"
                         "<span class='l'>%s</span><span class='v' style='color:%s'>%s</span></div>"
                         % (bg, fg, esc(s.get("label", "")), fg, esc(s.get("valor", ""))))
        parts.append("</div>")
    if d.get("resumen_cabecera"):
        parts.append("<div class='lede'>%s</div>" % rich(d["resumen_cabecera"]))
    parts.append("</header>")

    for sec in d.get("secciones", []):
        parts.append("<section id='s%s'>" % esc(sec.get("n", "")))
        parts.append("<h2><span class='num'>%s</span>%s</h2>" % (esc(sec.get("n", "")), esc(sec.get("titulo", ""))))
        if sec.get("kpis"):
            parts.append("<div class='kpis'>")
            for k in sec["kpis"]:
                parts.append("<div class='kpi'><span class='k'>%s</span><span class='val'>%s</span>%s</div>"
                             % (esc(k.get("label", "")), esc(k.get("valor", "—")),
                                ("<span class='n'>%s</span>" % esc(k["nota"])) if k.get("nota") else ""))
            parts.append("</div>")
        for b in sec.get("bloques", []):
            parts.append(bloque(b))
        parts.append("</section>")

    if d.get("fuentes"):
        parts.append("<section id='fuentes'><h2><span class='num'>F</span>Fuentes</h2><ul class='src'>")
        for f in d["fuentes"]:
            if isinstance(f, str):
                f = {"titulo": f, "url": ""}
            if f.get("url"):
                parts.append("<li><a href='%s'>%s</a></li>" % (esc(f["url"]), esc(f.get("titulo") or f["url"])))
            else:
                parts.append("<li>%s</li>" % esc(f.get("titulo", "")))
        parts.append("</ul></section>")

    parts.append("<div class='disc'>Este documento es descriptivo y de uso interno. Recopila datos "
                 "públicos de mercado y no constituye asesoramiento financiero ni una recomendación "
                 "de compra o venta. Los datos de 24 h dependen de la hora de corte indicada en la cabecera "
                 "y pueden diferir entre agregadores; cuando hay discrepancias relevantes se reportan como rango.</div>")
    parts.append("<footer class='rep'>cuestionario_cripto_v1 · pipeline ai-trader · reporte_%s.html · %s</footer>"
                 % (tk, esc(d.get("fecha", ""))))
    parts.append("</div></body></html>")
    return "".join(parts)

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)
    req = [s.get("n") for s in data.get("secciones", [])]
    if req != list(range(1, 14)):
        sys.exit("ERROR: se esperaban las secciones 1..13 en orden, se encontró: %s" % req)
    d = os.path.dirname(os.path.abspath(dst))
    os.makedirs(d, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(render(data))
    print("OK %s (%d bytes)" % (dst, os.path.getsize(dst)))
