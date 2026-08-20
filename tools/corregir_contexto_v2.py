#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrige la sección 3 (contexto global), la 10 (sentimiento) y la 6 (derivados)
de los reportes que heredaron el contexto v1 erróneo, re-renderiza el HTML y
corrige P05-P08 en las respuestas. Ver _log.txt para la trazabilidad."""
import json, os, subprocess, sys

BASE = "/root/ai-trader/data/signals_raw/ai_reports/2026-08-20"
DAT = os.path.join(BASE, "_datos")
REND = "/root/ai-trader/tools/render_reporte.py"
VAL = "/root/ai-trader/tools/validar_respuestas.py"

# 24h propio del activo (rango entre fuentes) -> para la tabla activo vs mercado
MOV = {
    "DOT":  ("+3,3% a +4,9%",  "Rezagado"),
    "LINK": ("+2,8% a +8,9%",  "En línea / rezagado"),
    "LTC":  ("+3,5% a +5,7%",  "Rezagado"),
    "APT":  ("+2,4% a +6,6%",  "Rezagado"),
    "ARB":  ("+15,7% a +16,3%","Muy por delante"),
    "OP":   ("+11,1% a +11,8%","Por delante"),
    "INJ":  ("+14,2% a +14,5%","Muy por delante"),
    "FIL":  ("+7,8% a +8,7%",  "En línea"),
    "ETC":  ("−0,4% a +0,5%",  "Muy por detrás"),
}
# Solo arreglo de sentimiento (su sección 3 ya era correcta)
SOLO_SENTIMIENTO = {"SOL", "BNB", "XRP"}

KPIS_S3 = [
    {"label": "BTC 24h",          "valor": "+7% a +11,6%",  "nota": "$69.300–71.900 · tocó $70.000, máximo desde junio"},
    {"label": "ETH 24h",          "valor": "+17,6% a +18,8%","nota": "$2.252–2.279 · gran ganador de la sesión"},
    {"label": "Cap. total cripto","valor": "$2,37–2,49 T",  "nota": "+7,5% a +9,4% en 24h"},
    {"label": "Dominancia BTC",   "valor": "56,2%–59,2%",   "nota": "dominancia ETH ≈10,9%"},
    {"label": "Fear & Greed",     "valor": "59–62 (Codicia)","nota": "41–46 ayer · 34 hace una semana"},
    {"label": "Altcoin Season",   "valor": "36/100",        "nota": "bajó desde 43: el dinero rota hacia BTC"},
]

NOTA_CORR = ("**Nota de corrección de datos (v2).** Una versión previa de este reporte fijaba el contexto "
             "global en la mañana del 19-ago (BTC $64.339, Fear & Greed 41–46) y marcaba erróneamente como "
             "«caché obsoleta» los datos de agregador que mostraban BTC cerca de $70.000. Esa lectura era "
             "incorrecta: los agregadores mostraban datos en vivo y el movimiento se produjo en la sesión "
             "americana del miércoles 19-ago. Esta sección refleja el contexto verificado.")

PAR_MOV = ("El mercado rompió al alza en la sesión americana del **miércoles 19-ago**: CoinDesk lo resume como "
           "«*Bitcoin briefly hits $70,000 for the first time since June — the largest crypto asset rose more "
           "than 7% on Wednesday after several catalysts sent crypto-related assets higher*». El **catalizador "
           "principal fue macro**: el Tesoro de EE. UU. **duplicó el tamaño de sus recompras de deuda del tramo "
           "largo** ($2 B → $4 B), lo que reactivó el apetito por riesgo en todos los activos. Se sumaron la "
           "propuesta de la **SEC «Regulation Crypto Assets»**, el trabajo de la **CFTC** para traer "
           "**Hyperliquid** a EE. UU. y el impulso renovado de la **CLARITY Act** en el Congreso.")

PAR_SQUEEZE = ("La mecánica del movimiento fue un **short squeeze de libro**: ≈**$3,0 B liquidados en 24 h con "
               "~92% de posiciones cortas**, el mayor desde 2021. Eso importa para la interpretación: buena "
               "parte del avance es **cobertura forzada de cortos**, no necesariamente demanda nueva sostenida.")

LISTA_S3 = [
    {"tono": "alcista", "texto": "**Risk-on generalizado** con expansión de volumen global ($55–117 B según fuente) y entradas récord en ETF spot de BTC (≈+$517 M) y ETH (≈+$189 M)."},
    {"tono": "alcista", "texto": "**Fear & Greed salta a 59–62 (Codicia)** desde 41–46 el día previo y 34 hace una semana: una mejora de ~25–28 puntos en siete sesiones."},
    {"tono": "bajista", "texto": "**Altcoin Season Index cae de 43 a 36**: el capital rota hacia BTC, no hacia las altcoins. La dominancia de BTC sube a la banda 56,2–59,2%."},
    {"tono": "bajista", "texto": "**El bear market estructural sigue intacto**: BTC ≈−43% interanual y ≈−44% desde el ATH de $126.198,07 (6-oct-2025); ETH ≈−57% interanual. Este movimiento no revierte la tendencia mayor."},
    {"tono": "aviso",   "texto": "**Riesgo de reversión**: actas del FOMC con tres disidencias a favor de subir tipos (Logan, Hammack, Kashkari), Jackson Hole a finales de mes con Kevin Warsh, y el conflicto de Irán / Oriente Medio sin resolución."},
]

NOTA_METODO = ("**Método.** El rango ancho de BTC ($69,3 k–$71,9 k) refleja cortes horarios distintos dentro de "
               "una sesión muy volátil, no un error de fuente. Todos los datos de 24 h dependen de la hora de "
               "corte indicada en la cabecera.")

KPIS_S10 = {
    "fear": {"label": "Fear & Greed hoy", "valor": "59–62 (Codicia)", "nota": "desde 41–46 el día previo"},
    "sem":  {"label": "F&G hace 1 semana", "valor": "34 (Miedo)", "nota": "≈13–15 ago"},
    "var":  {"label": "Variación semanal F&G", "valor": "+25 a +28 puntos", "nota": "mejora muy fuerte"},
    "asi":  {"label": "Altcoin Season Index", "valor": "36/100", "nota": "bajó desde 43: sin rotación a altcoins"},
}

PAR_S10 = ("El **Fear & Greed global salta a 59–62, zona de CODICIA**, desde 41–46 el día anterior y **34 hace "
           "una semana**: una mejora de **~25–28 puntos** en siete sesiones, uno de los giros de sentimiento más "
           "bruscos del trimestre. En paralelo, el **Altcoin Season Index cae de 43 a 36**, señal de que el "
           "capital se concentra en BTC y no rota hacia el resto del mercado.")

OBSOLETO = ["41–46", "41-46", "frontera Miedo/Neutral", "Miedo/Neutral", "+7 a +12 puntos",
            "43/100", "mejora de entre 7 y 12", "todavía sin codicia", "sin codicia"]


def limpia_bloques(bloques):
    """Elimina párrafos/notas que contienen cifras del contexto v1 ya corregidas."""
    out = []
    for b in bloques:
        if b.get("tipo") in ("parrafo", "nota") and any(o in b.get("texto", "") for o in OBSOLETO):
            continue
        out.append(b)
    return out


def patch_s3(sec, tk):
    mov, veredicto = MOV[tk]
    sec["kpis"] = list(KPIS_S3)
    sec["bloques"] = [
        {"tipo": "nota", "texto": NOTA_CORR},
        {"tipo": "parrafo", "texto": PAR_MOV},
        {"tipo": "parrafo", "texto": PAR_SQUEEZE},
        {"tipo": "tabla", "titulo": "%s frente al mercado (24 h)" % tk,
         "headers": ["Referencia", "Variación 24h", "Lectura"],
         "rows": [
             ["BTC", "+7% a +11,6%", "Motor del movimiento"],
             ["ETH", "+17,6% a +18,8%", "Máxima beta entre los grandes"],
             ["Cap. total cripto", "+7,5% a +9,4%", "Subida generalizada"],
             ["**%s**" % tk, "**%s**" % mov, "**%s**" % veredicto],
         ]},
        {"tipo": "lista", "items": LISTA_S3},
        {"tipo": "nota", "texto": NOTA_METODO},
    ]


def patch_s10(sec):
    kp = []
    vistos = set()
    for k in sec.get("kpis", []):
        lab = k.get("label", "").lower()
        if "fear" in lab or "f&g" in lab:
            if "semana" in lab and "sem" not in vistos:
                kp.append(KPIS_S10["sem"]); vistos.add("sem")
            elif "variac" in lab or "cambio" in lab:
                if "var" not in vistos:
                    kp.append(KPIS_S10["var"]); vistos.add("var")
            elif "fear" not in vistos:
                kp.append(KPIS_S10["fear"]); vistos.add("fear")
        elif "altcoin" in lab:
            if "asi" not in vistos:
                kp.append(KPIS_S10["asi"]); vistos.add("asi")
        else:
            kp.append(k)
    for clave in ("fear", "sem", "var", "asi"):
        if clave not in vistos:
            kp.insert(0, KPIS_S10[clave])
    sec["kpis"] = kp
    sec["bloques"] = [{"tipo": "parrafo", "texto": PAR_S10}] + limpia_bloques(sec.get("bloques", []))


def patch_s6(sec):
    txt = json.dumps(sec, ensure_ascii=False)
    if "92%" in txt or "$3,0 B" in txt or "3,02" in txt or "3,07" in txt:
        return
    sec["bloques"] = [{"tipo": "nota", "texto":
        "**Contexto agregado de derivados (19–20 ago).** El mercado registró ≈**$3,0 B de liquidaciones en 24 h "
        "con ~92% de posiciones cortas**, el mayor short squeeze desde 2021. Las lecturas de interés abierto "
        "(≈$120 B) y volumen de derivados (≈$151 B) corresponden al 19-ago, previas al pico."}] + sec.get("bloques", [])


RESP = {
    "P05": ("subida_fuerte", 2,
            "Sección 3 · Contexto global: BTC +7% a +11,6% en 24h ($69.300–71.900), tocando $70.000 por primera vez desde junio"),
    "P06": ("codicia", 1,
            "Sección 3 y 10: Fear & Greed en 59–62, zona de Codicia"),
    "P07": ("mejora_fuerte", 2,
            "Sección 10: F&G pasa de 34 (Miedo) hace una semana a 59–62, mejora de ~25–28 puntos"),
    "P08": ("risk_on_fuerte", 2,
            "Sección 3: risk-on generalizado, cap. total +7,5% a +9,4%, subidas en todo el mercado tras las recompras del Tesoro de EE. UU."),
}


def main():
    tickers = sorted(set(MOV) | SOLO_SENTIMIENTO)
    for tk in tickers:
        dpath = os.path.join(DAT, "datos_%s.json" % tk)
        d = json.load(open(dpath, encoding="utf-8"))
        secs = {s["n"]: s for s in d["secciones"]}
        if tk in MOV:
            patch_s3(secs[3], tk)
            patch_s6(secs[6])
        patch_s10(secs[10])
        json.dump(d, open(dpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        out = os.path.join(BASE, "reporte_%s.html" % tk)
        subprocess.run([sys.executable, REND, dpath, out], check=True, capture_output=True)

        rpath = os.path.join(BASE, "respuestas_%s.json" % tk)
        r = json.load(open(rpath, encoding="utf-8"))
        claves = RESP if tk in MOV else {k: RESP[k] for k in ("P06", "P07")}
        for qid, (op, val, ev) in claves.items():
            r["respuestas"][qid] = {"id_opcion": op, "valor": val, "evidencia": ev}
        json.dump(r, open(rpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        subprocess.run([sys.executable, VAL, rpath, "--fix"], check=True, capture_output=True)
        res = subprocess.run([sys.executable, VAL, rpath], capture_output=True, text=True)
        print("%-5s %s" % (tk, res.stdout.strip() or res.stderr.strip()))


if __name__ == "__main__":
    main()
