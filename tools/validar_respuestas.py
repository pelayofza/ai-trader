#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Valida (y opcionalmente corrige la puntuación agregada de) respuestas_{TICKER}.json.
Uso: python3 validar_respuestas.py respuestas_DOGE.json [--fix]"""
import json, sys, os

CUEST = "/root/ai-trader/config/cuestionario_cripto_v1.json"

def interpretar(media):
    if media is None:      return "sin_datos"
    if media <= -1.0:      return "muy_bajista"
    if media <= -0.35:     return "bajista"
    if media <  0.35:      return "neutral"
    if media <  1.0:       return "alcista"
    return "muy_alcista"

def main():
    path = sys.argv[1]
    fix = "--fix" in sys.argv
    q = json.load(open(CUEST, encoding="utf-8"))
    banco = {p["id"]: {o["id_opcion"]: o["valor"] for o in p["opciones"]} for p in q["preguntas"]}
    orden = [p["id"] for p in q["preguntas"]]
    d = json.load(open(path, encoding="utf-8"))
    err = []

    for campo in ("schema_version", "cuestionario", "activo", "ticker", "fecha", "reporte_fuente",
                  "respuestas", "puntuacion_agregada"):
        if campo not in d:
            err.append("falta el campo raíz '%s'" % campo)
    if d.get("cuestionario") != "cuestionario_cripto_v1":
        err.append("campo 'cuestionario' incorrecto: %r" % d.get("cuestionario"))

    r = d.get("respuestas", {})
    if len(r) != 30:
        err.append("se esperaban 30 respuestas, hay %d" % len(r))
    if list(r.keys()) != orden:
        faltan = [x for x in orden if x not in r]
        sobran = [x for x in r if x not in banco]
        if faltan: err.append("faltan preguntas: %s" % faltan)
        if sobran: err.append("ids desconocidos: %s" % sobran)

    suma, resp, sind = 0, 0, 0
    for qid in orden:
        if qid not in r:
            continue
        a = r[qid]
        for c in ("id_opcion", "valor", "evidencia"):
            if c not in a:
                err.append("%s: falta '%s'" % (qid, c))
        op = a.get("id_opcion")
        if op not in banco[qid]:
            err.append("%s: id_opcion '%s' no existe (válidas: %s)" % (qid, op, list(banco[qid])))
            continue
        esperado = banco[qid][op]
        if a.get("valor") != esperado:
            err.append("%s: valor %r no coincide con el cuestionario (%r)" % (qid, a.get("valor"), esperado))
        ev = (a.get("evidencia") or "").strip()
        if len(ev) < 8:
            err.append("%s: 'evidencia' vacía o demasiado corta" % qid)
        if esperado is None:
            sind += 1
        else:
            suma += esperado; resp += 1

    media = round(suma / resp, 4) if resp else None
    calc = {"suma_valores": suma, "preguntas_respondidas": resp,
            "preguntas_sin_datos": sind, "media": media, "interpretacion": interpretar(media)}
    pa = d.get("puntuacion_agregada", {})
    if fix:
        d["puntuacion_agregada"] = calc
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    else:
        for k, v in calc.items():
            got = pa.get(k)
            if k == "media" and isinstance(got, (int, float)) and v is not None:
                if abs(got - v) > 0.005:
                    err.append("puntuacion_agregada.media=%s, calculada=%s" % (got, v))
            elif k == "interpretacion":
                if got not in (v, None) and got != v:
                    err.append("puntuacion_agregada.interpretacion='%s', calculada='%s'" % (got, v))
            elif got != v:
                err.append("puntuacion_agregada.%s=%s, calculada=%s" % (k, got, v))

    name = os.path.basename(path)
    if err:
        print("FALLO %s (%d problemas)" % (name, len(err)))
        for e in err:
            print("   - " + e)
        sys.exit(1)
    print("OK %s | 30 respuestas | suma=%s respondidas=%s sin_datos=%s media=%s (%s)"
          % (name, calc["suma_valores"], calc["preguntas_respondidas"],
             calc["preguntas_sin_datos"], calc["media"], calc["interpretacion"]))

if __name__ == "__main__":
    main()
