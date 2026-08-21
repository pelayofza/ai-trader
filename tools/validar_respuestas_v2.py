#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valida (y opcionalmente corrige la puntuacion agregada de) respuestas_{TICKER}.json v2.

    python3 validar_respuestas_v2.py respuestas_DOGE.json [--fix] [--cuestionario RUTA]

PROBADO el 2026-08-21, y hasta entonces no lo estaba: se escribio en una sesion sin Python.
Se corrio sobre los 24 ficheros reales del dia (24 OK, 0 fallos) y, lo que de verdad importa,
sobre tres copias CORROMPIDAS a proposito, para comprobar que no se limita a imprimir OK:

  - una `fuente_ts` posterior a la hora de corte  -> la detecta como look-ahead
  - un `sin_datos` con valor 0 y disponible 1     -> detecta las dos cosas (era el bug de v1)
  - `n_fuentes_contrastadas` = 1 en el ancla      -> lo rechaza

No lo importa nadie: anadirlo no puede romper el paquete.

Diferencias con validar_respuestas.py (v1), que se conserva intacto para los ficheros v1:
  - 37 preguntas P01..P37 en 'respuestas', mas P30 en el bloque 'benchmark_llm'.
  - Solo suman las preguntas con sumable=true en el cuestionario (29 de 37).
  - 'no_aplica' vale null, no 0, y se distingue de 'sin_datos' por el campo 'estado'.
  - Comprueba el bloque 'ancla' (precio numerico + timestamp + 2 fuentes contrastadas).
  - Comprueba la regla point-in-time: ninguna 'fuente_ts' posterior a 'hora_corte_utc'.
  - Comprueba 'valor_crudo' en las preguntas con metrica_cruda medidas.
  - La ruta del cuestionario se resuelve relativa al repo, sin rutas absolutas embebidas.
"""
import json
import sys
import os
from datetime import datetime, timezone

AQUI = os.path.dirname(os.path.abspath(__file__))
CUEST_POR_DEFECTO = os.path.join(AQUI, "..", "config", "cuestionario_cripto_v2.json")
ASSETS_POR_DEFECTO = os.path.join(AQUI, "..", "config", "assets.json")

ESTADOS = ("medido", "sin_datos", "no_aplica")


def interpretar(media):
    if media is None:
        return "sin_datos"
    if media <= -1.0:
        return "muy_bajista"
    if media <= -0.35:
        return "bajista"
    if media < 0.35:
        return "neutral"
    if media < 1.0:
        return "alcista"
    return "muy_alcista"


def _ts(valor):
    """Parsea un ISO-8601 con Z final. Devuelve None si no es parseable.

    Si viene sin zona horaria se asume UTC: de lo contrario comparar un naive
    con un aware revienta con TypeError en vez de dar un error de validacion.
    """
    if not isinstance(valor, str) or not valor:
        return None
    try:
        t = datetime.fromisoformat(valor.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("uso: validar_respuestas_v2.py respuestas_TICKER.json [--fix] [--cuestionario RUTA]")
    path = args[0]
    fix = "--fix" in sys.argv
    cuest_path = CUEST_POR_DEFECTO
    if "--cuestionario" in sys.argv:
        cuest_path = sys.argv[sys.argv.index("--cuestionario") + 1]

    with open(cuest_path, encoding="utf-8") as fh:
        q = json.load(fh)
    preguntas = q["preguntas"]
    banco = {p["id"]: {o["id_opcion"]: o["valor"] for o in p["opciones"]} for p in preguntas}
    meta = {p["id"]: p for p in preguntas}
    # P30 vive en benchmark_llm, no en 'respuestas'.
    orden = [p["id"] for p in preguntas if p["id"] != "P30"]
    sumables = [p["id"] for p in preguntas if p.get("sumable") is True]

    assets = {}
    if os.path.exists(ASSETS_POR_DEFECTO):
        with open(ASSETS_POR_DEFECTO, encoding="utf-8") as fh:
            assets = {a["ticker"]: a for a in json.load(fh)["activos"]}

    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    err = []

    # --- campos raiz -------------------------------------------------------
    for campo in ("schema_version", "cuestionario", "activo", "ticker", "fecha",
                  "hora_corte_utc", "reporte_fuente", "ancla", "respuestas",
                  "benchmark_llm", "puntuacion_agregada", "cobertura"):
        if campo not in d:
            err.append("falta el campo raiz '%s'" % campo)
    if d.get("cuestionario") != "cuestionario_cripto_v2":
        err.append("campo 'cuestionario' incorrecto: %r" % d.get("cuestionario"))
    if str(d.get("schema_version")) != "2.0":
        err.append("schema_version deberia ser '2.0', es %r" % d.get("schema_version"))

    corte = _ts(d.get("hora_corte_utc"))
    if corte is None:
        err.append("'hora_corte_utc' ausente o no es ISO-8601: %r" % d.get("hora_corte_utc"))

    # --- ancla: lo unico sin lo cual el dia es irrecuperable ---------------
    a = d.get("ancla") or {}
    if not isinstance(a.get("precio_usd"), (int, float)) or isinstance(a.get("precio_usd"), bool):
        err.append("ancla.precio_usd debe ser un numero, es %r" % a.get("precio_usd"))
    if _ts(a.get("ts_utc")) is None:
        err.append("ancla.ts_utc ausente o no es ISO-8601: %r" % a.get("ts_utc"))
    for campo in ("fuente", "url", "exchange_ref", "par"):
        valor = a.get(campo)
        if not isinstance(valor, str) or not valor.strip():
            err.append("ancla.%s ausente o vacio" % campo)
    if not isinstance(a.get("n_fuentes_contrastadas"), int) or a.get("n_fuentes_contrastadas", 0) < 2:
        err.append("ancla.n_fuentes_contrastadas debe ser un entero >= 2, es %r"
                   % a.get("n_fuentes_contrastadas"))
    rango = a.get("precio_rango_usd")
    if rango is not None:
        if (not isinstance(rango, list) or len(rango) != 2
                or not all(isinstance(x, (int, float)) for x in rango) or rango[0] > rango[1]):
            err.append("ancla.precio_rango_usd debe ser [min, max] con min <= max, es %r" % rango)

    # --- respuestas --------------------------------------------------------
    r = d.get("respuestas", {})
    if len(r) != len(orden):
        err.append("se esperaban %d respuestas, hay %d" % (len(orden), len(r)))
    if list(r.keys()) != orden:
        faltan = [x for x in orden if x not in r]
        sobran = [x for x in r if x not in orden]
        if faltan:
            err.append("faltan preguntas: %s" % faltan)
        if sobran:
            err.append("ids inesperados en 'respuestas': %s" % sobran)
    if "P30" in r:
        err.append("P30 no va en 'respuestas': es benchmark y vive en 'benchmark_llm'")

    ticker = d.get("ticker")
    activo_cfg = assets.get(ticker, {})

    suma, resp, sind, nap, disp_sumables = 0, 0, 0, 0, 0
    for qid in orden:
        if qid not in r:
            continue
        ans = r[qid]
        m = meta[qid]
        for c in ("id_opcion", "valor", "estado", "disponible", "origen", "evidencia"):
            if c not in ans:
                err.append("%s: falta '%s'" % (qid, c))
        op = ans.get("id_opcion")
        if op not in banco[qid]:
            err.append("%s: id_opcion '%s' no existe (validas: %s)"
                       % (qid, op, list(banco[qid])))
            continue
        esperado = banco[qid][op]
        if ans.get("valor") != esperado:
            err.append("%s: valor %r no coincide con el cuestionario (%r)"
                       % (qid, ans.get("valor"), esperado))

        estado = ans.get("estado")
        if estado not in ESTADOS:
            err.append("%s: estado %r no es uno de %s" % (qid, estado, list(ESTADOS)))
        # coherencia estado <-> id_opcion <-> disponible
        if op == "sin_datos" and estado != "sin_datos":
            err.append("%s: id_opcion 'sin_datos' con estado '%s'" % (qid, estado))
        if op == "no_aplica" and estado != "no_aplica":
            err.append("%s: id_opcion 'no_aplica' con estado '%s'" % (qid, estado))
        if op not in ("sin_datos", "no_aplica") and estado != "medido":
            err.append("%s: id_opcion '%s' deberia ir con estado 'medido', va con '%s'"
                       % (qid, op, estado))
        esperada_disp = 1 if estado == "medido" else 0
        if ans.get("disponible") != esperada_disp:
            err.append("%s: disponible=%r, deberia ser %d con estado '%s'"
                       % (qid, ans.get("disponible"), esperada_disp, estado))

        if ans.get("origen") not in ("medicion", "reporte", "fuente_directa"):
            err.append("%s: origen %r no valido" % (qid, ans.get("origen")))

        ev = (ans.get("evidencia") or "").strip()
        if len(ev) < 8:
            err.append("%s: 'evidencia' vacia o demasiado corta" % qid)

        # valor_crudo obligatorio si la pregunta tiene metrica y esta medida
        if m.get("metrica_cruda") and estado == "medido":
            vc = ans.get("valor_crudo")
            if not isinstance(vc, (int, float)) or isinstance(vc, bool):
                err.append("%s: falta 'valor_crudo' numerico (metrica %s)"
                           % (qid, m["metrica_cruda"]["campo"]))

        # regla point-in-time
        fts = _ts(ans.get("fuente_ts"))
        if fts is not None and corte is not None and fts > corte:
            err.append("%s: fuente_ts %s es POSTERIOR a la hora de corte %s (look-ahead)"
                       % (qid, ans.get("fuente_ts"), d.get("hora_corte_utc")))

        # no_aplica solo si assets.json lo respalda
        if estado == "no_aplica" and activo_cfg:
            if qid in ("P13", "P14") and activo_cfg.get("producto_cotizado") != "no":
                err.append("%s: 'no_aplica' pero assets.json dice producto_cotizado=%r "
                           "(solo 'no' lo habilita; 'desconocido' obliga a sin_datos)"
                           % (qid, activo_cfg.get("producto_cotizado")))
            if qid == "P36" and activo_cfg.get("desbloqueos_programados") != "no":
                err.append("P36: 'no_aplica' pero assets.json dice desbloqueos_programados=%r"
                           % activo_cfg.get("desbloqueos_programados"))

        if estado == "sin_datos":
            sind += 1
        elif estado == "no_aplica":
            nap += 1
        if qid in sumables:
            if esperado is not None:
                suma += esperado
                resp += 1
            if estado == "medido":
                disp_sumables += 1

    # --- benchmark_llm -----------------------------------------------------
    b = d.get("benchmark_llm") or {}
    if "P30" not in b:
        err.append("falta 'benchmark_llm.P30'")
    else:
        op30 = b["P30"].get("id_opcion")
        if op30 not in banco["P30"]:
            err.append("benchmark_llm.P30: id_opcion '%s' no existe" % op30)
        elif b["P30"].get("valor") != banco["P30"][op30]:
            err.append("benchmark_llm.P30: valor %r no coincide con el cuestionario (%r)"
                       % (b["P30"].get("valor"), banco["P30"][op30]))

    if resp != disp_sumables:
        err.append("incoherencia: %d sumables con valor no nulo pero %d con estado 'medido'"
                   % (resp, disp_sumables))

    media = round(suma / resp, 4) if resp else None
    calc = {
        "usar_en_entrenamiento": False,
        "preguntas_sumadas": len(sumables),
        "suma_valores": suma,
        "preguntas_respondidas": resp,
        "preguntas_sin_datos": sind,
        "preguntas_no_aplica": nap,
        "media": media,
        "interpretacion": interpretar(media),
    }
    cob = {
        "preguntas_totales": len(orden) + 1,
        "sumables_totales": len(sumables),
        "sumables_disponibles": disp_sumables,
        "cobertura_sumables_pct": round(100.0 * disp_sumables / len(sumables), 2) if sumables else 0.0,
    }

    if fix:
        d["puntuacion_agregada"] = calc
        d["cobertura"] = cob
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    else:
        pa = d.get("puntuacion_agregada", {})
        for k, v in calc.items():
            got = pa.get(k)
            if k == "media" and isinstance(got, (int, float)) and v is not None:
                if abs(got - v) > 0.005:
                    err.append("puntuacion_agregada.media=%s, calculada=%s" % (got, v))
            elif got != v:
                err.append("puntuacion_agregada.%s=%s, calculada=%s" % (k, got, v))
        co = d.get("cobertura", {})
        for k, v in cob.items():
            got = co.get(k)
            if k == "cobertura_sumables_pct" and isinstance(got, (int, float)):
                if abs(got - v) > 0.05:
                    err.append("cobertura.%s=%s, calculada=%s" % (k, got, v))
            elif got != v:
                err.append("cobertura.%s=%s, calculada=%s" % (k, got, v))

    name = os.path.basename(path)
    if err:
        print("FALLO %s (%d problemas)" % (name, len(err)))
        for e in err:
            print("   - " + e)
        sys.exit(1)
    print("OK %s | %d respuestas | suma=%s sumables_resp=%s sin_datos=%s no_aplica=%s "
          "media=%s (%s) | cobertura=%.1f%%"
          % (name, len(orden), calc["suma_valores"], calc["preguntas_respondidas"],
             calc["preguntas_sin_datos"], calc["preguntas_no_aplica"], calc["media"],
             calc["interpretacion"], cob["cobertura_sumables_pct"]))


if __name__ == "__main__":
    main()
