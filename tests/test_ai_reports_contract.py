"""
El CONTRATO del reporte diario por activo, comprobado sin datos y sin red.

Por que existe este fichero. `config/cuestionario_cripto_v2.json`, `config/assets.json` y
sus companeros no los lee el paquete: los lee un agente externo (Claude Cowork) todos los
dias a las 08:00 Europe/Madrid, en otra maquina y en otro sandbox. Eso invierte el coste de
un error tipografico. Un JSON mal cerrado o un `id_opcion` repetido no rompe nada aqui --no
hay nada aqui que lo importe-- y rompe la ejecucion de manana por la manana, en un sitio
donde nadie esta mirando y donde el dia perdido no se recupera: la captura es
point-in-time y el 21 de agosto solo se puede medir el 21 de agosto.

Asi que lo que se comprueba es lo que se puede comprobar SIN la ejecucion del dia: que los
ficheros de entrada estan, que los ids son los que el validador espera, que toda pregunta
tiene salida para el hueco, y que el universo declara los dos campos que GOBIERNAN el
cuestionario. Lo que hace falta el fichero del dia para verlo --que una respuesta concreta
sea coherente-- es trabajo de `tools/validar_respuestas_v2.py`, que corre en la sesion del
agente y contra el fichero que acaba de escribir.

Lo que este test NO comprueba, y conviene no confundirlo: que las respuestas del dia sean
CIERTAS. Ningun test puede. El cuestionario no mide el mercado, mide lo que un conjunto de
fuentes publicas decia del mercado a una hora concreta.
"""
from __future__ import annotations

import json

import pytest

from golden_support import REPO_ROOT

from ai_trader.signals.ai_reports import (
    ANSWER_TEMPLATE,
    ASSETS,
    LABEL_SCHEMA,
    QUESTIONNAIRE,
    REQUIRED_INPUTS,
    STATES,
    contract_problems,
    load_contract,
    load_last_run,
    run_dates,
)


def _json(rel) -> dict:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def test_el_contrato_no_tiene_incoherencias() -> None:
    """El test de cabecera: si esto falla, manana a las ocho falla la captura.

    `contract_problems` devuelve prosa, no booleanos, para que el fallo diga QUE arreglar
    sin tener que abrir un JSON de cincuenta kilobytes a buscarlo."""
    problems = contract_problems(REPO_ROOT)
    assert problems == [], "El contrato del reporte diario tiene problemas:\n  - " + "\n  - ".join(
        problems
    )


@pytest.mark.parametrize("rel", REQUIRED_INPUTS, ids=lambda p: p.name)
def test_cada_fichero_de_entrada_existe(rel) -> None:
    """Los seis ficheros que el prompt del agente da por hechos.

    El prompt vive FUERA del repo, en la tarea de Cowork, asi que no hay forma de que un
    borrado aqui se note alli hasta que la ejecucion falla. Esta lista es el unico sitio
    del repo donde esa dependencia esta escrita."""
    assert (REPO_ROOT / rel).exists(), f"{rel.as_posix()} lo necesita la tarea diaria"


def test_las_37_preguntas_y_las_29_que_suman() -> None:
    """Las cifras que el validador imprime y que la plantilla da por hechas.

    Se fijan aqui a proposito. Anadir una pregunta es legitimo, pero obliga a pasar por
    este test, por el validador y por el prompt a la vez -- que es exactamente lo que hay
    que hacer cuando cambia el esquema de un dataset que ya tiene dias capturados."""
    contract = load_contract(REPO_ROOT)
    assert contract is not None, "sin cuestionario o sin universo no hay contrato que leer"
    assert contract["n_questions"] == 37
    assert contract["n_sumable"] == 29
    # 36 en `respuestas` (P01..P37 sin P30) mas P30 sola en `benchmark_llm`. El validador
    # imprime "36 respuestas" y falla si cuenta otra cosa.
    assert contract["n_questions"] - 1 == 36
    assert contract["benchmark_question"] == "P30"


def test_p30_es_benchmark_y_no_entra_en_la_suma() -> None:
    """La circularidad que la v2 vino a quitar, convertida en test.

    En la v1 el cuestionario entrevistaba al redactor: P30 era la conclusion del propio
    reporte y sumaba. Si alguien la marca `sumable` otra vez, la agregada del dia vuelve a
    incluir el juicio del que escribio el texto, y el fallo es silencioso -- las cifras
    siguen saliendo, solo que miden al agente."""
    questions = {q["id"]: q for q in _json(QUESTIONNAIRE)["preguntas"]}
    assert questions["P30"]["sumable"] is False
    assert questions["P30"]["uso"] != "feature"

    template = _json(ANSWER_TEMPLATE)
    assert "P30" in template["benchmark_llm"], "P30 vive en benchmark_llm, no en respuestas"
    assert "P30" not in template["respuestas"]
    assert template["puntuacion_agregada"]["usar_en_entrenamiento"] is False


def test_toda_pregunta_ofrece_salida_para_el_hueco() -> None:
    """Sin `sin_datos`/`no_aplica` la unica alternativa del agente es inventarse un numero.

    Y un valor inventado es indistinguible de uno medido en cuanto esta en el fichero: no
    se detecta a posteriori y contamina lo que se entrene con el."""
    faltan = [
        q["id"]
        for q in _json(QUESTIONNAIRE)["preguntas"]
        if not {"sin_datos", "no_aplica"} & {o["id_opcion"] for o in q["opciones"]}
    ]
    assert faltan == [], f"preguntas sin salida para el hueco: {faltan}"


def test_sin_datos_y_no_aplica_no_colisionan_en_cero() -> None:
    """El error concreto de la v1: `no_aplica` valia 0 y chocaba con 'flujos planos'.

    Son cosas opuestas -- la metrica no existe para el activo, frente a existe y no se
    encontro -- y meterlas las dos en el mismo cero mete un dato falso donde habia un
    hueco. En la v2 las dos llevan `valor: null` y se distinguen por `estado`."""
    for q in _json(QUESTIONNAIRE)["preguntas"]:
        for option in q["opciones"]:
            if option["id_opcion"] in ("sin_datos", "no_aplica"):
                assert option["valor"] is None, (
                    f"{q['id']}/{option['id_opcion']} vale {option['valor']!r} y debe ser null"
                )

    template = _json(ANSWER_TEMPLATE)
    for qid, answer in template["respuestas"].items():
        if answer["estado"] in ("sin_datos", "no_aplica"):
            assert answer["valor"] is None, f"{qid}: {answer['estado']} con valor no nulo"
            assert answer["disponible"] == 0, f"{qid}: {answer['estado']} con disponible=1"
        else:
            assert answer["disponible"] == 1, f"{qid}: medido con disponible=0"
        assert answer["estado"] in STATES


def test_las_preguntas_de_medicion_no_dejan_juicio_al_agente() -> None:
    """`origen_esperado: medicion` promete que la opcion sale del numero, mecanicamente.

    Si una de esas no trae `derivacion`, la promesa es falsa: el agente tiene que decidir,
    y entonces la columna mide al anotador y no al mercado. Es la diferencia entre una
    pregunta con techo de senal alto y una con acuerdo entre anotadores desconocido."""
    sin_derivacion = [
        q["id"]
        for q in _json(QUESTIONNAIRE)["preguntas"]
        if q["origen_esperado"] == "medicion" and not q.get("derivacion")
    ]
    assert sin_derivacion == []


def test_el_universo_gobierna_las_dos_preguntas_que_dependen_de_el() -> None:
    """`assets.json` manda, y `desconocido` NO habilita `no_aplica`.

    La distincion es la que impide que un hueco de investigacion se cuele como un
    'no existe': `no` esta verificado, `desconocido` es que nadie lo comprobo todavia, y
    solo el primero autoriza `no_aplica` en P13/P14 y P36."""
    rows = _json(ASSETS)["activos"]
    for row in rows:
        for field in ("producto_cotizado", "desbloqueos_programados"):
            assert row[field] in ("si", "no", "desconocido"), (
                f"{row['ticker']}: {field} = {row[field]!r}"
            )

    contract = load_contract(REPO_ROOT)
    universe = contract["universe"]
    assert universe["n_assets"] == len(rows)
    assert len(set(universe["tickers"])) == len(rows), "hay tickers repetidos"
    # Que existan `desconocido` no es un defecto: es el estado honesto de la verificacion.
    # Se afirma para que el dia que alguien los cierre a `si`/`no` se vea aqui.
    assert sum(universe["listed_product"].values()) == len(rows)
    assert sum(universe["unlocks"].values()) == len(rows)


def test_el_ancla_esta_en_la_plantilla_y_en_el_esquema_de_etiquetas() -> None:
    """Sin ancla el dia es irrecuperable, y es el unico campo del que eso es cierto.

    Todo lo demas de una ejecucion se puede volver a mirar mas tarde con mejor criterio.
    El precio a la hora de corte, no: si no se guardo como numero, no hay forma de calcular
    a posteriori que paso despues, y el dia entero deja de servir como dato de
    entrenamiento por bueno que fuese el cuestionario."""
    anchor = _json(ANSWER_TEMPLATE)["ancla"]
    for field in ("precio_usd", "ts_utc", "n_fuentes_contrastadas"):
        assert field in anchor, f"la plantilla del ancla no declara {field}"
    assert anchor["n_fuentes_contrastadas"] >= 2, "el ancla se contrasta con >=2 fuentes"

    fields = _json(LABEL_SCHEMA)["campos_por_ticker"]
    assert "ancla_precio_usd" in fields and "ancla_ts_utc" in fields
    # La etiqueta que de verdad importa y la que alinea el label con lo que el sistema
    # puede aportar: neto de costes, y en exceso sobre BTC porque el sistema es long/short.
    assert "ret_14d_neto" in fields
    assert "ret_14d_exceso_btc" in fields
    # MFE/MAE: un corto que acaba en +0% pero paso por -18% te liquida antes de tener
    # razon. El retorno final no lo captura.
    assert "mfe_14d" in fields and "mae_14d" in fields


def test_la_ultima_ejecucion_es_coherente_si_la_hay() -> None:
    """Se salta cuando no hay datos, y eso es lo correcto en un clon recien hecho.

    `data/signals_raw/` esta en el .gitignore: la captura diaria no se versiona. Un clon
    limpio tiene el contrato entero y cero ejecuciones, que no es un fallo. Cuando SI hay
    ejecucion, lo que se comprueba es la coherencia interna del dia, no sus cifras."""
    run = load_last_run(REPO_ROOT)
    if run is None:
        pytest.skip("no hay ninguna ejecucion capturada en este clon")

    assert run["date"] in run_dates(REPO_ROOT)
    assert run["n_assets"] > 0
    # El trio por activo: reporte, respuestas y medidas. Que uno de los tres falte
    # significa que el activo se quedo a medias, y el dia lo tiene que decir.
    assert run["incomplete"] == [], f"activos sin el trio completo: {run['incomplete']}"
    assert run["n_complete"] == run["n_reports"] == run["n_answers"] == run["n_measures"]
    if run["coverage_mean_pct"] is not None:
        assert 0.0 <= run["coverage_mean_pct"] <= 100.0


# --- las dos formas del resumen, que es donde ya se rompio una vez ---------------------
#
# El 2026-08-22 la tarea externa cambio la forma de `_resumen.json` sin avisar: `activos`
# paso de lista de filas a recuento entero, las filas se mudaron a `tickers` y los dos
# bloques agregados se renombraron. El lector, escrito el dia anterior contra la forma
# vieja, reventaba con `object of type 'int' has no len()`, y con el se caia el generador
# del dashboard. Aqui quedan las DOS formas congeladas: la de ayer y la de hoy, con datos
# minimos escritos a mano, para que la proxima vez el fallo salga en la suite y no a las
# ocho de la manana.

_SUMMARY_OLD = {
    "fecha": "2026-08-21",
    "hora_corte_utc": "2026-08-21T06:00:00Z",
    "cuestionario": "cuestionario_cripto_v2",
    "n_activos": 2,
    "agregado": {
        "media_de_medias": 0.5,
        "desviacion_medias": 0.1,
        "cobertura_media_pct": 55.0,
        "cobertura_min_pct": 50.0,
        "cobertura_max_pct": 60.0,
        "reparto_p30": {"alcista": 2},
    },
    "diagnostico_cobertura_vs_media": {"pearson": 0.3, "spearman": 0.4},
    "activos": [
        {"ticker": "BTC", "media": 0.6, "interpretacion": "alcista", "cobertura_pct": 60.0,
         "p30": "alcista", "rango_media": 1, "percentil_media": 90.0,
         "percentil_cobertura": 80.0, "ancla_precio_usd": 75_000.0,
         "ancla_ts_utc": "2026-08-21T06:00:00Z"},
        {"ticker": "ETH", "media": 0.4, "interpretacion": "alcista", "cobertura_pct": 50.0,
         "p30": "alcista", "rango_media": 2, "percentil_media": 10.0,
         "percentil_cobertura": 20.0, "ancla_precio_usd": 3_000.0,
         "ancla_ts_utc": "2026-08-21T06:00:00Z"},
    ],
}

_SUMMARY_NEW = {
    "schema_version": "1.0",
    "fecha": "2026-08-22",
    "hora_corte_utc": "2026-08-22T06:00:00Z",
    "cuestionario": "cuestionario_cripto_v2",
    "activos": 2,
    "correlacion_media_cobertura": {"pearson": 0.3, "spearman": 0.4},
    "sesgo_agregado": {
        "media_de_medias": 0.5,
        "cobertura_media_pct": 55.0,
        "p30_distribucion": {"alcista": 2},
    },
    "tickers": {
        "BTC": {"media": 0.6, "interpretacion": "alcista", "cobertura_sumables_pct": 60.0,
                "sesgo_p30": "alcista", "rank_media": 1,
                "percentil_seccion_transversal": 90.0, "ancla_precio_usd": 75_000.0,
                "ancla_ts_utc": "2026-08-22T06:00:00Z"},
        "ETH": {"media": 0.4, "interpretacion": "alcista", "cobertura_sumables_pct": 50.0,
                "sesgo_p30": "alcista", "rank_media": 2,
                "percentil_seccion_transversal": 10.0, "ancla_precio_usd": 3_000.0,
                "ancla_ts_utc": "2026-08-22T06:00:00Z"},
    },
}


def _fake_run(tmp_path, summary: dict):
    day = tmp_path / "data" / "signals_raw" / "ai_reports" / summary["fecha"]
    (day / "_medidas").mkdir(parents=True)
    (day / "_resumen.json").write_text(json.dumps(summary), encoding="utf-8")
    for ticker in ("BTC", "ETH"):
        (day / f"reporte_{ticker}.html").write_text("<html></html>", encoding="utf-8")
        (day / f"respuestas_{ticker}.json").write_text("{}", encoding="utf-8")
        (day / "_medidas" / f"medidas_{ticker}.json").write_text("{}", encoding="utf-8")
    return load_last_run(tmp_path)


@pytest.mark.parametrize("summary", [_SUMMARY_OLD, _SUMMARY_NEW], ids=["forma_vieja", "forma_nueva"])
def test_las_dos_formas_del_resumen_se_leen_igual(tmp_path, summary) -> None:
    run = _fake_run(tmp_path, summary)

    assert run is not None
    assert run["n_assets"] == 2
    assert run["n_complete"] == 2 and run["incomplete"] == []
    assert run["mean_of_means"] == 0.5
    assert run["coverage_mean_pct"] == 55.0
    # Minimo y maximo: la forma nueva no los publica y se derivan de las filas, que es
    # aritmetica sin ambiguedad. La desviacion NO se deriva, y por eso puede faltar.
    assert run["coverage_min_pct"] == 50.0 and run["coverage_max_pct"] == 60.0
    assert run["p30_split"] == {"alcista": 2}
    assert run["coverage_bias_pearson"] == 0.3
    assert [r["ticker"] for r in run["rows"]] == ["BTC", "ETH"]
    btc = run["rows"][0]
    assert btc["mean"] == 0.6 and btc["coverage_pct"] == 60.0 and btc["rank"] == 1
    assert btc["p30"] == "alcista" and btc["anchor_usd"] == 75_000.0


def test_un_resumen_que_no_se_entiende_no_revienta(tmp_path) -> None:
    """El caso que de verdad se dio: una forma que el lector no conoce. Lo correcto es una
    vista vacia, no una excepcion que se lleva por delante el generador del dashboard."""
    run = _fake_run(tmp_path, {"fecha": "2026-08-23", "activos": {"lo que sea": 1}})

    assert run is not None
    assert run["rows"] == []
    assert run["n_assets"] == 2  # sale del recuento de ficheros, que si esta
    assert run["mean_of_means"] is None
