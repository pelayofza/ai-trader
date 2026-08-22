"""
EL REPORTE DIARIO POR ACTIVO: lector del contrato y de la ultima ejecucion.

Es la segunda via de captura del proyecto, y es de otra naturaleza que la primera. La de
`capture.py` va contra APIs y devuelve numeros; esta la ejecuta un agente externo (Claude
Cowork) todos los dias a las 08:00 Europe/Madrid, lee fuentes publicas de la web y devuelve
CATEGORIAS: 37 preguntas por activo, sobre 24 activos. No comparte una sola linea de codigo
con la otra --no toca `SignalStore`, no pasa por `normalize.py`, no entra en el radar-- y
por eso vive en su propio modulo en vez de colgarse de la maquinaria del catalogo.

QUE HACE ESTE FICHERO, Y QUE NO
-------------------------------
Solo LEE, y no toca red. Dos cosas, separadas a proposito porque tienen vidas distintas:

  * `load_contract()`  <- `config/`. Esta VERSIONADO. Cambia cuando alguien lo edita.
  * `load_last_run()`  <- `data/signals_raw/ai_reports/`. Esta en el .gitignore y crece
                          cada mañana a las ocho. NO es reproducible en otro clon.

Mezclar las dos en una sola estructura seria repetir el error que el propio pipeline vino a
arreglar: confundir lo que el contrato PROMETE con lo que una ejecucion MIDIO. Un clon
recien hecho tiene contrato entero y ninguna ejecucion, y eso no es un fallo.

LO QUE ESTE MODULO NO PROMETE
-----------------------------
Que los ficheros del dia esten bien. Comprueba que el CONTRATO es coherente
(`contract_problems`), que es lo que se puede verificar sin datos; validar una respuesta
concreta es trabajo de `tools/validar_respuestas_v2.py`, que corre en la sesion del agente
y contra el fichero que acaba de escribir.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Todo relativo a la raiz del repo, como el resto de rutas del paquete (ver `capture.py`).
CONFIG_DIR = Path("config")
AI_REPORTS_DIR = Path("data") / "signals_raw" / "ai_reports"

QUESTIONNAIRE = CONFIG_DIR / "cuestionario_cripto_v2.json"
ASSETS = CONFIG_DIR / "assets.json"
LABEL_SCHEMA = CONFIG_DIR / "esquema_etiquetas.json"
ANSWER_TEMPLATE = CONFIG_DIR / "plantilla_respuestas_v2.json"
REPORT_TEMPLATE = CONFIG_DIR / "plantilla_reporte.html"
AGENT_INSTRUCTIONS = CONFIG_DIR / "INSTRUCCIONES_AGENTE.md"
GLOBAL_CONTEXT = CONFIG_DIR / "contexto_global.md"

# Los seis ficheros que el agente necesita para arrancar. Se listan aqui y no en el prompt
# porque el prompt vive fuera del repo: si manana falta uno, esto lo dice hoy y no a las
# ocho de la mañana en un sandbox sin nadie mirando.
REQUIRED_INPUTS = (
    QUESTIONNAIRE,
    ASSETS,
    LABEL_SCHEMA,
    ANSWER_TEMPLATE,
    REPORT_TEMPLATE,
    AGENT_INSTRUCTIONS,
)

# La unica salida del dia que NO es por activo y que fija el contrato de etiquetas. Sin
# ancla el dia es irrecuperable: no hay forma de calcular a posteriori que paso despues.
SUMMARY_FILE = "_resumen.json"
LABELS_PREFIX = "etiquetas_"
MEASURES_DIR = "_medidas"

# El nombre CANONICO del dia lleva el ticker y nada mas. Una reejecucion añade sufijo
# (`_v2`, `_v3`) y una version archivada tambien (`_vOld`), y las dos conviven en la misma
# carpeta porque la regla del pipeline es no sobrescribir nunca. Contar ficheros a secas
# daria 48 reportes para 24 activos; lo que interesa es cuantos activos tienen su trio
# completo, y por separado cuantos ficheros hay de mas.
_CANONICAL = re.compile(r"^(?:reporte|respuestas|medidas)_([A-Z0-9]+)\.(?:html|json)$")


def _canonical_tickers(paths) -> set[str]:
    """Los tickers con fichero de nombre canonico. Los sufijados se ignoran aqui."""
    return {m.group(1) for m in (_CANONICAL.match(p.name) for p in paths) if m}

# `estado` y `motivo_null` son la mascara del dataset. Se declaran aqui porque el test del
# contrato los compara contra lo que el cuestionario ofrece como opciones.
STATES = ("medido", "sin_datos", "no_aplica")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------- contrato ----
def load_contract(root: Path) -> dict | None:
    """El contrato versionado: cuestionario, universo y esquema de etiquetas.

    Devuelve `None` si falta el cuestionario o el universo, que son los dos ficheros sin
    los cuales no hay nada que describir. Los demas se degradan a `None` dentro del dict:
    que falte la plantilla de reporte empeora el HTML del agente, no invalida el resto.
    """
    questionnaire = _read_json(root / QUESTIONNAIRE)
    assets = _read_json(root / ASSETS)
    if not questionnaire or not assets:
        return None

    questions = questionnaire.get("preguntas") or []
    rows = assets.get("activos") or []
    labels = _read_json(root / LABEL_SCHEMA) or {}

    by_origin = Counter(q.get("origen_esperado") for q in questions)
    by_section = Counter(q.get("seccion") for q in questions)
    sumable = [q for q in questions if q.get("sumable")]

    return {
        "questionnaire_id": questionnaire.get("id_cuestionario"),
        "schema_version": questionnaire.get("schema_version"),
        "replaces": questionnaire.get("reemplaza_a"),
        "n_questions": len(questions),
        "n_sumable": len(sumable),
        # Las que NO suman llevan su motivo en el propio cuestionario: son estado
        # descriptivo (P16, P25, P31-P35) o el juicio del redactor (P30). Sumarlas seria
        # afirmar una interpretacion que le toca al modelo.
        "excluded_from_sum": list(
            (questionnaire.get("reglas_de_agregacion") or {}).get("excluidas_de_la_suma") or []
        ),
        # `medicion` = la opcion sale de aplicar una derivacion a un numero, sin juicio.
        # `reporte` = juicio narrativo. `cualquiera` = de donde haya dato, y es la que
        # rompe la circularidad de la v1, porque admite fuente ajena al reporte.
        "by_origin": dict(by_origin),
        "by_section": dict(by_section),
        "n_with_raw_metric": sum(1 for q in questions if q.get("metrica_cruda")),
        "sections": [
            {"key": key, "n": n} for key, n in sorted(by_section.items(), key=lambda kv: -kv[1])
        ],
        "benchmark_question": "P30",
        "states": list(STATES),
        "universe": {
            "n_assets": len(rows),
            "updated": assets.get("actualizado"),
            "tickers": [r.get("ticker") for r in rows],
            # Los dos campos que GOBIERNAN el cuestionario desde el fichero de activos:
            # `no` habilita `no_aplica`; `desconocido` NO lo habilita y obliga a
            # `sin_datos`. Son cosas opuestas y el recuento deja ver cuanto pesa cada una.
            "listed_product": dict(Counter(r.get("producto_cotizado") for r in rows)),
            "unlocks": dict(Counter(r.get("desbloqueos_programados") for r in rows)),
        },
        "labels": {
            "id": labels.get("id"),
            # El horizonte objetivo del sistema, y la razon por la que el proceso que las
            # rellena todavia no existe: es calculo numerico sobre mercado, que segun la
            # Regla 4 del proyecto no es un cambio mecanico.
            "fields": [k for k in (labels.get("campos_por_ticker") or {}) if k != "ticker"],
            "pending": True,
        },
    }


def contract_problems(root: Path) -> list[str]:
    """Incoherencias del contrato, en prosa. Lista vacia = coherente.

    Se comprueba lo que rompe la ejecucion del dia siguiente SIN NECESIDAD DE DATOS: que
    los ficheros de entrada existan, que los ids sean los que el validador espera, que
    toda opcion tenga valor y que la derivacion de cada pregunta dura apunte a una metrica
    declarada. Es barato y cierra el hueco entre editar `config/` y descubrir el fallo a
    las ocho de la mañana siguiente en un sandbox.
    """
    problems: list[str] = []

    for rel in REQUIRED_INPUTS:
        if not (root / rel).exists():
            problems.append(f"falta el fichero de entrada {rel.as_posix()}")

    questionnaire = _read_json(root / QUESTIONNAIRE)
    if not questionnaire:
        problems.append(f"{QUESTIONNAIRE.as_posix()} no es JSON valido")
        return problems

    questions = questionnaire.get("preguntas") or []
    ids = [q.get("id") for q in questions]
    expected = [f"P{i:02d}" for i in range(1, len(questions) + 1)]
    if ids != expected:
        problems.append(
            f"los ids del cuestionario no son P01..P{len(questions):02d} en orden: {ids[:3]}..."
        )

    # El validador y la plantilla dan por hecho que P30 existe, que NO suma y que vive en
    # `benchmark_llm`. Si alguien la marcase sumable, la agregada del dia pasaria a incluir
    # la conclusion del propio redactor: exactamente la circularidad que la v2 quito.
    benchmark = next((q for q in questions if q.get("id") == "P30"), None)
    if benchmark is None:
        problems.append("no existe P30, que es el benchmark del redactor")
    elif benchmark.get("sumable"):
        problems.append("P30 esta marcada sumable: es el benchmark, no puede entrar en la suma")

    rules = questionnaire.get("reglas_de_agregacion") or {}
    declared = set(rules.get("excluidas_de_la_suma") or [])
    actual = {q.get("id") for q in questions if not q.get("sumable")}
    if declared != actual:
        problems.append(
            "las excluidas de la suma que declara el cuestionario no son las que tienen "
            f"sumable=false: declara {sorted(declared)}, y son {sorted(actual)}"
        )

    for q in questions:
        qid = q.get("id")
        options = q.get("opciones") or []
        if not options:
            problems.append(f"{qid} no tiene opciones")
            continue
        seen = Counter(o.get("id_opcion") for o in options)
        duplicated = [k for k, n in seen.items() if n > 1]
        if duplicated:
            problems.append(f"{qid} repite id_opcion: {duplicated}")
        # Toda pregunta necesita una salida para el hueco. Sin ella el agente no puede
        # responder "no lo encontre" y la unica alternativa es inventarse un numero.
        if not ({"sin_datos", "no_aplica"} & set(seen)):
            problems.append(f"{qid} no ofrece ni sin_datos ni no_aplica")
        if q.get("origen_esperado") not in ("medicion", "reporte", "cualquiera"):
            problems.append(f"{qid} tiene origen_esperado desconocido: {q.get('origen_esperado')}")
        # Una pregunta de medicion sin derivacion deja la opcion al juicio del agente, que
        # es justo lo que 'medicion' promete que no pasa.
        if q.get("origen_esperado") == "medicion" and not q.get("derivacion"):
            problems.append(f"{qid} es de medicion y no declara derivacion")
        if q.get("metrica_cruda") and not (q["metrica_cruda"].get("campo")):
            problems.append(f"{qid} declara metrica_cruda sin campo")

    assets = _read_json(root / ASSETS)
    if assets is None:
        problems.append(f"{ASSETS.as_posix()} no es JSON valido")
        return problems

    rows = assets.get("activos") or []
    if not rows:
        problems.append(f"{ASSETS.as_posix()} no declara ningun activo")
    tickers = Counter(r.get("ticker") for r in rows)
    repeated = [t for t, n in tickers.items() if n > 1]
    if repeated:
        problems.append(f"{ASSETS.as_posix()} repite tickers: {repeated}")
    for row in rows:
        for field in ("producto_cotizado", "desbloqueos_programados"):
            if row.get(field) not in ("si", "no", "desconocido"):
                problems.append(
                    f"{row.get('ticker')}: {field} = {row.get(field)!r}; "
                    "solo valen si/no/desconocido"
                )
        if not row.get("par") or not row.get("exchange_ref"):
            problems.append(f"{row.get('ticker')}: sin par o sin exchange_ref de referencia")

    return problems


# ------------------------------------------------------------------ ejecuciones ----
def run_dates(root: Path) -> list[str]:
    """Las fechas con carpeta, de la mas antigua a la mas reciente. Vacia si no hay nada."""
    base = root / AI_REPORTS_DIR
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name[:1].isdigit())


# --- las dos formas del resumen del dia, y por que hay dos ---------------------------
#
# `_resumen.json` lo escribe la tarea externa, cuyo prompt vive FUERA del repo. El
# 2026-08-22 cambio de forma sin avisar: `activos` paso de ser la lista de filas por activo
# a ser el RECUENTO (un entero), las filas se mudaron a `tickers` (un dict, no una lista) y
# los dos bloques agregados se renombraron. El lector, escrito el dia anterior contra la
# forma vieja, reventaba con `object of type 'int' has no len()` -- y con el se caia el
# generador del dashboard, que lo lee.
#
# La leccion no es "poner un try": es que un lector de la frontera publica NO PUEDE dar por
# hecha una forma que no controla. Lo que se hace aqui es normalizar las dos a la MISMA
# estructura, con los nombres nuevos primero, para que el dia que aparezca una tercera se
# vea de un vistazo donde se anade. Lo que la forma nueva no publica se queda en None y la
# vista lo pinta como "—"; solo se deriva lo que es aritmetica indiscutible sobre las filas
# que si estan (minimo y maximo de una lista de coberturas), y nunca un estadistico cuya
# definicion habria que adivinar.
_ROW_ALIASES = {
    "mean": ("media",),
    "reading": ("interpretacion",),
    "coverage_pct": ("cobertura_sumables_pct", "cobertura_pct"),
    "p30": ("sesgo_p30", "p30"),
    "rank": ("rank_media", "rango_media"),
    "pct_mean": ("percentil_seccion_transversal", "percentil_media"),
    "pct_coverage": ("percentil_cobertura",),
    "anchor_usd": ("ancla_precio_usd",),
    "anchor_ts": ("ancla_ts_utc",),
}


def _pick(row: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in row:
            return row[key]
    return None


def _summary_rows(summary: dict) -> list[dict]:
    """Las filas por activo, vengan como lista con `ticker` dentro o como dict por ticker."""
    tickers = summary.get("tickers")
    if isinstance(tickers, dict):
        raw = [{"ticker": t, **v} for t, v in tickers.items() if isinstance(v, dict)]
    else:
        listed = summary.get("activos")
        raw = [r for r in listed if isinstance(r, dict)] if isinstance(listed, list) else []
    return [
        {"ticker": row.get("ticker"), **{k: _pick(row, keys) for k, keys in _ROW_ALIASES.items()}}
        for row in raw
    ]


def _summary_aggregate(summary: dict, rows: list[dict]) -> dict:
    block = summary.get("sesgo_agregado") or summary.get("agregado") or {}
    coverages = [r["coverage_pct"] for r in rows if isinstance(r["coverage_pct"], (int, float))]
    return {
        "mean_of_means": block.get("media_de_medias"),
        "spread_of_means": block.get("desviacion_medias"),
        "coverage_mean_pct": block.get("cobertura_media_pct"),
        # Derivados de las filas solo si el resumen no los trae. Min y max de una lista no
        # admiten dos definiciones; la desviacion de arriba si, y por eso esa no se deriva.
        "coverage_min_pct": block.get("cobertura_min_pct", min(coverages) if coverages else None),
        "coverage_max_pct": block.get("cobertura_max_pct", max(coverages) if coverages else None),
        "p30_split": block.get("p30_distribucion") or block.get("reparto_p30") or {},
    }


def load_last_run(root: Path) -> dict | None:
    """La ultima ejecucion con `_resumen.json`, o `None` si todavia no hay ninguna.

    OJO AL USARLO EN UN ARTEFACTO COMMITEADO: esto sale de `data/signals_raw/`, que esta
    fuera de git y crece cada mañana. Cambia solo con que pase un dia, igual que el bloque
    de paper trading en vivo, y por eso la caracterizacion lo enmascara en vez de
    congelarlo. Ver `tests/golden_support.py`.
    """
    dates = run_dates(root)
    for date in reversed(dates):
        day = root / AI_REPORTS_DIR / date
        summary = _read_json(day / SUMMARY_FILE)
        if not summary:
            continue

        rows = _summary_rows(summary)
        aggregate = _summary_aggregate(summary, rows)
        diagnostic = (
            summary.get("correlacion_media_cobertura")
            or summary.get("diagnostico_cobertura_vs_media")
            or {}
        )
        labels = next(day.glob(f"{LABELS_PREFIX}*.json"), None)

        measures_dir = day / MEASURES_DIR
        reports = _canonical_tickers(day.glob("reporte_*.html"))
        answers = _canonical_tickers(day.glob("respuestas_*.json"))
        measures = (
            _canonical_tickers(measures_dir.glob("medidas_*.json"))
            if measures_dir.is_dir()
            else set()
        )
        all_files = (
            len(list(day.glob("reporte_*.html")))
            + len(list(day.glob("respuestas_*.json")))
            + (len(list(measures_dir.glob("medidas_*.json"))) if measures_dir.is_dir() else 0)
        )

        return {
            "date": date,
            "cutoff_utc": summary.get("hora_corte_utc"),
            "questionnaire": summary.get("cuestionario"),
            # `activos` es un entero en la forma nueva y la lista de filas en la vieja: solo
            # cuenta como recuento cuando de verdad es un numero. El ultimo recurso son los
            # FICHEROS del dia, que estan ahi aunque el resumen sea ilegible entero.
            "n_assets": (
                summary.get("n_activos")
                or (summary["activos"] if isinstance(summary.get("activos"), int) else 0)
                or len(rows)
                or len(answers)
            ),
            "n_days_captured": len(dates),
            "first_date": dates[0],
            # Los tres artefactos por activo. Que las cifras coincidan es la comprobacion
            # barata de que ningun activo se quedo a medias: medida sin reporte significa
            # que el agente midio y fallo al narrar.
            "n_reports": len(reports),
            "n_answers": len(answers),
            "n_measures": len(measures),
            # Los que tienen el trio entero, que es la unidad util: una respuesta sin su
            # medida no es reconstruible y una medida sin respuesta no entra en la tabla.
            "n_complete": len(reports & answers & measures),
            "incomplete": sorted((reports | answers | measures) - (reports & answers & measures)),
            # Ficheros de mas sobre el canonico: reejecuciones (`_v2`) y versiones
            # archivadas (`_vOld`). No son basura --la regla es no sobrescribir jamas--
            # pero contarlos como si fueran activos inflaria el dia por dos.
            "n_extra_files": all_files - len(reports) - len(answers) - len(measures),
            "has_labels": labels is not None,
            **aggregate,
            # El sesgo a vigilar: si la cobertura correlaciona con la media, el ranking del
            # dia esta midiendo CUANTO se pudo medir de cada activo y no cuanto favorable es.
            "coverage_bias_pearson": diagnostic.get("pearson"),
            "coverage_bias_spearman": diagnostic.get("spearman"),
            "rows": rows,
        }
    return None
