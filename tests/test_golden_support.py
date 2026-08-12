"""
Tests del propio andamiaje de caracterizacion.

No es meta-testing gratuito: la primera version de `scrub()` sustituia el contador de
commits por su literal ("81"), y eso reescribia las series de precios del dashboard
(`81.32` -> `<NCOMMITS>.32`). Un scrubber que se come datos reales convierte los golden
en una red con agujeros justo donde importa: enmascara el cambio en vez de detectarlo.
"""
from __future__ import annotations

from golden_support import scrub


def test_scrub_no_toca_cifras_que_coincidan_con_el_contador_de_commits() -> None:
    """El fallo original, fijado: un numero suelto no es metadata."""
    payload = '{"series": [89.7, 81.32, 85.46, 81.78], "commit_count": "81"}'
    out = scrub(payload)

    assert '"commit_count": "<NCOMMITS>"' in out
    assert "81.32" in out, "el scrubber ha corrompido una serie de precios"
    assert "81.78" in out


def test_scrub_sustituye_la_metadata_del_commit() -> None:
    payload = '{"commit": "abc1234", "commit_count": "81", "generated_at": "2026-08-11"}'
    out = scrub(payload)

    assert out == '{"commit": "<COMMIT>", "commit_count": "<NCOMMITS>", "generated_at": "<GITDATE>"}'


def test_scrub_sustituye_marcas_de_tiempo() -> None:
    assert scrub('{"generated_at": "2026-08-12T12:31:15.007757+00:00"}') == (
        '{"generated_at": "<TS>"}'
    )
    assert scrub("2026-08-12 14:31:27,486 | INFO") == "<TS> | INFO"


def test_scrub_sustituye_el_recuento_de_tests() -> None:
    """Anadir un test cambia el artefacto sin cambiar el comportamiento."""
    assert scrub("<b>777 tests</b>") == "<b><NTESTS> tests</b>"


def test_scrub_neutraliza_la_metadata_en_prosa_de_docs() -> None:
    """El HTML commiteado —que es la referencia— lleva el hash de un commit ANTERIOR.

    Por eso va por patron y no por el valor de HEAD: si solo se sustituyese el hash
    actual, el lado de la referencia se quedaria con el suyo y el test fallaria en cada
    commit sin que nada hubiera cambiado."""
    viejo = ('generado desde el commit <span class="mono">eb661a1</span>\n'
             "(81 commits)2026-08-11 · se regenera")
    nuevo = ('generado desde el commit <span class="mono">33af4f6</span>\n'
             "(82 commits)2026-08-12 · se regenera")

    assert scrub(viejo) == scrub(nuevo)
    assert "<COMMIT>" in scrub(viejo)
    assert "(<NCOMMITS> commits)<GITDATE>" in scrub(viejo)


def test_scrub_no_toca_un_span_mono_que_no_sea_un_hash() -> None:
    """`<span class="mono">` tambien envuelve comandos; solo el hex de 7+ es metadata."""
    payload = '<span class="mono">python -m docs.build_docs</span>'
    assert scrub(payload) == payload


def test_scrub_ignora_un_hash_todo_numerico(monkeypatch) -> None:
    """Uno de cada veintisiete hashes cortos sale sin ninguna letra.

    Con uno asi, el reemplazo literal repetiria el error del contador de commits y se
    comeria cualquier cifra que lo contuviera. Los patrones anclados ya cubren ese caso,
    asi que el literal se salta."""
    import golden_support

    monkeypatch.setattr(golden_support, "_git", lambda *a: "1234567")

    payload = '{"volume": 91234567.0, "commit": "1234567"}'
    out = scrub(payload)

    assert "91234567.0" in out, "el scrubber ha corrompido una cifra"
    assert '"commit": "<COMMIT>"' in out, "la metadata anclada si debe sustituirse"


def test_scrub_normaliza_la_ruta_del_repo_y_los_finales_de_linea() -> None:
    from golden_support import REPO_ROOT

    out = scrub(f"root={REPO_ROOT}\r\nposix={str(REPO_ROOT).replace(chr(92), '/')}\r\n")
    assert out == "root=<ROOT>\nposix=<ROOT>\n"


def test_scrub_deja_intacto_lo_que_no_es_volatil() -> None:
    """Lo importante del contrato: todo lo demas debe pasar sin tocar."""
    payload = "Sharpe/Sortino:-2.80 / -3.33   Calmar: -1.842   Trades: 41   #aabbcc"
    assert scrub(payload) == payload
