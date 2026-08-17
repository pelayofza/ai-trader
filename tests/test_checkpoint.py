"""
El punto de guardado de los barridos largos.

Lo que hay que demostrar no es que el fichero se escriba: es que REANUDAR da exactamente el
mismo resultado que no haber parado nunca. Sin eso, un estudio pausado publicaria un informe
que nadie puede reproducir de una sentada, y eso no es evidencia.
"""
from __future__ import annotations

import json

import pytest

from ai_trader.shared.checkpoint import UnitCheckpoint, fingerprint


def rows_for(keys):
    return [{"key": k, "row": {"id": "/".join(map(str, k)), "score": float(i)}}
            for i, k in enumerate(keys)]


KEYS = [("cfg0", "off", "sc0", 0), ("cfg1", "off", "sc0", 0), ("cfg0", "on", "sc1", 1)]


class TestReanudarEsExacto:
    def test_lo_guardado_vuelve_igual(self, tmp_path):
        cp = UnitCheckpoint(tmp_path / "p.jsonl", "huella")
        for entry in rows_for(KEYS):
            cp.append(entry["key"], entry["row"])
        cp.close()

        done = UnitCheckpoint(tmp_path / "p.jsonl", "huella").load()

        assert set(done) == {tuple(k) for k in KEYS}
        assert done[("cfg1", "off", "sc0", 0)]["score"] == 1.0

    def test_una_pausa_no_cambia_el_conjunto_final(self, tmp_path):
        """Correr 3 de un tiron y correr 1 + pausa + 2 tienen que dar lo MISMO."""
        entero = UnitCheckpoint(tmp_path / "a.jsonl", "h")
        for e in rows_for(KEYS):
            entero.append(e["key"], e["row"])
        entero.close()

        tramo1 = UnitCheckpoint(tmp_path / "b.jsonl", "h")
        tramo1.append(rows_for(KEYS)[0]["key"], rows_for(KEYS)[0]["row"])
        tramo1.close()  # <- aqui se apaga el ordenador

        reanudado = UnitCheckpoint(tmp_path / "b.jsonl", "h")
        done = reanudado.load()
        pendientes = [e for e in rows_for(KEYS) if tuple(e["key"]) not in done]
        assert len(pendientes) == 2, "solo deberia quedar lo que no se corrio"
        for e in pendientes:
            reanudado.append(e["key"], e["row"])
        reanudado.close()

        assert (UnitCheckpoint(tmp_path / "a.jsonl", "h").load()
                == UnitCheckpoint(tmp_path / "b.jsonl", "h").load())


class TestLaHuellaEsLaGuarda:
    def test_otro_plan_no_se_reutiliza(self, tmp_path):
        """Mezclar dos planes daria un informe que no es de ninguno de los dos."""
        cp = UnitCheckpoint(tmp_path / "p.jsonl", "plan_viejo")
        cp.append(KEYS[0], {"score": 1.0})
        cp.close()

        done = UnitCheckpoint(tmp_path / "p.jsonl", "plan_nuevo").load()

        assert done == {}

    def test_lo_apartado_no_se_borra(self, tmp_path):
        """Si la huella cambio sin querer, lo corrido tiene que seguir ahi para mirarlo."""
        cp = UnitCheckpoint(tmp_path / "p.jsonl", "viejo")
        cp.append(KEYS[0], {"score": 1.0})
        cp.close()

        UnitCheckpoint(tmp_path / "p.jsonl", "nuevo").load()

        stale = tmp_path / "p.jsonl.stale"
        assert stale.exists()
        assert "score" in stale.read_text(encoding="utf-8")

    def test_la_huella_cambia_si_cambia_el_plan(self):
        assert fingerprint({"a": 1}, ["x"]) == fingerprint({"a": 1}, ["x"])
        assert fingerprint({"a": 1}, ["x"]) != fingerprint({"a": 2}, ["x"])
        assert fingerprint({"a": 1}, ["x"]) != fingerprint({"a": 1}, ["y"])

    def test_el_orden_de_las_claves_no_mueve_la_huella(self):
        assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


class TestUnaMuerteAMediaEscritura:
    def test_la_ultima_linea_rota_se_descarta_y_el_resto_se_conserva(self, tmp_path):
        """Es toda la razon de elegir JSONL: un JSON con corchetes habria que cerrarlo."""
        path = tmp_path / "p.jsonl"
        cp = UnitCheckpoint(path, "h")
        for e in rows_for(KEYS):
            cp.append(e["key"], e["row"])
        cp.close()
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"key": ["cfg9", "off", "sc9"')  # el proceso muere aqui

        done = UnitCheckpoint(path, "h").load()

        assert len(done) == len(KEYS)

    def test_un_fichero_que_no_existe_es_empezar_de_cero(self, tmp_path):
        assert UnitCheckpoint(tmp_path / "no_esta.jsonl", "h").load() == {}

    def test_un_fichero_vacio_no_revienta(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text("", encoding="utf-8")
        assert UnitCheckpoint(path, "h").load() == {}

    def test_una_cabecera_ilegible_no_revienta(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text("esto no es json\n", encoding="utf-8")
        assert UnitCheckpoint(path, "h").load() == {}


class TestCicloDeVida:
    def test_al_terminar_se_borra(self, tmp_path):
        """Dejarlo invita a reanudar un barrido que ya publico sus unidades."""
        path = tmp_path / "p.jsonl"
        cp = UnitCheckpoint(path, "h")
        cp.append(KEYS[0], {"score": 1.0})
        cp.discard()
        assert not path.exists()

    def test_la_cabecera_se_escribe_una_sola_vez(self, tmp_path):
        path = tmp_path / "p.jsonl"
        cp = UnitCheckpoint(path, "h")
        for e in rows_for(KEYS):
            cp.append(e["key"], e["row"])
        cp.close()
        cp2 = UnitCheckpoint(path, "h")
        cp2.append(("cfg2", "off", "sc2", 0), {"score": 9.0})
        cp2.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        headers = [ln for ln in lines if "fingerprint" in ln]
        assert len(headers) == 1
        assert json.loads(lines[0])["fingerprint"] == "h"
        assert len(UnitCheckpoint(path, "h").load()) == len(KEYS) + 1


@pytest.mark.parametrize("key", [("a", "b", "c", 0), ("a", "b", "c", 12)])
def test_la_clave_sobrevive_al_viaje_por_json(tmp_path, key):
    """La clave lleva un entero y JSON no tiene tuplas: si no se restaura igual, una unidad
    ya corrida se volveria a correr y la reanudacion no ahorraria nada."""
    cp = UnitCheckpoint(tmp_path / "p.jsonl", "h")
    cp.append(key, {"score": 1.0})
    cp.close()
    assert key in UnitCheckpoint(tmp_path / "p.jsonl", "h").load()
