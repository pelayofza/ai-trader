"""
Plataforma de ingesta de SENALES EXTERNAS: el esqueleto para que la fuente N+1 sea barata.

Cinco piezas, y ninguna sabe nada de estrategias ni del runner (esto NO esta cableado):

    catalog.py   Que fuentes existen, que producen y con que honestidad (`history_from`
                 MEDIDO, `pit`). Es una lista de declaraciones: no toca red.
    source.py    El puerto: `fetch_raw` (red, payload intacto) + `daily_from_raw` (PURA).
                 El registro de adaptadores arranca VACIO.
    store.py     Archivo crudo append-only en `data/signals_raw/` (no re-derivable) y
                 cache derivada desechable en `.cache/signals/`.
    capture.py   Recorre el catalogo y archiva. Se arranca YA: lo que no se capture hoy
                 de una fuente `forward_capture` no existira nunca.
    audit.py     Mide: cobertura del mapeo simbolo->entidad y del archivo por fuente,
                 entidad y ano.

El esquema canonico `(entity, day) -> features + observed` vive en `shared/signals.py`, y
la resolucion simbolo -> entidad en `shared/entities.py`, porque los dos son vocabulario
comun y no pertenecen a la ingesta.
"""
