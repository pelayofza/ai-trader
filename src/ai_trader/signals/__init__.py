"""
Plataforma de ingesta de SENALES EXTERNAS: de la declaracion al espacio de observacion.

Nueve piezas, y el orden en que se leen es el camino que recorre un dato:

    catalog.py   Que fuentes existen, que producen y con que honestidad (`history_from`
                 MEDIDO, `pit`). Es una lista de declaraciones: no toca red.
    source.py    El puerto: `fetch_raw` (red, payload intacto) + `daily_from_raw` (PURA).
    adapters/    Los diecisiete adaptadores: once continuos y seis de evento.
    store.py     Archivo crudo append-only en `data/signals_raw/` (no re-derivable) y
                 cache derivada desechable en `.cache/signals/`.
    capture.py   Recorre el catalogo y archiva. Se arranca YA: lo que no se capture hoy
                 de una fuente `forward_capture` no existira nunca.
    depth.py     MIDE desde cuando hay dato de verdad. Es lo unico que da derecho a
                 escribir un `history_from` en el catalogo.
    normalize.py Las dos varas de las series CONTINUAS: z contra la propia historia
                 (expansiva, causal) y z contra la seccion cruzada del dia.
    events.py    La codificacion de las series de EVENTO, que las dos z no pueden tocar:
                 dias-al-evento acotado, magnitud sobre su escala y ventana activa.
    feed.py      Del archivo a frames diarios. Es el unico camino por el que esto llega
                 al espacio de observacion.
    audit.py     Mide: cobertura del mapeo simbolo->entidad y del archivo por fuente,
                 entidad y ano.

DONDE ACABA ESTE PAQUETE. Aqui no hay ninguna estrategia ni ningun umbral de decision: lo
que consume estos frames es `observation/signal_radar.py`, que es quien los convierte en
las seis features que ve una decision. La dependencia va en ese sentido y solo en ese —el
radar importa de `signals/`, nunca al reves— para que la ingesta se pueda usar, medir y
publicar sin arrastrar nada de la capa de decision.

El esquema canonico `(entity, day) -> features + observed` vive en `shared/signals.py`, y
la resolucion simbolo -> entidad en `shared/entities.py`, porque los dos son vocabulario
comun y no pertenecen a la ingesta.
"""
