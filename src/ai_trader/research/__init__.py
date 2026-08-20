"""
INVESTIGACION APARCADA: el mundo sintetico y los estudios que lo usaban de sustrato.

NO SE TRABAJA AQUI salvo peticion explicita. Si estas leyendo esto porque una tarea te
trajo, comprueba primero que la tarea nombra `research/`: casi todo lo que parece vivir
aqui tiene su version viva fuera.

QUE ES ESTO
-----------
Durante meses, la apuesta del proyecto fue generar mundos sinteticos con los que rankear
estrategias: un mundo generado da distribuciones en vez de un unico camino historico, y
cubre regimenes que la historia no dio. La linea se llevo hasta el final y se midio:

- FIDELIDAD, conseguida. La libreria `ai_v3` acepta los nueve umbrales de hechos
  estilizados contra Binance (cobertura 35% -> 98%).
- TRANSFERENCIA, fallida. El Spearman entre el ranking real y el sintetico es -0,04 sobre
  16 configuraciones (IC95% por bloques [-0,44, +0,49], p = 0,89) y NEGATIVO (-0,67) sobre
  las nueve que operan de verdad en los dos mundos. La regla de aceptacion estaba escrita
  en el codigo ANTES de mirar (`transfer_study.RHO_ACCEPT = 0.30`).

Fidelidad no es transferencia: un mundo puede tener las colas, el agrupamiento de
volatilidad y la estructura de correlaciones del mercado y aun asi ordenar las estrategias
al reves. Eso es lo que se aprendio, y por eso el sintetico dejo de ser criterio de
seleccion. El sustrato que decide es ahora el historico real (`scoring/real_source.py`).

QUE SIGUE VIVO FUERA DE AQUI, Y NO SE DUPLICA
----------------------------------------------
Antes de aparcar la linea se saco lo que era del lado REAL y estaba dentro por accidente
historico -- porque estos estudios fueron los primeros que lo necesitaron:

    data/real_history.py        barras reales, cache offline, ventana cerrada
    scoring/real_substrate.py   sub-ventanas, universo cripto, auditoria de simbolos
    scoring/real_source.py      el sustrato del optimizador
    scoring/families.py         la rejilla de 64 configuraciones y su semilla
    scoring/signal_gate.py      con que parametro y valor se arma la puerta de senal

Los estudios de aqui importan de esos modulos; nunca al reves. La regla, medida y no
supuesta: de todo `src/ai_trader/` el UNICO modulo fuera de esta carpeta que importa
`ai_trader.research` es `cli.py`, y solo para el subcomando `synth`, que tambien esta
archivado. Ni `scoring/`, ni `backtest/`, ni `strategies/`, ni `app/` tocan nada de aqui:
el sistema que opera y el que puntua funcionan con esta carpeta entera borrada.

Fuera del paquete tambien la importan `dashboard/` y `docs/`, que leen los informes
publicados para el capitulo de investigacion archivada, y los tests de `tests/research/`.

QUE NO SE BORRA, Y POR QUE
---------------------------
Nada. Lo medido no se tira: la evidencia esta publicada en `data/fidelity/`,
`data/transfer/`, `data/calibration/`, `data/validation/`, `data/signal_channel/` y
`data/activity/`, y el dashboard y la metodologia la conservan en su capitulo de
investigacion archivada. Un resultado negativo caro es exactamente el que no hay que
repetir por haberlo borrado.

COMO SE REACTIVA
----------------
Los comandos siguen funcionando, con la ruta nueva:

    .venv\\Scripts\\python.exe -m ai_trader.research.fidelity_study --offline
    .venv\\Scripts\\python.exe -m ai_trader.research.transfer_study --offline
    .venv\\Scripts\\python.exe -m ai_trader.cli synth list

Y para volver a puntuar sobre una libreria generada hay que pedirlo explicitamente:

    from ai_trader.research.synthetic_source import SyntheticSampleSource
    run_optimization("crypto_momentum", source=SyntheticSampleSource.build())
"""
