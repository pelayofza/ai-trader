# ai-trader — instrucciones para Claude Code

## Regla 1: `src/ai_trader/research/` y `tests/research/` están APARCADOS

**No trabajes ahí salvo que la petición los nombre explícitamente.** No los leas "por
contexto", no propongas mejoras sobre ellos, no los incluyas en refactors generales y no
los cuentes al describir el estado del proyecto.

Qué contienen: el generador de datos sintéticos y los seis estudios que lo usaban como
sustrato (fidelidad, transferencia, validación, pesos, canal de señal, actividad). La línea
se llevó hasta el final y se midió: **la fidelidad se consiguió** (cobertura 35% → 98%, los
nueve umbrales aceptados) y **la transferencia falló** (ρ de Spearman entre el ranking real
y el sintético = −0,04; −0,67 entre las que operan de verdad). Fidelidad no es
transferencia, así que el sintético dejó de ser criterio de selección.

**No se borra nada.** Un resultado negativo caro es exactamente el que no hay que repetir
por haberlo tirado. Está ahí, entero y funcionando, y el dashboard y la metodología lo
conservan en su capítulo de investigación archivada.

Lo mismo aplica a estos datos, que son sus informes publicados: `data/synthetic/`,
`data/fidelity/`, `data/transfer/`, `data/validation/`, `data/calibration/`,
`data/signal_channel/`, `data/activity/`.

El aislamiento está **comprobado, no prometido**: `tests/test_research_isolation.py`
verifica que ningún paquete vivo importa de ahí, y ejecuta el runner, el motor, el
optimizador y el CLI con `ai_trader.research` bloqueado. Si tocas esa frontera, ese test lo
dice.

## Regla 2: dónde se trabaja, y en qué orden

Las tres líneas activas, en orden de dependencia:

1. **Captura de datos reales** para robustecer las señales externas (`signals/`).
2. **Procesamiento y calidad** de esos datos: análisis individualizado, fuente por fuente.
3. **Generación de estrategias** a partir de ellos (`scoring/`, sustrato real).

El criterio de prioridad cambió el 2026-08-20 y conviene entenderlo antes de proponer
trabajo: antes mandaba la asimetría de coste del juez —"un juez malo contamina todo lo que
puntúe"—, y eso produjo un instrumento excelente y una herramienta que no opera nada. Ahora
manda **poner la herramienta a funcionar**, aceptando el riesgo de sobreajuste que trae. Es
mejor tener algo corriendo con sobreajuste, y atacarlo después con evidencia de calendario,
que seguir refinando el juez de un backtest que no decide nada.

Consecuencia práctica: **no propongas trabajo de rigor de backtesting** (afinar PBO/DSR,
re-calibrar pesos, más esquemas de validación) salvo que se pida. Se retiró del roadmap a
propósito.

## Regla 3: el entorno

- `poetry run` **está roto** (falla mudo, exit 1). Invoca `.venv\Scripts\python.exe`
  directamente para todo: `pytest`, `ruff`, `-m ai_trader.cli`.
- Verificación única: `.\scripts\verify.ps1` (ruff + suite + árbol limpio). Tarda ~22 min
  completa; `-Fast` salta los `veryslow`. **Córrela en background** y no toques el árbol
  mientras corre: su tercer paso compara `git status` antes y después.
- `dashboard/index.html` y `docs/metodologia.html` son **su propia referencia** en los tests
  de caracterización. Si cambias algo que los afecta, regenéralos
  (`-m dashboard.build_dashboard`, `-m docs.build_docs`) y commitea el resultado.
- `tools/` y `config/*.md|json` alimentan una tarea diaria externa (Claude Cowork) que
  escribe en `data/signals_raw/ai_reports/`. No son código del paquete. **Sus rutas son una
  frontera pública**: el prompt de esa tarea vive fuera del repo y no se entera de un
  renombrado, así que mover uno de esos ficheros rompe la ejecución de la mañana siguiente
  y el día perdido no se recupera —la captura es point-in-time—. `verify.ps1` tiene un paso
  para ellos (`contrato`) y `tests/test_ai_reports_contract.py` comprueba el contrato.

## Regla 4: cómo se hacen los cambios aquí

Protocolo ultraconservador: **"prefiero el 10% de la deuda eliminada y cero regresiones que
el 80% con un bug silencioso"**. Ante la duda, NO tocar y preguntar — no hay penalización
por preguntar de más.

- Un cambio conceptual = un commit. Verificación después de cada uno.
- Cálculo numérico, estadística, semillas, métricas o backtesting: nunca son un cambio
  "mecánico". Pregunta antes.
- Cruzar una frontera pública (CLI, formato de fichero, esquema de datos, claves de config):
  no se toca sin acuerdo explícito.
- Antes de unificar dos cosas que parecen iguales, **demuéstralo** (ejecuta ambas y compara),
  no lo razones.
- Si un golden cambia aunque sea en el último decimal y no era el objetivo: `git revert`.

## Lo que el proyecto NO promete

El backtest no puede probar que algo funciona, sólo descartar. Aprobar el gate significa
haber batido a comprar y aguantar **en la cola mala** y haber operado lo suficiente para que
el score signifique algo — no es una promesa de rentabilidad. La única evidencia que no se
puede sobreajustar es el diario del paper trading, y ésa se compra con tiempo de calendario.
