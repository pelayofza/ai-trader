from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ai_trader.config import DEFAULT_CONFIG_PATH, load_config
from ai_trader.data.market_data import MarketDataService
from ai_trader.main import build_runner
from ai_trader.notifications.base import NullNotifier
# Las claves de los enriquecedores se importan arriba, y no de forma perezosa como el resto
# de `synth`, porque son las `choices` de un argumento: el parser las necesita al
# construirse. `observation_worlds` solo importa de `synthetic/`, asi que no arrastra nada.
from ai_trader.synthetic.observation_worlds import ENRICHERS as _ENRICHERS

ENRICHER_KEYS: tuple[str, ...] = tuple(_ENRICHERS)

logger = logging.getLogger(__name__)


def _build(config_path: Path):
    config = load_config(config_path)
    service = MarketDataService()
    return build_runner(config, service, NullNotifier())


def cmd_run_cycle(args: argparse.Namespace) -> int:
    runner = _build(args.config)
    results = runner.run_cycle()

    print(f"Executions: {len(results)}")
    for result in results:
        print(f"  {result.status.value:>10} | {result.symbol} | {result.message}")

    print()
    print(runner.get_performance_report())
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    import json

    from ai_trader.backtest.engine import BacktestEngine, parse_date
    from ai_trader.backtest.validation import SCHEME_SINGLE

    config = load_config(args.config)

    if args.synthetic:
        bars, start, end = _synthetic_bars(config, args)
    else:
        if not (args.start and args.end):
            print("--start and --end are required unless --synthetic is given.")
            return 2
        start, end = parse_date(args.start), parse_date(args.end)
        bars = _real_bars(config, start, end)

    if args.validation != SCHEME_SINGLE:
        return _run_multiwindow(config, bars, start, end, args)

    engine = BacktestEngine.from_bars(config, bars, starting_equity=args.capital)
    result = engine.run(
        start=start,
        end=end,
        split_ratio=args.split,
        split_date=parse_date(args.split_date) if args.split_date else None,
    )

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    _print_window("TRAIN (in-sample)", result.train)
    _print_window("TEST (out-of-sample)", result.test)
    print("=" * 48)
    w = result.headline_weights
    print(
        f"HEADLINE SCORE (out-of-sample): {result.headline_score:+.4f}"
        f"   [Sharpe - {w.lambda_turnover}*turnover - {w.kappa_maxdd}*maxDD]"
    )
    print(
        "Aviso: corte unico, una sola ventana OOS. Para la distribucion multiventana "
        "con purga y embargo usa --validation walk_forward|cpcv."
    )
    return 0


def _real_bars(config, start, end) -> dict:
    """Historico real precargado UNA vez, con calentamiento para que el primer dia de la
    primera ventana ya tenga lookback completo. Es la misma precarga que hacia el motor
    por dentro; se sube aqui para que el backtest de corte unico y la validacion
    multiventana partan de EXACTAMENTE las mismas barras."""
    from datetime import timedelta

    from ai_trader.data.backtest_source import HistoricalDataSource

    warmup = timedelta(days=config.runner.lookback_days + 30)
    return HistoricalDataSource.fetch_bars(
        MarketDataService(), config.runner.symbols, start - warmup, end
    )


def _synthetic_bars(config, args):
    """Muestra sintetica almacenada: LIBRERIA:ESCENARIO:PATH.

    La ventana se deriva del manifiesto dejando calentamiento para el lookback."""
    from ai_trader.synthetic.service import sample_window
    from ai_trader.synthetic.store import SyntheticStore

    parts = args.synthetic.split(":")
    if len(parts) != 3:
        raise SystemExit("--synthetic expects LIBRARY:SCENARIO:PATH (e.g. lib1:calm_bull:0)")
    library_id, scenario_id, path_index = parts[0], parts[1], int(parts[2])

    store = SyntheticStore(args.synthetic_root) if args.synthetic_root else SyntheticStore()
    manifest = store.load_manifest(library_id)
    bars = store.load_bars(library_id, scenario_id, path_index)

    start, end = sample_window(manifest, warmup_days=config.runner.lookback_days + 30)
    logger.info("Synthetic backtest | %s / %s / path %s", library_id, scenario_id, path_index)
    return bars, start, end


def _run_multiwindow(config, bars, start, end, args: argparse.Namespace) -> int:
    """Validacion multiventana: varias ventanas OOS con purga y embargo, agregadas en
    una distribucion robusta en vez de en un unico numero."""
    import json

    from ai_trader.scoring.activity import DEFAULT_ACTIVITY_FLOOR
    from ai_trader.scoring.multiwindow import validate_multiwindow

    result = validate_multiwindow(
        config, None, bars, start, end,
        scheme=args.validation,
        n_folds=args.folds,
        n_groups=args.groups,
        n_test_groups=args.test_groups,
        purge_days=args.purge,
        embargo_days=args.embargo,
        starting_equity=args.capital,
        run_train=args.with_train,
    )

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    s = result.stats
    print(f"=== VALIDACION {result.scheme.upper()} | {len(result.folds)} ventanas "
          f"| purga {result.purge_days}d | embargo {result.embargo_days}d ===")
    print(f"  Fuga temporal auditada: {'sin fuga' if result.leakage_ok else 'CON FUGA'}")
    print(f"  Calendario cubierto: {result.coverage['calendar_days_covered']} dias "
          f"(reuso x{result.coverage['reuse_factor']})")
    print()
    print(f"  {'ventana':<12} {'desde':<11} {'hasta':<11} {'tramos':>6} {'trades':>7} "
          f"{'sharpe':>8} {'headline':>10}")
    for fold in result.folds:
        print(f"  {fold.label:<12} {fold.test_start.date()!s:<11} {fold.test_end.date()!s:<11} "
              f"{fold.n_test_blocks:>6} {fold.num_trades:>7} {fold.sharpe:>8.2f} "
              f"{fold.score:>+10.4f}")
    print()
    print("=" * 72)
    print(f"  RECOMPENSA (CVaR@{s.alpha:.0%} de las ventanas): {s.reward:+.4f}")
    print(f"  mediana {result.median:+.4f} | media {s.mean:+.4f} | std {s.std:.4f} "
          f"| peor {s.worst:+.4f} | mejor {s.best:+.4f}")
    # La actividad va PEGADA a la recompensa: un CVaR de 0 puede ser "no perdio" o "no
    # jugo", y solo esta linea lo distingue. Ver ai_trader.scoring.activity.
    a = result.activity
    if a is not None:
        floor = DEFAULT_ACTIVITY_FLOOR
        print(f"  ACTIVIDAD: {a.trades_per_window:.1f} operaciones por ventana "
              f"(mediana {a.median_trades_per_window:.0f}) | "
              f"{a.zero_windows}/{a.n_windows} ventanas vacias ({a.zero_window_pct:.0f}%)")
        print(f"  Rankeable: {'SI' if result.rankable else 'NO'} "
              f"(suelo: >= {floor.min_median_trades_per_window:.0f} ops en la ventana "
              f"mediana, <= {floor.max_zero_window_pct:.0f}% vacias)"
              + ("" if result.rankable
                 else " -> " + "; ".join(floor.reasons(a))))
    if result.single_split_score is not None:
        print(f"  Corte unico 70/30 de referencia: {result.single_split_score:+.4f} "
              f"-> optimismo {result.optimism:+.4f}")
    if result.baseline_gate is not None:
        g = result.baseline_gate
        print(f"  Gate: {'APROBADO' if g.approved else 'RECHAZADO'} "
              f"(mejor baseline {g.best_name} {g.best_reward:+.4f}, margen {g.margin:+.4f})")
        if g.beats_baselines and not g.eligible:
            print("        bate a los baselines pero NO es rankeable: batirlos sin operar "
                  "es batirlos por no jugar")
    return 0


def cmd_synth_generate(args: argparse.Namespace) -> int:
    from ai_trader.synthetic.designer import ClaudeScenarioDesigner, TemplateScenarioDesigner
    from ai_trader.synthetic.service import SyntheticDataService
    from ai_trader.synthetic.store import SyntheticStore

    designer = ClaudeScenarioDesigner() if args.ai else TemplateScenarioDesigner()
    store = SyntheticStore(args.synthetic_root) if args.synthetic_root else SyntheticStore()
    service = SyntheticDataService(designer, store=store)

    manifest = service.generate(
        args.library,
        n_scenarios=args.scenarios,
        n_paths=args.paths,
        horizon_days=args.horizon,
        seed_base=args.seed,
    )
    print(
        f"Generated library '{manifest.library_id}' "
        f"({manifest.num_scenarios} scenarios x {manifest.n_paths} paths = "
        f"{manifest.num_samples} samples) using {manifest.designer}."
    )
    print(f"Horizon: {manifest.horizon_days} days | anchor: {manifest.anchor[:10]}")
    return 0


def cmd_synth_add_paths(args: argparse.Namespace) -> int:
    from ai_trader.synthetic.designer import TemplateScenarioDesigner
    from ai_trader.synthetic.service import SyntheticDataService
    from ai_trader.synthetic.store import SyntheticStore

    store = SyntheticStore(args.synthetic_root) if args.synthetic_root else SyntheticStore()
    # El disenador no se usa al regenerar (los escenarios ya estan en disco); se pasa
    # uno cualquiera valido solo para construir el servicio.
    service = SyntheticDataService(TemplateScenarioDesigner(), store=store)

    manifest = service.resynthesize(args.library, n_paths=args.paths)
    print(
        f"Resynthesized '{manifest.library_id}' from stored scenarios (NO API call): "
        f"{manifest.num_scenarios} scenarios x {manifest.n_paths} paths = "
        f"{manifest.num_samples} samples."
    )
    return 0


def cmd_synth_derive(args: argparse.Namespace) -> int:
    """
    Deriva una libreria de otra aplicando un enriquecedor DECLARADO. No llama a la IA.

    Es el camino por el que se produjeron ai_v2 y ai_v3, y hasta hoy la unica de las tres
    operaciones del servicio sin puerta de entrada: se corrieron desde una consola que nadie
    puede repetir. `--enricher` toma sus opciones del registro, asi que `--help` lista los
    mundos derivables y esa lista no se puede desincronizar del codigo.
    """
    from ai_trader.synthetic.designer import TemplateScenarioDesigner
    from ai_trader.synthetic.observation_worlds import ENRICHERS
    from ai_trader.synthetic.service import SyntheticDataService
    from ai_trader.synthetic.store import SyntheticStore

    store = SyntheticStore(args.synthetic_root) if args.synthetic_root else SyntheticStore()
    # El disenador no se usa al derivar (los escenarios ya estan en disco); se pasa uno
    # cualquiera valido solo para construir el servicio, igual que en add-paths.
    service = SyntheticDataService(TemplateScenarioDesigner(), store=store)

    manifest = service.derive_library(
        args.source, args.target, enricher=ENRICHERS[args.enricher], n_paths=args.paths
    )
    print(
        f"Derived '{manifest.library_id}' from '{args.source}' via '{args.enricher}' "
        f"(NO API call): {manifest.num_scenarios} scenarios x {manifest.n_paths} paths = "
        f"{manifest.num_samples} samples."
    )
    print(f"Designer chain: {manifest.designer}")
    return 0


def cmd_synth_list(args: argparse.Namespace) -> int:
    from ai_trader.synthetic.store import SyntheticStore

    store = SyntheticStore(args.synthetic_root) if args.synthetic_root else SyntheticStore()
    libraries = store.list_libraries()
    if not libraries:
        print("No synthetic libraries found.")
        return 0

    for library_id in libraries:
        m = store.load_manifest(library_id)
        print(f"=== {library_id} | {m.num_samples} samples | {m.designer} | {m.created_at[:10]} ===")
        for sc in m.scenarios:
            print(f"  {sc['id']:<28} {sc['name']}")
        print()
    return 0


def cmd_signals_catalog(args: argparse.Namespace) -> int:
    """El catalogo de fuentes tal y como esta declarado. No toca red ni disco."""
    import json

    from ai_trader.signals.catalog import CATALOG, catalog_summary

    if args.json:
        print(json.dumps([s.as_dict() for s in CATALOG], indent=2))
        return 0

    summary = catalog_summary()
    print(f"=== CATALOGO DE SENALES | {summary['n_sources']} fuentes | "
          f"{summary['n_features']} features ===")
    print(f"  {'clave':<24} {'tier':<5} {'scope':<7} {'codificacion':<12} {'pit':<18} "
          f"{'historia':<14} {'ADV tipico':>13} feats")
    for source in CATALOG:
        history = source.history_from.isoformat() if source.history_from else "solo adelante"
        adv = f"{source.typical_adv_usd:,.0f}" if source.typical_adv_usd else "-"
        print(f"  {source.key:<24} {source.tier:<5} {source.scope:<7} "
              f"{source.encoding_kind:<12} {source.pit:<18} {history:<14} {adv:>13} "
              f"{len(source.features)}")
    print()
    print(f"  Backtesteables HOY: {summary['n_backtestable']}/{summary['n_sources']} "
          f"(el resto no tiene history_from MEDIDO: solo existen hacia adelante)")
    print(f"  Sin credencial: {summary['n_open']}/{summary['n_sources']}")
    print("  Codificacion: " + " | ".join(
        f"{kind} {count}" for kind, count in summary["by_encoding"].items()
    ))
    if summary["min_adv_usd"]:
        # LA CIFRA QUE HAY QUE VER ANTES DE ESCALAR, y por eso sale sola y en negativo: no
        # es "el ADV medio del catalogo" sino el de la fuente que menos admite.
        print(f"  ADV declarado en {summary['n_with_adv']} fuentes; la mas estrecha vive "
              f"en entidades de {summary['min_adv_usd']:,.0f} USD/dia")
    return 0


def cmd_signals_capture(args: argparse.Namespace) -> int:
    """Una pasada de captura. Se puede correr HOY: sin adaptadores archiva 0 y lo declara."""
    from ai_trader.signals.capture import capture

    config = load_config(args.config)
    report = capture(config.runner.symbols)

    print(f"=== CAPTURA | {report.n_sources} fuentes | {report.records} registros ===")
    for item in report.sources:
        state = "sin adaptador" if not item.connected else (item.error or f"{item.records} reg.")
        print(f"  {item.source_key:<24} {item.tier:<3} {len(item.entities):>3} ent.  {state}")
    print()
    print(f"  Conectadas {report.n_connected}/{report.n_sources} | "
          f"pendientes {report.n_pending} | fallidas {report.n_failed}")
    if report.n_connected == 0:
        print("  Aviso: aun no hay ningun adaptador escrito. El archivo empieza a tener "
              "fondo el dia que se registre el primero (signals.source.register_adapter).")
    return 0


def cmd_signals_audit(args: argparse.Namespace) -> int:
    """Cobertura MEDIDA: del mapeo simbolo->entidad y del archivo crudo."""
    import json

    from ai_trader.signals.audit import audit_archive, audit_entities

    config = load_config(args.config)
    entities = audit_entities(config.runner.symbols)
    archive = audit_archive()

    if args.json:
        print(json.dumps({"entities": entities.as_dict(), "archive": archive.as_dict()}, indent=2))
        return 0

    print(f"=== ENTIDADES | {entities.n_symbols} simbolos | "
          f"cobertura {entities.coverage_pct:.1f}% ===")
    counts = entities.by_source
    print(f"  regla {counts['rule']} | overrides {counts['override']} | "
          f"sin resolver {counts['unmapped']}")
    if entities.unmapped:
        print(f"  Sin resolver: {', '.join(entities.unmapped)}")
    for key, symbols in entities.collisions.items():
        print(f"  {key} <- {', '.join(symbols)}")
    print()

    print(f"=== ARCHIVO CRUDO | {archive.root} | {archive.records} registros ===")
    for source in archive.sources:
        if not source.years:
            continue
        for row in source.years:
            pct = "n/a" if row.coverage_pct is None else f"{row.coverage_pct:.0f}%"
            print(f"  {source.source_key:<24} {row.entity:<12} {row.year} "
                  f"{row.days:>4}d  cobertura {pct:>5}  hueco max {row.max_gap_days}d")
    if archive.records == 0:
        print("  Vacio: todavia no se ha capturado nada.")
    return 0


def cmd_signals_depth(args: argparse.Namespace) -> int:
    """MIDE la profundidad de cada fuente. Es lo unico que da derecho a escribir un
    `history_from` en el catalogo, y deja la evidencia en data/signals/history_depth.json."""
    import json

    from ai_trader.signals.catalog import get_source
    from ai_trader.signals.depth import declared_vs_measured, measure_depth

    config = load_config(args.config)
    sources = [get_source(args.source)] if args.source else None
    report = measure_depth(config.runner.symbols, sources=sources, from_archive=args.from_archive)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    print(f"=== PROFUNDIDAD MEDIDA | {report.n_measured}/{len(report.sources)} fuentes ===")
    for item in report.sources:
        state = item.error[:38] if item.error else f"{item.days:>6} dias, {len(item.entities)} ent."
        print(f"  {item.source_key:<24} {item.method:<9} {str(item.first_day):<12} {state}")

    mismatches = {
        key: row
        for key, row in declared_vs_measured().items()
        if row["matches"] is False or (row["declared"] and not row["measured"])
    }
    print()
    if mismatches:
        print("  DESAJUSTES entre lo declarado y lo medido:")
        for key, row in mismatches.items():
            print(f"    {key:<24} declarado {row['declared']} != medido {row['measured']}")
    else:
        print("  Lo declarado en el catalogo coincide con lo medido.")
    return 0


def cmd_signals_features(args: argparse.Namespace) -> int:
    """Deriva el archivo crudo y publica el panel NORMALIZADO. No toca red."""
    import json

    from ai_trader.signals.capture import connect_adapters
    from ai_trader.signals.catalog import CATALOG, get_source
    from ai_trader.signals.normalize import normalization_coverage, normalization_spec, panel
    from ai_trader.signals.source import build_adapter
    from ai_trader.signals.store import SignalStore

    connect_adapters()
    store = SignalStore()
    sources = [get_source(args.source)] if args.source else list(CATALOG)

    frames = {}
    for source in sources:
        adapter = build_adapter(source)
        records = store.read(source.key) if adapter is not None else []
        if adapter is None or not records:
            continue
        frames[source.key] = adapter.daily_from_raw(records)

    wide = panel(frames)
    coverage = normalization_coverage(wide)

    if args.json:
        print(json.dumps({"coverage": coverage, "spec": normalization_spec()}, indent=2))
        return 0

    print(f"=== PANEL NORMALIZADO | {len(frames)} fuentes derivadas ===")
    for key, frame in sorted(frames.items()):
        days = frame.index.get_level_values("day")
        span = f"{days.min().date()} -> {days.max().date()}" if len(frame) else "vacio"
        print(f"  {key:<24} {len(frame):>7} filas  {span}")
    print()
    print(f"  Panel: {coverage['rows']} filas x {coverage['columns']} columnas "
          f"({coverage.get('entities', 0)} entidades)")
    print(f"  Cobertura z propia: {coverage['self_pct']:.1f}%   "
          f"transversal: {coverage['cross_pct']:.1f}%")
    print(f"  Politica: mediana / IQR-1.349, recorte +-{normalization_spec()['clip']}, "
          f"huecos a NaN (no a 0)")
    return 0


def cmd_signals_events(args: argparse.Namespace) -> int:
    """
    Cuenta los EVENTOS POOLED por fuente y publica el recuento. No toca red.

    Es la cifra que sustituye a la creencia: "muestras de decenas" era una afirmacion que
    nadie habia medido, y esta orden es la que la mide. La unidad es el evento NORMALIZADO
    —% del float, % del ADV— y no el token, que es lo que permite ponerlos todos en la
    misma distribucion.
    """
    import json

    from ai_trader.observation.signal_radar import SIGNAL_FEATURES
    from ai_trader.observation.signal_themes import (
        THEME_NAMES,
        ThemedSignalRadarProvider,
        theme_reading,
    )
    from ai_trader.shared.clock import LiveClock
    from ai_trader.signals.events import pool_report, write_pool_report
    from ai_trader.signals.feed import load_frames

    config = load_config(args.config)
    frames = load_frames(raw_root=config.signals.raw_root or None)
    report = pool_report(frames)
    path = write_pool_report(report)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"=== EVENTOS POOLED | {report['n_event_sources']} fuentes de evento | "
          f"{report['pooled_events_total']} eventos ===")
    for key, row in report["sources"].items():
        window = (
            f"{row['first_day']} -> {row['last_day']}" if row["first_day"] else "sin dato"
        )
        print(f"  {key:<24} {row['pooled_events']:>6} eventos  {row['entities']:>3} ent.  "
              f"{'anunciado' if row['announced'] else 'sin aviso':<10} {window}")
    print()
    print(f"  Tope dias-al-evento: {report['spec']['days_ahead_cap']:.0f} | "
          f"ventana posterior: {report['spec']['days_active_cap']:.0f} | "
          f"recorte de magnitud: +-{report['spec']['magnitude_clip']:.0f}")

    maps = report.get("price_maps") or {}
    if maps.get("sources"):
        # Aparte del recuento pooled a proposito: la unidad de observacion de un mapa es la
        # FOTO DIARIA y no el evento, y sumarlos diria que hay mas eventos de los que hay.
        print()
        print(f"  --- mapas de precios ({maps['n_sources']} fuentes, "
              f"tope {report['spec']['price_map']['distance_cap_pct']:.0f}% de precio) ---")
        for key, row in maps["sources"].items():
            window = (
                f"{row['first_day']} -> {row['last_day']}" if row["first_day"] else "sin dato"
            )
            print(f"  {key:<24} {row['snapshots']:>6} fotos    {row['entities']:>3} ent.  "
                  f"{'':<10} {window}")
    print(f"  Recuento publicado en {path}")

    if args.symbol:
        radar = ThemedSignalRadarProvider(frames, LiveClock())
        features = radar.features(args.symbol)
        report = radar.coverage_report()
        print()
        print(f"=== RADAR | {args.symbol} ===")
        for name in SIGNAL_FEATURES:
            print(f"  {name:<26} {features[name]:+.3f}")
        print(f"  Fuentes: {len(report['asset_sources'])} por activo, "
              f"{len(report['market_sources'])} de mercado")
        print()
        print(f"=== TEMAS | {args.symbol} ===")
        # La columna que importa es la ultima: un tema por debajo del minimo no es un tema
        # neutro, es un tema del que hoy no se sabe nada, y su puerta NO se evalua.
        print(f"  {'tema':<14} {'tono':>7} {'intens.':>8} {'cobert.':>8}  {'fuentes':>9}  legible")
        for theme in THEME_NAMES:
            reading = theme_reading(features, theme)
            block = report["themes"][theme]
            sources = f"{len(block['loaded'])}/{block['denominator']}"
            print(
                f"  {theme:<14} {reading.tone:>+7.3f} {reading.intensity:>8.3f} "
                f"{reading.coverage:>8.3f}  {sources:>9}  {'si' if reading.readable else 'NO'}"
            )
    return 0


def cmd_signals_adv(args: argparse.Namespace) -> int:
    """El ADV de las entidades donde vive cada senal. Ver `signals/liquidity.py`.

    Es la operacion gemela de `signals depth` y con el mismo contrato: MIDE, escribe el
    registro y compara con lo que declara el catalogo. `--from-ledger` no toca red, que es
    lo que hace falta para auditar sin volver a pedir tres venues."""
    import json

    from ai_trader.signals.liquidity import (
        declared_vs_measured_adv,
        liquidity_summary,
        measure_liquidity,
    )

    if not args.from_ledger:
        measure_liquidity()

    summary = liquidity_summary()
    comparison = declared_vs_measured_adv()

    if args.json:
        print(json.dumps({"summary": summary, "declared_vs_measured": comparison}, indent=2))
        return 0

    print(f"=== ADV DE LAS ENTIDADES | {summary['n_venues']} venues medidos | "
          f"{summary['n_declared']} fuentes lo declaran ===")
    print(f"  {'venue':<14} {'entidades':>10} {'con volumen':>12} {'mediana 24h':>16} "
          f"{'decil inf.':>16} {'maximo':>18}")
    for name, row in summary["venues"].items():
        print(f"  {name:<14} {row['n_entities'] or 0:>10} {row['n_traded'] or 0:>12} "
              f"{row['median_usd'] or 0:>16,.0f} {row['p10_usd'] or 0:>16,.0f} "
              f"{row['max_usd'] or 0:>18,.0f}")
    print()
    for key, row in comparison.items():
        state = (
            "sin medir" if not row["measured_usd"]
            else "no declarado" if not row["declared_usd"]
            else "ok" if row["within_tolerance"]
            else "FUERA DE TOLERANCIA"
        )
        print(f"  {key:<24} declarado {str(row['declared_usd'] or '-'):>16} "
              f"medido {row['measured_usd'] or 0:>16,.0f}  {state}")
    print()
    print(f"  La fuente mas estrecha es '{summary['thinnest_source']}': vive en entidades de "
          f"{summary['thinnest_adv_usd'] or 0:,.0f} USD/dia")
    print(f"  Tolerancia del test: x{summary['tolerance_factor']:.0f} (un orden de magnitud: "
          f"la pregunta que contesta el campo es si cabe tamano, no cuanto exactamente)")
    return 0


def cmd_signals_dat(args: argparse.Namespace) -> int:
    """
    Compone la distribucion de mNAV de las tesorerias cotizadas. No toca red.

    Es la gemela de `signals events` y con el mismo contrato: lee el archivo crudo, deriva
    con el adaptador REAL y publica la cifra que sustituye a la creencia. Aqui esa cifra es
    el N —observaciones de companıa agrupadas sobre la cohorte, no eventos de una— y, al
    lado, con cuantas companias distintas se compuso: doscientas observaciones de tres
    companias y doscientas de cuarenta sostienen inferencias distintas.
    """
    import json

    from ai_trader.signals.adapters.treasuries import (
        cohort_report,
        declared_vs_measured_lag,
        write_cohort_report,
    )
    from ai_trader.signals.capture import connect_adapters
    from ai_trader.signals.store import SignalStore

    config = load_config(args.config)
    connect_adapters()
    records = SignalStore(config.signals.raw_root or None).read("dat_mnav")
    report = cohort_report(records)
    path = write_cohort_report(report)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"=== TESORERIAS COTIZADAS | {report['companies']} de "
          f"{report['companies_examined']} companias | {report['pooled_observations']} "
          f"observaciones pooled ===")
    lag = report["median_disclosure_lag_days"]
    print(f"  Retraso MEDIDO de las tenencias: {lag if lag is not None else '-'} dias "
          f"(la fila se fecha en el dia de PUBLICACION, nunca en el de referencia)")
    print()
    print(f"  {'activo':<8} {'compan.':>8} {'obs.':>6} {'bajo 1x':>9} {'mediana':>9} "
          f"{'p25':>8}  ultima")
    for asset, block in report["assets"].items():
        last = block["latest"] or {}
        below = f"{last['below_nav_share']:.0%}" if last else "-"
        median = f"{1 + last['mnav_gap']:.2f}x" if last else "-"
        p25 = f"{last['mnav_p25']:.2f}x" if last else "-"
        print(f"  {asset:<8} {block['n_companies']:>8} {block['observations']:>6} "
              f"{below:>9} {median:>9} {p25:>8}  {last.get('day', '-')}")

    if report["rejections"]:
        print()
        print("  --- por que se cayeron las demas (cobertura explicita, no un hueco) ---")
        for reason, count in report["rejections"].items():
            print(f"  {count:>4}  {reason}")

    if args.rejected:
        print()
        for row in report["rejected"]:
            print(f"  {str(row.get('ticker') or '?'):<8} {str(row.get('name') or '')[:34]:<36} "
                  f"{row.get('reason')}")

    row = declared_vs_measured_lag()
    state = (
        "sin medir" if row["measured_days"] is None
        else "no declarado" if row["declared_days"] is None
        else "ok" if row["matches"] else "FUERA DE TOLERANCIA"
    )
    print()
    print(f"  Retraso declarado en el catalogo: {row['declared_days'] or '-'} | "
          f"medido: {row['measured_days'] or '-'} | {state}")
    print(f"  Umbrales declarados: tesoro >= {report['policy']['treasury_min_asset_share']:.0%} "
          f"del activo | tolerancia del precio implicito x"
          f"{report['policy']['unit_price_tolerance']:.2f} | cohorte minima "
          f"{report['policy']['min_cohort']}")
    print(f"  Informe publicado en {path}")
    return 0


def _print_window(title, window) -> None:
    m = window.metrics
    print(f"=== {title} | {window.start.date()} -> {window.end.date()} ===")
    print(f"  Equity:        {m.starting_equity:,.0f} -> {m.ending_equity:,.0f} USD")
    print(f"  Total return:  {m.total_return_pct:+.2f}%   CAGR: {m.cagr_pct:+.2f}%")
    print(f"  Max drawdown:  {m.max_drawdown_pct:.2f}%")
    print(f"  Sharpe/Sortino:{m.sharpe:.2f} / {m.sortino:.2f}   Calmar: {m.calmar:.3f}")
    pf = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "n/a"
    print(f"  Trades: {m.num_trades}   Win rate: {m.win_rate_pct:.1f}%   Profit factor: {pf}")
    print(f"  Turnover: {m.turnover:.4f} (notional rotado/dia, en equity inicial)")
    print(f"  Fees paid: {m.total_fees_usd:,.2f} USD")
    print()


def cmd_report(args: argparse.Namespace) -> int:
    runner = _build(args.config)

    reports = {
        "status": runner.get_status,
        "positions": runner.get_positions_report,
        "risk": runner.get_risk_report,
        "history": runner.get_history_report,
        "performance": runner.get_performance_report,
        "symbols": runner.get_symbols_report,
    }

    for name in args.which or ["status", "positions", "performance"]:
        print(f"=== {name} ===")
        print(reports[name]())
        print()

    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="ai-trader",
        description="Headless control of ai-trader. Runs without the Telegram bot.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.getenv("AI_TRADER_CONFIG", DEFAULT_CONFIG_PATH)),
        help="Path to the TOML config file.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run-cycle", help="Run a single trading cycle and print the outcome.")

    bt = sub.add_parser("backtest", help="Backtest the configured strategies over history.")
    bt.add_argument("--start", help="Start date, YYYY-MM-DD (required for real data).")
    bt.add_argument("--end", help="End date, YYYY-MM-DD (required for real data).")
    bt.add_argument(
        "--capital", type=float, default=10_000.0, help="Starting equity (default 10000)."
    )
    bt.add_argument(
        "--validation", default="single_split",
        choices=["single_split", "walk_forward", "cpcv"],
        help="Validation scheme. 'single_split' is the legacy 70/30 (one OOS window); "
             "'walk_forward' and 'cpcv' run several purged/embargoed OOS windows and "
             "aggregate them into a distribution.",
    )
    bt.add_argument(
        "--folds", type=int, default=4,
        help="Walk-forward: number of OOS windows (default 4).",
    )
    bt.add_argument(
        "--groups", type=int, default=6,
        help="CPCV: number of groups the range is split into (default 6).",
    )
    bt.add_argument(
        "--test-groups", type=int, default=2,
        help="CPCV: groups used as test per combination (default 2 -> C(6,2)=15 folds).",
    )
    bt.add_argument(
        "--purge", type=int, default=None,
        help="Days of train purged right before each test block. Defaults to the "
             "runner's max_holding_days (how long a position can stay alive).",
    )
    bt.add_argument(
        "--embargo", type=int, default=None,
        help="Days of train embargoed right after each test block. Defaults to 1%% of "
             "the range (minimum 1 day).",
    )
    bt.add_argument(
        "--with-train", action="store_true",
        help="Also run the train window of every fold (slower; only a reference, since "
             "nothing is fitted inside a backtest).",
    )
    bt.add_argument(
        "--split", type=float, default=0.7,
        help="Train/test ratio of the legacy single split (default 0.7). "
             "Ignored if --split-date or --validation is given.",
    )
    bt.add_argument(
        "--split-date", default=None,
        help="Explicit train/test cutoff, YYYY-MM-DD. Overrides --split.",
    )
    bt.add_argument(
        "--synthetic", default=None,
        help="Run over a stored synthetic sample: LIBRARY:SCENARIO:PATH. Ignores --start/--end.",
    )
    bt.add_argument(
        "--synthetic-root", default=None,
        help="Root dir of synthetic libraries (default data/synthetic).",
    )
    bt.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    synth = sub.add_parser("synth", help="Generate and inspect synthetic market data.")
    synth_sub = synth.add_subparsers(dest="synth_command", required=True)

    gen = synth_sub.add_parser("generate", help="Design scenarios and synthesize OHLCV paths.")
    gen.add_argument("--library", required=True, help="Library id to create/overwrite.")
    gen.add_argument("--scenarios", type=int, default=24, help="Number of scenarios (default 24).")
    gen.add_argument("--paths", type=int, default=30, help="Monte Carlo paths per scenario (30).")
    gen.add_argument("--horizon", type=int, default=730, help="Days per path (default 730).")
    gen.add_argument("--seed", type=int, default=1_000, help="Base RNG seed (default 1000).")
    gen.add_argument(
        "--ai", action="store_true",
        help="Use Claude to design scenarios (needs ANTHROPIC_API_KEY). Default: offline templates.",
    )
    gen.add_argument("--synthetic-root", default=None, help="Root dir (default data/synthetic).")

    addp = synth_sub.add_parser(
        "add-paths",
        help="Regenerate/extend Monte Carlo paths from stored scenarios (NO API call).",
    )
    addp.add_argument("--library", required=True, help="Existing library id.")
    addp.add_argument(
        "--paths", type=int, required=True,
        help="New TOTAL paths per scenario. Existing paths stay identical; extras are added.",
    )
    addp.add_argument("--synthetic-root", default=None, help="Root dir (default data/synthetic).")

    der = synth_sub.add_parser(
        "derive",
        help="Derive a library from another with a declared enricher (NO API call).",
    )
    # `--from` es palabra reservada de Python: el destino tiene que llamarse de otra forma.
    der.add_argument("--from", dest="source", required=True, help="Source library id.")
    der.add_argument("--to", dest="target", required=True, help="Library id to create/overwrite.")
    der.add_argument(
        "--enricher", required=True, choices=sorted(ENRICHER_KEYS),
        help=(
            "Declared enricher: 'v2' = microstructure (made ai_v2/ai_v3); "
            "'v4' = microstructure + the five thematic observation channels."
        ),
    )
    der.add_argument(
        "--paths", type=int, default=None,
        help="Paths per scenario. Defaults to the source library's.",
    )
    der.add_argument("--synthetic-root", default=None, help="Root dir (default data/synthetic).")

    lst = synth_sub.add_parser("list", help="List stored synthetic libraries.")
    lst.add_argument("--synthetic-root", default=None, help="Root dir (default data/synthetic).")

    signals = sub.add_parser(
        "signals",
        help="Catalogo de senales externas, captura hacia adelante y auditoria de cobertura.",
    )
    signals_sub = signals.add_subparsers(dest="signals_command", required=True)

    cat = signals_sub.add_parser("catalog", help="Lista las fuentes declaradas.")
    cat.add_argument("--json", action="store_true", help="Emit the catalog as JSON.")

    cap = signals_sub.add_parser(
        "capture",
        help="Recorre el catalogo y archiva lo que haya. Corre aunque no haya adaptadores.",
    )
    cap.add_argument("--json", action="store_true", help="Emit the report as JSON.")

    aud = signals_sub.add_parser("audit", help="Cobertura de entidades y del archivo crudo.")
    aud.add_argument("--json", action="store_true", help="Emit the audit as JSON.")

    dep = signals_sub.add_parser(
        "depth",
        help="MIDE desde cuando hay dato de verdad y escribe el registro de mediciones.",
    )
    dep.add_argument("--json", action="store_true", help="Emit the ledger as JSON.")
    dep.add_argument(
        "--from-archive",
        action="store_true",
        help="No toca red: mide sobre lo ya archivado (unica opcion en forward_capture).",
    )
    dep.add_argument("--source", default=None, help="Medir solo esta fuente del catalogo.")

    feat = signals_sub.add_parser(
        "features",
        help="Publica el panel NORMALIZADO (z propia + z transversal) desde el archivo.",
    )
    feat.add_argument("--json", action="store_true", help="Emit the coverage summary as JSON.")
    feat.add_argument("--source", default=None, help="Publicar solo esta fuente del catalogo.")

    ev = signals_sub.add_parser(
        "events",
        help="CUENTA los eventos pooled por fuente y publica el radar que ven las puertas.",
    )
    ev.add_argument("--json", action="store_true", help="Emit the pool report as JSON.")
    ev.add_argument("--symbol", default=None, help="Ademas, imprime el radar de este simbolo.")

    adv = signals_sub.add_parser(
        "adv",
        help="MIDE el ADV de las entidades donde vive cada senal y escribe el registro.",
    )
    adv.add_argument("--json", action="store_true", help="Emit the ledger summary as JSON.")
    adv.add_argument(
        "--from-ledger",
        action="store_true",
        help="No toca red: lee la ultima medicion y la compara con lo declarado.",
    )

    dat = signals_sub.add_parser(
        "dat",
        help="Compone la distribucion de mNAV de las tesorerias cotizadas y publica el N.",
    )
    dat.add_argument("--json", action="store_true", help="Emit the cohort report as JSON.")
    dat.add_argument(
        "--rejected",
        action="store_true",
        help="Lista companıa a companıa las que NO entraron en la cohorte, con su motivo.",
    )

    report_parser = sub.add_parser("report", help="Print reports without running a cycle.")
    report_parser.add_argument(
        "which",
        nargs="*",
        choices=["status", "positions", "risk", "history", "performance", "symbols"],
        help="Reports to print. Defaults to status, positions and performance.",
    )

    args = parser.parse_args(argv)

    if args.command == "synth":
        synth_handlers = {
            "generate": cmd_synth_generate,
            "add-paths": cmd_synth_add_paths,
            "derive": cmd_synth_derive,
            "list": cmd_synth_list,
        }
        return synth_handlers[args.synth_command](args)

    if args.command == "signals":
        signal_handlers = {
            "catalog": cmd_signals_catalog,
            "capture": cmd_signals_capture,
            "audit": cmd_signals_audit,
            "depth": cmd_signals_depth,
            "features": cmd_signals_features,
            "events": cmd_signals_events,
            "adv": cmd_signals_adv,
            "dat": cmd_signals_dat,
        }
        return signal_handlers[args.signals_command](args)

    handlers = {"run-cycle": cmd_run_cycle, "backtest": cmd_backtest, "report": cmd_report}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
