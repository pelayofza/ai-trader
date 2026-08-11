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

    lst = synth_sub.add_parser("list", help="List stored synthetic libraries.")
    lst.add_argument("--synthetic-root", default=None, help="Root dir (default data/synthetic).")

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
            "list": cmd_synth_list,
        }
        return synth_handlers[args.synth_command](args)

    handlers = {"run-cycle": cmd_run_cycle, "backtest": cmd_backtest, "report": cmd_report}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
