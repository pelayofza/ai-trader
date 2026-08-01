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

    config = load_config(args.config)
    service = MarketDataService()

    engine = BacktestEngine(config, service, starting_equity=args.capital)
    result = engine.run(
        start=parse_date(args.start),
        end=parse_date(args.end),
        split_ratio=args.split,
        split_date=parse_date(args.split_date) if args.split_date else None,
    )

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    _print_window("TRAIN (in-sample)", result.train)
    _print_window("TEST (out-of-sample)", result.test)
    print("=" * 48)
    print(f"HEADLINE SCORE (Calmar, out-of-sample): {result.headline_score:.3f}")
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
    bt.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    bt.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    bt.add_argument(
        "--capital", type=float, default=10_000.0, help="Starting equity (default 10000)."
    )
    bt.add_argument(
        "--split", type=float, default=0.7,
        help="Train/test ratio (default 0.7). Ignored if --split-date is given.",
    )
    bt.add_argument(
        "--split-date", default=None,
        help="Explicit train/test cutoff, YYYY-MM-DD. Overrides --split.",
    )
    bt.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    report_parser = sub.add_parser("report", help="Print reports without running a cycle.")
    report_parser.add_argument(
        "which",
        nargs="*",
        choices=["status", "positions", "risk", "history", "performance", "symbols"],
        help="Reports to print. Defaults to status, positions and performance.",
    )

    args = parser.parse_args(argv)

    handlers = {"run-cycle": cmd_run_cycle, "backtest": cmd_backtest, "report": cmd_report}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
