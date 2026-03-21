from ai_trader.app.runner import RunnerConfig, TradingRunner
from ai_trader.data.market_data import MarketDataService
from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.risk.engine import RiskEngine, RiskLimits
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import OrderRequest, OrderType, Side


class DummyStrategy:
    strategy_id = "dummy"

    def generate_signal(self, symbol: str, bars):
        return None


svc = MarketDataService()

runner = TradingRunner(
    config=RunnerConfig(
        symbols=["BTC/USDT"],
        lookback_days=30,
        max_holding_days=10,
    ),
    market_data_reader=svc,
    strategies=[DummyStrategy()],
    risk_engine=RiskEngine(RiskLimits()),
    execution_engine=PaperExecutionEngine(PaperExecutionConfig()),
    polymarket_execution_engine=PolymarketPaperExecutionEngine(),
)

markets = svc.search_prediction_markets("bitcoin", limit=3)
if not markets:
    raise RuntimeError("No prediction markets found")

market = markets[0]
if market.yes_token is None:
    raise RuntimeError("No yes token found")

order = OrderRequest(
    symbol=f"PM::{market.slug}",
    side=Side.BUY,
    size=10,
    order_type=OrderType.MARKET,
    strategy_id="manual_polymarket_test",
    venue=Venue.POLYMARKET,
    asset_class=AssetClass.PREDICTION,
    instrument_id=market.yes_token.token_id,
    outcome=market.yes_token.outcome,
    metadata={
        "question": market.question,
        "slug": market.slug,
    },
)

result = runner.submit_order(order)

print("EXECUTION")
print(result)

print("\nPOSITIONS REPORT")
print(runner.get_positions_report())

print("\nPERFORMANCE REPORT")
print(runner.get_performance_report())