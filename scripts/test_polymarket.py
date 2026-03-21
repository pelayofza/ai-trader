from ai_trader.data.market_data import MarketDataService
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import OrderRequest, OrderType, Side

svc = MarketDataService()
engine = PolymarketPaperExecutionEngine()

markets = svc.search_prediction_markets("bitcoin", limit=3)
print("Markets:", len(markets))

if not markets:
    raise RuntimeError("No markets found")

market = markets[0]
print("Question:", market.question)
print("Slug:", market.slug)

if market.yes_token is None:
    raise RuntimeError("First market has no yes token")

order = OrderRequest(
    symbol=f"PM::{market.slug}",
    side=Side.BUY,
    size=10,
    order_type=OrderType.MARKET,
    strategy_id="manual_test",
    venue=Venue.POLYMARKET,
    asset_class=AssetClass.PREDICTION,
    instrument_id=market.yes_token.token_id,
    outcome=market.yes_token.outcome,
    metadata={
        "question": market.question,
        "slug": market.slug,
    },
)

result = engine.execute(order)
print("Execution success:", result.success)
print("Status:", result.status.value)
print("Order ID:", result.order_id)
print("Filled price:", result.filled_price)
print("Filled size:", result.filled_size)
print("Fees:", result.fees)
print("Venue:", result.venue.value if result.venue else None)
print("Asset class:", result.asset_class.value if result.asset_class else None)
print("Instrument ID:", result.instrument_id)
print("Outcome:", result.outcome)

fills = engine.list_fills()
print("Stored fills:", len(fills))
if fills:
    print("Last fill:", fills[-1])