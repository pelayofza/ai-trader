from ai_trader.data.market_data import MarketDataService

svc = MarketDataService()

markets = svc.search_prediction_markets("bitcoin", limit=10)
print("Markets:", len(markets))

for i, m in enumerate(markets[:5], start=1):
    print(f"\n[{i}]")
    print("Question:", m.question)
    print("Slug:", m.slug)
    print("Active:", m.active, "Closed:", m.closed, "Archived:", m.archived)
    print("YES:", m.yes_token)
    print("NO:", m.no_token)

if markets and markets[0].yes_token:
    mid = svc.get_prediction_midpoint(markets[0].yes_token.token_id)
    print("\nMid:", mid)