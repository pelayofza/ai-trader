from ai_trader.data.market_data import MarketDataService

svc = MarketDataService()

markets = svc.search_prediction_markets("bitcoin", limit=5)
print("Markets:", len(markets))

if markets:
    m = markets[0]
    print("Question:", m.question)
    print("YES:", m.yes_token)
    print("NO:", m.no_token)

    if m.yes_token:
        mid = svc.get_prediction_midpoint(m.yes_token.token_id)
        print("Mid:", mid)