from ai_trader.data.market_data import MarketDataService

svc = MarketDataService()

markets = svc.search_prediction_markets("bitcoin", limit=3)
print("Markets:", len(markets))

for i, m in enumerate(markets, start=1):
    print(f"\n[{i}] {m.question}")
    print("Slug:", m.slug)
    print("YES:", m.yes_token)
    print("NO:", m.no_token)

if markets:
    market = markets[0]

    if market.yes_token:
        print("\n--- YES TOKEN CHECK ---")
        print("Token ID:", market.yes_token.token_id)

        mid = svc.get_prediction_midpoint(market.yes_token.token_id)
        print("Mid:", mid)

        book = svc.get_prediction_orderbook(market.yes_token.token_id)
        print("Book keys:", list(book.keys()) if isinstance(book, dict) else book)