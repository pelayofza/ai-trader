from ai_trader.data.market_data import MarketDataService

svc = MarketDataService()

for query in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]:
    print(f"\n=== QUERY: {query} ===")
    markets = svc.search_prediction_markets(query, limit=5)
    print("Markets:", len(markets))

    for i, m in enumerate(markets[:3], start=1):
        print(f"[{i}] {m.question}")
        print("Slug:", m.slug)
        print("YES:", m.yes_token)
        print("NO:", m.no_token)

        if m.yes_token:
            mid = svc.get_prediction_midpoint(m.yes_token.token_id)
            print("Mid:", mid)