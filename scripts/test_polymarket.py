from ai_trader.main import build_runner
from ai_trader.data.market_data import MarketDataService

svc = MarketDataService()
runner = build_runner(svc)

results = runner.run_cycle()

print("Executions:", len(results))
for item in results:
    print(item)

print("\nPOSITIONS")
print(runner.get_positions_report())

print("\nPERFORMANCE")
print(runner.get_performance_report())