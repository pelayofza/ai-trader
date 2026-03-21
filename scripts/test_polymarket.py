from ai_trader.main import build_runner

runner = build_runner()

results = runner.run_cycle()

print("Executions:", len(results))
for item in results:
    print(item)

print("\nPOSITIONS")
print(runner.get_positions_report())

print("\nPERFORMANCE")
print(runner.get_performance_report())