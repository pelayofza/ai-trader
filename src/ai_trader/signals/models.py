from dataclasses import dataclass

@dataclass
class TradeProposal:
    symbol: str
    side: str              # "LONG" / "SHORT" / "NONE"
    entry: float | None
    stop: float | None
    target: float | None
    rr: float | None
    reason: str