"""Algorithm-registry entry point for FairTrade-SP/-ATE. The actual from-scratch
implementation lives in fl/federated/fairtrade.py; this module just re-exports it."""
from fl.federated.fairtrade import FairTradeATE, FairTradeSP

__all__ = ["FairTradeATE", "FairTradeSP"]
