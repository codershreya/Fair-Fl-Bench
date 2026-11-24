"""Algorithm-registry entry point for GlobalGroup-SP/-EO. The actual from-scratch
implementation lives in fl/federated/globalgroup.py; this module just re-exports it."""
from fl.federated.globalgroup import GlobalGroupEO, GlobalGroupSP

__all__ = ["GlobalGroupEO", "GlobalGroupSP"]
