"""Algorithm-registry entry points for TrustFed-SP / TrustFed-EO. The actual from-scratch
implementation lives in fl/federated/trustfed.py; this module just re-exports it."""
from fl.federated.trustfed import TrustFedEO, TrustFedSP

__all__ = ["TrustFedSP", "TrustFedEO"]
