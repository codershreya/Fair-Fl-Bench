"""Algorithm-registry entry point for FedMinMax. The actual from-scratch implementation lives
in fl/federated/fedminmax.py; this module just re-exports it."""
from fl.federated.fedminmax import FedMinMax

__all__ = ["FedMinMax"]
