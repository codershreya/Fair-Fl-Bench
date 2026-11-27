"""Algorithm-registry entry point for FedFACT. The actual from-scratch implementation lives in
fl/federated/fedfact.py; this module just re-exports it."""
from fl.federated.fedfact import FedFACTIn, FedFACTPost

__all__ = ["FedFACTIn", "FedFACTPost"]
