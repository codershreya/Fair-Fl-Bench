"""Algorithm-registry entry point for GIFAIR-FL. The actual from-scratch implementation lives
in fl/federated/gifair.py; this module just re-exports it."""
from fl.federated.gifair import GIFAIRFL

__all__ = ["GIFAIRFL"]
