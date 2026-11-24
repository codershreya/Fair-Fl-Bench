"""Algorithm-registry entry point for EOFedMinMax. The actual from-scratch implementation
lives in fl/federated/eofedminmax.py; this module just re-exports it."""
from fl.federated.eofedminmax import EOFedMinMax

__all__ = ["EOFedMinMax"]
