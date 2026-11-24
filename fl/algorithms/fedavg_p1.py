"""Algorithm-registry entry point for FedAvg-P1. The actual from-scratch implementation lives in
fl/federated/fedavg_p1.py (shared by every caller: this registry, the equivalence harness, and
eventually the full benchmark runner); this module just re-exports it under the name the
runner/registry expects.
"""
from fl.federated.fedavg_p1 import FedAvgP1

__all__ = ["FedAvgP1"]
