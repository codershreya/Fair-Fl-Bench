"""Algorithm-registry entry point for FedAvg-P2. The actual from-scratch implementation lives in
fl/federated/fedavg_p2.py; this module just re-exports it under the name the runner/registry
expects (same pattern as fl/algorithms/fedavg_p1.py).
"""
from fl.federated.fedavg_p2 import FedAvgP2

__all__ = ["FedAvgP2"]
