"""Algorithm-registry entry point for GFL / GFL-GBCFSS-EO / GFL-GBCFSS-DP. The actual
from-scratch implementation lives in fl/federated/gfl.py; this module just re-exports it."""
from fl.federated.gfl import GFL, GFLGBCFSSDP, GFLGBCFSSEO

__all__ = ["GFL", "GFLGBCFSSDP", "GFLGBCFSSEO"]
