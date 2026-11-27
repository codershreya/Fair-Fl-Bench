"""Registry of from-scratch algorithm implementations, keyed by the SAME variant names the
oracle uses (reference/oracle_fairbench/fairbench/algos/__init__.py::REGISTRY) so results are
directly comparable. Only the variants built so far are registered; Phase 6 adds the rest
(GIFAIR-FL, FairTrade-SP/-ATE, GFL and its two identity variants)."""
from fl.algorithms.eofedminmax import EOFedMinMax
from fl.algorithms.fedavg_p1 import FedAvgP1
from fl.algorithms.fedavg_p2 import FedAvgP2
from fl.algorithms.fairtrade import FairTradeATE, FairTradeSP
from fl.algorithms.fedfact import FedFACTIn, FedFACTPost
from fl.algorithms.fedminmax import FedMinMax
from fl.algorithms.gfl import GFL, GFLGBCFSSDP, GFLGBCFSSEO
from fl.algorithms.gifair import GIFAIRFL
from fl.algorithms.globalgroup import GlobalGroupEO, GlobalGroupSP
from fl.algorithms.trustfed import TrustFedEO, TrustFedSP

REGISTRY = {
    "FedAvg-P1": FedAvgP1,
    "FedAvg-P2": FedAvgP2,
    "TrustFed-SP": TrustFedSP,
    "TrustFed-EO": TrustFedEO,
    "FedMinMax": FedMinMax,
    "EOFedMinMax": EOFedMinMax,
    "GlobalGroup-SP": GlobalGroupSP,
    "GlobalGroup-EO": GlobalGroupEO,
    "GIFAIR-FL": GIFAIRFL,
    "FairTrade-SP": FairTradeSP,
    "FairTrade-ATE": FairTradeATE,
    "GFL": GFL,
    "GFL-GBCFSS-EO": GFLGBCFSSEO,
    "GFL-GBCFSS-DP": GFLGBCFSSDP,
    "FedFACT-In": FedFACTIn,
    "FedFACT-Post": FedFACTPost,
}

__all__ = ["REGISTRY"]
