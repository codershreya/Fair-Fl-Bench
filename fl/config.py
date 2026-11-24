"""Shared configuration / constants for the unified Fair-FL benchmark.

Nothing here is dataset-loading logic (see fl/data/loaders.py); this module only holds the
numeric constants and the dataset registry that the rest of the codebase reads, so the shared
protocol (Section 3 of the project spec) lives in one place and is never hard-coded elsewhere.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
REFERENCE = PROJECT_ROOT / "reference"
P1_REPO = REFERENCE / "paper1_repo" / "FairnessBenchmarkingFL"
P2_REPO = REFERENCE / "paper2_repo" / "Fair_Federated_Learning"
RESULTS = PROJECT_ROOT / "results"
RESULTS_RUNS = RESULTS / "runs"
RESULTS_LOGS = RESULTS / "logs"
RESULTS_SUMMARY = RESULTS / "summary"
RESULTS_EQUIVALENCE = RESULTS / "equivalence"

# ---------------------------------------------------------------------------------------------
# Shared protocol constants (Section 3)
# ---------------------------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# IID client settings and attribute-based (non-IID) client settings
IID_K_VALUES = [3, 5, 10]
ATTR_K_VALUES = [3, 5]
AFFINITY_EPS = 0.20  # support floor; the only eps value used for the harmonised benchmark

# Feasibility thresholds for the attribute-based partitioner
MIN_CLIENT_TRAIN = 200   # minimum training samples per client
MIN_CELL = 10            # minimum samples per (A, Y) cell per client
MAX_PARTITION_ATTEMPTS = 20

# Preprocessing
MAX_ONEHOT_LEVELS = 30   # categorical cols with more train levels than this are integer-coded

# Harmonised tier: shared round count for every algorithm; per-algorithm items are in
# configs/algorithms.yaml (defining hyperparameters are never overridden here).
HARMONISED_ROUNDS = 50
CLIENT_FRACTION_C = 1.0

EXPECTED_ROW_COUNTS = {
    "adult": 48_842,
    "bank": 45_211,
    "cac": 1_994,
    "kdd": 299_285,
    "acs": 1_655_429,
}

# Dataset registry: source, partition-variable kind, sensitive attribute, and the human-readable
# meaning of A=1 / A=0 *if* the raw/"natural" encoding (not yet train-base-rate-oriented) were used.
# The actual A=1 group is decided per split/seed from TRAIN labels only (see fl/data/orientation.py)
# and may be the flip of the "raw1" group named here.
DATASETS = {
    "adult": dict(
        source="data/raw/adult.csv (Paper-2 copy of UCI adult.data + adult.test, 48,842 rows)",
        p_kind="binary", sensitive="sex",
        a_raw1_name="Male", a_raw0_name="Female",
        target="income", target_positive=">50K",
    ),
    "bank": dict(
        source="data/raw/bank.csv (Paper-2 copy of UCI bank-full, 45,211 rows)",
        p_kind="categorical", sensitive="marital",
        a_raw1_name="married", a_raw0_name="not married (single+divorced)",
        target="y", target_positive="yes",
    ),
    "cac": dict(
        source="data/raw/communities.csv (UCI Communities & Crime, Paper-1 copy, 1,994x128; "
               "27 columns dropped per Komiyama et al. 2018)",
        p_kind="continuous", sensitive="racepctblack",
        a_raw1_name="racepctblack > TRAIN median", a_raw0_name="racepctblack <= TRAIN median",
        target="ViolentCrimesPerPop", target_positive="> TRAIN median (adverse outcome)",
    ),
    "kdd": dict(
        source="data/raw/kdd.csv (extracted from Paper-1's kdd.zip, 299,285 rows; "
               "NOT Paper-2's 199,523-row file)",
        p_kind="binary", sensitive="sex",
        a_raw1_name="Male", a_raw0_name="Female",
        target="class", target_positive="1",
    ),
    "acs": dict(
        source="data/raw/acs/{state}_data.csv (50 state files, Paper-1 copy, ACSIncome features "
               "via folktables, 1,655,429 rows total)",
        p_kind="binary", sensitive="SEX",
        a_raw1_name="Male (SEX=1)", a_raw0_name="Female (SEX=2)",
        target="PINCP", target_positive="True (> $50k)",
    ),
}

ACS_STATES = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
]

# Komiyama et al. (2018) drop list for Communities & Crime, read from Paper 1's
# DatasetLoader/load_cac_data.py::DROP_COLS (reference/paper1_repo/.../load_cac_data.py).
CAC_DROP_COLS = [
    "community", "communityname", "county",
    "OtherPerCap",
    "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop",
    "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop",
    "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp",
    "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz",
    "PolicAveOTWorked", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr",
    "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
]
assert len(CAC_DROP_COLS) == 27, "Komiyama drop list must have 27 columns (spec Section 3)"
