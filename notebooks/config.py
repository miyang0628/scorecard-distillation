# config.py
# Central configuration for the whole pipeline. Every notebook imports this,
# so you change experiment settings in ONE place.
#
# >>> REAL-DATA RUN. Refuses to silently fall back to synthetic data or the
#     proxy teacher. Run notebook 00 FIRST after any change here (it regenerates
#     the split/features and prints a GO/NO-GO banner). Quick synthetic test:
#     DATASET="synthetic"; STRICT_TEACHER=False

import os as _os
from pathlib import Path as _P

# --- load local secrets from .env (e.g. TABPFN_TOKEN); never commit .env -----
for _env in (_P(__file__).resolve().parent / ".env",
             _P(__file__).resolve().parent.parent / ".env"):
    if _env.exists():
        for _ln in _env.read_text().splitlines():
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                _os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        break

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET = "home_credit"                 # "synthetic" | "home_credit" | "lending_club"
FALLBACK_TO_SYNTHETIC = False           # False -> hard error if the real CSV is missing
HOME_CREDIT_FILE = "application_train.csv"
LENDING_CLUB_FILE = "accepted_2007_to_2018Q4.csv"

# Row subsampling (class ratio preserved). None = all rows.
SUBSAMPLE_N = {"synthetic": 12000, "home_credit": 16000, "lending_club": 300000}

ENCODE_CATEGORICALS = False
CATEGORICAL_MAX_CARD = 20

# Cap #features (top-K by |WoE-target corr| on train). None = all.
# Home Credit has ~104 numerics; 25 keeps GPU teacher + 50-repeat ablation fast
# and matches scorecard practice (Siddiqi: 8-15 typical).
MAX_FEATURES = 25

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
USE_GPU = "auto"                        # CUDA for the TabPFN teacher; students exact/CPU

# ---------------------------------------------------------------------------
# Teacher (notebook 01)
# ---------------------------------------------------------------------------
TEACHER_KIND = "tabpfn"
STRICT_TEACHER = True                   # raise instead of silently using the proxy teacher
TEACHER_ROW_CAP = 10000                 # balanced TabPFN v2 context cap
TEACHER_PREDICT_BATCH = 1000            # GPU predict batch (lower if OOM: 500/250)
# Local TabPFN v2 checkpoint (.ckpt) in the PROJECT ROOT (parent of notebooks/).
TABPFN_MODEL_PATH = str(_P(__file__).resolve().parent.parent / "tabpfn-v2-classifier.ckpt")

# ---------------------------------------------------------------------------
# Experiment scale
# ---------------------------------------------------------------------------
N_REPEATS = 50
N_BINS = 5
LAMBDA_MIX = 0.30
N_BOOT = 500
P_GRID = [8, 10, 12, 15]
B_GRID = [3, 5, 7]

RANDOM_STATE = 42