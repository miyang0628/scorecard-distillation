# data_loaders.py
# Dataset loaders for the pipeline. Each returns (X: DataFrame, y: np.ndarray,
# source: str, monotone: dict|None). X holds NUMERIC features with NaN preserved
# (XGBoost and the scorecard both handle missing natively). Categorical encoding,
# if enabled, happens AFTER the train/es/eval split (notebook 00) to avoid leakage.

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

try:
    import config as CFG
except Exception:                       # minimal fallback if config missing
    class CFG:  # type: ignore
        RANDOM_STATE = 42
        SUBSAMPLE_N = {}
        HOME_CREDIT_FILE = "application_train.csv"
        LENDING_CLUB_FILE = "accepted_2007_to_2018Q4.csv"


# ---------------------------------------------------------------------------
# Synthetic credit data (XOR interactions => a genuine additive-model gap)
# ---------------------------------------------------------------------------
MONO_SYNTH = {
    "age": -1, "income": -1, "emp_length": -1, "debt_ratio": +1,
    "utilization": +1, "n_delinquencies": +1, "credit_history_len": -1,
    "n_inquiries": +1, "n_open_lines": +1,
}


def make_synthetic_credit(n, rng):
    def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
    age = np.clip(rng.normal(42, 12, n), 19, 85)
    income = np.exp(rng.normal(10.7, 0.5, n))
    emp = np.clip(rng.gamma(3.0, 2.0, n), 0, 40)
    dti = np.clip(rng.beta(2.0, 5.0, n) * 1.5, 0, 1.5)
    util = np.clip(rng.beta(2.0, 4.0, n), 0, 1.2)
    nd = rng.poisson(0.4, n).astype(float)
    hist = np.clip(rng.gamma(4.0, 3.0, n), 0.5, 45)
    ninq = rng.poisson(1.1, n).astype(float)
    opn = np.clip(rng.poisson(8, n), 1, 40).astype(float)
    add = (-5.30 - 0.0364 * (age - 42) - 1.105 * (np.log(income) - 10.7)
           - 0.078 * emp + 1.00 * dti + 1.20 * util + 0.50 * nd
           - 0.0585 * hist + 0.40 * ninq + 0.0182 * (opn - 8))
    xor1 = ((util > 0.4).astype(int) ^ (dti > 0.35).astype(int)).astype(float)
    xor2 = ((ninq >= 1).astype(int) ^ (nd >= 1).astype(int)).astype(float)
    z = add + 2.2 * xor1 + 1.8 * xor2 + rng.normal(0, 0.22, n)
    y = (rng.random(n) < sig(z)).astype(int)
    X = pd.DataFrame({
        "age": age, "income": income, "emp_length": emp, "debt_ratio": dti,
        "utilization": util, "n_delinquencies": nd, "credit_history_len": hist,
        "n_inquiries": ninq, "n_open_lines": opn})
    X.loc[rng.random(n) < 0.08, "emp_length"] = np.nan
    X.loc[rng.random(n) < 0.05, "credit_history_len"] = np.nan
    return X, y


def load_synthetic(n=12000, seed=CFG.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    X, y = make_synthetic_credit(n, rng)
    return X, y, "synthetic", MONO_SYNTH


# ---------------------------------------------------------------------------
# Home Credit Default Risk  (Kaggle: application_train.csv)
#   target = TARGET (1 = default/late), id = SK_ID_CURR
# ---------------------------------------------------------------------------
def load_home_credit(path, subsample=None, seed=CFG.RANDOM_STATE):
    df = pd.read_csv(path)
    if "TARGET" not in df.columns:
        raise ValueError("application_train.csv must contain a TARGET column")
    y = df["TARGET"].astype(int).to_numpy()
    drop = [c for c in ["TARGET", "SK_ID_CURR"] if c in df.columns]
    X = _finalize_features(df.drop(columns=drop))
    # DAYS_* are negative days-before-application; leave as-is (monotone handled later)
    X, y = _subsample(X, y, subsample, seed)
    return X, y, "home_credit", None


# ---------------------------------------------------------------------------
# Lending Club accepted loans  (Kaggle: accepted_2007_to_2018Q4.csv)
#   target derived from loan_status; drop post-origination leakage columns.
# ---------------------------------------------------------------------------
LC_GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
LC_BAD = {"Charged Off", "Default",
          "Does not meet the credit policy. Status:Charged Off"}
# outcome / post-origination fields that would leak the label
LC_LEAKAGE = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "last_pymnt_d", "next_pymnt_d", "out_prncp",
    "out_prncp_inv", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "settlement_amount", "settlement_percentage",
    "settlement_term", "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_date", "settlement_status", "hardship_*",
    # LendingClub's own risk pricing (encodes their model -> drop for a clean study)
    "grade", "sub_grade", "int_rate",
]
LC_ID = ["id", "member_id", "url", "loan_status"]


def _lc_parse_numeric_strings(X):
    """Parse LC's stringy numerics: term, revol_util, emp_length."""
    if "term" in X:
        X["term"] = (X["term"].astype(str).str.extract(r"(\d+)")[0]
                     .astype(float))
    for col in ["revol_util", "int_rate"]:
        if col in X and X[col].dtype == object:
            X[col] = (X[col].astype(str).str.replace("%", "", regex=False)
                      .replace("nan", np.nan).astype(float))
    if "emp_length" in X:
        s = X["emp_length"].astype(str)
        s = s.str.replace("< 1 year", "0", regex=False)
        s = s.str.replace("10+ years", "10", regex=False)
        X["emp_length"] = s.str.extract(r"(\d+)")[0].astype(float)
    return X


def load_lending_club(path, subsample=None, seed=CFG.RANDOM_STATE):
    df = pd.read_csv(path, low_memory=False)
    if "loan_status" not in df.columns:
        raise ValueError("accepted CSV must contain a loan_status column")
    status = df["loan_status"].astype(str)
    keep = status.isin(LC_GOOD | LC_BAD)
    df = df[keep].reset_index(drop=True)
    y = df["loan_status"].isin(LC_BAD).astype(int).to_numpy()
    # drop leakage / id columns (support simple wildcards like hardship_*)
    drop = set(LC_ID)
    for c in LC_LEAKAGE:
        if c.endswith("*"):
            pref = c[:-1]
            drop |= {col for col in df.columns if col.startswith(pref)}
        elif c in df.columns:
            drop.add(c)
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = _lc_parse_numeric_strings(X)
    X = _finalize_features(X)
    # drop columns that are entirely NaN or constant
    X = X.loc[:, X.notna().any()]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    X, y = _subsample(X, y, subsample, seed)
    return X, y, "lending_club", None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _finalize_features(X):
    """Keep numeric features; also keep low-cardinality object columns (as
    'category' dtype, to be target-encoded post-split) when ENCODE_CATEGORICALS."""
    if getattr(CFG, "ENCODE_CATEGORICALS", False):
        maxc = getattr(CFG, "CATEGORICAL_MAX_CARD", 20)
        num = X.select_dtypes(include=[np.number])
        obj = X.select_dtypes(include=["object"])
        keep_obj = [c for c in obj.columns if obj[c].nunique(dropna=True) <= maxc]
        out = pd.concat([num, obj[keep_obj].astype("category")], axis=1)
        return out.copy()
    return X.select_dtypes(include=[np.number]).copy()


def _subsample(X, y, n, seed):
    if n is None or n >= len(X):
        return X.reset_index(drop=True), y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), int(n), replace=False)   # class ratio preserved
    return X.iloc[idx].reset_index(drop=True), y[idx]


def fit_target_encoding(Xtr, y, cols, smoothing=50.0):
    """Smoothed mean-target encoding fit on TRAIN only."""
    prior = float(np.mean(y))
    maps = {}
    for c in cols:
        s = Xtr[c].astype("object")
        stats = pd.DataFrame({"c": s.to_numpy(), "y": y}).groupby("c")["y"]
        agg = stats.agg(["mean", "count"])
        enc = (agg["mean"] * agg["count"] + prior * smoothing) / (agg["count"] + smoothing)
        maps[c] = (enc.to_dict(), prior)
    return maps


def apply_target_encoding(X, maps):
    out = X.copy()
    for c, (m, prior) in maps.items():
        out[c] = X[c].astype("object").map(m).fillna(prior).astype(float)
    return out


def load_dataset():
    """Dispatch on config.DATASET; raise a clear error if a real CSV is missing."""
    ds = getattr(CFG, "DATASET", "synthetic")
    from pathlib import Path
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    sub = getattr(CFG, "SUBSAMPLE_N", {}).get(ds, None)
    try:
        if ds == "synthetic":
            return load_synthetic(getattr(CFG, "SUBSAMPLE_N", {}).get("synthetic", 12000))
        if ds == "home_credit":
            return load_home_credit(raw / CFG.HOME_CREDIT_FILE, subsample=sub)
        if ds == "lending_club":
            return load_lending_club(raw / CFG.LENDING_CLUB_FILE, subsample=sub)
        raise ValueError(f"unknown DATASET {ds}")
    except FileNotFoundError:
        fname = (CFG.HOME_CREDIT_FILE if ds == "home_credit"
                 else CFG.LENDING_CLUB_FILE if ds == "lending_club" else "")
        if getattr(CFG, "FALLBACK_TO_SYNTHETIC", False):
            print(f"[{ds}] raw file '{fname}' not found under data/raw/; "
                  f"falling back to synthetic data.")
            return load_synthetic(getattr(CFG, "SUBSAMPLE_N", {}).get("synthetic", 12000))
        raise FileNotFoundError(
            f"DATASET='{ds}' but '{fname}' was not found in data/raw/. "
            f"Place the Kaggle CSV there, or set FALLBACK_TO_SYNTHETIC=True in "
            f"config.py for a synthetic smoke test.")