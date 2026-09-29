# common.py
# Shared utilities for "Distilling Transformers into Scorecards".
# Grayscale figure style, PNG+PDF export at dpi=600, lossless stump->scorecard
# extraction, PDO quantization, calibration metrics (ECE, Brier-Murphy
# decomposition, Spiegelhalter Z), and the fast DeLong paired-AUC test.
# All notebooks import from this module.

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------------------------
# Paths (resolved relative to this file, so notebooks work from any cwd)
# ----------------------------------------------------------------------------
COMMON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMMON_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
for _d in (DATA_DIR, RAW_DIR, PROC_DIR, RESULTS_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

try:
    import config as _CFG
    RANDOM_STATE = getattr(_CFG, "RANDOM_STATE", 42)
except Exception:
    _CFG = None
    RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Compute devices
#   Student stump boosting uses tree_method="exact" on CPU: it is required for
#   *exact* lossless reduction (hist quantizes split thresholds into histogram
#   bins, which breaks reconstruction from the tree dump) and it is fast anyway
#   because the students are fit on features already discretized to a few bins.
#   GPU acceleration is reserved for the TabPFN teacher (torch), via torch_device().
# ----------------------------------------------------------------------------
_TORCH_DEVICE = None


def torch_device():
    """Return 'cuda' or 'cpu' for the teacher, per config.USE_GPU."""
    global _TORCH_DEVICE
    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE
    pref = getattr(_CFG, "USE_GPU", "auto") if _CFG is not None else "auto"
    if pref is False:
        _TORCH_DEVICE = "cpu"; return _TORCH_DEVICE
    try:
        import torch
        _TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _TORCH_DEVICE = "cpu"
    if pref is True and _TORCH_DEVICE != "cuda":
        _TORCH_DEVICE = "cuda"  # honor an explicit request even if probe failed
    return _TORCH_DEVICE


def xgb_base_params(base_score=0.5, eta=0.2):
    """Depth-1 (stump) boosting params. exact/CPU => exact lossless reduction."""
    return dict(max_depth=1, eta=eta, objective="binary:logistic",
                base_score=base_score, tree_method="exact", device="cpu",
                eval_metric="auc", min_child_weight=5, reg_lambda=1.0)

# ----------------------------------------------------------------------------
# Grayscale figure style
# ----------------------------------------------------------------------------
# Sequential grayscale ramp for ordered categories (light -> dark), avoiding
# pure white and pure black endpoints so all series stay visible on white.
GRAY_SEQUENTIAL = ["#111111", "#4d4d4d", "#7f7f7f", "#a6a6a6", "#cccccc"]
# Distinct grays for a small number of nominal categories.
GRAY_CATEGORICAL = ["#1a1a1a", "#808080", "#bdbdbd", "#595959", "#d9d9d9"]


def set_grayscale_style():
    """Apply a consistent grayscale, publication-oriented Matplotlib/Seaborn
    style. Call once at the top of every notebook."""
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update({
        "figure.dpi": 120,           # on-screen; export dpi is set in save_fig
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.prop_cycle": mpl.cycler(color=GRAY_CATEGORICAL),
        "image.cmap": "Greys",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    sns.set_palette(GRAY_CATEGORICAL)


def save_fig(fig, name, tight=True):
    """Save a figure to results/figures as BOTH .png and .pdf at dpi=600.
    No caption is written onto the canvas (captions live in the manuscript).
    `name` is the file stem, e.g. 'pareto_complexity'."""
    stem = str(name)
    kw = dict(dpi=600, facecolor="white")
    if tight:
        kw["bbox_inches"] = "tight"
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, **kw)
    fig.savefig(pdf, **kw)
    print(f"[saved] {png.name}  &  {pdf.name}")
    return png, pdf


def save_table(df, name, index=False, float_format="%.4f"):
    """Save a results table to results/tables as CSV (and a LaTeX sibling)."""
    stem = str(name)
    csv = TAB_DIR / f"{stem}.csv"
    df.to_csv(csv, index=index)
    try:
        tex = TAB_DIR / f"{stem}.tex"
        df.to_latex(tex, index=index, float_format=lambda v: (float_format % v))
    except Exception as e:  # LaTeX export is best-effort
        warnings.warn(f"LaTeX export failed for {stem}: {e}")
    print(f"[saved] {csv.name}")
    return csv


# ----------------------------------------------------------------------------
# Core probability <-> log-odds helpers
# ----------------------------------------------------------------------------
def sigmoid(f):
    f = np.clip(np.asarray(f, dtype=float), -35, 35)
    return 1.0 / (1.0 + np.exp(-f))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------
# Discrimination: AUC + fast DeLong paired test
#   Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm".
# ----------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _group_preds_by_label(*preds, y):
    y = np.asarray(y)
    order = (-y).argsort(kind="mergesort")  # positives first
    label_1_count = int(y.sum())
    stacked = np.vstack([np.asarray(p)[order] for p in preds])
    return stacked, label_1_count


def auc(y, p):
    """AUC via the DeLong midrank estimator (matches sklearn roc_auc_score)."""
    stacked, m = _group_preds_by_label(p, y=y)
    aucs, _ = _fast_delong(stacked, m)
    return float(aucs[0])


def delong_test(y, p1, p2):
    """Paired DeLong test comparing AUC(p1) vs AUC(p2) on the same labels y.
    Returns dict(auc1, auc2, diff, z, p_value, cov). Two-sided p-value."""
    stacked, m = _group_preds_by_label(p1, p2, y=y)
    aucs, cov = _fast_delong(stacked, m)
    a1, a2 = float(aucs[0]), float(aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z, pval = 0.0, 1.0
    else:
        z = (a1 - a2) / np.sqrt(var)
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc1": a1, "auc2": a2, "diff": a1 - a2,
            "z": float(z), "p_value": float(pval),
            "var": float(var)}


# ----------------------------------------------------------------------------
# Calibration metrics
# ----------------------------------------------------------------------------
def _bin_ids(p, n_bins=10, strategy="quantile"):
    p = np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
    else:  # uniform
        edges = np.linspace(0, 1, n_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    ids = np.digitize(p, edges[1:-1], right=False)
    return ids


def ece(y, p, n_bins=10, strategy="quantile"):
    """Expected Calibration Error (|acc - conf|, weighted by bin mass)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ids = _bin_ids(p, n_bins, strategy)
    N = len(y)
    e = 0.0
    for b in np.unique(ids):
        m = ids == b
        w = m.sum() / N
        e += w * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_murphy(y, p, n_bins=10, strategy="quantile"):
    """Murphy (1973) 3-component decomposition of the Brier score:

        Brier = Reliability - Resolution + Uncertainty

    - reliability  (calibration error, lower is better)
    - resolution   (= refinement, discrimination information, higher is better)
    - uncertainty  (base-rate variance, fixed by the data)

    Binning is on the predicted probability. Returns a dict; `check` is the
    reconstructed Brier (reliability - resolution + uncertainty) and should
    match `brier` up to binning granularity."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    N = len(y)
    obar = y.mean()
    ids = _bin_ids(p, n_bins, strategy)
    reliability = 0.0
    resolution = 0.0
    for b in np.unique(ids):
        m = ids == b
        nk = m.sum()
        if nk == 0:
            continue
        pk = p[m].mean()      # mean forecast in bin
        ok = y[m].mean()      # observed frequency in bin
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - obar) ** 2
    reliability /= N
    resolution /= N
    uncertainty = obar * (1 - obar)
    return {
        "brier": brier_score(y, p),
        "reliability": float(reliability),
        "resolution": float(resolution),   # == refinement
        "refinement": float(resolution),
        "uncertainty": float(uncertainty),
        "check": float(reliability - resolution + uncertainty),
    }


def spiegelhalter_z(y, p, eps=1e-6):
    """Spiegelhalter's Z test of calibration. Under perfect calibration
    Z ~ N(0,1); |Z| large => miscalibrated. Returns dict(z, p_value)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = num / den if den > 0 else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(pval)}


# ----------------------------------------------------------------------------
# Bootstrap CI (paired, over rows)
# ----------------------------------------------------------------------------
def bootstrap_ci(stat_fn, *arrays, n_boot=500, alpha=0.05, seed=RANDOM_STATE):
    """Percentile bootstrap CI for a statistic computed on aligned arrays.
    stat_fn receives resampled copies of each array in `arrays`."""
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = np.empty(n_boot)
    idx_all = np.arange(N)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=N, replace=True)
        vals[b] = stat_fn(*[np.asarray(a)[idx] for a in arrays])
    lo = np.percentile(vals, 100 * alpha / 2)
    hi = np.percentile(vals, 100 * (1 - alpha / 2))
    return {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi),
            "std": float(vals.std(ddof=1))}


# ----------------------------------------------------------------------------
# Lossless reduction: stump ensemble  ->  additive step-function scorecard
#   (Map A of the loss ledger: an algebraic re-arrangement, no approximation.)
# ----------------------------------------------------------------------------
class Scorecard:
    """Additive step-function scorecard f(x) = intercept + sum_j g_j(x_j),
    where each g_j is piecewise-constant over bins of feature j.

    Attributes
    ----------
    intercept : float                 global log-odds offset (base margin)
    features  : list[str]             feature order
    edges     : dict[str, np.ndarray] internal split thresholds per feature
    contribs  : dict[str, np.ndarray] log-odds contribution c_{j,b} per bin
                                      (len = len(edges)+1)
    missing   : dict[str, float]      contribution for a missing value
    points    : dict[str, np.ndarray] integer points per bin (after quantize)
    points_missing : dict[str, int]
    pdo, grid, base_points : quantization settings (None until quantized)
    """

    def __init__(self, intercept, features, edges, contribs, missing):
        self.intercept = float(intercept)
        self.features = list(features)
        self.edges = {k: np.asarray(v, dtype=float) for k, v in edges.items()}
        self.contribs = {k: np.asarray(v, dtype=float) for k, v in contribs.items()}
        self.missing = dict(missing)
        self.points = None
        self.points_missing = None
        self.pdo = None
        self.grid = None
        self.base_points = None

    def _bin_index(self, feat, x):
        edges = self.edges[feat]
        x = np.asarray(x, dtype=float)
        idx = np.digitize(x, edges, right=False)  # 0..len(edges)
        return idx

    def margin(self, X):
        """Continuous log-odds margin f(x) from the additive contributions."""
        X = pd.DataFrame(X).reset_index(drop=True)
        f = np.full(len(X), self.intercept, dtype=float)
        for feat in self.features:
            col = X[feat].to_numpy(dtype=float)
            nan = np.isnan(col)
            idx = self._bin_index(feat, np.where(nan, 0.0, col))
            c = self.contribs[feat][idx]
            c = np.where(nan, self.missing.get(feat, 0.0), c)
            f += c
        return f

    def predict_proba(self, X):
        return sigmoid(self.margin(X))

    # --- quantization (Map A round(): the ONLY information-losing step) ------
    def quantize(self, pdo=20.0, grid=1, base_points=600.0):
        """Points_{j,b} = round( -(PDO/ln2) * c_{j,b} / grid ) * grid.
        Higher score => lower risk (standard credit convention)."""
        scale = -pdo / np.log(2.0)
        self.points = {}
        self.points_missing = {}
        for feat in self.features:
            raw = scale * self.contribs[feat] / grid
            self.points[feat] = np.round(raw) * grid
            self.points_missing[feat] = float(
                np.round(scale * self.missing.get(feat, 0.0) / grid) * grid)
        self.base_points = float(base_points + scale * self.intercept)
        self.pdo, self.grid = float(pdo), int(grid)
        return self

    def total_points(self, X):
        """Integer total score = base + sum of per-feature bin points."""
        assert self.points is not None, "call quantize() first"
        X = pd.DataFrame(X).reset_index(drop=True)
        s = np.full(len(X), self.base_points, dtype=float)
        for feat in self.features:
            col = X[feat].to_numpy(dtype=float)
            nan = np.isnan(col)
            idx = self._bin_index(feat, np.where(nan, 0.0, col))
            pts = self.points[feat][idx]
            pts = np.where(nan, self.points_missing[feat], pts)
            s += pts
        return s

    def quantized_margin(self, X):
        """Recover a log-odds margin from integer points (inverse PDO map),
        so calibration can be measured on the deployed integer scorecard."""
        assert self.points is not None, "call quantize() first"
        scale = -self.pdo / np.log(2.0)
        pts = self.total_points(X)
        return (pts - self.base_points) / scale + self.intercept

    def quantized_proba(self, X):
        return sigmoid(self.quantized_margin(X))

    def n_cells(self):
        """Total number of scorecard cells C = sum_j (#bins of feature j)."""
        return int(sum(len(self.contribs[f]) for f in self.features))

    def summary_table(self):
        rows = []
        for feat in self.features:
            edges = self.edges[feat]
            lo = np.concatenate([[-np.inf], edges])
            hi = np.concatenate([edges, [np.inf]])
            for b in range(len(self.contribs[feat])):
                row = {"feature": feat, "bin": b,
                       "lo": lo[b], "hi": hi[b],
                       "logodds": self.contribs[feat][b]}
                if self.points is not None:
                    row["points"] = self.points[feat][b]
                rows.append(row)
        return pd.DataFrame(rows)


def extract_scorecard_from_xgb(booster, feature_names):
    """Perform Map A: collect all depth-1 stump splits from a trained XGBoost
    booster and re-arrange them into an additive per-feature step function.
    Learning rate is already folded into XGBoost leaf values, so summing leaf
    values reconstructs the margin exactly (verified by assertion elsewhere).
    """
    df = booster.trees_to_dataframe()
    # base margin (XGBoost >=2 stores base_score as a probability, sometimes
    # as a bracketed vector string like '[1.34E-1]')
    try:
        cfg = json.loads(booster.save_config())
        raw = str(cfg["learner"]["learner_model_param"]["base_score"])
        raw = raw.strip().lstrip("[").rstrip("]").split(",")[0]
        base_score = float(raw)
    except Exception:
        base_score = 0.5
    intercept = logit(np.array([base_score]))[0] if 0 < base_score < 1 else float(base_score)

    # gather split thresholds per feature
    splits = {f: [] for f in feature_names}
    stump_specs = []  # (feature, threshold, leaf_yes, leaf_no, leaf_missing)
    for tree_id, g in df.groupby("Tree"):
        root = g[g["Node"] == 0].iloc[0]
        feat = root["Feature"]
        if feat == "Leaf":
            # constant tree (no split) -> adds a scalar to every row
            intercept += float(root["Gain"])
            continue
        thr = float(root["Split"])
        node_map = {int(r["Node"]): r for _, r in g.iterrows()}
        yes_id = int(str(root["Yes"]).split("-")[-1])
        no_id = int(str(root["No"]).split("-")[-1])
        miss_id = int(str(root["Missing"]).split("-")[-1])
        leaf_yes = float(node_map[yes_id]["Gain"])   # x < thr
        leaf_no = float(node_map[no_id]["Gain"])      # x >= thr
        leaf_missing = float(node_map[miss_id]["Gain"])
        splits.setdefault(feat, []).append(thr)
        stump_specs.append((feat, thr, leaf_yes, leaf_no, leaf_missing))

    used = [f for f in feature_names if len(splits.get(f, [])) > 0]
    edges = {f: np.unique(np.array(splits[f], dtype=float)) for f in used}
    contribs = {f: np.zeros(len(edges[f]) + 1, dtype=float) for f in used}
    missing = {f: 0.0 for f in used}

    for (feat, thr, ly, ln, lm) in stump_specs:
        e = edges[feat]
        # bin b covers [e[b-1], e[b]); representative point = midpoint
        left = np.concatenate([[e[0] - 1.0], e])
        right = np.concatenate([e, [e[-1] + 1.0]])
        mids = (left + right) / 2.0
        add = np.where(mids < thr, ly, ln)
        contribs[feat] += add
        missing[feat] += lm

    return Scorecard(intercept, used, edges, contribs, missing)


# ----------------------------------------------------------------------------
# Standard quantile binning (used by the WoE baseline and 'standard' structure)
# ----------------------------------------------------------------------------
def quantile_edges(x, n_bins):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    e = np.unique(np.quantile(x, qs))
    return e


def woe_transform(x, y, edges, eps=0.5):
    """Weight-of-evidence transform of feature x given bin edges and labels y.
    Returns (woe_values_per_row, woe_table dict bin->woe)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.digitize(np.where(np.isnan(x), -np.inf, x), edges, right=False)
    tot_pos = y.sum() + eps
    tot_neg = (1 - y).sum() + eps
    woe_map = {}
    out = np.zeros(len(x))
    for b in np.unique(idx):
        m = idx == b
        pos = y[m].sum() + eps
        neg = (1 - y[m]).sum() + eps
        woe = np.log((pos / tot_pos) / (neg / tot_neg))
        woe_map[int(b)] = woe
        out[m] = woe
    return out, woe_map


# ----------------------------------------------------------------------------
# Binning structure (degree of freedom 1)
#   'standard'     -> quantile edges of the raw feature (greedy/agnostic)
#   'TFM_induced'  -> edges placed where the TEACHER probability changes, via a
#                     per-feature regression tree fit on p^T (H2 structure).
# Both return a dict feature -> sorted edge array; apply_binning maps a raw
# feature to its ordinal bin index (NaN preserved for native missing handling).
# ----------------------------------------------------------------------------
def fit_binning(X, target, features, n_bins=5, method="standard",
                random_state=RANDOM_STATE):
    from sklearn.tree import DecisionTreeRegressor
    binning = {}
    for f in features:
        x = X[f].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        if method == "standard":
            edges = quantile_edges(x[ok], n_bins)
        elif method == "TFM_induced":
            tree = DecisionTreeRegressor(max_leaf_nodes=max(2, n_bins),
                                         min_samples_leaf=max(20, len(x) // 100),
                                         random_state=random_state)
            tree.fit(x[ok].reshape(-1, 1), np.asarray(target)[ok])
            thr = tree.tree_.threshold[tree.tree_.feature == 0]
            edges = np.unique(np.sort(thr.astype(float)))
        else:
            raise ValueError(method)
        if len(edges) == 0:                       # degenerate -> single bin
            edges = np.array([np.nanmedian(x)])
        binning[f] = edges
    return binning


def apply_binning(X, binning):
    """Map raw features to ordinal bin indices (float), preserving NaN."""
    out = {}
    for f, edges in binning.items():
        x = X[f].to_numpy(dtype=float)
        nan = np.isnan(x)
        idx = np.digitize(np.where(nan, 0.0, x), edges, right=False).astype(float)
        idx[nan] = np.nan
        out[f] = idx
    return pd.DataFrame(out, index=X.index)


def fit_stump_scorecard(Xtr_b, target, Xes_b, yes_hard, monotone=None,
                        eta=0.2, max_rounds=500, esr=30, fixed_rounds=None):
    """Shared distillation engine: depth-1 soft-logistic boosting with AUC
    early stopping on the es split (hard labels), returned as a lossless
    Scorecard. `fixed_rounds` overrides early stopping (device 2, iso-AUC).
    Requires xgboost."""
    import xgboost as xgb
    dtrain = xgb.DMatrix(Xtr_b, label=np.asarray(target, dtype=float),
                         feature_names=list(Xtr_b.columns))
    deval = xgb.DMatrix(Xes_b, label=np.asarray(yes_hard, dtype=float),
                        feature_names=list(Xes_b.columns))
    params = xgb_base_params(
        base_score=float(np.clip(np.mean(target), 1e-3, 1 - 1e-3)), eta=eta)
    if monotone is not None:
        params["monotone_constraints"] = "(" + ",".join(
            str(int(monotone.get(f, 0))) for f in Xtr_b.columns) + ")"
    if fixed_rounds is not None:
        bst = xgb.train(params, dtrain, num_boost_round=int(fixed_rounds),
                        verbose_eval=False)
    else:
        bst = xgb.train(params, dtrain, num_boost_round=max_rounds,
                        evals=[(deval, "es")], early_stopping_rounds=esr,
                        verbose_eval=False)
    sc = extract_scorecard_from_xgb(bst, list(Xtr_b.columns))
    sc._best_rounds = int(getattr(bst, "best_iteration", bst.num_boosted_rounds()) or
                          bst.num_boosted_rounds())
    return sc


if __name__ == "__main__":
    print("common.py loaded")
