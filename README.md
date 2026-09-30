# Distilling Transformers into Scorecards

Reproducibility code for the paper **"Distilling Transformers into Scorecards:
Bridging Tabular Foundation Models and Interpretable Additive Credit Scoring"**
(under review). *Author, affiliation, and funding information are withheld for
double-blind peer review.*

A tabular foundation model (TFM) is a strong but opaque teacher; a regulated
credit **scorecard** — attributes binned, mapped to points, and summed — is a
transparent but constrained student. This repository turns the first into the
second and measures exactly what is lost.

The method rests on two ideas:

- **Lossless reduction (Map A).** A depth-one gradient-boosting ensemble *is*
  already an additive scorecard: grouping the stumps by split feature and summing
  their leaf contributions is an exact algebraic identity, verified to
  floating-point precision. Only the final integer rounding loses information.
- **Loss dualization.** The *approximation* loss of the constrained additive
  class (Map B) and the *quantization* loss of integerization (Map A round) are
  driven by different design dials, reported on separate axes, and **never
  summed**.

On top of this, the teacher is distilled into the scorecard by **target
substitution**; three sequential gates separate genuine *calibration transfer*
from label softness and from discrimination. Notebook `08` adds a robustness
study: incumbent/glassbox comparators, a subgroup-fairness audit, and replication
on a second portfolio.

---

## Repository layout

```
scorecard-distillation/
├── notebooks/
│   ├── config.py                    # all experiment settings live here
│   ├── common.py                    # grayscale style, lossless extraction,
│   │                                #   DeLong, Brier–Murphy, ECE, PDO
│   │                                #   quantization, binning, bootstrap CIs
│   ├── data_loaders.py              # dataset loaders + leakage handling
│   ├── 00_data_prep.ipynb           # load, triple split, WoE, monotone screening
│   ├── 01_teacher_tfm.ipynb         # TFM teacher inference + calibration
│   ├── 02_lossless_reduction.ipynb  # Map A: exact reduction, step functions
│   ├── 03_distillation_engine.ipynb # target substitution, 2×2 ablation, temperature
│   ├── 04_quantization.ipynb        # Map A round: three levels of quantization loss
│   ├── 05_decision_simulation.ipynb # policy-space decision-impact simulation
│   ├── 06_calibration_gates.ipynb   # calibration-transfer gates α / β / γ
│   ├── 07_pareto_synthesis.ipynb    # complexity–performance Pareto, loss ledger
│   └── 08_robustness_and_baselines.ipynb  # comparators, fairness, external portfolio
├── results/
│   ├── figures/                     # grayscale, dpi 600, PNG + PDF
│   └── tables/                      # CSV + LaTeX
├── requirements.txt
├── LICENSE
└── README.md
```

Raw and processed data are **not** included (see *Data*). Run notebook `00`
first; notebooks `01`–`08` read its artifacts from `data/processed/`.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

The core pipeline (reduction, distillation, quantization, gates, figures) runs on
**CPU only**. The student ensembles are deliberately fit with an *exact* tree
method on CPU, which is what makes the lossless reduction hold at the level of the
reported thresholds — a histogram/GPU tree method quantizes split points and
breaks exactness.

Optional dependencies:

```bash
# real TabPFN v2 teacher (GPU) — match the CUDA build to your driver
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install tabpfn

# glassbox comparator (EBM) used in notebook 08
pip install interpret
```

A GPU is used **only** for the TabPFN teacher; everything else is CPU.

---

## Data

The experiments use two public retail-credit datasets, neither redistributed here:

- **Home Credit Default Risk** (`application_train.csv`), from its Kaggle
  competition page — the primary portfolio. Place the CSV at
  `data/raw/application_train.csv`.
- **Taiwan credit-card default** (UCI / OpenML `default-of-credit-card-clients`) —
  the external-validation portfolio, fetched automatically by notebook `08` via
  OpenML.

The Home Credit loader recognizes `TARGET` (1 = default) and `SK_ID_CURR` (id) and
uses the numeric features (the external bureau scores `EXT_SOURCE_*`, `AMT_*`,
`DAYS_*` carry most of the signal). Missing values are preserved and handled
natively by both teacher and student.

---

## Reproducing the results

Open `notebooks/` and run `00` → `08` in order:

| Notebook | Produces | Paper artifact |
| --- | --- | --- |
| `00_data_prep` | triple split, WoE, monotone screening | feature-monotonicity figure |
| `01_teacher_tfm` | teacher probabilities + calibration | teacher-quality table, reliability diagram |
| `02_lossless_reduction` | exact reduction check, per-feature step functions | lossless-verification table, step-function figure |
| `03_distillation_engine` | 2×2 ablation, gap recovery, DeLong | ablation table + figure |
| `04_quantization` | three-level quantization loss | quantization table + Pareto figure |
| `05_decision_simulation` | approval-rate policy space | approved-default-rate + migration figures |
| `06_calibration_gates` | gates α / β / γ | gates table + gate-β reliability figure |
| `07_pareto_synthesis` | complexity–performance frontier, loss ledger | complexity-Pareto figure |
| `08_robustness_and_baselines` | logistic + EBM comparators, gender fairness, Taiwan replication | comparators / fairness / external tables + figures |

All figures are written to `results/figures/` in grayscale at dpi 600 (PNG and
PDF); all tables to `results/tables/` as CSV and LaTeX. The conceptual/schematic
figures in the paper (pipeline, reduction identity, loss dualization, ablation
design, gate flow) are drawn separately and are not produced by these notebooks.

---

## Configuration (`notebooks/config.py`)

Everything is set in one place. The settings used for the reported experiments:

| Key | Value | Meaning |
| --- | --- | --- |
| `DATASET` | `"home_credit"` | primary dataset selector |
| `FALLBACK_TO_SYNTHETIC` | `False` | never silently substitute synthetic data |
| `STRICT_TEACHER` | `True` | never silently fall back to a proxy teacher |
| `SUBSAMPLE_N` (home_credit) | `16000` | class-ratio-preserving subsample |
| `MAX_FEATURES` | `25` | top-K features by WoE–target correlation |
| `TEACHER_ROW_CAP` | `10000` | TabPFN v2 in-context cap |
| `N_REPEATS` | `50` | bootstrap repetitions for the ablation |
| `N_BINS` / `LAMBDA_MIX` / `N_BOOT` | `5` / `0.30` / `500` | bins, teacher-mix λ, CI resamples |

The teacher's in-context set is sampled to **preserve the ~8% prevalence** (not
class-balanced), so its probabilities stay calibrated to the deployment base rate;
a balanced context badly over-predicts.

> **Note.** The pipeline is designed to *fail loudly*: with the settings above it
> refuses to run on synthetic data or a proxy teacher, so a demo run can never be
> mistaken for the reported results. A quick synthetic smoke test is available by
> setting `DATASET="synthetic"` and `STRICT_TEACHER=False`.

---

## What notebook 08 checks

- **Comparators.** The distilled scorecard is placed beside the industry-standard
  logistic weight-of-evidence scorecard and a strong glassbox model, the
  explainable boosting machine (EBM). On Home Credit the distilled scorecard leads
  both on discrimination while matching their calibration.
- **Fairness.** A gender audit at a common approval rate reports the adverse-impact
  ratio and equal-opportunity difference; no model breaches the four-fifths
  threshold, and the distilled scorecard is at least as fair as the hard-label
  baseline.
- **External portfolio.** The ablation and the calibration-transfer test are
  replicated on the Taiwan default portfolio. The accuracy decomposition is
  portfolio-specific, but the transfer of calibration is stable.

---

## Notes on scope

The design also contains a loader for an alternative portfolio and a
proxy-teacher path for environments without a GPU; these are development
conveniences and are **not** part of the reported experiments, which use Home
Credit (primary) and Taiwan default (external) with a real TabPFN v2 teacher.

---

## License

See `LICENSE`. A permissive open-source license applies; the copyright line is
kept anonymous during double-blind review and will be completed on
de-anonymization.
