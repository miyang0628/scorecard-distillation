# Distilling Transformers into Scorecards

Reproducibility code for the paper **"Distilling Transformers into Scorecards"**
(under review). *Author, affiliation, and funding information are withheld for
double-blind peer review.*

A tabular foundation model (TFM) is a strong but opaque teacher; a regulated
credit **scorecard** — attributes binned, mapped to points, and summed — is a
transparent but constrained student. This repository turns the first into the
second and measures exactly what is lost, on the public *Home Credit Default
Risk* portfolio.

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
substitution**, and the resulting calibration is traced with three sequential
gates that separate genuine *calibration transfer* from mere label softness and
from discrimination.

---

## Repository layout

```
scorecard-distillation/
├── notebooks/
│   ├── config.py                    # all experiment settings live here
│   ├── common.py                    # shared utilities: grayscale style, lossless
│   │                                #   extraction, DeLong, Brier–Murphy, ECE,
│   │                                #   PDO quantization, binning, bootstrap CIs
│   ├── data_loaders.py              # dataset loaders + leakage handling
│   ├── 00_data_prep.ipynb           # load, triple split, WoE, monotone screening
│   ├── 01_teacher_tfm.ipynb         # TFM teacher inference + calibration
│   ├── 02_lossless_reduction.ipynb  # Map A: exact reduction, three DOF
│   ├── 03_distillation_engine.ipynb # target substitution, 2×2 ablation, temperature
│   ├── 04_quantization.ipynb        # Map A round: three levels of quantization loss
│   ├── 05_decision_simulation.ipynb # policy-space decision-impact simulation
│   ├── 06_calibration_gates.ipynb   # calibration-transfer gates α / β / γ
│   └── 07_pareto_synthesis.ipynb    # complexity–performance Pareto, loss ledger
├── results/
│   ├── figures/                     # grayscale, dpi 600, PNG + PDF
│   └── tables/                      # CSV + LaTeX
├── requirements.txt
├── LICENSE
└── README.md
```

Raw and processed data are **not** included (see *Data*). Run notebook `00`
first; the remaining notebooks read its artifacts from `data/processed/`.

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

A GPU is used **only** for the optional real TabPFN v2 teacher:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install tabpfn
```

---

## Data

The experiments use the public **Home Credit Default Risk** dataset
(`application_train.csv`), available from its Kaggle competition page. The data
are **not redistributed here**; download them under Kaggle's terms and place the
CSV at:

```
data/raw/application_train.csv
```

The loader recognizes `TARGET` (1 = default) and `SK_ID_CURR` (id), and uses the
numeric features (the external bureau scores `EXT_SOURCE_*`, `AMT_*`, `DAYS_*`,
etc. carry most of the signal). Missing values are preserved and handled natively
by both teacher and student.

---

## Reproducing the results

Open `notebooks/` and run `00` → `07` in order:

| Notebook | Produces | Paper artifact |
| --- | --- | --- |
| `00_data_prep` | triple split, WoE, monotone screening | Fig. (feature monotonicity) |
| `01_teacher_tfm` | teacher probabilities + calibration | teacher-quality table, reliability diagram |
| `02_lossless_reduction` | exact reduction check, per-feature step functions | lossless-verification table, step-function figure |
| `03_distillation_engine` | 2×2 ablation, gap recovery, DeLong | ablation table + figure |
| `04_quantization` | three-level quantization loss | quantization table + Pareto figure |
| `05_decision_simulation` | approval-rate policy space | approved-default-rate + migration figures |
| `06_calibration_gates` | gates α / β / γ | gates table + gate-β reliability figure |
| `07_pareto_synthesis` | complexity–performance frontier, loss ledger | complexity-Pareto figure |

All figures are written to `results/figures/` in grayscale at dpi 600 (PNG and
PDF); all tables to `results/tables/` as CSV and LaTeX. The conceptual/schematic
figures in the paper (pipeline, reduction identity, loss dualization, ablation
design, gate flow) are drawn separately and are not produced by these notebooks.

---

## Configuration (`notebooks/config.py`)

Everything is set in one place. The settings used for the reported experiments:

| Key | Value | Meaning |
| --- | --- | --- |
| `DATASET` | `"home_credit"` | dataset selector |
| `FALLBACK_TO_SYNTHETIC` | `False` | never silently substitute synthetic data |
| `STRICT_TEACHER` | `True` | never silently fall back to a proxy teacher |
| `SUBSAMPLE_N` (home_credit) | `16000` | class-ratio-preserving subsample |
| `MAX_FEATURES` | `25` | top-K features by WoE–target correlation |
| `TEACHER_ROW_CAP` | `10000` | TabPFN v2 in-context cap |
| `N_REPEATS` | `50` | bootstrap repetitions for the ablation |
| `N_BINS` / `LAMBDA_MIX` / `N_BOOT` | `5` / `0.30` / `500` | bins, teacher-mix λ, CI resamples |

The teacher's in-context set is sampled to **preserve the ~8% prevalence** (not
class-balanced), so the teacher's probabilities stay calibrated to the deployment
base rate; a balanced context badly over-predicts.

> **Note.** The pipeline is designed to *fail loudly*: with the settings above it
> refuses to run on synthetic data or a proxy teacher, so a demo run can never be
> mistaken for the reported results. A quick synthetic smoke test is available by
> setting `DATASET="synthetic"` and `STRICT_TEACHER=False`.

---

## Notes on scope

The design also contains loaders for an alternative portfolio and a
proxy-teacher path for environments without a GPU; these are development
conveniences and are **not** part of the reported experiments, which use Home
Credit with a real TabPFN v2 teacher throughout.

---

## License

See `LICENSE`. A permissive open-source license applies; the copyright line is
kept anonymous during double-blind review and will be completed on
de-anonymization.
