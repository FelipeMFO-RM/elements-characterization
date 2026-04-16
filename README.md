# Elements Characterization

Unsupervised characterization and classification of copper alloy samples using
OES (Optical Emission Spectrometry) composition data from the SPECTRO Cu-10-F
method. (Information provided by the suppliers of grains/scrap).

---

## Objective

Given a set of pure copper samples with measured elemental compositions, build
an unsupervised pipeline that:

1. **Groups samples** into meaningful families based on their impurity profiles.
2. **Handles measurement uncertainty** — OES results carry standard deviations
   per element; this enables the application of Monte Carlo augmentation to propagate this uncertainty into the clustering.
3. **Ranks elements by relevance** — not all 26 measured elements carry equal
   predictive power; a tier system separates severe conductivity drivers from
   low-impact or near-zero-variance elements.
4. **Classifies new samples** — once clusters are fitted, any incoming
   composition vector can be projected into the same space and assigned to the
   most appropriate cluster without re-training.

The end goal is a robust, physics-informed fingerprinting system for copper
stock that can feed features in the supervised models to control key variables
(conductivity and tensile strength) prediction.

---

## Reasoning

### Why unsupervised learning?

At the start of this project there are no labelled quality outcomes
(conductivity %, tensile strength, etc.) for the 9 available samples.
Unsupervised clustering lets us discover natural groupings in the impurity
profile space before any labelled data is collected. These groups become the
hypothesis that future supervised models will refine.

### Why Monte Carlo augmentation?

OES measurements are not exact values — each element comes with a mean (`val`)
and standard deviation (`sd`). With only 9 samples, fitting a clustering model
on point estimates is statistically fragile. Monte Carlo augmentation draws
`N` realisations from each element's normal distribution, expanding the
effective dataset to `9 × N` rows while respecting the measurement
uncertainty. Two strategies are compared:

| Dataset | `below_limit_zero` | Description |
|---|---|---|
| A | `False` | 9 original point estimates — baseline |
| B | `True` | 900 MC draws; below-detection elements fixed at 0.0 |
| C | `False` | 900 MC draws; below-detection elements also sampled from N(val, sd) |

OBS: The N as 100 is set as an example, it is a flexible hyperparameter.

### Why four clustering algorithms?

Each algorithm makes different assumptions about cluster shape and size. Running
all four in parallel on the same data — and comparing their Silhouette scores —
gives a consensus view rather than trusting a single method:

| Algorithm | Key assumption |
|---|---|
| K-Means | Spherical, equal-variance clusters |
| K-Means++ | Same as K-Means, smarter initialisation |
| Hierarchical | No shape assumption; agglomerative (Ward linkage) |
| GMM | Elliptical clusters; probabilistic soft assignment |

EDIT 16/04: K-Means ++ and K-Means show almost similar performance, in future versions
work with only one of them for the sake of simplicity.

### Why a tier and discard system?

Not all 26 elements contribute equally to the properties of copper. Some are
always at the detection limit (zero variance in this dataset) and provide no
discriminative signal. Others have strong physical reasons to drive conductivity
or mechanical behaviour. The tier system makes the feature-selection choice
explicit and reproducible, rather than relying solely on statistical variance
filtering.

The tier system has the tiers, the elements that were chosen as more impactful.
```
Which elements have the biggest impact on conductivity?

In copper, electrical/thermal conductivity is overwhelmingly governed by elements that go into solid solution rather than forming precipitates. The most damaging impurities to conductivity, roughly ranked by their resistivity contribution per ppm in copper, are:

Tier 1 — Severe impact (high resistivity coefficient): P, Fe, Si, As, Ti, Sb. Phosphorus is the classic killer — even 20-40 ppm of P in solid solution can drop conductivity from 101% IACS to ~99% IACS. Iron and silicon are also very potent. Looking at your data, Fe ranges from ~9 ppm (1A sac1) to ~15 ppm (1B sac4), and Si ranges from ~8 to 17 ppm. These are significant.

Tier 2 — Moderate impact: Sn, Mn, Ni, Cr, Al. Tin is interesting because your samples show a huge range — from below detection (<20 ppm) in the 1A samples up to 199 ppm in 1B sac4 and 195 ppm in 1B sac6. That's a 10x spread and will definitely show up in conductivity.

Tier 3 — Lower impact but present: Ag, Zn, Pb, Mg, B, Te. Silver is actually nearly transparent to conductivity (it barely affects it), so despite being present at ~73-96 ppm in some samples, Ag matters less than you'd think.

For your models, the key conductivity drivers from your data are likely Sn, Fe, Si, Pb, and Ag — not because they're all equally potent per atom, but because they show the most variance across your samples.

---

Which elements have the biggest impact on tensile strength?

Strengthening in copper comes from solid solution hardening and, for some elements, precipitation hardening or grain boundary effects.

The most potent strengtheners in your composition range are Sn (strong solid solution hardener — and you have big variance here), Pb (doesn't strengthen much but drastically affects ductility and hot workability — your 2N samples have ~300-330 ppm Pb vs. <3 ppm in the 1A/1B samples, which is a massive difference), Fe (can form precipitates that pin grain boundaries), and Si and P (both solid solution strengtheners).

The 2N samples are clearly a different material class from the 1A and 1B — much higher Pb, higher Sn, higher Zn, higher Ni, higher Ag. The 2N samples are essentially lower purity copper (99.67–99.94%) and will have measurably different mechanical properties.


```

And the discard logic: Several elements are at or below detection limits across all samples and show essentially zero variance: Mn (< 0.00040 everywhere), Cr (< 0.00030 everywhere), Cd (< 0.00030 everywhere), Be (< 0.00010 everywhere), S (< 0.00020 everywhere), Al (< 0.00050 everywhere), Au (< 0.00060 everywhere), Zr (< 0.00030 everywhere), Pt (< 0.0020 everywhere), Co (< 0.0010 everywhere), Ti (< 0.00020 mostly, one at 0.00024).

From a modeling perspective, if a feature has near-zero variance, it provides no predictive power. You could safely drop these ~11 elements and keep roughly 15 that show meaningful variation.


---

## Information Gathered (from the first samples)

### Samples

9 samples measured with the SPECTRO Cu-10-F OES equipment (date: 04/12/2025).
Each measurement reports mean concentration in %, standard deviation, and a
detection-limit flag per element.

| Sample | Group | Cu (%) | Notable characteristics |
|---|---|---|---|
| 1A sac1 | 1A | 99.98 | Highest purity; all impurities near or below detection limit |
| 1A sac2 | 1A | 99.98 | Identical impurity profile to 1A sac1 |
| 1B sac1 | 1B | 99.97 | First appearance of measurable Sn (137 ppm) |
| 1B sac2 | 1B | 99.98 | Moderate Sn (37 ppm), low impurity variance |
| 1B sac3 | 1B | 99.97 | First measurable Pb (100 ppm); moderate Sn |
| 1B sac4 | 1B | 99.95 | Highest Sn (199 ppm), Zn (101 ppm), and Pb (53 ppm) in group |
| 1B sac6 | 1B | 99.96 | High Sn (195 ppm) with nearly zero Pb |
| 2N sac1 | 2N | 99.94 | High Pb (332 ppm); elevated Zn, Ni above detection limit |
| 2N sac2 | 2N | 99.67 | Extreme Pb (3020 ppm) — clearly the lowest purity sample |

**Natural groups** (validated by hierarchical clustering):
- **Group 1A** — highest purity, all impurities at or below detection limit.
  Likely cathode-grade or ETP copper.
- **Group 1B** — intermediate purity, characterised by meaningful Sn content
  (33–199 ppm). Probably a different production lot or supplier.
- **Group 2N** — lowest purity (99.67–99.94 %), dominated by very high Pb
  (332–3020 ppm). Likely recycled or secondary copper.

### Element tiers

Defined in [`config/elements_considerations.py`](config/elements_considerations.py)
based on the known resistivity contribution per ppm in solid-solution copper
and on the observed variance across the 9 samples.

#### Tier 1 — Severe conductivity impact
> P, Fe, Si, As, Sb

High resistivity coefficient per ppm. Phosphorus is the classic killer — even
20–40 ppm can drop conductivity from 101 % IACS to ~99 % IACS. Fe and Si are
similarly potent. As and Sb are solid-solution embrittlers that also affect
conductivity.

#### Tier 2 — Moderate impact
> Sn, Mn, Ni, Cr, Al

Sn shows the largest absolute variance in this dataset (~2 ppm in 1A up to
199 ppm in 1B sac4) and is a strong solid-solution hardener. Mn, Cr, and Al
are at detection limits in this dataset but are included here as domain
knowledge — they matter in other copper datasets.

#### Tier 3 — Lower impact, but variable
> Ag, Zn, Pb, Mg, B, Te

Ag is nearly transparent to conductivity, but Pb is the key discriminator
between the 2N group and all others (3020 ppm vs < 3 ppm). Zn, Mg, B, and Te
show moderate variance and contribute to mechanical property differences.

#### Discard — near-zero variance in this dataset
> Cu, Mn, Cr, Cd, Be, S, Al, Au, Zr, Pt, Co, Ti, Bi

These elements are at or below their detection limits across all 9 samples.
Cu is the base matrix (~99 %) and dominates any distance metric without
providing discriminative information. Removing this list allows clustering
algorithms to focus on the true impurity variation.

> **Note:** Tier 2 and Discard overlap intentionally. Tier lists reflect domain
> physics; the Discard list reflects the statistical reality of *this* dataset.
> A future dataset with higher Mn variation would promote Mn out of Discard.

---

## Future Applications

### Classifying new samples

Once clusters have been fitted on the reference dataset (e.g. Dataset C —
MC-augmented with `below_limit_zero=False`), any new OES measurement can be
classified without re-training:

```python
from src.feature_engineering.FeatureEngineering import FeatureEngineering as feng
from src.modeling.modeling import Modeling as modl

# Build the reference clusters once
df_ref = feng.build_dataset(ALL_SAMPLES, below_limit_zero=False,
                             mc_augment=True, n_mc=100, seed=42)
_, results_ref = modl.run_all(df_ref, ks=(2, 3, 4), seed=42)

# Classify a new measurement dict
df_new = feng.build_dataset(NEW_SAMPLES, below_limit_zero=False)
predictions = modl.predict_new(df_new, results_ref, k=3)
```

`predict_new` uses the scaler fitted on the reference dataset so the new
sample is projected into the same standardised space. For Hierarchical
clustering (which has no native `predict`), nearest-centroid assignment is used
automatically.

### Tier-based re-runs

The feature-selection knobs at the top of each notebook let you re-run the
entire pipeline with a single change:

```python
# Tier 1 only (most physically significant features)
TIERS          = ["tier1"]
DROP_DISCARDED = True

# Tier 1 + 2
TIERS          = ["tier1", "tier2"]
DROP_DISCARDED = True
```

This directly answers the question: *does adding moderately relevant elements
change which group a sample belongs to?*

### Towards supervised prediction

When laboratory measurements (conductivity % IACS, Vickers hardness, tensile
strength) become available for the 9 samples, the cluster labels generated here
become a compact categorical feature — or a validation target. Recommended
path:

1. Collect % IACS via four-point probe or eddy current for each of the 9 samples.
2. Use cluster label + raw element concentrations as features.
3. Train a regression model (XGBoost, GPR) on conductivity as the target.
4. Validate: do samples within the same cluster have similar conductivity?

### Expanding the reference set

New samples should be added to `compositions.py` following the existing format
and `ALL_SAMPLES` dict updated. The pipeline re-runs without any other changes.

---

## How to Initialize

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/FelipeMFO-RM/elements-characterization.git
cd elements-characterization

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch JupyterLab
jupyter lab
```

### Repository structure

```
elements-characterization/
├── config/
│   └── elements_considerations.py   # Tier lists and DISCARD definition
├── data/
│   └── raw/
│       └── elements_composition/
│           └── compositions.py      # All 9 OES measurements
├── notebooks/
│   ├── exploration/
│   │   └── eda_clusterization.ipynb # EDA and initial visualisation
│   └── modeling/
│       └── modeling_clusterization.ipynb  # Full clustering pipeline
├── src/
│   ├── DataLoader.py                # to_flat: raw dict → flat row
│   ├── feature_engineering/
│   │   └── FeatureEngineering.py    # monte_carlo, build_dataset, tier filters
│   ├── modeling/
│   │   └── modeling.py              # run_all, predict_new
│   └── visualization/
│       └── Plots.py                 # All plotting methods
└── requirements.txt
```

### TODO — Setup script (next steps)

The following items are planned for a `scripts/setup.py` or `Makefile` that
will automate environment configuration end-to-end:

```
TODO: scripts/setup.py
──────────────────────────────────────────────────────────────────
[ ] Create virtualenv and install requirements.txt
[ ] Enable to change compositions.py in case there are more than 9 samples
[ ] Run the full clustering pipeline headlessly and save
    summary tables to data/outputs/ as CSV
[ ] Generate a baseline output (classifications + PCA plots)
    for all the possible datasets (A, B, C), tier combinations receiving new inputs
[ ] Do the point above as a python runnable in-out / plug-and-play script
──────────────────────────────────────────────────────────────────
```
