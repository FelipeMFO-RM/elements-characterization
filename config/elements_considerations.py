# ──────────────────────────────────────────────────────────────────────────────
# Element classification for Cu-10-F SPECTRO samples
# Physical impact on conductivity and mechanical properties of copper.
# Reference: OPUS LLM domain reasoning on Cu impurity effects (April 2026)
# ──────────────────────────────────────────────────────────────────────────────

# Elements at or below detection limits across all 9 samples with essentially
# zero variance — no predictive power for this dataset, safe to exclude.
# Cu is also excluded: it is the base matrix (~99 %) and dominates any
# distance metric without contributing discriminative information.
DISCARD: list[str] = [
    "Cu",   # base matrix — always ~99 %, drives variance trivially
    "Mn",   # always at detection limit (0.0004 %), zero variance
    "Cr",   # always at detection limit (0.0003 %), zero variance
    "Cd",   # always at detection limit (0.0003 %), zero variance
    "Be",   # always at detection limit (0.0001 %), zero variance
    "S",    # always at detection limit (0.0002 %), zero variance
    "Al",   # always at detection limit (0.0005 %), zero variance
    "Au",   # always at detection limit (0.0006 %), zero variance
    "Zr",   # always at detection limit (0.0003 %), zero variance
    "Pt",   # always at detection limit (0.0020 %), zero variance
    "Co",   # always at detection limit (0.0010 %), zero variance
    "Ti",   # almost entirely at detection limit, near-zero variance
    "Bi",   # always at detection limit (0.0005 %), zero variance
]

# Tier 1 — Severe conductivity impact (high resistivity coefficient per ppm)
# P is the classic killer; Fe and Si are also very potent.
# As and Sb contribute meaningfully and show variance across samples.
TIER_1: list[str] = ["P", "Fe", "Si", "As", "Sb"]

# Tier 2 — Moderate conductivity / solid-solution hardening impact
# Sn shows the largest absolute variance across samples (~2–199 ppm range).
# Mn, Cr, Al are flagged in DISCARD for this dataset (zero variance) but are
# kept here as domain knowledge — they matter in other Cu datasets.
TIER_2: list[str] = ["Sn", "Mn", "Ni", "Cr", "Al"]

# Tier 3 — Lower conductivity impact but present and variable across samples
# Pb is critical for distinguishing the 2N group (300+ ppm vs <3 ppm in 1A/1B).
TIER_3: list[str] = ["Ag", "Zn", "Pb", "Mg", "B", "Te"]

# Convenience mapping — pass tier names as strings to FeatureEngineering
TIER_MAP: dict[str, list[str]] = {
    "tier1": TIER_1,
    "tier2": TIER_2,
    "tier3": TIER_3,
}

# ──────────────────────────────────────────────────────────────────────────────
# Periodic group classification (conductivity-reduction studies)
# Source: data/structured/periodic_group_elements_factor.tsv
# Note: group numbering follows copper-alloy industry convention,
#       not modern IUPAC group numbers.
# ──────────────────────────────────────────────────────────────────────────────

PERIODIC_GROUPS: dict[str, list[str]] = {
    "I":    ["Ag", "Au"],           # noble metals — low impact
    "II":   ["Zn", "Cd"],
    "III":  ["Al"],                 # very high factor (500)
    "IV":   ["Si", "Sn", "Pb"],
    "V":    ["P", "As", "Sb", "Bi"],  # P has highest factor (3000)
    "VI":   ["S", "Te"],            # O not measured by SPECTRO Cu-10-F
    "VIII": ["Fe"],                 # factor 140
}

# Elements present in the compositions but absent from the reduction table
UNCATEGORIZED: list[str] = [
    "Mn", "Ni", "Mg", "Cr", "Co", "B", "Ti", "Pt", "Be", "Zr",
]
