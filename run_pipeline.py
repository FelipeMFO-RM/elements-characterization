#!/usr/bin/env python
"""
run_pipeline.py
───────────────
Elements Characterization Pipeline — script entry point.

Edit config.yaml (project root) to set all hyperparameters and define new
samples to classify, then run:

    python run_pipeline.py
    python run_pipeline.py --config path/to/other_config.yaml

Outputs are written to:
    data/outputs/<run_name>_<YYYYMMDD_HHMMSS>/

Output files
────────────
  silhouette_summary.csv          — all Silhouette / Inertia / BIC / AIC scores
  dataset_A_silhouette_heatmap.png
  dataset_A_pca_clusters.png
  dataset_B_silhouette_heatmap.png
  dataset_B_pca_clusters.png
  dataset_C_silhouette_heatmap.png
  dataset_C_pca_clusters.png
  comparison_silhouette_bars.png  — cross-dataset grouped bar chart
  new_samples_predictions.csv     — cluster label per algorithm at reference K
  new_samples_pca.png             — PCA overlay of new samples on fitted clusters
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: saves files, no display pop-up
import matplotlib.pyplot as plt  # noqa: E402 — must come after matplotlib.use()

# ── Project root on sys.path so src.* and config.* imports resolve ──────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.DataLoader import DataLoader as load  # noqa: E402
from src.feature_engineering.FeatureEngineering import (  # noqa: E402
    FeatureEngineering as feng,
)
from src.modeling.modeling import Modeling as modl  # noqa: E402
from src.visualization.Plots import Plots as plot  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_output_dir(cfg: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / "outputs" / f"{cfg.get('run_name', 'run')}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_new_samples(
    cfg: dict,
    all_samples: dict,
    all_elements: list[str],
) -> pd.DataFrame:
    """Build the DataFrame of new samples to classify from config.

    Supports two sources (both can be active simultaneously):
    - ``blends`` — auto-generated convex combinations of existing samples
    - ``manual_samples``— user-defined compositions in config.yaml
    """
    frames = []

    # ── Blend samples ──────────────────────────────────────────────────────
    blend_cfg = cfg["classification"].get("blends", {})
    if blend_cfg.get("enabled", False):
        rng = np.random.default_rng(blend_cfg["seed"])
        flat = {
            name: load.to_flat(s, below_limit_zero=False)
            for name, s in all_samples.items()
        }
        names = list(flat.keys())
        for i in range(blend_cfg["n"]):
            s1, s2 = rng.choice(names, size=2, replace=False)
            alpha = float(rng.uniform(0.2, 0.8))
            row = {e: alpha * flat[s1][e] + (1 - alpha) * flat[s2][e] for e in flat[s1]}
            label = f"blend_{i+1}  ({s1} × {s2}  α={alpha:.2f})"
            frames.append(pd.DataFrame([row], index=[label]))

    # ── Manual samples ─────────────────────────────────────────────────────
    for entry in cfg["classification"].get("manual_samples") or []:
        row = {e: 0.0 for e in all_elements}  # default every element to 0
        row.update(entry["elements"])
        frames.append(pd.DataFrame([row], index=[entry["name"]]))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df.index.name = "sample"
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run(config_path: str) -> None:
    cfg = _load_config(config_path)
    out = _make_output_dir(cfg)
    print(f"\n{'─'*60}")
    print(f"  Elements Characterization Pipeline")
    print(f"  Run name : {cfg.get('run_name', 'run')}")
    print(f"  Output   : {out}")
    print(f"{'─'*60}\n")

    # ── Load compositions ──────────────────────────────────────────────────
    comp_path = str(ROOT / "data" / "raw" / "elements_composition")
    if comp_path not in sys.path:
        sys.path.insert(0, comp_path)
    from compositions import ALL_SAMPLES  # noqa: PLC0415

    SEED = cfg["seed"]
    KS = tuple(cfg["ks"])
    ALGOS = cfg["algorithms"]
    N_MC = cfg["n_mc"]
    TIERS = cfg.get("tiers")  # None or list[str]
    DROP_DISCARDED = cfg.get("drop_discarded", False)

    # ── Build datasets ─────────────────────────────────────────────────────
    print("[1/4] Building datasets...")
    DATASET_META = {
        "A": ("Original — 9 point estimates", False, False),
        "B": ("MC blz=True — 900 rows", True, True),
        "C": ("MC blz=False — 900 rows", False, True),
    }
    datasets: dict[str, pd.DataFrame] = {}
    for key, (label, blz, mc) in DATASET_META.items():
        datasets[key] = feng.build_dataset(
            ALL_SAMPLES,
            below_limit_zero=blz,
            mc_augment=mc,
            n_mc=N_MC,
            seed=SEED,
            tiers=TIERS,
            drop_discarded=DROP_DISCARDED,
        )
        print(f"  Dataset {key}: {datasets[key].shape}  — {label}")

    # ── Clustering ─────────────────────────────────────────────────────────
    print("\n[2/4] Clustering...")
    summaries: dict[str, pd.DataFrame] = {}
    results_all: dict[str, dict] = {}
    for key, df in datasets.items():
        print(f"  Running on Dataset {key}...")
        summaries[key], results_all[key] = modl.run_all(df, ks=KS, seed=SEED)

    # ── Save summary CSV ───────────────────────────────────────────────────
    summary_combined = pd.concat(
        [s.assign(Dataset=k) for k, s in summaries.items()]
    ).reset_index()
    summary_combined.to_csv(out / "silhouette_summary.csv", index=False)
    print(f"\n  Saved: silhouette_summary.csv")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[3/4] Generating plots...")
    for key, (label, _, _) in DATASET_META.items():
        plot.plot_silhouette_heatmap(
            summaries[key],
            title=f"Silhouette — Dataset {key} ({label})",
            output_path=str(out / f"dataset_{key}_silhouette_heatmap.png"),
        )
        plot.plot_clusters_pca(
            df_feat=datasets[key],
            algos=ALGOS,
            results=results_all[key],
            title=f"Dataset {key} — {label}",
            ks=KS,
            output_path=str(out / f"dataset_{key}_pca_clusters.png"),
        )
        print(f"  Dataset {key}: heatmap + PCA saved")

    plot.plot_silhouette_comparison(
        summaries=[
            (summaries["A"], "A · Original"),
            (summaries["B"], "B · MC blz=True"),
            (summaries["C"], "C · MC blz=False"),
        ],
        ks=KS,
        output_path=str(out / "comparison_silhouette_bars.png"),
    )
    print("  Cross-dataset comparison saved")

    # ── New sample classification ──────────────────────────────────────────
    print("\n[4/4] Classifying new samples...")
    ref_key = cfg["classification"]["reference_dataset"]
    ref_k = cfg["classification"]["reference_k"]

    all_elements = list(next(iter(ALL_SAMPLES.values())).keys())
    df_new = _build_new_samples(cfg, ALL_SAMPLES, all_elements)

    if df_new.empty:
        print("  No new samples defined in config — skipping.")
    else:
        print(
            f"  {len(df_new)} new samples  →  "
            f"Dataset {ref_key} clusters  (K={ref_k})"
        )

        # Predictions at reference K (saved as CSV)
        predictions = modl.predict_new(df_new, results_all[ref_key], k=ref_k)
        predictions.to_csv(out / "new_samples_predictions.csv")
        print(f"  Saved: new_samples_predictions.csv")

        # PCA overlay across all Ks
        predictions_by_k = {
            k: modl.predict_new(df_new, results_all[ref_key], k=k) for k in KS
        }
        plot.plot_new_samples_pca(
            df_train=datasets[ref_key],
            df_new=df_new,
            results=results_all[ref_key],
            predictions_by_k=predictions_by_k,
            algos=ALGOS,
            ks=KS,
            title=f"New samples — classified against Dataset {ref_key} clusters",
            output_path=str(out / "new_samples_pca.png"),
        )
        print("  Saved: new_samples_pca.png")

    print(f"\n{'─'*60}")
    print(f"  Done.  All outputs in:")
    print(f"  {out}")
    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elements Characterization Pipeline",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()
    run(args.config)
