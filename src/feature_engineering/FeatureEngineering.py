import numpy as np
import pandas as pd

from src.DataLoader import DataLoader as load
from config.elements_considerations import DISCARD, TIER_MAP

# Element full-name → symbol mapping for the periodic reduction TSV
_ELEMENT_NAME_TO_SYMBOL: dict[str, str] = {
    "Silver": "Ag", "Gold": "Au", "Zinc": "Zn", "Cadmium": "Cd",
    "Aluminum": "Al", "Silicon": "Si", "Tin": "Sn", "Lead": "Pb",
    "Phosphorus": "P", "Arsenic": "As", "Antimony": "Sb",
    "Bismuth": "Bi", "Oxygen": "O", "Sulphur": "S",
    "Tellurium": "Te", "Iron": "Fe",
}


class FeatureEngineering:

    # ------------------------------------------------------------------
    # Monte Carlo sampling
    # ------------------------------------------------------------------

    @staticmethod
    def monte_carlo(
        sample: dict[str, dict],
        n: int = 1000,
        below_limit_zero: bool = True,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Draw *n* Monte Carlo samples from each element's distribution.

        Parameters
        ----------
        sample:
            Dict produced by ``compositions.py``, where each value has keys
            ``val``, ``sd``, ``rsd``, ``below_limit``.
        n:
            Number of Monte Carlo draws.
        below_limit_zero:
            Controls how elements flagged as ``below_limit=True`` are handled:

            - ``True``  — element is fixed at **0.0** for all draws (the true
              value is unknown beyond "it is below the detection floor", so no
              variance is injected).
            - ``False`` — element is treated as a numerical measurement and
              **sampled from N(val, sd)** using the reported detection-limit
              value as the mean.  Use this when you want the sampling to
              propagate the uncertainty even for near-limit elements.
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Shape ``(n, n_elements)``.  Each row is one realisation of the
            full composition vector.
        """
        rng = np.random.default_rng(seed)

        columns: dict[str, np.ndarray] = {}
        for elem, info in sample.items():
            is_below = info["below_limit"]
            val, sd = info["val"], info["sd"]

            if sd == 0.0 or (is_below and below_limit_zero):
                fill = 0.0 if (is_below and below_limit_zero) else val
                columns[elem] = np.full(n, fill)
            else:
                columns[elem] = rng.normal(loc=val, scale=sd, size=n)

        return pd.DataFrame(columns)

    # ------------------------------------------------------------------
    # Feature-selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_discarded(df: pd.DataFrame) -> pd.DataFrame:
        """Drop DISCARD elements from *df* (columns only, in-place safe).

        Non-numeric columns (e.g. ``sample_name``) are never touched.
        Silently skips elements not present in the DataFrame.
        """
        return df.drop(columns=[c for c in DISCARD if c in df.columns])

    @staticmethod
    def _filter_tiers(df: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
        """Keep only the numeric columns belonging to *tiers*.

        Non-numeric columns (e.g. ``sample_name``) are always preserved.
        Column order follows the original DataFrame.

        Parameters
        ----------
        tiers:
            One or more of ``"tier1"``, ``"tier2"``, ``"tier3"``.
        """
        keep_elements: set[str] = set()
        for t in tiers:
            keep_elements.update(TIER_MAP[t.lower()])

        non_numeric = df.select_dtypes(exclude="number").columns.tolist()
        feat_cols = [c for c in df.select_dtypes(include="number").columns
                     if c in keep_elements]
        return df[non_numeric + feat_cols]

    # ------------------------------------------------------------------
    # Master dataset builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_dataset(
        all_samples: dict,
        below_limit_zero: bool = False,
        mc_augment: bool = False,
        n_mc: int = 100,
        seed: int = 42,
        tiers: list[str] | None = None,
        drop_discarded: bool = False,
    ) -> pd.DataFrame:
        """Build a feature-ready DataFrame from raw sample dicts.

        Combines raw loading, optional Monte Carlo augmentation, optional
        DISCARD filtering, and optional tier-based column selection into a
        single call so notebooks stay clean.

        Parameters
        ----------
        all_samples:
            ``{sample_name: sample_dict}`` mapping from ``compositions.py``.
        below_limit_zero:
            Forwarded to ``DataLoader.to_flat`` (no-MC path) or
            ``FeatureEngineering.monte_carlo`` (MC path).
            ``True``  → below-detection elements set to 0.0.
            ``False`` → nominal detection-limit value is used as-is.
        mc_augment:
            ``False`` → one row per sample (point estimates,
                        Dataset A style).
            ``True``  → ``n_mc`` Monte Carlo draws per sample
                        (Datasets B/C style).
        n_mc:
            Number of MC draws per sample. Ignored when ``mc_augment=False``.
        seed:
            Random seed forwarded to the MC sampler.
        tiers:
            ``None``            → keep all elements (default).
            ``["tier1"]``       → Tier 1 elements only.
            ``["tier1","tier2"]``→ Tier 1 + Tier 2.
            ``["tier1","tier2","tier3"]`` → all tier elements (same as None
                                            unless ``drop_discarded=True``).
        drop_discarded:
            ``True``  → remove DISCARD elements before returning.
            ``False`` → keep all columns (default, current behaviour).

        Returns
        -------
        pd.DataFrame
            Numeric feature columns only (no ``sample_name`` metadata).
            For the no-MC path the index is the sample name.
            For the MC path the index is a plain integer range.
        """
        fe = FeatureEngineering

        # ── 1. Build raw DataFrame ────────────────────────────────────
        if mc_augment:
            frames = []
            for name, sample in all_samples.items():
                draws = fe.monte_carlo(sample, n=n_mc,
                                       below_limit_zero=below_limit_zero,
                                       seed=seed)
                frames.append(draws)
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.DataFrame(
                {name: load.to_flat(sample, below_limit_zero=below_limit_zero)
                 for name, sample in all_samples.items()}
            ).T
            df.index.name = "sample"

        # ── 2. Drop near-zero-variance elements (optional) ────────────
        if drop_discarded:
            df = fe._drop_discarded(df)

        # ── 3. Filter to selected tiers (optional) ────────────────────
        if tiers is not None:
            df = fe._filter_tiers(df, tiers)

        return df

    # ------------------------------------------------------------------
    # Periodic group feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def load_periodic_table(tsv_path: str) -> pd.DataFrame:
        """Load and clean the periodic-group conductivity-reduction TSV.

        Parameters
        ----------
        tsv_path:
            Path to ``periodic_group_elements_factor.tsv``.

        Returns
        -------
        pd.DataFrame with columns:
            ``symbol``       — element symbol (e.g. ``"Fe"``)
            ``group``        — periodic group string (e.g. ``"V"``)
            ``factor``       — conductivity reduction factor
            ``atomic_weight``— atomic weight
        """
        group_fixes = {"1V": "IV"}   # correct typo in source file
        rows = []
        with open(tsv_path, encoding="utf-8") as f:
            next(f)                  # skip header line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip().rstrip(".").strip()
                         for p in line.split(";")]
                if len(parts) < 4:
                    continue
                raw_group = parts[0]
                raw_name = parts[1]
                try:
                    factor = float(parts[2])
                    aw = float(parts[3])
                except ValueError:
                    continue
                group = group_fixes.get(raw_group, raw_group)
                symbol = _ELEMENT_NAME_TO_SYMBOL.get(raw_name)
                if symbol is not None:
                    rows.append({"symbol": symbol, "group": group,
                                 "factor": factor, "atomic_weight": aw})
        return pd.DataFrame(rows)

    @staticmethod
    def build_group_pct(
        df: pd.DataFrame,
        periodic_df: pd.DataFrame,
        include_other: bool = True,
    ) -> pd.DataFrame:
        """Aggregate element concentrations by periodic group.

        Parameters
        ----------
        df:
            Composition DataFrame (numeric columns = element symbols).
        periodic_df:
            Output of ``load_periodic_table``.
        include_other:
            If ``True``, an extra column ``group_Other_pct`` sums all
            numeric columns not mapped to any group (excluding Cu).

        Returns
        -------
        pd.DataFrame with columns ``group_{G}_pct`` for each group G,
        index preserved from *df*.
        """
        sym_to_group = dict(zip(periodic_df["symbol"], periodic_df["group"]))
        known = set(periodic_df["symbol"])
        base_metals = {"Cu"}
        num_cols = df.select_dtypes(include="number").columns.tolist()

        result: dict[str, pd.Series] = {}
        for group in sorted(periodic_df["group"].unique()):
            elems = [c for c in num_cols if sym_to_group.get(c) == group]
            result[f"group_{group}_pct"] = (
                df[elems].sum(axis=1) if elems
                else pd.Series(0.0, index=df.index)
            )

        if include_other:
            other = [c for c in num_cols
                     if c not in known and c not in base_metals]
            if other:
                result["group_Other_pct"] = df[other].sum(axis=1)

        return pd.DataFrame(result, index=df.index)

    @staticmethod
    def build_reduction_impact(
        df: pd.DataFrame,
        periodic_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute conductivity reduction impact per periodic group.

        For each group G:
            ``reduction_G = Σ (conc_i × factor_i)``

        Also adds ``total_reduction`` = sum across all groups.

        Parameters
        ----------
        df:
            Composition DataFrame (numeric columns = element symbols).
        periodic_df:
            Output of ``load_periodic_table``.

        Returns
        -------
        pd.DataFrame with one column per group plus ``total_reduction``,
        index preserved from *df*.
        """
        sym_to_group = dict(zip(periodic_df["symbol"], periodic_df["group"]))
        sym_to_factor = dict(zip(periodic_df["symbol"], periodic_df["factor"]))
        num_cols = df.select_dtypes(include="number").columns.tolist()

        result: dict[str, pd.Series] = {}
        for group in sorted(periodic_df["group"].unique()):
            elems = [c for c in num_cols
                     if sym_to_group.get(c) == group]
            if elems:
                result[f"reduction_{group}"] = sum(
                    df[c] * sym_to_factor[c] for c in elems
                )
            else:
                result[f"reduction_{group}"] = pd.Series(0.0, index=df.index)

        out = pd.DataFrame(result, index=df.index)
        out["total_reduction"] = out.sum(axis=1)
        return out

    @staticmethod
    def build_periodic_features(
        df: pd.DataFrame,
        periodic_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine group percentage and reduction impact into one DataFrame.

        Convenience wrapper around ``build_group_pct`` +
        ``build_reduction_impact``.
        """
        fe = FeatureEngineering
        return pd.concat(
            [fe.build_group_pct(df, periodic_df),
             fe.build_reduction_impact(df, periodic_df)],
            axis=1,
        )
