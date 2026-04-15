import numpy as np
import pandas as pd

from src.DataLoader import DataLoader as load


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
        from config.elements_considerations import DISCARD
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
        from config.elements_considerations import TIER_MAP
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
