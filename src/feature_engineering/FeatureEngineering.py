import numpy as np
import pandas as pd


class FeatureEngineering:

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

        Notes
        -----
        The Normal approximation is reasonable for OES measurements far from
        the detection limit.  Near the limit (high ``rsd``), the distribution
        is asymmetric; a truncated-normal or log-normal would be more
        appropriate in those cases.

        Elements whose ``sd == 0.0`` are always deterministic regardless of
        ``below_limit_zero``, since there is no empirical spread to sample from.

        Examples
        --------
        >>> from src.feature_engineering.FeatureEngineering import FeatureEngineering as fe
        >>> draws = fe.monte_carlo(sac_1B_sac3, n=5000, seed=42)
        >>> draws["Pb"].mean()   # ≈ 0.0100
        >>> draws["Pb"].std()    # ≈ 0.00049
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
