class DataLoader:
    """Class for loading"""

    @staticmethod
    def to_flat(
        sample: dict[str, dict],
        below_limit_zero: bool = False,
    ) -> dict[str, float]:
        """Convert a detailed sample dict to a flat {element: value} dict.

        Parameters
        ----------
        sample:
            Dict produced by this module, where each value is a sub-dict
            with keys ``val``, ``sd``, ``rsd``, ``below_limit``.
        below_limit_zero:
            If True, elements flagged as below detection limit are set
            to 0.0 in the output; otherwise their nominal ``val`` is
            used as-is.

        Returns
        -------
        dict[str, float]
            Flat mapping of element symbol to concentration (%).

        Examples
        --------
        >>> flat = to_flat(sac_1A_sac1, below_limit_zero=True)
        >>> flat["Zn"]
        0.0
        >>> flat["Cu"]
        99.98
        """
        return {
            elem: (0.0 if (below_limit_zero and info["below_limit"])
                   else info["val"])
            for elem, info in sample.items()
        }
