from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class Modeling:
    """Unsupervised clustering algorithms for element-composition data.

    All public ``run_*`` methods share the same contract:
    - Accept a numeric ``pd.DataFrame`` (samples × elements).
    - Scale features internally via ``StandardScaler`` (zero-variance
      columns are dropped so Cu ≈ 99.9 % does not dominate).
    - Return a result ``dict`` with at minimum ``labels`` and ``silhouette``.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale(df: pd.DataFrame) -> np.ndarray:
        """Drop zero-variance columns then StandardScale."""
        numeric = df.select_dtypes(include="number")
        non_const = numeric.loc[:, numeric.std() > 0]
        return StandardScaler().fit_transform(non_const.values)

    # ------------------------------------------------------------------
    # Individual algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def run_kmeans(
        df: pd.DataFrame,
        k: int,
        seed: int = 42,
    ) -> dict:
        """Vanilla K-Means (``init='random'``, 10 re-starts)."""
        X = Modeling._scale(df)
        model = KMeans(n_clusters=k, init="random", n_init=10,
                       random_state=seed)
        labels = model.fit_predict(X)
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
            "inertia": model.inertia_,
        }

    @staticmethod
    def run_kmeans_plus(
        df: pd.DataFrame,
        k: int,
        seed: int = 42,
    ) -> dict:
        """K-Means++ (``init='k-means++'``, 10 re-starts)."""
        X = Modeling._scale(df)
        model = KMeans(n_clusters=k, init="k-means++", n_init=10,
                       random_state=seed)
        labels = model.fit_predict(X)
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
            "inertia": model.inertia_,
        }

    @staticmethod
    def run_hierarchical(
        df: pd.DataFrame,
        k: int,
        linkage: str = "ward",
    ) -> dict:
        """Agglomerative Hierarchical Clustering (default linkage: ward)."""
        X = Modeling._scale(df)
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
        }

    @staticmethod
    def run_gmm(
        df: pd.DataFrame,
        k: int,
        seed: int = 42,
    ) -> dict:
        """Gaussian Mixture Model — distribution-based clustering."""
        X = Modeling._scale(df)
        model = GaussianMixture(n_components=k, random_state=seed)
        model.fit(X)
        labels = model.predict(X)
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
            "bic": model.bic(X),
            "aic": model.aic(X),
        }

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    @staticmethod
    def run_all(
        df: pd.DataFrame,
        ks: tuple[int, ...] = (2, 3, 4),
        seed: int = 42,
    ) -> tuple[pd.DataFrame, dict]:
        """Run every algorithm for every K and aggregate results.

        Parameters
        ----------
        df:
            Feature matrix — numeric columns only; ``sample_name`` or
            other string columns are ignored by ``_scale`` automatically.
        ks:
            Cluster counts to evaluate.
        seed:
            Random seed forwarded to stochastic algorithms.

        Returns
        -------
        summary : pd.DataFrame
            One row per (Algorithm, K) with Silhouette, Inertia (KMeans
            variants), and BIC/AIC (GMM).  Index is (Algorithm, K).
        results : dict[str, dict[int, dict]]
            Full objects: ``results[algo_name][k]`` → result dict
            containing ``labels``, ``model``, ``silhouette``, etc.
        """
        runners: dict[str, callable] = {
            "KMeans": lambda k: Modeling.run_kmeans(df, k, seed=seed),
            "KMeans++": lambda k: Modeling.run_kmeans_plus(df, k, seed=seed),
            "Hierarchical": lambda k: Modeling.run_hierarchical(df, k),
            "GMM": lambda k: Modeling.run_gmm(df, k, seed=seed),
        }

        rows: list[dict] = []
        results: dict[str, dict[int, dict]] = {name: {} for name in runners}

        for algo, fn in runners.items():
            for k in ks:
                res = fn(k)
                results[algo][k] = res
                row: dict = {
                    "Algorithm": algo,
                    "K": k,
                    "Silhouette": round(res["silhouette"], 4),
                }
                if "inertia" in res:
                    row["Inertia"] = round(res["inertia"], 4)
                if "bic" in res:
                    row["BIC"] = round(res["bic"], 2)
                    row["AIC"] = round(res["aic"], 2)
                rows.append(row)

        summary = pd.DataFrame(rows).set_index(["Algorithm", "K"]).sort_index()
        return summary, results
