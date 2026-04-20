from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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
    - Return a result ``dict`` with at minimum ``labels``, ``silhouette``,
      ``scaler``, and ``cols`` (so new samples can be projected later).
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale(df: pd.DataFrame) -> tuple[np.ndarray,
                                          StandardScaler, list[str]]:
        """Drop zero-variance columns, StandardScale,
        return (X, scaler, cols)."""
        numeric = df.select_dtypes(include="number")
        non_const = numeric.loc[:, numeric.std() > 0]
        scaler = StandardScaler()
        X = scaler.fit_transform(non_const.values)
        return X, scaler, list(non_const.columns)

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
        X, scaler, cols = Modeling._scale(df)
        model = KMeans(n_clusters=k, init="random", n_init=10,
                       random_state=seed)
        labels = model.fit_predict(X)
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
            "inertia": model.inertia_,
            "scaler": scaler,
            "cols": cols,
        }

    @staticmethod
    def run_kmeans_plus(
        df: pd.DataFrame,
        k: int,
        seed: int = 42,
    ) -> dict:
        """K-Means++ (``init='k-means++'``, 10 re-starts)."""
        X, scaler, cols = Modeling._scale(df)
        model = KMeans(n_clusters=k, init="k-means++", n_init=10,
                       random_state=seed)
        labels = model.fit_predict(X)
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
            "inertia": model.inertia_,
            "scaler": scaler,
            "cols": cols,
        }

    @staticmethod
    def run_hierarchical(
        df: pd.DataFrame,
        k: int,
        linkage: str = "ward",
    ) -> dict:
        """Agglomerative Hierarchical Clustering (default linkage: ward)."""
        X, scaler, cols = Modeling._scale(df)
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
            "scaler": scaler,
            "cols": cols,
            "centroids": centroids,
        }

    @staticmethod
    def run_gmm(
        df: pd.DataFrame,
        k: int,
        seed: int = 42,
    ) -> dict:
        """Gaussian Mixture Model — distribution-based clustering."""
        X, scaler, cols = Modeling._scale(df)
        model = GaussianMixture(n_components=k, random_state=seed)
        model.fit(X)
        labels = model.predict(X)
        return {
            "labels": labels,
            "model": model,
            "silhouette": silhouette_score(X, labels),
            "bic": model.bic(X),
            "aic": model.aic(X),
            "scaler": scaler,
            "cols": cols,
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
            containing ``labels``, ``model``, ``silhouette``,
            ``scaler``, ``cols``, etc.
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

    # ------------------------------------------------------------------
    # PCA variance analysis
    # ------------------------------------------------------------------

    @staticmethod
    def pca_variance_analysis(
        df: pd.DataFrame,
        max_pcs: int = 5,
        seed: int = 42,
    ) -> dict:
        """Fit PCA and return explained-variance data for 1..max_pcs components.

        Useful for comparing how many principal components are needed to
        capture a given fraction of variance across different feature sets.

        Parameters
        ----------
        df:
            Feature matrix — scaled internally via ``_scale``.
        max_pcs:
            Maximum number of principal components to evaluate.
            Automatically clamped to ``min(n_samples - 1, n_features)``.
        seed:
            PCA random state.

        Returns
        -------
        dict with keys:
            ``n_components``          — list [1, 2, ..., n]
            ``variance_per_pc``       — explained variance ratio per PC
            ``cumulative_variance``   — cumulative explained variance ratio
            ``n_features``            — number of features after scaling
        """
        from sklearn.decomposition import PCA
        X, _, _ = Modeling._scale(df)
        n = min(max_pcs, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n, random_state=seed)
        pca.fit(X)
        var = list(pca.explained_variance_ratio_)
        return {
            "n_components":        list(range(1, n + 1)),
            "variance_per_pc":     var,
            "cumulative_variance": list(np.cumsum(var)),
            "n_features":          X.shape[1],
        }

    # ------------------------------------------------------------------
    # Attach cluster labels to a DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def attach_labels(
        df: pd.DataFrame,
        results: dict,
        ks: tuple[int, ...] | None = None,
    ) -> pd.DataFrame:
        """Return *df* with cluster label columns appended.

        One column per (algorithm, K) combination, named
        ``{algo_slug}_k{k}`` (e.g. ``KMeans_k2``, ``KMeanspp_k3``).

        Parameters
        ----------
        df:
            The feature DataFrame whose index aligns with the labels
            stored in *results* (i.e. the same DataFrame passed to
            ``run_all``).
        results:
            The ``results`` dict returned by ``run_all``.
        ks:
            Subset of K values to attach.  Defaults to all Ks present
            in *results*.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with label columns added; original columns
            are untouched.
        """
        out = df.copy()
        for algo, algo_results in results.items():
            slug = algo.replace("+", "p").replace(" ", "_")
            for k, res in algo_results.items():
                if ks is not None and k not in ks:
                    continue
                out[f"{slug}_k{k}"] = res["labels"]
        return out

    # ------------------------------------------------------------------
    # Predict new samples
    # ------------------------------------------------------------------

    @staticmethod
    def predict_new(
        df_new: pd.DataFrame,
        results: dict,
        k: int,
    ) -> pd.DataFrame:
        """Predict cluster labels for new samples using all fitted algorithms.

        Parameters
        ----------
        df_new:
            New samples feature matrix (same columns as training data).
        results:
            The ``results`` dict returned by ``run_all``.
        k:
            Which cluster count to use for prediction.

        Returns
        -------
        pd.DataFrame
            One column per algorithm, one row per new sample, with
            the predicted cluster label (0-indexed integer).
        """
        preds = {}
        for algo, algo_results in results.items():
            res = algo_results[k]
            X_new = res["scaler"].transform(df_new[res["cols"]].values)
            model = res["model"]
            if hasattr(model, "predict"):
                preds[algo] = model.predict(X_new)
            else:
                # AgglomerativeClustering has no predict — nearest centroid
                preds[algo] = cdist(X_new, res["centroids"]).argmin(axis=1)
        return pd.DataFrame(preds, index=df_new.index)
