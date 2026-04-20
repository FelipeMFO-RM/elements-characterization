import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]


class Plots:
    @staticmethod
    def plot_clusters_pca(
        df_feat: pd.DataFrame,
        algos: list,
        results: dict,
        title: str,
        ks: tuple,
        seed: int = 42,
        output_path: str | None = None,
    ) -> None:
        """PCA 2-D scatter: rows = K values, columns = algorithms.

        Parameters
        ----------
        output_path:
            If provided, save the figure to this path instead of (or in
            addition to) displaying it.  Useful for the pipeline script.
        """
        numeric = df_feat.select_dtypes(include="number")
        non_const = numeric.loc[:, numeric.std() > 0]
        X_scaled = StandardScaler().fit_transform(non_const.values)
        coords = PCA(n_components=2, random_state=seed).fit_transform(X_scaled)

        n_rows, n_cols = len(ks), len(algos)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

        for r, k in enumerate(ks):
            for c, algo in enumerate(algos):
                ax = axes[r, c]
                labels = results[algo][k]["labels"]
                sil = results[algo][k]["silhouette"]
                for cluster_id in range(k):
                    mask = labels == cluster_id
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        c=COLORS[cluster_id],
                        label=f"C{cluster_id}",
                        alpha=0.7,
                        s=30,
                        edgecolors="none",
                    )
                ax.set_title(f"{algo} · K={k}\nSil={sil:.3f}", fontsize=9)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.legend(fontsize=7, markerscale=1.2)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_silhouette_comparison(
        summaries: list[tuple[pd.DataFrame, str]],
        ks: tuple,
        title: str = "Silhouette score — all algorithms & datasets",
        output_path: str | None = None,
    ) -> None:
        """Grouped bar chart comparing silhouette scores across datasets,
        one subplot per K.

        Parameters
        ----------
        output_path:
            If provided, save the figure to this path.
        """
        comp_flat = pd.concat([
            summary[["Silhouette"]].assign(Dataset=label)
            for summary, label in summaries
        ]).reset_index()

        fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4),
                                 sharey=True)
        fig.suptitle(title, fontweight="bold")

        for ax, k in zip(axes, ks):
            subset = comp_flat[comp_flat["K"] == k]
            sns.barplot(data=subset, x="Algorithm", y="Silhouette",
                        hue="Dataset", ax=ax)
            ax.set_title(f"K = {k}")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_new_samples_pca(
        df_train: pd.DataFrame,
        df_new: pd.DataFrame,
        results: dict,
        predictions_by_k: dict,
        algos: list,
        ks: tuple,
        title: str,
        seed: int = 42,
        output_path: str | None = None,
    ) -> None:
        """PCA scatter overlaying new sample predictions on fitted
        training clusters.

        Training points are shown faded; new samples appear as stars coloured
        by their predicted cluster.

        Parameters
        ----------
        df_train:
            Feature matrix used to fit the clusters.
        df_new:
            New samples to classify (same numeric columns as df_train).
        results:
            The ``results`` dict returned by ``Modeling.run_all``.
        predictions_by_k:
            ``{k: pd.DataFrame(index=df_new.index, columns=algos)}`` as
            returned by ``Modeling.predict_new``.
        algos:
            Algorithm names to include — one column of subplots each.
        ks:
            Cluster counts to include — one row of subplots each.
        title:
            Figure super-title.
        seed:
            PCA random state.
        output_path:
            If provided, save the figure to this path.
        """
        ref = results[algos[0]][ks[0]]
        scaler, cols = ref["scaler"], ref["cols"]

        pca = PCA(n_components=2, random_state=seed)
        coords_train = \
            pca.fit_transform(scaler.transform(df_train[cols].values))
        coords_new = pca.transform(scaler.transform(df_new[cols].values))

        n_rows, n_cols = len(ks), len(algos)
        fig, axes = \
            plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

        for r, k in enumerate(ks):
            for c, algo in enumerate(algos):
                ax = axes[r, c]
                train_labels = results[algo][k]["labels"]
                pred_labels = predictions_by_k[k][algo].values

                for cluster_id in range(k):
                    mask = train_labels == cluster_id
                    ax.scatter(
                        coords_train[mask, 0],
                        coords_train[mask, 1],
                        c=COLORS[cluster_id],
                        alpha=0.2,
                        s=20,
                        edgecolors="none",
                    )

                for cluster_id in range(k):
                    mask = pred_labels == cluster_id
                    if mask.any():
                        ax.scatter(
                            coords_new[mask, 0],
                            coords_new[mask, 1],
                            c=COLORS[cluster_id],
                            marker="*",
                            s=220,
                            edgecolors="black",
                            linewidths=0.5,
                            label=f"C{cluster_id}",
                            zorder=5,
                        )

                sil = results[algo][k]["silhouette"]
                ax.set_title(f"{algo} · K={k}  Sil={sil:.3f}", fontsize=9)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.legend(fontsize=7, markerscale=0.8, title="new → cluster")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_pca_variance(
        analyses: dict[str, dict],
        title: str = "PCA Variance Analysis",
        output_path: str | None = None,
    ) -> None:
        """Scree + cumulative variance chart for one or more feature sets.

        Each key in *analyses* produces one subplot.  Bars show the
        individual explained variance per PC; the red line shows the
        cumulative total.  Cumulative values are annotated at each point
        for easy reading of "how many PCs reach 90 %?".

        Parameters
        ----------
        analyses:
            ``{label: result_dict}`` where each ``result_dict`` is the
            output of ``Modeling.pca_variance_analysis``.
        title:
            Figure super-title.
        output_path:
            If provided, save the figure to this path.
        """
        n = len(analyses)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]
        fig.suptitle(title, fontweight="bold", y=1.02)

        for ax, (label, res) in zip(axes, analyses.items()):
            pcs = res["n_components"]
            var = [v * 100 for v in res["variance_per_pc"]]
            cum = [c * 100 for c in res["cumulative_variance"]]

            ax.bar(pcs, var, color="steelblue", alpha=0.75,
                   label="Individual PC")
            ax.plot(pcs, cum, "o-", color="tomato", linewidth=2,
                    markersize=5, label="Cumulative")

            for n_pc, c in zip(pcs, cum):
                ax.annotate(
                    f"{c:.0f}%",
                    (n_pc, c),
                    textcoords="offset points",
                    xytext=(0, 7),
                    fontsize=8,
                    ha="center",
                    color="tomato",
                )

            n_feat = res["n_features"]
            ax.set_title(f"{label}\n({n_feat} features after scaling)",
                         fontsize=9)
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained variance (%)")
            ax.set_ylim(0, 115)
            ax.set_xticks(pcs)
            ax.legend(fontsize=8)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_silhouette_heatmap(
        summary: pd.DataFrame,
        title: str,
        output_path: str | None = None,
    ) -> None:
        """Heatmap of silhouette scores: algorithms vs K.

        Parameters
        ----------
        output_path:
            If provided, save the figure to this path.
        """
        sil = summary["Silhouette"].unstack(level="K")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(
            sil,
            annot=True,
            fmt=".3f",
            cmap="YlGn",
            linewidths=0.5,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_title(title, fontweight="bold")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
