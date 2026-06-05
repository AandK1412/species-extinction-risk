"""
interdependence.py
------------------
Reproduces Section 3 of the report:
  * Principal Component Analysis (scree plot, cumulative variance, loadings)
  * K-means and hierarchical clustering, with a cross-tabulation of the two
    solutions to check agreement (the report found strong agreement).

Outputs:
    outputs/pca_scree.png
    outputs/pca_loadings.png
    outputs/cluster_comparison.png
    prints variance explained, retained components, and a cluster crosstab
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import NUMERIC_VARS, ensure_output_dir, load_data

PCA_VARS = NUMERIC_VARS + [
    "Protected_Area_Presence",
    "Invasive_Species_Presence",
]


def run_pca(df, out_dir):
    X = StandardScaler().fit_transform(df[PCA_VARS].values)
    pca = PCA().fit(X)
    eigenvalues = pca.explained_variance_
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    kaiser = int((eigenvalues > 1).sum())
    n_for_75 = int(np.searchsorted(cum_var, 0.75) + 1)
    print("PCA")
    print(f"  Components with eigenvalue > 1 (Kaiser): {kaiser}")
    print(f"  Components for ~75% variance: {n_for_75}")
    print(f"  Cumulative variance (first 5): "
          f"{np.round(cum_var[:5], 3).tolist()}")

    # Scree + cumulative variance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    comps = np.arange(1, len(eigenvalues) + 1)
    ax1.plot(comps, eigenvalues, "o-", color="#4C72B0")
    ax1.axhline(1, color="grey", ls="--", lw=1, label="Kaiser (eigenvalue = 1)")
    ax1.set_title("Figure 3.1/3.2: Scree Plot")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.legend()

    ax2.plot(comps, cum_var, "o-", color="#C44E52")
    ax2.axhline(0.75, color="grey", ls="--", lw=1, label="75% threshold")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_scree.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")

    # Loadings plot for the first two components (Figure 3.4)
    loadings = pca.components_[:2].T
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    for i, var in enumerate(PCA_VARS):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 head_width=0.02, color="#4C72B0", alpha=0.7)
        ax.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1,
                var.replace("_", " "), fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Figure 3.4: PCA Loadings (PC1 vs PC2)")
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_loadings.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")

    return X


def run_clustering(X, out_dir, k=3):
    # K-means
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    km_labels = km.labels_

    # Hierarchical (Ward)
    Z = linkage(X, method="ward")
    hc_labels = fcluster(Z, t=k, criterion="maxclust")

    crosstab = pd.crosstab(
        pd.Series(km_labels, name="KMeans"),
        pd.Series(hc_labels, name="Hierarchical"),
    )
    print("\nClustering (k = %d)" % k)
    print("  KMeans vs Hierarchical cross-tabulation:")
    print(crosstab.to_string())

    # Agreement: max overlap per kmeans cluster / total
    agreement = crosstab.max(axis=1).sum() / crosstab.values.sum()
    print(f"  Best-match agreement between methods: {agreement:.1%}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(crosstab.values, cmap="Blues")
    ax.set_xticks(range(crosstab.shape[1]))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticks(range(crosstab.shape[0]))
    ax.set_yticklabels(crosstab.index)
    ax.set_xlabel("Hierarchical cluster")
    ax.set_ylabel("K-means cluster")
    ax.set_title("Figure 3.13: K-means vs Hierarchical Agreement")
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            ax.text(j, i, crosstab.values[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, "cluster_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    df = load_data()
    out_dir = ensure_output_dir()
    X = run_pca(df, out_dir)
    run_clustering(X, out_dir, k=3)


if __name__ == "__main__":
    main()
