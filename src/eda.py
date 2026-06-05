"""
eda.py
------
Reproduces Section 2 of the report: variable distributions (Figures 2.1-2.8)
and the key scatterplot relationships (Figures 2.9-2.13).

Outputs:
    outputs/distributions.png
    outputs/scatter_relationships.png
    prints a summary table to the console
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend so it runs without a display
import matplotlib.pyplot as plt

from utils import NUMERIC_VARS, TARGET, ensure_output_dir, load_data


def plot_distributions(df, out_dir):
    binary_vars = ["Protected_Area_Presence", "Invasive_Species_Presence"]
    vars_to_plot = NUMERIC_VARS + binary_vars

    n = len(vars_to_plot)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.ravel()

    for ax, var in zip(axes, vars_to_plot):
        if var in binary_vars:
            counts = df[var].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
        else:
            ax.hist(df[var], bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(var.replace("_", " "))
        ax.set_ylabel("Count")

    for ax in axes[len(vars_to_plot):]:
        ax.axis("off")

    fig.suptitle("Figure 2.1-2.8: Variable Distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "distributions.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_relationships(df, out_dir):
    pairs = [
        ("Illegal_Wildlife_Trade_Rate", "Food_Availability"),
        ("Human_Population_Density", "Food_Availability"),
        ("Climate_Change_Impact", "Pollution_Level"),
        ("Human_Population_Density", "Species_Reproduction_Rate"),
        ("Climate_Change_Impact", "Species_Reproduction_Rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (x, y) in zip(axes, pairs):
        colors = df[TARGET].map({0: "#55A868", 1: "#C44E52"})
        ax.scatter(df[x], df[y], c=colors, s=8, alpha=0.5)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.set_title(f"{x.replace('_', ' ')} vs {y.replace('_', ' ')}")
    axes[-1].axis("off")
    # simple legend
    axes[-1].scatter([], [], c="#C44E52", label="At risk (1)")
    axes[-1].scatter([], [], c="#55A868", label="Not at risk (0)")
    axes[-1].legend(loc="center", fontsize=12)
    fig.suptitle("Figure 2.9-2.13: Scatterplot Relationships", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "scatter_relationships.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    df = load_data()
    out_dir = ensure_output_dir()

    print(f"Records: {len(df)}  |  Variables: {df.shape[1]}")
    print(f"Extinction risk prevalence: {df[TARGET].mean():.1%}\n")
    print("Numeric summary:")
    print(df[NUMERIC_VARS].describe().T.round(2))

    plot_distributions(df, out_dir)
    plot_relationships(df, out_dir)


if __name__ == "__main__":
    main()
