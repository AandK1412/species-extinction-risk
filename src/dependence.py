"""
dependence.py
-------------
Reproduces Section 4 of the report:
  * Full logistic regression (all 8 predictors) with significance via statsmodels
    if available, otherwise via scikit-learn coefficients.
  * Reduced logistic regression (drops Food Availability and Reproduction Rate).
  * Decision tree with the root split on Pollution Level and feature importances.

Outputs:
    outputs/decision_tree.png
    prints coefficient / significance tables and tree rules
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from utils import PREDICTORS, REDUCED_PREDICTORS, TARGET, ensure_output_dir, load_data

try:
    import statsmodels.api as sm

    HAVE_SM = True
except Exception:
    HAVE_SM = False


def logistic_report(df, predictors, label):
    X = df[predictors].values
    y = df[TARGET].values
    Xs = StandardScaler().fit_transform(X)

    print(f"\n=== Logistic Regression: {label} ({len(predictors)} predictors) ===")
    if HAVE_SM:
        Xc = sm.add_constant(Xs)
        model = sm.Logit(y, Xc).fit(disp=False)
        # McFadden pseudo R-squared (comparable to JMP's R-square (U))
        print(f"  Pseudo R-squared (McFadden): {model.prsquared:.4f}")
        table = pd.DataFrame(
            {
                "coef": model.params[1:],
                "p_value": model.pvalues[1:],
            },
            index=predictors,
        )
        table["significant (p<0.05)"] = table["p_value"] < 0.05
        table = table.reindex(table["coef"].abs().sort_values(ascending=False).index)
        print(table.round(4).to_string())
    else:
        clf = LogisticRegression(max_iter=1000).fit(Xs, y)
        table = pd.Series(clf.coef_[0], index=predictors)
        table = table.reindex(table.abs().sort_values(ascending=False).index)
        print("  (install statsmodels for p-values; showing coefficients)")
        print(table.round(4).to_string())


def decision_tree_report(df, out_dir, max_depth=4):
    X = df[PREDICTORS].values
    y = df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(Xtr, ytr)

    print("\n=== Decision Tree ===")
    print(f"  Training accuracy:   {tree.score(Xtr, ytr):.3f}")
    print(f"  Validation accuracy: {tree.score(Xte, yte):.3f}")
    root_feature = PREDICTORS[tree.tree_.feature[0]]
    root_threshold = tree.tree_.threshold[0]
    print(f"  Root split: {root_feature} <= {root_threshold:.2f}")

    importances = (
        pd.Series(tree.feature_importances_, index=PREDICTORS)
        .sort_values(ascending=False)
    )
    print("  Feature importances:")
    print(importances.round(4).to_string())

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        tree,
        feature_names=[p.replace("_", " ") for p in PREDICTORS],
        class_names=["Not at risk", "At risk"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Figure 4.8: Decision Tree (depth-limited for readability)")
    fig.tight_layout()
    path = os.path.join(out_dir, "decision_tree.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")

    print("\n  Text rules (top levels):")
    print(export_text(tree, feature_names=[p for p in PREDICTORS], max_depth=2))


def main():
    df = load_data()
    out_dir = ensure_output_dir()
    logistic_report(df, PREDICTORS, "Full model")
    logistic_report(df, REDUCED_PREDICTORS, "Reduced model")
    decision_tree_report(df, out_dir)


if __name__ == "__main__":
    main()
