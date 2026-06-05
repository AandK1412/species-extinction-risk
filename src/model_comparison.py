"""
model_comparison.py
-------------------
Reproduces Section 5 of the report: trains four classifiers and compares them
on training and validation data using AUC, combined ROC curves, and lift curves.

Models:
  * Logistic Regression
  * Neural Network (MLPClassifier)
  * Support Vector Machine (RBF kernel, probability=True)
  * Stacking Ensemble (LR + Decision Tree + SVM, meta-learner = Logistic Reg.)

Outputs:
    outputs/roc_comparison.png
    outputs/lift_comparison.png
    prints a Training/Validation AUC comparison table (Table 5.1)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import PREDICTORS, TARGET, ensure_output_dir, load_data


def build_models():
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    nn = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(6,),
            alpha=0.3,
            early_stopping=False,
            max_iter=800,
            random_state=42,
        ),
    )
    svm = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)
    )
    ensemble = StackingClassifier(
        estimators=[
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("svm", make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True,
                                                        random_state=42))),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )
    return {
        "Logistic Regression": lr,
        "Neural Network": nn,
        "SVM": svm,
        "Ensemble": ensemble,
    }


def lift_points(y_true, y_score, bins=20):
    """Return (cumulative population fraction, lift) for a lift curve."""
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    base_rate = y_sorted.mean()
    cum_frac, lift = [], []
    n = len(y_sorted)
    for b in range(1, bins + 1):
        k = max(1, int(n * b / bins))
        cum_frac.append(b / bins)
        lift.append((y_sorted[:k].mean() / base_rate) if base_rate > 0 else 0)
    return np.array(cum_frac), np.array(lift)


def main():
    df = load_data()
    out_dir = ensure_output_dir()
    X = df[PREDICTORS].values
    y = df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results = {}
    roc_fig, roc_axes = plt.subplots(1, 2, figsize=(14, 6))
    lift_fig, lift_axes = plt.subplots(1, 2, figsize=(14, 6))

    rows = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        s_tr = model.predict_proba(Xtr)[:, 1]
        s_te = model.predict_proba(Xte)[:, 1]
        auc_tr = roc_auc_score(ytr, s_tr)
        auc_te = roc_auc_score(yte, s_te)
        results[name] = (auc_tr, auc_te)
        rows.append({"Model": name, "Training AUC": round(auc_tr, 3),
                     "Validation AUC": round(auc_te, 3)})

        # ROC
        for ax, (yy, ss, split) in zip(
            roc_axes, [(ytr, s_tr, "Training"), (yte, s_te, "Validation")]
        ):
            fpr, tpr, _ = roc_curve(yy, ss)
            ax.plot(fpr, tpr, label=f"{name}")
            ax.set_title(f"{split} ROC")

        # Lift
        for ax, (yy, ss, split) in zip(
            lift_axes, [(ytr, s_tr, "Training"), (yte, s_te, "Validation")]
        ):
            cf, lf = lift_points(yy, ss)
            ax.plot(cf, lf, label=f"{name}")
            ax.set_title(f"{split} Lift")

    for ax in roc_axes:
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    roc_fig.suptitle("Figure 5.9: Combined ROC Curves", fontsize=14)
    roc_fig.tight_layout(rect=[0, 0, 1, 0.96])
    roc_path = os.path.join(out_dir, "roc_comparison.png")
    roc_fig.savefig(roc_path, dpi=120)
    plt.close(roc_fig)

    for ax in lift_axes:
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_xlabel("Fraction of population (ranked by risk)")
        ax.set_ylabel("Lift")
        ax.legend()
    lift_fig.suptitle("Figure 5.10: Combined Lift Curves", fontsize=14)
    lift_fig.tight_layout(rect=[0, 0, 1, 0.96])
    lift_path = os.path.join(out_dir, "lift_comparison.png")
    lift_fig.savefig(lift_path, dpi=120)
    plt.close(lift_fig)

    table = pd.DataFrame(rows).set_index("Model")
    print("\nTable 5.1: Model Comparison (AUC)")
    print(table.to_string())
    best = table["Validation AUC"].idxmax()
    print(f"\nBest validation AUC: {best} ({table.loc[best, 'Validation AUC']})")
    print(f"Saved {roc_path}")
    print(f"Saved {lift_path}")


if __name__ == "__main__":
    main()
