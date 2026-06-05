# Predicting Species Extinction Risk

A machine-learning study that predicts whether a species is **at risk of
extinction** from environmental, biological, and human (anthropogenic)
indicators. The goal is to help conservation teams rank where limited resources
will have the greatest impact.

This repository is a **reproducible Python implementation** of Team 04's
business-intelligence project (*Progress Report 3: Data Exploration*). The
original analysis was carried out in JMP and documented in a written report; the
code here regenerates the dataset and reproduces every stage of that analysis
with `scikit-learn`, `statsmodels`, and `matplotlib`.

> **Note on data:** the team's dataset was synthetic (AI-generated) and was not
> included with the report, so `src/generate_dataset.py` reconstructs an
> equivalent dataset matching the documented schema, distributions, and findings.
> Numbers will differ slightly from the report, but the conclusions hold:
> pollution is the dominant driver, protected areas are protective, and the four
> models land around AUC 0.70.

## What it does

| Stage | Script | Report section |
|---|---|---|
| Generate the synthetic dataset (1,885 records) | `src/generate_dataset.py` | §2.1 |
| Exploratory data analysis — distributions & scatterplots | `src/eda.py` | §2.2 |
| Interdependence analysis — PCA & clustering | `src/interdependence.py` | §3 |
| Dependence analysis — logistic regression & decision tree | `src/dependence.py` | §4 |
| Model comparison — LR vs NN vs SVM vs stacking ensemble | `src/model_comparison.py` | §5 |

## Key findings (reproduced)

- **Pollution Level** is the single strongest predictor of extinction risk; in
  the decision tree it forms the root split (~67 on a 0–100 scale).
- **Human Population Density** and **Illegal Wildlife Trade Rate** are the next
  most important drivers, followed by **Climate Change Impact** and
  **Invasive Species Presence**.
- **Protected Area Presence** lowers extinction risk (negative coefficient).
- **Food Availability** and **Species Reproduction Rate** are weak / not
  statistically significant.
- All four classifiers achieve moderate, comparable performance (validation
  AUC ≈ 0.70–0.73), with the ensemble and neural network competitive with the
  interpretable logistic regression.

## Project structure

```
species-extinction-risk/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                     # data dictionary
│   └── species_extinction_risk.csv   # created by generate_dataset.py
├── src/
│   ├── generate_dataset.py
│   ├── utils.py                      # shared loaders / config
│   ├── eda.py
│   ├── interdependence.py
│   ├── dependence.py
│   └── model_comparison.py
├── outputs/                          # figures written by the scripts
└── reports/
    └── ReportPR3_Team04.docx         # the original written report
```

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/AandK1412/species-extinction-risk.git
cd species-extinction-risk

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Run the scripts from the project root, in order. Each writes its figures to
`outputs/` and prints a summary to the console.

```bash
python src/generate_dataset.py      # 1. create data/species_extinction_risk.csv
python src/eda.py                   # 2. distributions + scatterplots
python src/interdependence.py       # 3. PCA + clustering
python src/dependence.py            # 4. logistic regression + decision tree
python src/model_comparison.py      # 5. four-model comparison (AUC, ROC, lift)
```

## Methods

- **PCA** with Kaiser (eigenvalue > 1), cumulative-variance (75%), and scree
  (elbow) criteria for component retention, plus a PC1–PC2 loadings plot.
- **Clustering** with both K-means and Ward hierarchical clustering, cross-
  tabulated to check that the two solutions agree.
- **Logistic regression** in full (8-predictor) and reduced (6-predictor) forms,
  with McFadden pseudo-R² and p-values (via `statsmodels`).
- **Decision tree** with feature importances and human-readable split rules.
- **Model comparison** across logistic regression, a neural network
  (`MLPClassifier`), an RBF-kernel SVM, and a stacking ensemble, evaluated by
  training/validation AUC, combined ROC curves, and lift curves.

## References

1. Cardillo, M., Purvis, A., Sechrest, W., Gittleman, J. L., Bielby, J., & Mace, G. M. (2005). Multiple causes of high extinction risk in large mammal species. *Science*, 309(5738), 1239–1241.
2. Purvis, A., Gittleman, J. L., Cowlishaw, G., & Mace, G. M. (2000). Predicting extinction risk in declining species. *Proceedings of the Royal Society B*, 267(1456), 1947–1952.
3. Böhm, M., Collen, B., Baillie, J. E. M., et al. (2016). Threats to the world's freshwater biodiversity. *Conservation Letters*, 9(2), 80–89.
4. Davidson, A. D., Hamilton, M. J., Boyer, A. G., Brown, J. H., & Ceballos, G. (2012). Drivers and hotspots of extinction risk in marine mammals. *Proceedings of the Royal Society B*, 279(1730), 2197–2205.
5. Di Marco, M., Brooks, T., Cuttelod, A., et al. (2014). Quantifying the relative importance of global conservation strategies. *Nature Communications*, 5, 1–8.

## Authors

Aakash Nihalaney, Anrunya Patole, Harsh Sahu, Shreyas Desai

## License

Released under the MIT License. See [LICENSE](LICENSE).
