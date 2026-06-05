"""Shared helpers for the analysis scripts."""

import os

import pandas as pd

DATA_PATH = os.path.join("data", "species_extinction_risk.csv")
OUTPUT_DIR = "outputs"

# Eight predictors used by the dependence / comparison models (Section 4-5).
PREDICTORS = [
    "Pollution_Level",
    "Illegal_Wildlife_Trade_Rate",
    "Food_Availability",
    "Human_Population_Density",
    "Species_Reproduction_Rate",
    "Climate_Change_Impact",
    "Protected_Area_Presence",
    "Invasive_Species_Presence",
]

# Six predictors retained in the reduced logistic model (Section 4.1).
REDUCED_PREDICTORS = [
    "Pollution_Level",
    "Illegal_Wildlife_Trade_Rate",
    "Human_Population_Density",
    "Climate_Change_Impact",
    "Protected_Area_Presence",
    "Invasive_Species_Presence",
]

NUMERIC_VARS = [
    "Pollution_Level",
    "Illegal_Wildlife_Trade_Rate",
    "Food_Availability",
    "Human_Population_Density",
    "Species_Reproduction_Rate",
    "Climate_Change_Impact",
]

TARGET = "Extinction_Risk"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/generate_dataset.py` first."
        )
    return pd.read_csv(DATA_PATH)


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
