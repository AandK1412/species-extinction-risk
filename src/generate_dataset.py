"""
generate_dataset.py
-------------------
Regenerates the synthetic "Species Extinction Risk Dataset" described in the
Team 04 report (Progress Report 3: Data Exploration).

The team's original dataset was created with generative AI and was not shipped
with the report, so this script reconstructs an equivalent dataset that matches:

  * the documented schema (Table 2.1.1),
  * the documented variable distributions (Section 2.2), and
  * the documented relationships / model findings (Sections 4-5):
      - Pollution Level is the strongest driver of extinction risk,
      - Human Population Density and Illegal Wildlife Trade Rate are next,
      - Climate Change Impact and Invasive Species Presence add risk,
      - Protected Area Presence reduces risk (negative coefficient),
      - Food Availability and Reproduction Rate are weak / not significant,
      - enough noise that models land around AUC ~0.70 (not a trivial 1.0).

Because the data is randomly generated, exact figures will differ slightly from
the report, but the qualitative conclusions are reproduced. A fixed random seed
keeps the output reproducible.

Usage:
    python src/generate_dataset.py
Output:
    data/species_extinction_risk.csv  (1885 rows)
"""

import os

import numpy as np
import pandas as pd

SEED = 42
N_RECORDS = 1885
OUTPUT_PATH = os.path.join("data", "species_extinction_risk.csv")

SPECIES = [
    "Snow Leopard", "Bengal Tiger", "African Elephant", "Giant Panda",
    "Mountain Gorilla", "Hawksbill Turtle", "Blue Whale", "Orangutan",
    "Black Rhino", "Vaquita", "Amur Leopard", "Saola", "Red Panda",
    "Pangolin", "Axolotl", "Iberian Lynx", "Kakapo", "Javan Rhino",
    "Sea Otter", "Galapagos Penguin", "Asian Elephant", "Sumatran Tiger",
    "Whooping Crane", "California Condor", "Hellbender Salamander",
]

REGIONS = [
    "North America", "South America", "Europe", "Sub-Saharan Africa",
    "South Asia", "Southeast Asia", "East Asia", "Oceania",
    "Central Asia", "Middle East",
]


def _scaled_beta(rng, a, b, size, lo=0.0, hi=100.0):
    """Beta sample rescaled to [lo, hi]."""
    return lo + (hi - lo) * rng.beta(a, b, size)


def generate(seed: int = SEED, n: int = N_RECORDS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Categorical descriptors -------------------------------------------
    species = rng.choice(SPECIES, size=n)
    region = rng.choice(REGIONS, size=n)

    # --- Numerical predictors (match documented distributions) -------------
    # Pollution Level: slightly left-skewed, mass 40-90.
    pollution = _scaled_beta(rng, 5, 2, n)  # skewed toward high values

    # Illegal Wildlife Trade Rate: fairly even, mass 30-70.
    trade = np.clip(rng.normal(50, 13, n), 0, 100)

    # Food Availability: strongly right-skewed, mass 0-40.
    food = _scaled_beta(rng, 1.5, 4, n)  # skewed toward low values

    # Human Population Density: pronounced right skew, mostly low, rare >195.
    human_density = rng.gamma(shape=2.0, scale=18.0, size=n)
    human_density = np.clip(human_density, 0, 260)

    # Species Reproduction Rate: bell shaped around ~2 births/yr.
    reproduction = np.clip(rng.normal(2.0, 0.35, n), 0.2, None)
    # a few high-fecundity outliers
    outlier_idx = rng.choice(n, size=max(1, n // 120), replace=False)
    reproduction[outlier_idx] += rng.uniform(1.5, 4.0, size=outlier_idx.size)

    # Climate Change Impact (%): slightly left-skewed, mass 60-95.
    climate = _scaled_beta(rng, 6, 2, n)

    # --- Binary predictors (near 50/50) ------------------------------------
    protected = rng.integers(0, 2, n)
    invasive = rng.integers(0, 2, n)

    # --- Target: Extinction Risk -------------------------------------------
    # Build a log-odds as a noisy linear combination of standardized drivers.
    def z(x):
        return (x - x.mean()) / x.std()

    logit = (
        -0.30
        + 1.65 * z(pollution)          # dominant driver
        + 0.85 * z(human_density)      # strong
        + 0.70 * z(trade)              # strong
        + 0.55 * z(climate)            # moderate
        + 0.45 * (invasive - 0.5) * 2  # adds risk
        - 1.15 * (protected - 0.5) * 2  # protection reduces risk (~ -1.15)
        + 0.05 * z(food)               # weak / not significant
        + 0.04 * z(reproduction)       # weak / not significant
    )
    # Noise controls separability so AUC lands around ~0.70.
    logit += rng.normal(0, 3.3, n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    extinction_risk = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "Species": species,
            "Region": region,
            "Pollution_Level": np.round(pollution, 2),
            "Illegal_Wildlife_Trade_Rate": np.round(trade, 2),
            "Food_Availability": np.round(food, 2),
            "Human_Population_Density": np.round(human_density, 2),
            "Species_Reproduction_Rate": np.round(reproduction, 2),
            "Climate_Change_Impact": np.round(climate, 2),
            "Protected_Area_Presence": protected,
            "Invasive_Species_Presence": invasive,
            "Extinction_Risk": extinction_risk,
        }
    )
    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(f"Extinction risk prevalence: {df['Extinction_Risk'].mean():.1%}")
    print(df.describe(include="all").T.head(20))


if __name__ == "__main__":
    main()
