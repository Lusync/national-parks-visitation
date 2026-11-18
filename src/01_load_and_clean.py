"""
01_load_and_clean.py
National Park Visitation Project
Loads the raw data, performs initial cleaning, and creates derived features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "US-National-Parks_RecreationVisits_1979-2024.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "parks_visits_cleaned.csv"

def load_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV."""
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning and feature engineering steps:
    - Ensure correct dtypes
    - Identify first year each park appears (establishment proxy)
    - Create park_age (years since first appearance)
    - Create COVID period labels
    - Create log-transformed visits
    - Flag zero-visit records
    """
    df = df.copy()

    # Standardize column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Dtypes
    df["Year"] = df["Year"].astype(int)
    df["RecreationVisits"] = df["RecreationVisits"].astype(int)

    # --- Handle late-starting parks (do NOT impute pre-existence years) ---
    # First year each park appears in the dataset
    first_year = df.groupby("ParkName")["Year"].transform("min")
    df["first_year"] = first_year
    df["park_age"] = df["Year"] - df["first_year"]  # 0 in the first recorded year

    # --- COVID period ---
    def assign_period(year):
        if year < 2020:
            return "Pre-pandemic"
        elif year in [2020, 2021]:
            return "Pandemic"
        else:
            return "Recovery"

    df["covid_period"] = df["Year"].apply(assign_period)

    # --- Log transform (add 1 to handle any zeros safely) ---
    df["log_visits"] = np.log1p(df["RecreationVisits"])

    # --- Flag zero visits ---
    df["is_zero_visits"] = df["RecreationVisits"] == 0

    # Order columns nicely
    cols = [
        "ParkName", "Region", "State", "Year", "RecreationVisits",
        "log_visits", "first_year", "park_age", "covid_period", "is_zero_visits"
    ]
    df = df[cols]

    return df

def summarize(df: pd.DataFrame) -> None:
    """Print useful summary information."""
    print("\n=== CLEANED DATA SUMMARY ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nYear range: {df['Year'].min()} – {df['Year'].max()}")
    print(f"Number of unique parks: {df['ParkName'].nunique()}")

    print("\n--- Parks that start after 1979 ---")
    late = (
        df.groupby("ParkName")["first_year"]
        .min()
        .loc[lambda x: x > 1979]
        .sort_values()
    )
    print(late.to_string())
    print(f"\nTotal parks starting after 1979: {len(late)}")

    print("\n--- Zero-visit records ---")
    zeros = df[df["is_zero_visits"]]
    print(f"Count: {len(zeros)}")
    if len(zeros) > 0:
        print(zeros[["ParkName", "Year", "RecreationVisits"]].to_string(index=False))

    print("\n--- COVID period counts ---")
    print(df["covid_period"].value_counts().sort_index())

    print("\n--- RecreationVisits distribution ---")
    print(df["RecreationVisits"].describe().round(0))

    print("\n--- Skewness (raw visits) ---")
    print(f"{df['RecreationVisits'].skew():.3f}")

def main():
    df_raw = load_data(RAW_PATH)
    df_clean = clean_and_engineer(df_raw)

    # Save cleaned data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"\nCleaned data saved to: {PROCESSED_PATH}")

    summarize(df_clean)

    return df_clean

if __name__ == "__main__":
    main()
