"""
02_eda.py
Exploratory Data Analysis for National Park Visitation project.
Generates summary tables and key figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "parks_visits_cleaned.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

def load_cleaned():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded cleaned data: {df.shape}")
    return df

def summary_tables(df: pd.DataFrame) -> None:
    print("\n=== SUMMARY TABLES ===")

    print("\n--- Overall descriptive statistics (RecreationVisits) ---")
    print(df["RecreationVisits"].describe().round(0))

    print("\n--- Mean visits by Region ---")
    region_summary = (
        df.groupby("Region")["RecreationVisits"]
        .agg(["mean", "median", "count"])
        .round(0)
        .sort_values("mean", ascending=False)
    )
    print(region_summary)

    print("\n--- Mean visits by COVID period ---")
    period_summary = (
        df.groupby("covid_period")["RecreationVisits"]
        .agg(["mean", "median", "count"])
        .round(0)
        .reindex(["Pre-pandemic", "Pandemic", "Recovery"])
    )
    print(period_summary)

    print("\n--- Top 10 parks by average annual visits ---")
    top_parks = (
        df.groupby("ParkName")["RecreationVisits"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(0)
    )
    print(top_parks)

def plot_visits_over_time(df: pd.DataFrame) -> None:
    """Total and average visits per year, with COVID periods shaded."""
    yearly = df.groupby("Year").agg(
        total_visits=("RecreationVisits", "sum"),
        avg_visits=("RecreationVisits", "mean"),
        n_parks=("ParkName", "nunique")
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Total visits
    axes[0].plot(yearly["Year"], yearly["total_visits"] / 1e6, color="#2c7bb6", linewidth=2)
    axes[0].axvspan(2019.5, 2021.5, color="red", alpha=0.12, label="Pandemic (2020-2021)")
    axes[0].set_ylabel("Total Visits (millions)")
    axes[0].set_title("Total National Park Recreation Visits by Year")
    axes[0].legend(loc="upper left")

    # Average visits
    axes[1].plot(yearly["Year"], yearly["avg_visits"] / 1e6, color="#d7191c", linewidth=2)
    axes[1].axvspan(2019.5, 2021.5, color="red", alpha=0.12)
    axes[1].set_ylabel("Average Visits per Park (millions)")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Average Visits per Park by Year")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_visits_over_time.png")
    plt.close()
    print("Saved: 01_visits_over_time.png")

def plot_region_comparison(df: pd.DataFrame) -> None:
    """Average visits by Region."""
    region_avg = (
        df.groupby("Region")["RecreationVisits"]
        .mean()
        .sort_values(ascending=True)
        / 1e6
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    region_avg.plot(kind="barh", ax=ax, color="#2c7bb6")
    ax.set_xlabel("Average Annual Visits (millions)")
    ax.set_title("Average Annual Recreation Visits by NPS Region")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_avg_visits_by_region.png")
    plt.close()
    print("Saved: 02_avg_visits_by_region.png")

def plot_covid_comparison(df: pd.DataFrame) -> None:
    """Boxplot of visits by COVID period (log scale for readability)."""
    period_order = ["Pre-pandemic", "Pandemic", "Recovery"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x="covid_period", y="log_visits",
        order=period_order, ax=ax, palette="Set2"
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("Log(Visits + 1)")
    ax.set_title("Distribution of Park Visits by COVID Period (log scale)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_covid_period_boxplot.png")
    plt.close()
    print("Saved: 03_covid_period_boxplot.png")

def plot_distribution(df: pd.DataFrame) -> None:
    """Raw vs log-transformed distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(df["RecreationVisits"] / 1e6, bins=40, color="#2c7bb6", edgecolor="white")
    axes[0].set_xlabel("Recreation Visits (millions)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Raw Distribution (highly skewed)")

    axes[1].hist(df["log_visits"], bins=40, color="#d7191c", edgecolor="white")
    axes[1].set_xlabel("Log(Visits + 1)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Log-Transformed Distribution")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_visits_distribution.png")
    plt.close()
    print("Saved: 04_visits_distribution.png")

def plot_top_parks(df: pd.DataFrame) -> None:
    """Top 15 parks by average visits."""
    top = (
        df.groupby("ParkName")["RecreationVisits"]
        .mean()
        .sort_values(ascending=True)
        .tail(15)
        / 1e6
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind="barh", ax=ax, color="#2c7bb6")
    ax.set_xlabel("Average Annual Visits (millions)")
    ax.set_title("Top 15 National Parks by Average Annual Visits")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_top_parks.png")
    plt.close()
    print("Saved: 05_top_parks.png")

def plot_covid_change_by_region(df: pd.DataFrame) -> None:
    """Mean visits by region and COVID period."""
    summary = (
        df.groupby(["Region", "covid_period"])["RecreationVisits"]
        .mean()
        .unstack()
        .reindex(columns=["Pre-pandemic", "Pandemic", "Recovery"])
        / 1e6
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Average Visits (millions)")
    ax.set_title("Average Visits by Region and COVID Period")
    ax.legend(title="Period")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_covid_by_region.png")
    plt.close()
    print("Saved: 06_covid_by_region.png")

def main():
    df = load_cleaned()
    summary_tables(df)

    print("\n=== GENERATING FIGURES ===")
    plot_visits_over_time(df)
    plot_region_comparison(df)
    plot_covid_comparison(df)
    plot_distribution(df)
    plot_top_parks(df)
    plot_covid_change_by_region(df)

    print(f"\nAll figures saved to: {FIG_DIR}")

if __name__ == "__main__":
    main()
