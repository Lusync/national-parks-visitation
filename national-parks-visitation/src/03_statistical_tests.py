"""
03_statistical_tests.py
Formal statistical analysis for National Park Visitation project.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "parks_visits_cleaned.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {df.shape[0]} rows\n")
    return df

def test_region_differences(df: pd.DataFrame) -> None:
    """
    Test whether mean visitation differs across NPS regions.
    Because of strong skew, we run both ANOVA and Kruskal-Wallis.
    """
    print("=" * 60)
    print("TEST 1: Differences in visitation across NPS Regions")
    print("=" * 60)

    print("\nNull hypothesis (H0): Mean recreation visits are equal across all regions.")
    print("Alternative (H1): At least one region has a different mean.")

    groups = [group["RecreationVisits"].values for name, group in df.groupby("Region")]

    # One-way ANOVA
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  p-value     = {p_anova:.6f}")

    # Kruskal-Wallis (non-parametric, more appropriate given skew)
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis (non-parametric):")
    print(f"  H-statistic = {h_stat:.4f}")
    print(f"  p-value     = {p_kw:.6f}")

    if p_kw < 0.05:
        print("\nInterpretation: Reject H0. There is strong evidence that median visitation differs across regions.")
    else:
        print("\nInterpretation: Fail to reject H0.")

def test_covid_periods(df: pd.DataFrame) -> None:
    """
    Compare visitation across the three COVID-defined periods.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Differences across COVID periods")
    print("=" * 60)

    print("\nNull hypothesis (H0): Mean recreation visits are equal across Pre-pandemic, Pandemic, and Recovery periods.")
    print("Alternative (H1): At least one period differs.")

    period_order = ["Pre-pandemic", "Pandemic", "Recovery"]
    groups = [df.loc[df["covid_period"] == p, "RecreationVisits"].values for p in period_order]

    # ANOVA
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  p-value     = {p_anova:.6f}")

    # Kruskal-Wallis
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis:")
    print(f"  H-statistic = {h_stat:.4f}")
    print(f"  p-value     = {p_kw:.6f}")

    # Pairwise Mann-Whitney (with simple Bonferroni note)
    print("\nPairwise Mann-Whitney U tests (Bonferroni-adjusted alpha ≈ 0.0167):")
    pairs = [("Pre-pandemic", "Pandemic"), ("Pre-pandemic", "Recovery"), ("Pandemic", "Recovery")]
    for a, b in pairs:
        x = df.loc[df["covid_period"] == a, "RecreationVisits"]
        y = df.loc[df["covid_period"] == b, "RecreationVisits"]
        u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
        print(f"  {a} vs {b}: U = {u_stat:.1f}, p = {p_val:.6f}")

    print("\nInterpretation guidance: Look at both the overall test and the pairwise results.")
    print("Note: Pandemic period has far fewer observations (only 2 years).")

def test_year_correlation(df: pd.DataFrame) -> None:
    """Correlation between Year and visits."""
    print("\n" + "=" * 60)
    print("TEST 3: Correlation between Year and RecreationVisits")
    print("=" * 60)

    print("\nNull hypothesis (H0): There is no linear association between Year and visits (rho = 0).")

    # Pearson
    r_pearson, p_pearson = stats.pearsonr(df["Year"], df["RecreationVisits"])
    print(f"\nPearson correlation:")
    print(f"  r = {r_pearson:.4f}, p = {p_pearson:.6f}")

    # Spearman (more robust to skew and non-linearity)
    r_spear, p_spear = stats.spearmanr(df["Year"], df["RecreationVisits"])
    print(f"\nSpearman rank correlation:")
    print(f"  rho = {r_spear:.4f}, p = {p_spear:.6f}")

    if p_spear < 0.05:
        direction = "positive" if r_spear > 0 else "negative"
        print(f"\nInterpretation: Significant {direction} association between year and visits.")
    else:
        print("\nInterpretation: No significant monotonic association detected.")

def main():
    df = load_data()
    test_region_differences(df)
    test_covid_periods(df)
    test_year_correlation(df)
    print("\n" + "=" * 60)
    print("Statistical tests complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
