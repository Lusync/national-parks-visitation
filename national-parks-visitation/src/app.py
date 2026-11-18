"""
National Park Visitation Analysis
Streamlit interactive report
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="National Park Visitation Analysis",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "parks_visits_cleaned.csv"
FIG_DIR = ROOT / "figures"

sns.set_theme(style="whitegrid", palette="muted")

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset & Cleaning",
        "Exploratory Analysis",
        "Statistical Tests",
        "Machine Learning",
        "AI Assistant (LangChain)",
        "Conclusions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "U.S. National Parks Recreation Visits (1979–2024)\n\n"
    "Interactive report for the full data science workflow."
)

# -----------------------------------------------------------------------------
# PAGE: Project Overview
# -----------------------------------------------------------------------------
if page == "Project Overview":
    st.title("🏞️ National Park Visitation Analysis")
    st.markdown("### Predicting and understanding recreation visits across U.S. National Parks")

    st.markdown("""
    **Research Question**  
    Can we predict annual recreation visits to a U.S. National Park from its region, 
    state, temporal patterns, and previous-year visits? How did visitation patterns 
    change across the pre-pandemic (before 2020), pandemic (2020–2021), and recovery 
    (2022–2024) periods?

    **Dataset**  
    Annual recreation visit counts for the 63 U.S. National Parks from 1979 to 2024.  
    Source: National Park Service Visitor Use Statistics (cleaned version).

    **Workflow covered in this app**
    - Data cleaning and feature engineering
    - Exploratory data analysis
    - Formal statistical hypothesis testing
    - Baseline and advanced machine learning models
    - Interpretation and conclusions
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parks", df["ParkName"].nunique())
    col2.metric("Years", f"{df['Year'].min()}–{df['Year'].max()}")
    col3.metric("Total Records", f"{len(df):,}")
    col4.metric("Avg Visits", f"{df['RecreationVisits'].mean()/1e6:.2f}M")

# -----------------------------------------------------------------------------
# PAGE: Dataset & Cleaning
# -----------------------------------------------------------------------------
elif page == "Dataset & Cleaning":
    st.title("Dataset & Cleaning")

    st.subheader("Raw structure after cleaning")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Key cleaning decisions")
    st.markdown("""
    - **No missing values** were present in the cleaned source file.
    - **8 parks** first appear after 1979 (they did not exist yet).  
      These earlier years were **not imputed**. Each park’s series begins at its first recorded year.
    - **4 zero-visit records** were flagged (`is_zero_visits`).
    - Created features:
        - `first_year` / `park_age`
        - `covid_period` (Pre-pandemic / Pandemic / Recovery)
        - `log_visits` (log1p transform to address strong right skew ≈ 3.42)
        - Later: `lag1_log_visits` (previous year’s visits)
    """)

    st.subheader("Parks that start after 1979")
    late = (
        df.groupby("ParkName")["first_year"]
        .min()
        .loc[lambda x: x > 1979]
        .sort_values()
    )
    st.dataframe(late.reset_index().rename(columns={"first_year": "First Year"}), use_container_width=True)

    st.subheader("Distribution: Raw vs Log-transformed")
    if (FIG_DIR / "04_visits_distribution.png").exists():
        st.image(str(FIG_DIR / "04_visits_distribution.png"), use_container_width=True)
    else:
        st.info("Figure not found — run the EDA script first.")

# -----------------------------------------------------------------------------
# PAGE: Exploratory Analysis
# -----------------------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    st.subheader("Visits over time")
    if (FIG_DIR / "01_visits_over_time.png").exists():
        st.image(str(FIG_DIR / "01_visits_over_time.png"), use_container_width=True)

    st.subheader("Average visits by Region")
    if (FIG_DIR / "02_avg_visits_by_region.png").exists():
        st.image(str(FIG_DIR / "02_avg_visits_by_region.png"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("COVID period distribution (log scale)")
        if (FIG_DIR / "03_covid_period_boxplot.png").exists():
            st.image(str(FIG_DIR / "03_covid_period_boxplot.png"), use_container_width=True)
    with col2:
        st.subheader("COVID impact by Region")
        if (FIG_DIR / "06_covid_by_region.png").exists():
            st.image(str(FIG_DIR / "06_covid_by_region.png"), use_container_width=True)

    st.subheader("Top parks by average annual visits")
    if (FIG_DIR / "05_top_parks.png").exists():
        st.image(str(FIG_DIR / "05_top_parks.png"), use_container_width=True)

    st.subheader("Summary statistics by COVID period")
    period_summary = (
        df.groupby("covid_period")["RecreationVisits"]
        .agg(["mean", "median", "count"])
        .round(0)
        .reindex(["Pre-pandemic", "Pandemic", "Recovery"])
    )
    st.dataframe(period_summary, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE: Statistical Tests
# -----------------------------------------------------------------------------
elif page == "Statistical Tests":
    st.title("Statistical Analysis & Hypothesis Testing")

    st.markdown("### Test 1: Do mean visits differ across NPS Regions?")
    st.markdown("""
    **H₀**: Mean recreation visits are equal across all regions.  
    **H₁**: At least one region differs.
    """)

    groups = [g["RecreationVisits"].values for _, g in df.groupby("Region")]
    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kw = stats.kruskal(*groups)

    c1, c2 = st.columns(2)
    c1.metric("ANOVA F-statistic", f"{f_stat:.3f}")
    c1.metric("ANOVA p-value", f"{p_anova:.2e}")
    c2.metric("Kruskal-Wallis H", f"{h_stat:.3f}")
    c2.metric("Kruskal-Wallis p-value", f"{p_kw:.2e}")

    st.success("Result: Reject H₀. Strong evidence that visitation differs across regions.")

    st.markdown("---")
    st.markdown("### Test 2: Differences across COVID periods")
    st.markdown("""
    **H₀**: Mean visits are equal across Pre-pandemic, Pandemic, and Recovery.  
    **H₁**: At least one period differs.
    """)

    period_order = ["Pre-pandemic", "Pandemic", "Recovery"]
    groups_p = [df.loc[df["covid_period"] == p, "RecreationVisits"].values for p in period_order]
    f_p, p_anova_p = stats.f_oneway(*groups_p)
    h_p, p_kw_p = stats.kruskal(*groups_p)

    c1, c2 = st.columns(2)
    c1.metric("ANOVA F", f"{f_p:.3f}", help=f"p = {p_anova_p:.4f}")
    c2.metric("Kruskal-Wallis H", f"{h_p:.3f}", help=f"p = {p_kw_p:.4f}")

    st.markdown("**Pairwise Mann-Whitney U tests**")
    pairs = [("Pre-pandemic", "Pandemic"), ("Pre-pandemic", "Recovery"), ("Pandemic", "Recovery")]
    rows = []
    for a, b in pairs:
        x = df.loc[df["covid_period"] == a, "RecreationVisits"]
        y = df.loc[df["covid_period"] == b, "RecreationVisits"]
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"Comparison": f"{a} vs {b}", "U": round(u, 1), "p-value": f"{p:.4f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.info("Note: The pandemic period contains only two years, so statistical power is limited.")

    st.markdown("---")
    st.markdown("### Test 3: Association between Year and Visits")
    r_p, p_p = stats.pearsonr(df["Year"], df["RecreationVisits"])
    r_s, p_s = stats.spearmanr(df["Year"], df["RecreationVisits"])

    c1, c2 = st.columns(2)
    c1.metric("Pearson r", f"{r_p:.4f}", help=f"p = {p_p:.2e}")
    c2.metric("Spearman ρ", f"{r_s:.4f}", help=f"p = {p_s:.4f}")
    st.success("Small but statistically significant positive association over time.")

# -----------------------------------------------------------------------------
# PAGE: Machine Learning
# -----------------------------------------------------------------------------
elif page == "Machine Learning":
    st.title("Machine Learning Models")

    st.markdown("""
    **Target**: `log_visits` (log1p of RecreationVisits) to reduce skew.  
    **Features**: Year, park_age, Region, covid_period, lag1_log_visits (previous year).  
    **Validation**: Temporal split — train on years before 2020, test on 2020–2024.
    """)

    st.subheader("Model Performance (Test set: 2020–2024)")

    results = pd.DataFrame({
        "Model": ["Linear Regression (Baseline)", "Random Forest", "HistGradientBoosting"],
        "RMSE (log)": [0.5511, 0.5937, 0.6313],
        "MAE (log)": [0.3174, 0.3277, 0.3479],
        "R²": [0.8945, 0.8775, 0.8615],
    })
    st.dataframe(results, use_container_width=True)

    st.success("Best performing model on the temporal hold-out: **Linear Regression** (R² ≈ 0.89).")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline predictions")
        if (FIG_DIR / "07_baseline_pred_vs_actual.png").exists():
            st.image(str(FIG_DIR / "07_baseline_pred_vs_actual.png"), use_container_width=True)
    with col2:
        st.subheader("Random Forest predictions")
        if (FIG_DIR / "08_rf_pred_vs_actual.png").exists():
            st.image(str(FIG_DIR / "08_rf_pred_vs_actual.png"), use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    if (FIG_DIR / "09_rf_feature_importance.png").exists():
        st.image(str(FIG_DIR / "09_rf_feature_importance.png"), use_container_width=True)

    st.markdown("""
    **Interpretation**  
    The previous year’s visits dominate predictive power. Region and COVID period 
    contribute additional signal. The strong performance of the linear model suggests 
    that, once lag is accounted for, the remaining relationships are relatively linear 
    on the log scale.
    """)

# -----------------------------------------------------------------------------
# PAGE: AI Assistant (LangChain placeholder)
# -----------------------------------------------------------------------------
elif page == "AI Assistant (LangChain)":
    st.title("AI-Assisted Analysis")

    st.markdown("""
    This section will integrate an LLM through **LangChain** + an API 
    (as required by the project).

    **Planned capabilities**
    - Ask natural-language questions about the data and results
    - Get help interpreting statistical tests and model outputs
    - Critically evaluate AI suggestions (accept / modify / reject) and document the decision

    **Status**: Placeholder — LangChain integration coming next.
    """)

    st.info("Once the API key and LangChain chain are configured, this page will become interactive.")

    user_q = st.text_input("Example question (not yet connected):", 
                           placeholder="Why did Linear Regression outperform Random Forest on the 2020–2024 test set?")
    if user_q:
        st.write("*(LLM response will appear here once LangChain is wired up.)*")

# -----------------------------------------------------------------------------
# PAGE: Conclusions
# -----------------------------------------------------------------------------
elif page == "Conclusions":
    st.title("Conclusions")

    st.markdown("""
    ### Main Findings

    1. **Regional differences are large and statistically significant.**  
       Southeast and Northeast parks have substantially higher average visitation than Alaska.

    2. **COVID impact is visible but mixed in formal tests.**  
       Overall visitation recovered in 2022–2024; some pairwise comparisons reach significance, 
       but the short pandemic window limits power.

    3. **Previous-year visits are by far the strongest predictor.**  
       Adding a one-year lag raised test-set R² from ~0.45 to ~0.89 for the linear model.

    4. **A simple linear model currently generalizes best** across the COVID structural break, 
       outperforming Random Forest and HistGradientBoosting on the 2020–2024 hold-out.

    ### Limitations
    - Many external drivers (weather, economy, fuel prices, park-specific events, social media) 
      are not in the dataset.
    - The pandemic period contains only two years.
    - Park-level random effects or more sophisticated time-series models could be explored next.

    ### Next Steps for the Project
    - Complete LangChain + LLM integration and document AI interactions
    - Polish Streamlit UI and export options
    - Finalize README and GitHub repository structure
    """)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("National Park Visitation Project")
