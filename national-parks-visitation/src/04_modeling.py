"""
04_modeling.py
Baseline + advanced machine learning models for National Park visitation.
Includes lagged visits feature for temporal dependence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "parks_visits_cleaned.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-year visits for each park."""
    df = df.sort_values(["ParkName", "Year"]).copy()
    df["lag1_visits"] = df.groupby("ParkName")["RecreationVisits"].shift(1)
    df["lag1_log_visits"] = df.groupby("ParkName")["log_visits"].shift(1)
    # Drop the first year of each park (no lag available)
    df = df.dropna(subset=["lag1_log_visits"]).reset_index(drop=True)
    print(f"After adding lag features and dropping first years: {df.shape[0]} rows")
    return df

def prepare_features(df: pd.DataFrame):
    feature_cols = ["Year", "park_age", "Region", "covid_period", "lag1_log_visits"]
    target_col = "log_visits"

    data = df[feature_cols + [target_col, "RecreationVisits"]].dropna().copy()
    X = data[feature_cols]
    y = data[target_col]
    y_raw = data["RecreationVisits"]
    return X, y, y_raw

def build_preprocessor():
    numeric_features = ["Year", "park_age", "lag1_log_visits"]
    categorical_features = ["Region", "covid_period"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    return preprocessor

def evaluate(y_true, y_pred, label: str):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{label}")
    print(f"  RMSE (log scale) = {rmse:.4f}")
    print(f"  MAE  (log scale) = {mae:.4f}")
    print(f"  R²             = {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def temporal_split(X, y, y_raw, test_start=2020):
    train_mask = X["Year"] < test_start
    test_mask = X["Year"] >= test_start

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    y_raw_train, y_raw_test = y_raw.loc[train_mask], y_raw.loc[test_mask]

    print(f"Temporal split (test years >= {test_start}):")
    print(f"  Train: {len(X_train)} rows ({X_train['Year'].min()}-{X_train['Year'].max()})")
    print(f"  Test:  {len(X_test)} rows ({X_test['Year'].min()}-{X_test['Year'].max()})")
    return X_train, X_test, y_train, y_test, y_raw_train, y_raw_test

def plot_pred_vs_actual(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual log(Visits + 1)")
    ax.set_ylabel("Predicted log(Visits + 1)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_feature_importance(importances, feature_names, title, filename):
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(idx)), importances[idx][::-1], color="#2c7bb6")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def main():
    df = load_data()
    df = add_lag_features(df)
    X, y, y_raw = prepare_features(df)

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = temporal_split(
        X, y, y_raw, test_start=2020
    )

    preprocessor = build_preprocessor()

    # ---- Baseline: Linear Regression ----
    print("\n" + "=" * 60)
    print("BASELINE MODEL: Linear Regression (with lag)")
    print("=" * 60)

    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    metrics_baseline = evaluate(y_test, y_pred_baseline, "Linear Regression (Test)")

    plot_pred_vs_actual(
        y_test, y_pred_baseline,
        "Baseline: Linear Regression (with lag)",
        "07_baseline_pred_vs_actual.png"
    )

    # ---- Advanced: Random Forest ----
    print("\n" + "=" * 60)
    print("ADVANCED MODEL: Random Forest (with lag)")
    print("=" * 60)

    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_rf = evaluate(y_test, y_pred_rf, "Random Forest (Test)")

    plot_pred_vs_actual(
        y_test, y_pred_rf,
        "Advanced: Random Forest (with lag)",
        "08_rf_pred_vs_actual.png"
    )

    # Feature names and importance
    ohe = rf.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["Region", "covid_period"]))
    feature_names = ["Year", "park_age", "lag1_log_visits"] + cat_names

    plot_feature_importance(
        rf.named_steps["regressor"].feature_importances_,
        feature_names,
        "Random Forest Feature Importances",
        "09_rf_feature_importance.png"
    )

    # ---- Extra: HistGradientBoosting ----
    print("\n" + "=" * 60)
    print("ADDITIONAL MODEL: HistGradientBoosting")
    print("=" * 60)

    hgb = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=8,
            learning_rate=0.08,
            random_state=RANDOM_STATE
        ))
    ])
    hgb.fit(X_train, y_train)
    y_pred_hgb = hgb.predict(X_test)
    metrics_hgb = evaluate(y_test, y_pred_hgb, "HistGradientBoosting (Test)")

    # ---- Comparison ----
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test set, log scale, with lag feature)")
    print("=" * 60)
    print(f"{'Model':<28} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 60)
    print(f"{'Linear Regression':<28} {metrics_baseline['rmse']:>10.4f} {metrics_baseline['mae']:>10.4f} {metrics_baseline['r2']:>10.4f}")
    print(f"{'Random Forest':<28} {metrics_rf['rmse']:>10.4f} {metrics_rf['mae']:>10.4f} {metrics_rf['r2']:>10.4f}")
    print(f"{'HistGradientBoosting':<28} {metrics_hgb['rmse']:>10.4f} {metrics_hgb['mae']:>10.4f} {metrics_hgb['r2']:>10.4f}")

    print("\nNotes:")
    print("- Added lag1_log_visits (previous year) to capture temporal dependence.")
    print("- Temporal split: train < 2020, test 2020-2024.")
    print("- Target remains log(Visits + 1).")

if __name__ == "__main__":
    main()
