# National Park Visitation Analysis

Interactive data science project exploring annual recreation visits to the 63 U.S. National Parks (1979–2024).

## Research Question

Can we predict annual recreation visits to a U.S. National Park from its region, temporal patterns, and previous-year visits? How did visitation change across the pre-pandemic (before 2020), pandemic (2020–2021), and recovery (2022–2024) periods?

## Dataset

- **Source**: U.S. National Park Service Visitor Use Statistics (cleaned version)
- **File**: `data/raw/US-National-Parks_RecreationVisits_1979-2024.csv`
- **Records**: 2,846 rows × 63 parks × years 1979–2024
- **Target**: `RecreationVisits`

## Project Structure

```
national-parks-visitation/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned data with engineered features
├── src/
│   ├── 01_load_and_clean.py
│   ├── 02_eda.py
│   ├── 03_statistical_tests.py
│   ├── 04_modeling.py
│   └── app.py                # Streamlit application
├── figures/                  # Generated plots
├── notebooks/
├── report/
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn streamlit
```

### 2. Reproduce analysis (optional)
```bash
python src/01_load_and_clean.py
python src/02_eda.py
python src/03_statistical_tests.py
python src/04_modeling.py
```

### 3. Launch the Streamlit app
```bash
streamlit run src/app.py
```

## Key Results (Summary)

- Strong regional differences in visitation (Kruskal-Wallis p ≈ 0).
- Previous-year visits are the dominant predictor.
- Linear Regression achieved R² ≈ 0.89 on the 2020–2024 temporal hold-out after including a lag feature.
- COVID-period effects are visible in the data; formal tests are mixed due to the short pandemic window.

## Requirements Covered

- Data cleaning & preprocessing (including proper handling of parks that did not yet exist)
- Exploratory data analysis with visualizations
- Formal statistical hypothesis testing
- Baseline + advanced machine learning models with evaluation
- Interactive Streamlit report
- (LangChain + LLM integration — in progress)
