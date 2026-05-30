# AI Impact on the Job Market (2030)

An end-to-end data science project exploring how AI may affect job displacement by 2030, using the Kaggle dataset [AI Impact in Future on Jobs Market in 2030](https://www.kaggle.com/datasets/muhammadwaqas023/ai-impact-in-future-on-jobs-market-in-2030). The project covers data loading, cleaning, exploratory data analysis, feature encoding, visualization, and predictive modeling.

> **Note:** After completing the modeling stage, the dataset was assessed to be likely synthetic — distributions are too uniform, correlations are near-zero across all features, and the target variable (`AI_Replacement_Risk`) shows no meaningful signal from any predictor. For this reason, the project stops at predictive modeling and will not continue to deployment or further analysis.

---

## Project Structure

```
AI Impact/
├── data/
│   ├── raw/               # Original downloaded dataset (AI_Impact_Raw.csv)
│   └── processed/         # Encoded dataset ready for modeling (AI_Impact_Encoded.csv)
├── Notebooks/
│   └── AI_Impact_Orchestrator.ipynb   # End-to-end notebook tying all stages together
├── src/
│   ├── data_loader.py     # Downloads dataset from Kaggle and saves to data/raw/
│   ├── cleaning.py        # Data diagnosis and categorical-to-numeric conversion
│   ├── EDA.py             # Summary stats, value counts, correlation analysis
│   ├── visualizing.py     # Univariate, bivariate, and heatmap visualizations
│   ├── encoding.py        # Feature encoding (ordinal, one-hot, binary, multi-label)
│   └── model_train.py     # Baseline, Linear Regression, and Random Forest regressors
└── README.md
```

---

## Pipeline

| Stage | File | Description |
|---|---|---|
| 1. Data Loading | `data_loader.py` | Downloads dataset via `kagglehub`, saves raw CSV |
| 2. Cleaning / Diagnosis | `cleaning.py` | Checks shape, dtypes, missingness, duplicates |
| 3. EDA | `EDA.py` | Value counts, descriptive stats, correlation matrix |
| 4. Visualization | `visualizing.py` | 7-section visual EDA (distributions, scatter, boxplot, heatmap) |
| 5. Encoding | `encoding.py` | Ordinal, one-hot, binary, and multi-label encoding; saves encoded CSV |
| 6. Modeling | `model_train.py` | Train/test split → scaling → baseline/LR/RF → MAE, RMSE, R² metrics |

---

## Target Variable

**`AI_Replacement_Risk`** — a continuous score from 0.0 to 1.0 representing the estimated risk of a job being replaced by AI.

### Other Key Columns

| Column | Range / Type | Notes |
|---|---|---|
| `Future_Demand_Score` | 0.0 – 1.0 | Projected future demand for the role |
| `Performance_Score` | 2.0 – 5.0 | Employee performance rating |
| `Job_Satisfaction` | 1.0 – 5.0 | Self-reported job satisfaction |
| `Required_Skills` | multi-value string | Comma-separated skills; multi-label encoded |
| `Automation_Level` | Low / Medium / High | Ordinal encoded |
| `Hiring_Trend_2026` | Declining / Stable / Growing | Ordinal encoded |

---

## Encoding Strategy

| Column(s) | Strategy |
|---|---|
| `Education_Level`, `Automation_Level`, `AI_Tool_Usage`, `Company_Size`, `Hiring_Trend_2026` | Ordinal encoding (natural order preserved) |
| `Industry`, `Country`, `Job_Title`, `Remote_Work_Possibility` | One-hot encoding |
| `Upskilling_Needed` | Binary encoding (Yes → 1, No → 0) |
| `Required_Skills` | Multi-label binarization |
| `Employee_ID` | Dropped (identifier, no signal) |

---

## Models Trained

- **Baseline** — `DummyRegressor(strategy="mean")`
- **Linear Regression** — `sklearn.linear_model.LinearRegression`
- **Random Forest** — `RandomForestRegressor(n_estimators=200, random_state=42)`

Evaluation metrics: MAE, RMSE, R²

---

## Setup

**Requirements:** Python 3.10+, a Kaggle API token configured (`~/.kaggle/kaggle.json`).

```bash
# Install dependencies
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn

# Run the orchestrator notebook, or run stages individually from src/
cd src
python data_loader.py   # download raw data
python EDA.py           # print EDA summary
python model_train.py   # train and evaluate models
```

---

## Why the Project Stops Here

The dataset appears to be synthetically generated. Across all EDA and modeling stages, every feature showed near-zero correlation with `AI_Replacement_Risk`, all models performed at or near the dummy baseline, and the distributions of key columns are suspiciously uniform. Drawing conclusions or building a deployed model on top of synthetic data with no real signal would not be meaningful, so this project ends at the predictive modeling stage.

That said, this was a valuable learning experience — it served as a reminder to be more cautious and deliberate when selecting a dataset before starting a project. Checking provenance, inspecting distributions early, and verifying that the data has real-world signal are steps worth doing before investing time in a full pipeline.
