"""
Modeling stage for AI_Impact_Raw.csv  (target = AI_Replacement_Risk, regression).

Pipeline:  encode -> split -> scale -> train (baseline + 2 models) -> evaluate -> plot

Produces two results figures in 'model_plots/':
  - predicted vs actual scatter
  - residuals vs predicted
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import data_loader as ld
import encoding as enc

df = ld.load_data()
X = enc.predictors(df)
y = df["AI_Replacement_Risk"]

# ======================================================================
# 1) Train / test split  (do this BEFORE scaling to avoid leakage)
# ======================================================================
def train_test_split(X, y, test_size=0.2, random_state=42):
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
      )
  return X_train, X_test, y_train, y_test

# ======================================================================
# 2) Scale  (fit on train only, transform both)
# ======================================================================
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s

# ======================================================================
# 3) Train models: a dumb baseline + two real regressors
# ======================================================================
def train_models(X_train_s, y_train, X_test_s, y_test):
  models = {
      "Baseline (mean)":  DummyRegressor(strategy="mean"),
      "Linear Regression": LinearRegression(),
      "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
  }

  results = {}
  for name, model in models.items():
      model.fit(X_train_s, y_train)
      preds = model.predict(X_test_s)
      results[name] = {
          "model": model,
          "preds": preds,
          "MAE":  mean_absolute_error(y_test, preds),
          "RMSE": root_mean_squared_error(y_test, preds),
          "R2":   r2_score(y_test, preds),
      }
  return results

# ======================================================================
# 4) Report metrics
# ======================================================================
def get_results(results):
  print(f"{'Model':<20}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
  print("-" * 44)
  for name, r in results.items():
      print(f"{name:<20}{r['MAE']:>8.3f}{r['RMSE']:>8.3f}{r['R2']:>8.3f}")

# ======================================================================
# 5a) Predicted vs actual
# ======================================================================
def plot_results(results, y_test):
  # Use the best real model (by R2, excluding baseline) for the plots
  best = max((n for n in results if n != "Baseline (mean)"),
            key=lambda n: results[n]["R2"])
  preds = results[best]["preds"]
  residuals = y_test - preds

  fig, ax = plt.subplots(figsize=(7, 7))
  ax.scatter(y_test, preds, alpha=0.4, edgecolor="none", color="#4C72B0")
  lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
  ax.plot(lims, lims, "--", color="red", label="perfect prediction")
  ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
  ax.set_title(f"Predicted vs Actual — {best}")
  ax.legend()
  fig.tight_layout()
  plt.show()

# ======================================================================
# 5b) Residual plot
# ======================================================================
  fig, ax = plt.subplots(figsize=(7, 5))
  ax.scatter(preds, residuals, alpha=0.4, edgecolor="none", color="#55A868")
  ax.axhline(0, color="red", linestyle="--")
  ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (actual - predicted)")
  ax.set_title(f"Residual Plot — {best}")
  fig.tight_layout()
  plt.show()