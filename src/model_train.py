"""
Modeling stage for AI_Impact_Raw.csv  (target = AI_Replacement_Risk, regression).

Pipeline:  encode -> split -> scale -> train (baseline + 2 models) -> evaluate -> plot

Produces two results figures in 'model_plots/':
  - predicted vs actual scatter
  - residuals vs predicted
"""

import os
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
print(len(X.columns), "features after encoding")
y = df["AI_Replacement_Risk"]

# ======================================================================
# 1) Train / test split  (do this BEFORE scaling to avoid leakage)
# ======================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)