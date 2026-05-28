import os

import matplotlib
matplotlib.use("Qt5Agg")   # use Qt instead of Tk because it didn't respond
import seaborn as sns 
import data_loader as dl
import matplotlib.pyplot as plt

df = dl.load_data()

"""
Comprehensive early-EDA visualizations for AI_Impact_Raw.csv.
Run on cleaned, PRE-encoded data (categorical labels stay readable).

Covers the full framework:
  1. Univariate  - numeric histograms
  2. Univariate  - categorical counts
  3. Target      - distribution of AI_Replacement_Risk
  4. Bivariate   - numeric features vs target (scatter)
  5. Bivariate   - categorical features vs target (boxplot)
  6. Feature x feature - correlation heatmap
  7. Data quality      - missing-value counts

"""

df = dl.load_data()
target = "AI_Replacement_Risk"

numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in df.select_dtypes(exclude="number").columns
                    if c not in ("Employee_ID", "Required_Skills")]
# numeric features = numeric columns minus the target itself
numeric_features = [c for c in numeric_cols if c != target]


def grid(items, ncols=3, size=(5, 5)):
    """Helper: make a subplot grid sized to the number of items."""
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(size[0] * ncols, size[1] * nrows),
                             constrained_layout=True)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.set_visible(False)          # hide unused cells
    return fig, axes

# ----------------------------------------------------------------------
# 1) Univariate: numeric distributions
# ----------------------------------------------------------------------
def univariate_numeric(df, numeric_cols):
    fig, axes = grid(numeric_cols)
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
    fig.suptitle("1. Numeric Distributions", fontsize=15)
    fig.subplots_adjust(top=0.93)
    plt.show()

# ----------------------------------------------------------------------
# 2) Univariate: categorical counts
# ----------------------------------------------------------------------
def univariate_categorical(df, categorical_cols):
    fig, axes = grid(categorical_cols)
    for ax, col in zip(axes, categorical_cols):
        order = df[col].value_counts().index
        sns.countplot(y=df[col], order=order, ax=ax, color="#55A868")
        ax.set_title(col); ax.set_ylabel("")
    fig.suptitle("2. Categorical Counts", fontsize=15)
    fig.subplots_adjust(top=0.93)
    plt.show()

# ----------------------------------------------------------------------
# 3) Target distribution
# ----------------------------------------------------------------------
def target_distribution(df, target):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df[target], kde=True, ax=ax, color="#C44E52")
    ax.set_title(f"3. Target Distribution: {target}")
    plt.show()

# ----------------------------------------------------------------------
# 4) Bivariate: numeric features vs target (scatter)
#    Cloud with no slope = no relationship.
# ----------------------------------------------------------------------
def bivariate_numeric(df, numeric_features, target):
    fig, axes = grid(numeric_features)
    for ax, col in zip(axes, numeric_features):
        sns.regplot(x=df[col], y=df[target], ax=ax,
                    scatter_kws={"alpha": 0.5,"s": 12, "color": "#4C72B0"},
                    line_kws={"color": "#C44E52"})
    fig.suptitle("4. Numeric Features vs Target", fontsize=15)
    fig.subplots_adjust(top=0.93)
plt.show()

# ----------------------------------------------------------------------
# 5) Bivariate: categorical features vs target (boxplot)
#    Boxes all at the same level = category tells you nothing.
# ----------------------------------------------------------------------
def bivariate_categorical(df, categorical_cols, target):
    fig, axes = grid(categorical_cols)
    for ax, col in zip(axes, categorical_cols):
        sns.boxplot(x=df[col], y=df[target], ax=ax, color="#55A868")
        ax.set_title(f"{target} by {col}")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("")
    fig.suptitle("5. Categorical Features vs Target", fontsize=15)
    fig.subplots_adjust(top=0.93)
    plt.show()

# ----------------------------------------------------------------------
# 6) Feature x feature: correlation heatmap
# ----------------------------------------------------------------------
def correlation_heatmap(df, numeric_cols):
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("6. Correlation Heatmap")
    plt.show()

# ----------------------------------------------------------------------
# 7) Data quality: missing-value counts
# ----------------------------------------------------------------------
def data_quality_missing(df):
    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#35207B")
    ax.set_title("7. Missing Values per Column")
    ax.set_xlabel("count of NaN")
    plt.show()
    print("Total missing values:", int(missing.sum()))