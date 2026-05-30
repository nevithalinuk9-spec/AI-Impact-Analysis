import pandas as pd
import numpy as np
import data_loader as dl

df = dl.load_data()

def EDA(df: pd.DataFrame=df) -> None:
    print("==DataFrame info==\n")
    print(df.info())
    for col in df.columns:
        print(f"\n{df[col].value_counts()}")
    print("\n==Numeric columns summary==\n")
    print(df.describe().round(2))
    print("\n==Categorical columns summary==\n")
    print(df.describe(include="object").T)
    print("\n==Missing values summary==\n")
    print(df.isnull().sum())
    print("\n==Unique values per categorical column==\n")
    for col in df.select_dtypes(include="object").columns:
        print(f"{col}: {df[col].nunique()} unique values")
    print("\n==Ranges of key columns==\n")
    # Ranges columns summary
    print(df[["AI_Replacement_Risk", "Future_Demand_Score", "Performance_Score", "Job_Satisfaction"]].describe().round(2))

"""
Found out that,
1. AI_Replacement_Risk -> 0.0 - 1.0 (1.0 being the highest)
2. Future_Demand_Score -> 0.0 - 1.0 (1.0 being the highest)
3. Performance Score -> 2.0 - 5.0 (5.0 being the highest)
4. Job_Satisfaction -> 1.0 - 5.0 (5.0 being the highest)
5. Required_Skills has multiple values per column therefore it needs to be exploded when analysing or visualizing
"""
def correlation_analysis(df: pd.DataFrame=df) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()

if __name__ == "__main__":
    EDA(df)