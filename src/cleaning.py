import pandas as pd
import data_loader as ld

df = ld.load_data()

def diagnosis(df: pd.DataFrame=df) -> None: 
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing_per_column": df.isnull().sum(),
        "missing_pct": df.isnull().mean().mul(100).round(2),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_summary": df.describe(),
    }

    # Per column unique value counts per catergorical columns
    str_cols = df.select_dtypes(include="str").columns
    report["unique_counts"] = {col: df[col].nunique() for col in str_cols}
    # Memory footprint
    report["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    for key, value in report.items():
        print(f"\n=={key}==\n")
        print(value)
diagnosis()
