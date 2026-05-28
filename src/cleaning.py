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
    
    # Check if the str column types can be numeric
    for col in df.select_dtypes(include="str").columns:
        try:
            pd.to_numeric(df[col])
            print(f"{col}: could be numeric")
        except (ValueError, TypeError):
            pass
diagnosis()

# Convert categorical to numeric using factorize
columns = ["Education_Level", "Remote_Work_Possibility", "Automation_Level", "Company_Size", "AI_Tool_Usage", "Upskilling_Needed", "Hiring_Trend_2026"] 
def categorical_to_numeric(df: pd.DataFrame=df, columns: list=columns) -> pd.DataFrame:
    df_copy = df.copy()
    # selecting only the catergorical columns and applying factorize to convert them to numeric
    for col in columns:
        df_copy[col] = pd.factorize(df_copy[col])[0]
    return df_copy

df1 = categorical_to_numeric()
