import pandas as pd
import data_loader as ld

df = ld.load_data()

def diagnosis(df: pd.DataFrame=df) -> None: 
    # Shape and types
    print(df.shape)
    print(df.dtypes)

    # Missing values
    print(df.isnull().sum())

    # Duplicates
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Summary stats for numeric columns
    print(df.describe())

    # Unique values in categorical columns
    for col in df.select_dtypes(include="str").columns:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())

diagnosis()