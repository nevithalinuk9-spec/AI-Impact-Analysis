"""
Encode the categorical variables in AI_Impact_Raw.csv.
 
Strategy:
  - Ordinal encoding  : Education_Level, Automation_Level, AI_Tool_Usage,
                        Company_Size, Hiring_Trend_2026   (have a natural order)
  - Binary encoding   : Upskilling_Needed                 (Yes/No -> 1/0)
  - One-hot encoding  : Industry, Country, Job_Title, Remote_Work_Possibility
  - Multi-label       : Required_Skills                   (comma-separated lists) -> I'll decide later if I want to do multi-hot encoding or just explode the rows for analysis and visualization
  - Dropped           : Employee_ID                       (pure identifier)
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import data_loader as ld

df = ld.load_data()

# Defining the columns for each encoding strategy

ordinal_cols = ["Education_Level", "Automation_Level", "AI_Tool_Usage",
                "Company_Size", "Hiring_Trend_2026"]
 
# Order matters: list each column's categories from lowest -> highest
ordinal_categories = [
    ["High School", "Bachelor", "Master", "PhD"],   # Education_Level
    ["Low", "Medium", "High"],                       # Automation_Level
    ["Low", "Moderate", "High"],                      # AI_Tool_Usage
    ["Startup", "Medium", "Enterprise"],             # Company_Size
    ["Declining", "Stable", "Growing"],              # Hiring_Trend_2026
]
 
onehot_cols = ["Industry", "Country", "Job_Title", "Remote_Work_Possibility"]

# Binary encoding for Upskilling_Needed
def binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    df["Upskilling_Needed"] = df["Upskilling_Needed"].map({"No": 0, "Yes": 1})
    #print("Binary encoding applied to 'Upskilling_Needed'")
    return df

# ordinal encoding for the specified columns
def ordinal_encode(df: pd.DataFrame, columns: list=ordinal_cols) -> pd.DataFrame:
    encoder = OrdinalEncoder(categories=ordinal_categories)
    df[columns] = encoder.fit_transform(df[columns])
    #print(f"Ordinal encoding applied to columns: {columns}")
    return df

# one-hot encoding for the specified columns
def onehot_encode(df: pd.DataFrame, columns: list=onehot_cols) -> pd.DataFrame:
  ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
  encoded = ohe.fit_transform(df[columns])

  # Build a labeled dataframe and join it back, dropping the originals
  encoded_df = pd.DataFrame(
      encoded,
      columns=ohe.get_feature_names_out(columns),
      index=df.index,
  )
  df = pd.concat([df.drop(columns=columns), encoded_df], axis=1)
  #print(f"One-hot encoding applied to columns: {columns}")
  return df

def multi_label_encode(df: pd.DataFrame, column: str="Required_Skills") -> pd.DataFrame:
    skills = df["Required_Skills"].str.split(",").apply(lambda l: [s.strip() for s in l])
    mlb = MultiLabelBinarizer()
    skills_df = pd.DataFrame(mlb.fit_transform(skills),
                             columns=[f"skill_{s}" for s in mlb.classes_],
                             index=df.index)
    df = pd.concat([df, skills_df], axis=1)
    return df
    

df = binary_encode(df)
df = ordinal_encode(df)
print(df.shape)
df = onehot_encode(df)
print(df.filter(regex="^(Industry|Country|Job_Title|Remote_Work_Possibility)_").shape)
df = multi_label_encode(df)
print(df.filter(regex="^skill_").shape)



