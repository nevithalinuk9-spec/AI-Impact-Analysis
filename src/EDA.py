import pandas as pd
import numpy as np
import data_loader as dl

df = dl.load_data()
for col in df.columns:
    #print(f"\n{df[col].value_counts()}")


print(df["Job_Growth_2030"].describe().round(2))
print(df[["AI_Replacement_Risk", "Future_Demand_Score", "Performance_Score", "Job_Satisfaction"]].describe().round(2))