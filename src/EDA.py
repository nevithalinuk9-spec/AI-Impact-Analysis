import pandas as pd
import numpy as np
import data_loader as dl

df = dl.load_data()
#for col in df.columns:
 #   print(f"\n{df[col].value_counts()}")

#print(df["Job_Growth_2030"].describe().round(2))
#print(df[["AI_Replacement_Risk", "Future_Demand_Score", "Performance_Score", "Job_Satisfaction"]].describe().round(2))

"""
Found out that,
1. AI_Replacement_Risk -> 0.0 - 1.0 (1.0 being the highest)
2. Future_Demand_Score -> 0.0 - 1.0 (1.0 being the highest)
3. Performance Score -> 2.0 - 5.0 (5.0 being the highest)
4. Job_Satisfaction -> 1.0 - 5.0 (5.0 being the highest)
5. Required_Skills has multiple values per column therefore it needs to be exploded when analysing or visualizing
"""
#print(df.info())
#print(df["Job_Title"].value_counts())




