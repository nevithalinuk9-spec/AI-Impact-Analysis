import matplotlib
matplotlib.use("Qt5Agg")   # use Qt instead of Tk because it didn't respond
import seaborn as sns 
import data_loader as dl
import matplotlib.pyplot as plt

df = dl.load_data()
plt.bar(df["Job_Title"], df["Average_Salary_USD"], data=df)
plt.show()