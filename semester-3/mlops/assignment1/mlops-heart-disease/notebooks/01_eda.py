# # Exploratory Data Analysis (EDA)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/heart_clean.csv")
df.head()

plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Class Balance: Heart Disease Presence")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Samples")
plt.show()

df.hist(figsize=(14,10), bins=20)
plt.suptitle("Feature Distributions (Histograms)", fontsize=16)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap of Features")
plt.show()