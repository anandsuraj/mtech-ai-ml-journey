#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import urllib.request

# Create directories
os.makedirs("../data/raw", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)

# Download dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
raw_path = "../data/raw/heart.csv"
urllib.request.urlretrieve(url, raw_path)
print("Downloaded dataset to", raw_path)

# Define column names
column_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Load dataset
df = pd.read_csv(raw_path, names=column_names)

# Replace '?' with NaN and convert to numeric
df = df.replace("?", pd.NA)
df = df.apply(pd.to_numeric)

print("Missing values per column:\n", df.isna().sum())

# Drop rows with missing values
print("Before:", df.shape)
df_clean = df.dropna().copy()
print("After:", df_clean.shape)

# Convert target to binary (0 = no disease, 1 = disease)
df_clean.loc[:, "target"] = (df_clean["target"] != 0).astype(int)

print("Target distribution:\n", df_clean["target"].value_counts())

# Save cleaned dataset
clean_path = "../data/processed/heart_clean.csv"
df_clean.to_csv(clean_path, index=False)
print("Cleaned dataset saved to", clean_path)


# In[3]:


import os
print(os.path.abspath("../data/raw/heart.csv"))


# In[ ]:




