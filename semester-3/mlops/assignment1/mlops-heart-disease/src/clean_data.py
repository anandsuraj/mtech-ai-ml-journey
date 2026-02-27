from src.preprocess import load_data, drop_missing, fix_target

df = load_data("data/raw/heart.csv")
df = drop_missing(df)
df = fix_target(df)

df.to_csv("data/processed/heart_clean.csv", index=False)
print("Saved cleaned dataset to data/processed/heart_clean.csv")