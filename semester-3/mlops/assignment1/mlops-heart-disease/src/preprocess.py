import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

def load_data(path: str):
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS
    )
    return df

def drop_missing(df):
    df_clean = df.replace("?", pd.NA)
    df_clean = df_clean.dropna()
    return df_clean

def find_missing_values(df):
    return (df == "?").sum()

def fix_target(df):
    df = df.copy()
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df

def build_preprocessor():
    numeric_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak"
    ]

    categorical_features = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal"
    ]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", "passthrough", categorical_features)
        ]
    )

    return preprocessor

def split_features_target(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y