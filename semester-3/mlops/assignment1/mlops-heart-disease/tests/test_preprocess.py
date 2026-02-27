import pandas as pd
import pytest
from src.preprocess import (
    load_data,
    drop_missing,
    fix_target,
    split_features_target,
    build_preprocessor
)

def test_load_data_not_empty():
    df = load_data("data/heart.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] >= 14   # dataset should have at least 14 features

def test_drop_missing_reduces_rows():
    df = load_data("data/heart.csv")
    df_clean = drop_missing(df)
    # Either rows reduce or remain same (if no missing values)
    assert df_clean.shape[0] <= df.shape[0]

def test_target_is_binary():
    df = load_data("data/heart.csv")
    df = drop_missing(df)
    df = fix_target(df)
    assert set(df["target"].unique()).issubset({0, 1})

def test_split_features_target():
    df = pd.DataFrame({"age":[29,40], "chol":[200,180], "target":[0,1]})
    X, y = split_features_target(df)
    assert "target" not in X.columns
    assert len(y) == len(X)

def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None

def test_preprocessing_pipeline_output_rows():
    df = load_data("data/heart.csv")
    df = drop_missing(df)
    df = fix_target(df)
    X, y = split_features_target(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Row count must remain same after transformation
    assert X_transformed.shape[0] == X.shape[0]