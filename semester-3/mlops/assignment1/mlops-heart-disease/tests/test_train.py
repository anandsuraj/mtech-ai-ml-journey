import os
import pytest
from sklearn.pipeline import Pipeline
from src.train import train_logistic_regression, train_random_forest, get_env
from src.preprocess import load_data, drop_missing, fix_target, split_features_target
from sklearn.model_selection import train_test_split


def prepare_data():
    """Helper to load and preprocess the dataset once per test."""
    df = load_data("data/heart.csv")
    df = drop_missing(df)
    df = fix_target(df)
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_train_logistic_regression():
    X_train, X_test, y_train, y_test = prepare_data()
    env = get_env()

    pipeline, acc = train_logistic_regression(X_train, y_train, X_test, y_test, version=1, env=env)

    assert pipeline is not None
    assert isinstance(pipeline, Pipeline)
    assert acc > 0.75


def test_train_random_forest():
    X_train, X_test, y_train, y_test = prepare_data()
    env = get_env()

    pipeline, acc = train_random_forest(X_train, y_train, X_test, y_test, version=1, env=env)

    assert pipeline is not None
    assert isinstance(pipeline, Pipeline)
    assert acc > 0.75


def test_artifacts_created():
    X_train, X_test, y_train, y_test = prepare_data()
    env = get_env()

    # Run both trainings once
    lr_pipeline, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test, version=1, env=env)
    rf_pipeline, rf_acc = train_random_forest(X_train, y_train, X_test, y_test, version=1, env=env)

    # Find latest version folder inside artifacts/{env}
    assert os.path.exists("artifacts")
    env_dir = os.path.join("artifacts", env)
    assert os.path.exists(env_dir), f"No artifacts found for env={env}"

    versions = [d for d in os.listdir(env_dir) if d.startswith("v")]
    assert versions, "No versioned artifact folders found"
    latest_version = sorted(versions)[-1]
    artifacts_dir = os.path.join(env_dir, latest_version)

    # Check Logistic Regression artifacts
    files = os.listdir(artifacts_dir)
    assert any("LogisticRegression_confusion_matrix" in f for f in files)
    assert any("LogisticRegression_classification_report" in f for f in files)

    # Check Random Forest artifacts
    assert any("RandomForest_confusion_matrix" in f for f in files)
    assert any("RandomForest_classification_report" in f for f in files)
