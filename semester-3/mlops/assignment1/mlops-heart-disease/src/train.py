import os
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)

from src.preprocess import (
    load_data,
    drop_missing,
    fix_target,
    split_features_target,
    build_preprocessor
)

def get_env():
    return os.getenv("ENV", "development")

def get_next_model_version(base_path):
    if not os.path.exists(base_path):
        return 1
    versions = [
        int(d.replace("v", ""))
        for d in os.listdir(base_path)
        if d.startswith("v") and d.replace("v", "").isdigit()
    ]
    if not versions:
        return 1
    return max(versions, default=0) + 1

def log_confusion_matrix(y_true, y_pred, model_name, version, env, output_dir="artifacts"):
    run_dir = os.path.join(output_dir, env, f"v{version}")
    os.makedirs(run_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_file = os.path.join(run_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_file)
    plt.close()

    mlflow.log_artifact(cm_file, artifact_path="evaluation")

def log_classification_report(y_true, y_pred, model_name, version,env, output_dir="artifacts"):
    run_dir = os.path.join(output_dir, env, f"v{version}")
    os.makedirs(run_dir, exist_ok=True)

    report = classification_report(y_true, y_pred, output_dict=True)
    report_file = os.path.join(run_dir, f"{model_name}_classification_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    mlflow.log_artifact(report_file, artifact_path="evaluation")

def log_cross_validation_metrics(pipeline, X, y, model_name,version, env, output_dir="artifacts"):
    # Allow CV folds to be set via environment variable, default = 5
    cv_folds = int(os.getenv("CV_FOLDS", 5))
    scoring = ["accuracy", "precision", "recall", "roc_auc", "f1"]

    scores = cross_validate(pipeline, X, y, cv=cv_folds, scoring=scoring)

    # Log metrics to MLflow
    cv_results = {}
    for metric in scoring:
        mean_score = scores[f"test_{metric}"].mean()
        mlflow.log_metric(f"cv_{metric}", mean_score)
        cv_results[metric] = mean_score
        print(f"CV {model_name} {metric}: {mean_score:.4f}")

    # Save CV metrics as JSON artifact
    run_dir = os.path.join(output_dir, env, f"v{version}")
    os.makedirs(run_dir, exist_ok=True)

    cv_file = os.path.join(run_dir, f"{model_name}_classification_report.json")
    with open(cv_file, "w") as f:
        json.dump(cv_results, f, indent=4)

    mlflow.log_artifact(cv_file, artifact_path="evaluation")

def evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name, version, env):
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    f1 = f1_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_param("model_type", model_name)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    if roc_auc is not None:
        mlflow.log_metric("roc_auc", roc_auc)

    # Log metrics and save artifacts
    log_confusion_matrix(y_test, y_pred, model_name, version, env)
    log_classification_report(y_test, y_pred, model_name, version, env)
    log_cross_validation_metrics(pipeline, X_train, y_train, model_name,version, env)

    # Console output
    print(f"{model_name} Metrics:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC   : {roc_auc:.4f}")

    return acc

def train_logistic_regression(X_train, y_train, X_test, y_test, version,env):
    with mlflow.start_run():
        mlflow.log_params({
            "model": "LogisticRegression",
            "max_iter": 1000,
            "random_state":42
        })

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000,random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        acc = evaluate_model(pipeline, X_train, y_train, X_test, y_test, "LogisticRegression", version, env)

        mlflow.sklearn.log_model(sk_model=pipeline, name="heart_disease_model_lr")
        return pipeline, acc

def train_random_forest(X_train, y_train, X_test, y_test, version,env):
    with mlflow.start_run():
        mlflow.log_params({
            "model": "RandomForest",
            "n_estimators": 100,
            "random_state": 42
        })

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        acc = evaluate_model(pipeline, X_train, y_train, X_test, y_test, "RandomForest", version, env)

        mlflow.sklearn.log_model(sk_model=pipeline, name="heart_disease_model_rf")
        return pipeline, acc

def promote_model_to_production(version):
    src = f"model/development/v{version}"
    dst = f"model/production/v{version}"

    os.makedirs("model/production", exist_ok=True)

    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"✅ Model v{version} promoted to production")

if __name__ == "__main__":
    mlflow.set_experiment("heart-disease-prediction")

    env_file = os.getenv("ENV_FILE", "config/dev.env")
    load_dotenv(env_file)

    if os.getenv("TRAINING_ENABLED") != "true":
        raise RuntimeError("Training disabled in this environment")

    # Load and preprocess data once
    df = load_data("data/heart.csv")
    df = drop_missing(df)
    df = fix_target(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get next version
    env = get_env()
    model_base_path = f"model/{env}"
    version = get_next_model_version(model_base_path)


    lr_model, lr_acc = None, None
    rf_model, rf_acc = None, None

    # Train both models with version-aware artifact logging
    try:
        lr_model, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test, version,env)
    except Exception as e:
        print(f"Logistic Regression Failed: {e}")

    try:
        rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test, version,env)
    except Exception as e:
        print(f"Random Forest Failed: {e}")

    # Compare accuracies safely
    if lr_acc is not None and (rf_acc is None or lr_acc >= rf_acc):
        best_model, best_acc, best_name = lr_model, lr_acc, "LogisticRegression"
    elif rf_acc is not None:
        best_model, best_acc, best_name = rf_model, rf_acc, "RandomForest"
    else:
        raise RuntimeError("Both models failed to train")

    # Save best model
    model_dir = f"{model_base_path}/v{version}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, f"{model_dir}/heart_model.pkl")

    print(f"\nBest Model: {best_name} with Accuracy: {best_acc:.4f}")
    print(f"Model saved at: {model_dir}/heart_model.pkl")

    promote_model_to_production(version)