import os
import sys
import yaml
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from src.versioning import get_latest_version

# =========================
# Promotion Threshold
# =========================
PROMOTION_ACCURACY_THRESHOLD = 0.65


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_test_data(env: str, version: str):
    path = f"data/processed/{env}/{version}/dataset.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path)
    return data["X_test"], data["y_test"]


def load_model(env: str, version: str):
    path = f"models/{env}/{version}/model.h5"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    return tf.keras.models.load_model(path)


def save_confusion_matrix(y_true, y_pred, env, version):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs("artifacts", exist_ok=True)
    path = f"artifacts/confusion_matrix_{env}_{version}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def save_roc_curve(y_true, probs, env, version):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    os.makedirs("artifacts", exist_ok=True)
    path = f"artifacts/roc_curve_{env}_{version}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path, roc_auc


def evaluate(config_path: str):
    cfg = load_config(config_path)
    env = cfg["env"]

    # 🔑 Auto-detect latest trained model
    version = get_latest_version(f"models/{env}")

    mlflow.set_experiment(cfg["mlflow_experiment"])

    X_test, y_test = load_test_data(env, version)
    model = load_model(env, version)

    with mlflow.start_run(run_name=f"{env}_{version}_evaluation"):
        mlflow.set_tags({
            "env": env,
            "version": version,
            "stage": "evaluation"
        })

        # =========================
        # Predictions
        # =========================
        probs = model.predict(X_test).flatten()
        preds = (probs > 0.5).astype(int)

        # =========================
        # Metrics
        # =========================
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1
        })

        # =========================
        # Visual Artifacts
        # =========================
        cm_path = save_confusion_matrix(y_test, preds, env, version)
        roc_path, roc_auc = save_roc_curve(y_test, probs, env, version)

        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
        mlflow.log_artifact(roc_path, artifact_path="evaluation")

        # =========================
        # Classification Report
        # =========================
        report = classification_report(
            y_test, preds, target_names=["Cat", "Dog"]
        )

        report_path = f"artifacts/classification_report_{env}_{version}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path, artifact_path="evaluation")

        # =========================
        # Promotion Decision
        # =========================
        promote = acc >= PROMOTION_ACCURACY_THRESHOLD
        mlflow.log_param("promotion_candidate", promote)

        print("\n==============================")
        print("✅ Evaluation Completed")
        print(f"Environment : {env}")
        print(f"Version     : {version}")
        print(f"Accuracy    : {acc:.4f}")
        print(f"ROC-AUC     : {roc_auc:.4f}")
        print(f"Promote     : {promote}")
        print("==============================\n")

        print(report)

        # 🔴 Fail CI/CD if model is not good enough
        if not promote:
            print("❌ Model did NOT meet promotion threshold")
            sys.exit(1)


if __name__ == "__main__":
    evaluate("configs/dev.yaml")
