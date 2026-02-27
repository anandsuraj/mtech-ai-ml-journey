import csv
import os
from datetime import datetime
import pandas as pd

LOG_FILE = "logs/predictions.csv"


# ================================
# LOG PREDICTIONS
# ================================
def log_prediction(y_true, y_pred, confidence):
    """
    Logs every prediction.
    true_label can be None/unknown in production.
    """
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "true_label",
                "prediction",
                "confidence"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            y_true if y_true else "unknown",
            y_pred,
            float(confidence)
        ])


# ================================
# PERFORMANCE SUMMARY
# ================================
def get_performance_summary():
    """
    Returns monitoring statistics.

    IMPORTANT:
    - total_predictions counts ALL requests
    - accuracy uses only rows with true labels
    """

    if not os.path.exists(LOG_FILE):
        return {"message": "No prediction data available yet"}

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        return {"message": "No predictions logged"}

    # ----------------------------
    # TOTAL PREDICTIONS
    # ----------------------------
    total_predictions = len(df)

    # ----------------------------
    # ONLY EVALUATED DATA
    # (true labels available)
    # ----------------------------
    evaluated_df = df[
        (df["true_label"].notna()) &
        (df["true_label"] != "unknown")
    ]

    evaluated_predictions = len(evaluated_df)

    response = {
        "total_predictions": int(total_predictions),
        #"evaluated_predictions": int(evaluated_predictions),
        "avg_confidence": round(
            float(df["confidence"].mean()), 4
        ),
        "class_distribution": (
            df["prediction"]
            .value_counts()
            .to_dict()
        )
    }

    # ----------------------------
    # ACCURACY (only if labels exist)
    # ----------------------------
    if evaluated_predictions > 0:
        correct = (
            evaluated_df["true_label"]
            == evaluated_df["prediction"]
        ).sum()

        accuracy = correct / evaluated_predictions

        response["correct_predictions"] = int(correct)
        response["accuracy"] = round(float(accuracy), 4)

    return response