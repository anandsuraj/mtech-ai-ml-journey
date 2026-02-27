import io
import numpy as np
import tensorflow as tf
import time
import logging
import os
import cv2
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

from src.versioning import get_latest_version
from src.performance_tracking import (
    get_performance_summary
)

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    REGISTRY
)

# ==========================
# LOGGING
# ==========================
logging.basicConfig(level=logging.INFO)

# ==========================
# SAFE METRICS (pytest-safe)
# ==========================
def get_or_create_counter():
    try:
        return Counter(
            "prediction_requests_total",
            "Total number of prediction requests"
        )
    except ValueError:
        return REGISTRY._names_to_collectors["prediction_requests_total"]


def get_or_create_histogram():
    try:
        return Histogram(
            "prediction_latency_seconds",
            "Latency of prediction requests"
        )
    except ValueError:
        return REGISTRY._names_to_collectors["prediction_latency_seconds"]


REQUEST_COUNTER = get_or_create_counter()
PREDICTION_LATENCY = get_or_create_histogram()

REQUEST_COUNT = 0

# ==========================
# APP
# ==========================
app = FastAPI(title="Cats vs Dogs Inference API")

# ==========================
# LOGGING PREDICTIONS
# ==========================
def log_prediction(true_label, prediction, confidence):

    os.makedirs("logs", exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "true_label": true_label if true_label else "unknown",
        "prediction": prediction,
        "confidence": float(confidence)
    }

    df = pd.DataFrame([row])

    file_path = "logs/predictions.csv"

    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


# ==========================
# MODEL LOADING
# ==========================
BASE_PATH = "models/prod"
model = None
prod_version = None


def load_model_if_available():
    """
    Lazy-load model safely.
    Prevents crashes during pytest / CI.
    """
    global model, prod_version

    if model is not None:
        return model

    if not os.path.exists(BASE_PATH):
        logging.warning("models/prod does not exist")
        return None

    if not os.listdir(BASE_PATH):
        logging.warning("models/prod is empty")
        return None

    prod_version = get_latest_version(BASE_PATH)

    if prod_version is None:
        logging.warning("No model version found")
        return None

    model_path = f"{BASE_PATH}/{prod_version}/model.h5"

    if not os.path.exists(model_path):
        logging.warning(f"Model file not found: {model_path}")
        return None

    logging.info(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


# ==========================
# PREPROCESS (MATCH TRAINING)
# ==========================
def preprocess(image_bytes):
    """
    Match training preprocessing:
    - Decode with OpenCV (BGR)
    - Resize to 224x224
    - Normalize to float32 [0,1]
    - Add batch dimension
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)



# ==========================
# ROUTES
# ==========================
@app.get("/health")
def health():
    load_model_if_available()

    return {
        "status": "healthy",
        "model_version": prod_version
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    true_label: str = None
):
    global REQUEST_COUNT

    loaded_model = load_model_if_available()

    if loaded_model is None:
        return {"error": "Model not available"}

    image_bytes = await file.read()

    start = time.time()

    with PREDICTION_LATENCY.time():
        image = preprocess(image_bytes)
        prob = loaded_model.predict(image)[0][0]

    latency = time.time() - start
    REQUEST_COUNT += 1

    # ==========================
    # LABEL MAPPING (CORRECT)
    # cats = 0, dogs = 1
    # ==========================
    label = "dog" if prob > 0.5 else "cat"
    #label = "cat" if prob > 0.5 else "dog"

    # log prediction always
    log_prediction(
        true_label if true_label is not None else "unknown",
        label,
        float(prob)
    )

    logging.info(
        f"Prediction | label={label} | prob={prob:.3f} "
        f"| latency={latency:.3f}s | requests={REQUEST_COUNT}"
    )

    REQUEST_COUNTER.inc()

    return {
        "label": label,
        "confidence": float(prob),
        "model_version": prod_version,
        "latency": latency,
        "request_count": REQUEST_COUNT
    }


@app.get("/performance")
def performance():
    return get_performance_summary()


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain; version=0.0.4"
    )