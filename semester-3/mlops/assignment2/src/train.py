import os
import yaml
import mlflow
import mlflow.keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from src.versioning import get_next_version

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(env, version):
    data = np.load(f"data/processed/{env}/{version}/dataset.npz")
    return data["X_train"], data["y_train"], data["X_val"], data["y_val"]

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():

    # ===============================
    # PRETRAINED BACKBONE
    # ===============================
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # ===============================
    # CLASSIFICATION HEAD
    # ===============================
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train(config_path):
    cfg = load_config(config_path)
    env = cfg["env"]

    data_version = sorted(os.listdir(f"data/processed/{env}"))[-1]
    model_base = f"models/{env}"
    model_version = get_next_version(model_base)

    mlflow.set_experiment(cfg["mlflow_experiment"])

    X_train, y_train, X_val, y_val = load_data(env, data_version)

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    with mlflow.start_run(run_name=f"{env}_{model_version}"):
        mlflow.log_params(cfg)
        mlflow.set_tags({"env": env, "version": model_version})

        model = build_model()
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=cfg["batch_size"]),
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
        )

        os.makedirs(f"{model_base}/{model_version}", exist_ok=True)
        model.save(f"{model_base}/{model_version}/model.h5")

        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
        mlflow.keras.log_model(model, "model")

        print(f"✅ Model trained: {env}/{model_version}")
        return model_version

if __name__ == "__main__":
    train("configs/dev.yaml")
