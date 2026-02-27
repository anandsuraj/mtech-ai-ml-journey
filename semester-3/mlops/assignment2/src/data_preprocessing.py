import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from src.versioning import get_next_version

logging.basicConfig(level=logging.INFO)

IMG_SIZE = 224
RANDOM_STATE = 42

def load_images(folder, label, max_images=500):
    X, y = [], []

    files = os.listdir(folder)[:max_images]   # ⭐ LIMIT DATASET

    for file in files:
        if not file.lower().endswith((".jpg", ".png")):
            continue

        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label)

    print(f"Loaded {len(X)} images from {folder}")
    return np.array(X), np.array(y)


def preprocess(env="dev", raw_dir="data/raw"):
    base_output = f"data/processed/{env}"
    version = get_next_version(base_output)
    output_dir = f"{base_output}/{version}"
    os.makedirs(output_dir, exist_ok=True)

    X_cats, y_cats = load_images(os.path.join(raw_dir, "cats"), 0)
    X_dogs, y_dogs = load_images(os.path.join(raw_dir, "dogs"), 1)

    X = np.concatenate([X_cats, X_dogs])
    y = np.concatenate([y_cats, y_dogs])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    np.savez(
        f"{output_dir}/dataset.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )

    logging.info(f"✅ Data prepared for {env}/{version}")
    return version


if __name__ == "__main__":
    preprocess(env="dev")
