import joblib
import pandas as pd

# Load trained model
model = joblib.load("../model/development/v1/heart_model.pkl")

# Define multiple samples
samples = pd.DataFrame([
    {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0, "thal": 6
    },
    {
        "age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 240, "fbs": 0,
        "restecg": 1, "thalach": 160, "exang": 1, "oldpeak": 1.0,
        "slope": 2, "ca": 1, "thal": 7
    }
])

# Get predictions and probabilities
preds = model.predict(samples)
probas = model.predict_proba(samples)

# Iterate through each sample
for i, (pred, proba) in enumerate(zip(preds, probas), start=1):
    confidence = max(proba)
    disease_prob = proba[1]

    result = {
        "sample_id": i,
        "predicted_label": int(pred),
        "predicted_class": "Heart Disease" if pred == 1 else "No Heart Disease",
        "probability_no_disease": round(float(proba[0]), 4),
        "probability_heart_disease": round(float(proba[1]), 4),
        "model_confidence": round(float(confidence), 4)
    }

    print(f"\n--- Sample {i} ---")
    for k, v in result.items():
        print(f"{k:30s}: {v}")

    if result["probability_heart_disease"] >= 0.5:
        print("Clinical interpretation: High likelihood of heart disease")
    else:
        print("Clinical interpretation: Low likelihood of heart disease")