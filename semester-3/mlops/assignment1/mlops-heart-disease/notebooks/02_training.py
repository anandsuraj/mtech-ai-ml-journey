import sys
import os
import mlflow
import mlflow.sklearn
import joblib

# Add project root to path
sys.path.append(os.path.abspath(".."))

# Import training functions
from src.train import train_logistic_regression, train_random_forest, train_test_split
from src.preprocess import (
    load_data,
    drop_missing,
    fix_target,
    split_features_target,
    build_preprocessor
)
# Example: assume you already have train/test splits and version defined
# Load and preprocess data once
df = load_data("../data/heart.csv")
df = drop_missing(df)
df = fix_target(df)
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train both models with version-aware artifact logging
lr_model, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test, 1)
rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test, 1)

# Compare accuracies
if lr_acc >= rf_acc:
    best_model, best_acc, best_name = lr_model, lr_acc, "LogisticRegression"
else:
    best_model, best_acc, best_name = rf_model, rf_acc, "RandomForest"

# Print metrics
print("Logistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)
print(f"Best Model: {best_name} with Accuracy: {best_acc}")