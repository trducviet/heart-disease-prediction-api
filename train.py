"""Train and persist the preprocessing artifacts and Logistic Regression model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
CATEGORICAL_COLUMNS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def train(data_path: Path) -> None:
    data = pd.read_csv(data_path)
    missing_columns = set(FEATURE_COLUMNS + ["HeartDisease"]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    label_encoders: dict[str, LabelEncoder] = {}
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        label_encoders[column] = encoder

    features = data[FEATURE_COLUMNS]
    labels = data["HeartDisease"]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    selector = SelectKBest(score_func=chi2, k=5)
    selected_features = selector.fit_transform(scaled_features, labels)

    x_train, x_test, y_train, y_test = train_test_split(
        selected_features, labels, test_size=0.2, random_state=42
    )
    model = LogisticRegression(C=10, penalty="l2", solver="lbfgs")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    print(f"Selected features: {list(features.columns[selector.get_support()])}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "logistic_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(selector, MODEL_DIR / "selector.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    arguments = parser.parse_args()
    train(arguments.data)
