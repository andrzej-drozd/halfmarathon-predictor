import os
import json
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import load_all_races
from src.features import build_features


MODEL_NAME = "halfmarathon_linear.joblib"
MODEL_PATH = os.path.join("models", MODEL_NAME)
META_PATH = os.path.join("models", "halfmarathon_linear.metadata.json")

FEATURES = ["t5k_s", "age", "sex_M"]


def train_model():
    # 1) Load raw race data
    races = load_all_races((2023, 2024))

    # 2) Build features per year and combine
    feats = []
    for year, df in races.items():
        f = build_features(df, race_year=year)
        f["race_year"] = year
        feats.append(f)

    data = pd.concat(feats, ignore_index=True)

    # 3) Build training matrix WITH FEATURE NAMES (DataFrame)
    X = pd.DataFrame({
        "t5k_s": data["t5k_s"].astype(float),
        "age": data["age"].astype(int),
        "sex_M": (data["sex"] == "M").astype(int),
    })[FEATURES]

    y = data["t21k_s"].astype(float)

    # 4) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = model.predict(X_val)

    mae_sec = mean_absolute_error(y_val, y_pred)
    rmse_sec = mean_squared_error(y_val, y_pred) ** 0.5

    print("=== Validation metrics ===")
    print(f"MAE:  {mae_sec/60:.2f} min")
    print(f"RMSE: {rmse_sec/60:.2f} min")

    # 7) Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # 8) Save metadata (optional but useful)
    meta = {
        "model_name": MODEL_NAME,
        "features": FEATURES,
        "n_rows": int(len(data)),
        "mae_sec": float(mae_sec),
        "rmse_sec": float(rmse_sec),
        "mae_min": float(mae_sec / 60),
        "rmse_min": float(rmse_sec / 60),
        "coef": {k: float(v) for k, v in zip(FEATURES, model.coef_)},
        "intercept": float(model.intercept_),
        "years": sorted([int(y) for y in races.keys()]),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Metadata saved to: {META_PATH}")

    return model


if __name__ == "__main__":
    train_model()
