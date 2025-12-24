import os
import joblib
from dotenv import load_dotenv, find_dotenv
import boto3
import pandas as pd


MODEL_NAME = "halfmarathon_linear.joblib"
LOCAL_MODEL_PATH = os.path.join("models", MODEL_NAME)
SPACES_MODEL_KEY = f"models/{MODEL_NAME}"


def load_env():
    env_path = find_dotenv()
    if not env_path:
        raise FileNotFoundError("Could not find .env")
    load_dotenv(env_path, override=True)


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('DO_SPACES_REGION')}.digitaloceanspaces.com",
        aws_access_key_id=os.getenv("DO_SPACES_KEY"),
        aws_secret_access_key=os.getenv("DO_SPACES_SECRET"),
    )


def download_model_from_spaces():
    """
    Downloads model from DigitalOcean Spaces if not available locally.
    """
    load_env()

    bucket = os.getenv("DO_SPACES_BUCKET")
    if not bucket:
        raise ValueError("Missing DO_SPACES_BUCKET")

    os.makedirs("models", exist_ok=True)

    client = _s3_client()
    client.download_file(bucket, SPACES_MODEL_KEY, LOCAL_MODEL_PATH)
    print(f"Model downloaded from Spaces to {LOCAL_MODEL_PATH}")


def load_model():
    """
    Loads model from local disk; downloads from Spaces if needed.
    """
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model_from_spaces()

    return joblib.load(LOCAL_MODEL_PATH)


def predict_halfmarathon_time(*, t5k_s: float, age: int, sex: str) -> float:
    model = load_model()

    sex_M = 1 if str(sex).upper() == "M" else 0

    X = pd.DataFrame(
        [[t5k_s, age, sex_M]],
        columns=["t5k_s", "age", "sex_M"]
    )

    y_pred = model.predict(X)
    return float(y_pred[0])

