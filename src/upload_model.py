import os
from dotenv import load_dotenv, find_dotenv
import boto3


MODEL_LOCAL_PATH = os.path.join("models", "halfmarathon_linear.joblib")
MODEL_SPACES_KEY = "models/halfmarathon_linear.joblib"


def load_env():
    env_path = find_dotenv()
    if not env_path:
        raise FileNotFoundError("Could not find .env")
    load_dotenv(env_path, override=True)


def upload_file(local_path: str, bucket: str, object_key: str, region: str):
    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=os.getenv("DO_SPACES_KEY"),
        aws_secret_access_key=os.getenv("DO_SPACES_SECRET"),
    )

    client.upload_file(local_path, bucket, object_key)
    print(f"Uploaded: {local_path} -> s3://{bucket}/{object_key}")


if __name__ == "__main__":
    load_env()

    bucket = os.getenv("DO_SPACES_BUCKET")
    region = os.getenv("DO_SPACES_REGION")

    if not bucket or not region:
        raise ValueError("Missing DO_SPACES_BUCKET / DO_SPACES_REGION in .env")

    if not os.path.exists(MODEL_LOCAL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_LOCAL_PATH}")

    upload_file(MODEL_LOCAL_PATH, bucket, MODEL_SPACES_KEY, region)
