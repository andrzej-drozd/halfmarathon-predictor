import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv


def load_env() -> None:
    """
    Loads .env variables. Uses override=True to avoid stale empty vars from the environment.
    """
    env_path = find_dotenv()
    if not env_path:
        raise FileNotFoundError("Could not find .env (find_dotenv returned empty path).")
    load_dotenv(env_path, override=True)


def _storage_options():
    key = os.getenv("DO_SPACES_KEY")
    secret = os.getenv("DO_SPACES_SECRET")
    region = os.getenv("DO_SPACES_REGION")

    if not key or not secret or not region:
        raise ValueError("Missing DO_SPACES_KEY / DO_SPACES_SECRET / DO_SPACES_REGION in environment.")

    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": f"https://{region}.digitaloceanspaces.com"},
    }


def load_race_csv(year: int) -> pd.DataFrame:
    """
    Loads a single year's CSV from DigitalOcean Spaces.
    Expected path: s3://{bucket}/{prefix}/{year}.csv
    """
    load_env()

    bucket = os.getenv("DO_SPACES_BUCKET")
    prefix = os.getenv("DO_SPACES_PREFIX")

    if not bucket or not prefix:
        raise ValueError("Missing DO_SPACES_BUCKET / DO_SPACES_PREFIX in environment.")

    # normalize prefix (no trailing slash)
    prefix = prefix.rstrip("/")

    path = f"s3://{bucket}/{prefix}/{year}.csv"
    df = pd.read_csv(path, sep=";", storage_options=_storage_options())
    return df


def load_all_races(years=(2023, 2024)) -> dict[int, pd.DataFrame]:
    """
    Loads multiple years into a dict: {year: dataframe}.
    """
    return {year: load_race_csv(year) for year in years}
