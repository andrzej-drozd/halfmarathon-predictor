import numpy as np
import pandas as pd


def time_to_seconds(x):
    """
    Converts time strings 'MM:SS' or 'HH:MM:SS' to seconds.
    Returns np.nan if parsing fails.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if not s:
        return np.nan

    parts = s.split(":")
    try:
        if len(parts) == 2:  # MM:SS
            m, sec = parts
            return int(m) * 60 + int(sec)
        elif len(parts) == 3:  # HH:MM:SS
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(sec)
    except ValueError:
        return np.nan

    return np.nan


def build_features(df: pd.DataFrame, race_year: int) -> pd.DataFrame:
    """
    Feature selection + feature engineering.

    Input:
        df         - raw dataframe loaded from CSV
        race_year - year of the race (used to compute age)

    Output:
        DataFrame with columns:
            sex (str: 'M' or 'K')
            age (int)
            t5k_s (float)
            t21k_s (float)
    """

    out = df[["Płeć", "Rocznik", "5 km Czas", "Czas"]].copy()

    # age
    out["birth_year"] = pd.to_numeric(out["Rocznik"], errors="coerce")
    out["age"] = race_year - out["birth_year"]

    # times
    out["t5k_s"] = out["5 km Czas"].map(time_to_seconds)
    out["t21k_s"] = out["Czas"].map(time_to_seconds)

    # sex
    out["sex"] = out["Płeć"].astype(str).str.strip().str.upper()
    out.loc[~out["sex"].isin(["M", "K"]), "sex"] = np.nan

    # drop incomplete
    out = out.dropna(subset=["sex", "age", "t5k_s", "t21k_s"])

    # sanity filters
    out = out[(out["age"] >= 10) & (out["age"] <= 90)]
    out = out[(out["t5k_s"] >= 12 * 60) & (out["t5k_s"] <= 60 * 60)]
    out = out[(out["t21k_s"] >= 60 * 60) & (out["t21k_s"] <= 5 * 60 * 60)]

    return out[["sex", "age", "t5k_s", "t21k_s"]]

