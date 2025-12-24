import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI


# ---------------------------
# Small helpers (parsing/normalization)
# ---------------------------

_TIME_MMSS_RE = re.compile(r"^\s*(\d{1,2})\s*:\s*([0-5]\d)\s*$")


def _mmss_to_seconds(mmss: str) -> Optional[float]:
    m = _TIME_MMSS_RE.match(str(mmss))
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    return float(mm * 60 + ss)


def _normalize_sex(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().upper()

    # common Polish words
    if s in ("M", "MALE", "MAN", "MĘŻCZYZNA", "MEZCZYZNA", "MĘŻCZYZNA.", "MEZCZYZNA."):
        return "M"
    if s in ("K", "F", "FEMALE", "WOMAN", "KOBIETA", "KOBIETA."):
        return "K"

    # sometimes model returns "mężczyzna"/"kobieta"
    if "MEZCZYZ" in s or "MĘŻCZYZ" in s:
        return "M"
    if "KOBIET" in s:
        return "K"

    # fallback for exact
    if s in ("M", "K"):
        return s
    return None


def _normalize_age(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        a = int(float(val))
        return a
    except Exception:
        return None


def _normalize_t5k(val: Any) -> Optional[str]:
    """
    Try to normalize to MM:SS string if possible.
    Accepts:
      - "25:00"
      - "25 min", "25 minut", "25 minutes" (assume 00 seconds)
      - numeric seconds (handled elsewhere)
    """
    if val is None or val == "":
        return None

    s = str(val).strip()

    # direct MM:SS
    if _TIME_MMSS_RE.match(s):
        mm, ss = s.split(":")
        return f"{int(mm):02d}:{int(ss):02d}"

    # "25 min", "25 minut", "25 minutes"
    m = re.search(r"(\d{1,2})\s*(min|mins|minute|minutes|minut|minuty|minuta)\b", s, flags=re.I)
    if m:
        mm = int(m.group(1))
        return f"{mm:02d}:00"

    # "25m 10s"
    m2 = re.search(r"(\d{1,2})\s*m(?:in)?\s*(\d{1,2})\s*s", s, flags=re.I)
    if m2:
        mm = int(m2.group(1))
        ss = int(m2.group(2))
        ss = max(0, min(59, ss))
        return f"{mm:02d}:{ss:02d}"

    return None


def _compute_missing(sex: Optional[str], age: Optional[int], t5k_s: Optional[float], t5k: Optional[str]) -> list[str]:
    missing = []
    if sex not in ("M", "K"):
        missing.append("sex")
    if age is None:
        missing.append("age")
    # one of them should be present; prefer seconds
    if t5k_s is None and (t5k is None or t5k.strip() == ""):
        missing.append("t5k")
    return missing


# ---------------------------
# Main function
# ---------------------------

def extract_runner_profile(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract {sex, age, t5k (MM:SS), t5k_s (seconds)} from free-form Polish text.

    IMPORTANT:
    - api_key is used only for this call; never stored.
    - returns dict with keys: sex, age, t5k, t5k_s, missing
    """
    text = str(text or "").strip()

    # Default empty response
    out: Dict[str, Any] = {
        "sex": None,
        "age": None,
        "t5k": None,
        "t5k_s": None,
        "missing": ["sex", "age", "t5k"],
    }
    if not text:
        return out

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    system = (
        "Jesteś parserem danych wejściowych dla aplikacji biegowej. "
        "Wyciągnij z tekstu użytkownika: płeć (M/K), wiek (liczba całkowita), czas na 5 km. "
        "Zwróć WYŁĄCZNIE JSON w jednym obiekcie, bez komentarzy.\n\n"
        "Wymagany format:\n"
        "{\n"
        '  "sex": "M" | "K" | null,\n'
        '  "age": 40 | null,\n'
        '  "t5k": "MM:SS" | null,\n'
        '  "t5k_s": 1500 | null\n'
        "}\n\n"
        "Zasady:\n"
        "- Jeśli czas jest podany jako '25 minut' to t5k = '25:00'.\n"
        "- Jeśli podano czas jako liczba sekund, ustaw t5k_s.\n"
        "- Jeśli czegoś nie ma, ustaw null.\n"
    )

    user = f"Tekst użytkownika:\n{text}"

    # Ask model to produce JSON only
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = resp.choices[0].message.content or ""

    # Try to locate JSON inside (just in case)
    raw = raw.strip()
    json_str = raw
    if not raw.startswith("{"):
        # naive extraction of first JSON object
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            json_str = m.group(0)

    try:
        data = json.loads(json_str)
    except Exception:
        # fallback: return minimal + missing
        return out

    # Normalize fields
    sex = _normalize_sex(data.get("sex"))
    age = _normalize_age(data.get("age"))

    t5k = _normalize_t5k(data.get("t5k"))
    t5k_s = None

    # If numeric seconds provided, accept it
    try:
        if data.get("t5k_s") not in (None, ""):
            t5k_s = float(data.get("t5k_s"))
    except Exception:
        t5k_s = None

    # If no seconds but we have MM:SS, compute seconds
    if t5k_s is None and t5k:
        t5k_s = _mmss_to_seconds(t5k)

    missing = _compute_missing(sex, age, t5k_s, t5k)

    return {
        "sex": sex,
        "age": age,
        "t5k": t5k,
        "t5k_s": t5k_s,
        "missing": missing,
    }
