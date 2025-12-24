import json
import os
import re
from typing import Any, Dict, Optional


def _time_to_seconds(s: str) -> Optional[float]:
    """
    Accepts 'MM:SS' or 'HH:MM:SS' and returns seconds.
    Returns None if cannot parse.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    parts = s.split(":")
    try:
        if len(parts) == 2:  # MM:SS
            m, sec = parts
            return int(m) * 60 + int(sec)
        if len(parts) == 3:  # HH:MM:SS
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(sec)
    except Exception:
        return None

    return None


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Tries hard to parse JSON from a model output.
    """
    text = (text or "").strip()

    # direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to extract first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError("Model did not return valid JSON.")


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def _get_openai_client(openai_api_key: str):
    """
    Returns an OpenAI client; if Langfuse keys are present, uses Langfuse OpenAI wrapper.
    """
    if not openai_api_key:
        raise ValueError("Missing OpenAI API key.")

    # Prefer Langfuse wrapper when configured
    if _langfuse_enabled():
        # Langfuse OpenAI integration: https://langfuse.com/guides/cookbook/integration_openai_sdk
        from langfuse.openai import OpenAI  # type: ignore
        return OpenAI(api_key=openai_api_key)

    # Fallback: plain OpenAI client
    from openai import OpenAI  # type: ignore
    return OpenAI(api_key=openai_api_key)


def extract_runner_profile(
    text: str,
    openai_api_key: str,
    *,
    model: str = None,
) -> Dict[str, Any]:
    """
    Extracts runner profile from free text.
    Returns dict:
      {
        "sex": "M"|"K"|None,
        "age": int|None,
        "t5k": "MM:SS"|None,
        "t5k_s": float|None,
        "missing": [..]
      }
    """
    text = (text or "").strip()
    if not text:
        return {"sex": None, "age": None, "t5k": None, "t5k_s": None, "missing": ["text"]}

    client = _get_openai_client(openai_api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system = (
        "Jesteś parserem danych biegowych. "
        "Masz wyciągnąć z tekstu użytkownika: płeć (M/K), wiek (10–90) i czas na 5 km. "
        "Zwróć WYŁĄCZNIE poprawny JSON zgodny ze schematem."
    )

    user = f"""
Tekst użytkownika:
\"\"\"{text}\"\"\"

Zasady:
- sex: "M" jeśli mężczyzna / facet / male, "K" jeśli kobieta / female; jeśli niepewne -> null
- age: liczba całkowita 10–90; jeśli brak -> null
- t5k: czas 5 km w formacie "MM:SS" (np. 25:00). Jeśli użytkownik podał "25 minut", to ustaw "25:00".
- t5k_s: sekundy jako liczba (float lub int). Jeśli nie potrafisz policzyć, ustaw null.
- missing: lista brakujących pól spośród: ["sex","age","t5k"] (t5k_s nie jest wymagane, bo możemy policzyć z t5k)

Zwróć JSON o dokładnie takich kluczach:
{{
  "sex": "M" | "K" | null,
  "age": number | null,
  "t5k": string | null,
  "t5k_s": number | null,
  "missing": array
}}
""".strip()

    # Langfuse wrapper parses extra attributes (name, metadata) and logs a trace
    # Important: do NOT include the openai_api_key in metadata.
    resp = client.chat.completions.create(
        name="extract_runner_profile",
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        metadata={
            "langfuse_tags": ["halfmarathon-predictor", "extract"],
        },
    )

    content = resp.choices[0].message.content or "{}"
    data = _safe_json_loads(content)

    # Normalize keys + types
    sex = data.get("sex", None)
    if isinstance(sex, str):
        sex = sex.strip().upper()
        if sex not in ("M", "K"):
            sex = None

    age = data.get("age", None)
    try:
        if age is not None and age != "":
            age = int(age)
    except Exception:
        age = None

    t5k = data.get("t5k", None)
    if isinstance(t5k, str):
        t5k = t5k.strip()
        if not t5k:
            t5k = None

    t5k_s = data.get("t5k_s", None)
    try:
        if t5k_s is not None and t5k_s != "":
            t5k_s = float(t5k_s)
        else:
            t5k_s = None
    except Exception:
        t5k_s = None

    missing = data.get("missing", [])
    if not isinstance(missing, list):
        missing = []

    # Post-process: if we have t5k but not t5k_s, compute locally
    if t5k_s is None and t5k:
        computed = _time_to_seconds(t5k)
        if computed is not None:
            t5k_s = float(computed)

    # Recompute missing defensively (don’t trust model 100%)
    missing_clean = []
    if sex is None:
        missing_clean.append("sex")
    if age is None:
        missing_clean.append("age")
    if t5k is None and t5k_s is None:
        missing_clean.append("t5k")

    return {
        "sex": sex,
        "age": age,
        "t5k": t5k,
        "t5k_s": t5k_s,
        "missing": missing_clean,
    }
