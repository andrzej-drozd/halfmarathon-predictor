import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

# Langfuse (opcjonalnie) — jeśli paczka jest zainstalowana i env są ustawione, to zadziała
try:
    from langfuse import Langfuse
except Exception:
    Langfuse = None


SYSTEM_PROMPT = """
Jesteś parserem danych biegacza. Twoim zadaniem jest WYŁĄCZNIE ekstrakcja danych
z krótkiego tekstu po polsku.

Zwróć JSON w formacie:
{
  "sex": "M" | "K" | null,
  "age": number | null,
  "t5k": "MM:SS" | "HH:MM:SS" | null,
  "t5k_s": number | null
}

Zasady:
- Płeć: "M" lub "K" (mężczyzna/chłopak/pan => M, kobieta/pani => K).
- Wiek: liczba całkowita (np. "mam 40 lat" => 40).
- 5 km czas: jeśli jest podany jako "25:00" lub "0:25:00" etc => t5k.
- Jeśli czas podany jako "25 minut" albo "1500 sekund" => możesz wypełnić t5k_s.
- Jeśli czegoś nie ma w tekście, ustaw null.
- Zwróć TYLKO czysty JSON. Bez komentarzy i bez markdown.
""".strip()


def _seconds_to_mmss(seconds: int) -> str:
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def _normalize_extracted(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Upewniamy się, że pola są zawsze obecne
    sex = obj.get("sex", None)
    age = obj.get("age", None)
    t5k = obj.get("t5k", None)
    t5k_s = obj.get("t5k_s", None)

    # Normalizacja sex
    if isinstance(sex, str):
        sex = sex.strip().upper()
        if sex not in ("M", "K"):
            sex = None
    else:
        sex = None

    # Normalizacja age
    try:
        if age is not None and age != "":
            age = int(age)
        else:
            age = None
    except Exception:
        age = None

    # Normalizacja t5k_s
    try:
        if t5k_s is not None and t5k_s != "":
            t5k_s = float(t5k_s)
        else:
            t5k_s = None
    except Exception:
        t5k_s = None

    # Jeśli t5k_s jest, a t5k nie ma — wypełnij t5k w MM:SS
    if t5k is None and t5k_s is not None:
        try:
            t5k = _seconds_to_mmss(int(round(t5k_s)))
        except Exception:
            t5k = None

    # Missing list (jak u Ciebie w UI)
    missing = []
    if not sex:
        missing.append("sex")
    if age is None:
        missing.append("age")
    # “czas” traktujemy jako t5k lub t5k_s
    if (t5k is None or str(t5k).strip() == "") and t5k_s is None:
        missing.append("t5k")

    return {
        "sex": sex,
        "age": age,
        "t5k": t5k,
        "t5k_s": t5k_s,
        "missing": missing,
    }


def extract_runner_profile(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract runner profile from free-form Polish text.

    IMPORTANT:
    - api_key is provided by user (session-only) from UI.
    - If api_key is None, tries OPENAI_API_KEY from env (useful locally).
    Returns dict:
      {"sex": ..., "age": ..., "t5k": ..., "t5k_s": ..., "missing": [...]}
    """
    text = (text or "").strip()
    if not text:
        return {"sex": None, "age": None, "t5k": None, "t5k_s": None, "missing": ["sex", "age", "t5k"]}

    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        # brak klucza => zwracamy brakujące, ale bez crasha
        return {"sex": None, "age": None, "t5k": None, "t5k_s": None, "missing": ["sex", "age", "t5k"]}

    # Langfuse (opcjonalnie)
    langfuse = None
    if Langfuse is not None:
        pub = os.getenv("LANGFUSE_PUBLIC_KEY")
        sec = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if pub and sec:
            try:
                langfuse = Langfuse(public_key=pub, secret_key=sec, host=host)
            except Exception:
                langfuse = None

    client = OpenAI(api_key=key)

    trace = None
    span = None
    if langfuse:
        try:
            trace = langfuse.trace(name="extract_runner_profile")
            span = trace.span(
                name="openai.chat.completions.create",
                input={"text": text},
                metadata={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")},
            )
        except Exception:
            trace = None
            span = None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        content = resp.choices[0].message.content or ""
        content = content.strip()

        # model ma zwrócić JSON; próbujemy zdekodować
        parsed = json.loads(content)

        out = _normalize_extracted(parsed)

        if span:
            try:
                span.end(output=out)
            except Exception:
                pass
        if trace:
            try:
                trace.end()
            except Exception:
                pass

        return out

    except Exception as e:
        # fail-safe: nie wywalaj całej aplikacji
        out = {"sex": None, "age": None, "t5k": None, "t5k_s": None, "missing": ["sex", "age", "t5k"]}
        if span:
            try:
                span.end(output={"error": str(e)})
            except Exception:
                pass
        if trace:
            try:
                trace.end()
            except Exception:
                pass
        return out
