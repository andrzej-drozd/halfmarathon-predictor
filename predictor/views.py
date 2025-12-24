import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from src.model import predict_halfmarathon_time
from predictor.llm import extract_runner_profile


# ---------------------------
# helpers
# ---------------------------

def _format_hhmmss(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _format_mmss(seconds: float) -> str:
    s = int(round(seconds))
    mm = s // 60
    ss = s % 60
    return f"{mm:02d}:{ss:02d}"


def _time_to_seconds(s: str) -> float:
    """
    Accepts 'MM:SS' or 'HH:MM:SS' and returns seconds.
    """
    s = str(s).strip()
    parts = s.split(":")
    if len(parts) == 2:  # MM:SS
        m, sec = parts
        return int(m) * 60 + int(sec)
    if len(parts) == 3:  # HH:MM:SS
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + int(sec)
    raise ValueError("Invalid time format. Use MM:SS or HH:MM:SS")


def _postprocess_extracted(extracted: dict) -> dict:
    """
    If LLM returned t5k_s but not t5k string, fill t5k as MM:SS for UI consistency.
    """
    try:
        if extracted.get("t5k") in (None, "") and extracted.get("t5k_s") not in (None, ""):
            extracted["t5k"] = _format_mmss(float(extracted["t5k_s"]))
    except Exception:
        pass
    return extracted


def parse_input(payload: dict):
    """
    Wspólna walidacja dla API i formularza.
    Przyjmuje dict i zwraca (sex, age, t5k_s).
    Rzuca ValueError z czytelnym komunikatem.
    """
    if "sex" not in payload or payload.get("sex") in (None, ""):
        raise ValueError("Brakuje pola: sex (M/K).")
    if "age" not in payload or payload.get("age") in (None, ""):
        raise ValueError("Brakuje pola: age (wiek).")

    # time: either t5k_s or t5k
    if ("t5k_s" not in payload or payload.get("t5k_s") in (None, "")) and (
        "t5k" not in payload or payload.get("t5k") in (None, "")
    ):
        raise ValueError("Brakuje czasu 5 km: podaj t5k_s (sekundy) albo t5k w formacie 'MM:SS' (np. 25:00).")

    sex = str(payload["sex"]).strip().upper()
    if sex not in ("M", "K"):
        raise ValueError("sex musi być 'M' albo 'K'.")

    try:
        age = int(payload["age"])
    except Exception:
        raise ValueError("age musi być liczbą całkowitą.")

    # parse time
    try:
        if "t5k_s" in payload and payload.get("t5k_s") not in (None, ""):
            t5k_s = float(payload["t5k_s"])
        else:
            t5k_s = float(_time_to_seconds(payload["t5k"]))
    except Exception:
        raise ValueError("Czas 5 km jest niepoprawny. Użyj t5k_s (sekundy) albo t5k np. '25:00'.")

    # ranges (te same co w pipeline)
    if not (10 <= age <= 90):
        raise ValueError("Wiek musi być w zakresie 10–90.")
    if not (12 * 60 <= t5k_s <= 60 * 60):
        raise ValueError("Czas 5 km musi być w zakresie 12:00–60:00.")

    return sex, age, t5k_s


def _ctx_from_session(request):
    """
    Build UI context from session.
    """
    return {
        "form_data": request.session.get("form_data", {"sex": "M", "age": "40", "t5k": "25:00"}),
        "text_data": request.session.get("text_data", {"text": ""}),
        "result": request.session.get("result"),
        "text_result": request.session.get("text_result"),
        "error_left": request.session.get("error_left"),
        "error_right": request.session.get("error_right"),
    }


# ---------------------------
# UI views (HTML)
# ---------------------------

@require_GET
def home(request):
    # DEV MODE: zawsze czyść sesję przy wejściu na /
    for k in [
        "form_data",
        "text_data",
        "result",
        "text_result",
        "error_left",
        "error_right",
    ]:
        request.session.pop(k, None)

    ctx = _ctx_from_session(request)
    return render(request, "predictor/home.html", ctx)



@require_POST
def predict_form(request):
    """
    Left side (classic form). Persist inputs + result in session.
    Does NOT touch right-side stored values/results.
    """
    sex_in = request.POST.get("sex", "")
    age_in = request.POST.get("age", "")
    t5k_in = request.POST.get("t5k", "")

    # always persist what user typed
    request.session["form_data"] = {"sex": sex_in, "age": age_in, "t5k": t5k_in}

    try:
        sex, age, t5k_s = parse_input({"sex": sex_in, "age": age_in, "t5k": t5k_in})
        t21k_s = predict_halfmarathon_time(t5k_s=t5k_s, age=age, sex=sex)

        request.session["result"] = {"t21k_s": t21k_s, "t21k_hhmmss": _format_hhmmss(t21k_s)}
        request.session.pop("error_left", None)

    except Exception as e:
        # key: show error in left "Wynik" and clear stale left result
        request.session["error_left"] = str(e)
        request.session["result"] = None

    ctx = _ctx_from_session(request)
    return render(request, "predictor/home.html", ctx)


@require_POST
def predict_text_form(request):
    """
    Right side (LLM text). Persist textarea + text_result in session.
    Does NOT touch left-side stored values/results.
    """
    text = str(request.POST.get("text", "")).strip()
    request.session["text_data"] = {"text": text}

    # clear stale right-side error/result for this attempt; left stays intact
    request.session.pop("error_right", None)
    request.session["text_result"] = None

    if not text:
        request.session["error_right"] = "Brakuje tekstu wejściowego."
        ctx = _ctx_from_session(request)
        return render(request, "predictor/home.html", ctx)

    extracted = _postprocess_extracted(extract_runner_profile(text))
    missing = extracted.get("missing", []) or []

    # If LLM reports missing fields -> show as "missing" (not as hard error_right)
    if missing:
        request.session["text_result"] = {
            "input_text": text,
            "extracted": extracted,
            "missing": missing,
        }
        ctx = _ctx_from_session(request)
        return render(request, "predictor/home.html", ctx)

    payload2 = {
        "sex": extracted.get("sex"),
        "age": extracted.get("age"),
        "t5k": extracted.get("t5k"),
        "t5k_s": extracted.get("t5k_s"),
    }

    try:
        sex, age, t5k_s = parse_input(payload2)
        t21k_s = predict_halfmarathon_time(t5k_s=t5k_s, age=age, sex=sex)

        request.session["text_result"] = {
            "input_text": text,
            "extracted": extracted,
            "prediction": {
                "t21k_s": t21k_s,
                "t21k_hhmmss": _format_hhmmss(t21k_s),
            },
        }

    except Exception as e:
        request.session["error_right"] = str(e)
        request.session["text_result"] = None

    ctx = _ctx_from_session(request)
    return render(request, "predictor/home.html", ctx)


# ---------------------------
# JSON API endpoints
# ---------------------------

@csrf_exempt
@require_POST
def predict(request):
    """
    POST /predict
    JSON body examples:
      {"sex":"M","age":40,"t5k_s":1500}
      {"sex":"M","age":40,"t5k":"25:00"}
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    try:
        sex, age, t5k_s = parse_input(payload)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    t21k_s = predict_halfmarathon_time(t5k_s=t5k_s, age=age, sex=sex)

    return JsonResponse({
        "input": {"sex": sex, "age": age, "t5k_s": t5k_s},
        "prediction": {"t21k_s": t21k_s, "t21k_hhmmss": _format_hhmmss(t21k_s)},
    })


@csrf_exempt
@require_POST
def parse(request):
    """
    POST /parse
    {"text": "..."} -> {"extracted": {...}}
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    text = str(payload.get("text", "")).strip()
    if not text:
        return JsonResponse({"error": "Missing field: text"}, status=400)

    extracted = _postprocess_extracted(extract_runner_profile(text))
    return JsonResponse({"extracted": extracted})


@csrf_exempt
@require_POST
def predict_text(request):
    """
    POST /predict_text
    {"text": "..."} -> extracted + prediction (or missing)
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    text = str(payload.get("text", "")).strip()
    if not text:
        return JsonResponse({"error": "Missing field: text"}, status=400)

    extracted = _postprocess_extracted(extract_runner_profile(text))
    missing = extracted.get("missing", []) or []

    if missing:
        return JsonResponse({
            "error": "Brakuje danych do predykcji.",
            "missing": missing,
            "extracted": extracted,
        }, status=400)

    payload2 = {
        "sex": extracted.get("sex"),
        "age": extracted.get("age"),
        "t5k": extracted.get("t5k"),
        "t5k_s": extracted.get("t5k_s"),
    }

    try:
        sex, age, t5k_s = parse_input(payload2)
    except ValueError as e:
        return JsonResponse({"error": str(e), "extracted": extracted}, status=400)

    t21k_s = predict_halfmarathon_time(t5k_s=t5k_s, age=age, sex=sex)

    return JsonResponse({
        "input_text": text,
        "extracted": extracted,
        "input": {"sex": sex, "age": age, "t5k_s": t5k_s},
        "prediction": {"t21k_s": t21k_s, "t21k_hhmmss": _format_hhmmss(t21k_s)},
    })

@csrf_exempt
@require_POST
def predict_text_ui(request):
    """
    AJAX endpoint used by the right panel.
    Body: {"text": "...", "openai_api_key": "..."}
    Returns JSON suitable to update the right-side Result box without page reload.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body."}, status=400)

    text = str(payload.get("text", "")).strip()
    api_key = str(payload.get("openai_api_key", "")).strip()

    if not text:
        return JsonResponse({"ok": False, "error": "Brakuje tekstu wejściowego."}, status=400)

    if not api_key:
        return JsonResponse({"ok": False, "error": "Brakuje klucza OpenAI."}, status=400)

    # IMPORTANT: do not store api_key anywhere; use only for this call
    try:
        extracted = _postprocess_extracted(extract_runner_profile(text, api_key=api_key))
    except TypeError:
        # If your extract_runner_profile() doesn't accept api_key yet, you'll add it in step 2.
        return JsonResponse({"ok": False, "error": "LLM layer not updated: extract_runner_profile(text, api_key=...) missing."}, status=500)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"LLM error: {e}"}, status=500)

    missing = extracted.get("missing", []) or []
    if missing:
        return JsonResponse({
            "ok": True,
            "missing": missing,
            "extracted": extracted,
        })

    payload2 = {
        "sex": extracted.get("sex"),
        "age": extracted.get("age"),
        "t5k": extracted.get("t5k"),
        "t5k_s": extracted.get("t5k_s"),
    }

    try:
        sex, age, t5k_s = parse_input(payload2)
        t21k_s = predict_halfmarathon_time(t5k_s=t5k_s, age=age, sex=sex)
        return JsonResponse({
            "ok": True,
            "prediction": {"t21k_s": t21k_s, "t21k_hhmmss": _format_hhmmss(t21k_s)},
            "extracted": extracted,
        })
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e), "extracted": extracted}, status=400)

@csrf_exempt
@require_POST
def predict_form_ui(request):
    """
    AJAX endpoint used by the left panel (classic form).
    Body: {"sex":"M","age":40,"t5k":"25:00"}  OR {"sex":"M","age":40,"t5k_s":1500}
    Returns JSON to update left Result box without page reload.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body."}, status=400)

    try:
        sex, age, t5k_s = parse_input(payload)
        t21k_s = predict_halfmarathon_time(t5k_s=t5k_s, age=age, sex=sex)
        return JsonResponse({
            "ok": True,
            "input": {"sex": sex, "age": age, "t5k_s": t5k_s},
            "prediction": {"t21k_s": t21k_s, "t21k_hhmmss": _format_hhmmss(t21k_s)},
        })
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)
