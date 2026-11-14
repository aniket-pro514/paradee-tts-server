import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import io
import gc
import inspect
import threading
import traceback
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(title="Scribble Paradee 1.3 Lightweight Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

VALID_VOICES = {
    "hf_alpha",
    "hm_omega",
    "af_heart",
    "am_puck",
    "bf_isabella",
    "bm_fable",
}

tts_engine = None
paradee_mod = None
inference_lock = threading.Lock()

def ensure_engine():
    global tts_engine, paradee_mod
    if tts_engine is not None or paradee_mod is not None:
        return
    import paradee
    paradee_mod = paradee
    if hasattr(paradee, "Paradee"):
        tts_engine = paradee.Paradee()
    elif hasattr(paradee, "TTS"):
        tts_engine = paradee.TTS()
    gc.collect()
    print("✅ Single Paradee Engine loaded cleanly!")

@app.on_event("startup")
def startup_event():
    ensure_engine()

def call_paradee(text: str, voice: str, speed: float):
    ensure_engine()
    target = tts_engine if tts_engine is not None else paradee_mod

    for name in ("synthesize", "generate", "speak", "create", "__call__"):
        fn = getattr(target, name, None)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs = {}
        if "voice" in params:
            kwargs["voice"] = voice
        if "speed" in params:
            kwargs["speed"] = speed
        return fn(text, **kwargs)

    raise RuntimeError("No synthesis method found on Paradee object")

@app.get("/")
def health():
    return {"status": "ok", "model": "Paradee 1.3 Single-RAM Instance"}

@app.get("/debug")
def debug_info():
    """Safe inspection without loading extra models or crashing RAM!"""
    ensure_engine()
    target = tts_engine if tts_engine is not None else paradee_mod
    methods = {}
    for name in dir(target):
        if not name.startswith("_"):
            attr = getattr(target, name, None)
            if callable(attr):
                try:
                    methods[name] = str(inspect.signature(attr))
                except Exception:
                    methods[name] = "callable"
            else:
                methods[name] = str(type(attr).__name__)

    # Test 1 short English call safely and show exact error if it fails
    en_test = {}
    try:
        res = call_paradee("Hello", "af_heart", 1.0)
        en_test["af_heart"] = f"SUCCESS: {type(res)}"
    except Exception as e:
        en_test["af_heart_error"] = traceback.format_exc()

    return {
        "engine_type": str(type(target)),
        "attributes_and_methods": methods,
        "english_test": en_test,
    }

@app.get("/tts")
@app.post("/tts")
def tts_endpoint(
    text: str = Query(..., min_length=1, max_length=1500),
    voice: str = Query("hf_alpha"),
    speed: float = Query(1.0, ge=0.5, le=2.0),
):
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Empty text")

    chosen_voice = voice if voice in VALID_VOICES else "hf_alpha"

    try:
        with inference_lock:
            result = call_paradee(clean_text, chosen_voice, speed)

        sample_rate = 24000
        if isinstance(result, tuple):
            audio_data = result[0]
            if len(result) > 1 and isinstance(result[1], (int, float)):
                sample_rate = int(result[1])
        elif hasattr(result, "audio"):
            audio_data = result.audio
            sample_rate = int(getattr(result, "sample_rate", 24000))
        elif isinstance(result, (bytes, bytearray)):
            return Response(content=bytes(result), media_type="audio/wav")
        else:
            audio_data = result

        audio_np = np.asarray(audio_data, dtype=np.float32).squeeze()
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_np, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()
        gc.collect()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "public, max-age=604800",
                "X-Paradee-Voice": chosen_voice,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{exc}")
