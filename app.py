import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import io
import inspect
import threading
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(title="Scribble Paradee 1.3 TTS Server")

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
paradee_module = None
inference_lock = threading.Lock()

@app.on_event("startup")
def load_paradee_engine():
    global tts_engine, paradee_module
    import paradee
    paradee_module = paradee
    if hasattr(paradee, "Paradee"):
        tts_engine = paradee.Paradee()
    elif hasattr(paradee, "TTS"):
        tts_engine = paradee.TTS()
    print("✅ Paradee 1.3 Engine loaded into RAM!")

def run_paradee_inference(text: str, voice: str, speed: float):
    target = tts_engine if tts_engine is not None else paradee_module

    for method_name in ("synthesize", "generate", "speak", "__call__"):
        fn = getattr(target, method_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            kwargs = {}
            if "voice" in sig.parameters:
                kwargs["voice"] = voice
            if "speed" in sig.parameters:
                kwargs["speed"] = speed
            if "lang" in sig.parameters:
                if voice.startswith("h"):
                    kwargs["lang"] = "hi"
                elif voice.startswith("b"):
                    kwargs["lang"] = "en-gb"
                else:
                    kwargs["lang"] = "en-us"

            try:
                return fn(text, **kwargs)
            except Exception:
                # Fallback with only voice parameter if lang/speed rejected
                return fn(text, voice=voice)

    raise RuntimeError("Could not find synthesis method on paradee module.")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "Paradee 1.3 (8.5MB INT8 ONNX)",
        "voices": list(VALID_VOICES),
    }

@app.get("/tts")
@app.post("/tts")
def synthesize_speech(
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
            result = run_paradee_inference(clean_text, chosen_voice, speed)

        sample_rate = 24000
        if isinstance(result, tuple):
            audio_data = result[0]
            if len(result) > 1 and isinstance(result[1], (int, float)):
                sample_rate = int(result[1])
        elif hasattr(result, "audio"):
            audio_data = result.audio
            sample_rate = int(getattr(result, "sample_rate", 24000))
        elif isinstance(result, (bytes, bytearray)):
            return Response(
                content=bytes(result),
                media_type="audio/wav",
                headers={"Cache-Control": "public, max-age=604800"},
            )
        else:
            audio_data = result

        audio_np = np.asarray(audio_data, dtype=np.float32).squeeze()

        wav_io = io.BytesIO()
        sf.write(wav_io, audio_np, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "public, max-age=604800, s-maxage=604800",
                "X-Paradee-Voice": chosen_voice,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Paradee synthesis error: {exc}")
