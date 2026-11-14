import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import io
import re
import inspect
import threading
import subprocess
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(title="Scribble Paradee 1.3 All-Voice Server")

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

tts_instances = {}
paradee_mod = None
inference_lock = threading.Lock()
audio_cache = {}

def get_paradee_module():
    global paradee_mod
    if paradee_mod is None:
        import paradee
        paradee_mod = paradee
    return paradee_mod

def get_engine_for_voice(voice: str):
    """Creates or retrieves a cached Paradee instance matching the voice language."""
    pm = get_paradee_module()
    prefix = voice[0]  # 'h' for Hindi, 'a' for US English, 'b' for UK English

    if prefix in tts_instances:
        return tts_instances[prefix]

    cls = getattr(pm, "Paradee", None) or getattr(pm, "TTS", None)
    if cls is None:
        return pm

    # Try initializing with language code if constructor supports it
    sig = inspect.signature(cls.__init__)
    for lang_val in (prefix, "hi" if prefix == "h" else "en"):
        try:
            if "lang" in sig.parameters:
                inst = cls(lang=lang_val)
                tts_instances[prefix] = inst
                return inst
            elif "language" in sig.parameters:
                inst = cls(language=lang_val)
                tts_instances[prefix] = inst
                return inst
        except Exception:
            pass

    if "default" not in tts_instances:
        tts_instances["default"] = cls()
    tts_instances[prefix] = tts_instances["default"]
    return tts_instances[prefix]

@app.on_event("startup")
def startup_load():
    try:
        get_engine_for_voice("hf_alpha")
        print("✅ Paradee 1.3 Base Engine Ready!")
    except Exception as e:
        print("Startup note:", e)

def split_long_sentence(text: str, max_chars: int = 220):
    """Prevents ONNX 510 token overflow crashes on long paragraphs."""
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return [clean] if clean else []
    words = clean.split(" ")
    chunks, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks

def extract_audio_array(result):
    sample_rate = 24000
    if isinstance(result, tuple):
        audio_data = result[0]
        if len(result) > 1 and isinstance(result[1], (int, float)):
            sample_rate = int(result[1])
    elif hasattr(result, "audio"):
        audio_data = result.audio
        sample_rate = int(getattr(result, "sample_rate", 24000))
    else:
        audio_data = result
    arr = np.asarray(audio_data, dtype=np.float32).squeeze()
    return arr, sample_rate

def synthesize_single_chunk(chunk: str, voice: str, speed: float):
    engine = get_engine_for_voice(voice)
    prefix = voice[0]  # 'h', 'a', or 'b'
    lang_candidates = [prefix, "hi" if prefix == "h" else ("en-gb" if prefix == "b" else "en-us"), "en"]

    last_err = None
    for method_name in ("synthesize", "generate", "speak", "create", "__call__"):
        fn = getattr(engine, method_name, None)
        if not callable(fn):
            continue

        sig = inspect.signature(fn)
        params = sig.parameters

        # Try calling with proper language codes first, then without lang
        for lang_try in lang_candidates + [None]:
            try:
                kwargs = {}
                if "voice" in params:
                    kwargs["voice"] = voice
                if "speed" in params:
                    kwargs["speed"] = speed
                if lang_try is not None:
                    if "lang" in params:
                        kwargs["lang"] = lang_try
                    elif "language" in params:
                        kwargs["language"] = lang_try
                res = fn(chunk, **kwargs)
                if res is not None:
                    return extract_audio_array(res)
            except Exception as e:
                last_err = e

    # Direct ONNX + espeak-ng fallback if Paradee's English wrapper failed
    try:
        espeak_voice = "hi" if prefix == "h" else ("en-gb" if prefix == "b" else "en-us")
        phonemes = subprocess.check_output(
            ["espeak-ng", "-q", "--ipa", "-v", espeak_voice, chunk],
            text=True
        ).strip()
        if phonemes and hasattr(engine, "synthesize_phonemes"):
            res = engine.synthesize_phonemes(phonemes, voice=voice, speed=speed)
            return extract_audio_array(res)
    except Exception:
        pass

    raise RuntimeError(f"Voice '{voice}' failed: {last_err}")

@app.get("/debug")
def debug_paradee():
    """Shows exact Paradee methods and voice test results in browser."""
    pm = get_paradee_module()
    eng = get_engine_for_voice("hf_alpha")
    info = {
        "module_dir": [x for x in dir(pm) if not x.startswith("_")],
        "engine_dir": [x for x in dir(eng) if not x.startswith("_")],
        "tests": {},
    }
    for v, txt in [("hf_alpha", "नमस्ते"), ("af_heart", "Hello world"), ("bf_isabella", "Good evening")]:
        try:
            arr, sr = synthesize_single_chunk(txt, v, 1.0)
            info["tests"][v] = f"OK ({len(arr)} samples at {sr}Hz)"
        except Exception as e:
            info["tests"][v] = f"ERROR: {repr(e)}"
    return info

@app.get("/tts")
@app.post("/tts")
def synthesize_speech(
    text: str = Query(..., min_length=1, max_length=2000),
    voice: str = Query("hf_alpha"),
    speed: float = Query(1.0, ge=0.5, le=2.0),
):
    # Clean special symbols that break phonemizer
    clean_text = re.sub(r"[*~_`<>@#$%^&+=|\\]", " ", text).strip()
    if not re.search(r"[\w\u0900-\u097F]", clean_text):
        # Return 0.15s silence if line only has punctuation like "..." or "—"
        silence = np.zeros(int(24000 * 0.15), dtype=np.float32)
        wav_io = io.BytesIO()
        sf.write(wav_io, silence, 24000, format="WAV", subtype="PCM_16")
        return Response(content=wav_io.getvalue(), media_type="audio/wav")

    chosen_voice = voice if voice in VALID_VOICES else "hf_alpha"
    cache_key = f"{chosen_voice}:{speed}:{clean_text}"

    if cache_key in audio_cache:
        return Response(content=audio_cache[cache_key], media_type="audio/wav")

    try:
        chunks = split_long_sentence(clean_text, 200)
        audio_segments = []
        final_sr = 24000

        with inference_lock:
            for ch in chunks:
                arr, sr = synthesize_single_chunk(ch, chosen_voice, speed)
                final_sr = sr
                if arr.size > 0:
                    audio_segments.append(arr)

        if not audio_segments:
            raise RuntimeError("Empty audio generated")

        combined_audio = np.concatenate(audio_segments)
        wav_io = io.BytesIO()
        sf.write(wav_io, combined_audio, final_sr, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        if len(audio_cache) > 35:
            audio_cache.pop(next(iter(audio_cache)), None)
        audio_cache[cache_key] = wav_bytes

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "public, max-age=604800",
                "X-Paradee-Voice": chosen_voice,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Paradee error ({chosen_voice}): {exc}")
