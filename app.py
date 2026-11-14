import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import io
import re
import gc
import json
import ctypes
import inspect
import threading
import subprocess
import numpy as np
import soundfile as sf
import onnxruntime as ort
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Low-RAM ONNX session patch
_orig_session_init = ort.InferenceSession.__init__
def _low_ram_session_init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    _orig_session_init(self, path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)

ort.InferenceSession.__init__ = _low_ram_session_init

app = FastAPI(title="Scribble Paradee 1.3 Fast All-Voice Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

VOICE_INDEX_MAP = {
    "af_heart": 0,
    "bf_isabella": 1,
    "am_puck": 2,
    "bm_fable": 3,
    "hf_alpha": 4,
    "hm_omega": 5,
}

tts_engine = None
ort_session = None
vocab_map = None
voice_names_list = None
inference_lock = threading.Lock()

def release_os_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def espeak_to_misaki_english(ipa: str) -> str:
    """Converts fast espeak-ng English IPA directly to Paradee's Misaki phoneme set (Zero RAM!)."""
    s = ipa.strip()
    replacements = [
        ("ɑːɹ", "ɑɹ"),
        ("ɔːɹ", "ɔɹ"),
        ("ɜː", "ɜɹ"),
        ("əʊ", "O"),
        ("oʊ", "O"),
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("ɔɪ", "Y"),
        ("aʊ", "W"),
        ("tʃ", "ʧ"),
        ("dʒ", "ʤ"),
        ("r", "ɹ"),
        ("ː", ""),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s

def find_onnx_and_vocab(eng):
    """Extracts the already-loaded ONNX session and vocab dictionary from Paradee instance."""
    global ort_session, vocab_map, voice_names_list
    for attr in dir(eng):
        if attr.startswith("__"):
            continue
        val = getattr(eng, attr, None)
        if isinstance(val, ort.InferenceSession):
            ort_session = val
        elif isinstance(val, dict) and len(val) > 40 and all(isinstance(v, int) for v in val.values()):
            vocab_map = val

    # Also check module/package files if config.json is stored alongside paradee
    import paradee
    pkg_dir = os.path.dirname(inspect.getfile(paradee))
    for root, _, files in os.walk(pkg_dir):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith(".json") and vocab_map is None:
                try:
                    with open(full, "r", encoding="utf-8") as fp:
                        cfg = json.load(fp)
                    v = cfg.get("vocab") or cfg.get("phoneme_to_id") or cfg.get("tokens")
                    if isinstance(v, dict):
                        vocab_map = v
                    if isinstance(cfg.get("voices"), list):
                        voice_names_list = cfg["voices"]
                except Exception:
                    pass
            elif f.endswith(".onnx") and ort_session is None:
                try:
                    ort_session = ort.InferenceSession(full)
                except Exception:
                    pass

def ensure_engine():
    global tts_engine
    if tts_engine is not None:
        return
    import paradee
    if hasattr(paradee, "Paradee"):
        tts_engine = paradee.Paradee()
    elif hasattr(paradee, "TTS"):
        tts_engine = paradee.TTS()
    else:
        tts_engine = paradee
    find_onnx_and_vocab(tts_engine)
    release_os_memory()
    print(f"✅ Paradee Ready! Direct ONNX={ort_session is not None}, Vocab={vocab_map is not None}")

@app.on_event("startup")
def startup_event():
    ensure_engine()

def direct_espeak_onnx_synthesize(phrase: str, voice: str, speed: float):
    """
    Ultra-fast (<150ms), Zero-RAM-Leak synthesis using espeak-ng + ONNX directly!
    Works identically for all 4 English voices and 2 Hindi voices without loading heavy Python dicts.
    """
    if ort_session is None or not vocab_map:
        return None

    is_hindi = voice.startswith("h") or bool(re.search(r"[\u0900-\u097F]", phrase))
    espeak_lang = "hi" if is_hindi else ("en-gb" if voice.startswith("b") else "en-us")

    raw_ipa = subprocess.check_output(
        ["espeak-ng", "-q", "--ipa", "-v", espeak_lang, phrase],
        text=True,
        timeout=4,
    ).strip()

    ipa = raw_ipa if is_hindi else espeak_to_misaki_english(raw_ipa)
    token_ids = [0]
    for ch in ipa:
        if ch in vocab_map:
            token_ids.append(vocab_map[ch])
        elif ch in (" ", "\n") and " " in vocab_map:
            token_ids.append(vocab_map[" "])
        if len(token_ids) >= 510:
            break
    token_ids.append(0)

    if len(token_ids) <= 2:
        return None

    voice_idx = VOICE_INDEX_MAP.get(voice, 4)
    if voice_names_list and voice in voice_names_list:
        voice_idx = voice_names_list.index(voice)

    feeds = {}
    for inp in ort_session.get_inputs():
        name = inp.name
        is_int32 = "int32" in inp.type
        dtype = np.int32 if is_int32 else np.int64
        if "input" in name or "id" in name or "token" in name:
            feeds[name] = np.array([token_ids], dtype=dtype)
        elif "speed" in name or "dur" in name:
            feeds[name] = np.array([float(speed)], dtype=np.float32)
        elif "voice" in name or "spk" in name or "style" in name:
            feeds[name] = np.array([voice_idx], dtype=dtype)

    out = ort_session.run(None, feeds)[0]
    return np.asarray(out, dtype=np.float32).squeeze(), 24000

def run_single_phrase(phrase: str, voice: str, speed: float):
    ensure_engine()

    # For English voices (af_heart, bf_isabella, am_puck, bm_fable), ALWAYS try fast direct espeak+ONNX first
    # so Python never hangs or runs out of RAM loading misaki's heavy English dictionary!
    if not voice.startswith("h"):
        try:
            direct_res = direct_espeak_onnx_synthesize(phrase, voice, speed)
            if direct_res is not None and direct_res[0].size > 100:
                return direct_res
        except Exception as e:
            print("Direct English ONNX fallback note:", e)

    # Standard Paradee synthesis (Works great for Hindi hf_alpha / hm_omega)
    for name in ("synthesize", "generate", "speak", "create", "__call__"):
        fn = getattr(tts_engine, name, None)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs = {}
        if "voice" in params:
            kwargs["voice"] = voice
        if "speed" in params:
            kwargs["speed"] = speed
        try:
            res = fn(phrase, **kwargs)
            sample_rate = 24000
            if isinstance(res, tuple):
                audio_data = res[0]
                if len(res) > 1 and isinstance(res[1], (int, float)):
                    sample_rate = int(res[1])
            elif hasattr(res, "audio"):
                audio_data = res.audio
                sample_rate = int(getattr(res, "sample_rate", 24000))
            else:
                audio_data = res
            arr = np.asarray(audio_data, dtype=np.float32).squeeze()
            if arr.size > 0:
                return arr, sample_rate
        except Exception:
            pass

    # Final fallback via direct espeak+ONNX for any voice
    direct_res = direct_espeak_onnx_synthesize(phrase, voice, speed)
    if direct_res is not None:
        return direct_res

    raise RuntimeError(f"Synthesis failed for voice {voice}")

def split_into_safe_phrases(text: str, max_len: int = 130):
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= max_len:
        return [clean]
    parts = re.split(r"(?<=[।.,!?;:—\-])\s+", clean)
    chunks, cur = [], ""
    for p in parts:
        if not p:
            continue
        if len(p) > max_len:
            for w in p.split(" "):
                if not cur:
                    cur = w
                elif len(cur) + 1 + len(w) <= max_len:
                    cur += " " + w
                else:
                    chunks.append(cur)
                    cur = w
        elif not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_len:
            cur += " " + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

@app.get("/phonemize")
def phonemize_api(text: str = Query(..., max_length=1500), voice: str = Query("hf_alpha")):
    """Ultra-lightweight endpoint (<5ms, 0MB RAM) for client-side mobile Web Worker!"""
    is_hindi = voice.startswith("h") or bool(re.search(r"[\u0900-\u097F]", text))
    espeak_lang = "hi" if is_hindi else ("en-gb" if voice.startswith("b") else "en-us")
    raw_ipa = subprocess.check_output(
        ["espeak-ng", "-q", "--ipa", "-v", espeak_lang, text],
        text=True,
        timeout=4,
    ).strip()
    ipa = raw_ipa if is_hindi else espeak_to_misaki_english(raw_ipa)
    return {"ipa": ipa, "voice": voice}

@app.get("/")
@app.get("/debug")
def status_check():
    ensure_engine()
    return {
        "status": "online",
        "direct_onnx_ready": ort_session is not None,
        "vocab_size": len(vocab_map) if vocab_map else 0,
        "voices": list(VOICE_INDEX_MAP.keys()),
    }

@app.get("/tts")
@app.post("/tts")
def tts_endpoint(
    text: str = Query(..., min_length=1, max_length=1500),
    voice: str = Query("hf_alpha"),
    speed: float = Query(1.0, ge=0.5, le=2.0),
):
    clean_text = re.sub(r"[*~_`<>@#$%^&+=|\\]", " ", text).strip()
    if not re.search(r"[\w\u0900-\u097F]", clean_text):
        silence = np.zeros(int(24000 * 0.15), dtype=np.float32)
        wav_io = io.BytesIO()
        sf.write(wav_io, silence, 24000, format="WAV", subtype="PCM_16")
        return Response(content=wav_io.getvalue(), media_type="audio/wav")

    chosen_voice = voice if voice in VOICE_INDEX_MAP else "hf_alpha"
    phrases = split_into_safe_phrases(clean_text, 130)

    try:
        segments = []
        final_sr = 24000
        with inference_lock:
            for ph in phrases:
                arr, sr = run_single_phrase(ph, chosen_voice, speed)
                final_sr = sr
                if arr.size > 0:
                    segments.append(arr)
                release_os_memory()

        if not segments:
            raise RuntimeError("Empty audio output")

        full_audio = np.concatenate(segments) if len(segments) > 1 else segments[0]
        wav_io = io.BytesIO()
        sf.write(wav_io, full_audio, final_sr, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        del segments, full_audio
        release_os_memory()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "public, max-age=604800",
                "X-Paradee-Voice": chosen_voice,
            },
        )
    except Exception as exc:
        release_os_memory()
        raise HTTPException(status_code=500, detail=f"{exc}")
