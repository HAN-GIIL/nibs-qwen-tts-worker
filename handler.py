"""RunPod Serverless Handler — Qwen3-TTS 1.7B Voice Clone on CUDA + bf16 + flash-attn3.

Input:
    {
        "text": str,
        "ref_audio_base64": str,   # wav bytes, base64 encoded
        "ref_text": str,           # transcript of ref audio
        "language": "Korean",      # optional, default "Korean"
    }

Output:
    {
        "audio_base64": str,       # mp3 bytes, base64 encoded
        "sample_rate": int,
        "duration": float,
    }
"""
print("[Qwen-TTS] handler.py module loading...", flush=True)
import os, io, base64, hashlib, time, sys, traceback
try:
    import numpy as np
    import soundfile as sf
    import torch
    import runpod
    print(f"[Qwen-TTS] imports OK. torch={torch.__version__} cuda_avail={torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[Qwen-TTS] FATAL import error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Network Volume은 /runpod-volume에 마운트됨. 볼륨 있으면 거기 저장 → 콜드스타트마다 재다운로드 생략
_VOLUME_BASE = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/models"
MODEL_PATH = os.environ.get("MODEL_PATH", f"{_VOLUME_BASE}/Qwen3-TTS-12Hz-1.7B-Base")
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

_model = None


def _ensure_model_downloaded():
    """MODEL_PATH 없으면 HF에서 다운로드."""
    import os
    cfg = os.path.join(MODEL_PATH, "config.json")
    if os.path.exists(cfg):
        return
    print(f"[Qwen-TTS] Downloading {MODEL_REPO} → {MODEL_PATH}...", flush=True)
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_REPO, local_dir=MODEL_PATH)
    print(f"[Qwen-TTS] Download done in {time.time()-t0:.1f}s", flush=True)


def _load_model():
    global _model
    if _model is not None:
        return _model
    _ensure_model_downloaded()
    print(f"[Qwen-TTS] Loading {MODEL_PATH} (cuda + bf16 + flash-attn3)...", flush=True)
    t0 = time.time()
    from qwen_tts import Qwen3TTSModel
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="kernels-community/flash-attn3",
    )
    print(f"[Qwen-TTS] Loaded in {time.time()-t0:.1f}s", flush=True)
    return _model


def _save_ref_audio(ref_audio_b64: str) -> str:
    """base64 wav → /tmp/ref_{hash}.wav"""
    data = base64.b64decode(ref_audio_b64)
    h = hashlib.md5(data).hexdigest()[:16]
    path = f"/tmp/ref_{h}.wav"
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


def _encode_mp3(audio: np.ndarray, sr: int) -> str:
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sr, format="WAV")
    wav_buf.seek(0)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_wav(wav_buf)
        mp3_buf = io.BytesIO()
        seg.export(mp3_buf, format="mp3", bitrate="192k")
        return base64.b64encode(mp3_buf.getvalue()).decode()
    except Exception as e:
        print(f"[Qwen-TTS] mp3 encode failed ({e}), returning wav", flush=True)
        return base64.b64encode(wav_buf.getvalue()).decode()


def handler(event):
    inp = event.get("input") or {}
    text = inp.get("text", "").strip()
    ref_audio_b64 = inp.get("ref_audio_base64")
    ref_text = (inp.get("ref_text") or "").strip()
    language = inp.get("language", "Korean")

    if not text:
        return {"error": "text is required"}
    if not ref_audio_b64:
        return {"error": "ref_audio_base64 is required"}
    if not ref_text:
        return {"error": "ref_text is required"}

    try:
        # HF Space 방식 그대로: 파일 → soundfile로 numpy 로드 → (wav, sr) 튜플로 전달
        import io as _io
        ref_bytes = base64.b64decode(ref_audio_b64)
        wav, ref_sr = sf.read(_io.BytesIO(ref_bytes))
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wav = wav.astype(np.float32)
        # HF Space _normalize_audio와 동일: 이미 float이면 max>1 일 때만 정규화, clip
        m = float(np.max(np.abs(wav))) if wav.size else 0.0
        if m > 1.0 + 1e-6:
            wav = wav / (m + 1e-12)
        wav = np.clip(wav, -1.0, 1.0)
        audio_tuple = (wav, int(ref_sr))

        model = _load_model()

        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text,
            x_vector_only_mode=False,
            max_new_tokens=2048,
        )
        audio = wavs[0]
        gen_s = time.time() - t0
        duration = len(audio) / sr
        print(f"[Qwen-TTS] {len(text)} chars → {duration:.2f}s audio in {gen_s:.1f}s", flush=True)

        audio_b64 = _encode_mp3(audio, sr)
        return {
            "audio_base64": audio_b64,
            "sample_rate": int(sr),
            "duration": float(duration),
            "generation_seconds": float(gen_s),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print("[Qwen-TTS] Starting RunPod serverless loop...", flush=True)
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"[Qwen-TTS] serverless.start crashed: {e}", flush=True)
        traceback.print_exc()
        raise
    print("[Qwen-TTS] serverless.start returned (shouldn't happen)", flush=True)
