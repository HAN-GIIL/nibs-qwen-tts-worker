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
    # HF Space가 쓰는 kernels-community/flash-attn3는 우리 이미지에서 silently fallback됨
    # → sdpa (PyTorch 내장 fused)로 명시. H100/4090에서 fast + accurate
    import traceback as _tb
    attn = os.environ.get("ATTN_IMPL", "sdpa")
    print(f"[Qwen-TTS] Trying attn_implementation={attn}", flush=True)
    try:
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, device_map="cuda", dtype=torch.bfloat16, attn_implementation=attn,
        )
    except Exception as e:
        print(f"[Qwen-TTS] {attn} failed ({e}), falling back to eager", flush=True)
        _tb.print_exc()
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, device_map="cuda", dtype=torch.bfloat16, attn_implementation="eager",
        )
    # 어떤 attention이 실제로 박혔는지 로그
    try:
        inner = _model.model
        attn_used = getattr(inner.config, "_attn_implementation", "unknown")
        print(f"[Qwen-TTS] Actual _attn_implementation={attn_used}", flush=True)
    except Exception:
        pass
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


_whisper = None

def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("[Qwen-TTS] Loading whisper base...", flush=True)
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper


def _trim_tail_by_whisper(audio: np.ndarray, sr: int, body_chars: str, keep_after_s: float = 0.8) -> np.ndarray:
    """패딩 포함한 audio에서 본문 마지막 단어 + keep_after_s까지만 남김.
    body_chars: 원본 텍스트의 한글만 모은 문자열."""
    import re, tempfile
    try:
        wm = _get_whisper()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, sr)
        segs, _ = wm.transcribe(tmp.name, language="ko", word_timestamps=True)
        words = []
        for seg in segs:
            if seg.words:
                words.extend(seg.words)
        if not words:
            return audio
        # 본문 마지막 단어 위치 (glove pattern: whisper 단어가 body_chars에 포함되는 마지막 index)
        last_idx = -1
        for i, w in enumerate(words):
            wc = re.sub(r'[^가-힣]', '', w.word)
            if not wc:
                continue
            if wc in body_chars or any(wc[:k] in body_chars for k in range(len(wc), 1, -1)):
                last_idx = i
        if last_idx >= 0:
            cut_time = min(words[last_idx].end + keep_after_s, len(audio) / sr)
            print(f"[Qwen-TTS] tail trim: last_word='{words[last_idx].word.strip()}' end={words[last_idx].end:.2f}s cut={cut_time:.2f}s", flush=True)
            return audio[:int(sr * cut_time)]
    except Exception as e:
        print(f"[Qwen-TTS] whisper trim err: {e}", flush=True)
    return audio


def _noise_gate_front(audio: np.ndarray, sr: int, duration_s: float = 0.4,
                       threshold: float = 0.035, win_ms: int = 10,
                       attenuation: float = 0.05) -> np.ndarray:
    """앞 duration_s 구간의 저에너지 wideband 노이즈(솨악) 감쇠."""
    end = min(int(sr * duration_s), len(audio))
    win = int(sr * win_ms / 1000)
    if end < win * 2:
        return audio
    result = audio.astype(np.float32, copy=True)
    n_win = end // win
    gains = np.ones(n_win + 2, dtype=np.float32)
    for k in range(n_win):
        seg = audio[k*win:(k+1)*win]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64)**2)))
        gains[k] = attenuation if rms < threshold else 1.0
    gains[-2] = 1.0; gains[-1] = 1.0
    gain_env = np.interp(np.arange(end),
                         np.arange(n_win + 2) * win + win // 2, gains)
    result[:end] *= gain_env
    return result


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
        # HF Space _normalize_audio와 동일: float이면 max>1 일 때만 정규화, clip
        m = float(np.max(np.abs(wav))) if wav.size else 0.0
        if m > 1.0 + 1e-6:
            wav = wav / (m + 1e-12)
        wav = np.clip(wav, -1.0, 1.0)
        # Qwen 모델은 24kHz 기반. ref가 다르면 speech_tokenizer encode 결과 어긋남 → 리샘플
        TARGET_SR = 24000
        if ref_sr != TARGET_SR:
            import librosa as _lr
            wav = _lr.resample(wav, orig_sr=int(ref_sr), target_sr=TARGET_SR).astype(np.float32)
            ref_sr = TARGET_SR
        # ref 끝에 0.5s 묵음 패딩 — decoder 전이 garble이 본문 밖(묵음 위)에서 일어나게
        silence = np.zeros(int(ref_sr * 0.5), dtype=np.float32)
        wav = np.concatenate([wav, silence])
        audio_tuple = (wav, int(ref_sr))
        print(f"[Qwen-TTS] ref wav: shape={wav.shape}, sr={ref_sr}", flush=True)

        model = _load_model()

        # 텍스트에 꼬리 패딩 추가 — 마지막 음절 잘림 방지. whisper로 본문 끝에서 잘라냄.
        import re as _re
        body_chars = _re.sub(r'[^가-힣]', '', text)
        text_padded = text
        if text_padded and text_padded[-1] not in '.!?':
            text_padded += '.'
        text_padded = text_padded + ' 그럼 다음에 또 봬요.'

        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text_padded,
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text,
            x_vector_only_mode=False,
            max_new_tokens=2048,
        )
        audio = wavs[0]
        gen_s = time.time() - t0
        # 후처리 순서: 꼬리 trim (본문 끝 + 0.8s) → 앞 노이즈 게이트
        audio = _trim_tail_by_whisper(audio, sr, body_chars)
        audio = _noise_gate_front(audio, sr)
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
