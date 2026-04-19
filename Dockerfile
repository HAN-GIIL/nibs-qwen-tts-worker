# RunPod serverless Qwen3-TTS (CUDA 12.4 + bf16 + flash-attn3 via kernels-community)
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base

# ffmpeg (pydub), libsndfile1 (soundfile), build tools for pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# HF Space에서 가져온 qwen_tts 소스를 PYTHONPATH로 직접 사용
# PyPI qwen_tts와 내부 구현이 다를 수 있어 HF Space와 동일 코드로 맞춤
COPY qwen_tts /app/qwen_tts
ENV PYTHONPATH=/app

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
