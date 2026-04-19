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
# 먼저 일반 deps 설치 → 그다음 qwen_tts를 --no-deps로 (gradio/fastapi 제외)
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --no-deps qwen_tts

# 모델은 빌드 타임에 굽지 않음 (GHA 디스크 한계) → 첫 콜드스타트 때 handler가 자동 다운로드
# RunPod Network Volume을 마운트해두면 워커 재시작 시에도 캐시 공유됨

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
