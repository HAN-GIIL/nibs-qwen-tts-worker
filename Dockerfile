# RunPod serverless Qwen3-TTS (CUDA 12.4 + bf16 + flash-attn3 via kernels-community)
FROM runpod/pytorch:2.8.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base

# ffmpeg for pydub mp3 encode
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt hf_transfer

# 모델은 빌드 타임에 굽지 않음 (GHA 디스크 한계) → 첫 콜드스타트 때 handler가 자동 다운로드
# RunPod Network Volume을 마운트해두면 워커 재시작 시에도 캐시 공유됨

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
