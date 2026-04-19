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

# Pre-download model at build time → 런타임 콜드스타트 단축
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/models/Qwen3-TTS-12Hz-1.7B-Base')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
