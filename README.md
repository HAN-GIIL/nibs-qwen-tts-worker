# RunPod Serverless Qwen3-TTS 배포 가이드

## 1. Docker 이미지 빌드 + 푸시

Docker Hub 계정 가정 (`YOUR_DOCKERHUB`). 이미지 크기 ~15GB (모델 프리다운로드 포함).

```bash
cd /Users/t_y/Desktop/Nibs-Backend/runpod

# 빌드 (ARM Mac이면 --platform linux/amd64 필수)
docker buildx build --platform linux/amd64 -t YOUR_DOCKERHUB/qwen3-tts-runpod:latest --push .
```

Docker Hub 안 쓰려면 RunPod Serverless에 Git 연동도 가능(위 경로에 Dockerfile만 있으면 자동 빌드).

## 2. RunPod Serverless 엔드포인트 생성

1. https://www.runpod.io/console/serverless → **+ New Endpoint**
2. **Docker Image**: `YOUR_DOCKERHUB/qwen3-tts-runpod:latest`
3. **GPU**: RTX 4090 (24GB) 선택. (1.7B bf16 = ~3.4GB, 여유 충분)
4. **Max Workers**: 3 (동시 요청 허용)
5. **Idle Timeout**: 5 초 (짧게 = 비용 절감)
6. **Active (Min) Workers**: **1** (always-warm. 콜드스타트 제거. 시간당 과금)
7. **Container Disk**: 25 GB (모델 포함)
8. **Execution Timeout**: 600 초
9. 환경변수 (선택): `MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base` (기본값)
10. Create → 엔드포인트 ID 복사

## 3. Mac 백엔드 환경변수 설정

Mac에서 서버 실행 전:

```bash
export RUNPOD_ENDPOINT_ID="xxxxxxxxxxxx"
export RUNPOD_API_KEY="rpa_xxxxxxxxxxxxxxxxxxxx"
```

또는 `~/Desktop/Nibs-Backend/.env` 파일에 저장 (서버 실행 스크립트에서 로드).

로컬 fallback으로 강제하려면 `RUNPOD_LOCAL=1` 설정.

## 4. 테스트

```bash
# API 직접 호출 테스트
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {
    "text": "안녕하세요. 테스트 음성입니다.",
    "ref_audio_base64": "<base64-encoded wav>",
    "ref_text": "<ref transcript>",
    "language": "Korean"
  }}'
```

## 5. 비용 예측

- RTX 4090: ~$0.40/hr active
- Always-warm 1 worker: $0.40 × 24 × 30 = **~$288/월 상한** (실제 요청 적으면 비례 감소)
- 콜드 허용하면 (Active 0) 요청당 과금만, $0/hr idle

## 트러블슈팅

- **콜드 스타트 첫 요청 90s~**: 정상. 모델 VRAM 로드 중. Active 1 이면 회피.
- **`flash-attn3` 설치 실패**: `kernels-community/flash-attn3` 는 H100/A100/4090 등에서만. RTX 3090 이하는 `attn_implementation="sdpa"`로 변경 필요.
- **OOM**: 1.7B bf16은 3.4GB. 24GB 4090에선 여유. 다른 워크로드 섞이면 체크.
