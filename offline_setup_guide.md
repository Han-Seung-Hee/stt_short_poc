# 오프라인 설치 가이드 (Mac Apple Silicon M1)

폐쇄망(Air-gapped) M1 Mac에서 STT + 요약 PoC를 구동하기 위한 단계별 설치 가이드입니다.

> **전제 조건**: 외부망 접근 가능한 Mac이 별도로 있어야 합니다 (다운로드용).

---

## 1단계: 외부망 Mac에서 다운로드

### 1-1. Python 설치 파일

```bash
# python.org에서 macOS용 유니버셜 설치 파일(.pkg) 다운로드
# https://www.python.org/downloads/
# 예: python-3.11.8-macos11.pkg
curl -O https://www.python.org/ftp/python/3.11.8/python-3.11.8-macos11.pkg
```

### 1-2. Python 패키지 (wheel 파일)

```bash
# 프로젝트 디렉토리에서 wheel 파일을 다운로드
mkdir -p offline_packages/wheels
cd offline_packages

# requirements.txt 기반으로 모든 의존성을 wheel로 다운로드
# --platform: Apple Silicon Mac 전용
pip download \
  -r requirements.txt \
  -d ./wheels \
  --platform macosx_11_0_arm64 \
  --python-version 3.11 \
  --only-binary=:all:

# 순수 Python 패키지는 --platform 없이 추가 다운로드
pip download \
  -r requirements.txt \
  -d ./wheels \
  --no-binary=:none:

# ※ 일부 패키지가 누락될 수 있으므로, 동일 사양의 Mac에서
#    가상환경을 만들어 설치 테스트 후 wheel을 복사하는 것이 가장 확실합니다:
# python -m venv test_env
# source test_env/bin/activate
# pip install -r requirements.txt
# pip wheel -r requirements.txt -w ./wheels
```

### 1-3. Whisper 모델 (mlx-whisper용)

```bash
# HuggingFace에서 mlx-community 모델 다운로드
# huggingface-cli 사용 (pip install huggingface-hub)
huggingface-cli download mlx-community/whisper-small-mlx --local-dir ./models/whisper-small-mlx

# 또는 git lfs 사용
# git lfs install
# git clone https://huggingface.co/mlx-community/whisper-small-mlx ./models/whisper-small-mlx

# ※ 한국어 품질 향상이 필요하면 medium도 추가 다운로드
# huggingface-cli download mlx-community/whisper-medium-mlx --local-dir ./models/whisper-medium-mlx
```

### 1-4. Ollama 설치 파일 + LLM 모델

```bash
# Ollama Mac 설치 파일 다운로드
curl -L -o Ollama-darwin.zip https://ollama.com/download/Ollama-darwin.zip

# Ollama를 설치 & 실행 후 모델 다운로드
# (외부망 Mac에서 한 번 실행해야 모델 파일을 받을 수 있습니다)
ollama pull qwen2.5:7b

# 다운로드된 모델 파일 위치 확인
# Mac: ~/.ollama/models/
# 이 디렉토리 전체를 USB에 복사합니다
```

### 1-5. Xcode Command Line Tools (오프라인 설치용)

```bash
# Apple Developer 사이트에서 .dmg 다운로드
# https://developer.apple.com/download/all/
# "Command Line Tools for Xcode 15.x" 검색 후 다운로드
# ※ Apple ID 로그인 필요
```

### 1-6. USB에 복사할 파일 목록

```
USB/
├── python-3.11.8-macos11.pkg          # Python 설치파일
├── Command_Line_Tools_for_Xcode_15.dmg # Xcode CLT
├── Ollama-darwin.zip                   # Ollama 앱
├── wheels/                             # Python wheel 파일들
│   ├── fastapi-*.whl
│   ├── uvicorn-*.whl
│   ├── mlx-*.whl
│   ├── mlx_whisper-*.whl
│   └── ... (모든 의존성)
├── models/
│   └── whisper-small-mlx/              # Whisper 모델
│       ├── config.json
│       ├── model.safetensors
│       └── ...
├── ollama_models/                      # Ollama 모델 (~/.ollama/models/ 복사)
│   ├── manifests/
│   └── blobs/
└── stt_short/                          # 프로젝트 소스코드
    ├── app/
    ├── config.yaml
    ├── requirements.txt
    └── ...
```

---

## 2단계: 내부망 Mac 설치

### 2-1. Xcode Command Line Tools 설치

```bash
# .dmg 파일을 더블클릭하여 설치
# 또는 이미 설치되어 있는지 확인:
xcode-select -p
# /Library/Developer/CommandLineTools 가 출력되면 이미 설치됨
```

### 2-2. Python 설치

```bash
# .pkg 파일을 더블클릭하여 설치 (관리자 권한 필요)
# 설치 후 확인:
python3 --version
# Python 3.11.8
```

### 2-3. 프로젝트 세팅

```bash
# USB에서 프로젝트 소스를 원하는 위치에 복사
cp -r /Volumes/USB/stt_short ~/projects/stt_short
cd ~/projects/stt_short

# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 오프라인 wheel 설치
pip install --no-index --find-links=/Volumes/USB/wheels -r requirements.txt

# 설치 확인
python -c "import fastapi; import mlx_whisper; import httpx; print('✓ 모든 패키지 설치 완료')"
```

### 2-4. Whisper 모델 배치

```bash
# USB에서 모델 복사
cp -r /Volumes/USB/models/whisper-small-mlx ~/projects/stt_short/models/whisper-small-mlx

# config.yaml에서 model_path 설정
# stt:
#   model_path: "/Users/사용자명/projects/stt_short/models/whisper-small-mlx"
```

### 2-5. Ollama 설치 및 모델 로드

```bash
# Ollama 앱 설치
unzip /Volumes/USB/Ollama-darwin.zip -d /Applications/

# Ollama 모델 파일 복사 (외부망에서 받아온 것)
mkdir -p ~/.ollama/models
cp -r /Volumes/USB/ollama_models/* ~/.ollama/models/

# Ollama 앱 실행 (Applications에서 더블클릭 또는)
open /Applications/Ollama.app

# 모델 인식 확인
ollama list
# qwen2.5:7b 가 목록에 있어야 함
```

### 2-6. 서버 실행 및 테스트

```bash
cd ~/projects/stt_short
source .venv/bin/activate

# 서버 시작
python -m app.main

# 또는
uvicorn app.main:app --host 0.0.0.0 --port 8000

# --- 다른 터미널에서 테스트 ---

# 헬스체크
curl http://localhost:8000/api/v1/health

# STT + 요약 통합 테스트
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@test_audio.wav" \
  -F "language=ko" \
  -F "chunk_enabled=false"

# STT만 테스트
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@test_audio.wav"

# 요약만 테스트 (STT 없이)
curl -X POST http://localhost:8000/api/v1/summarize \
  -F "text=고객이 신용카드 한도 증액을 요청했습니다. 상담사는 소득증빙 서류 제출이 필요하다고 안내했습니다. 고객은 3일 내 서류를 제출하기로 했습니다."
```

---

## 3단계: 트러블슈팅

### pydub FFmpeg 필요 (청크 분할 사용 시)

pydub는 내부적으로 ffmpeg를 사용합니다. 청크 분할 기능을 사용하려면:

```bash
# 옵션 A: 외부망에서 ffmpeg 정적 바이너리 다운로드
# https://evermeet.cx/ffmpeg/ 에서 macOS arm64용 다운로드
# USB로 반입 후 /usr/local/bin/ 에 복사

# 옵션 B: 청크 분할 미사용 시 ffmpeg 불필요
# config.yaml에서 chunk.enabled: false (기본값)
```

### Ollama 연결 실패

```bash
# Ollama 프로세스 확인
pgrep -fl ollama

# Ollama가 실행 중이 아니면 재시작
open /Applications/Ollama.app

# 포트 확인 (기본 11434)
lsof -i :11434
```

### 메모리 부족

```bash
# 현재 메모리 사용량 확인
vm_stat | head -10

# Whisper 모델을 small → base로 축소
# config.yaml: stt.model_name: "base"

# Ollama 모델을 더 작은 것으로 교체
# ollama pull gemma-2-2b-it (사전에 외부망에서 다운로드 필요)
```
