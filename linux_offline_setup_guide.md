# Linux 폐쇄망(Air-Gapped) 오프라인 셋업 가이드

본 문서는 외부 인터넷이 완벽히 차단된 폐쇄망 Linux 서버(테스트 서버 및 운영 서버)에 STT + 요약 파이프라인을 구축하기 위한 "사전 준비물"과 "서버실 반입 후 설치 절차"를 A to Z까지 설명합니다.

---

## 🚀 1단계: 외부망(인터넷 접속 가능) PC에서 파일 준비
서버실에 들어가기 전, 외부 인터넷이 연결된 **[서버와 동일한 OS 환경의 Linux PC]** 혹은 랩탑에서 다음 항목들을 모두 다운로드하여 이동식 매체(USB, 외장하드 등)에 담습니다.

### 1. Python 패키지 의존성 (.whl 파일들)
동일한 Python 버전(3.10 또는 3.11)의 외부 PC 터미널에서 다음 명령을 실행하여 모든 의존성 패키지 파일을 `wheels` 폴더로 다운받습니다.

```bash
mkdir -p offline_packages/wheels
cd offline_packages

# 프로젝트의 requirements.txt 파일을 복사해 왔다고 가정
pip download -r requirements.txt -d ./wheels

# Linux STT 필수 패키지도 추가 다운로드 (requirements.txt 에 주석처리 되어있을 경우)
pip download faster-whisper -d ./wheels

# ⭐ 해당 폐쇄망 서버가 NVIDIA GPU 장착 서버라면 PyTorch(CUDA 포함) 모델도 필수 다운로드
# pip download torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -d ./wheels
```

### 2. FFmpeg 정적(Static) 바이너리
화자 분리(채널 쪼개기)를 위한 필수 인프라 패키지입니다. (폐쇄망에서는 `apt-get` 사용 불가)
- 다운로드 경로: https://johnvansickle.com/ffmpeg/
- `ffmpeg-release-amd64-static.tar.xz` (AMD64 기준) 파일을 통째로 다운로드합니다.

### 3. Ollama (LLM 구동 엔진) 실행 바이너리
Linux용 수동 설치 바이너리를 직접 다운받습니다.
```bash
curl -L -o ollama-linux-amd64 https://ollama.com/download/ollama-linux-amd64
```

### 4. Qwen2.5:7b (LLM) 모델 파일
외부 망 PC에 임시로 Ollama를 설치하고, 터미널에서 모델을 다운받은 뒤 모델 통 파일을 통째로 복사해옵니다.
```bash
# 외부 PC에서 모델 다운로드
ollama pull qwen2.5:7b

# 다운로드가 완료되면 해당 캐시(모델) 폴더를 통째로 USB로 복사합니다.
# Linux의 경우 보통 ~/.ollama/models 혹은 /usr/share/ollama/.ollama/models 에 위치.
cp -r ~/.ollama/models /path/to/USB/ollama_models
```

### 5. STT 모델 (faster-whisper/small)
Hugging Face 서버 망 분리에 대비해, CTranslate2 포맷으로 변환되어 있는 STT 모델 파일을 다운받습니다.
- 접속 위치: `https://huggingface.co/Systran/faster-whisper-small` (브라우저 또는 git lfs 환경)
- `.bin`, `.json`, `.txt` 확장자를 가진 이 레포지토리 안의 **모든 파일**을 다운받아 `models/faster-whisper-small` 폴더에 담습니다.

### ✅ USB 최종 준비물 폴더 트리 체크
망가져서 서버실 밖으로 두 번 나오지 않기 위해 아래 트리 구조가 다 있는지 체크하세요.
```text
USB_DRIVE/
├── wheels/                         # pip -d 로 구운 .whl 패키지 뭉치들
├── ffmpeg-release-amd64-static.tar.xz # 정적 우분투 데비안계 용 FFmpeg
├── ollama-linux-amd64              # Ollama 실행 파일
├── ollama_models/                  # Ollama LLM 모델 데이터 블롭 (blobs, manifests 등)
└── models/
    └── faster-whisper-small/       # 다운받은 STT 엔진 모델 파일 조각들
```

---

## 🛠️ 2단계: 폐쇄망 Linux 서버 내부 설치 (서버실 작업)

### 1. 소스코드 및 파일 복사
USB에 담아온 파일들을 서버 내 특정 공간(예: `/opt/stt_poc`)으로 모두 복사합니다.

### 2. FFmpeg 수동 설치 및 PATH 등록
압축을 풀고 어느 경로에서나 실행할 수 있도록 바이너리 디렉토리로 이동시킵니다.
```bash
tar -xf /path/to/USB/ffmpeg-release-amd64-static.tar.xz

# 압축풀린 디렉토리로 이동
cd ffmpeg-*-static

# bin 유틸에 덮어쓰기 권한
sudo cp ffmpeg ffprobe /usr/local/bin/

# 잘 동작하는지 확인 (버전 메시지 출력 시 정상)
ffmpeg -version
```

### 3. Python 가상환경 및 오프라인 패키지 빌드
외부망을 바라보지 않도록 `--no-index` 플래그를 사용해 로컬 폴더 기준으로 설치합니다.
```bash
cd /opt/stt_poc
python3 -m venv .venv
source .venv/bin/activate

# 주의: 밖(외부)을 찾지 않고 USB경로 안경로만 찾아가며 의존성을 맞춥니다.
pip install --no-index --find-links=/path/to/USB/wheels -r requirements.txt
pip install --no-index --find-links=/path/to/USB/wheels faster-whisper
```

### 4. Ollama 백그라운드 서버 구동 및 모델 이식
권한 부여 후 구동 및 모델의 원래 위치에 USB의 모델 파일 트리를 그대로 이식합니다.
```bash
# 실행 파일 시스템 위치로 이동 (전역 실행)
sudo cp /path/to/USB/ollama-linux-amd64 /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama

# 현재 로그인한 구동 계정 홈 디렉토리에 빈 모델 폴더 생성 후 복사
mkdir -p ~/.ollama/models
cp -R /path/to/USB/ollama_models/* ~/.ollama/models/

# 백그라운드로 서버 리스닝
ollama serve &

# 엔터 한번 쳐서 백그라운드 떨구고, 리스트 뱉어내는지 확인
ollama list
```
`NAME: qwen2.5:7b` 등의 목록이 화면에 뜬다면 LLM 모델 이식 완벽 성공입니다.

### 5. config.yaml 경로 로컬 바라보게 수정
앱이 인터넷(HuggingFace)에서 모델을 다운받으려 하지 않고, 로컬 USB에서 가져다 복사해 둔 경로를 쓰도록 강제합니다.
```yaml
stt:
  engine: "faster"
  model_name: "small"
  device: "cpu"
  model_path: "/opt/stt_poc/models/faster-whisper-small" # 반드시 절대 경로 지정
  speaker_separation: true     # 화자 분리 켜기
```

### 6. 서버 최종 구동 및 검증
```bash
# Uvicorn 런
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
방화벽 8000 포트가 비어있다면 외부 네트워크 브라우저에서 서버 IP를 입력하여 데모(PoC) 화면이 뜬다면 성공입니다.
