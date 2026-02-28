# Linux 실행 가이드 (저사양 테스트 서버용)

이 가이드는 GPU가 없거나 스펙이 낮은 Linux 테스트 서버 상에서 PoC를 어떻게 구동하는지 설명합니다. Linux 환경에서는 Apple Silicon 전용인 `mlx-whisper` 대신, CPU 최적화가 탁월한 **`faster-whisper`** 기반으로 엔진을 교체하여 실행합니다.

---

## 1. 사전 준비물 및 요구사항
- 운영체제: Ubuntu / CentOS 등 일반 상용 Linux
- 메모리: 최소 8GB 이상 (권장 16GB)
- CPU: 멀티코어 권장 (스레드 수가 속도에 직결됨)
- Python 3.10 또는 3.11 이상 설치됨

---

## 2. 엔진 설정 교체 (`config.yaml`)

테스트 서버 환경에 맞게 `config.yaml`의 엔진 설정을 **`mlx`에서 `faster`로 변경**해야 합니다.

```yaml
stt:
  engine: "faster"           # << 기존 "mlx"를 "faster"로 변경
  model_name: "small"        # 사양이 아주 낮다면 "base"나 "tiny"로 하향 조정 권장
  language: "ko"
  device: "cpu"              # ⭐ NVIDIA GPU가 있는 서버라면 "cuda"로 변경하세요!
  speaker_separation: true   # 스테레오 파일인 경우 켜면 화자 분리 진행
  chunk:
    enabled: false
```

- **팁:** 저사양 서버의 경우, 요약을 제외하고 단순 STT 테스트만 원한다면, 모델 크기를 `base`로 내려서 속도를 챙기는 것도 방법입니다.
- **GPU 팁:** NVIDIA GPU가 장착된 서버라면 `device: "cuda"`로 변경 시 STT 처리 속도가 **10배 이상 비약적으로 상승**하며, 모델 크기를 `large-v3`로 올려 완벽한 정확도를 노릴 수 있습니다.

---

## 3. Linux 서버 패키지 설치

### 3-1. 시스템 패키지 설치 (오디오 처리용 FFmpeg)
pydub가 오디오를 분할(화자 분리 처리)하려면 OS 단에 `ffmpeg`가 설치되어 있어야 합니다.
```bash
# Ubuntu의 경우
sudo apt-get update
sudo apt-get install ffmpeg -y

# CentOS의 경우
sudo yum install ffmpeg -y
```

### 3-2. Python 패키지 설치
이미 프로젝트에 `requirements.txt`가 있지만, Linux 전용 엔진인 `faster-whisper`를 명시적으로 추가 설치해주면 좋습니다.
```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 기본 패키지 설치
pip install -r requirements.txt

# Linux용 외부 엔진 설치 (CPU 사용 시 기본)
pip install faster-whisper

# ⭐ 만약 NVIDIA GPU 서버라면 추가 모듈이 필요합니다.
# requirements.txt 파일 하단의 PyTorch(GPU 가속) 부분 주석을 해제한 후 설치하세요.
```

**[NVIDIA GPU 셋업 추가 정보]**
Linux에 GPU가 달려있는데 `device: "cuda"` 설정 시 라이브러리 로드 에러(libcudnn 등)가 발생하는 경우가 꽤 많습니다. 이를 미리 방어하기 위해, `requirements.txt` 하단에 명시된 `torch`, `torchvision`, `torchaudio` (CUDA 11.8 호환) 패키지들의 주석을 해제하고 함께 설치하시는 것을 강력히 권장합니다.

---

## 4. Ollama (LLM) 리눅스 설치
서버 상단 메뉴바가 없는 Linux 환경에서는 커맨드라인으로 백그라운드 서비스를 올립니다.

```bash
# Ollama 공식 설치 스크립트 실행 (GPU가 있다면 스크립트가 알아서 인식합니다)
curl -fsSL https://ollama.com/install.sh | sh

# Ollama 백그라운드 서비스 시작 (터미널을 하나 더 열거나 서비스를 백그라운드로 띄움)
ollama serve &

# 모델 다운로드
# GPU가 빵빵하다면 qwen2.5:14b 나 32b 급 모델을 올려 요약 품질을 극대화할 수 있습니다!
ollama pull qwen2.5:7b
```

---

## 5. 서버 실행 및 접속 테스트

```bash
# FastAPI 서버 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

방화벽(UFW 등)이 있다면 8000번 포트를 열어주시고, 브라우저에서 `http://서버IP:8000/index.html`로 접속하면 Mac에서 보셨던 것과 100% 동일한 화면과 기능을 사용하실 수 있습니다!

---

## 🚨 저사양 서버 트러블슈팅 팁
1. **CPU 100% 치솟고 느림:**
   Linux CPU에서 `faster-whisper`를 돌리면 100% 풀로드를 땡깁니다. 정상입니다! 단지 시간이 Mac보다 수 배 이상 오래 걸립니다. (특히 `speaker_separation: true`로 스테레오를 두 번 돌리면 대기시간이 엄청 길어질 가능성이 높습니다.)
2. **Killed 오류 뜸 (메모리 부족):**
   RAM이 8GB 이하인 서버에서 7B 모델과 STT를 동시에 올릴 때 뻗을 수 있습니다. 이때는 `qwen2.5:7b` 모델을 `qwen2.5:3b` 등으로 낮추거나, STT 모델을 `base`로 낮추세요.
