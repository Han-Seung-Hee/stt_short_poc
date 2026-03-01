# Air-Gapped STT + Summarization PoC

폐쇄망(Air-gapped) Apple Silicon Mac 및 Linux 환경에서 고객 상담 음성파일(.wav)을 텍스트로 변환(STT)하고, 핵심 내용을 3줄로 요약하는 오프라인 AI API 서비스입니다.

> [!CAUTION]
> **[필독] 폐쇄망 환경 Windows OS (Server 및 10/11) 배포 불가 안내**
> 본 오프라인 AI 파이프라인은 최대한의 성능과 안정성을 위해 Linux 및 MacOS 환경을 표준으로 설계되었습니다. 고객사 환경이 망이 분리된 **Windows Server 계열(2019/2022 등) 또는 일반 데스크톱 Windows OS(10, 11 등)일 경우, 다음의 치명적인 이슈들로 인해 시스템 구성 및 배포가 원천적으로 불가합니다.**
> 
> 1. **의존성 문제 (DLL 충돌):** 수 GB에 달하는 C++ 빌드 환경과 AI 수학 연산(PyTorch, CTranslate2) 라이브러리들을 윈도우 오프라인망에 수동으로 밀어넣는 과정에서 높은 확률로 환경변수 꼬임 및 DLL 충돌이 발생합니다.
> 2. **서버 셧다운 (좀비 프로세스):** 화자 분리(L/R)를 수행하는 핵심 코어인 FFmpeg가 Windows 데몬(서비스) 환경에서는 좀비 프로세스로 남아 누적되어 메인 서버 메모리 부족으로 인한 시스템 중단이 100% 발생합니다.
> 3. **성능 저하 문제 (도커 오버헤드):** Docker에서 구동하는 경우, Windows의 극한 가상화 층(Hyper-V 오버헤드) 때문에 AI 인퍼런스 속도가 70-90% 이상 수직 하락합니다.
> 4. **GPU 세팅 문제:** 폐쇄된 Windows에 NVIDIA CUDA 드라이버와 cuDNN 바이너리를 직접 연동(수동으로 연동해야 함)하다가 컴포넌트나 레지스트리가 꼬여 블루스크린이 발생하는 치명적 위험이 존재합니다.
> 
> **💡 결론:** 글로벌 오픈소스 AI 생태계의 표준은 Linux입니다. 시스템 안정성과 처리 속도를 조금도 보장할 수 없는 Windows(Server/PC 등) 환경은 일절 지원하지 않으며, 정상적인 작동을 위해 **반드시 운영체제가 Linux(Ubuntu, CentOS 등)인 서버나 PC를 할당**받으시기 바랍니다.

## 아키텍처

```
                    ┌────────────────────────────-─┐
                    │       FastAPI Server         │
                    │       (port 9123)            │
                    │                              │
  WAV Upload ─────▶ │  POST /api/v1/process        │
                    │       │                      │
                    │       ▼                      │
                    │  ┌──────────────┐            │
                    │  │  STT Engine  │ mlx-whisper│
                    │  │(스테레오 분리)│ pydub        │
                    │  └──────┬───────┘            │
                    │         │ transcript         │
                    │         ▼                    │
                    │  ┌──────────────┐            │
                    │  │  LLM Engine  │ Ollama     │
                    │  │  (3줄 요약)   │ qwen2.5:7b |
                    │  └──────┬───────┘            │
                    │         │ summary            │
                    │         ▼                    │
                    │   JSON Response ─────────────┼──▶ { transcript, summary }
                    └─────────────────────────────-┘
```


## 관련 문서
- `offline_setup_guide.md`: Mac 환경 (폐쇄망) 오프라인 설치 가이드
- `linux_execution_guide.md`: 인터넷이 되는 Linux (테스트/운영 서버) 스펙 및 세팅 가이드
- `linux_offline_setup_guide.md`: **[필독] 완전 폐쇄망(Air-Gapped) Linux 서버 수동 설치 가이드**
- `java_call_guide.md`: 외부 시스템(Java/Spring Boot) 타임아웃 및 API 연동 가이드
- `architecture.drawio`: 시스템 아키텍처 다이어그램 (draw.io 파일)

## 빠른 시작

### 사전 요건
- Python 3.11+
- Ollama 실행 중 (`ollama serve`)
- Qwen2.5 7B 모델 로드 완료 (명령어: `ollama run qwen2.5:7b`)

### 설치 및 실행

**1. 파이썬 가상환경 설정 및 의존성 설치**
```bash
# 가상환경 생성 (최초 1회)
python3 -m venv .venv

# 가상환경 활성화 (필수)
source .venv/bin/activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

**2. AI Hub 데이터 테스트 준비 (옵션)**
AI Hub 데이터처럼 여러 개로 쪼개진 모노 채널 데이터 파일을 가지고 있다면, 테스트를 위해 하나로 합쳐야 합니다. (이 작업은 PoC 테스트 정확도 평가를 위해 필요합니다)
```bash
# base_data 폴더 안에 있는 특정 통화 ID 데이터 병합 및 정답지 추출
python prepare_data.py MEN0005946

# 실행 완료 후 test_ready 폴더에 병합된 WAV 파일과 대본(TXT) 파일이 생성됩니다.
```

**3. 백그라운드 LLM 실행 확인**
*   로컬 Mac 환경에서 `Ollama` 애플리케이션이 실행 중인지 확인합니다. (상단 메뉴바 등)
*   사용할 모델(`qwen2.5:7b` 등)이 이미 `ollama run`을 통해 로드 가능한 상태여야 합니다.

**4. STT + 요약 API 서버 시작**
가상환경(`(.venv)`)이 활성화된 터미널에서 아래 명령어로 서버를 가동합니다.
```bash
kill -9 $(lsof -t -i:9123) 2>/dev/null; .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 9123
```
*`Application startup complete.` 문구가 뜨면 서버 가동 완료입니다.*

**5. 웹 브라우저에서 클라이언트 열기**
서버가 켜진 상태에서, 터미널(새 탭) 또는 Finder를 통해 `index.html` 파일을 열어 시연을 진행합니다.
```bash
open index.html
```

```bash
# 헬스체크
curl http://localhost:9123/api/v1/health
```

- **통합 프로세스 (WAV → STT + 3줄 요약)**
```bash
curl -X POST http://localhost:9123/api/v1/process \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test1.wav" \
  -F "language=ko"
```

- **STT 전용 (음성 → 텍스트만 추출)**
```bash
curl -X POST http://localhost:9123/api/v1/transcribe \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test1.wav" \
  -F "language=ko"
```

- **LLM 전용 (텍스트 → 3줄 요약만)**
```bash
curl -X POST http://localhost:9123/api/v1/summarize \
  -H "accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=여기에 분석할 긴 텍스트를 입력하세요. 고객 불만 사항 등..."
```

*(선택)* 서버가 실행 상태일때 `chunk` 기반 처리를 하고 싶다면 API 요청 시 옵션을 주면 됩니다:
```bash
curl -X POST http://localhost:9123/api/v1/process \
  -F "file=@long_audio.wav" \
  -F "chunk_enabled=true" \
  -F "chunk_length_sec=300"
```

### 응답 예시

```json
{
  "status": "success",
  "transcript": "고객님 안녕하세요. 신용카드 한도 관련 문의입니다...",
  "segments": [
    {"start": 0.0, "end": 3.5, "text": "고객님 안녕하세요."},
    {"start": 3.5, "end": 8.2, "text": "신용카드 한도 관련 문의입니다."}
  ],
  "summary": "- 고객이 신용카드 한도 증액을 요청하셨습니다.\n- 상담사가 소득증빙 서류 제출이 필요하다고 안내하였습니다.\n- 고객이 3일 내 서류를 제출하기로 약속하셨습니다.",
  "model_info": {
    "stt": "mlx-whisper/small",
    "llm": "qwen2.5:7b"
  },
  "processing_time": {
    "stt_sec": 12.34,
    "llm_sec": 8.56,
    "total_sec": 20.90
  },
  "error": null
}
```

## 설정

`config.yaml`에서 모든 파라미터를 변경할 수 있으며, 환경변수로 오버라이드도 가능합니다.

| 환경변수 | 설명 | 기본값 |
|----------|------|--------|
| `STT_ENGINE` | STT 엔진 (`mlx` / `faster`) | `mlx` |
| `STT_MODEL_NAME` | Whisper 모델 크기 | `small` |
| `STT_MODEL_PATH` | 로컬 모델 경로 | (자동) |
| `STT_CHUNK_ENABLED` | 청크 분할 활성화 | `false` |
| `LLM_MODEL_NAME` | Ollama 모델명 | `qwen2.5:7b` |
| `LLM_TIMEOUT_SEC` | LLM 응답 타임아웃(초) | `120` |

## 메모리 관리 (16GB Unified Memory)

| 상태 | 메모리 사용 |
|------|-----------|
| STT (small) 단독 | ~1GB |
| STT (medium) 단독 | ~1.5GB |
| LLM (7B) 단독 | ~4.5GB |
| STT + LLM 동시 | ~6GB |
| **16GB 여유분** | **~10GB** |

### 최적화 팁
- STT와 LLM은 **순차 처리**됩니다 (파이프라인 기본 동작)
- STT 모델은 첫 요청 시 로드되고, GC를 통해 해제를 유도합니다
- Ollama는 별도 프로세스이므로 Python 메모리와 격리됩니다
- 메모리 부족 시 `config.yaml`에서 `stt.model_name`을 `base`로 축소하세요

## Linux 서버 확장

향후 GPU 없는 저사양 Linux 서버 혹은 NVIDIA GPU가 장착된 고사양 서버로 이전 및 실행하는 구체적인 방법은 **`linux_execution_guide.md`** 문서를 참고하세요.

> [!WARNING]
> **폐쇄망(Air-Gapped) 리눅스 서버 배포 시 주의사항 (담당자 필독!!)**
> 서버가 외부 인터넷과 완전히 단선된 폐쇄망 서버실에 있을 경우, 절대 "알아서 환경 잡으세요"라고 하시면 안 됩니다. (`pip install` 1줄조차 무조건 에러납니다.)
> 
> 외부망(인터넷) PC에서 다음 파일들을 **USB에 사전에 담아 들어가서 수동 설치(A to Z)** 해야 합니다.
> - **Python 패키지 (사전 빌드 .whl)**
> - **FFmpeg (정적 바이너리 타르)**
> - **Ollama 실행 파일 (바이너리 다운로드본)**
> - **LLM 모델 (qwen2.5:7b 통파일 아카이브)**
> - **STT 모델 (faster-whisper CT2 로컬 폴더)**
> 
> 인프라 담당자와의 원활한 설치 협업을 위해, 상세한 다운로드 방법 명령어 및 수동 설치 커맨드를 명시한 **`linux_offline_setup_guide.md`** 문서를 작성해 두었으니 이 문서를 전달해 주시기 바랍니다.

기본적으로(인터넷 연결 시)는 아래의 세 가지 스텝을 따릅니다:
1. OS 환경에 `ffmpeg` 설치 및 `pip install faster-whisper`
2. `config.yaml`에서 `stt.engine: "faster"` 로 변경 (NVIDIA GPU가 있다면 `stt.device: "cuda"` 추가)
3. Ollama Linux 버전 설치 및 모델 풀(`ollama pull qwen2.5:7b`)

```yaml
# config.yaml (Linux)
stt:
  engine: "faster"
  model_name: "small" # GPU 사용 시 large-v3 권장
  device: "cpu"       # GPU 사용 시 "cuda"로 변경
  model_path: "/path/to/whisper-small-ct2"
```

## 프로젝트 구조

```
stt_short/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 진입점
│   ├── config.py             # 설정 관리
│   ├── pipeline.py           # STT→요약 오케스트레이션
│   ├── stt/
│   │   ├── base.py           # STT 추상 인터페이스
│   │   ├── mlx_engine.py     # Mac: mlx-whisper
│   │   └── faster_engine.py  # Linux: faster-whisper
│   └── llm/
│       ├── base.py           # LLM 추상 인터페이스
│       └── ollama_engine.py  # Ollama REST API
├── config.yaml               # 설정 파일
├── requirements.txt          # Python 의존성
├── offline_setup_guide.md    # 오프라인 설치 가이드
├── linux_execution_guide.md  # Linux 테스트 서버 실행 가이드
├── linux_offline_setup_guide.md # 완전 폐쇄망 수동 설치 가이드
├── java_call_guide.md        # 백엔드(Java/Spring) 연동 가이드
└── README.md                 # 이 파일
```
