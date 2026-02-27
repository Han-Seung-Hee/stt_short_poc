# Air-Gapped STT + Summarization PoC

폐쇄망(Air-gapped) Apple Silicon Mac 환경에서 고객 상담 음성파일(.wav)을 텍스트로 변환(STT)하고, 핵심 내용을 3줄로 요약하는 오프라인 AI API 서비스입니다.

## 아키텍처

```
                    ┌─────────────────────────────┐
                    │       FastAPI Server         │
                    │       (port 8000)            │
                    │                              │
  WAV Upload ─────▶│  POST /api/v1/process        │
                    │       │                      │
                    │       ▼                      │
                    │  ┌──────────────┐            │
                    │  │  STT Engine  │ mlx-whisper│
                    │  │  (Metal GPU) │ (Mac)      │
                    │  └──────┬───────┘            │
                    │         │ transcript          │
                    │         ▼                     │
                    │  ┌──────────────┐            │
                    │  │  LLM Engine  │ Ollama     │
                    │  │  (3줄 요약)  │ REST API   │
                    │  └──────┬───────┘            │
                    │         │ summary             │
                    │         ▼                     │
                    │   JSON Response ─────────────┼──▶ { transcript, summary }
                    └─────────────────────────────┘
```

## 빠른 시작

### 사전 요건
- Python 3.11+
- Ollama 실행 중 (`ollama serve`)
- EEVE-Korean-10.8B 모델 로드 완료

### 설치 및 실행

```bash
# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치 (온라인)
pip install -r requirements.txt

# 또는 오프라인 설치
pip install --no-index --find-links=./wheels -r requirements.txt

# 서버 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API 사용

```bash
# 헬스체크
curl http://localhost:8000/api/v1/health

# STT + 요약 통합
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@sample.wav" \
  -F "language=ko"

# STT만
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@sample.wav"

# 요약만
curl -X POST http://localhost:8000/api/v1/summarize \
  -F "text=상담 내용 텍스트..."

# 청크 분할 활성화 (긴 음성)
curl -X POST http://localhost:8000/api/v1/process \
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
    "llm": "EEVE-Korean-10.8B-v1.0:Q4_K_M"
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
| `LLM_MODEL_NAME` | Ollama 모델명 | `EEVE-Korean-10.8B-v1.0:Q4_K_M` |
| `LLM_TIMEOUT_SEC` | LLM 응답 타임아웃(초) | `120` |

## 메모리 관리 (16GB Unified Memory)

| 상태 | 메모리 사용 |
|------|-----------|
| STT (small) 단독 | ~1GB |
| STT (medium) 단독 | ~1.5GB |
| LLM (10.8B-Q4) 단독 | ~7GB |
| STT + LLM 동시 | ~8~10GB |
| **16GB 여유분** | **~6~8GB** |

### 최적화 팁
- STT와 LLM은 **순차 처리**됩니다 (파이프라인 기본 동작)
- STT 모델은 첫 요청 시 로드되고, GC를 통해 해제를 유도합니다
- Ollama는 별도 프로세스이므로 Python 메모리와 격리됩니다
- 메모리 부족 시 `config.yaml`에서 `stt.model_name`을 `base`로 축소하세요

## Linux 서버 확장

향후 GPU 없는 Linux 서버로 이전 시:

1. `requirements.txt`에서 `mlx`, `mlx-whisper` 주석 처리 → `faster-whisper` 추가
2. `config.yaml`에서 `stt.engine: "faster"` 로 변경
3. CTranslate2 형식의 Whisper 모델로 교체
4. Ollama Linux 버전 설치 (동일한 모델 사용 가능)

```yaml
# config.yaml (Linux)
stt:
  engine: "faster"
  model_name: "small"
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
└── README.md                 # 이 파일
```
