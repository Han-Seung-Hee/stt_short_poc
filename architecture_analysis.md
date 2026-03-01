# 프로젝트 구조 및 데이터 흐름 분석 (개인 공부용)

이 프로젝트는 **관심사의 분리(Separation of Concerns)**와 **의존성 주입(Dependency Injection)** 패턴을 잘 따르고 있어, 파이썬 기반 백엔드 아키텍처를 학습하기에 매우 좋은 구조를 가지고 있습니다.

---

## 1. 프로젝트 폴더 구조 핵심 요약

프로젝트의 뼈대는 큼직하게 **실행 진입점, 설정 관리, 파이프라인(오케스트레이터), 개별 엔진(STT/LLM)** 으로 나뉩니다.

```text
├── config.yaml               # (최상위) 서버 구동에 필요한 모든 설정값을 담은 텍스트 파일
├── app/
│   ├── main.py               # [진입점] FastAPI 서버 정의, API 엔드포인트(URL) 라우팅
│   ├── config.py             # [설정 관리] config.yaml을 파이썬 객체로 파싱하고 관리
│   ├── pipeline.py           # [오케스트레이터] STT와 LLM을 조합해 순서대로 실행하는 감독관
│   ├── stt/                  # (기능 1: STT 엔진 모음)
│   │   ├── base.py           # STT 엔진들이 무조건 지켜야 할 규칙(인터페이스) 정의
│   │   ├── mlx_engine.py     # Mac 전용 STT 엔진 (Apple Metal 가속)
│   │   └── faster_engine.py  # Linux 전용 STT 엔진 (CPU/NVIDIA GPU 가속)
│   └── llm/                  # (기능 2: LLM 엔진 모음)
│       ├── base.py           # LLM 엔진들이 무조건 지켜야 할 규칙(인터페이스) 정의
│       └── ollama_engine.py  # 로컬 통신 기반 Ollama 요약 엔진
```

---

## 2. 설정 파일(`config.yaml`)을 읽어오는 상세 원리

설정을 담당하는 곳은 `app/config.py`입니다. 이 파일은 서버가 구동될 때 가장 먼저 실행되어 중추적인 역할을 합니다.

1. **`dataclass` 정의:** `config.py` 안에는 `STTConfig`, `LLMConfig` 같은 클래스들이 정의되어 있습니다. (이 코드는 `config.yaml`의 구조와 1:1 매칭됩니다.)
2. **`load_config()` 함수 (파싱 및 오버라이드):**
   * 서버가 시작될 때 이 함수가 호출되어 디스크에 있는 `config.yaml` 파일을 읽어옵니다. (`yaml.safe_load()`)
   * **환경변수 덮어쓰기:** 읽어온 직후, `_apply_env_overrides()` 함수를 통해 OS의 환경변수(예: `export STT_ENGINE=faster`)가 있는지 체크합니다. 만약 있다면, yaml에 적힌 값보다 **환경변수 값을 우선적으로 덮어씁니다.** (도커나 클라우드 배포 시 매우 유용한 패턴입니다.)ㅍ
   * 최종적으로 파싱된 딕셔너리를 `AppConfig`라는 하나의 파이썬 객체(Data Class)로 예쁘게 포장해서 반환합니다.
3. **`get_config()` 함수 (싱글톤 패턴):**
   * 이렇게 만들어진 `AppConfig` 객체는 전역 변수 `_config`에 캐싱(저장)됩니다.
   * 이후 `main.py`나 `pipeline.py` 등 어디서든 설정을 원할 때 `get_config()`를 호출하는데, 매번 파일을 읽는 게 아니라 **맨 처음 캐싱해 둔 객체를 그대로 반환(싱글톤)**하여 디스크 I/O 속도 저하를 막습니다.

---

## 3. 서버 기동 시 동작 플로우 (Startup Flow)

터미널에서 `uvicorn app.main:app --host 0.0.0.0 --port 9123`을 입력했을 때, `app/main.py` 내부에서 어떤 일이 일어나는지 살펴봅시다.

1. **FastAPI 앱 객체 생성:** `app = FastAPI(...)` 코드가 실행되며 빈 서버 껍데기를 만듭니다.
2. **설정 로드:** `config = get_config()`가 호출되어 앞서 설명한 `config.yaml`을 메모리에 올립니다.
3. **의존성 팩토리 함수 실행 (\@app.on_event("startup")):** (FastAPI의 최신 방식인 `lifespan` 혹은 시작 스크립트로 구성됨)
   * `create_stt_engine(config)`: config의 `stt.engine` 값(mlx인지 faster인지)을 보고 **다형성(Polymorphism)**을 활용해 적절한 파이썬 엔진 클래스를 메모리에 띄웁니다.
   * `create_llm_engine(config)`: 동일한 방식으로 Ollama 연결 객체를 만듭니다.
4. **파이프라인 결합:** 생성된 `stt_engine`과 `llm_engine`, 그리고 `config`를 인자로 넣어서 `ProcessingPipeline(stt_engine, llm_engine, config)` 오케스트레이터 객체를 생성해 전역(`global app_pipeline`)에 대기시킵니다.
5. **리스닝 상태 전환:** 서버가 9123포트를 열고 `POST /api/v1/process` 요청이 오기를 기다립니다.

---

## 4. API 요청 유입 시 동작 플로우 (Request Flow)

사용자가 웹(index.html)에서 파일을 올리고 "처리 시작"을 누르면, FastAPI의 라우터 단에서부터 물 흐르듯 코드가 실행됩니다.

### ① 요청 수신 (`app/main.py`의 `@app.post("/api/v1/process")` 함수)
*   FastAPI가 올라온 `.wav` 파일을 받아 임시 파일(`UploadFile`) 형태로 저장합니다.
*   요청 파라미터(language, chunk 활성화 여부 등)를 파싱합니다.

### ② 파이프라인에 작업 지시 (`app/main.py` -> `app/pipeline.py`)
*   `main.py`는 자기가 다 처리하지 않고, 아까 띄워둔 전담 매니저인 `app_pipeline.process(audio_path, ...)` 함수를 호출하여 일을 던집니다.

### ③ STT 처리 (`app/pipeline.py` -> `app/stt/mlx_engine.py`)
*   파이프라인 입장에서는 그게 mlx인지 faster인지 알 바 아닙니다. 그냥 `base.py` 인터페이스에 적힌 대로 `self.stt_engine.transcribe(...)`를 호출할 뿐입니다.
*   `mlx_engine.py` 내부에서:
    *   `config.yaml`에 `speaker_separation: true`로 되어 있다면, `pydub`를 써서 스테레오를 좌/우로 찢습니다.
    *   `mlx-whisper` 모델을 호출해 음성을 텍스트로 뽑아내고 메모리에 올립니다.
*   뽑힌 텍스트와 메타데이터가 `STTResult` 객체로 포장되어 파이프라인으로 반환됩니다.

### ④ 요약 처리 (`app/pipeline.py` -> `app/llm/ollama_engine.py`)
*   파이프라인은 앞선 STT의 결과(`STTResult.text`)를 받아서, 이번엔 `self.llm_engine.summarize(text)`를 호출합니다.
*   `ollama_engine.py` 내부에서:
    *   `config.yaml`에 `prompt_type`이 "few_shot"으로 설정되어 있다면, 시스템 프롬프트(상담원/고객 분류 규칙 및 예시)를 덧붙입니다.
    *   `httpx` (HTTP 클라이언트) 라이브러리를 통해 로컬의 `http://localhost:11434` 포트에 떠있는 Ollama 프로그램에 비동기로 REST API 요청을 쏩니다.
    *   Ollama가 생성해 준 답변을 받아냅니다.

### ⑤ 결과 포장 및 응답 (`app/pipeline.py` -> `app/main.py`)
*   파이프라인이 STT 결과와 LLM 요약 결과를 합쳐서 `PipelineResult`라는 예쁜 바구니에 담아 `main.py`로 돌려줍니다.
*   `main.py`는 이를 프론트엔드가 이해하기 쉬운 `JSON` 포맷으로 최종 변환하여 HTTP 응답(Response 200 OK)을 반환합니다.

이렇게 **설정 -> 진입점 -> 파이프라인 매니저 -> 개별 실무자(엔진)** 구조로 역할이 뚜렷하게 나뉘어 있기 때문에, 유지보수 시 STT 엔진을 뜯어고친다고 해서 메인 API 코드나 요약 코드를 건드릴 필요가 없는 좋은 구조입니다!
