"""FastAPI 기반 STT + 요약 API 서버.

엔드포인트:
  POST /api/v1/process   — WAV 업로드 → STT + 요약 통합 반환
  POST /api/v1/transcribe — WAV 업로드 → STT 결과만 반환
  POST /api/v1/summarize  — 텍스트 입력 → 3줄 요약 반환
  GET  /api/v1/health     — 헬스체크
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config, AppConfig
from app.pipeline import ProcessingPipeline, PipelineResult
from app.stt.base import STTEngine
from app.llm.base import LLMEngine

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === 엔진 팩토리 ===

def create_stt_engine(config: AppConfig) -> STTEngine:
    """설정에 따라 적절한 STT 엔진을 생성합니다."""
    engine_type = config.stt.engine.lower()

    if engine_type == "mlx":
        from app.stt.mlx_engine import MLXWhisperEngine
        return MLXWhisperEngine(
            model_name=config.stt.model_name,
            model_path=config.stt.model_path,
        )
    elif engine_type == "faster":
        from app.stt.faster_engine import FasterWhisperEngine
        return FasterWhisperEngine(
            model_name=config.stt.model_name,
            model_path=config.stt.model_path,
        )
    else:
        raise ValueError(f"지원하지 않는 STT 엔진: {engine_type}. 'mlx' 또는 'faster'를 사용하세요.")


def create_llm_engine(config: AppConfig) -> LLMEngine:
    """설정에 따라 적절한 LLM 엔진을 생성합니다."""
    engine_type = config.llm.engine.lower()

    if engine_type == "ollama":
        from app.llm.ollama_engine import OllamaEngine
        return OllamaEngine(
            base_url=config.llm.base_url,
            model_name=config.llm.model_name,
            timeout_sec=config.llm.timeout_sec,
            max_retries=config.llm.max_retries,
        )
    else:
        raise ValueError(f"지원하지 않는 LLM 엔진: {engine_type}. 'ollama'를 사용하세요.")


# === 전역 상태 ===
pipeline: Optional[ProcessingPipeline] = None
stt_engine: Optional[STTEngine] = None
llm_engine: Optional[LLMEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 엔진을 초기화/정리합니다."""
    global pipeline, stt_engine, llm_engine

    config = get_config()
    logger.info("=" * 60)
    logger.info("  Air-Gapped STT + Summarization PoC Server")
    logger.info("=" * 60)
    logger.info(f"  STT 엔진: {config.stt.engine} (모델: {config.stt.model_name})")
    logger.info(f"  LLM 엔진: {config.llm.engine} (모델: {config.llm.model_name})")
    logger.info(f"  청크 분할: {'ON' if config.stt.chunk.enabled else 'OFF'} ({config.stt.chunk.length_sec}초)")
    logger.info(f"  서버: {config.server.host}:{config.server.port}")
    logger.info("=" * 60)

    # 엔진 생성
    stt_engine = create_stt_engine(config)
    llm_engine = create_llm_engine(config)

    # 가용성 체크
    if not stt_engine.is_available():
        logger.warning(f"⚠️  STT 엔진({stt_engine.get_engine_name()})을 사용할 수 없습니다.")
    else:
        logger.info(f"✓ STT 엔진 준비: {stt_engine.get_engine_name()}")

    if not llm_engine.is_available():
        logger.warning(
            f"⚠️  LLM 엔진({llm_engine.get_engine_name()})에 연결할 수 없습니다. "
            "Ollama가 실행 중인지 확인하세요."
        )
    else:
        logger.info(f"✓ LLM 엔진 준비: {llm_engine.get_engine_name()}")

    pipeline = ProcessingPipeline(stt_engine, llm_engine, config)
    logger.info("✓ 파이프라인 준비 완료")

    yield

    # 종료 정리
    logger.info("서버 종료 중...")
    pipeline = None
    stt_engine = None
    llm_engine = None


# === FastAPI 앱 ===
app = FastAPI(
    title="Air-Gapped STT + Summarization PoC",
    description="폐쇄망 환경에서 음성파일을 텍스트로 변환하고 3줄 요약을 제공하는 API",
    version="0.1.0",
    lifespan=lifespan,
)

# === CORS 설정 추가 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (PoC용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_wav_file(file: UploadFile) -> None:
    """업로드 파일이 WAV인지 검증합니다."""
    if file.filename and not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="WAV 파일만 지원합니다. (.wav 확장자)",
        )

    config = get_config()
    max_bytes = config.server.max_file_size_mb * 1024 * 1024
    # Content-Length 헤더가 있는 경우 사전 검증
    if file.size and file.size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기 초과: 최대 {config.server.max_file_size_mb}MB",
        )


async def _save_upload_to_temp(file: UploadFile) -> str:
    """업로드 파일을 임시 파일에 저장하고 경로를 반환합니다."""
    config = get_config()
    max_bytes = config.server.max_file_size_mb * 1024 * 1024

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        total_read = 0
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB씩 읽기
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > max_bytes:
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail=f"파일 크기 초과: 최대 {config.server.max_file_size_mb}MB",
                )
            tmp.write(chunk)
        return tmp.name


# === 엔드포인트 ===

@app.get("/api/v1/health")
async def health_check():
    """서버 및 엔진 상태를 확인합니다."""
    config = get_config()
    return {
        "status": "ok",
        "stt_engine": {
            "name": stt_engine.get_engine_name() if stt_engine else "not loaded",
            "available": stt_engine.is_available() if stt_engine else False,
        },
        "llm_engine": {
            "name": llm_engine.get_engine_name() if llm_engine else "not loaded",
            "available": llm_engine.is_available() if llm_engine else False,
        },
        "config": {
            "stt_model": config.stt.model_name,
            "llm_model": config.llm.model_name,
            "chunk_enabled": config.stt.chunk.enabled,
        },
    }


@app.post("/api/v1/process")
async def process_audio(
    file: UploadFile = File(..., description="WAV 음성 파일"),
    language: str = Form(default="ko", description="언어 코드"),
    chunk_enabled: Optional[bool] = Form(default=None, description="청크 분할 활성화 여부"),
    chunk_length_sec: Optional[int] = Form(default=None, description="청크 길이(초)"),
):
    """WAV 파일을 STT + 요약까지 통합 처리합니다.

    - WAV 파일 업로드
    - mlx-whisper로 음성 인식 (STT)
    - Ollama LLM으로 3줄 요약
    - 통합 결과 JSON 반환
    """
    _validate_wav_file(file)

    if pipeline is None:
        raise HTTPException(status_code=503, detail="파이프라인이 초기화되지 않았습니다.")

    tmp_path = await _save_upload_to_temp(file)

    try:
        result: PipelineResult = pipeline.process(
            audio_path=tmp_path,
            language=language,
            chunk_enabled=chunk_enabled,
            chunk_length_sec=chunk_length_sec,
        )

        status_code = 200 if result.status == "success" else 207  # 207 = partial
        return JSONResponse(content=result.to_dict(), status_code=status_code)

    except Exception as e:
        logger.error(f"처리 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

    finally:
        # 임시 파일 정리
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/v1/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="WAV 음성 파일"),
    language: str = Form(default="ko", description="언어 코드"),
    chunk_enabled: Optional[bool] = Form(default=None, description="청크 분할 활성화 여부"),
    chunk_length_sec: Optional[int] = Form(default=None, description="청크 길이(초)"),
):
    """WAV 파일을 텍스트로 변환합니다 (STT만 수행)."""
    _validate_wav_file(file)

    if stt_engine is None:
        raise HTTPException(status_code=503, detail="STT 엔진이 초기화되지 않았습니다.")

    config = get_config()
    tmp_path = await _save_upload_to_temp(file)

    try:
        use_chunk = chunk_enabled if chunk_enabled is not None else config.stt.chunk.enabled
        chunk_sec = chunk_length_sec or config.stt.chunk.length_sec

        stt_result = stt_engine.transcribe(
            audio_path=tmp_path,
            language=language,
            chunk_enabled=use_chunk,
            chunk_length_sec=chunk_sec,
        )

        return {
            "status": "success",
            "transcript": stt_result.text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in stt_result.segments
            ],
            "language": stt_result.language,
            "model": stt_result.model_name,
            "duration_sec": round(stt_result.duration_sec, 2),
        }

    except Exception as e:
        logger.error(f"STT 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT 처리 중 오류 발생: {str(e)}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/v1/summarize")
async def summarize_text(
    text: str = Form(..., description="요약할 텍스트"),
    system_prompt: Optional[str] = Form(default=None, description="커스텀 시스템 프롬프트"),
):
    """텍스트를 3줄로 요약합니다 (LLM만 수행)."""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM 엔진이 초기화되지 않았습니다.")

    try:
        summary_result = llm_engine.summarize(
            text=text,
            system_prompt=system_prompt,
        )

        return {
            "status": "success",
            "summary": summary_result.summary,
            "model": summary_result.model_name,
            "tokens": {
                "prompt": summary_result.prompt_tokens,
                "eval": summary_result.eval_tokens,
            },
            "duration_sec": round(summary_result.elapsed_sec, 2),
        }

    except Exception as e:
        logger.error(f"요약 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"요약 처리 중 오류 발생: {str(e)}")


# === 직접 실행 지원 ===
if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,  # 프로덕션에서는 False
        log_level="info",
    )
