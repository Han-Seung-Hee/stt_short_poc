"""STT → 요약 통합 파이프라인.

음성 파일 입력부터 최종 요약까지의 전체 흐름을 오케스트레이션합니다.
메모리 관리를 위해 STT와 LLM을 순차적으로 실행합니다.
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import AppConfig
from app.stt.base import STTEngine, STTResult
from app.llm.base import LLMEngine, SummaryResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """통합 파이프라인 처리 결과."""
    # STT 결과
    transcript: str = ""
    segments: list = field(default_factory=list)
    stt_model: str = ""
    stt_duration_sec: float = 0.0

    # 요약 결과
    summary: str = ""
    llm_model: str = ""
    llm_duration_sec: float = 0.0

    # 전체 처리
    total_duration_sec: float = 0.0
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """API 응답용 딕셔너리로 변환합니다."""
        return {
            "status": self.status,
            "transcript": self.transcript,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in self.segments
            ],
            "summary": self.summary,
            "model_info": {
                "stt": self.stt_model,
                "llm": self.llm_model,
            },
            "processing_time": {
                "stt_sec": round(self.stt_duration_sec, 2),
                "llm_sec": round(self.llm_duration_sec, 2),
                "total_sec": round(self.total_duration_sec, 2),
            },
            "error": self.error,
        }


class ProcessingPipeline:
    """STT → LLM 요약 통합 파이프라인.

    메모리 관리:
    - STT 처리 완료 후 gc.collect() 호출 (Unified Memory 반환 유도)
    - LLM은 Ollama 별도 프로세스에서 실행되므로 Python 메모리와 분리
    """

    def __init__(self, stt_engine: STTEngine, llm_engine: LLMEngine, config: AppConfig):
        self.stt_engine = stt_engine
        self.llm_engine = llm_engine
        self.config = config

    def process(
        self,
        audio_path: str,
        language: Optional[str] = None,
        chunk_enabled: Optional[bool] = None,
        chunk_length_sec: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> PipelineResult:
        """WAV 파일을 STT → 요약까지 처리합니다.

        Args:
            audio_path: WAV 파일 경로.
            language: 언어 코드 (None이면 설정값 사용).
            chunk_enabled: 청크 분할 여부 (None이면 설정값 사용).
            chunk_length_sec: 청크 길이(초) (None이면 설정값 사용).
            system_prompt: 커스텀 요약 프롬프트.

        Returns:
            PipelineResult: 통합 처리 결과.
        """
        result = PipelineResult()
        total_start = time.time()

        # 매개변수 기본값 적용
        lang = language or self.config.stt.language
        use_chunk = chunk_enabled if chunk_enabled is not None else self.config.stt.chunk.enabled
        chunk_sec = chunk_length_sec or self.config.stt.chunk.length_sec

        # === 1단계: STT ===
        try:
            logger.info(f"=== STT 시작: {audio_path} (lang={lang}, chunk={use_chunk}) ===")

            stt_result: STTResult = self.stt_engine.transcribe(
                audio_path=audio_path,
                language=lang,
                chunk_enabled=use_chunk,
                chunk_length_sec=chunk_sec,
            )

            result.transcript = stt_result.text
            result.segments = stt_result.segments
            result.stt_model = stt_result.model_name
            result.stt_duration_sec = stt_result.duration_sec

            logger.info(f"STT 완료: {len(stt_result.text)}자 인식")

        except Exception as e:
            logger.error(f"STT 실패: {e}", exc_info=True)
            result.status = "error"
            result.error = f"STT 처리 실패: {str(e)}"
            result.total_duration_sec = time.time() - total_start
            return result

        # STT 후 메모리 정리 (Unified Memory 반환 유도)
        gc.collect()
        logger.debug("STT 후 GC 수행 완료")

        # === 2단계: 요약 ===
        if not stt_result.text.strip():
            logger.warning("STT 결과가 비어있어 요약을 건너뜁니다.")
            result.summary = "(인식된 텍스트가 없습니다)"
            result.total_duration_sec = time.time() - total_start
            return result

        try:
            logger.info(f"=== 요약 시작: {len(stt_result.text)}자 입력 ===")

            summary_result: SummaryResult = self.llm_engine.summarize(
                text=stt_result.text,
                system_prompt=system_prompt,
            )

            result.summary = summary_result.summary
            result.llm_model = summary_result.model_name
            result.llm_duration_sec = summary_result.elapsed_sec

            logger.info(f"요약 완료: {len(summary_result.summary)}자")

        except Exception as e:
            logger.error(f"요약 실패: {e}", exc_info=True)
            result.status = "partial"
            result.error = f"요약 처리 실패 (STT는 성공): {str(e)}"

        result.total_duration_sec = time.time() - total_start

        logger.info(
            f"=== 파이프라인 완료: "
            f"STT {result.stt_duration_sec:.1f}s + "
            f"LLM {result.llm_duration_sec:.1f}s = "
            f"총 {result.total_duration_sec:.1f}s ==="
        )

        return result
