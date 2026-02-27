"""mlx-whisper 기반 STT 엔진 (Apple Silicon Mac 전용).

Metal 가속을 활용하여 음성을 텍스트로 변환합니다.
청크 분할 옵션을 지원하며, 긴 음성도 안정적으로 처리합니다.
"""

import gc
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from app.stt.base import STTEngine, STTResult, Segment

logger = logging.getLogger(__name__)


class MLXWhisperEngine(STTEngine):
    """mlx-whisper를 사용하는 STT 엔진.

    Apple Silicon의 Metal 가속을 활용합니다.
    오프라인 환경에서는 model_path에 로컬 모델 경로를 지정합니다.
    """

    def __init__(
        self,
        model_name: str = "small",
        model_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: Whisper 모델 크기 (tiny, base, small, medium, large-v3).
            model_path: 로컬 모델 디렉토리 경로. None이면 mlx-community 기본 경로 사용.
        """
        self.model_name = model_name
        self.model_path = model_path
        self._resolved_model = self._resolve_model_path()

    def _resolve_model_path(self) -> str:
        """모델 경로를 결정합니다.

        오프라인 환경에서는 model_path에 로컬 디렉토리를 지정해야 합니다.
        예: /path/to/models/mlx-community/whisper-small-mlx
        """
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"로컬 모델 경로 사용: {self.model_path}")
            return self.model_path

        # 온라인 환경 또는 HuggingFace 캐시 사용 시
        model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
        }
        resolved = model_map.get(self.model_name, f"mlx-community/whisper-{self.model_name}-mlx")
        logger.info(f"모델 식별자 사용: {resolved}")
        return resolved

    def transcribe(
        self,
        audio_path: str,
        language: str = "ko",
        chunk_enabled: bool = False,
        chunk_length_sec: int = 300,
    ) -> STTResult:
        """음성 파일을 텍스트로 변환합니다.

        Args:
            audio_path: WAV 파일 경로.
            language: 인식 언어 코드.
            chunk_enabled: 청크 분할 처리 여부.
            chunk_length_sec: 청크 단위 길이(초).

        Returns:
            STTResult: 인식 결과.
        """
        import mlx_whisper

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

        start_time = time.time()

        if chunk_enabled:
            result = self._transcribe_chunked(
                audio_path, language, chunk_length_sec
            )
        else:
            result = self._transcribe_single(audio_path, language)

        elapsed = time.time() - start_time
        result.duration_sec = elapsed
        result.model_name = f"mlx-whisper/{self.model_name}"

        logger.info(
            f"STT 완료: {len(result.text)}자, "
            f"{len(result.segments)}세그먼트, "
            f"{elapsed:.1f}초 소요"
        )

        return result

    def _transcribe_single(self, audio_path: str, language: str) -> STTResult:
        """단일 파일 전체를 한번에 처리합니다."""
        import mlx_whisper

        raw = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self._resolved_model,
            language=language,
            fp16=True,
            verbose=False,
        )

        segments = []
        for seg in raw.get("segments", []):
            segments.append(Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
            ))

        return STTResult(
            text=raw.get("text", "").strip(),
            segments=segments,
            language=language,
        )

    def _transcribe_chunked(
        self,
        audio_path: str,
        language: str,
        chunk_length_sec: int,
    ) -> STTResult:
        """음성을 청크로 나누어 순차 처리합니다.

        pydub를 사용하여 WAV를 분할하고, 각 청크를 개별 처리 후 결과를 병합합니다.
        """
        from pydub import AudioSegment

        logger.info(f"청크 분할 모드: {chunk_length_sec}초 단위")

        audio = AudioSegment.from_wav(audio_path)
        total_ms = len(audio)
        chunk_ms = chunk_length_sec * 1000

        all_text_parts: list[str] = []
        all_segments: list[Segment] = []
        time_offset = 0.0

        chunk_count = (total_ms + chunk_ms - 1) // chunk_ms
        for i in range(chunk_count):
            start_ms = i * chunk_ms
            end_ms = min((i + 1) * chunk_ms, total_ms)
            chunk_audio = audio[start_ms:end_ms]

            logger.info(f"청크 {i + 1}/{chunk_count} 처리 중 ({start_ms / 1000:.0f}s ~ {end_ms / 1000:.0f}s)")

            # 임시 파일로 저장하여 처리
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                chunk_audio.export(tmp.name, format="wav")
                chunk_result = self._transcribe_single(tmp.name, language)
                Path(tmp.name).unlink(missing_ok=True)

            # 세그먼트 시간 오프셋 보정
            for seg in chunk_result.segments:
                all_segments.append(Segment(
                    start=seg.start + time_offset,
                    end=seg.end + time_offset,
                    text=seg.text,
                ))

            all_text_parts.append(chunk_result.text)
            time_offset = end_ms / 1000.0

            # 청크 간 메모리 정리
            gc.collect()

        return STTResult(
            text=" ".join(all_text_parts),
            segments=all_segments,
            language=language,
        )

    def is_available(self) -> bool:
        """mlx-whisper 사용 가능 여부를 확인합니다."""
        try:
            import mlx_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def get_engine_name(self) -> str:
        return f"mlx-whisper ({self.model_name})"
