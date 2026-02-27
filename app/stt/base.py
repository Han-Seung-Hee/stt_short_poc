"""STT 엔진 추상 인터페이스.

Mac(mlx-whisper)과 Linux(faster-whisper)를 동일한 인터페이스로 사용하기 위한 ABC.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Segment:
    """개별 음성 인식 구간."""
    start: float   # 시작 시간(초)
    end: float     # 종료 시간(초)
    text: str      # 인식된 텍스트


@dataclass
class STTResult:
    """STT 처리 결과."""
    text: str                            # 전체 인식 텍스트
    segments: list[Segment] = field(default_factory=list)
    language: str = "ko"                 # 감지/지정 언어
    duration_sec: float = 0.0            # 원본 오디오 길이(초)
    model_name: str = ""                 # 사용된 모델


class STTEngine(ABC):
    """STT 엔진 추상 기반 클래스.

    모든 STT 구현체는 이 클래스를 상속하고 transcribe()를 구현해야 합니다.
    """

    @abstractmethod
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
            language: 인식 언어 코드 (기본: "ko").
            chunk_enabled: 청크 분할 처리 여부.
            chunk_length_sec: 청크 단위 길이(초). 기본 300초(5분).

        Returns:
            STTResult: 인식 결과.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """현재 환경에서 이 엔진을 사용할 수 있는지 확인합니다."""
        ...

    @abstractmethod
    def get_engine_name(self) -> str:
        """엔진 이름을 반환합니다."""
        ...
