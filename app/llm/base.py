"""LLM 요약 엔진 추상 인터페이스.

Ollama 등 다양한 LLM 백엔드를 동일한 인터페이스로 사용하기 위한 ABC.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SummaryResult:
    """요약 처리 결과."""
    summary: str            # 요약 텍스트
    model_name: str = ""    # 사용된 모델명
    prompt_tokens: int = 0  # 입력 토큰 수 (지원 시)
    eval_tokens: int = 0    # 출력 토큰 수 (지원 시)
    elapsed_sec: float = 0.0  # 처리 시간


class LLMEngine(ABC):
    """LLM 요약 엔진 추상 기반 클래스.

    모든 LLM 구현체는 이 클래스를 상속하고 summarize()를 구현해야 합니다.
    """

    @abstractmethod
    def summarize(
        self,
        text: str,
        system_prompt: Optional[str] = None,
    ) -> SummaryResult:
        """텍스트를 요약합니다.

        Args:
            text: 요약할 텍스트.
            system_prompt: 시스템 프롬프트 (None이면 기본 프롬프트 사용).

        Returns:
            SummaryResult: 요약 결과.
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
