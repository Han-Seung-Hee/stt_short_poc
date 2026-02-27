"""Ollama 기반 LLM 요약 엔진.

로컬 Ollama REST API를 호출하여 텍스트를 3줄로 요약합니다.
Mac/Linux 양쪽에서 동일하게 동작합니다.
"""

import logging
import time
from typing import Optional

import httpx

from app.llm.base import LLMEngine, SummaryResult

logger = logging.getLogger(__name__)

# 기본 한국어 3줄 요약 시스템 프롬프트
DEFAULT_SYSTEM_PROMPT = """당신은 전문 요약 AI입니다.
사용자가 제공하는 고객 상담 통화 내용을 읽고, 핵심 내용을 정확히 3줄로 요약해 주세요.

규칙:
1. 반드시 3줄로 요약합니다. 각 줄은 하나의 핵심 포인트를 담습니다.
2. 각 줄은 "- " 로 시작합니다.
3. 고객의 요청사항, 상담사의 응대 내용, 결론/후속 조치를 포함합니다.
4. 존댓말(합쇼체)을 사용합니다.
5. 불필요한 인사말이나 감탄사는 제외합니다."""


class OllamaEngine(LLMEngine):
    """Ollama REST API를 사용하는 LLM 요약 엔진.

    Ollama가 localhost에서 실행 중이어야 합니다.
    Mac: `ollama serve` 또는 Ollama.app 실행
    Linux: `ollama serve`
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "EEVE-Korean-10.8B-v1.0:Q4_K_M",
        timeout_sec: int = 120,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def summarize(
        self,
        text: str,
        system_prompt: Optional[str] = None,
    ) -> SummaryResult:
        """텍스트를 3줄로 요약합니다.

        Args:
            text: 요약할 텍스트 (STT 결과).
            system_prompt: 커스텀 시스템 프롬프트. None이면 기본 프롬프트 사용.

        Returns:
            SummaryResult: 요약 결과.
        """
        if not text.strip():
            return SummaryResult(
                summary="(입력 텍스트가 비어있습니다)",
                model_name=self.model_name,
            )

        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_message = f"다음 고객 상담 통화 내용을 3줄로 요약해 주세요:\n\n{text}"

        start_time = time.time()

        # Ollama /api/chat 엔드포인트 사용 (대화형)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,       # 낮은 temperature로 일관된 요약
                "top_p": 0.9,
                "num_predict": 512,       # 3줄 요약이므로 짧게
            },
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Ollama 요약 요청 (시도 {attempt}/{self.max_retries}): "
                    f"모델={self.model_name}, 입력길이={len(text)}자"
                )

                with httpx.Client(timeout=self.timeout_sec) as client:
                    response = client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
                    response.raise_for_status()

                data = response.json()
                elapsed = time.time() - start_time

                summary_text = data.get("message", {}).get("content", "").strip()

                # 토큰 정보 추출 (Ollama가 제공하는 경우)
                prompt_tokens = data.get("prompt_eval_count", 0)
                eval_tokens = data.get("eval_count", 0)

                logger.info(
                    f"요약 완료: {len(summary_text)}자, "
                    f"prompt_tokens={prompt_tokens}, eval_tokens={eval_tokens}, "
                    f"{elapsed:.1f}초 소요"
                )

                return SummaryResult(
                    summary=summary_text,
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    eval_tokens=eval_tokens,
                    elapsed_sec=elapsed,
                )

            except httpx.ConnectError as e:
                last_error = e
                logger.error(
                    f"Ollama 연결 실패 (시도 {attempt}): {e}. "
                    "Ollama가 실행 중인지 확인하세요."
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"Ollama HTTP 오류 (시도 {attempt}): {e}")
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Ollama 타임아웃 (시도 {attempt}): {self.timeout_sec}초 초과. "
                    "모델 로딩 중일 수 있습니다."
                )
            except Exception as e:
                last_error = e
                logger.error(f"Ollama 예외 (시도 {attempt}): {e}")

        raise RuntimeError(
            f"Ollama 요약 실패 ({self.max_retries}회 시도): {last_error}"
        )

    def is_available(self) -> bool:
        """Ollama 서버가 응답하는지 확인합니다."""
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def get_engine_name(self) -> str:
        return f"ollama ({self.model_name})"

    def list_models(self) -> list[str]:
        """Ollama에 로드된 모델 목록을 반환합니다."""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            return []
