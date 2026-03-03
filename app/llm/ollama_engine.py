"""Ollama 기반 LLM 요약 엔진.

로컬 Ollama REST API를 호출하여 텍스트를 3줄로 요약합니다.
Mac/Linux 양쪽에서 동일하게 동작합니다.
"""

import logging
import time
import re
from typing import Optional, List

import httpx

from app.llm.base import LLMEngine, SummaryResult

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """당신은 전문 요약 AI입니다.
사용자가 제공하는 고객 상담 통화 내용을 읽고, 반드시 지정된 JSON 형식으로만 대답해 주세요.
절대 다른 언어(특히 중국어)를 사용하지 말고, 오직 한국어(Korean)로만 요약하세요.

[지정된 JSON 양식]
{
  "full_summary": "전체 대화의 맥락과 주요 과정을 2~3줄로 요약",
  "three_lines": [
    "핵심사항 1",
    "핵심사항 2",
    "핵심사항 3"
  ],
  "is_profanity": false,
  "profanity_words": []
}

규칙:
1. 답변은 오직 JSON 형식으로만 작성해야 하며, 백틱(```) 등 불필요한 문구를 넣지 마세요.
2. 고객이 욕설이나 비속어를 썼다면 is_profanity 를 true 로, 해당 단어를 profanity_words 리스트에 넣으세요.
"""

FEW_SHOT_SYSTEM_PROMPT = """당신은 고객 상담 통화 내용 전문 요약 AI입니다.
사용자가 제공하는 통화 내용을 읽고, 반드시 아래의 JSON 형식으로만 대답해 주세요. 그 외 설명이나 마크다운 백틱(```json)을 붙이지 마세요.

형식:
{
  "full_summary": "전체 대화 내용을 2~3줄로 이어서 작성한 줄거리 (비속어가 감지되면 해당 내용도 포함)",
  "three_lines": [
    "[주제] 불만 사항 요약 등",
    "[상담원 대응] 해결책이나 상황 설명 요약",
    "[결론] 합의된 조치 요약"
  ],
  "is_profanity": false,
  "profanity_words": []
}

규칙:
1. 답변은 오직 위 JSON 형식으로만 작성해야 하며, 다른 여분의 텍스트를 포함하지 마세요.
2. full_summary 에는 대화의 시작부터 끝까지 전체 상황을 자연스럽게 요약합니다.
3. three_lines 에는 명사형 종결이 아닌, 각 줄마다 구체적인 문장(또는 구절)으로 명확히 추출합니다. 절대로 단순 단어(예: "욕설")만 달랑 적지 마세요.
4. 고객이 욕설, 비속어, 협박성 발언을 했다면 is_profanity 를 true 로, 해당 단어들을 profanity_words 리스트에 넣으세요.
5. 반드시 오직 한국어(Korean)로만 작성하세요. 중국어 등 타 언어 사용은 절대 금지됩니다.
"""

class OllamaEngine(LLMEngine):
    """Ollama REST API를 사용하는 LLM 요약 엔진.

    Ollama가 localhost에서 실행 중이어야 합니다.
    Mac: `ollama serve` 또는 Ollama.app 실행
    Linux: `ollama serve`
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b",
        timeout_sec: int = 3600,
        max_retries: int = 3,
        prompt_type: str = "default",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.prompt_type = prompt_type

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

        # 정제(refine) 단계 호출인지 확인 (system_prompt에 '교정기' 등의 키워드가 있는지 확인)
        is_refine = system_prompt and "교정기" in system_prompt

        if is_refine:
            prompt = system_prompt
            user_message = f"다음 주어진 텍스트의 오탈자만 원문 길이를 유지하며 교정해서 다시 출력해 주세요:\n\n{text}"
            num_predict = 4096  # 전체 텍스트 출력을 위해 토큰 수 대폭 확장
        else:
            # 설정(config)에 따른 프롬프트 분기
            if self.prompt_type == "few_shot":
                prompt = FEW_SHOT_SYSTEM_PROMPT
                user_message = f"다음 고객 상담 통화 내용을 [지정된 양식]에 맞춰 완벽하게 요약해 주세요:\n\n{text}"
            else:
                prompt = DEFAULT_SYSTEM_PROMPT
                user_message = f"다음 고객 상담 통화 내용을 핵심 3줄로 요약해 주세요:\n\n{text}"
            num_predict = 512

        start_time = time.time()

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": "json" if not is_refine else None,
            "options": {
                "temperature": 0.3,       # 낮은 temperature로 일관된 교정/요약
                "top_p": 0.9,
                "num_predict": num_predict,
                "num_ctx": 4096,          # 긴 문서 처리를 위해 컨텍스트 윈도우 확보
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

                if len(summary_text) == 0:
                    raise ValueError("Ollama가 빈 문자열(0자)을 반환했습니다. 컨텍스트 초과 또는 모델 로딩 오류일 수 있습니다.")
                    
                # 강력한 중국어 필터링 (발견 시 즉각 에러 발생시켜 클라이언트에 노출 차단)
                if re.search(r'[\u4e00-\u9fff]', summary_text):
                    logger.error("🚨 [치명적 오류] 요약 결과에 중국어가 감지되었습니다. 출력을 강제 차단합니다.")
                    raise ValueError("환각(Hallucination) 감지. 엄격한 차단 정책에 의해 응답이 폐기되었습니다.")

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

    def list_models(self) -> List[str]:
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
