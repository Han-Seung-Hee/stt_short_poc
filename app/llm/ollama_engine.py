"""Ollama 기반 LLM 요약 엔진.

로컬 Ollama REST API를 호출하여 텍스트를 3줄로 요약합니다.
Mac/Linux 양쪽에서 동일하게 동작합니다.
"""

import logging
import time
from typing import Optional, List

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

FEW_SHOT_SYSTEM_PROMPT = """당신은 고객 상담 통화 내용 전문 요약 AI입니다.
사용자가 제공하는 통화 내용을 읽고, 아래 [지정된 양식]에 맞춰 완벽하게 규칙대로 요약해 주세요.

[지정된 양식]
[주제]
- (고객의 주된 문의 현황 및 불만 사항 요약)
[상담사의 응대 내용]
- (상담사가 제공한 해결책이나 상황 설명 요약)
[결론]
- (고객의 최종 반응 및 향후 합의된 조치 요약)

[예시]
통화 전문: 
고객: 진료비 후불결제 안되나요? 예전에 병원 갔더니 수납 대기가 너무 힘들더라구요. 그리고 제 카드를 등록해두면 다른 결제에 쓰이거나 해킹당하는거 아니에요?
상담사: 고객님, 후불결제를 신청하시면 검사마다 수납하실 필요가 없습니다. 등록하신 카드는 병원 결제 외에는 절대 사용되지 않으며 철저히 보안 유지됩니다. 안심하셔도 됩니다.
상담사: 그리고 만약 창구 대기가 힘드시면 수납 창구 옆 키오스크를 이용해 보세요. 자원봉사자분들이 도와주실 것입니다. 
고객: 아 기계는 사실 제가 좀 어려워서요... 그래도 자꾸 해봐야겠죠. 알겠습니다. 다음 병원 가면 키오스크 한번 써볼게요. 수고하세요.

요약 결과:
[주제]
- 고객은 병원의 후불결제 서비스 및 수납 창구 대기에 따른 불편함과 카드 등록 보안을 문의함
[상담사의 응대 내용]
- 상담사는 후불결제 서비스의 장점과 철저한 보안성을 안내하고, 수납 대기 시간을 줄이기 위한 방안으로 키오스크 이용을 권장함
[결론]
- 고객은 키오스크 사용에 부담을 느꼈으나 상담사의 제안을 받아들여 다음 방문 시 시도해보기로 함

규칙:
1. 반드시 위 [지정된 양식]의 3가지 항목(주제, 응대 내용, 결론)을 제목 포함하여 지켜서 작성하세요.
2. 각 항목의 내용은 "- "로 시작하는 1~2개의 명확한 문장으로 작성하세요.
3. 고객의 감정 표현이나 대화의 세세한 뉘앙스보다는 "주요 사실 관계와 결과"만 뽑아 간결하게 요약하세요.
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

        # Ollama /api/chat 엔드포인트 사용 (대화형)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
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
