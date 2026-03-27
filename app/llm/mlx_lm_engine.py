"""MLX-LM 기반 네이티브 로컬 LLM 엔진."""

import logging
import time
from typing import Optional

from app.llm.base import LLMEngine, SummaryResult

logger = logging.getLogger(__name__)

class MlxLMEngine(LLMEngine):
    """mlx-lm을 사용하여 애플 실리콘에서 네이티브로 요약을 수행합니다."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        prompt_type: str = "default",
    ):
        self._model_name = model_name
        self._prompt_type = prompt_type
        
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            from mlx_lm import load
            logger.info(f"MLX-LM: 모델 로드 시작 ({self._model_name}) - 수초 소요(최초 다운로드 시 약간 더 소요됨)")
            self.model, self.tokenizer = load(self._model_name)
            logger.info("MLX-LM: 모델 로드 완료")
        except Exception as e:
            logger.error(f"MLX-LM 모델 로드 실패: {e}")

    def summarize(self, text: str, system_prompt: Optional[str] = None) -> SummaryResult:
        from mlx_lm import generate
        
        start_time = time.time()
        
        if system_prompt is None:
            if self._prompt_type == "few_shot":
                system_prompt = """당신은 고객 상담 통화 내용 전문 요약 AI입니다.
사용자가 제공하는 통화 내용을 읽고, 반드시 아래의 JSON 형식으로만 대답해 주세요. 그 외 설명이나 마크다운 마크업(```json)을 붙이지 마세요.

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
4. 고객 또는 상담사가 **명백한 심한 욕설이나 거친 비속어(예: ㅅㅂ, 미친, 개새끼 등)**를 사용한 경우에만 is_profanity를 true로 하고, 해당 단어를 profanity_words 리스트에 넣으세요. 단순히 불만을 토로하거나 한탄조("피 말리네요", "말도 안 되죠", "그냥 끊으세요") 등의 일상적 표현은 비속어가 아니므로 절대 감지하지 마세요.
5. 반드시 오직 한국어(Korean)로만 작성하세요. 중국어 등 타 언어 사용은 절대 금지됩니다."""
            else:
                system_prompt = "다음 텍스트를 읽고 가장 중요한 핵심 내용 3줄로 요약해 주세요."
                
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
        logger.info(f"MLX-LM 요약 생성 시작")
        
        try:
            # generate 함수 호출. OOM 방지를 위해 파라미터 최적화
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt, 
                verbose=False,
                max_tokens=800
            )
        except Exception as e:
            logger.error(f"MLX-LM 생성 중 오류: {e}")
            raise RuntimeError(f"MLX-LM 생성 실패: {e}")
        
        elapsed_sec = time.time() - start_time
        
        return SummaryResult(
            summary=response.strip(),
            model_name=self._model_name,
            prompt_tokens=0,
            eval_tokens=0,
            elapsed_sec=elapsed_sec
        )

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def get_engine_name(self) -> str:
        return f"mlx-lm ({self._model_name})"
