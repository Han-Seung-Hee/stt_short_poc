"""faster-whisper 기반 STT 엔진 (Linux CPU 전용 — 향후 확장용 스텁).

GPU 없는 Linux 서버에서 CPU 모드로 Whisper를 실행합니다.
현재는 인터페이스만 구현되어 있으며, 실제 구동에 필요한 패키지는
Linux 환경에 맞춰 별도 설치합니다.
"""

import logging
from typing import Optional

from app.stt.base import STTEngine, STTResult, Segment

logger = logging.getLogger(__name__)


class FasterWhisperEngine(STTEngine):
    """faster-whisper를 사용하는 STT 엔진 (Linux CPU용).

    향후 Linux 서버 배포 시 활성화합니다.
    설치: pip install faster-whisper
    모델: CTranslate2 형식의 Whisper 모델 필요.
    """

    def __init__(
        self,
        model_name: str = "small",
        model_path: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        """모델을 로드합니다."""
        try:
            from faster_whisper import WhisperModel

            model_path = self.model_path or self.model_name
            self._model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(
                f"faster-whisper 모델 로드 완료: {model_path} "
                f"(device={self.device}, compute_type={self.compute_type})"
            )
        except ImportError:
            raise RuntimeError(
                "faster-whisper가 설치되지 않았습니다. "
                "Linux 환경에서 'pip install faster-whisper'를 실행하세요."
            )

    def transcribe(
        self,
        audio_path: str,
        language: str = "ko",
        chunk_enabled: bool = False,
        chunk_length_sec: int = 300,
        speaker_separation: bool = False,
    ) -> STTResult:
        """음성 파일을 텍스트로 변환합니다."""
        if self._model is None:
            self._load_model()

        import time
        import tempfile
        from pathlib import Path

        start_time = time.time()
        
        is_stereo_separated = False
        if speaker_separation:
            from pydub import AudioSegment
            audio_info = AudioSegment.from_wav(audio_path)
            if audio_info.channels == 2:
                is_stereo_separated = True
            else:
                logger.warning("스테레오 분리 옵션이 켜져 있으나, 음성이 모노입니다. 기본 모드로 진행합니다.")

        def _transcribe_file(path: str) -> list:
            segments_gen, _ = self._model.transcribe(
                path,
                language=language,
                beam_size=5,
                vad_filter=True,
                initial_prompt="다음은 고객센터 상담원과 고객의 통화 내용입니다. 자연스러운 한국어 존댓말이 사용됩니다."
            )
            segs = []
            for seg in segments_gen:
                segs.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
            return segs

        if is_stereo_separated:
            from pydub import AudioSegment
            logger.info("============== 스테레오 화자 분리 모드로 인식 시작 (혼합 STT 후 에너지 기반 분리, Linux CPU) ==============")
            audio = AudioSegment.from_wav(audio_path)
            left, right = audio.split_to_mono()
            
            # 1. 원본 믹스본을 통해 시계열 유지 통 STT
            all_segments = _transcribe_file(audio_path)
            
            # 2. RMS 비교로 화자 분리
            for seg in all_segments:
                start_ms = int(seg.start * 1000)
                end_ms = int(seg.end * 1000)
                
                chunk_l = left[start_ms:end_ms]
                chunk_r = right[start_ms:end_ms]
                
                if chunk_l.rms > chunk_r.rms:
                    seg.text = f"상담사: {seg.text}"
                else:
                    seg.text = f"고객: {seg.text}"
        else:
            all_segments = _transcribe_file(audio_path)

        elapsed = time.time() - start_time
        
        formatted_text = "\n".join([seg.text for seg in all_segments])

        return STTResult(
            text=formatted_text,
            segments=all_segments,
            language=language,
            duration_sec=elapsed,
            model_name=f"faster-whisper/{self.model_name}",
        )

    def is_available(self) -> bool:
        """faster-whisper 사용 가능 여부를 확인합니다."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except ImportError:
            return False

    def get_engine_name(self) -> str:
        return f"faster-whisper ({self.model_name}, {self.device})"
