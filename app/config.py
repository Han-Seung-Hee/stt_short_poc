"""애플리케이션 설정 관리.

config.yaml 파일과 환경변수를 통합하여 설정을 관리합니다.
환경변수는 YAML 설정을 오버라이드합니다.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class STTChunkConfig:
    """청크 분할 설정."""
    enabled: bool = False
    length_sec: int = 300  # 5분


@dataclass
class STTConfig:
    """STT 엔진 설정."""
    engine: str = "mlx"            # "mlx" | "faster"
    model_name: str = "small"      # tiny, base, small, medium, large-v3
    model_path: Optional[str] = None  # 로컬 모델 경로 (None이면 기본 경로)
    language: str = "ko"
    chunk: STTChunkConfig = field(default_factory=STTChunkConfig)
    speaker_separation: bool = False  # 스테레오 채널 분리 화자 인식


@dataclass
class LLMConfig:
    """LLM 엔진 설정."""
    engine: str = "ollama"
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b"
    timeout_sec: int = 120
    max_retries: int = 2
    refine_enabled: bool = False  # 빠른 요약을 위해 기본값 False
    prompt_type: str = "default"  # "default" | "few_shot"


@dataclass
class ServerConfig:
    """서버 설정."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 100


@dataclass
class AppConfig:
    """전체 애플리케이션 설정."""
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """딕셔너리를 깊은 병합합니다."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(data: dict) -> dict:
    """환경변수로 설정을 오버라이드합니다.

    환경변수 규칙:
      STT_ENGINE, STT_MODEL_NAME, STT_MODEL_PATH, STT_LANGUAGE
      STT_CHUNK_ENABLED, STT_CHUNK_LENGTH_SEC
      LLM_ENGINE, LLM_BASE_URL, LLM_MODEL_NAME, LLM_TIMEOUT_SEC
      SERVER_HOST, SERVER_PORT, SERVER_MAX_FILE_SIZE_MB
    """
    env_mapping = {
        ("stt", "engine"): "STT_ENGINE",
        ("stt", "model_name"): "STT_MODEL_NAME",
        ("stt", "model_path"): "STT_MODEL_PATH",
        ("stt", "language"): "STT_LANGUAGE",
        ("stt", "chunk", "enabled"): "STT_CHUNK_ENABLED",
        ("stt", "chunk", "length_sec"): "STT_CHUNK_LENGTH_SEC",
        ("llm", "engine"): "LLM_ENGINE",
        ("llm", "base_url"): "LLM_BASE_URL",
        ("llm", "model_name"): "LLM_MODEL_NAME",
        ("llm", "timeout_sec"): "LLM_TIMEOUT_SEC",
        ("llm", "max_retries"): "LLM_MAX_RETRIES",
        ("server", "host"): "SERVER_HOST",
        ("server", "port"): "SERVER_PORT",
        ("server", "max_file_size_mb"): "SERVER_MAX_FILE_SIZE_MB",
    }

    for keys, env_var in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # 타입 변환
            current = data
            for k in keys[:-1]:
                current = current.setdefault(k, {})

            # 불리언/정수 자동 변환
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass

            current[keys[-1]] = value

    return data


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """설정 파일을 로드합니다.

    Args:
        config_path: YAML 설정 파일 경로. None이면 프로젝트 루트의 config.yaml 사용.

    Returns:
        AppConfig: 로드된 설정.
    """
    if config_path is None:
        # 프로젝트 루트의 config.yaml을 기본으로 사용
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / "config.yaml")

    data = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # 환경변수 오버라이드 적용
    data = _apply_env_overrides(data)

    # dataclass 생성
    stt_data = data.get("stt", {})
    chunk_data = stt_data.pop("chunk", {}) if isinstance(stt_data, dict) else {}
    llm_data = data.get("llm", {})
    server_data = data.get("server", {})

    chunk_config = STTChunkConfig(**chunk_data) if chunk_data else STTChunkConfig()
    stt_config = STTConfig(chunk=chunk_config, **stt_data) if stt_data else STTConfig()
    llm_config = LLMConfig(**llm_data) if llm_data else LLMConfig()
    server_config = ServerConfig(**server_data) if server_data else ServerConfig()

    return AppConfig(
        stt=stt_config,
        llm=llm_config,
        server=server_config,
    )


# 싱글톤 설정 인스턴스
_config: Optional[AppConfig] = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """설정 싱글톤을 반환합니다."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config() -> None:
    """설정 싱글톤을 초기화합니다 (테스트용)."""
    global _config
    _config = None
