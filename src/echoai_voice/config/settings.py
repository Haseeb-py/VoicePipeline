"""Application settings and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv_file(dotenv_path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into the process environment."""

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Typed application settings loaded exclusively from the environment."""

    deepgram_api_key: str
    groq_api_key: str
    groq_model: str
    whisper_model: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    default_tts_provider: str
    default_tts_voice: str
    persona_dir: Path
    low_confidence_threshold: float
    log_level: str


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache application settings after initializing dotenv."""

    project_root = _get_project_root()
    _load_dotenv_file(project_root / ".env")
    return AppSettings(
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        # Keep this aligned with a generally available Groq production model.
        # The previous Llama model was retired and caused every response request
        # to fail with a 404 before any TTS audio could be generated.
        groq_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo"),
        livekit_url=os.getenv("LIVEKIT_URL", ""),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        default_tts_provider=os.getenv("DEFAULT_TTS_PROVIDER", "deepgram"),
        default_tts_voice=os.getenv("DEFAULT_TTS_VOICE", "aura-2-asteria-en"),
        persona_dir=Path(os.getenv("PERSONA_DIR", str(project_root / "config" / "personas"))),
        low_confidence_threshold=float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.72")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
