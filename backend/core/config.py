"""
Centralized application settings and configuration.

Loads validated environment configurations from .env using Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    ENV: str = Field(default="development", description="Runtime environment: development or production")
    DEBUG: bool = Field(default=False, description="Debug mode toggle")

    HOST: str = Field(default="127.0.0.1", description="Bind address. Keep 127.0.0.1 unless you need LAN/container access")
    PORT: int = Field(default=8000, description="Bind port")
    TRUST_PROXY_HEADERS: bool = Field(
        default=False,
        description="Trust X-Forwarded-For for client identity. Enable ONLY behind a trusted reverse proxy",
    )

    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated allowed CORS origins",
    )

    DATABASE_URL: str = Field(
        default="sqlite:///./voice_studio.db",
        description="SQLAlchemy database connection URI",
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """
        Validate database URL, falling back to SQLite if an external placeholder is detected.

        Args:
            v (str): Raw database URI input string.

        Returns:
            str: Validated connection URI string.

        Raises:
            None
        """
        backend_dir = Path(__file__).resolve().parent.parent
        default_sqlite = f"sqlite:///{(backend_dir / 'voice_studio.db').as_posix()}"
        if not v or "USER:PASSWORD" in v or "DBNAME" in v:
            return default_sqlite
        # Anchor relative SQLite paths to backend/ so launching from another folder
        # never creates a stray second database.
        if v.startswith("sqlite:///./") or (v.startswith("sqlite:///") and not v.startswith("sqlite:////")
                                             and not Path(v[len("sqlite:///"):]).is_absolute()):
            rel = v[len("sqlite:///"):]
            return f"sqlite:///{(backend_dir / rel).resolve().as_posix()}"
        return v

    TMP_DIR: str = Field(default="storage/tmp", description="Temporary files storage directory")
    VOICES_DIR: str = Field(default="voices", description="Voice reference and model assets directory")

    MAX_UPLOAD_MB: int = Field(default=25, description="Maximum permitted upload size in megabytes")
    MAX_AUDIO_SECONDS: int = Field(default=120, description="Maximum audio length in seconds")
    MAX_TEXT_CHARS: int = Field(default=30000, description="Maximum script text character count (long-form enabled)")

    # Long-form chunk pipeline settings
    CHUNK_MAX_CHARS: int = Field(default=220, description="Maximum characters per synthesis chunk")
    CHUNK_PAUSE_SENTENCE_MS: int = Field(default=250, description="Silence gap between sentence chunks in milliseconds")
    CHUNK_PAUSE_PARAGRAPH_MS: int = Field(default=600, description="Silence gap between paragraph chunks in milliseconds")
    CHUNK_TIMEOUT_SECONDS: int = Field(default=180, description="Timeout per individual synthesis chunk in seconds")
    JOB_MAX_SECONDS: int = Field(default=7200, description="Absolute wall-clock limit for a single job (2 hours)")

    JOB_TTL_MINUTES: int = Field(default=120, description="Job artifact retention time in minutes, counted from job completion")
    QUEUE_MAX_SIZE: int = Field(default=10, description="Maximum queue capacity for inference jobs")
    JOB_TIMEOUT_SECONDS: int = Field(default=180, description="Legacy single-call timeout in seconds (kept for VC jobs)")
    RATE_LIMIT_PER_MINUTE: int = Field(default=10, description="Max requests permitted per minute per IP")

    # Storage hygiene (prevents junk build-up)
    CLEANUP_INTERVAL_SECONDS: int = Field(default=600, description="How often the janitor purges expired/orphaned files")
    FAILED_JOB_TTL_MINUTES: int = Field(default=60, description="Keep failed-job chunks this long for resume, then purge")
    CONDS_CACHE_MAX_FILES: int = Field(default=10, description="Max cached speaker-conditioning files kept on disk")
    LOG_MAX_BYTES: int = Field(default=1_000_000, description="Rotate app.log at this size")
    LOG_BACKUP_COUNT: int = Field(default=1, description="Rotated log files to keep")

    # Inference resource policy
    CHATTERBOX_DEVICE: str = Field(default="auto", description="auto | cuda | cpu")
    OUTPUT_SAMPLE_RATE: int = Field(default=24000, description="Output WAV sample rate (model native 24 kHz = smallest, no quality loss)")
    MODEL_IDLE_UNLOAD_SECONDS: int = Field(default=300, description="Free GPU memory after this many idle seconds (0 = never)")

    # Per-client generation cap
    CLIENT_DAILY_MINUTES: int = Field(default=120, description="Maximum generated audio minutes per client (hashed IP) per day")

    GROQ_API_KEY: str = Field(default="", description="Optional Groq API key")
    GEMINI_API_KEY: str = Field(default="", description="Optional Gemini API key")
    OPENROUTER_API_KEY: str = Field(default="", description="Optional OpenRouter API key")

    EXPOSE_EXPERIMENTAL_VOICES: bool = Field(
        default=False,
        description="Whether experimental voice entries are exposed in public endpoints",
    )

    CHATTERBOX_PYTHON_PATH: str = Field(
        default="",
        description="Path to python executable with chatterbox dependencies installed. If empty, checks backend/.venv_chatterbox or system environment",
    )
    CHATTERBOX_IN_PROCESS: bool = Field(
        default=True,
        description="Run resident Chatterbox model in-process for low latency",
    )
    TTS_CONFIDENCE_GOOD_THRESHOLD: float = Field(
        default=0.70,
        description="Speaker similarity threshold for 'good' confidence label",
    )
    TTS_CONFIDENCE_FAIR_THRESHOLD: float = Field(
        default=0.56,
        description="Speaker similarity threshold for 'fair' confidence label",
    )

    @property
    def backend_dir(self) -> Path:
        """
        Absolute Path to the backend root directory.

        Returns:
            Path: Backend directory path.
        """
        return Path(__file__).resolve().parent.parent

    @property
    def tmp_path(self) -> Path:
        """
        Absolute Path to the temporary storage directory inside backend.

        Returns:
            Path: Resolved tmp directory path.
        """
        p = Path(self.TMP_DIR)
        return p if p.is_absolute() else (self.backend_dir / p)

    @property
    def voices_path(self) -> Path:
        """
        Absolute Path to the voices directory inside backend.

        Returns:
            Path: Resolved voices directory path.
        """
        p = Path(self.VOICES_DIR)
        return p if p.is_absolute() else (self.backend_dir / p)

    @property
    def cors_origins_list(self) -> List[str]:
        """
        Parse comma-separated allowed origins into a typed list.

        Returns:
            List[str]: List of origin strings trimmed of whitespace.

        Raises:
            None
        """
        if not self.ALLOWED_ORIGINS:
            return ["*"] if self.DEBUG else []
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    """
    Retrieve cached singleton instance of application settings.

    Returns:
        Settings: Validated process-wide settings instance.

    Raises:
        None
    """
    return Settings()
