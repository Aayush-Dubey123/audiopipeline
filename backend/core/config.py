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
    DEBUG: bool = Field(default=True, description="Debug mode toggle")

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
        if not v or "USER:PASSWORD" in v or "DBNAME" in v:
            return "sqlite:///./voice_studio.db"
        return v

    TMP_DIR: str = Field(default="storage/tmp", description="Temporary files storage directory")
    VOICES_DIR: str = Field(default="voices", description="Voice reference and model assets directory")

    MAX_UPLOAD_MB: int = Field(default=25, description="Maximum permitted upload size in megabytes")
    MAX_AUDIO_SECONDS: int = Field(default=120, description="Maximum audio length in seconds")
    MAX_TEXT_CHARS: int = Field(default=1000, description="Maximum script text character count")

    JOB_TTL_MINUTES: int = Field(default=60, description="Job artifact retention time in minutes")
    QUEUE_MAX_SIZE: int = Field(default=10, description="Maximum queue capacity for inference jobs")
    JOB_TIMEOUT_SECONDS: int = Field(default=180, description="Timeout per job in seconds")
    RATE_LIMIT_PER_MINUTE: int = Field(default=10, description="Max requests permitted per minute per IP")

    GROQ_API_KEY: str = Field(default="", description="Optional Groq API key")
    GEMINI_API_KEY: str = Field(default="", description="Optional Gemini API key")
    OPENROUTER_API_KEY: str = Field(default="", description="Optional OpenRouter API key")

    EXPOSE_EXPERIMENTAL_VOICES: bool = Field(
        default=False,
        description="Whether experimental voice entries are exposed in public endpoints",
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
