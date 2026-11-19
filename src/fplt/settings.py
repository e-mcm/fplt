"""Process settings loaded from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paths and the FPL entry this install optimises for."""

    model_config = SettingsConfigDict(
        env_prefix="FPLT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    entry_id: int | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, loaded once."""
    return Settings()
