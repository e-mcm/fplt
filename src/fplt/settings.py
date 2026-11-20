"""Process settings loaded from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paths and the FPL entry this install optimises for.

    ``fpl_session`` is the ``pl_profile`` cookie value from
    fantasy.premierleague.com. It is required for the live squad via
    ``/api/my-team/{entry_id}/`` (selling prices and free transfers). Never
    commit a real cookie value.
    """

    model_config = SettingsConfigDict(
        env_prefix="FPLT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    entry_id: int | None = None
    fpl_session: str | None = None
    ntfy_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_to: str | None = None
    smtp_use_tls: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, loaded once."""
    return Settings()
