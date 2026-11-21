"""Process settings loaded from the environment."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


def bearer_token(value: str | None) -> str | None:
    """Return a raw JWT, dropping a ``Bearer `` prefix and surrounding quotes.

    Args:
        value: Token as pasted from DevTools, or ``None``.

    Returns:
        The token with no scheme prefix, or ``None`` when blank.
    """
    if value is None:
        return None
    token = value.strip().strip('"').strip("'").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


class Settings(BaseSettings):
    """Paths and the FPL entry this install optimises for.

    ``fpl_api_token`` is the Bearer value from the ``x-api-authorization``
    header on a logged-in ``my-team`` request. It authenticates
    ``/api/my-team/{entry_id}/`` (selling prices and free transfers).
    ``fpl_session`` is the retired ``pl_profile`` cookie and is still sent
    when set. Never commit a real token or cookie.
    """

    model_config = SettingsConfigDict(
        env_prefix="FPLT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    entry_id: int | None = None
    fpl_api_token: str | None = None
    fpl_session: str | None = None
    ntfy_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_to: str | None = None
    smtp_use_tls: bool = True

    @field_validator("fpl_api_token", mode="before")
    @classmethod
    def _normalise_api_token(cls, value: object) -> object:
        if isinstance(value, str):
            return bearer_token(value)
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Skip the repo ``.env`` under pytest so a local entry id cannot leak in.

        Args:
            settings_cls: The settings class being built.
            init_settings: Values passed to the constructor.
            env_settings: Process environment.
            dotenv_settings: The ``.env`` file next to the working directory.
            file_secret_settings: File-based secrets source.

        Returns:
            Sources in priority order. Pytest keeps init and process env only.
        """
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return init_settings, env_settings
        return init_settings, env_settings, dotenv_settings, file_secret_settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, loaded once."""
    return Settings()
