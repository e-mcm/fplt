"""Process settings loaded from the environment."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
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


def user_env_file() -> Path:
    """Return the out-of-repo secrets file path.

    Honours ``FPLT_ENV_FILE`` (same as ``deploy/launchd/run-with-env.sh``),
    otherwise ``~/.config/fplt/env``. Prefer ``fplt auth set`` over editing
    the repo ``.env``.
    """
    override = os.environ.get("FPLT_ENV_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "fplt" / "env"


class Settings(BaseSettings):
    """Paths and the FPL entry this install optimises for.

    ``fpl_api_token`` is the Bearer value from the ``x-api-authorization``
    header on a logged-in ``my-team`` request. It authenticates
    ``/api/my-team/{entry_id}/`` (selling prices and free transfers) and is
    required for ``fplt apply --confirm`` (``POST /transfers/``, lineup).
    ``fpl_session`` is the retired ``pl_profile`` cookie and is still sent
    when set. ``api_football_key`` is the API-Sports key for injuries
    ingest (``FPLT_API_FOOTBALL_KEY``); the fetch script also accepts
    ``API_FOOTBALL_KEY`` / ``APISPORTS_KEY``. Never commit a real token,
    cookie, or API key; store them with ``fplt auth`` / env under
    ``~/.config/fplt/env``.
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
    api_football_key: str | None = None
    # Experimental Touchline Wire predicted-XI fetch (off by default).
    touchline_lineups_enabled: bool = False
    touchline_lineups_min_interval_seconds: int = 300
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
        """Load process env, then ``~/.config/fplt/env``, then repo ``.env``.

        User config beats the checkout ``.env`` so ``fplt auth set`` wins over
        a stale or accidental repo paste. Pytest keeps init and process env
        only so a local entry id cannot leak into tests.

        Args:
            settings_cls: The settings class being built.
            init_settings: Values passed to the constructor.
            env_settings: Process environment.
            dotenv_settings: The ``.env`` file next to the working directory.
            file_secret_settings: File-based secrets source.

        Returns:
            Sources in priority order (first wins).
        """
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return init_settings, env_settings
        user_dotenv = DotEnvSettingsSource(
            settings_cls,
            env_file=user_env_file(),
            env_file_encoding="utf-8",
        )
        return (
            init_settings,
            env_settings,
            user_dotenv,
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, loaded once."""
    return Settings()
