"""Process settings loaded from the environment."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class SettingsConfigError(ValueError):
    """A required setting is missing or empty (operator config, not a bug)."""


_ENTRY_ID_UNSET = "FPLT_ENTRY_ID is not set (GitHub: environment 'r2' secret)"

# Credential fields held as ``SecretStr`` (or already so). Used for log redaction.
_SECRET_FIELD_NAMES: tuple[str, ...] = (
    "fpl_api_token",
    "fpl_session",
    "api_football_key",
    "r2_access_key_id",
    "r2_secret_access_key",
    "ntfy_url",
    "smtp_password",
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


def secret_value(value: SecretStr | str | None) -> str | None:
    """Unwrap a ``SecretStr`` (or plain string) to a non-empty secret.

    Args:
        value: Credential from ``Settings``, or ``None``.

    Returns:
        The underlying string, or ``None`` when unset or blank.
    """
    if value is None:
        return None
    if isinstance(value, SecretStr):
        text = value.get_secret_value()
    else:
        text = value
    return text or None


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
    ``/api/my-team/{entry_id}/`` when available (authoritative live squad)
    and is **required for ``fplt apply --confirm``** (``POST /transfers/``,
    lineup). Planning with ``FPLT_ENTRY_ID`` does not need it: missing or
    expired tokens fall back to ``public_squad_state``. ``fpl_session`` is
    the retired ``pl_profile`` cookie and is still sent when set.
    ``api_football_key`` is the API-Sports key for injuries ingest
    (``FPLT_API_FOOTBALL_KEY``); the fetch script also accepts
    ``API_FOOTBALL_KEY`` / ``APISPORTS_KEY``. ``r2_*`` / ``data_bucket`` /
    ``lab_bucket`` configure the Cloudflare R2 remotes used by
    ``fplt.data.remote``. Credential fields are ``SecretStr`` so ``repr`` /
    validation errors never print them. Never commit a real token, cookie,
    API key, or R2 secret; store them with ``fplt auth`` / env under
    ``~/.config/fplt/env``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FPLT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    entry_id: int | None = None
    fpl_api_token: SecretStr | None = None
    fpl_session: SecretStr | None = None
    api_football_key: SecretStr | None = None
    # Cloudflare R2 (unset data_bucket ⇒ snapshots-branch fetch behaviour).
    r2_account_id: str | None = None
    r2_access_key_id: SecretStr | None = None
    r2_secret_access_key: SecretStr | None = None
    data_bucket: str | None = None
    lab_bucket: str | None = None
    # Experimental Touchline Wire predicted-XI fetch (off by default).
    touchline_lineups_enabled: bool = False
    touchline_lineups_min_interval_seconds: int = 300
    ntfy_url: SecretStr | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str | None = None
    smtp_to: str | None = None
    smtp_use_tls: bool = True

    @field_validator("fpl_api_token", mode="before")
    @classmethod
    def _normalise_api_token(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return bearer_token(value)
        return value

    def configured_secrets(self) -> list[str]:
        """Return non-empty credential strings for log redaction.

        Returns:
            Underlying secret values from every credential field that is set.
        """
        out: list[str] = []
        for name in _SECRET_FIELD_NAMES:
            raw = secret_value(getattr(self, name))
            if raw is not None:
                out.append(raw)
        return out

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


def require_entry_id(settings: Settings | None = None) -> int:
    """Return ``entry_id`` or raise a clear config error when unset.

    Args:
        settings: Process settings; defaults to ``get_settings()``.

    Returns:
        The configured FPL manager entry id.

    Raises:
        SettingsConfigError: ``FPLT_ENTRY_ID`` is unset (including empty env).
    """
    cfg = get_settings() if settings is None else settings
    if cfg.entry_id is None:
        raise SettingsConfigError(_ENTRY_ID_UNSET)
    return cfg.entry_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, loaded once."""
    return Settings()
