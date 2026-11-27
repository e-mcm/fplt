"""Pydantic schema for ``raw/vaastav/captured_at=…/_source.json``."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VaastavSourceFile(BaseModel):
    """One mirrored file inside a vaastav raw capture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("path")
    @classmethod
    def _posix_relative(cls, value: str) -> str:
        cleaned = value.strip().replace("\\", "/")
        if not cleaned or cleaned.startswith("/") or ".." in cleaned.split("/"):
            raise ValueError("path must be a relative POSIX path without '..'")
        return cleaned

    @field_validator("sha256")
    @classmethod
    def _lowercase_hex(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return cleaned


class VaastavSourceManifest(BaseModel):
    """Provenance manifest written next to mirrored vaastav files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(min_length=1)
    commit_sha: str = Field(min_length=7)
    captured_at: datetime
    seasons: list[str] = Field(default_factory=list[str])
    files: list[VaastavSourceFile] = Field(default_factory=list[VaastavSourceFile])

    @field_validator("commit_sha")
    @classmethod
    def _hex_sha(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if len(cleaned) < 7 or any(ch not in "0123456789abcdef" for ch in cleaned):
            raise ValueError("commit_sha must be lowercase hex (at least 7 chars)")
        return cleaned

    @field_validator("seasons")
    @classmethod
    def _season_labels(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for raw in value:
            label = raw.strip()
            if len(label) != 7 or label[4] != "-":
                raise ValueError(f"invalid season label: {raw!r}")
            out.append(label)
        return out
