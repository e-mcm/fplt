"""Vaastav raw ``_source.json`` schema validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fplt.schemas.vaastav_source import VaastavSourceFile, VaastavSourceManifest


def test_source_file_normalises_path_and_sha() -> None:
    entry = VaastavSourceFile(
        path="data/2024-25/gws/merged_gw.csv",
        bytes=12,
        sha256="A" * 64,
    )
    assert entry.path == "data/2024-25/gws/merged_gw.csv"
    assert entry.sha256 == "a" * 64


def test_source_file_rejects_bad_path() -> None:
    with pytest.raises(ValidationError):
        VaastavSourceFile(path="../secret", bytes=1, sha256="a" * 64)
    with pytest.raises(ValidationError):
        VaastavSourceFile(path="/abs", bytes=1, sha256="a" * 64)


def test_manifest_round_trip() -> None:
    manifest = VaastavSourceManifest(
        repo="https://example.test/vaastav.git",
        commit_sha="abcdef1",
        captured_at=datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
        seasons=["2024-25", "2025-26"],
        files=[
            VaastavSourceFile(
                path="data/2024-25/gws/merged_gw.csv",
                bytes=4,
                sha256="b" * 64,
            )
        ],
    )
    loaded = VaastavSourceManifest.model_validate_json(manifest.model_dump_json())
    assert loaded.commit_sha == "abcdef1"
    assert loaded.seasons == ["2024-25", "2025-26"]
    assert loaded.files[0].bytes == 4


def test_manifest_rejects_bad_season_label() -> None:
    with pytest.raises(ValidationError):
        VaastavSourceManifest(
            repo="r",
            commit_sha="abcdef1",
            captured_at=datetime(2026, 1, 1, tzinfo=UTC),
            seasons=["2024"],
            files=[],
        )
