"""Tests for the public GitHub Pages simple-index builder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.mc_simple_index import main


def _fake_wheel(dir_: Path, name: str, payload: bytes) -> Path:
    path = dir_ / name
    path.write_bytes(payload)
    return path


def test_builds_pep503_index(tmp_path: Path) -> None:
    """New wheels land under wheels/ with sha256 hrefs in simple/fplt-mc."""
    wheels = tmp_path / "dist"
    wheels.mkdir()
    name = "fplt_mc-0.1.0-cp312-abi3-manylinux_2_28_x86_64.whl"
    payload = b"wheel-bytes-v1"
    _fake_wheel(wheels, name, payload)
    out = tmp_path / "site"

    assert main(["--wheels", str(wheels), "--out", str(out)]) == 0

    published = out / "wheels" / name
    assert published.read_bytes() == payload
    digest = hashlib.sha256(payload).hexdigest()
    index = (out / "simple" / "fplt-mc" / "index.html").read_text(encoding="utf-8")
    assert name in index
    assert f"sha256={digest}" in index
    assert (out / "index.html").is_file()


def test_refuses_overwrite_different_bytes(tmp_path: Path) -> None:
    """Same filename with different content must fail (immutable publish)."""
    wheels = tmp_path / "dist"
    wheels.mkdir()
    name = "fplt_mc-0.1.0-cp312-abi3-linux_x86_64.whl"
    out = tmp_path / "site"
    _fake_wheel(wheels, name, b"aaa")
    assert main(["--wheels", str(wheels), "--out", str(out)]) == 0

    _fake_wheel(wheels, name, b"bbb")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--wheels", str(wheels), "--out", str(out)])


def test_idempotent_same_bytes(tmp_path: Path) -> None:
    """Re-publishing identical bytes is a no-op success."""
    wheels = tmp_path / "dist"
    wheels.mkdir()
    name = "fplt_mc-0.1.0-py3-none-any.whl"
    _fake_wheel(wheels, name, b"same")
    out = tmp_path / "site"
    assert main(["--wheels", str(wheels), "--out", str(out)]) == 0
    assert main(["--wheels", str(wheels), "--out", str(out)]) == 0
