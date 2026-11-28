"""#340: push_raw_upload.sh sets uploaded only after rclone exits 0."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_HELPER = _REPO / "scripts" / "push_raw_upload.sh"
_INSTALL = _REPO / ".github" / "actions" / "install-rclone" / "action.yml"
_RUNBOOK = _REPO / "docs" / "ops-runbook.md"


def _write_fake_rclone(
    bin_dir: Path,
    *,
    copy_rc: int = 0,
    marker_rc: int | None = None,
    check_rc: int = 0,
) -> Path:
    """Install a stub ``rclone`` that records argv and returns configured codes."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    log = bin_dir / "rclone-calls.log"
    # Second copy (``_source.json``) may fail independently of the payload pass.
    second_copy_rc = copy_rc if marker_rc is None else marker_rc
    script = bin_dir / "rclone"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            echo "$@" >>"{log}"
            cmd="${{1:-}}"
            if [ "$cmd" = "copy" ]; then
              n="$(grep -c '^copy ' "{log}" || true)"
              if [ "$n" -eq 1 ]; then
                exit {copy_rc}
              fi
              exit {second_copy_rc}
            fi
            if [ "$cmd" = "check" ]; then
              exit {check_rc}
            fi
            echo "unexpected rclone argv: $*" >&2
            exit 99
            """
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return log


def _run_helper(
    tmp_path: Path,
    *,
    src: Path,
    dest: str = "fplt:fplt-data/raw",
    copy_rc: int = 0,
    marker_rc: int | None = None,
    check_rc: int = 0,
    transfers: str | None = None,
    checkers: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``push_raw_upload.sh`` with a fake rclone on PATH."""
    bin_dir = tmp_path / "bin"
    _write_fake_rclone(bin_dir, copy_rc=copy_rc, marker_rc=marker_rc, check_rc=check_rc)
    out = tmp_path / "github_output"
    out.write_text("", encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "GITHUB_OUTPUT": str(out),
        "HOME": str(tmp_path),
    }
    if transfers is not None:
        env["RCLONE_TRANSFERS"] = transfers
    if checkers is not None:
        env["RCLONE_CHECKERS"] = checkers
    return subprocess.run(
        ["bash", str(_HELPER), str(src), dest],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )


def test_helper_sets_uploaded_true_after_successful_copy_and_check(
    tmp_path: Path,
) -> None:
    """Non-empty src + rclone copy/check exit 0 → uploaded=true."""
    src = tmp_path / "raw"
    src.mkdir()
    (src / "file.json.gz").write_bytes(b"payload")
    result = _run_helper(tmp_path, src=src)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "uploaded=true" in (tmp_path / "github_output").read_text(encoding="utf-8")
    calls = (tmp_path / "bin" / "rclone-calls.log").read_text(encoding="utf-8")
    assert "copy " in calls
    assert "check --one-way " in calls


def test_helper_uploads_source_json_in_second_pass(tmp_path: Path) -> None:
    """Payload excludes ``_source.json``; marker pass includes it last (#353)."""
    src = tmp_path / "raw"
    capture = src / "vaastav" / "captured_at=2026-11-28T00-00-00Z"
    (capture / "data").mkdir(parents=True)
    (capture / "data" / "file.csv").write_text("a\n", encoding="utf-8")
    (capture / "_source.json").write_text("{}\n", encoding="utf-8")
    result = _run_helper(tmp_path, src=src)
    assert result.returncode == 0, result.stderr + result.stdout
    log_text = (tmp_path / "bin" / "rclone-calls.log").read_text(encoding="utf-8")
    lines = log_text.splitlines()
    copies = [line for line in lines if line.startswith("copy ")]
    assert len(copies) == 2
    assert "--exclude **/_source.json" in copies[0]
    assert "--include **/_source.json" in copies[1]
    assert "--transfers 32" in copies[0]
    assert "--checkers 32" in copies[0]
    assert lines[-1].startswith("check --one-way ")


def test_helper_transfers_checkers_overridable(tmp_path: Path) -> None:
    """``RCLONE_TRANSFERS`` / ``RCLONE_CHECKERS`` override the defaults (#353)."""
    src = tmp_path / "raw"
    src.mkdir()
    (src / "file.json.gz").write_bytes(b"payload")
    result = _run_helper(tmp_path, src=src, transfers="8", checkers="16")
    assert result.returncode == 0, result.stderr + result.stdout
    calls = (tmp_path / "bin" / "rclone-calls.log").read_text(encoding="utf-8")
    assert "--transfers 8" in calls
    assert "--checkers 16" in calls


def test_helper_sets_uploaded_false_when_copy_fails(tmp_path: Path) -> None:
    """Non-zero rclone copy must not claim uploaded=true (#340)."""
    src = tmp_path / "raw"
    src.mkdir()
    (src / "file.json.gz").write_bytes(b"payload")
    result = _run_helper(tmp_path, src=src, copy_rc=1)
    assert result.returncode == 1
    out = (tmp_path / "github_output").read_text(encoding="utf-8")
    assert "uploaded=false" in out
    assert "uploaded=true" not in out
    calls = (tmp_path / "bin" / "rclone-calls.log").read_text(encoding="utf-8")
    assert calls.count("copy ") == 1
    assert "check " not in calls


def test_helper_sets_uploaded_false_when_marker_copy_fails(tmp_path: Path) -> None:
    """Payload success with ``_source.json`` copy failure → uploaded=false (#353)."""
    src = tmp_path / "raw"
    src.mkdir()
    (src / "file.json.gz").write_bytes(b"payload")
    result = _run_helper(tmp_path, src=src, marker_rc=3)
    assert result.returncode == 3
    out = (tmp_path / "github_output").read_text(encoding="utf-8")
    assert "uploaded=false" in out
    assert "uploaded=true" not in out
    calls = (tmp_path / "bin" / "rclone-calls.log").read_text(encoding="utf-8")
    assert calls.count("copy ") == 2
    assert "check " not in calls


def test_helper_sets_uploaded_false_when_check_fails(tmp_path: Path) -> None:
    """Copy success with check failure still reports uploaded=false."""
    src = tmp_path / "raw"
    src.mkdir()
    (src / "file.json.gz").write_bytes(b"payload")
    result = _run_helper(tmp_path, src=src, check_rc=2)
    assert result.returncode == 2
    out = (tmp_path / "github_output").read_text(encoding="utf-8")
    assert "uploaded=false" in out
    assert "uploaded=true" not in out


def test_helper_empty_src_sets_uploaded_true_without_rclone(tmp_path: Path) -> None:
    """Empty capture dir is success with nothing to transfer."""
    src = tmp_path / "raw"
    src.mkdir()
    result = _run_helper(tmp_path, src=src)
    assert result.returncode == 0
    assert "nothing new" in result.stdout
    assert "uploaded=true" in (tmp_path / "github_output").read_text(encoding="utf-8")
    assert not (tmp_path / "bin" / "rclone-calls.log").exists()


def test_helper_rejects_dot_path_components(tmp_path: Path) -> None:
    """A source tree with ``.foo/bar`` exits non-zero and skips rclone (#339)."""
    src = tmp_path / "raw"
    bad = src / ".foo"
    bad.mkdir(parents=True)
    (bad / "bar").write_text("nope\n", encoding="utf-8")
    (src / "ok.txt").write_text("ok\n", encoding="utf-8")
    result = _run_helper(tmp_path, src=src)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "::error::" in combined
    assert "uploaded=false" in (tmp_path / "github_output").read_text(encoding="utf-8")
    assert "uploaded=true" not in (tmp_path / "github_output").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "bin" / "rclone-calls.log").exists()


def test_push_raw_action_exposes_transfers_checkers_inputs() -> None:
    """Composite action defaults transfers/checkers to 32 and forwards them (#353)."""
    action = (_REPO / ".github" / "actions" / "push-raw" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert 'default: "32"' in action
    assert "RCLONE_TRANSFERS: ${{ inputs.transfers }}" in action
    assert "RCLONE_CHECKERS: ${{ inputs.checkers }}" in action


def test_install_rclone_default_matches_ops_runbook_pin() -> None:
    """Action default and ops-runbook Tooling pins must stay in sync (#340)."""
    doc = yaml.safe_load(_INSTALL.read_text(encoding="utf-8"))
    version = str(doc["inputs"]["version"]["default"])
    runbook = _RUNBOOK.read_text(encoding="utf-8")
    assert re.search(
        rf"rclone\s*\|\s*\*\*{re.escape(version)}\*\*",
        runbook,
    ), f"ops-runbook pin missing rclone {version}"


def test_install_rclone_verifies_sha256sums() -> None:
    """Pinned install must checksum-verify the release zip against SHA256SUMS."""
    text = _INSTALL.read_text(encoding="utf-8")
    assert "SHA256SUMS" in text
    assert "checksum mismatch" in text
    assert "apt-get" not in text
