"""#309: actionlint + zizmor wrapper fails on medium+ findings."""

from __future__ import annotations

import json
import stat
import textwrap
from pathlib import Path

from scripts.check_workflow_security_lints import run_actionlint, run_zizmor

_REPO = Path(__file__).resolve().parents[2]


def _write_exec(path: Path, body: str) -> Path:
    """Write an executable shell script at ``path``."""
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_run_zizmor_fails_on_active_finding(tmp_path: Path) -> None:
    """A non-ignored medium finding makes the wrapper exit 1."""
    fake = _write_exec(
        tmp_path / "zizmor",
        textwrap.dedent(
            """\
            #!/bin/sh
            cat <<'EOF'
            [{"ident": "artipacked", "ignored": false,
              "determinations": {"severity": "Medium"},
              "locations": []}]
            EOF
            """
        ),
    )
    assert run_zizmor(fake, tmp_path) == 1


def test_run_zizmor_ignores_suppressed_finding(tmp_path: Path) -> None:
    """Ignored findings do not fail the wrapper."""
    fake = _write_exec(
        tmp_path / "zizmor",
        textwrap.dedent(
            """\
            #!/bin/sh
            cat <<'EOF'
            [{"ident": "artipacked", "ignored": true,
              "determinations": {"severity": "Medium"},
              "locations": []}]
            EOF
            """
        ),
    )
    assert run_zizmor(fake, tmp_path) == 0


def test_run_actionlint_propagates_exit_code(tmp_path: Path) -> None:
    """actionlint non-zero exit fails the wrapper step."""
    bad = _write_exec(tmp_path / "actionlint", "#!/bin/sh\nexit 2\n")
    assert run_actionlint(bad, tmp_path) == 2
    good = _write_exec(tmp_path / "actionlint-ok", "#!/bin/sh\nexit 0\n")
    assert run_actionlint(good, tmp_path) == 0


def test_run_zizmor_rejects_invalid_json(tmp_path: Path) -> None:
    """Garbage stdout from zizmor is a hard failure."""
    fake = _write_exec(
        tmp_path / "zizmor",
        "#!/bin/sh\necho not-json\n",
    )
    assert run_zizmor(fake, tmp_path) == 1


def test_live_workflows_json_roundtrip_shape() -> None:
    """Sanity: real zizmor JSON (when on PATH) is a list of objects."""
    # Optional: only asserts structure if zizmor is available locally.
    import shutil
    import subprocess

    zizmor = shutil.which("zizmor")
    if zizmor is None:
        return
    proc = subprocess.run(
        [
            zizmor,
            str(_REPO / ".github" / "workflows"),
            "--min-severity",
            "medium",
            "--format",
            "json",
        ],
        cwd=str(_REPO),
        check=False,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout or "[]")
    assert isinstance(data, list)
    active = [f for f in data if not f.get("ignored")]
    assert active == []
