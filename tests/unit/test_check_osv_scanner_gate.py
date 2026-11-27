"""#309: osv-scanner gate detects the known-vulnerable fixture."""

from __future__ import annotations

import json
import stat
import textwrap
from pathlib import Path

import pytest

from scripts.check_osv_scanner_gate import main, scan_fixture


def _write_exec(path: Path, body: str) -> Path:
    """Write an executable shell script at ``path``."""
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_scan_fixture_collects_vulnerabilities(tmp_path: Path) -> None:
    """Wrapper parses osv-scanner JSON and returns vuln objects."""
    report = {
        "results": [
            {
                "packages": [
                    {
                        "package": {"name": "pyyaml", "version": "5.1"},
                        "vulnerabilities": [{"id": "GHSA-test"}],
                    }
                ]
            }
        ]
    }
    argv_log = tmp_path / "argv.txt"
    fake = _write_exec(
        tmp_path / "osv-scanner",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf '%s\\n' "$*" > "{argv_log}"
            cat <<'EOF'
            {json.dumps(report)}
            EOF
            """
        ),
    )
    vulns = scan_fixture(fake, tmp_path)
    assert len(vulns) == 1
    assert vulns[0]["id"] == "GHSA-test"
    argv = argv_log.read_text(encoding="utf-8")
    assert "--config" in argv
    assert "osv-scanner.toml" in argv


def test_scan_fixture_empty_report_is_empty_list(tmp_path: Path) -> None:
    """A clean report yields no vulnerabilities."""
    fake = _write_exec(
        tmp_path / "osv-scanner",
        "#!/bin/sh\necho '{\"results\":[]}'\n",
    )
    assert scan_fixture(fake, tmp_path) == []


@pytest.mark.slow
@pytest.mark.enable_socket
def test_live_vulnerable_fixture_is_detected() -> None:
    """Acceptance: real osv-scanner reports vulns on the pinned fixture."""
    assert main([]) == 0
