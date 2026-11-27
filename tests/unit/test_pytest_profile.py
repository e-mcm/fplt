"""Unit tests for ``scripts/pytest_profile`` JUnit ranking helpers."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import pytest

from scripts.pytest_profile import (
    CaseDuration,
    aggregate_by_file,
    format_duration_table,
    format_file_table,
    main,
    parse_junit_durations,
)


def _write_junit(path: Path, cases: list[tuple[str, str, str, float]]) -> None:
    """Write a minimal pytest-shaped JUnit XML file.

    Args:
        path: Destination path.
        cases: ``(classname, name, file, seconds)`` rows.
    """
    root = Element("testsuites")
    suite = SubElement(root, "testsuite", name="pytest", tests=str(len(cases)))
    for classname, name, file_attr, seconds in cases:
        SubElement(
            suite,
            "testcase",
            classname=classname,
            name=name,
            file=file_attr,
            time=f"{seconds:.6f}",
        )
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_parse_junit_durations_sorts_descending(tmp_path: Path) -> None:
    """Slowest cases come first; nodeids prefer the ``file`` attribute."""
    junit = tmp_path / "out.xml"
    _write_junit(
        junit,
        [
            ("tests.unit.test_a", "test_fast", "tests/unit/test_a.py", 0.1),
            ("tests.unit.test_b", "test_slow", "tests/unit/test_b.py", 3.5),
            ("tests.unit.test_c", "test_mid", "tests/unit/test_c.py", 1.2),
        ],
    )
    parsed = parse_junit_durations(junit)
    assert [c.seconds for c in parsed] == [3.5, 1.2, 0.1]
    assert parsed[0].nodeid == "tests/unit/test_b.py::test_slow"
    assert parsed[0].suite == "tests/unit/test_b.py"


def test_parse_junit_missing_file_raises(tmp_path: Path) -> None:
    """Absent JUnit path is a hard error."""
    with pytest.raises(FileNotFoundError, match="JUnit report not found"):
        parse_junit_durations(tmp_path / "missing.xml")


def test_parse_junit_rejects_wrong_root(tmp_path: Path) -> None:
    """Non-JUnit roots fail with ValueError."""
    bad = tmp_path / "bad.xml"
    bad.write_text("<html><body/></html>\n", encoding="utf-8")
    with pytest.raises(ValueError, match="testsuites/testsuite"):
        parse_junit_durations(bad)


def test_aggregate_by_file_sums_cases() -> None:
    """File totals sum case times and count rows."""
    cases = [
        CaseDuration("a.py::t1", 2.0, "a", "t1", "a.py"),
        CaseDuration("a.py::t2", 1.0, "a", "t2", "a.py"),
        CaseDuration("b.py::t3", 4.0, "b", "t3", "b.py"),
    ]
    rows = aggregate_by_file(cases)
    assert rows[0] == ("b.py", 4.0, 1)
    assert rows[1] == ("a.py", 3.0, 2)


def test_format_tables_include_headers() -> None:
    """Human tables include headers and ranked rows."""
    cases = [
        CaseDuration("tests/unit/x.py::test_z", 2.5, "x", "test_z", "tests/unit/x.py"),
    ]
    case_table = format_duration_table(cases, top=5, min_seconds=0.0)
    assert "seconds" in case_table
    assert "test_z" in case_table
    file_table = format_file_table([("tests/unit/x.py", 2.5, 1)], top=5)
    assert "suite" in file_table
    assert "tests/unit/x.py" in file_table


def test_main_parse_only_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--parse-only`` ranks without invoking pytest and can write JSON."""
    junit = tmp_path / "junit.xml"
    out = tmp_path / "durations.json"
    _write_junit(
        junit,
        [
            ("tests.unit.slow", "test_big", "tests/unit/slow.py", 5.0),
            ("tests.unit.fast", "test_tiny", "tests/unit/fast.py", 0.05),
        ],
    )
    rc = main(
        [
            "--parse-only",
            str(junit),
            "--top",
            "10",
            "--min-seconds",
            "0.0",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "test_big" in printed
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_cases"] == 2
    assert payload["top_cases"][0]["name"] == "test_big"
    assert payload["top_files"][0]["suite"] == "tests/unit/slow.py"
