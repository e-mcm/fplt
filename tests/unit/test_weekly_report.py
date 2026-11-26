"""Weekly decision report writer: files + idempotent notify (ops-08)."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request

import polars as pl

from fplt.optimise.weekly_report import (
    format_weekly_markdown,
    report_dir,
    rollup_season_to_date,
    run_weekly_report,
)
from fplt.settings import Settings
from tests.unit.test_decision_scoring import _decision_gw


def test_write_and_rerun_are_idempotent_and_notify_once(tmp_path: Path) -> None:
    gw1 = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        captain_index=0,
    )
    posts: list[Request] = []

    def fake_post(request: Request) -> object:
        posts.append(request)
        return object()

    settings = Settings(ntfy_url="https://ntfy.sh/test-topic")
    first = run_weekly_report(
        tmp_path,
        season="2024-25",
        gw=1,
        decisions=gw1,
        settings=settings,
        post=fake_post,
        load_rank=lambda _root, _season: None,
    )
    out = report_dir(tmp_path, 1)
    assert (out / "weekly.md").exists()
    assert (out / "weekly.parquet").exists()
    assert (tmp_path / "reports" / "season_to_date.parquet").exists()
    assert first.notified is True
    assert first.notify_skipped_reason is None
    assert len(posts) == 1
    md_first = (out / "weekly.md").read_text(encoding="utf-8")

    second = run_weekly_report(
        tmp_path,
        season="2024-25",
        gw=1,
        decisions=gw1,
        settings=settings,
        post=fake_post,
        load_rank=lambda _root, _season: None,
    )
    assert second.notified is False
    assert second.notify_skipped_reason == "already_notified"
    assert len(posts) == 1
    assert (out / "weekly.md").read_text(encoding="utf-8") == md_first
    assert second.content_sha256 == first.content_sha256


def test_notify_skipped_when_no_secret(tmp_path: Path) -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    posts: list[Request] = []

    def fake_post(request: Request) -> object:
        posts.append(request)
        return object()

    result = run_weekly_report(
        tmp_path,
        season="2024-25",
        gw=1,
        decisions=gw1,
        settings=Settings(),  # no ntfy / smtp
        post=fake_post,
        load_rank=lambda _root, _season: None,
    )
    assert result.notified is False
    assert result.notify_skipped_reason == "no_notify_channel"
    assert posts == []
    # Marker written so a later secret does not re-notify unchanged content.
    assert (report_dir(tmp_path, 1) / "notified.json").exists()
    again = run_weekly_report(
        tmp_path,
        season="2024-25",
        gw=1,
        decisions=gw1,
        settings=Settings(ntfy_url="https://ntfy.sh/later"),
        post=fake_post,
        load_rank=lambda _root, _season: None,
    )
    assert again.notified is False
    assert again.notify_skipped_reason == "already_notified"
    assert posts == []


def test_season_to_date_rollup_sums_weeks(tmp_path: Path) -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    gw2 = gw1.with_columns(pl.lit(2).cast(pl.Int64).alias("gw"))
    result = run_weekly_report(
        tmp_path,
        season="2024-25",
        gw=2,
        decisions=pl.concat([gw1, gw2]),
        notify=False,
        load_rank=lambda _root, _season: None,
    )
    from fplt.optimise.decision_scoring import score_gameweeks

    std = result.season_to_date
    weekly = score_gameweeks(pl.concat([gw1, gw2]))
    assert std.height == 1
    assert int(std["gameweeks"][0]) == 2
    assert int(std["through_gw"][0]) == 2
    assert int(std["season_points"][0]) == int(weekly["realised_points"].sum())
    assert float(std["plan_ev_sum"][0]) == float(weekly["plan_ev"].sum())


def test_format_weekly_markdown_includes_std_and_links() -> None:
    from fplt.optimise.decision_scoring import score_gameweeks

    gw1 = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        captain_index=0,
    )
    weekly = score_gameweeks(gw1)
    std = rollup_season_to_date(weekly, through_gw=1)
    text = format_weekly_markdown(
        weekly, std, links={"Leaderboard": "https://example.test/lb"}
    )
    assert "Weekly decision report" in text
    assert "Season to date" in text
    assert "Leaderboard" in text
    assert "Cap regret" in text
