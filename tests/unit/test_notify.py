"""ntfy and SMTP alerts no-op without config and accept injectable fakes."""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
from urllib.request import Request

import dagster as dg
import pytest

from fplt.automation.notify import email, notify, notify_recommendation
from fplt.orchestration.assets import pre_deadline_plan, recommendation_report
from fplt.settings import Settings, get_settings


class _FakeSmtp:
    """In-memory SMTP stand-in for :func:`email` tests."""

    instances: list[_FakeSmtp] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.tls = False
        self.login_args: tuple[str, str] | None = None
        self.sent: EmailMessage | None = None
        type(self).instances.append(self)

    def starttls(self) -> None:
        self.tls = True

    def login(self, user: str, password: str) -> None:
        self.login_args = (user, password)

    def send_message(self, msg: EmailMessage) -> None:
        self.sent = msg

    def __enter__(self) -> _FakeSmtp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_notify_posts_only_when_a_url_is_set() -> None:
    sent: list[Request] = []

    def post(request: Request) -> object:
        sent.append(request)
        return None

    assert notify(None, "title", "body", post) is False
    assert notify("", "title", "body", post) is False
    assert sent == []
    assert notify("https://ntfy.sh/fplt", "title", "hello", post) is True
    assert sent[0].data == b"hello"
    assert sent[0].get_header("Title") == "title"
    assert sent[0].get_header("Priority") is None
    assert (
        notify(
            "https://ntfy.sh/fplt",
            "urgent",
            "missing plan",
            post,
            priority="5",
        )
        is True
    )
    assert sent[1].get_header("Priority") == "5"


def test_email_is_noop_without_host_from_or_to() -> None:
    _FakeSmtp.instances.clear()
    fake = _FakeSmtp
    assert email(None, 587, "a@b.c", "d@e.f", "s", "body", smtp_factory=fake) is False
    assert email("smtp", 587, None, "d@e.f", "s", "body", smtp_factory=fake) is False
    assert email("smtp", 587, "a@b.c", None, "s", "body", smtp_factory=fake) is False
    assert email("", 587, "a@b.c", "d@e.f", "s", "body", smtp_factory=fake) is False
    assert _FakeSmtp.instances == []


def test_email_sends_plain_text_over_tls_with_login() -> None:
    _FakeSmtp.instances.clear()
    assert (
        email(
            "smtp.example.com",
            587,
            "from@example.com",
            "to@example.com",
            "subject",
            "report body",
            username="user",
            password="secret",
            use_tls=True,
            smtp_factory=_FakeSmtp,
        )
        is True
    )
    client = _FakeSmtp.instances[0]
    assert client.host == "smtp.example.com"
    assert client.port == 587
    assert client.tls is True
    assert client.login_args == ("user", "secret")
    assert client.sent is not None
    assert client.sent["Subject"] == "subject"
    assert client.sent["From"] == "from@example.com"
    assert client.sent["To"] == "to@example.com"
    assert client.sent.get_content().strip() == "report body"


def test_notify_recommendation_shares_message_across_channels() -> None:
    _FakeSmtp.instances.clear()
    posts: list[Request] = []

    def post(request: Request) -> object:
        posts.append(request)
        return None

    settings = Settings(
        ntfy_url="https://ntfy.sh/fplt",
        smtp_host="smtp.example.com",
        smtp_from="from@example.com",
        smtp_to="to@example.com",
    )
    assert (
        notify_recommendation(
            settings,
            "fplt recommendation",
            "shared body",
            post=post,
            smtp_factory=_FakeSmtp,
        )
        is True
    )
    assert posts[0].data == b"shared body"
    assert _FakeSmtp.instances[0].sent is not None
    assert _FakeSmtp.instances[0].sent.get_content().strip() == "shared body"


def test_notify_recommendation_noop_when_nothing_configured() -> None:
    _FakeSmtp.instances.clear()
    posts: list[Request] = []

    def post(request: Request) -> object:
        posts.append(request)
        return None

    assert (
        notify_recommendation(
            Settings(),
            "title",
            "body",
            post=post,
            smtp_factory=_FakeSmtp,
        )
        is False
    )
    assert posts == []
    assert _FakeSmtp.instances == []


def test_recommendation_report_notifies_only_when_plans_were_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    calls: list[tuple[str, str]] = []

    def fake_notify(settings: Settings, title: str, message: str, **_: object) -> bool:
        calls.append((title, message))
        return True

    monkeypatch.setattr("fplt.orchestration.assets.notify_recommendation", fake_notify)
    monkeypatch.setattr(
        "fplt.orchestration.assets.catch_up",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr("fplt.orchestration.assets.reports", lambda _d: "")
    result = dg.materialize([recommendation_report])
    assert result.success
    assert calls == []

    key = type("K", (), {"season": "2024-25", "gw": 1})()
    monkeypatch.setattr(
        "fplt.orchestration.assets.catch_up",
        lambda *_a, **_k: [key],
    )
    monkeypatch.setattr(
        "fplt.orchestration.assets.reports", lambda _d: "# GW 1\ntransfers"
    )
    result = dg.materialize([recommendation_report])
    assert result.success
    assert calls == [("fplt recommendation", "# GW 1\ntransfers")]
    get_settings.cache_clear()


def test_pre_deadline_plan_notifies_only_when_a_deadline_was_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    calls: list[str] = []

    def fake_notify(settings: Settings, title: str, message: str, **_: object) -> bool:
        calls.append(message)
        return True

    monkeypatch.setattr("fplt.orchestration.assets.notify_recommendation", fake_notify)
    monkeypatch.setattr(
        "fplt.orchestration.assets.pre_deadline",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("fplt.orchestration.assets.reports", lambda _d: "old")
    result = dg.materialize([pre_deadline_plan])
    assert result.success
    assert calls == []

    key = type("K", (), {"season": "2024-25", "gw": 3})()
    monkeypatch.setattr(
        "fplt.orchestration.assets.pre_deadline",
        lambda *_a, **_k: key,
    )
    monkeypatch.setattr("fplt.orchestration.assets.reports", lambda _d: "# GW 3\nxi")
    result = dg.materialize([pre_deadline_plan])
    assert result.success
    assert calls == ["# GW 3\nxi"]
    get_settings.cache_clear()
