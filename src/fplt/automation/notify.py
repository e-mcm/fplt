"""Push alerts via ntfy and optional SMTP. Missing config is a no-op."""

from __future__ import annotations

import smtplib
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any
from urllib.request import Request, urlopen

from fplt.settings import Settings

_Post = Callable[[Request], object]
_SmtpFactory = Callable[[str, int], Any]


def notify(
    url: str | None,
    title: str,
    message: str,
    post: _Post | None = None,
    *,
    priority: str | None = None,
) -> bool:
    """POST ``message`` to an ntfy topic.

    Args:
        url: Topic URL, for example ``https://ntfy.sh/fplt-alerts``. ``None``
            and ``""`` send nothing.
        title: Notification title. ntfy reads it from the ``Title`` header.
        message: Body text.
        post: HTTP sender. Defaults to ``urllib.request.urlopen``. Tests pass
            a fake so this never touches the network.
        priority: Optional ntfy ``Priority`` header (``5`` = urgent).

    Returns:
        True when a request was sent.
    """
    if not url:
        return False
    headers = {"Title": title, "Content-Type": "text/plain"}
    if priority:
        headers["Priority"] = priority
    request = Request(
        url,
        data=message.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    sender = urlopen if post is None else post
    sender(request)
    return True


def email(
    host: str | None,
    port: int,
    mail_from: str | None,
    mail_to: str | None,
    subject: str,
    body: str,
    *,
    username: str | None = None,
    password: str | None = None,
    use_tls: bool = True,
    smtp_factory: _SmtpFactory | None = None,
) -> bool:
    """Send ``body`` as a plain-text email over SMTP.

    Args:
        host: SMTP hostname. ``None`` and ``""`` send nothing.
        port: SMTP port (typically 587 with STARTTLS).
        mail_from: Envelope and ``From`` address. Required to send.
        mail_to: Recipient address. Required to send.
        subject: Message subject.
        body: Plain-text body (shared with the ntfy path).
        username: Optional SMTP login user.
        password: Optional SMTP login password.
        use_tls: Call ``starttls`` before login/send when True.
        smtp_factory: Builds an SMTP client. Defaults to ``smtplib.SMTP``.
            Tests pass a fake so this never touches the network.

    Returns:
        True when a message was handed to the SMTP client.
    """
    if not host or not mail_from or not mail_to:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(body)
    factory: _SmtpFactory = smtp_factory or smtplib.SMTP
    with factory(host, port) as client:
        if use_tls:
            client.starttls()
        if username:
            client.login(username, password or "")
        client.send_message(msg)
    return True


def notify_recommendation(
    settings: Settings,
    title: str,
    message: str,
    *,
    post: _Post | None = None,
    smtp_factory: _SmtpFactory | None = None,
    priority: str | None = None,
) -> bool:
    """Push a plan report via ntfy and/or email when those channels are set.

    Uses the same ``message`` string for both channels. Missing ``ntfy_url`` or
    incomplete SMTP settings leave that channel as a no-op.

    Args:
        settings: Process settings (``ntfy_url`` and ``smtp_*`` fields).
        title: ntfy title / email subject.
        message: Report body, typically concatenated ``report.md`` text.
        post: Optional ntfy HTTP fake for tests.
        smtp_factory: Optional SMTP fake for tests.
        priority: Optional ntfy ``Priority`` (``5`` = urgent for no-plan alarm).

    Returns:
        True when at least one channel sent.
    """
    ntfy_sent = notify(settings.ntfy_url, title, message, post=post, priority=priority)
    email_sent = email(
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_from,
        settings.smtp_to,
        title,
        message,
        username=settings.smtp_user,
        password=settings.smtp_password,
        use_tls=settings.smtp_use_tls,
        smtp_factory=smtp_factory,
    )
    return ntfy_sent or email_sent
