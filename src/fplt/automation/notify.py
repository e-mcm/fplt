"""Send a run-failure alert. With no URL configured, alerting is a no-op."""

from __future__ import annotations

from collections.abc import Callable
from urllib.request import Request, urlopen

_Post = Callable[[Request], object]


def notify(
    url: str | None,
    title: str,
    message: str,
    post: _Post | None = None,
) -> bool:
    """POST ``message`` to an ntfy topic.

    Args:
        url: Topic URL, for example ``https://ntfy.sh/fplt-alerts``. ``None``
            and ``""`` send nothing.
        title: Notification title. ntfy reads it from the ``Title`` header.
        message: Body text.
        post: HTTP sender. Defaults to ``urllib.request.urlopen``. Tests pass
            a fake so this never touches the network.

    Returns:
        True when a request was sent.
    """
    if not url:
        return False
    request = Request(
        url,
        data=message.encode("utf-8"),
        headers={"Title": title, "Content-Type": "text/plain"},
        method="POST",
    )
    sender = urlopen if post is None else post
    sender(request)
    return True
