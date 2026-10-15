"""One module per official job feed. Each exposes `async def fetch(client, account) -> list[Job]`."""

from __future__ import annotations

import httpx

USER_AGENT = "JobzhubClassifier/0.1 (+https://jobzhub.com; support@jobzhub.com)"


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=httpx.Timeout(45.0, connect=15.0),
        follow_redirects=True,
    )
