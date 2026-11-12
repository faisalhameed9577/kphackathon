"""HTTP helper with timeouts, retries and exponential back-off for the research APIs."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

import requests

from utils.helpers import get_logger, redact_secrets

log = get_logger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ResearchAPIError(Exception):
    """Raised when a research API cannot be reached or returns an error."""


def request_with_retry(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]],
    *,
    source: str,
    timeout: int,
    max_retries: int,
    backoff: float,
    accept_statuses: Iterable[int] = (),
    secrets: Iterable[str] = (),
) -> requests.Response:
    """
    GET ``url`` with a bounded number of retries.

    * Timeouts, connection errors and HTTP 429/5xx are retried (exponential back-off).
    * Other HTTP errors raise ``ResearchAPIError`` immediately.
    * ``accept_statuses`` lets a caller handle e.g. 404 itself.
    * Error messages never contain the URL, so API keys cannot leak.
    """
    secrets = tuple(secrets)
    accept = set(accept_statuses)
    last_problem = "no response"

    for attempt in range(max(1, max_retries)):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_problem = f"network problem ({type(exc).__name__})"
            log.warning("%s attempt %d failed: %s", source, attempt + 1, redact_secrets(exc, secrets))
        else:
            if response.status_code in accept or response.status_code < 400:
                return response
            if response.status_code in RETRYABLE_STATUS:
                last_problem = f"HTTP {response.status_code}"
                log.warning("%s attempt %d got HTTP %s", source, attempt + 1, response.status_code)
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit() and attempt < max_retries - 1:
                    time.sleep(min(int(retry_after), 30))
                    continue
            else:
                raise ResearchAPIError(f"{source} returned HTTP {response.status_code}.")

        if attempt < max_retries - 1:
            time.sleep(backoff * (2 ** attempt))

    raise ResearchAPIError(
        f"{source} is not responding ({last_problem}) after {max_retries} attempts. "
        "Check your internet connection or try again in a minute."
    )
