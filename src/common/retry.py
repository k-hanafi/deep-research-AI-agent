"""
Async retry logic with exponential backoff.

Retries on transient errors (rate limits, server errors, timeouts)
but NOT on permanent errors (bad request, auth failure, JSON parse).
"""

import asyncio
import logging
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {
    429,  # Rate limit
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# Exception types that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


async def async_retry(
    fn: Callable[..., T],
    *args,
    max_retries: int = 3,
    delay_base: float = 1.0,
    operation_name: str = "",
    **kwargs,
) -> T:
    """
    Call an async function with exponential backoff on transient failures.

    Retry on:
    - HTTP 429 (rate limit), 500/502/503/504 (server error)
    - httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError

    Do NOT retry on:
    - HTTP 400 (bad request), 401 (auth), 404 (not found)
    - json.JSONDecodeError (response parse error)
    - Any other non-transient error

    Args:
        fn: The async callable to invoke.
        *args: Positional arguments passed to fn.
        max_retries: Maximum number of retries (0 = no retries, just one attempt).
        delay_base: Base delay in seconds; doubles each retry (1s, 2s, 4s, ...).
        operation_name: Human-readable label for log messages.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        The result of fn(*args, **kwargs).

    Raises:
        The last exception if all retries are exhausted.
    """
    label = operation_name or fn.__name__
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status not in RETRYABLE_STATUS_CODES:
                # Permanent error — don't retry
                raise

            last_exception = e
            if attempt < max_retries:
                delay = delay_base * (2 ** attempt)
                logger.warning(
                    "%s: HTTP %d on attempt %d/%d, retrying in %.1fs",
                    label, status, attempt + 1, max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "%s: HTTP %d — all %d attempts exhausted",
                    label, status, max_retries + 1,
                )

        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                delay = delay_base * (2 ** attempt)
                logger.warning(
                    "%s: %s on attempt %d/%d, retrying in %.1fs",
                    label, type(e).__name__, attempt + 1, max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "%s: %s — all %d attempts exhausted",
                    label, type(e).__name__, max_retries + 1,
                )

    # All retries exhausted — re-raise the last exception
    raise last_exception  # type: ignore[misc]
