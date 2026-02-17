"""
Website Health Check

Checks if a company's homepage is alive using HEAD requests.
Used as a quick signal for online presence before running Tavily search.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from ..config import PROCESSING


@dataclass
class WebsiteStatus:
    """
    Result of checking a company's website.
    
    Attributes:
        url: The URL that was checked.
        is_alive: Whether the site responded successfully.
        status_code: HTTP status code (if any response received).
        final_url: URL after redirects (if different from original).
        response_time_ms: How long the request took.
        error: Error message if the check failed.
    """
    url: str
    is_alive: bool
    status_code: Optional[int] = None
    final_url: Optional[str] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def is_redirect(self) -> bool:
        """Did the URL redirect to a different domain?"""
        if not self.final_url or not self.url:
            return False
        
        original_domain = urlparse(self.url).netloc.lower()
        final_domain = urlparse(self.final_url).netloc.lower()
        
        # Strip www. for comparison
        original_domain = original_domain.replace("www.", "")
        final_domain = final_domain.replace("www.", "")
        
        return original_domain != final_domain


async def check_website(
    url: str,
    timeout: float = PROCESSING.http_timeout,
    follow_redirects: bool = True,
    client: Optional[httpx.AsyncClient] = None,
) -> WebsiteStatus:
    """
    Check if a website is alive using a HEAD request.
    
    Uses HEAD instead of GET because:
    - Faster (no response body)
    - Lower bandwidth
    - Sufficient to check if site is up
    
    Args:
        url: The URL to check.
        timeout: Request timeout in seconds.
        follow_redirects: Whether to follow redirects.
        client: Optional shared httpx.AsyncClient for connection pooling.
    
    Returns:
        WebsiteStatus with check results.
    """
    # Normalize URL
    if not url:
        return WebsiteStatus(
            url=url or "",
            is_alive=False,
            error="No URL provided"
        )
    
    # Ensure URL has scheme
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    
    async def _do_check(c: httpx.AsyncClient) -> WebsiteStatus:
        try:
            # Try HEAD first (faster)
            start_time = asyncio.get_event_loop().time()
            response = await c.head(url)
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Some servers don't support HEAD, fall back to GET
            if response.status_code == 405:  # Method Not Allowed
                start_time = asyncio.get_event_loop().time()
                response = await c.get(url)
                elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Consider 2xx and 3xx as "alive"
            is_alive = 200 <= response.status_code < 400
            
            return WebsiteStatus(
                url=url,
                is_alive=is_alive,
                status_code=response.status_code,
                final_url=str(response.url) if response.url != url else None,
                response_time_ms=round(elapsed_ms, 1),
            )
            
        except httpx.TimeoutException:
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error="Timeout"
            )
        except httpx.ConnectError as e:
            # DNS failure, connection refused, etc.
            error_msg = str(e)
            if "Name or service not known" in error_msg or "getaddrinfo failed" in error_msg:
                error_msg = "DNS lookup failed"
            elif "Connection refused" in error_msg:
                error_msg = "Connection refused"
            else:
                error_msg = "Connection failed"
            
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error=error_msg
            )
        except httpx.TooManyRedirects:
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error="Too many redirects"
            )
        except Exception as e:
            # SSL errors, other network issues
            error_type = type(e).__name__
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error=f"{error_type}: {str(e)[:100]}"
            )
    
    if client is not None:
        return await _do_check(client)
    
    # Fallback: create a one-off client (backward compat for simple scripts)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://ubc.ca)",
            "Accept": "text/html,application/xhtml+xml",
        }
    ) as new_client:
        return await _do_check(new_client)


async def check_websites_batch(
    urls: list[str],
    max_concurrent: int = PROCESSING.max_concurrent_requests
) -> list[WebsiteStatus]:
    """
    Check multiple websites concurrently.
    
    Uses semaphore to limit concurrency and avoid overwhelming
    network/getting rate limited.
    
    Args:
        urls: List of URLs to check.
        max_concurrent: Maximum simultaneous requests.
    
    Returns:
        List of WebsiteStatus in same order as input URLs.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def check_with_semaphore(url: str) -> WebsiteStatus:
        async with semaphore:
            return await check_website(url)
    
    tasks = [check_with_semaphore(url) for url in urls]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────────────────────
# SYNCHRONOUS WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def check_website_sync(url: str) -> WebsiteStatus:
    """Synchronous wrapper for check_website."""
    return asyncio.run(check_website(url))


def check_websites_batch_sync(urls: list[str]) -> list[WebsiteStatus]:
    """Synchronous wrapper for check_websites_batch."""
    return asyncio.run(check_websites_batch(urls))
