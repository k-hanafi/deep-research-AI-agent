"""
Async Rate Limiter

Enforces requests-per-minute (RPM) limits using a sliding window.
Designed for use with Tavily and OpenAI API calls at scale.

Usage:
    limiter = AsyncRateLimiter(rpm=950)

    # Before each API call:
    await limiter.acquire()
    response = await client.post(...)

The limiter is safe for concurrent use with asyncio.gather().
"""

import asyncio
import time
from collections import deque


class AsyncRateLimiter:
    """
    Sliding-window rate limiter for async API calls.
    
    Tracks request timestamps over a 60-second window and blocks
    when the limit would be exceeded. Thread-safe via asyncio.Lock.
    
    Args:
        rpm: Maximum requests per minute.
        name: Optional label for logging (e.g. "tavily", "openai").
    """
    
    def __init__(self, rpm: int, name: str = ""):
        self.rpm = rpm
        self.name = name
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._total_wait_seconds = 0.0
    
    async def acquire(self) -> None:
        """
        Wait until a request slot is available, then reserve it.
        
        Call this before every API request. If the RPM limit has been
        reached, this will sleep until the oldest request in the window
        expires (i.e., falls outside the 60-second window).
        """
        async with self._lock:
            now = time.monotonic()
            
            # Evict timestamps older than 60 seconds
            while self._timestamps and now - self._timestamps[0] >= 60.0:
                self._timestamps.popleft()
            
            # If at capacity, wait for the oldest request to expire
            if len(self._timestamps) >= self.rpm:
                wait = 60.0 - (now - self._timestamps[0]) + 0.01  # small buffer
                if wait > 0:
                    self._total_wait_seconds += wait
                    await asyncio.sleep(wait)
                # Re-evict after sleeping
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
            
            self._timestamps.append(time.monotonic())
            self._total_requests += 1
    
    @property
    def current_window_count(self) -> int:
        """Number of requests in the current 60-second window."""
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60.0:
            self._timestamps.popleft()
        return len(self._timestamps)
    
    @property
    def stats(self) -> dict:
        """Return usage statistics for monitoring."""
        return {
            "name": self.name,
            "rpm_limit": self.rpm,
            "total_requests": self._total_requests,
            "total_wait_seconds": round(self._total_wait_seconds, 2),
            "current_window": self.current_window_count,
        }
    
    def __repr__(self) -> str:
        return (
            f"AsyncRateLimiter(name={self.name!r}, rpm={self.rpm}, "
            f"requests={self._total_requests}, waited={self._total_wait_seconds:.1f}s)"
        )
