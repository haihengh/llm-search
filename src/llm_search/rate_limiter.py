"""Token bucket rate limiter per client IP."""

import time
from threading import Lock
from typing import Optional

from .config import settings


class TokenBucket:
    """Single token bucket — refills at `rate` tokens per second, max `capacity`."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed, False if denied."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


class RateLimiter:
    """Per-client-IP rate limiter using token buckets."""

    def __init__(self, per_minute: Optional[int] = None):
        limit = per_minute if per_minute is not None else settings.rate_limit_per_minute
        self._rate = limit / 60.0  # tokens per second
        self._capacity = limit  # burst capacity
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()
        self._hits: int = 0

    def _get_bucket(self, client_id: str) -> TokenBucket:
        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(
                    rate=self._rate, capacity=self._capacity
                )
            return self._buckets[client_id]

    def is_allowed(self, client_id: str) -> bool:
        bucket = self._get_bucket(client_id)
        allowed = bucket.consume(1)
        if not allowed:
            self._hits += 1
        return allowed

    @property
    def hits(self) -> int:
        return self._hits


# Module-level singleton
rate_limiter = RateLimiter()
