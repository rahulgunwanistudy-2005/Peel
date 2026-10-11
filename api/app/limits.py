"""Per-IP token bucket and upload size limits."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fastapi import Request, UploadFile
from peel.io import MAX_BYTES
from peel.types import ErrorCode, PeelError

from app.config import settings


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Classic token bucket: ``capacity`` requests in a burst, one more every ``refill`` seconds."""

    def __init__(self, capacity: int, refill_seconds: float) -> None:
        self.capacity = float(capacity)
        self.refill_seconds = refill_seconds
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def take(self, key: str, cost: float = 1.0) -> float:
        """Spend ``cost`` tokens. Returns 0 on success, else seconds until enough tokens exist."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = _Bucket(self.capacity, now)
            bucket.tokens = min(
                self.capacity, bucket.tokens + (now - bucket.updated) / self.refill_seconds
            )
            bucket.updated = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                if len(self._buckets) > 5000:
                    self._evict(now)
                return 0.0
            return (cost - bucket.tokens) * self.refill_seconds

    def _evict(self, now: float) -> None:
        stale = [k for k, b in self._buckets.items() if now - b.updated > 600]
        for key in stale:
            del self._buckets[key]


limiter = RateLimiter(settings.rate_capacity, settings.rate_refill_seconds)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(cost: float = 1.0):  # type: ignore[no-untyped-def]
    """FastAPI dependency factory. Heavy routes cost a full token, light ones a fraction."""

    def dependency(request: Request) -> None:
        wait = limiter.take(client_ip(request), cost)
        if wait > 0:
            raise PeelError(
                ErrorCode.RATE_LIMITED,
                f"Too many requests from your address. Try again in {max(1, round(wait))} seconds.",
                status=429,
            )

    return dependency


async def read_upload(upload: UploadFile, label: str) -> bytes:
    """Read at most the limit plus one byte, so an oversized upload is rejected without buffering it."""
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise PeelError(
            ErrorCode.IMAGE_TOO_LARGE, f"The {label} is over the 15 MB limit.", status=413
        )
    if not data:
        raise PeelError(ErrorCode.INVALID_IMAGE, f"The {label} is empty.")
    return data
