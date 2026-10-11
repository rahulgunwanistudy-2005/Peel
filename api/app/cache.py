"""In-memory job store: 30-minute TTL, byte-budgeted LRU, content-hash lookup.

Nothing is written to disk. When a job expires its images are gone.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from peel.arrays import U8
from peel.pipeline import PeelOutput

from app.config import settings
from app.schemas import ChainResult, ErrorDetail, JobKind, JobState, PeelResult, ProgressEvent

MEMORY_BUDGET_BYTES = 1_500_000_000


@dataclass
class Job:
    id: str
    kind: JobKind
    state: JobState = "queued"
    progress: ProgressEvent = field(default_factory=lambda: ProgressEvent(stage="queued", pct=0.0))
    events: list[ProgressEvent] = field(default_factory=list)
    output: PeelOutput | None = None
    result: PeelResult | None = None
    chain: ChainResult | None = None
    error: ErrorDetail | None = None
    content_hash: str | None = None
    permanent: bool = False
    version: int = 0
    created: float = field(default_factory=time.monotonic)
    touched: float = field(default_factory=time.monotonic)
    # name -> function that renders the image; encoded bytes are memoized in ``encoded``.
    renderers: dict[str, Callable[[], U8]] = field(default_factory=dict)
    encoded: dict[tuple[str, int], bytes] = field(default_factory=dict)
    # Examples start with pre-rendered files and are analysed lazily, on the first recomposite.
    source: tuple[bytes, bytes, str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, stage: str, pct: float, label: str = "") -> None:
        event = ProgressEvent(stage=stage, pct=round(min(max(pct, 0.0), 1.0), 3), label=label)
        self.progress = event
        self.events.append(event)

    def size_bytes(self) -> int:
        total = sum(len(b) for b in self.encoded.values())
        if self.output is not None:
            a = self.output.analysis
            total += self.output.peeled.nbytes + a.original.nbytes + a.aligned.nbytes
        return total


class JobStore:
    def __init__(self, ttl_seconds: int, max_jobs: int) -> None:
        self.ttl = ttl_seconds
        self.max_jobs = max_jobs
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._by_hash: dict[str, str] = {}
        self._lock = threading.RLock()

    def create(self, kind: JobKind, *, job_id: str | None = None, permanent: bool = False) -> Job:
        job = Job(id=job_id or uuid.uuid4().hex[:16], kind=kind, permanent=permanent)
        with self._lock:
            self._jobs[job.id] = job
            self._purge()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._purge()
            job = self._jobs.get(job_id)
            if job is not None:
                job.touched = time.monotonic()
                self._jobs.move_to_end(job_id)
            return job

    def by_hash(self, content_hash: str) -> Job | None:
        with self._lock:
            job_id = self._by_hash.get(content_hash)
        job = self.get(job_id) if job_id else None
        return job if job is not None and job.state == "done" else None

    def index(self, job: Job, content_hash: str) -> None:
        with self._lock:
            job.content_hash = content_hash
            self._by_hash[content_hash] = job.id

    def permanent_jobs(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.permanent]

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def _drop(self, job_id: str) -> None:
        job = self._jobs.pop(job_id, None)
        if job is not None and job.content_hash and self._by_hash.get(job.content_hash) == job_id:
            del self._by_hash[job.content_hash]

    def _purge(self) -> None:
        now = time.monotonic()
        disposable = [j for j in self._jobs.values() if not j.permanent]
        for job in disposable:
            if now - job.touched > self.ttl:
                self._drop(job.id)
        disposable = [j for j in self._jobs.values() if not j.permanent and j.state != "running"]
        total = sum(j.size_bytes() for j in self._jobs.values())
        while disposable and (len(disposable) > self.max_jobs or total > MEMORY_BUDGET_BYTES):
            oldest = disposable.pop(0)  # OrderedDict order is least recently used first
            total -= oldest.size_bytes()
            self._drop(oldest.id)


store = JobStore(settings.job_ttl_seconds, settings.max_jobs)
