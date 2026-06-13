"""Job queue abstraction for ARGUS inference workers.

Selected via QUEUE_BACKEND env var:
  memory — asyncio.Queue (default, single-process)
  sqs    — AWS SQS (stub, implemented in Phase 7)
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod


class JobQueue(ABC):
    """Abstract job queue that dispatches job IDs to workers."""

    @abstractmethod
    async def enqueue(self, job_id: str) -> None:
        """Add a job ID to the queue.

        Args:
            job_id: UUID string of the job to process.
        """

    @abstractmethod
    async def dequeue(self) -> str:
        """Block until a job ID is available and return it.

        Returns:
            The next job ID to process.
        """

    @abstractmethod
    async def clear(self) -> int:
        """Remove all pending items without processing them.

        Returns:
            Number of items removed.
        """


class InMemoryQueue(JobQueue):
    """In-process asyncio queue — single-process deployments only."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, job_id: str) -> None:
        await self._q.put(job_id)

    async def dequeue(self) -> str:
        return await self._q.get()

    async def clear(self) -> int:
        count = 0
        while True:
            try:
                self._q.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                return count


class SQSQueue(JobQueue):
    """AWS SQS queue backend — implemented in Phase 7."""

    async def enqueue(self, job_id: str) -> None:
        raise NotImplementedError("SQS queue not implemented until Phase 7")

    async def dequeue(self) -> str:
        raise NotImplementedError("SQS queue not implemented until Phase 7")

    async def clear(self) -> int:
        raise NotImplementedError("SQS queue not implemented until Phase 7")


def get_queue() -> JobQueue:
    """Factory: return queue backend selected by QUEUE_BACKEND env var.

    Returns:
        Configured JobQueue instance.

    Raises:
        ValueError: If QUEUE_BACKEND is not a known value.
    """
    backend = os.environ.get("QUEUE_BACKEND", "memory").lower()
    if backend == "memory":
        return InMemoryQueue()
    if backend == "sqs":
        return SQSQueue()
    raise ValueError(f"Unknown QUEUE_BACKEND: {backend!r}")


if __name__ == "__main__":
    import asyncio

    async def _smoke() -> None:
        q = InMemoryQueue()
        await q.enqueue("job-abc")
        result = await q.dequeue()
        assert result == "job-abc"
        print("InMemoryQueue smoke test passed.")

    asyncio.run(_smoke())
