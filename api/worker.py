"""Standalone worker entry point for GPU inference containers.

Consumes job IDs from the queue (SQS in production), runs the inference
pipeline, and writes results to the shared database.  The HTTP API is
handled by a separate container; this process only processes jobs.

Run with:
    python -m api.worker
"""

from __future__ import annotations

import asyncio
import logging
import os

from db.models import get_engine, get_session_factory, init_db

logger = logging.getLogger(__name__)


async def _main() -> None:
    from api.queue import get_queue
    from api.storage import get_storage
    from api.main import _process_job

    engine = get_engine()
    await init_db(engine)
    session_factory = get_session_factory(engine)
    storage = get_storage()
    queue = get_queue()

    # Minimal FastAPI-like state object so _process_job can be reused
    class _State:
        pass

    class _App:
        state = _State()

    app = _App()
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.queue = queue
    app.state.model_loaded = False

    logger.info("Worker started — waiting for jobs (QUEUE_BACKEND=%s)", os.environ.get("QUEUE_BACKEND", "memory"))

    while True:
        job_id = await queue.dequeue()
        logger.info("Processing job %s", job_id)
        try:
            await _process_job(job_id, app)
            logger.info("Job %s complete", job_id)
        except Exception:
            logger.exception("Job %s failed", job_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    asyncio.run(_main())
