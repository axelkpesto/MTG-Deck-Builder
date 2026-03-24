"""Redis-backed worker for asynchronous deck-generation jobs."""
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv()

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import vector_db_server as server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deckgen-worker")

REDIS_URL = os.environ["REDIS_URL"]
QUEUE_KEY = os.environ.get("DECKGEN_QUEUE_KEY", "deckgen:queue")
JOB_PREFIX = os.environ.get("DECKGEN_JOB_PREFIX", "deckgen:job:")
JOB_TTL_SECONDS = int(os.environ.get("DECKGEN_JOB_TTL_SECONDS", "86400"))
POLL_TIMEOUT_SECONDS = 5

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def job_key(job_id: str) -> str:
    """Return the Redis key used to store one job."""
    return f"{JOB_PREFIX}{job_id}"


def update_job(job_id: str, **fields: Any) -> None:
    """Write job fields and refresh TTL."""
    payload = {k: "" if v is None else str(v) for k, v in fields.items()}
    payload["updated_at"] = str(int(time.time()))
    key = job_key(job_id)
    redis_client.hset(key, mapping=payload)
    redis_client.expire(key, JOB_TTL_SECONDS)


def mark_failed(job_id: str, message: str, tb: str = "") -> None:
    """Persist a failed job state."""
    update_job(job_id, status="failed", error=message, traceback=tb)


def process_job(job_id: str) -> None:
    """Run deck generation for a single queued job."""
    key = job_key(job_id)
    job = redis_client.hgetall(key)
    if not job:
        logger.warning("Skipping missing job %s", job_id)
        return

    commander = (job.get("commander") or "").strip()
    if not commander:
        mark_failed(job_id, "Missing commander name in job payload")
        return

    logger.info("Starting job %s for commander %s", job_id, commander)
    update_job(job_id, status="running", error="", started_at=str(int(time.time())))

    try:
        resolved_commander = server.resolve_card_id(commander)
        result = server.gen.generate(resolved_commander)
    except KeyError:
        mark_failed(job_id, f"Commander not found: {commander}")
        return
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("Deck generation job %s failed", job_id)
        mark_failed(job_id, str(exc), tb)
        return

    update_job(job_id, status="done", result=json.dumps(result), error="")
    logger.info("Completed job %s", job_id)


def main() -> None:
    """Consume queued deck-generation jobs forever."""
    logger.info("Deckgen worker started. Waiting on %s", QUEUE_KEY)
    while True:
        try:
            item = redis_client.blpop(QUEUE_KEY, timeout=POLL_TIMEOUT_SECONDS)
        except redis.RedisError as exc:
            logger.warning("Redis pop failed: %s", exc)
            time.sleep(2)
            continue

        if item is None:
            continue

        _, job_id = item
        process_job(job_id)


if __name__ == "__main__":
    main()
