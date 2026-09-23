"""Celery application: broker + result backend for offloading analysis jobs
from the API container to the worker sidecar (see k8s-deploy.yml, entrypoint.sh).

Both containers in a pod share this same module/image; only the process each
one runs differs (uvicorn vs. `celery -A celery_app worker`).
"""
import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL")

# No REDIS_URL (e.g. local dev with no Redis running) -> use an in-memory
# broker and run tasks eagerly (synchronously, in the caller's process), so
# `task.delay(...)` still works with no separate worker process required.
BROKER_URL = REDIS_URL or "memory://"
RESULT_BACKEND = REDIS_URL or "cache+memory://"

celery_app = Celery("imagedetection", broker=BROKER_URL, backend=RESULT_BACKEND)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
    # DINOv2/torch hold onto GPU/CPU memory per-process; recycling workers
    # periodically avoids slow unbounded growth across many jobs.
    worker_max_tasks_per_child=int(os.getenv("CELERY_MAX_TASKS_PER_CHILD", "50")),
    # False (the default) so that in eager mode (no REDIS_URL) a bad request
    # comes back as a normal FAILURE result from .delay(), same as the
    # non-eager path — not an exception raised straight out of the endpoint.
    task_always_eager=REDIS_URL is None,
    task_eager_propagates=False,
    # Without this, GET /analyze/{job_id} can't look the result back up by id
    # in eager mode — the task runs synchronously inside .delay() but its
    # result is otherwise discarded rather than written to the backend.
    task_store_eager_result=True,
)

# tasks.py registers its @celery_app.task on import; the celery CLI is pointed
# at "celery_app.celery_app" but still needs tasks.py imported for the worker
# process to know about the task, hence the explicit import (not autodiscovery
# — there's no tasks/ package here, just a flat tasks.py).
import tasks  # noqa: E402,F401
