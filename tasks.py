"""Celery task definitions. Kept separate from celery_app.py so the Celery
app object and its task registry have one clear owner each."""
import time

import tracer  # noqa: F401  -- must import first: patches Celery before use
from celery_app import celery_app
from pipeline import run_analysis


@celery_app.task(bind=True, name="imagedetection.run_analysis", acks_late=True)
def run_analysis_task(self, payload: dict) -> dict:
    started = time.monotonic()
    try:
        result = run_analysis(payload)
        tracer.incr("analyze.succeeded")
        return result
    except Exception:
        tracer.incr("analyze.failed")
        raise
    finally:
        tracer.timing("analyze.duration_ms", (time.monotonic() - started) * 1000)
