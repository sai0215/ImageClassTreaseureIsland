"""Datadog APM tracing + custom metrics setup, shared by every process this
image runs (uvicorn, celery worker, celery beat, flower — see entrypoint.sh).

Import this before anything else so ddtrace's monkey-patching wraps
FastAPI/Starlette, Celery, and redis-py before those libraries are otherwise
imported. entrypoint.sh does a `python3 -c "import tracer"` preflight for
every role so a broken Datadog config fails the container fast, at startup,
instead of silently swallowing traces later.

Tracing is opt-in on DD_AGENT_HOST being set, so local dev / CI with no
Datadog Agent nearby just runs with tracing disabled rather than erroring.
"""
import os

import ddtrace
from datadog import initialize, statsd

DD_AGENT_HOST = os.getenv("DD_AGENT_HOST")
DD_ENABLED = bool(DD_AGENT_HOST) and os.getenv("DD_TRACE_ENABLED", "true").lower() != "false"

if DD_ENABLED:
    ddtrace.patch_all()
    ddtrace.config.fastapi["service_name"] = os.getenv("DD_SERVICE", "imagedetection-api")

    # DD_SERVICE / DD_ENV / DD_VERSION / DD_AGENT_HOST / DD_LOGS_INJECTION are
    # all read directly from the environment by ddtrace itself (see
    # k8s-deploy.yml); initialize() below only wires up the separate
    # dogstatsd client used for the custom counters/timers in incr()/timing().
    initialize(
        statsd_host=DD_AGENT_HOST,
        statsd_port=int(os.getenv("DD_DOGSTATSD_PORT", "8125")),
    )
else:
    ddtrace.tracer.enabled = False

tracer = ddtrace.tracer


def incr(metric: str, value: int = 1, tags: list | None = None) -> None:
    """Increment a dogstatsd counter. No-op when Datadog isn't configured
    (local dev), so call sites don't need their own DD_ENABLED checks."""
    if not DD_ENABLED:
        return
    statsd.increment(f"imagedetection.{metric}", value=value, tags=tags or [])


def timing(metric: str, value_ms: float, tags: list | None = None) -> None:
    """Record a dogstatsd timing (histogram of durations in ms)."""
    if not DD_ENABLED:
        return
    statsd.timing(f"imagedetection.{metric}", value_ms, tags=tags or [])
