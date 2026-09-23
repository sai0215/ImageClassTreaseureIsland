#!/usr/bin/env bash
# Shared Docker entrypoint for every container built from this image (see
# Dockerfile). Which process actually runs is picked by the first argument,
# so the same image serves both containers in the pod (k8s-deploy.yml):
#   api container:    entrypoint.sh api
#   worker sidecar:   entrypoint.sh worker
# Anything else is exec'd as-is, so `docker run <image> bash` etc. still work.
set -euo pipefail

ROLE="${1:-api}"
shift || true

log() { echo "[entrypoint:${ROLE}] $*" >&2; }

# Fail fast (and loudly) instead of letting uvicorn/celery start against a
# broker that isn't reachable yet — matters most right after a fresh
# `docker compose up` / pod restart, before Azure Cache for Redis DNS/network
# has settled. Skipped entirely when REDIS_URL is unset, which is also how
# Celery decides to run tasks eagerly in-process (see celery_app.py) — no
# broker, nothing to wait for.
wait_for_redis() {
  if [ -z "${REDIS_URL:-}" ]; then
    log "REDIS_URL unset; running without a broker (Celery eager mode)"
    return 0
  fi

  log "waiting for Redis (${REDIS_URL%%@*}...) to accept connections"
  if ! python3 - <<'PYEOF'
import os
import sys
import time

import redis

deadline = time.time() + 30
last_err = None
while time.time() < deadline:
    try:
        redis.from_url(os.environ["REDIS_URL"]).ping()
        sys.exit(0)
    except Exception as exc:
        last_err = exc
        time.sleep(1)
print(f"redis never became reachable: {last_err}", file=sys.stderr)
sys.exit(1)
PYEOF
  then
    log "giving up waiting for Redis"
    exit 1
  fi
  log "Redis is reachable"
}

# Fail fast if Datadog tracing (tracer.py, imported first thing by main.py
# and tasks.py) is misconfigured — e.g. ddtrace/datadog not installed, or a
# bad DD_AGENT_HOST value blows up at import time — rather than finding out
# only once the first request/task tries to use it.
check_tracer() {
  log "checking Datadog tracer import (DD_AGENT_HOST=${DD_AGENT_HOST:-<unset, tracing disabled>})"
  python3 -c "import tracer" || {
    log "tracer.py failed to import; aborting"
    exit 1
  }
}

case "$ROLE" in
  api)
    check_tracer
    wait_for_redis
    log "starting uvicorn (FastAPI)"
    exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;

  worker)
    check_tracer
    wait_for_redis
    log "starting celery worker"
    exec celery -A celery_app worker \
      --loglevel="${CELERY_LOG_LEVEL:-info}" \
      --concurrency="${CELERY_CONCURRENCY:-2}" \
      --max-tasks-per-child="${CELERY_MAX_TASKS_PER_CHILD:-50}" \
      "$@"
    ;;

  beat)
    # Not wired into k8s-deploy.yml yet — here for when a periodic job (e.g.
    # nightly re-run + report rebuild) is added; run as its own single-replica
    # Deployment (Celery beat must never have more than one scheduler running).
    check_tracer
    wait_for_redis
    log "starting celery beat"
    exec celery -A celery_app beat --loglevel="${CELERY_LOG_LEVEL:-info}" "$@"
    ;;

  flower)
    # Celery monitoring UI: live task/worker view, retries, rate limits.
    # Run as its own single-replica Deployment + ClusterIP Service, not as a
    # third container in the api/worker pod, and put it behind the ingress
    # only with auth in front (it has no built-in login by default).
    check_tracer
    wait_for_redis
    log "starting flower on :${FLOWER_PORT:-5555}"
    flower_args=(--port="${FLOWER_PORT:-5555}")
    [ -n "${FLOWER_URL_PREFIX:-}" ] && flower_args+=(--url_prefix="${FLOWER_URL_PREFIX}")
    [ -n "${FLOWER_BASIC_AUTH:-}" ] && flower_args+=(--basic_auth="${FLOWER_BASIC_AUTH}")
    exec celery -A celery_app flower "${flower_args[@]}" "$@"
    ;;

  *)
    log "unrecognized role '${ROLE}', exec-ing it as a literal command"
    exec "$ROLE" "$@"
    ;;
esac
