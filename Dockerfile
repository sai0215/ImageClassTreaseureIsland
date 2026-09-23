# Treasure Island change-detection API image.
# Same image is used for both containers in the pod (see k8s-deploy.yml),
# dispatched by entrypoint.sh's first argument:
#   - "api" container    -> entrypoint.sh api    (uvicorn)
#   - "worker" sidecar    -> entrypoint.sh worker (celery worker)
#
# GDAL/rasterio/geopandas need native libgdal, and OpenCV needs libGL — both
# pulled in via apt before pip install so the wheels build/link cleanly.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgl1 \
        libglib2.0-0 \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x /app/entrypoint.sh

# Non-root: least-privilege container, required by most AKS pod security
# admission policies (Rancher's default PSP/PSA baseline included).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/analysis_output \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Only meaningful for the "api" role when run outside Kubernetes (e.g. plain
# `docker run`/compose) — under k8s, kubelet's own livenessProbe/readinessProbe
# in k8s-deploy.yml take over and this HEALTHCHECK is not consulted. The
# "worker" role doesn't serve HTTP, so this check is skipped for it via
# CELERY_WORKER env below.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD if [ "${CELERY_WORKER:-}" = "1" ]; then exit 0; fi; \
        python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
