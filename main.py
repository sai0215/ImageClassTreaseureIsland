"""FastAPI backend for the Treasure Island hurricane change-detection pipeline.

Wraps compare_images.py / dinov2_compare.py / combined_compare.py /
build_final_report.py (each a standalone script over a fixed PAIRS list, see
pipeline.py) behind a REST API. Analysis jobs are submitted as Celery tasks
(celery_app.py, tasks.py) so the actual SSIM/DINOv2 inference runs in the
"worker" sidecar container, not in this process — a slow multi-minute job
never risks this container's event loop or its liveness probe. Without
REDIS_URL configured, Celery runs tasks eagerly (in-process, synchronously),
so the API is still fully usable standalone for local dev with no broker.
"""
import tracer  # noqa: F401  -- must import first: patches FastAPI/redis before use

import json
import os
from typing import Optional

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from celery_app import celery_app
from pipeline import OUT_DIR, METRICS_PATH
from tasks import run_analysis_task

app = FastAPI(
    title="Treasure Island Change Detection API",
    description=(
        "Pre/post-hurricane satellite change-detection pipeline (ORB registration, "
        "SSIM, DINOv2 embeddings, per-building footprint scoring) exposed as a REST API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    pairs: Optional[list[str]] = None  # None = every pair defined in compare_images.PAIRS
    run_dinov2: bool = True
    run_combined: bool = True
    rebuild_report: bool = False


@app.get("/health/live", tags=["health"])
def liveness():
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
def readiness():
    if not OUT_DIR.exists():
        raise HTTPException(status_code=503, detail="analysis_output directory missing")
    broker_ok = True
    if os.getenv("REDIS_URL"):
        try:
            celery_app.connection().ensure_connection(max_retries=1)
        except Exception:
            broker_ok = False
    return {"status": "ready" if broker_ok else "degraded", "broker_connected": broker_ok}


@app.get("/pairs", tags=["pipeline"])
def list_pairs():
    from compare_images import PAIRS

    return [{"name": p["name"], "title": p["title"]} for p in PAIRS]


@app.get("/metrics", tags=["pipeline"])
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics computed yet. POST /analyze first.")
    return json.loads(METRICS_PATH.read_text())


@app.post("/analyze", tags=["pipeline"], status_code=202)
def analyze(req: AnalyzeRequest = AnalyzeRequest()):
    async_result = run_analysis_task.delay(req.model_dump())
    tracer.incr("analyze.queued", tags=[f"pairs:{len(req.pairs or [])}"])
    return {"job_id": async_result.id, "status": async_result.status}


@app.get("/analyze/{job_id}", tags=["pipeline"])
def analyze_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    body = {"job_id": job_id, "status": result.status}
    if result.status == "SUCCESS":
        body["result"] = result.result
    elif result.status == "FAILURE":
        body["error"] = str(result.result)
    return body


@app.get("/outputs/{filename}", tags=["pipeline"])
def get_output_file(filename: str):
    path = (OUT_DIR / filename).resolve()
    if path.parent != OUT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/report", tags=["pipeline"])
def get_report():
    report_path = OUT_DIR / "Final_Report.pdf"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not built yet. POST /analyze with rebuild_report=true.",
        )
    return FileResponse(report_path, media_type="application/pdf", filename="Final_Report.pdf")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD")),
    )
