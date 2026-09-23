"""Pure analysis pipeline logic — no FastAPI or Celery imports here, so it can
be called directly (tests, a REPL) or from the Celery task in tasks.py without
either module importing the other.
"""
import json
import subprocess
import sys
from pathlib import Path

# Must run before compare_images/dinov2_compare/combined_compare import
# matplotlib.pyplot: those scripts were written to run as a `python foo.py`
# main-thread process, but here they run inside a FastAPI threadpool worker
# thread or a Celery worker process, where matplotlib's default interactive
# backend (Mac's "macosx", or "Qt"/"Tk" elsewhere) either can't be created
# outside the main thread or has no display to attach to. "Agg" is the
# non-interactive, thread-safe raster backend — plots are still saved to PNG
# fine, they just never open a window (which nothing here wants anyway).
import matplotlib

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "analysis_output"
METRICS_PATH = OUT_DIR / "metrics.json"


def run_analysis(payload: dict) -> dict:
    """Run compare_images / dinov2_compare / combined_compare over the
    requested pairs, merge results into analysis_output/metrics.json, and
    optionally rebuild the PDF report. Returns the per-pair results computed
    in this call (not the full merged metrics.json)."""
    from compare_images import PAIRS, analyze_pair

    names = payload.get("pairs") or [p["name"] for p in PAIRS]
    pairs = [p for p in PAIRS if p["name"] in names]
    if not pairs:
        raise ValueError(f"No matching pairs for {names}")

    results = {}
    for pair in pairs:
        results[pair["name"]] = analyze_pair(pair)

    if payload.get("run_dinov2", True):
        from dinov2_compare import analyze_pair_dinov2

        for pair in pairs:
            results[pair["name"]].update(analyze_pair_dinov2(pair))

    if payload.get("run_combined", True):
        from combined_compare import analyze_pair_combined

        for pair in pairs:
            results[pair["name"]].update(analyze_pair_combined(pair))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if METRICS_PATH.exists():
        existing = {m["pair"]: m for m in json.loads(METRICS_PATH.read_text())}
    existing.update(results)
    METRICS_PATH.write_text(json.dumps(list(existing.values()), indent=2))

    if payload.get("rebuild_report", False):
        subprocess.run(
            [sys.executable, str(BASE_DIR / "build_final_report.py")],
            check=True,
            cwd=BASE_DIR,
        )

    return results
