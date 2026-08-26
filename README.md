# Treasure Island Hurricane Change Detection

Pre/post-hurricane satellite image comparison for Treasure Island, FL, covering
Hurricanes Helene (Sept 2024) and Milton (Oct 2024). Three independent sources —
Maxar, NOAA NGS, and Sentinel-2 — are each registered, differenced, and scored with
two complementary similarity methods to identify and quantify shoreline/storm damage.

## Data

`images/` contains six source PNG previews, forming three before/after pairs:

| Pair | Before | After |
|---|---|---|
| Maxar (sub-meter) | `Maxar_pre_20231104_TreasureIsland_preview.png` | `Maxar_post_20241010_TreasureIsland_preview.png` |
| NOAA NGS (storm-on-storm) | `NOAA_postHelene_20240930_TreasureIsland_preview.png` | `NOAA_postMilton_20241011_TreasureIsland_preview.png` |
| Sentinel-2 (10–20 m/px) | `S2_pre_20240919_preview.png` | `S2_post_20241014_preview.png` |

## Method

1. **Co-registration** — ORB feature matching + RANSAC homography warps the "after"
   image onto the "before" image's frame, since the two dates rarely share an exact
   footprint.
2. **No-data masking** — near-black border/letterbox pixels are excluded from every
   statistic.
3. **SSIM (Structural Similarity)** — a classical pixel/texture measure, robust to
   lighting and tide differences. `1 − SSIM` gives a per-pixel dissimilarity map.
4. **Absolute RGB difference** — a raw per-pixel color delta, useful for reading
   turbidity/sediment color shifts directly.
5. **Otsu-thresholded change mask** — an automatic threshold on the SSIM
   dissimilarity map, converted into a "% area changed" figure.
6. **DINOv2 embedding similarity** — a self-supervised vision transformer
   (`facebook/dinov2-base`) embeds each image. Cosine similarity on the global
   `[CLS]` token gives one semantic "how similar overall" score per pair; cosine
   similarity on the 16×16 grid of patch tokens (upsampled) gives a coarse semantic
   change map. This catches things SSIM misses (e.g. cloud-occluded content reading
   as very different from clear ground) and vice versa.

## Scripts

Run in this order from the project root, with the venv active:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python compare_images.py       # SSIM / abs-diff / Otsu — writes analysis_output/*.png + metrics.json
python dinov2_compare.py       # DINOv2 similarity — appends to the same metrics.json
python build_final_report.py   # Final 2-page PDF report
```

| Script | Produces |
|---|---|
| `compare_images.py` | Per-pair 6-panel figures (`analysis_output/<pair>.png`) and `analysis_output/metrics.json` |
| `dinov2_compare.py` | Per-pair DINOv2 dissimilarity figures (`analysis_output/<pair>_dinov2.png`); updates `metrics.json` |
| `build_final_report.py` | `analysis_output/Final_Report.pdf` — a 2-page classic-style summary report |

## Output

**`analysis_output/Final_Report.pdf`** is the main deliverable:
- Page 1 — method summary, results table, and a bar chart comparing SSIM vs. DINOv2
  similarity across all three pairs
- Page 2 — a Before/After/Change thumbnail grid for each pair, plus a written
  conclusion

`analysis_output/metrics.json` holds every raw number (SSIM, % area changed, mean
RGB shift, DINOv2 cosine similarity) behind the report.

## Key finding

All three sources agree the open Gulf-facing beach/dune line took substantial damage
while interior canal neighborhoods stayed comparatively stable. The one place SSIM
and DINOv2 disagree is the Sentinel-2 pair: SSIM rates it the *most* similar of the
three (cloud texture still reads as texture to a pixel-structure metric), while
DINOv2 rates it the *least* similar (a semantic model penalizes cloud occlusion
heavily) — a reminder that these are relative indicators, not calibrated damage
measurements, and that combining a classical and a learned similarity method
surfaces disagreements neither would catch alone.
