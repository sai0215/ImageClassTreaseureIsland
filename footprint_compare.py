"""Per-building damage-change scoring, adapted from HASTE's footprint-embedding
method (github.com/microsoft/haste, hastelib/src/hastegeo/workflows/embed_buildings.py).

HASTE's DINOv2 path: crop the raster to each building's bounding box, run DINOv2
on the crop at native resolution, mask the building polygon into the patch-token
grid, and average the masked tokens into one feature vector per building. HASTE
then trains a linear probe on a handful of labeled buildings.

We don't have damage labels, so instead of a linear probe we embed each footprint
in BOTH the pre- and post-event raster and use the cosine distance between the
two pooled vectors as an unsupervised per-building change score. This is the
footprint-scale analogue of dinov2_compare.py's dense tiled dissimilarity map,
except each crop is sized to one real building instead of an arbitrary tile —
so a small, real, localized change dominates its patch tokens instead of being
averaged away over an arbitrary 224px window.

Requires georeferenced rasters (GeoTIFF/COG with a real CRS + transform) for the
pre/post scenes — the flat preview PNGs in images/ have no geotransform and
can't be used here. Fill in GEO_PAIRS below once you have those rasters.
"""
import json

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
import torch

from dinov2_compare import PATCH, DEVICE, model, processor

FOOTPRINTS_PATH = "footprints/treasure_island_buildings.geojson"
OUT_DIR = "analysis_output"

# Fill in with georeferenced (GeoTIFF/COG) pre/post rasters once available.
# The flat PNGs in images/ have no geotransform and won't work here.
GEO_PAIRS = [
    # {
    #     "name": "Maxar_pre_vs_post_footprints",
    #     "title": "Maxar: pre vs post-Milton — per-building change",
    #     "pre_tif": "images_geo/Maxar_pre_20231104.tif",
    #     "post_tif": "images_geo/Maxar_post_20241010.tif",
    # },
]

MIN_CROP_PX = PATCH * 4     # smallest crop we'll embed a building from (4x4 patches)
PAD_FRAC = 0.25             # extra context padded around each building's bbox
BATCH_SIZE = 16


def _read_window_rgb(src, window):
    """Read a window from a rasterio dataset as HWC uint8 RGB."""
    arr = src.read(window=window, boundless=True, fill_value=0)
    if arr.shape[0] >= 3:
        arr = arr[:3]
    else:
        arr = np.repeat(arr[:1], 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return arr


def _crop_window_for_building(src, geom_bounds):
    minx, miny, maxx, maxy = geom_bounds
    row_start, col_start = src.index(minx, maxy)
    row_stop, col_stop = src.index(maxx, miny)
    row_start, row_stop = sorted([row_start, row_stop])
    col_start, col_stop = sorted([col_start, col_stop])

    h = max(row_stop - row_start, 1)
    w = max(col_stop - col_start, 1)
    pad_h = max(int(h * PAD_FRAC), (MIN_CROP_PX - h) // 2 + 1)
    pad_w = max(int(w * PAD_FRAC), (MIN_CROP_PX - w) // 2 + 1)

    row_start -= pad_h
    row_stop += pad_h
    col_start -= pad_w
    col_stop += pad_w

    size = max(row_stop - row_start, col_stop - col_start, MIN_CROP_PX)
    size -= size % PATCH
    size = max(size, PATCH)

    cy = (row_start + row_stop) // 2
    cx = (col_start + col_stop) // 2
    row_start, col_start = cy - size // 2, cx - size // 2

    return rasterio.windows.Window(col_start, row_start, size, size), size


def _polygon_patch_mask(geom, window, src_transform, size, grid):
    """Rasterize the building polygon at crop resolution, then max-pool down to
    the patch grid so a patch counts as "inside" if any of it overlaps the
    building (mirrors HASTE's token-grid masking)."""
    win_transform = rasterio.windows.transform(window, src_transform)
    px_mask = rasterio.features.rasterize(
        [(geom, 1)], out_shape=(size, size), transform=win_transform, fill=0, dtype=np.uint8
    )
    pooled = px_mask.reshape(grid, PATCH, grid, PATCH).max(axis=(1, 3))
    return pooled.astype(bool)


@torch.no_grad()
def _embed_crop(crop_rgb, size):
    inputs = processor(images=crop_rgb, do_resize=False, do_center_crop=False, return_tensors="pt").to(DEVICE)
    out = model(**inputs, interpolate_pos_encoding=True)
    grid = size // PATCH
    tokens = out.last_hidden_state[0, -grid * grid:, :]
    return tokens.reshape(grid, grid, -1).cpu().numpy(), grid


def _pooled_footprint_embedding(src, geom):
    window, size = _crop_window_for_building(src, geom.bounds)
    crop = _read_window_rgb(src, window)
    grid_feats, grid = _embed_crop(crop, size)
    patch_mask = _polygon_patch_mask(geom, window, src.transform, size, grid)
    if not patch_mask.any():
        patch_mask[grid // 2, grid // 2] = True  # degenerate polygon: fall back to center patch
    return grid_feats[patch_mask].mean(axis=0)


def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1 - np.dot(a, b))


def analyze_pair_footprints(pair, footprints):
    with rasterio.open(pair["pre_tif"]) as pre_src, rasterio.open(pair["post_tif"]) as post_src:
        fp_pre = footprints.to_crs(pre_src.crs)
        fp_post = footprints.to_crs(post_src.crs)

        b = pre_src.bounds
        in_bounds = fp_pre.geometry.apply(
            lambda g: g.bounds[0] >= b.left and g.bounds[2] <= b.right and g.bounds[1] >= b.bottom and g.bounds[3] <= b.top
        )

        scores = []
        for idx in footprints.index[in_bounds]:
            try:
                emb_pre = _pooled_footprint_embedding(pre_src, fp_pre.geometry.loc[idx])
                emb_post = _pooled_footprint_embedding(post_src, fp_post.geometry.loc[idx])
            except Exception as e:
                print(f"  skipping building {idx}: {e}")
                continue
            scores.append((idx, cosine_dist(emb_pre, emb_post)))

    result = footprints.loc[[i for i, _ in scores]].copy()
    result["change_score"] = [s for _, s in scores]
    out_path = f"{OUT_DIR}/{pair['name']}.geojson"
    result.to_file(out_path, driver="GeoJSON")
    print(f"  {len(result)} buildings scored -> {out_path}")
    return {
        "pair": pair["name"],
        "n_buildings_scored": len(result),
        "mean_change_score": float(result["change_score"].mean()) if len(result) else float("nan"),
        "footprints_geojson": out_path,
    }


if __name__ == "__main__":
    if not GEO_PAIRS:
        print(
            "GEO_PAIRS is empty — add georeferenced pre/post GeoTIFF paths for at "
            "least one scene before running this script. See the module docstring."
        )
        raise SystemExit(0)

    footprints = gpd.read_file(FOOTPRINTS_PATH)
    print(f"Loaded {len(footprints)} building footprints from {FOOTPRINTS_PATH}")

    all_metrics = []
    for pair in GEO_PAIRS:
        print(f"Footprint analysis: {pair['name']}...")
        all_metrics.append(analyze_pair_footprints(pair, footprints))

    with open(f"{OUT_DIR}/footprint_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved {OUT_DIR}/footprint_metrics.json")
