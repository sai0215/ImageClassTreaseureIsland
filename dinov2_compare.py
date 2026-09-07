import json
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
import cv2

from compare_images import PAIRS, IMG_DIR, OUT_DIR, load_rgb, valid_mask, align_b_to_a

MODEL_NAME = "facebook/dinov2-base"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Tile size matches DINOv2's native training resolution (224 = 16 * patch_size),
# so no position-embedding interpolation is needed. Stride < TILE gives overlap,
# which smooths tile-seam artifacts in the stitched map at the cost of more forward
# passes (STRIDE = TILE // 2 means ~4x the tiles of non-overlapping stitching).
TILE = 224
STRIDE = 112
BATCH_SIZE = 16

print(f"Loading {MODEL_NAME} on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
PATCH = model.config.patch_size  # 14


@torch.no_grad()
def embed_global(img_rgb_uint8):
    """Whole-image CLS embedding with no center crop, so the full frame (not just
    the center ~75-90% of it) contributes to the scene-level similarity score."""
    inputs = processor(images=img_rgb_uint8, do_center_crop=False, return_tensors="pt").to(DEVICE)
    out = model(**inputs, interpolate_pos_encoding=True)
    return out.last_hidden_state[0, 0].cpu().numpy()


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def patch_cosine_map(grid_a, grid_b):
    a = grid_a / (np.linalg.norm(grid_a, axis=-1, keepdims=True) + 1e-8)
    b = grid_b / (np.linalg.norm(grid_b, axis=-1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=-1)


def _tile_starts(dim, tile, stride):
    if dim <= tile:
        return [0]
    starts = list(range(0, dim - tile + 1, stride))
    last = dim - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def _patch_grids(crops, tile):
    """Embed a list of tile x tile x 3 crops at native resolution (no resize, no
    crop) and return their patch-token grids, shape (N, grid, grid, dim)."""
    grid = tile // PATCH
    grids = []
    for i in range(0, len(crops), BATCH_SIZE):
        chunk = crops[i:i + BATCH_SIZE]
        inputs = processor(images=chunk, do_resize=False, do_center_crop=False, return_tensors="pt").to(DEVICE)
        out = model(**inputs, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state
        patch_tokens = tokens[:, -grid * grid:, :].reshape(len(chunk), grid, grid, -1)
        grids.append(patch_tokens.cpu().numpy())
    return np.concatenate(grids, axis=0)


def dense_dissim_map(img_a, img_b):
    """Tile both (already pixel-aligned) images at native resolution, compute
    per-tile patch dissimilarity, and stitch (with averaging over overlaps) into
    a full-resolution map. Avoids both the center-crop content loss and the
    heavy single-pass upsample blur of a whole-image embed."""
    h, w = img_a.shape[:2]
    tile = min(TILE, h, w)
    tile -= tile % PATCH
    tile = max(tile, PATCH)

    ys = _tile_starts(h, tile, STRIDE)
    xs = _tile_starts(w, tile, STRIDE)
    coords = [(y, x) for y in ys for x in xs]

    crops_a = [img_a[y:y + tile, x:x + tile] for y, x in coords]
    crops_b = [img_b[y:y + tile, x:x + tile] for y, x in coords]

    grids_a = _patch_grids(crops_a, tile)
    grids_b = _patch_grids(crops_b, tile)

    dissim_sum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for (y, x), ga, gb in zip(coords, grids_a, grids_b):
        sim = patch_cosine_map(ga, gb)
        tile_dissim = cv2.resize((1 - sim).astype(np.float32), (tile, tile), interpolation=cv2.INTER_CUBIC)
        dissim_sum[y:y + tile, x:x + tile] += tile_dissim
        weight[y:y + tile, x:x + tile] += 1.0

    dissim_full = dissim_sum / np.maximum(weight, 1e-6)
    return np.clip(dissim_full, 0, None), len(coords), tile


def analyze_pair_dinov2(pair):
    a = load_rgb(f"{IMG_DIR}/{pair['a']}")
    b_raw = load_rgb(f"{IMG_DIR}/{pair['b']}")
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_raw, cv2.COLOR_RGB2GRAY)
    b_aligned, _, _ = align_b_to_a(a_gray, b_gray, b_raw)
    mask = valid_mask(a) & valid_mask(b_aligned)

    cls_a = embed_global(a)
    cls_b = embed_global(b_aligned)
    global_sim = cosine(cls_a, cls_b)

    dissim_full, n_tiles, tile_size = dense_dissim_map(a, b_aligned)
    dissim_masked = np.where(mask, dissim_full, 0)
    mean_patch_dissim = float(dissim_full[mask].mean()) if mask.any() else float("nan")

    metrics = {
        "dinov2_model": MODEL_NAME,
        "dinov2_global_cosine_sim": global_sim,
        "dinov2_mean_patch_dissim": mean_patch_dissim,
        "dinov2_tile_size": tile_size,
        "dinov2_tile_stride": STRIDE,
        "dinov2_n_tiles": n_tiles,
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    fig.suptitle(f"{pair['title']}  —  DINOv2 embedding similarity", fontsize=14, fontweight="bold")
    axes[0].imshow(a); axes[0].set_title("A (reference)", fontsize=9); axes[0].axis("off")
    axes[1].imshow(b_aligned); axes[1].set_title("B (aligned)", fontsize=9); axes[1].axis("off")
    im = axes[2].imshow(dissim_masked, cmap="viridis")
    axes[2].set_title(
        f"DINOv2 dense tiled dissimilarity (1-cos)\nglobal CLS cos-sim={global_sim:.3f}, "
        f"mean dissim={mean_patch_dissim:.3f}, {n_tiles} tiles @ {tile_size}px",
        fontsize=9,
    )
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = f"{OUT_DIR}/{pair['name']}_dinov2.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

    metrics["dinov2_figure"] = out_path
    return metrics


if __name__ == "__main__":
    with open(f"{OUT_DIR}/metrics.json") as f:
        existing = {m["pair"]: m for m in json.load(f)}

    for pair in PAIRS:
        print(f"DINOv2 analysis: {pair['name']}...")
        m = analyze_pair_dinov2(pair)
        existing[pair["name"]].update(m)
        print(json.dumps(m, indent=2))

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(list(existing.values()), f, indent=2)
    print(f"Updated {OUT_DIR}/metrics.json with DINOv2 metrics.")
