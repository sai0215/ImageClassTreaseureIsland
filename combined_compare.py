import json
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

from compare_images import PAIRS, IMG_DIR, OUT_DIR, load_rgb, valid_mask, align_b_to_a
from dinov2_compare import dense_dissim_map

# Colors for the overlay panel (RGB, 0-255)
COLOR_CONFIRMED = [255, 0, 0]      # both signals agree: most likely a real localized change
COLOR_FINE_ONLY = [0, 150, 255]    # SSIM flags it, DINOv2 doesn't: likely noise/misalignment/lighting
COLOR_SEMANTIC_ONLY = [255, 200, 0]  # DINOv2 flags it, SSIM doesn't: broad texture/context shift (e.g. water)


def otsu_mask(values_2d, valid_2d):
    valid_vals = values_2d[valid_2d]
    if valid_vals.size == 0 or valid_vals.max() <= valid_vals.min():
        return np.zeros_like(valid_2d)
    t = threshold_otsu(valid_vals)
    return (values_2d > t) & valid_2d


def analyze_pair_combined(pair):
    a = load_rgb(f"{IMG_DIR}/{pair['a']}")
    b_raw = load_rgb(f"{IMG_DIR}/{pair['b']}")
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_raw, cv2.COLOR_RGB2GRAY)
    b_aligned, _, _ = align_b_to_a(a_gray, b_gray, b_raw)
    mask = valid_mask(a) & valid_mask(b_aligned)

    # Fine-grained candidate detector: native-resolution windowed SSIM.
    # Sensitive to small localized pixel changes, but also to noise/misalignment/lighting.
    b_gray_aligned = cv2.cvtColor(b_aligned, cv2.COLOR_RGB2GRAY)
    _, ssim_map = ssim(a_gray, b_gray_aligned, full=True)
    ssim_dissim = np.clip(1 - ssim_map, 0, None)

    # Semantic filter: full-resolution stitched DINOv2 patch dissimilarity.
    # Robust to noise/lighting, but blind to changes much smaller than a patch.
    dino_dissim, n_tiles, tile_size = dense_dissim_map(a, b_aligned)

    ssim_candidates = otsu_mask(ssim_dissim, mask)
    dino_candidates = otsu_mask(dino_dissim, mask)

    confirmed = ssim_candidates & dino_candidates
    fine_only = ssim_candidates & ~dino_candidates
    semantic_only = dino_candidates & ~ssim_candidates

    valid_n = int(mask.sum())
    metrics = {
        "pair": pair["name"],
        "combined_tile_size": tile_size,
        "combined_n_tiles": n_tiles,
        "pct_ssim_candidates": 100.0 * int(ssim_candidates.sum()) / valid_n if valid_n else float("nan"),
        "pct_dino_candidates": 100.0 * int(dino_candidates.sum()) / valid_n if valid_n else float("nan"),
        "pct_confirmed_change": 100.0 * int(confirmed.sum()) / valid_n if valid_n else float("nan"),
        "pct_fine_only_likely_noise": 100.0 * int(fine_only.sum()) / valid_n if valid_n else float("nan"),
        "pct_semantic_only_broad": 100.0 * int(semantic_only.sum()) / valid_n if valid_n else float("nan"),
    }

    overlay = a.copy()
    overlay[semantic_only] = COLOR_SEMANTIC_ONLY
    overlay[fine_only] = COLOR_FINE_ONLY
    overlay[confirmed] = COLOR_CONFIRMED

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2))
    fig.suptitle(f"{pair['title']}  —  combined SSIM (fine) + DINOv2 (semantic) change detection", fontsize=13, fontweight="bold")

    axes[0].imshow(a); axes[0].set_title("A (reference)", fontsize=9); axes[0].axis("off")

    im1 = axes[1].imshow(np.where(mask, ssim_dissim, 0), cmap="inferno")
    axes[1].set_title(f"SSIM dissimilarity (fine)\n{metrics['pct_ssim_candidates']:.1f}% flagged", fontsize=9)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(np.where(mask, dino_dissim, 0), cmap="viridis")
    axes[2].set_title(f"DINOv2 dissimilarity (semantic)\n{metrics['pct_dino_candidates']:.1f}% flagged", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Confirmed (red)={metrics['pct_confirmed_change']:.1f}%  "
        f"Fine-only/likely noise (blue)={metrics['pct_fine_only_likely_noise']:.1f}%  "
        f"Semantic-only/broad (amber)={metrics['pct_semantic_only_broad']:.1f}%",
        fontsize=8,
    )
    axes[3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = f"{OUT_DIR}/{pair['name']}_combined.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

    metrics["combined_figure"] = out_path
    return metrics


if __name__ == "__main__":
    with open(f"{OUT_DIR}/metrics.json") as f:
        existing = {m["pair"]: m for m in json.load(f)}

    for pair in PAIRS:
        print(f"Combined analysis: {pair['name']}...")
        m = analyze_pair_combined(pair)
        existing[pair["name"]].update(m)
        print(json.dumps(m, indent=2))

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(list(existing.values()), f, indent=2)
    print(f"Updated {OUT_DIR}/metrics.json with combined metrics.")
