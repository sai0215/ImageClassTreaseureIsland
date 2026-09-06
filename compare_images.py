import json
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

IMG_DIR = "/Users/saivignesh/Desktop/projects/TAMU/ImageDetection/images"
OUT_DIR = "/Users/saivignesh/Desktop/projects/TAMU/ImageDetection/analysis_output"

PAIRS = [
    {
        "name": "Maxar_pre_vs_post",
        "title": "Maxar: 2023-11-04 (pre) vs 2024-10-10 (post-Milton)",
        "a": "Maxar_pre_20231104_TreasureIsland_preview.png",
        "b": "Maxar_post_20241010_TreasureIsland_preview.png",
    },
    {
        "name": "NOAA_postHelene_vs_postMilton",
        "title": "NOAA: post-Helene 2024-09-30 vs post-Milton 2024-10-11",
        "a": "NOAA_postHelene_20240930_TreasureIsland_preview.png",
        "b": "NOAA_postMilton_20241011_TreasureIsland_preview.png",
    },
    {
        "name": "S2_pre_vs_post",
        "title": "Sentinel-2: 2024-09-19 (pre) vs 2024-10-14 (post)",
        "a": "S2_pre_20240919_preview.png",
        "b": "S2_post_20241014_preview.png",
    },
]


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def valid_mask(img, thresh=8):
    return (img.astype(np.int32).sum(axis=2) > thresh)


def align_b_to_a(a_gray, b_gray, b_color):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(a_gray, None)
    kp2, des2 = orb.detectAndCompute(b_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return b_color, False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2) #lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 15:
        return b_color, False, len(good)

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return b_color, False, len(good)

    h, w = a_gray.shape[:2]
    aligned = cv2.warpPerspective(b_color, H, (w, h), borderValue=(0, 0, 0))
    inliers = int(mask.sum()) if mask is not None else 0
    return aligned, True, inliers


def analyze_pair(pair):
    a = load_rgb(f"{IMG_DIR}/{pair['a']}")
    b_raw = load_rgb(f"{IMG_DIR}/{pair['b']}")

    
    if b_raw.shape[:2] != a.shape[:2]:
        b_raw = np.array(Image.fromarray(b_raw).resize((a.shape[1], a.shape[0])))

    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_raw, cv2.COLOR_RGB2GRAY)

    b_aligned, aligned_ok, n_inliers = align_b_to_a(a_gray, b_gray, b_raw)
    b_gray_aligned = cv2.cvtColor(b_aligned, cv2.COLOR_RGB2GRAY)

    mask = valid_mask(a) & valid_mask(b_aligned)

    # SSIM (structural similarity) map
    score, ssim_map = ssim(a_gray, b_gray_aligned, full=True)
    dissim = 1 - ssim_map
    dissim_masked = np.where(mask, dissim, 0)

  
    ssim_score_valid = ssim_map[mask].mean() if mask.any() else float("nan")

    # absolute RGB diff
    diff_rgb = cv2.absdiff(a, b_aligned).astype(np.float32)
    diff_gray = diff_rgb.mean(axis=2)
    diff_gray_masked = np.where(mask, diff_gray, 0)

   #otsu thresholding to find changed areas
    valid_vals = dissim_masked[mask]
    change_pct = float("nan")
    change_mask = np.zeros_like(mask)
    if valid_vals.size > 0 and valid_vals.max() > valid_vals.min():
        t = threshold_otsu(valid_vals)
        change_mask = (dissim_masked > t) & mask
        change_pct = 100.0 * change_mask.sum() / mask.sum()

    
    mean_a = a[mask].mean(axis=0) if mask.any() else np.array([np.nan] * 3)
    mean_b = b_aligned[mask].mean(axis=0) if mask.any() else np.array([np.nan] * 3)

    metrics = {
        "pair": pair["name"],
        "aligned": aligned_ok,
        "orb_inliers": n_inliers,
        "ssim_global": float(score),
        "ssim_valid_region": float(ssim_score_valid),
        "mean_abs_pixel_diff_0_255": float(diff_gray_masked[mask].mean()) if mask.any() else float("nan"),
        "pct_area_changed_otsu": change_pct,
        "valid_pixel_fraction": float(mask.mean()),
        "mean_rgb_a": mean_a.tolist(),
        "mean_rgb_b": mean_b.tolist(),
        "mean_rgb_delta_b_minus_a": (mean_b - mean_a).tolist(),
    }

   
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(pair["title"], fontsize=15, fontweight="bold")

    axes[0, 0].imshow(a); axes[0, 0].set_title(f"A: {pair['a']}", fontsize=9)
    axes[0, 1].imshow(b_raw); axes[0, 1].set_title(f"B (raw): {pair['b']}", fontsize=9)
    axes[0, 2].imshow(b_aligned); axes[0, 2].set_title(
        f"B aligned to A (ORB inliers={n_inliers})", fontsize=9)

    im1 = axes[1, 0].imshow(dissim_masked, cmap="inferno", vmin=0, vmax=1)
    axes[1, 0].set_title(f"SSIM dissimilarity map\n(global SSIM={score:.3f}, valid={ssim_score_valid:.3f})", fontsize=9)
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(diff_gray_masked, cmap="magma", vmin=0, vmax=80)
    axes[1, 1].set_title(f"Absolute RGB diff (mean={metrics['mean_abs_pixel_diff_0_255']:.1f}/255)", fontsize=9)
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    overlay = a.copy()
    overlay[change_mask] = [255, 0, 0]
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f"Otsu change mask overlay\n({change_pct:.1f}% of valid area flagged)", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = f"{OUT_DIR}/{pair['name']}.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

    metrics["figure"] = out_path
    return metrics


if __name__ == "__main__":
    all_metrics = []
    for pair in PAIRS:
        print(f"Analyzing {pair['name']}...")
        m = analyze_pair(pair)
        all_metrics.append(m)
        print(json.dumps(m, indent=2))
        print()

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {OUT_DIR}/metrics.json")
