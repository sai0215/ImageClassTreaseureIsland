import json
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
import cv2

from compare_images import PAIRS, IMG_DIR, OUT_DIR, load_rgb, valid_mask, align_b_to_a

MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
PATCH = model.config.patch_size  # 14


@torch.no_grad()
def embed(img_rgb_uint8):
    inputs = processor(images=img_rgb_uint8, return_tensors="pt").to(DEVICE)
    h_in, w_in = inputs["pixel_values"].shape[-2:]
    out = model(**inputs)
    tokens = out.last_hidden_state[0]
    cls = tokens[0]
    n_patches = (h_in // PATCH) * (w_in // PATCH)
    patch_tokens = tokens[-n_patches:]
    grid_h, grid_w = h_in // PATCH, w_in // PATCH
    patch_grid = patch_tokens.reshape(grid_h, grid_w, -1)
    return cls.cpu().numpy(), patch_grid.cpu().numpy()


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def patch_cosine_map(grid_a, grid_b):
    a = grid_a / (np.linalg.norm(grid_a, axis=-1, keepdims=True) + 1e-8)
    b = grid_b / (np.linalg.norm(grid_b, axis=-1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=-1)


def analyze_pair_dinov2(pair):
    a = load_rgb(f"{IMG_DIR}/{pair['a']}")
    b_raw = load_rgb(f"{IMG_DIR}/{pair['b']}")
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_raw, cv2.COLOR_RGB2GRAY)
    b_aligned, _, _ = align_b_to_a(a_gray, b_gray, b_raw)
    mask = valid_mask(a) & valid_mask(b_aligned)

    cls_a, grid_a = embed(a)
    cls_b, grid_b = embed(b_aligned)

    global_sim = cosine(cls_a, cls_b)
    sim_map = patch_cosine_map(grid_a, grid_b)
    dissim_map = 1 - sim_map

    h, w = a.shape[:2]
    dissim_full = cv2.resize(dissim_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    dissim_full = np.clip(dissim_full, 0, None)
    dissim_masked = np.where(mask, dissim_full, 0)

    mean_patch_dissim = float(sim_map.size and (1 - sim_map).mean())

    metrics = {
        "dinov2_model": MODEL_NAME,
        "dinov2_global_cosine_sim": global_sim,
        "dinov2_mean_patch_dissim": mean_patch_dissim,
        "dinov2_patch_grid": list(sim_map.shape),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    fig.suptitle(f"{pair['title']}  —  DINOv2 embedding similarity", fontsize=14, fontweight="bold")
    axes[0].imshow(a); axes[0].set_title("A (reference)", fontsize=9); axes[0].axis("off")
    axes[1].imshow(b_aligned); axes[1].set_title("B (aligned)", fontsize=9); axes[1].axis("off")
    im = axes[2].imshow(dissim_masked, cmap="viridis")
    axes[2].set_title(
        f"DINOv2 patch dissimilarity (1-cos)\nglobal CLS cos-sim={global_sim:.3f}, "
        f"mean patch dissim={mean_patch_dissim:.3f}",
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
