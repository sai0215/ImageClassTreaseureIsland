"""
Two-page, classic-style PDF summary report for the Treasure Island change-detection analysis.
Page 1: header, method, results table, similarity bar chart.
Page 2: before/after/change thumbnail grid, conclusion, footer.
Reuses the same registration/SSIM/DINOv2 numbers already computed in metrics.json.
"""
import json
import textwrap
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu

from compare_images import PAIRS, IMG_DIR, OUT_DIR, load_rgb, valid_mask, align_b_to_a

plt.rcParams["font.family"] = "serif"

with open(f"{OUT_DIR}/metrics.json") as f:
    M = {m["pair"]: m for m in json.load(f)}

READS = {
    "Maxar_pre_vs_post": "Severe shoreline change",
    "NOAA_postHelene_vs_postMilton": "Modest incremental change",
    "S2_pre_vs_post": "Cloud-confounded, use cautiously",
}
SHORT_NAME = {
    "Maxar_pre_vs_post": "Maxar",
    "NOAA_postHelene_vs_postMilton": "NOAA",
    "S2_pre_vs_post": "Sentinel-2",
}

LEFT, RIGHT = 0.08, 0.92
WIDTH = RIGHT - LEFT


def square_crop_resize(img, size=320):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def build_pair_thumbs(pair, size=320):
    a = load_rgb(f"{IMG_DIR}/{pair['a']}")
    b_raw = load_rgb(f"{IMG_DIR}/{pair['b']}")
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_raw, cv2.COLOR_RGB2GRAY)
    b_aligned, _, _ = align_b_to_a(a_gray, b_gray, b_raw)
    mask = valid_mask(a) & valid_mask(b_aligned)

    a_g2 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_g2 = cv2.cvtColor(b_aligned, cv2.COLOR_RGB2GRAY)
    _, ssim_map = ssim(a_g2, b_g2, full=True)
    dissim = np.where(mask, 1 - ssim_map, 0)
    vals = dissim[mask]
    change_mask = np.zeros_like(mask)
    if vals.size and vals.max() > vals.min():
        t = threshold_otsu(vals)
        change_mask = (dissim > t) & mask

    overlay = a.astype(np.float32).copy()
    red = np.array([196, 46, 46], dtype=np.float32)
    overlay[change_mask] = 0.45 * overlay[change_mask] + 0.55 * red
    overlay = overlay.astype(np.uint8)

    return (
        square_crop_resize(a, size),
        square_crop_resize(b_aligned, size),
        square_crop_resize(overlay, size),
    )


class PageBuilder:
    """Top-down cursor layout helper, one instance per page/figure."""

    def __init__(self):
        self.fig = plt.figure(figsize=(8.5, 11))
        self.cursor = 0.97

    def take_axes(self, height, gap_after=0.014):
        top = self.cursor
        bottom = top - height
        ax = self.fig.add_axes([LEFT, bottom, WIDTH, height])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.cursor = bottom - gap_after
        return ax, top, bottom

    def rule(self, y=None, lw=1.1):
        y = self.cursor if y is None else y
        self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [y, y], color="black", linewidth=lw))

    def gap(self, amount):
        self.cursor -= amount


# ======================================================================
# PAGE 1 — Header, Method, Results Table, Similarity Bar Chart
# ======================================================================
p1 = PageBuilder()

ax_h, top, bottom = p1.take_axes(0.135, gap_after=0.0)
ax_h.text(0, 0.94, "Treasure Island, FL — Hurricane Change Detection",
           fontsize=18.5, fontweight="bold", va="top", ha="left")
ax_h.text(0, 0.56, "A pre/post-hurricane comparison of Maxar, NOAA, and Sentinel-2 satellite imagery",
           fontsize=12.5, style="italic", color="#333333", va="top", ha="left")
ax_h.text(0, 0.14,
           "Treasure Island, FL   |   2023-11-04 to 2024-10-14   |   Hurricanes Helene & Milton (2024)",
           fontsize=9, color="#555555", va="top", ha="left", family="monospace")
p1.rule(bottom)
p1.gap(0.028)

ax_m, top, bottom = p1.take_axes(0.135, gap_after=0.04)
method_text = (
    "Each before/after pair is co-registered with ORB feature matching and a RANSAC homography, "
    "since the two dates rarely share an exact footprint. Two independent similarity signals are "
    "then computed over the registered pair: Structural Similarity (SSIM), a classical pixel/texture "
    "measure robust to lighting and tide differences, and DINOv2, a self-supervised vision transformer "
    "whose embeddings capture semantic content rather than raw pixel structure. An Otsu threshold on "
    "the SSIM map produces the change-mask overlays shown on page 2."
)
ax_m.text(0, 1.0, "Method", fontsize=12, fontweight="bold", va="top", ha="left")
ax_m.text(0, 0.80, textwrap.fill(method_text, width=100), fontsize=10, va="top", ha="left", linespacing=1.6)

ax_t, top, bottom = p1.take_axes(0.16, gap_after=0.05)
ax_t.text(0, 1.0, "Results", fontsize=12, fontweight="bold", va="top", ha="left")

cols = ["Pair", "Dates", "SSIM", "% Changed", "DINOv2 sim", "Read"]
col_x = [0.0, 0.16, 0.42, 0.51, 0.63, 0.76]
head_y = 0.80
for cx, label in zip(col_x, cols):
    ax_t.text(cx, head_y, label, fontsize=9.2, fontweight="bold", va="top", ha="left")
ax_t.axhline(y=head_y - 0.10, xmin=0, xmax=1, color="black", linewidth=0.9)

row_y0, row_h = head_y - 0.24, 0.235
for i, pair in enumerate(PAIRS):
    m = M[pair["name"]]
    d_a = pair["a"].split("_")[2][:8]
    d_b = pair["b"].split("_")[2][:8]
    y = row_y0 - i * row_h
    ax_t.text(col_x[0], y, SHORT_NAME[pair["name"]], fontsize=9.8, va="top", ha="left", fontweight="bold")
    ax_t.text(col_x[1], y, f"{d_a} to {d_b}", fontsize=7.8, va="top", ha="left", family="monospace", color="#444")
    ax_t.text(col_x[2], y, f"{m['ssim_valid_region']:.3f}", fontsize=9.6, va="top", ha="left", family="monospace")
    ax_t.text(col_x[3], y, f"{m['pct_area_changed_otsu']:.1f}%", fontsize=9.6, va="top", ha="left", family="monospace")
    ax_t.text(col_x[4], y, f"{m['dinov2_global_cosine_sim']:.3f}", fontsize=9.6, va="top", ha="left", family="monospace")
    ax_t.text(col_x[5], y, READS[pair["name"]], fontsize=8.6, va="top", ha="left", color="#333")
    if i < len(PAIRS) - 1:
        ax_t.axhline(y=y - row_h + 0.06, xmin=0, xmax=1, color="#dddddd", linewidth=0.6)
ax_t.axhline(y=row_y0 - len(PAIRS) * row_h + 0.115, xmin=0, xmax=1, color="black", linewidth=0.9)

ax_bt, top, bottom = p1.take_axes(0.035, gap_after=0.0)
ax_bt.text(0, 1.0, "Similarity by Method", fontsize=12, fontweight="bold", va="top", ha="left")
ax_bt.text(0, 0.30, "1.0 = identical; higher = more similar", fontsize=8.6, color="#555555",
           va="top", ha="left", style="italic")

bar_h = 0.30
ax_b = p1.fig.add_axes([LEFT + 0.03, p1.cursor - bar_h, WIDTH - 0.06, bar_h])
p1.gap(bar_h + 0.02)

labels = [SHORT_NAME[p["name"]] for p in PAIRS]
ssim_vals = [M[p["name"]]["ssim_valid_region"] for p in PAIRS]
dino_vals = [M[p["name"]]["dinov2_global_cosine_sim"] for p in PAIRS]

x = np.arange(len(labels))
w = 0.32
b1 = ax_b.bar(x - w / 2, ssim_vals, width=w, color="#2b2b2b", label="SSIM (valid region)")
b2 = ax_b.bar(x + w / 2, dino_vals, width=w, color="#b7b7b7", edgecolor="#2b2b2b",
              linewidth=0.7, label="DINOv2 global cos-sim")

for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax_b.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.3f}",
                  ha="center", va="bottom", fontsize=9, family="monospace")

ax_b.set_xticks(x); ax_b.set_xticklabels(labels, fontsize=11)
ax_b.set_ylim(0, 1.32)
ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_yticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=9, family="monospace")
ax_b.tick_params(length=0)
for spine in ["top", "right", "left"]:
    ax_b.spines[spine].set_visible(False)
ax_b.spines["bottom"].set_color("#333333")
ax_b.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
ax_b.set_axisbelow(True)
ax_b.legend(loc="upper center", ncol=2, frameon=False, fontsize=9.5)

p1.fig.text(0.5, 0.015, "Page 1 of 2", fontsize=8, color="#888888", ha="center")

# ======================================================================
# PAGE 2 — Visual Evidence, Conclusion, Footer
# ======================================================================
p2 = PageBuilder()

ax_h2, top, bottom = p2.take_axes(0.05, gap_after=0.015)
ax_h2.text(0, 0.9, "Visual Evidence — Before / After / Change", fontsize=15, fontweight="bold",
           va="top", ha="left")
p2.rule(bottom + 0.005)

grid_h = 0.54
ax_sl, top, bottom = p2.take_axes(0.024, gap_after=0.0)
stage_x = [0.13, 0.36, 0.59]
stage_w = 0.21
for sx, stage in zip(stage_x, ["BEFORE", "AFTER", "CHANGE  (red = flagged)"]):
    ax_sl.text(sx + stage_w / 2, 0.15, stage, fontsize=8.6, ha="center", va="bottom",
               color="#444444", family="monospace", fontweight="bold")

row_gap = 0.015
row_h_grid = (grid_h - 2 * row_gap) / 3
grid_top_y = p2.cursor
p2.gap(grid_h + 0.02)

for r, pair in enumerate(PAIRS):
    thumbs = build_pair_thumbs(pair)
    row_top = grid_top_y - r * (row_h_grid + row_gap)
    row_bot = row_top - row_h_grid

    ax_label = p2.fig.add_axes([LEFT, row_bot, 0.11, row_h_grid]); ax_label.axis("off")
    ax_label.text(0.0, 0.5, SHORT_NAME[pair["name"]], fontsize=12, fontweight="bold",
                  va="center", ha="left", transform=ax_label.transAxes)

    for c, thumb in enumerate(thumbs):
        ax_img = p2.fig.add_axes([LEFT + stage_x[c], row_bot, stage_w, row_h_grid])
        ax_img.imshow(thumb)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#999999")
            spine.set_linewidth(0.7)

p2.rule()
p2.gap(0.015)

ax_c, top, bottom = p2.take_axes(0.17, gap_after=0.025)
conclusion = (
    "All three sources agree on the same basic story: the open Gulf-facing beach and dune line took "
    "substantial, visually confirmed damage between the pre-hurricane baseline and the post-Milton "
    "state, while interior canal neighborhoods stayed comparatively stable. NOAA's storm-on-storm pair "
    "shows Milton's incremental damage on top of Helene was real but smaller than the full pre/post "
    "change, and DINOv2 independently confirms it as the least-changed pair. The two methods disagree "
    "only on Sentinel-2: SSIM calls it most similar (cloud texture still reads as texture), DINOv2 "
    "calls it least similar (clouds break the semantic match) — a reminder that these percentages are "
    "relative indicators, not calibrated damage measurements."
)
ax_c.text(0, 1.0, "Conclusion", fontsize=13, fontweight="bold", va="top", ha="left")
ax_c.text(0, 0.83, textwrap.fill(conclusion, width=102), fontsize=9.2, va="top", ha="left", linespacing=1.4)

p2.rule()
p2.gap(0.012)
ax_f, top, bottom = p2.take_axes(0.04, gap_after=0.0)
footer_text = (
    "Method: ORB+RANSAC registration -> masked SSIM / Otsu change mask (OpenCV, scikit-image) "
    "+ DINOv2 (facebook/dinov2-base) cosine similarity."
)
ax_f.text(0, 1.0, textwrap.fill(footer_text, width=105), fontsize=7.4, color="#666666",
          va="top", ha="left", family="monospace", linespacing=1.45)

p2.fig.text(0.5, 0.015, "Page 2 of 2", fontsize=8, color="#888888", ha="center")

out_path = f"{OUT_DIR}/Final_Report.pdf"
with PdfPages(out_path) as pdf:
    pdf.savefig(p1.fig)
    pdf.savefig(p2.fig)

p1.fig.savefig(f"{OUT_DIR}/Final_Report_page1_preview.png", dpi=170)
p2.fig.savefig(f"{OUT_DIR}/Final_Report_page2_preview.png", dpi=170)
plt.close(p1.fig)
plt.close(p2.fig)
print("Saved", out_path, "(2 pages) and preview PNGs")
