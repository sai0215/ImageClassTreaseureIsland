import json

OUT = "/Users/saivignesh/Desktop/projects/TAMU/ImageDetection/analysis_output"
SCRATCH = "/private/tmp/claude-501/-Users-saivignesh-Desktop-projects-TAMU/29425213-5192-4f48-bc15-422f4c47f5a8/scratchpad"

with open(f"{OUT}/metrics.json") as f:
    metrics = {m["pair"]: m for m in json.load(f)}

imgs = {}
for key, fname in [
    ("maxar", "Maxar_pre_vs_post_web.jpg"),
    ("noaa", "NOAA_postHelene_vs_postMilton_web.jpg"),
    ("s2", "S2_pre_vs_post_web.jpg"),
    ("maxar_dino", "Maxar_pre_vs_post_dinov2_web.jpg"),
    ("noaa_dino", "NOAA_postHelene_vs_postMilton_dinov2_web.jpg"),
    ("s2_dino", "S2_pre_vs_post_dinov2_web.jpg"),
]:
    with open(f"{OUT}/{fname}.b64") as f:
        imgs[key] = f.read()

m_maxar = metrics["Maxar_pre_vs_post"]
m_noaa = metrics["NOAA_postHelene_vs_postMilton"]
m_s2 = metrics["S2_pre_vs_post"]

def rgbfmt(v):
    return f"{v:+.1f}"

html = f"""<title>Treasure Island Change Detection</title>
<style>
:root {{
  --bg: #EEF2F3;
  --surface: #FFFFFF;
  --surface-2: #E3E9EA;
  --ink: #16232B;
  --ink-soft: #4B5D66;
  --accent: #C96A2E;
  --accent-2: #1F6F78;
  --line: #CBD6D9;
  --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --display: 'Archivo', 'Arial Narrow', Arial, sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0E161C;
    --surface: #16212A;
    --surface-2: #1C2932;
    --ink: #E8EEF0;
    --ink-soft: #9FB2BA;
    --accent: #EB9257;
    --accent-2: #4CB2BB;
    --line: #26343E;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0E161C;
  --surface: #16212A;
  --surface-2: #1C2932;
  --ink: #E8EEF0;
  --ink-soft: #9FB2BA;
  --accent: #EB9257;
  --accent-2: #4CB2BB;
  --line: #26343E;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: var(--serif);
  margin: 0;
  padding: 0 20px 80px;
  line-height: 1.6;
}}
.wrap {{
  max-width: 880px;
  margin: 0 auto;
}}

/* header */
header.masthead {{
  padding: 64px 0 28px;
  border-bottom: 1px solid var(--line);
}}
.eyebrow {{
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent-2);
  margin: 0 0 14px;
}}
h1 {{
  font-family: var(--display);
  font-weight: 700;
  font-size: clamp(2rem, 4.4vw, 3.1rem);
  letter-spacing: -0.01em;
  line-height: 1.05;
  margin: 0 0 16px;
  text-wrap: balance;
  color: var(--ink);
}}
.dek {{
  font-size: 1.1rem;
  color: var(--ink-soft);
  max-width: 62ch;
  margin: 0 0 26px;
}}
.meta-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px 22px;
  font-family: var(--mono);
  font-size: 12.5px;
  color: var(--ink-soft);
}}
.meta-strip span b {{
  color: var(--ink);
  font-weight: 600;
}}

/* sections */
section {{ padding: 46px 0; border-bottom: 1px solid var(--line); }}
section:last-of-type {{ border-bottom: none; }}
h2 {{
  font-family: var(--display);
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: -0.005em;
  margin: 0 0 18px;
  color: var(--ink);
}}
p {{ margin: 0 0 14px; max-width: 68ch; }}
.lede p {{ font-size: 1.02rem; }}

/* method list */
ol.methods {{
  list-style: none;
  margin: 24px 0 0;
  padding: 0;
  display: grid;
  gap: 16px;
}}
ol.methods li {{
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 14px;
  align-items: baseline;
}}
ol.methods .num {{
  font-family: var(--mono);
  color: var(--accent);
  font-size: 13px;
  font-weight: 600;
  padding-top: 2px;
}}
ol.methods b {{
  font-family: var(--display);
  font-weight: 600;
  color: var(--ink);
}}
ol.methods span.body {{ color: var(--ink-soft); font-size: 0.96rem; }}

/* pair blocks */
.pair {{ padding: 54px 0; border-bottom: 1px solid var(--line); }}
.pair-head {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 10px 20px;
  margin-bottom: 6px;
}}
.pair-eyebrow {{
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
}}
.pair-dates {{
  font-family: var(--mono);
  font-size: 12px;
  color: var(--ink-soft);
}}
.pair h3 {{
  font-family: var(--display);
  font-weight: 700;
  font-size: 1.35rem;
  margin: 4px 0 20px;
  color: var(--ink);
}}
figure {{ margin: 0 0 22px; }}
figure img {{
  width: 100%;
  height: auto;
  display: block;
  border-radius: 3px;
  border: 1px solid var(--line);
}}
figcaption {{
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--ink-soft);
  margin-top: 8px;
  line-height: 1.5;
}}

/* metrics table */
.metrics-wrap {{ overflow-x: auto; margin: 0 0 22px; }}
table.metrics {{
  border-collapse: collapse;
  width: 100%;
  min-width: 480px;
  font-family: var(--mono);
  font-size: 13px;
  background: var(--surface);
  border: 1px solid var(--line);
}}
table.metrics th, table.metrics td {{
  text-align: left;
  padding: 9px 14px;
  border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums;
}}
table.metrics th {{
  font-weight: 600;
  color: var(--ink-soft);
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.06em;
  background: var(--surface-2);
}}
table.metrics tr:last-child td {{ border-bottom: none; }}
table.metrics td.val {{ color: var(--accent); font-weight: 600; }}

.interp {{ background: var(--surface); border: 1px solid var(--line); border-radius: 3px; padding: 20px 22px; }}
.interp p:last-child {{ margin-bottom: 0; }}
.interp .label {{
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--accent-2); margin: 0 0 10px;
}}

/* synthesis */
.callout {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: 3px;
  padding: 18px 22px;
  margin: 18px 0;
}}
.callout p {{ margin: 0; color: var(--ink-soft); font-size: 0.95rem; }}
.callout .label {{
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 8px; display:block;
}}

footer {{
  padding: 40px 0 0;
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--ink-soft);
}}
</style>

<div class="wrap">

  <header class="masthead">
    <p class="eyebrow">Change-Detection Field Report</p>
    <h1>Reading Two Hurricanes Off a Barrier Island</h1>
    <p class="dek">Three independent satellite archives capture the same stretch of Treasure Island, FL, before and after Hurricanes Helene and Milton (Sept–Oct 2024). Registering and differencing each pair, then reading it a second time through a learned vision embedding, turns three ordinary-looking image sets into a quantified map of where the coastline actually changed.</p>
    <div class="meta-strip">
      <span><b>Site</b> Treasure Island, FL</span>
      <span><b>Event window</b> 2023-11-04 → 2024-10-14</span>
      <span><b>Sources</b> Maxar · NOAA NGS · Sentinel-2</span>
      <span><b>Pairs analyzed</b> 3</span>
    </div>
  </header>

  <section class="lede">
    <h2>Method</h2>
    <p>Each before/after pair is treated as a small registration-and-difference pipeline rather than a raw pixel subtraction, since the two shots in a pair are rarely captured from the exact same footprint. Two independent similarity signals are computed on top of the same registered pair: a classical pixel/texture signal (SSIM) and a learned semantic signal (DINOv2 embeddings), because they disagree in informative ways.</p>
    <ol class="methods">
      <li><span class="num">01</span><div><b>Feature-based co-registration.</b> <span class="body">ORB keypoints are matched between the two dates and a RANSAC homography warps the later image onto the earlier one's frame, correcting the small footprint/rotation offsets between passes.</span></div></li>
      <li><span class="num">02</span><div><b>No-data masking.</b> <span class="body">Near-black border pixels (swath edges, letterboxing left behind by alignment) are excluded from every statistic so empty canvas can't masquerade as "change."</span></div></li>
      <li><span class="num">03</span><div><b>Structural similarity (SSIM).</b> <span class="body">A sliding-window SSIM map scores local structural agreement; 1 − SSIM highlights where texture and edges genuinely reorganized, which is far more robust to lighting/tide differences than a raw color diff.</span></div></li>
      <li><span class="num">04</span><div><b>Absolute RGB difference.</b> <span class="body">A straightforward per-pixel color delta, useful as a sanity check against the SSIM map and for reading turbidity/sediment color shifts directly.</span></div></li>
      <li><span class="num">05</span><div><b>Otsu-thresholded change mask.</b> <span class="body">An automatic threshold on the dissimilarity map converts the continuous SSIM signal into a binary "changed area," reported as a percentage of the valid (non-masked) frame.</span></div></li>
      <li><span class="num">06</span><div><b>DINOv2 embedding similarity.</b> <span class="body">A self-supervised vision transformer (facebook/dinov2-base) embeds each image; cosine similarity on the global [CLS] token gives one semantic "how similar overall" score per pair, and cosine similarity on the 16×16 grid of patch tokens (upsampled back to image size) gives a coarse semantic change map — sensitive to changes in what a region <em>is</em> (sand → water, roof → debris), not just how its pixels look.</span></div></li>
    </ol>
  </section>

  <!-- PAIR 1: MAXAR -->
  <section class="pair">
    <div class="pair-head">
      <span class="pair-eyebrow">Pair 01 — Maxar (sub-meter)</span>
      <span class="pair-dates">2023-11-04 → 2024-10-10</span>
    </div>
    <h3>Pre-season baseline vs. post-Milton</h3>
    <figure>
      <img src="data:image/jpeg;base64,{imgs['maxar']}" alt="Maxar pre vs post comparison grid" />
      <figcaption>Top row: source A, raw source B, and B co-registered onto A ({m_maxar['orb_inliers']} ORB inliers). Bottom row: SSIM dissimilarity map, absolute RGB difference, and the Otsu change mask overlaid in red.</figcaption>
    </figure>
    <div class="metrics-wrap">
      <table class="metrics">
        <thead><tr><th>Metric</th><th>Value</th><th>Reading</th></tr></thead>
        <tbody>
          <tr><td>SSIM (valid region)</td><td class="val">{m_maxar['ssim_valid_region']:.3f}</td><td>1.0 = identical structure</td></tr>
          <tr><td>Mean abs. pixel diff</td><td class="val">{m_maxar['mean_abs_pixel_diff_0_255']:.1f} / 255</td><td>raw color-shift magnitude</td></tr>
          <tr><td>Area flagged changed</td><td class="val">{m_maxar['pct_area_changed_otsu']:.1f}%</td><td>Otsu threshold on SSIM map</td></tr>
          <tr><td>Mean RGB shift (B − A)</td><td class="val">R {rgbfmt(m_maxar['mean_rgb_delta_b_minus_a'][0])} · G {rgbfmt(m_maxar['mean_rgb_delta_b_minus_a'][1])} · B {rgbfmt(m_maxar['mean_rgb_delta_b_minus_a'][2])}</td><td>redder, less green</td></tr>
          <tr><td>Valid frame coverage</td><td class="val">{m_maxar['valid_pixel_fraction']*100:.0f}%</td><td>excludes no-data border</td></tr>
          <tr><td>DINOv2 global cos-sim</td><td class="val">{m_maxar['dinov2_global_cosine_sim']:.3f}</td><td>1.0 = same semantic content</td></tr>
          <tr><td>DINOv2 mean patch dissim</td><td class="val">{m_maxar['dinov2_mean_patch_dissim']:.3f}</td><td>16×16 grid, 1−cos per patch</td></tr>
        </tbody>
      </table>
    </div>
    <div class="interp">
      <p class="label">Reading</p>
      <p>This is the sharpest and most trustworthy pair of the three. The SSIM map lights up almost the entire dune line and beachfront: the smooth white sand berms visible in 2023 are gone or badly reshaped in 2024, replaced by darker, wetter, more irregular sand and visible storm debris. Water color swings toward red and away from green (R {rgbfmt(m_maxar['mean_rgb_delta_b_minus_a'][0])}, G {rgbfmt(m_maxar['mean_rgb_delta_b_minus_a'][1])}), consistent with sediment-laden, turbid nearshore water rather than the clear turquoise Gulf water in the baseline shot. The interior canal neighborhood shows comparatively low SSIM dissimilarity, indicating the marina/residential blocks inland of the beach were structurally far less disturbed than the immediate shoreline. The 54% "changed" figure is inflated somewhat by genuine sub-meter texture differences (individual roof glare, boat positions) that SSIM is sensitive to at this resolution — treat it as an upper bound on real storm damage, not a literal damage footprint.</p>
    </div>
    <figure>
      <img src="data:image/jpeg;base64,{imgs['maxar_dino']}" alt="Maxar DINOv2 patch dissimilarity map" />
      <figcaption>DINOv2 patch-token dissimilarity map (coarse 16×16 grid, upsampled). Bright regions are patches whose learned visual content shifted most between dates — note it agrees with SSIM on the shoreline hotspot but is comparatively quiet over the interior canals, since DINOv2 sees "houses/boats/roads" as the same semantic content even when individual pixels moved.</figcaption>
    </figure>
  </section>

  <!-- PAIR 2: NOAA -->
  <section class="pair">
    <div class="pair-head">
      <span class="pair-eyebrow">Pair 02 — NOAA NGS emergency response</span>
      <span class="pair-dates">2024-09-30 (post-Helene) → 2024-10-11 (post-Milton)</span>
    </div>
    <h3>Storm-on-storm: what Milton added on top of Helene</h3>
    <figure>
      <img src="data:image/jpeg;base64,{imgs['noaa']}" alt="NOAA post-Helene vs post-Milton comparison grid" />
      <figcaption>Both frames are already post-storm. The comparison isolates incremental change from Milton striking a coastline Helene had already reworked. {m_noaa['orb_inliers']} ORB inliers used for registration.</figcaption>
    </figure>
    <div class="metrics-wrap">
      <table class="metrics">
        <thead><tr><th>Metric</th><th>Value</th><th>Reading</th></tr></thead>
        <tbody>
          <tr><td>SSIM (valid region)</td><td class="val">{m_noaa['ssim_valid_region']:.3f}</td><td>higher than Pair 01 — smaller net change</td></tr>
          <tr><td>Mean abs. pixel diff</td><td class="val">{m_noaa['mean_abs_pixel_diff_0_255']:.1f} / 255</td><td>raw color-shift magnitude</td></tr>
          <tr><td>Area flagged changed</td><td class="val">{m_noaa['pct_area_changed_otsu']:.1f}%</td><td>Otsu threshold on SSIM map</td></tr>
          <tr><td>Mean RGB shift (B − A)</td><td class="val">R {rgbfmt(m_noaa['mean_rgb_delta_b_minus_a'][0])} · G {rgbfmt(m_noaa['mean_rgb_delta_b_minus_a'][1])} · B {rgbfmt(m_noaa['mean_rgb_delta_b_minus_a'][2])}</td><td>uniformly darker</td></tr>
          <tr><td>Valid frame coverage</td><td class="val">{m_noaa['valid_pixel_fraction']*100:.0f}%</td><td>excludes no-data border</td></tr>
          <tr><td>DINOv2 global cos-sim</td><td class="val">{m_noaa['dinov2_global_cosine_sim']:.3f}</td><td>highest of the three pairs</td></tr>
          <tr><td>DINOv2 mean patch dissim</td><td class="val">{m_noaa['dinov2_mean_patch_dissim']:.3f}</td><td>16×16 grid, 1−cos per patch</td></tr>
        </tbody>
      </table>
    </div>
    <div class="interp">
      <p class="label">Reading</p>
      <p>Because both frames are already storm-damaged, SSIM here is measuring incremental change, and it's noticeably higher (0.60) than the Pair 01 baseline comparison — most of the coastline's basic structure held between the two events. The dissimilarity hotspots that remain are concentrated tightly along the beach/dune edge and the immediate shoreline, which reads as continued dune scarping and sand redistribution from Milton on ground Helene had already stripped. One caveat: the RGB shift is almost perfectly uniform across all three channels (R {rgbfmt(m_noaa['mean_rgb_delta_b_minus_a'][0])}, G {rgbfmt(m_noaa['mean_rgb_delta_b_minus_a'][1])}, B {rgbfmt(m_noaa['mean_rgb_delta_b_minus_a'][2])}) — a flat brightness drop rather than a color-selective shift like Pair 01's. That pattern is more consistent with the two NOAA collects being captured under different sun angle, tide, or processing/exposure settings than with real ground darkening, so the color metric should be read cautiously here; the SSIM structural signal is the more reliable read for this pair. DINOv2 agrees this is the <em>most</em> similar pair of the three (global cos-sim {m_noaa['dinov2_global_cosine_sim']:.3f}, the highest of all three comparisons) — a second, independent signal confirming that Milton's incremental damage on top of Helene was real but comparatively modest.</p>
    </div>
    <figure>
      <img src="data:image/jpeg;base64,{imgs['noaa_dino']}" alt="NOAA DINOv2 patch dissimilarity map" />
      <figcaption>DINOv2 patch dissimilarity map. Localized bright spots track isolated shoreline/dune segments; the broad interior is dark (semantically unchanged), matching the "incremental, not wholesale, change" read from SSIM.</figcaption>
    </figure>
  </section>

  <!-- PAIR 3: SENTINEL-2 -->
  <section class="pair">
    <div class="pair-head">
      <span class="pair-eyebrow">Pair 03 — Sentinel-2 (10–20 m/px, regional)</span>
      <span class="pair-dates">2024-09-19 (pre) → 2024-10-14 (post)</span>
    </div>
    <h3>Wide-area context, complicated by cloud cover</h3>
    <figure>
      <img src="data:image/jpeg;base64,{imgs['s2']}" alt="Sentinel-2 pre vs post comparison grid" />
      <figcaption>Regional Tampa Bay / barrier-island view. The pre-image carries substantial cloud cover (visible as bright blotches); the post-image is comparatively clear. {m_s2['orb_inliers']} ORB inliers used for registration.</figcaption>
    </figure>
    <div class="metrics-wrap">
      <table class="metrics">
        <thead><tr><th>Metric</th><th>Value</th><th>Reading</th></tr></thead>
        <tbody>
          <tr><td>SSIM (valid region)</td><td class="val">{m_s2['ssim_valid_region']:.3f}</td><td>lowest confidence of the three</td></tr>
          <tr><td>Mean abs. pixel diff</td><td class="val">{m_s2['mean_abs_pixel_diff_0_255']:.1f} / 255</td><td>highest of the three — cloud-driven</td></tr>
          <tr><td>Area flagged changed</td><td class="val">{m_s2['pct_area_changed_otsu']:.1f}%</td><td>Otsu threshold on SSIM map</td></tr>
          <tr><td>Mean RGB shift (B − A)</td><td class="val">R {rgbfmt(m_s2['mean_rgb_delta_b_minus_a'][0])} · G {rgbfmt(m_s2['mean_rgb_delta_b_minus_a'][1])} · B {rgbfmt(m_s2['mean_rgb_delta_b_minus_a'][2])}</td><td>darker overall (cloud removal)</td></tr>
          <tr><td>Valid frame coverage</td><td class="val">{m_s2['valid_pixel_fraction']*100:.0f}%</td><td>excludes no-data border</td></tr>
          <tr><td>DINOv2 global cos-sim</td><td class="val">{m_s2['dinov2_global_cosine_sim']:.3f}</td><td>lowest of the three pairs</td></tr>
          <tr><td>DINOv2 mean patch dissim</td><td class="val">{m_s2['dinov2_mean_patch_dissim']:.3f}</td><td>≈ Pair 01, despite higher SSIM</td></tr>
        </tbody>
      </table>
    </div>
    <div class="interp">
      <p class="label">Reading</p>
      <p>This pair carries the least reliable signal of the three. Its highest-of-the-set pixel-diff value ({m_s2['mean_abs_pixel_diff_0_255']:.1f}/255) is misleading on its own — the change mask visibly clusters in large, blob-shaped patches that correspond almost exactly to where clouds sat in the pre-image, not to any ground feature. Sun-glint on inland waterways and general seasonal vegetation drift account for most of the rest. Real signal is still present at 10–20 m resolution: a faint but consistent turbidity/color shift is visible along the outer coastline and near the barrier-island inlets in the post-image, agreeing directionally with the sediment-plume finding from Pair 01. For change detection specifically, this pair is best used as coarse regional context rather than a damage estimate.</p>
      <p>This is also the clearest case where the two similarity methods disagree, and the disagreement is itself informative. SSIM rates this pair the <em>most</em> similar of the three (0.647, higher than either Maxar or NOAA), because cloud texture is still texture — a sliding pixel-structure window doesn't strongly penalize a cloud sitting where ground used to be. DINOv2 rates it the <em>least</em> similar (global cos-sim {m_s2['dinov2_global_cosine_sim']:.3f}, the lowest of all three pairs; mean patch dissimilarity {m_s2['dinov2_mean_patch_dissim']:.3f}, on par with Pair 01's genuine storm damage), because a semantic embedding treats "cloud" and "city block" as two very different kinds of content, regardless of how locally smooth or textured either one is. Neither reading is wrong — they're measuring different things — but it means SSIM's 0.647 score for this pair should not be read as "not much changed here."</p>
    </div>
    <figure>
      <img src="data:image/jpeg;base64,{imgs['s2_dino']}" alt="Sentinel-2 DINOv2 patch dissimilarity map" />
      <figcaption>DINOv2 patch dissimilarity map. Broad, diffuse high-dissimilarity regions dominate the frame — consistent with a semantic embedding reacting to cloud-vs-clear content rather than to fine ground change, in contrast to the sharply localized hotspots in Pairs 01–02.</figcaption>
    </figure>
  </section>

  <section>
    <h2>Synthesis</h2>
    <p>All three sources agree on the same basic story despite very different sensors, resolutions, and capture conditions: the immediate beachfront and dune line at Treasure Island underwent substantial, visually confirmed structural change between the pre-hurricane baseline and the post-Milton state, and nearshore water consistently reads more turbid/sediment-laden afterward. The interior canal neighborhoods were comparatively stable across all three comparisons — the damage signal concentrates almost entirely on the open Gulf-facing shoreline. DINOv2's global cosine-similarity ranking (NOAA {m_noaa['dinov2_global_cosine_sim']:.3f} > Maxar {m_maxar['dinov2_global_cosine_sim']:.3f} > S2 {m_s2['dinov2_global_cosine_sim']:.3f}) independently confirms that the NOAA storm-on-storm pair is the most similar of the three, consistent with its higher SSIM score.</p>
    <p>The NOAA storm-on-storm comparison (Pair 02) suggests Milton's incremental damage on top of Helene, while real, was smaller in relative terms than the full pre-to-post-hurricane change captured by Maxar — consistent with Helene having already done much of the initial damage to the dune system before Milton arrived eleven days later.</p>
    <p>Running both a pixel/texture metric (SSIM) and a learned semantic metric (DINOv2) side by side paid off specifically on the Sentinel-2 pair, where they disagree: SSIM calls it the most similar pair of the three, DINOv2 calls it the least. That split traces directly to cloud cover in the pre-image — texture-based SSIM doesn't penalize a cloud much, a semantic embedding penalizes it heavily — and would have been invisible from either method alone.</p>
    <div class="callout">
      <span class="label">Limitations</span>
      <p>SSIM and pixel-diff are sensitive to anything that changes the frame — clouds, sun angle, tide stage, sensor differences, and JPEG-preview compression artifacts — not just storm damage. DINOv2 removes some of that sensitivity but adds its own bias: it was pretrained on natural/web imagery, not satellite data, and its 16×16 patch grid is far coarser than the SSIM map, so it can name a region as "changed" without localizing the change as precisely. None of these pairs are radiometrically calibrated to each other, so the "% area changed" and cosine-similarity figures should be read as relative, cross-comparable indicators of disturbance rather than an absolute damage measurement. The Sentinel-2 pair in particular is materially confounded by cloud cover and should not be used alone for damage quantification.</p>
    </div>
  </section>

  <footer>
    Method: ORB+RANSAC registration → masked SSIM / abs-diff / Otsu change mask (OpenCV, scikit-image, NumPy) + DINOv2 (facebook/dinov2-base, via Hugging Face Transformers) global and patch-level cosine similarity. Figures generated at analysis_output/*.png; source imagery in ImageDetection/images/.
  </footer>

</div>
"""

with open(f"{SCRATCH}/treasure_island_report.html", "w") as f:
    f.write(html)

print("wrote report, length:", len(html))
