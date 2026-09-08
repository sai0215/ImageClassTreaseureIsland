"""Train a per-building damage classifier on xBD, using the same DINOv2
footprint-pooling method as footprint_compare.py, so it can be applied to
Treasure Island building embeddings once georeferenced imagery is available.

This is the supervised upgrade to footprint_compare.py's raw cosine-distance
heuristic: instead of just measuring how much a building's embedding shifted
between pre/post, we train a linear probe (same recipe HASTE uses) on xBD's
~850k labeled buildings to predict an actual damage category from the
post-disaster embedding.

xBD access (gated, not automatable):
    1. Register at https://xview2.org and accept the terms.
    2. Download the "Datasets from the Challenge" tiles (~10GB compressed),
       excluding the holdout set.
    3. Unzip so you end up with:
         <xbd_root>/<disaster>/images/<chip>_pre_disaster.png
         <xbd_root>/<disaster>/images/<chip>_post_disaster.png
         <xbd_root>/<disaster>/labels/<chip>_post_disaster.json
       (pre_disaster.json exists too but every polygon in it is labeled
       "no-damage" by construction — it's just the footprint layer, so we
       don't use it for labels, only post_disaster.json.)

Usage (once you have the data):
    python3 train_xbd_probe.py --xbd-root /path/to/xBD --disasters hurricane-harvey hurricane-michael hurricane-florence

Run with no --xbd-root to see this message and exit.
"""
import argparse
import json
import os

import cv2
import joblib
import numpy as np
import torch
from shapely import wkt as shapely_wkt

from dinov2_compare import PATCH, DEVICE, model, processor

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
MIN_CROP_PX = PATCH * 4
PAD_FRAC = 0.25
BATCH_SIZE = 16


def _polygon_from_wkt(wkt_str):
    poly = shapely_wkt.loads(wkt_str)
    return np.array(poly.exterior.coords, dtype=np.float64)


def _crop_box_for_polygon(coords, img_w, img_h):
    minx, miny = coords.min(axis=0)
    maxx, maxy = coords.max(axis=0)
    h, w = maxy - miny, maxx - minx
    pad_h = max(h * PAD_FRAC, (MIN_CROP_PX - h) / 2 + 1)
    pad_w = max(w * PAD_FRAC, (MIN_CROP_PX - w) / 2 + 1)

    cy, cx = (miny + maxy) / 2, (minx + maxx) / 2
    size = max(h + 2 * pad_h, w + 2 * pad_w, MIN_CROP_PX)
    size = int(size - size % PATCH)
    size = max(size, PATCH)

    x0 = int(cx - size / 2)
    y0 = int(cy - size / 2)
    x0 = min(max(x0, 0), max(img_w - size, 0))
    y0 = min(max(y0, 0), max(img_h - size, 0))
    return x0, y0, size


def _polygon_patch_mask(coords, x0, y0, size, grid):
    local = (coords - [x0, y0]).astype(np.int32)
    px_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(px_mask, [local], 1)
    pooled = px_mask.reshape(grid, PATCH, grid, PATCH).max(axis=(1, 3))
    return pooled.astype(bool)


@torch.no_grad()
def _embed_crop(crop_rgb, size):
    inputs = processor(images=crop_rgb, do_resize=False, do_center_crop=False, return_tensors="pt").to(DEVICE)
    out = model(**inputs, interpolate_pos_encoding=True)
    grid = size // PATCH
    tokens = out.last_hidden_state[0, -grid * grid:, :]
    return tokens.reshape(grid, grid, -1).cpu().numpy(), grid


def _pooled_embedding(img_rgb, coords):
    h, w = img_rgb.shape[:2]
    x0, y0, size = _crop_box_for_polygon(coords, w, h)
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    sub = img_rgb[y0:min(y0 + size, h), x0:min(x0 + size, w)]
    crop[: sub.shape[0], : sub.shape[1]] = sub

    grid_feats, grid = _embed_crop(crop, size)
    patch_mask = _polygon_patch_mask(coords, x0, y0, size, grid)
    if not patch_mask.any():
        patch_mask[grid // 2, grid // 2] = True
    return grid_feats[patch_mask].mean(axis=0)


def iter_labeled_buildings(xbd_root, disasters=None, max_per_disaster=None):
    disaster_dirs = sorted(
        d for d in os.listdir(xbd_root)
        if os.path.isdir(os.path.join(xbd_root, d)) and (disasters is None or d in disasters)
    )
    for disaster in disaster_dirs:
        label_dir = os.path.join(xbd_root, disaster, "labels")
        image_dir = os.path.join(xbd_root, disaster, "images")
        if not os.path.isdir(label_dir):
            continue
        post_labels = sorted(f for f in os.listdir(label_dir) if f.endswith("_post_disaster.json"))
        n_from_this_disaster = 0
        for label_file in post_labels:
            chip = label_file[: -len("_post_disaster.json")]
            post_png = os.path.join(image_dir, f"{chip}_post_disaster.png")
            if not os.path.exists(post_png):
                continue
            with open(os.path.join(label_dir, label_file)) as f:
                labels = json.load(f)
            buildings = [
                (feat["properties"].get("uid"), feat["wkt"], feat["properties"].get("subtype"))
                for feat in labels["features"]["xy"]
            ]
            if not buildings:
                continue
            yield disaster, chip, post_png, buildings
            n_from_this_disaster += len(buildings)
            if max_per_disaster and n_from_this_disaster >= max_per_disaster:
                break


def extract_features(xbd_root, disasters=None, max_per_disaster=None):
    X, y = [], []
    cur_chip = None
    cur_img = None
    for disaster, chip, post_png, buildings in iter_labeled_buildings(xbd_root, disasters, max_per_disaster):
        if chip != cur_chip:
            cur_img = cv2.cvtColor(cv2.imread(post_png), cv2.COLOR_BGR2RGB)
            cur_chip = chip
        for uid, wkt_str, subtype in buildings:
            if subtype not in DAMAGE_CLASSES:
                continue  # skip "un-classified"
            coords = _polygon_from_wkt(wkt_str)
            try:
                emb = _pooled_embedding(cur_img, coords)
            except Exception as e:
                print(f"  skipping {chip}/{uid}: {e}")
                continue
            X.append(emb)
            y.append(subtype)
        print(f"  {disaster}/{chip}: {len(buildings)} buildings, running total {len(X)}")
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xbd-root", help="Path to the unzipped xBD dataset root")
    parser.add_argument("--disasters", nargs="*", default=None, help="Restrict to these disaster subfolders (default: all)")
    parser.add_argument("--max-per-disaster", type=int, default=2000, help="Cap buildings sampled per disaster (default 2000; use 0 for no cap)")
    parser.add_argument("--out", default="models/xbd_dinov2_probe.joblib")
    args = parser.parse_args()

    if not args.xbd_root:
        print(__doc__)
        return

    max_per_disaster = args.max_per_disaster or None
    print(f"Extracting DINOv2 footprint embeddings from {args.xbd_root} ...")
    X, y = extract_features(args.xbd_root, args.disasters, max_per_disaster)
    print(f"Extracted {len(X)} labeled building embeddings, classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    if len(X) == 0:
        print("No labeled buildings found — check --xbd-root and --disasters.")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)

    y_pred = clf.predict(scaler.transform(X_test))
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf, "classes": list(clf.classes_)}, args.out)
    print(f"Saved trained probe to {args.out}")


if __name__ == "__main__":
    main()
