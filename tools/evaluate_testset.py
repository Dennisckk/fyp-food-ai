from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

from PIL import Image, ImageOps
from ultralytics import YOLO


# ----------------------------
# Helpers
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess_image(img: Image.Image, max_side: int = 1280) -> Image.Image:
    """Match your app behavior: EXIF transpose + RGB + downscale."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def parse_data_yaml_names(data_yaml: Path) -> Dict[int, str]:
    """
    Minimal parser for data.yaml to read:
    names:
      0: egg
      1: rice
      ...
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml.resolve()}")

    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    names: Dict[int, str] = {}
    in_names = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("names:"):
            in_names = True
            continue

        if in_names:
            # stop if we hit another top-level key like "train:" or "path:"
            if ":" in line and not line[0].isdigit():
                # e.g., "train: images/train"
                break

            # expect "0: egg"
            if ":" in line:
                left, right = line.split(":", 1)
                left = left.strip()
                right = right.strip().strip('"').strip("'")
                if left.isdigit():
                    names[int(left)] = right

    if not names:
        raise ValueError("Could not parse names from data.yaml (names section empty).")

    return names


def load_calorie_db(calorie_db_path: Path) -> Dict[str, float]:
    import json
    if not calorie_db_path.exists():
        raise FileNotFoundError(f"calorie_db.json not found: {calorie_db_path.resolve()}")
    data = json.loads(calorie_db_path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def size_from_ratio(r: float) -> str:
    # same idea as your MVP (tune later)
    if r < 0.10:
        return "small"
    elif r < 0.25:
        return "medium"
    else:
        return "large"


MULTIPLIER = {
    "small": 0.7,
    "medium": 1.0,
    "large": 1.3,
}


@dataclass
class Detection:
    label: str
    confidence: float
    mask_area_px: int
    area_ratio: float
    size: str
    calories: float


def yolo_predict(
    model: YOLO,
    img: Image.Image,
    conf: float,
    iou: float,
    imgsz: int,
    allowed: Set[str] | None,
    calorie_db: Dict[str, float],
) -> Tuple[List[Detection], float]:
    """
    Returns (detections, total_calories) using:
    - masks area
    - dedupe: keep largest mask per label
    - size ratio: mask_area / max_mask_area (more stable across different resolutions)
    """
    results = model.predict(source=img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r = results[0]

    if r.boxes is None or r.masks is None:
        return [], 0.0

    names = r.names
    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes

    raw = []
    for i in range(len(boxes)):
        b = boxes[i]
        cls_id = int(b.cls[0])
        label = str(names.get(cls_id, cls_id)).lower().strip()
        confidence = float(b.conf[0])

        if allowed is not None and label not in allowed:
            continue

        mask_area_px = int((masks[i] > 0.5).sum())
        raw.append((label, confidence, mask_area_px))

    if not raw:
        return [], 0.0

    # Dedupe: keep largest mask per label
    by_label: Dict[str, List[Tuple[str, float, int]]] = {}
    for t in raw:
        by_label.setdefault(t[0], []).append(t)

    kept: List[Tuple[str, float, int]] = []
    for label, items in by_label.items():
        items.sort(key=lambda x: x[2], reverse=True)  # by mask_area_px
        kept.append(items[0])

    max_area = max(a for (_, _, a) in kept) or 1

    dets: List[Detection] = []
    total = 0.0
    for label, confv, area in sorted(kept, key=lambda x: x[2], reverse=True):
        ratio = area / max_area
        size = size_from_ratio(ratio)

        base = float(calorie_db.get(label, 0.0))
        mult = float(MULTIPLIER.get(size, 1.0))
        cal = base * mult

        dets.append(
            Detection(
                label=label,
                confidence=confv,
                mask_area_px=area,
                area_ratio=ratio,
                size=size,
                calories=cal,
            )
        )
        total += cal

    return dets, round(total, 2)


def gt_labels_for_image(labels_dir: Path, img_path: Path, id2name: Dict[int, str]) -> Set[str]:
    """
    Reads YOLO label txt for this image and returns set of GT labels (image-level).
    """
    txt = labels_dir / (img_path.stem + ".txt")
    if not txt.exists():
        return set()

    out = set()
    for line in txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        name = id2name.get(cls_id, str(cls_id))
        out.add(name.lower().strip())
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/ingredients-seg/data.yaml", help="Path to data.yaml")
    ap.add_argument("--weights", default="models/ingredient_yolo_seg.pt", help="Path to trained weights")
    ap.add_argument("--caldb", default="backend/calorie_db.json", help="Path to calorie_db.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    args = ap.parse_args()

    root = Path(".").resolve()
    data_yaml = root / args.data
    weights = root / args.weights
    caldb_path = root / args.caldb

    id2name = parse_data_yaml_names(data_yaml)
    allowed = {v.lower().strip() for v in id2name.values()}  # allowed labels = your dataset classes
    calorie_db = load_calorie_db(caldb_path)

    images_dir = root / "datasets" / "ingredients-seg" / "images" / args.split
    labels_dir = root / "datasets" / "ingredients-seg" / "labels" / args.split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir.resolve()}")

    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    per_img_csv = out_dir / "eval_per_image.csv"
    summary_csv = out_dir / "eval_summary.csv"
    no_det_txt = out_dir / "no_detections.txt"

    # Load model
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights.resolve()}")
    model = YOLO(str(weights))
    print("✅ Loaded model:", weights)

    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if args.limit and args.limit > 0:
        img_paths = img_paths[: args.limit]

    # Per-class counters (image-level presence)
    # TP: predicted label present AND GT present
    # FP: predicted label present but GT absent
    # FN: GT present but predicted absent
    labels = sorted(allowed)
    TP = {c: 0 for c in labels}
    FP = {c: 0 for c in labels}
    FN = {c: 0 for c in labels}

    # Track no detections
    no_det = []

    with per_img_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename",
            "gt_labels",
            "pred_labels",
            "num_pred",
            "avg_conf",
            "total_calories_estimate",
            "conf",
            "iou",
            "imgsz",
            "runtime_ms"
        ])

        for idx, img_path in enumerate(img_paths, start=1):
            t0 = time.time()

            # Load + preprocess
            img = Image.open(img_path)
            img = preprocess_image(img, max_side=args.max_side)

            # Predict
            dets, total_cal = yolo_predict(
                model=model,
                img=img,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                allowed=allowed,
                calorie_db=calorie_db,
            )

            dt_ms = int((time.time() - t0) * 1000)

            pred_set = {d.label for d in dets}
            gt_set = gt_labels_for_image(labels_dir, img_path, id2name)

            if not dets:
                no_det.append(img_path.name)

            # update per-class counts
            for c in labels:
                in_gt = c in gt_set
                in_pred = c in pred_set
                if in_gt and in_pred:
                    TP[c] += 1
                elif (not in_gt) and in_pred:
                    FP[c] += 1
                elif in_gt and (not in_pred):
                    FN[c] += 1

            avg_conf = round(sum(d.confidence for d in dets) / len(dets), 4) if dets else 0.0

            w.writerow([
                img_path.name,
                ";".join(sorted(gt_set)),
                ";".join(sorted(pred_set)),
                len(dets),
                avg_conf,
                total_cal,
                args.conf,
                args.iou,
                args.imgsz,
                dt_ms
            ])

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(img_paths)}...")

    # Write no detections list
    no_det_txt.write_text("\n".join(no_det), encoding="utf-8")

    # Write summary CSV (per-class precision/recall/F1, image-level)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "TP", "FP", "FN", "precision", "recall", "f1"])
        for c in labels:
            tp, fp, fn = TP[c], FP[c], FN[c]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            w.writerow([c, tp, fp, fn, round(prec, 4), round(rec, 4), round(f1, 4)])

    print("✅ Done!")
    print("Per-image CSV:", per_img_csv.resolve())
    print("Summary CSV :", summary_csv.resolve())
    print("No-detect list:", no_det_txt.resolve())


if __name__ == "__main__":
    main()
