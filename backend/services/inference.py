from ultralytics import YOLO
import numpy as np
# from PIL import Image
from pathlib import Path
from collections import defaultdict
import torch
import cv2
import io
from PIL import Image, ImageOps, ImageFile

# Put your trained weights later in: fyp-food-ai/models/ingredient_yolo.pt
ROOT = Path(__file__).resolve().parents[2]  # .../fyp-food-ai
CUSTOM_WEIGHTS = ROOT / "models" / "ingredient_yolo_seg.pt"

MODEL_PATH = str(CUSTOM_WEIGHTS) if CUSTOM_WEIGHTS.exists() else "yolov8n-seg.pt"
model = YOLO(MODEL_PATH)

print("✅ YOLO-SEG model loaded:", MODEL_PATH)

# Your target ingredient labels (once you train your own model, these should match your dataset class names)
ALLOWED = {"egg", "rice", "chicken", "fish", "beef", "noodles", "bread", "vegetable", "shrimp", "tofu"}

def predict_raw(pil_img: Image.Image, conf_thres: float = 0.25):
    """Return the Ultralytics Result object (with boxes/masks)."""
    pil_img = pil_img.convert("RGB")
    results = model.predict(source=pil_img, conf=conf_thres, iou=0.5, imgsz=640, verbose=False)
    r = results[0]

    # ✅ DEBUG (watch terminal output)
    print("RAW names:", r.names)
    print("RAW boxes:", 0 if r.boxes is None else len(r.boxes))
    print("RAW masks:", "None" if r.masks is None else tuple(r.masks.data.shape))
    if r.boxes is not None:
        n = min(len(r.boxes), 10)
        for i in range(n):
            b = r.boxes[i]
            print("  cls:", int(b.cls[0]), "conf:", float(b.conf[0]))

    return r

def render_annotated_png_bytes(result):
    """
    Convert Ultralytics rendered output (BGR numpy) into PNG bytes.
    """
    import io
    # Ultralytics plot() returns BGR image (numpy)
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]  # BGR -> RGB
    pil_out = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return buf.getvalue()

def size_from_ratio(r: float) -> str:
    # Simple thresholds (tune later)
    if r < 0.10:
        return "small"
    elif r < 0.25:
        return "medium"
    else:
        return "large"

def run_inference(pil_img: Image.Image, conf_thres: float = 0.35):
    pil_img = pil_img.convert("RGB")
    r = predict_raw(pil_img, conf_thres=conf_thres)

    detections = []
    names = r.names

    if r.masks is None or r.boxes is None:
        return detections

    masks = r.masks.data.cpu().numpy()
    mh, mw = masks.shape[1], masks.shape[2]
    image_area = mh * mw
    boxes = r.boxes

    for i in range(len(boxes)):
        b = boxes[i]
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        label = str(names.get(cls_id, cls_id))

        # Optional: filter out non-ingredient classes
        # if label not in ALLOWED:
        #     continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        mask_area_px = int((masks[i] > 0.5).sum())

        detections.append({
            "label": label,
            "confidence": round(conf, 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "mask_area_px": mask_area_px,
            "area_ratio": 0.0,   # fill later
            "size": "medium"     # fill later
        })

    if not detections:
        return []

    by_label = defaultdict(list)
    for d in detections:
        by_label[d["label"]].append(d)

    keep = []
    for lbl, items in by_label.items():
        items.sort(key=lambda x: x["mask_area_px"], reverse=True)
        keep.append(items[0])   # top 1 per label

    detections = keep

    max_area = max(d["mask_area_px"] for d in detections) or 1
    for d in detections:
        ratio = d["mask_area_px"] / max_area
        d["area_ratio"] = round(ratio, 6)
        d["size"] = size_from_ratio(ratio)

    detections.sort(key=lambda x: x["mask_area_px"], reverse=True)
    return detections

def _get_last_conv_layer(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def _encode_png(rgb: np.ndarray) -> bytes:
    """rgb: HxWx3 uint8 (RGB) -> PNG bytes"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def explain_png_bytes(pil_img: Image.Image, conf_thres: float = 0.25, imgsz: int = 640, alpha: float = 0.45) -> bytes:
    """
    Returns a heatmap-overlay PNG using predicted SEG masks weighted by confidence.
    This is more stable than true Grad-CAM for YOLO-seg.
    """
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # YOLO predict
    results = model.predict(source=pil_img, conf=conf_thres, imgsz=imgsz, verbose=False)
    if not results:
        return _encode_png(np.array(pil_img))

    r = results[0]

    # No detections or no masks -> return original
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return _encode_png(np.array(pil_img))

    masks = r.masks.data.detach().cpu().numpy()   # (N, Hm, Wm)
    confs = r.boxes.conf.detach().cpu().numpy()   # (N,)

    # Weighted heatmap: sum(mask_i * conf_i)
    heat = np.tensordot(confs, masks, axes=(0, 0))  # (Hm, Wm)
    heat = np.clip(heat, 0, None)
    if heat.max() > 0:
        heat = heat / heat.max()

    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR

    base = np.array(pil_img)  # RGB (H, W, 3)

    # Resize heatmap to original image size
    heat_color = cv2.resize(heat_color, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = (base * (1 - alpha) + heat_color * alpha).astype(np.uint8)
    return _encode_png(overlay)
