from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont  
from fastapi.templating import Jinja2Templates
from pathlib import Path
from PIL import Image
from .services.calories import estimate_calories
from .services.inference import run_inference
import io
from fastapi.responses import Response
from .services.inference import predict_raw, render_annotated_png_bytes
# from fastapi import Query
from PIL import Image, ImageOps, UnidentifiedImageError
import csv
from datetime import datetime
from uuid import uuid4
from pathlib import Path
# from fastapi import HTTPException
from fastapi import Response, HTTPException
import csv, json
from datetime import datetime
from fastapi.responses import FileResponse, JSONResponse
# from .services.inference import gradcam_png_bytes
from .services.inference import explain_png_bytes
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from backend.services.inference import explain_png_bytes
import cv2

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../fyp-food-ai
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HISTORY_CSV = OUTPUTS_DIR / "predictions_history.csv"

HISTORY_HEADERS = [
    "timestamp",
    "filename",
    "conf",
    "total_calories_estimate",
    "items",      # short summary like "fish:small:140 | rice:medium:220"
    "raw_json"    # full JSON for debugging
]

def _summarize_items(detections: list[dict]) -> str:
    parts = []
    for d in detections:
        label = d.get("label", "")
        size = d.get("size", "")
        cal = d.get("calories", "")
        parts.append(f"{label}:{size}:{cal}")
    return " | ".join(parts)

def append_history(payload: dict, conf: float):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    detections = payload.get("detections", []) or []
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": payload.get("filename", ""),
        "conf": conf,
        "total_calories_estimate": payload.get("total_calories_estimate", 0),
        "items": _summarize_items(detections),
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }

    file_exists = HISTORY_CSV.exists()
    with HISTORY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_HEADERS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

@app.get("/history")
def get_history(page: int = 1, page_size: int = 10, max_items: int = 0):
    """
    page_size = rows per page (you want fixed 10)
    max_items = cap how many newest rows are available in history (20/50/100). 0 = all
    """
    # sanitize
    if page < 1:
        page = 1

    if page_size < 1:
        page_size = 10
    if page_size > 50:
        page_size = 50  # safety

    if max_items < 0:
        max_items = 0
    if max_items > 500:
        max_items = 500  # safety

    if not HISTORY_CSV.exists():
        return {"page": page, "page_size": page_size, "max_items": max_items, "total": 0, "rows": []}

    with HISTORY_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # newest first
    rows = rows[::-1]

    # apply cap (max_items)
    if max_items > 0:
        rows = rows[:max_items]

    total = len(rows)

    start = (page - 1) * page_size
    end = start + page_size
    page_rows = rows[start:end]

    return {"page": page, "page_size": page_size, "max_items": max_items, "total": total, "rows": page_rows}

@app.get("/history.csv")
def download_history_csv(page: int = 1, page_size: int = 0, max_items: int = 0):
    """
    - If page_size=0: download ALL rows (but still obey max_items if >0)
    - If page_size>0: download ONLY that page slice (obeys max_items too)
    """
    if page < 1:
        page = 1

    if page_size < 0:
        page_size = 0
    if page_size > 200:
        page_size = 200

    if max_items < 0:
        max_items = 0
    if max_items > 500:
        max_items = 500

    if not HISTORY_CSV.exists():
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with HISTORY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_HEADERS)
            w.writeheader()

    with HISTORY_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # newest first
    rows = rows[::-1]

    # apply cap (max_items)
    if max_items > 0:
        rows = rows[:max_items]

    # If page_size > 0, export ONLY that page slice
    if page_size > 0:
        start = (page - 1) * page_size
        end = start + page_size
        rows = rows[start:end]
        filename = f"predictions_history_p{page}_ps{page_size}_cap{max_items or 'all'}.csv"
    else:
        filename = f"predictions_history_cap{max_items or 'all'}.csv"

    import io as _io
    buf = _io.StringIO()
    w = csv.DictWriter(buf, fieldnames=HISTORY_HEADERS)
    w.writeheader()
    for r in rows:
        w.writerow(r)

    csv_bytes = buf.getvalue().encode("utf-8")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=csv_bytes, media_type="text/csv", headers=headers)

# Allow your HTML page to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later change to your real frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# point templates to ../frontend
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR.parent / "frontend"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/report/{report_id}.csv")
def download_report_csv(report_id: str):
    reports = getattr(app.state, "reports", {})
    report = reports.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found (click Predict again)")

    csv_bytes = report_to_csv_bytes(report)
    headers = {"Content-Disposition": f'attachment; filename="report_{report_id}.csv"'}
    return Response(content=csv_bytes, media_type="text/csv", headers=headers)


ROOT = Path(__file__).resolve().parents[1]  # project root (…/fyp-food-ai)
OUTPUT_DIR = BASE_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PRED_CSV = OUTPUT_DIR / "predictions.csv"

def _ensure_csv_header():
    if not PRED_CSV.exists():
        with PRED_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","report_id","filename",
                "label","confidence","size","area_ratio","calories",
                "total_calories_estimate"
            ])

def append_prediction_to_csv(report: dict):
    _ensure_csv_header()
    ts = datetime.now().isoformat(timespec="seconds")
    report_id = report["report_id"]
    filename = report.get("filename", "")
    total = report.get("total_calories_estimate", 0)

    dets = report.get("detections", []) or []
    with PRED_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not dets:
            w.writerow([ts, report_id, filename, "", "", "", "", "", total])
            return
        for d in dets:
            w.writerow([
                ts, report_id, filename,
                d.get("label",""),
                d.get("confidence",""),
                d.get("size",""),
                d.get("area_ratio",""),
                d.get("calories",""),
                total
            ])

def report_to_csv_bytes(report: dict) -> bytes:
    import io as _io
    buf = _io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", report.get("filename", "")])
    w.writerow(["report_id", report.get("report_id", "")])
    w.writerow(["total_calories_estimate", report.get("total_calories_estimate", 0)])
    w.writerow([])
    w.writerow(["label","confidence","size","area_ratio","calories"])
    for d in report.get("detections", []) or []:
        w.writerow([
            d.get("label",""),
            d.get("confidence",""),
            d.get("size",""),
            d.get("area_ratio",""),
            d.get("calories","")
        ])
    return buf.getvalue().encode("utf-8")


def preprocess_upload(img: Image.Image, max_side: int = 1280) -> Image.Image:
    img = ImageOps.exif_transpose(img)  # fix rotation from phone EXIF
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))  # downscale (keeps aspect ratio)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.35)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    MAX_SIDE = 1280  # you can try 1024 or 1280
    w, h = img.size
    m = max(w, h)
    if m > MAX_SIDE:
        scale = MAX_SIDE / m
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    detections = run_inference(img, conf_thres=conf)
    detections, total_cal = estimate_calories(detections)

    report_id = uuid4().hex
    report = {
        "report_id": report_id,
        "filename": file.filename,
        "detections": detections,
        "total_calories_estimate": total_cal
    }

    # store in memory so /report/... can download it
    if not hasattr(app.state, "reports"):
        app.state.reports = {}
    app.state.reports[report_id] = report

    append_history(report, conf)
    append_prediction_to_csv(report)

    return report

def build_explanation_text(filename: str, conf: float, detections: list, total_cal: float) -> str:
    if not detections:
        return (
            "No ingredients were confidently detected. Try lowering the confidence threshold, "
            "move closer so the food fills the image, or use better lighting."
        )

    # sort by confidence (high → low)
    dets = sorted(detections, key=lambda d: float(d.get("confidence", 0)), reverse=True)

    top = dets[0]
    top_label = top.get("label", "ingredient")
    top_conf = float(top.get("confidence", 0))
    top_size = top.get("size", "-")
    top_cal = float(top.get("calories", 0))

    # top-3 summary
    top3 = dets[:3]
    top3_str = ", ".join([f"{d.get('label')} ({float(d.get('confidence',0)):.2f})" for d in top3])

    # portion summary
    size_counts = {}
    for d in dets:
        s = d.get("size", "-")
        size_counts[s] = size_counts.get(s, 0) + 1
    size_str = ", ".join([f"{k}:{v}" for k, v in size_counts.items() if k != "-"]) or "unknown"

    msg = (
        f"Detected {len(dets)} ingredient(s): {top3_str}. "
        f"The model focused most on **{top_label}** (confidence {top_conf:.2f}), "
        f"estimated portion **{top_size}**, about **{top_cal:.0f} kcal** for this item. "
        f"Portion distribution: {size_str}. "
        f"Total estimated calories: **{float(total_cal):.0f} kcal**."
    )
    return msg


@app.post("/explain_text")
async def explain_text(file: UploadFile = File(...), conf: float = 0.35):
    contents = await file.read()
    img = load_upload_image(contents)  # your existing loader

    detections = run_inference(img, conf_thres=conf)
    detections, total_cal = estimate_calories(detections)

    text = build_explanation_text(file.filename, conf, detections, total_cal)

    return JSONResponse({
        "filename": file.filename,
        "conf": conf,
        "num_detections": len(detections),
        "total_calories_estimate": total_cal,
        "explanation": text
    })

@app.post("/predict_annotated")
async def predict_annotated(file: UploadFile = File(...), conf: float = Query(0.35)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = preprocess_upload(img, max_side=1280)

    # IMPORTANT: set conf lower, because your model has predictions around 0.14 / 0.23
    r = predict_raw(img, conf_thres=conf)
    print("ANNOTATED DEBUG:", 0 if r.boxes is None else len(r.boxes), "masks:", r.masks is not None)

    png_bytes = render_annotated_png_bytes(r)
    return Response(content=png_bytes, media_type="image/png")

def load_upload_image(contents: bytes) -> Image.Image:
    """Loads jpg/png/webp safely and returns RGB PIL Image."""
    try:
        img = Image.open(io.BytesIO(contents))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except UnidentifiedImageError:
        # fallback for formats Pillow can't open (often WEBP if pillow built without webp)
        arr = np.frombuffer(contents, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

@app.post("/explain")
async def explain(file: UploadFile = File(...), conf: float = 0.35):
    contents = await file.read()
    img = load_upload_image(contents)

    png_bytes = explain_png_bytes(img, conf_thres=conf)  # from inference.py
    return Response(content=png_bytes, media_type="image/png")

def draw_boxes(img: Image.Image, detections: list[dict]):
    draw = ImageDraw.Draw(img)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["label"]} {d.get("size","")} ({d["confidence"]:.2f})'

        # box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        # label background + text
        text_w = draw.textlength(label)
        draw.rectangle([x1, y1 - 24, x1 + text_w + 10, y1], fill="red")
        draw.text((x1 + 5, y1 - 22), label, fill="white")
