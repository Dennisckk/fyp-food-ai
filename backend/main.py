from fastapi import FastAPI, UploadFile, File, Request
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
from fastapi import Query
from PIL import Image, ImageOps
import csv
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from fastapi import HTTPException

app = FastAPI()

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
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    detections = run_inference(img, conf_thres=0.35)
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

    # append to global CSV log
    append_prediction_to_csv(report)

    return report

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
