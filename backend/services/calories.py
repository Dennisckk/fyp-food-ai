import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
DB_PATH = ROOT / "backend" / "calorie_db.json"

# size multipliers (tune later)
MULTIPLIER = {
    "small": 0.5,
    "medium": 1.0,
    "large": 1.5
}

def load_db() -> dict:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    return {}

CAL_DB = load_db()

def estimate_calories(detections: list[dict]) -> tuple[list[dict], float]:
    total = 0.0
    out = []

    for d in detections:
        label = d.get("label", "")
        size = d.get("size", "medium")

        base = float(CAL_DB.get(label, 0))               # calories per serving
        mult = float(MULTIPLIER.get(size, 1.0))
        cal = base * mult

        d2 = dict(d)
        d2["calories"] = round(cal, 2)
        d2["calories_basis"] = "per_serving_x_size"
        total += cal
        out.append(d2)

    return out, round(total, 2)
