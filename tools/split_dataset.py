import random
import shutil
import re
from pathlib import Path

# ====== SETTINGS ======
SEED = 42
SRC = Path("raw_images")  # your current folder
OUT = Path("datasets/ingredients-seg/images")
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMAGES_PER_FORM = 5  # skip tiny folders (optional)

# ====== HELPERS ======
def slug(s: str) -> str:
    """Make a safe filename piece from folder names (spaces -> _, remove weird chars)."""
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s[:60] if len(s) > 60 else s

def ensure_dirs():
    for split in SPLITS:
        (OUT / split).mkdir(parents=True, exist_ok=True)

def split_items(items: list[Path]):
    n = len(items)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

def copy_set(items: list[Path], split_name: str, cls_name: str, form_name: str):
    cls_s = slug(cls_name)
    form_s = slug(form_name)

    for i, p in enumerate(items, start=1):
        # keep extension
        ext = p.suffix.lower()
        # unique name prevents overwrite
        new_name = f"{cls_s}__{form_s}__{i:04d}{ext}"
        dest = OUT / split_name / new_name
        shutil.copy2(p, dest)

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Cannot find folder: {SRC.resolve()}")

    random.seed(SEED)
    ensure_dirs()

    total = {"train": 0, "val": 0, "test": 0}

    for cls_dir in sorted([p for p in SRC.iterdir() if p.is_dir()]):
        for form_dir in sorted([p for p in cls_dir.iterdir() if p.is_dir()]):
            imgs = [p for p in form_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS]

            if len(imgs) < MIN_IMAGES_PER_FORM:
                print(f"Skip (too few): {cls_dir.name}/{form_dir.name} -> {len(imgs)}")
                continue

            random.shuffle(imgs)
            tr, va, te = split_items(imgs)

            copy_set(tr, "train", cls_dir.name, form_dir.name)
            copy_set(va, "val", cls_dir.name, form_dir.name)
            copy_set(te, "test", cls_dir.name, form_dir.name)

            total["train"] += len(tr)
            total["val"] += len(va)
            total["test"] += len(te)

            print(f"{cls_dir.name}/{form_dir.name}: {len(tr)} train, {len(va)} val, {len(te)} test")

    print("\nDONE. Totals:", total)
    print("Output folder:", OUT.resolve())

if __name__ == "__main__":
    main()
