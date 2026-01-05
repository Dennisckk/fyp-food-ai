from roboflow import Roboflow
from pathlib import Path

API_KEY = "YOUR_PRIVATE_API_KEY"
WORKSPACE = "your-workspace-id"
PROJECT = "your-project-id"

BASE = Path("datasets/ingredients-seg/images")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)

def upload_folder(folder: Path, split_name: str):
    for img in folder.iterdir():
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            project.upload(str(img), split=split_name)  # split is supported :contentReference[oaicite:1]{index=1}

upload_folder(BASE / "train", "train")
upload_folder(BASE / "val", "valid")
upload_folder(BASE / "test", "test")

print("Done uploading.")
