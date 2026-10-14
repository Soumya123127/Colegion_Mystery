"""Audit generated image/video assets and manifest paths."""

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest_path = ROOT / "data/manifests/asset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run python scripts/prepare_data.py first")
    manifest = json.loads(manifest_path.read_text())
    missing = []
    unreadable = []
    image_paths = list(manifest["images"].values()) + manifest.get("references", []) + manifest.get("portraits", [])
    video_records = [manifest["video"]] + manifest.get("additional_videos", [])
    video_paths = [record["path"] for record in video_records]
    dataset_path = ROOT / "data/manifests/dataset_manifest.json"
    if not dataset_path.is_file():
        missing.append("data/manifests/dataset_manifest.json")
        dataset = {"files": [], "model": {"required": False}}
    else:
        dataset = json.loads(dataset_path.read_text())
    dataset_entries = dataset.get("files", [])
    listed_paths = {entry.get("filename") for entry in dataset_entries}
    for entry in dataset_entries:
        relative = entry.get("filename")
        if not relative or not entry.get("purpose") or not entry.get("creation_method") or not entry.get("license"):
            unreadable.append("dataset manifest entry missing required provenance fields: {}".format(relative))
        elif not (ROOT / relative).is_file():
            missing.append(relative)
    for relative in image_paths + video_paths:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            image = cv2.imread(str(path))
            if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                unreadable.append(relative)
        else:
            video = cv2.VideoCapture(str(path))
            ok, frame = video.read()
            count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            video.release()
            if not ok or count <= 0 or fps <= 0 or frame is None:
                unreadable.append(relative)
    for record in video_records:
        path = ROOT / record["path"]
        if not path.is_file():
            continue
        video = cv2.VideoCapture(str(path))
        actual_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        if actual_frames != record["frame_count"] or abs(actual_fps - record["fps"]) > 0.1:
            unreadable.append("{} metadata differs from asset manifest".format(record["path"]))
    if dataset.get("model", {}).get("required") and not (ROOT / dataset["model"]["path"]).is_file():
        missing.append(dataset["model"]["path"])
    if not listed_paths.issuperset(image_paths + video_paths):
        unreadable.append("dataset manifest omits files listed by asset_manifest.json")
    if missing or unreadable:
        raise RuntimeError("Missing: {} | unreadable: {}".format(missing, unreadable))
    (ROOT / "outputs").mkdir(exist_ok=True)
    probe = ROOT / "outputs" / ".write-check"
    probe.write_text("ok")
    probe.unlink()
    model_message = "required weights present" if (ROOT / "models/yolov8n.onnx").is_file() else "optional YOLO weights not installed; no semantic inference will be reported"
    print("Asset audit passed: {} images, {} videos, dataset provenance and writable outputs; {}.".format(len(image_paths), len(video_paths), model_message))


if __name__ == "__main__":
    verify()