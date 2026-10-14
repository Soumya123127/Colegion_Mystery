"""Download and export official YOLOv8n COCO weights for OpenCV DNN."""

from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CHECKPOINT = MODEL_DIR / "yolov8n.pt"
ONNX_MODEL = MODEL_DIR / "yolov8n.onnx"
CHECKPOINT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        from urllib.request import urlretrieve

        print("Downloading official YOLOv8n COCO checkpoint (about 6.2 MB).")
        urlretrieve(CHECKPOINT_URL, CHECKPOINT)
    if not ONNX_MODEL.is_file():
        print("Exporting fixed-size 640x640 ONNX weights. First-time setup may take a few minutes.")
        exported = Path(YOLO(str(CHECKPOINT)).export(format="onnx", imgsz=640, opset=12, simplify=False, dynamic=False))
        if exported.resolve() != ONNX_MODEL.resolve():
            exported.replace(ONNX_MODEL)
    if not ONNX_MODEL.is_file():
        raise RuntimeError("Ultralytics export did not create {}".format(ONNX_MODEL))
    net = cv2.dnn.readNet(str(ONNX_MODEL))
    net.setInput(cv2.dnn.blobFromImage(cv2.imread(str(ROOT / "assets/images/cctv_corrupted.png")), 1 / 255.0, (640, 640), swapRB=True, crop=False))
    output = net.forward(net.getUnconnectedOutLayersNames())
    print("Verified OpenCV DNN forward pass; output shapes: {}".format([tuple(item.shape) for item in output]))
    print("Model weights are local and ignored by Git: {}".format(ONNX_MODEL))
    print("Source: Ultralytics YOLOv8n COCO, AGPL-3.0; see docs/ASSET_CREDITS.md.")


if __name__ == "__main__":
    main()
