"""Fine-tune and evaluate a custom YOLO detector, then export it for OpenCV."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))

from the_last_chai.object_detection import infer_image, load_model


def read_dataset_config(path: Path) -> Tuple[Dict[str, object], List[str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    raw_names = config.get("names")
    if isinstance(raw_names, list):
        names = raw_names
    elif isinstance(raw_names, dict):
        indexed_names = sorted((int(index), name) for index, name in raw_names.items())
        if [index for index, _ in indexed_names] != list(range(len(indexed_names))):
            raise ValueError("Class IDs in names must be consecutive integers starting at zero.")
        names = [name for _, name in indexed_names]
    else:
        raise ValueError("Dataset YAML must define class names as a list or indexed mapping.")
    if not names or not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Dataset class names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Dataset class names must be unique.")

    configured_root = Path(str(config["path"])) if config.get("path") else Path(".")
    dataset_root = configured_root if configured_root.is_absolute() else path.parent / configured_root
    resolved_splits = []
    for split in ("train", "val", "test"):
        source = config.get(split)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Dataset YAML must define a separate '{}' split.".format(split))
        split_path = Path(source)
        resolved = split_path if split_path.is_absolute() else dataset_root / split_path
        if not resolved.exists():
            raise ValueError("{} split does not exist: {}".format(split, resolved))
        resolved_splits.append(resolved.resolve())
    if len(set(resolved_splits)) != 3:
        raise ValueError("Train, validation and test splits must use distinct sources.")
    return config, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on a labeled dataset and export a custom OpenCV ONNX model.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "custom_yolo.yaml", help="YOLO dataset YAML with separate train, val and test entries")
    parser.add_argument("--weights", type=Path, default=MODEL_DIR / "yolov8n.pt", help="Pretrained YOLO checkpoint used for transfer learning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, for example cpu, mps or 0")
    parser.add_argument("--name", default="custom_yolo", help="Training run name under runs/custom_yolo/")
    arguments = parser.parse_args()

    dataset_path = arguments.data.expanduser().resolve()
    checkpoint = arguments.weights.expanduser().resolve()
    if not dataset_path.is_file():
        parser.error("Dataset YAML not found: {}. Copy data/custom_yolo.yaml.example and add annotated images first.".format(dataset_path))
    if not checkpoint.is_file():
        parser.error("Pretrained checkpoint not found: {}. Run python scripts/setup_model.py first.".format(checkpoint))
    try:
        _, class_names = read_dataset_config(dataset_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if min(arguments.epochs, arguments.imgsz, arguments.batch, arguments.patience) < 1 or arguments.workers < 0:
        parser.error("epochs, imgsz, batch and patience must be positive; workers cannot be negative.")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error("Install the optional model dependencies first: python -m pip install -e '.[model]' ({})".format(error))

    model = YOLO(str(checkpoint))
    train_options = {
        "data": str(dataset_path),
        "epochs": arguments.epochs,
        "imgsz": arguments.imgsz,
        "batch": arguments.batch,
        "patience": arguments.patience,
        "workers": arguments.workers,
        "project": str(ROOT / "runs" / "custom_yolo"),
        "name": arguments.name,
    }
    if arguments.device:
        train_options["device"] = arguments.device
    model.train(**train_options)
    best_checkpoint = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError("Training completed without a best.pt checkpoint: {}".format(best_checkpoint))

    best_model = YOLO(str(best_checkpoint))
    test_metrics = best_model.val(data=str(dataset_path), split="test", imgsz=arguments.imgsz, batch=arguments.batch)
    exported_path = Path(best_model.export(format="onnx", imgsz=640, dynamic=False, simplify=False, opset=12))
    if not exported_path.is_file():
        raise FileNotFoundError("Ultralytics export did not create an ONNX model: {}".format(exported_path))
    probe = np.zeros((640, 640, 3), dtype=np.uint8)
    infer_image(load_model(str(exported_path)), probe, confidence=0.99, class_names=class_names)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weights_target = MODEL_DIR / "custom_yolo.onnx"
    weights_temporary = MODEL_DIR / "custom_yolo.onnx.tmp"
    shutil.copyfile(exported_path, weights_temporary)
    weights_temporary.replace(weights_target)
    classes_target = MODEL_DIR / "custom_yolo_classes.json"
    classes_temporary = MODEL_DIR / "custom_yolo_classes.json.tmp"
    classes_temporary.write_text(json.dumps(class_names, indent=2) + "\n", encoding="utf-8")
    classes_temporary.replace(classes_target)
    metrics = {name: float(value) for name, value in test_metrics.results_dict.items()}
    report = {"checkpoint": str(best_checkpoint), "dataset": str(dataset_path), "class_names": class_names, "test_metrics": metrics}
    (MODEL_DIR / "custom_yolo_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Exported custom detector: {}".format(weights_target))
    print("Class names: {}".format(", ".join(class_names)))
    print("Held-out test metrics: {}".format(json.dumps(metrics, sort_keys=True)))
    print("The local portal will prefer this model on its next analysis request.")


if __name__ == "__main__":
    main()