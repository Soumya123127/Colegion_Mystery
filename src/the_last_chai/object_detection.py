"""Optional OpenCV-DNN ONNX detector. No inference is invented when weights are absent."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def load_model(weights_path: str, config_path: Optional[str] = None) -> cv2.dnn_Net:
    weights = Path(weights_path)
    if not weights.is_file():
        raise FileNotFoundError("YOLO ONNX weights are missing at {}. Download the documented model or use the labelled scene-comparison fallback.".format(weights))
    return cv2.dnn.readNet(str(weights)) if config_path is None else cv2.dnn.readNet(str(weights), str(config_path))


def validate_detections(detections: List[Dict[str, object]], width: int, height: int) -> bool:
    return all(0 <= d["x"] < width and 0 <= d["y"] < height and d["w"] > 0 and d["h"] > 0 and d["x"] + d["w"] <= width and d["y"] + d["h"] <= height for d in detections)


def infer_image(
    net: cv2.dnn_Net,
    image: np.ndarray,
    confidence: float = 0.35,
    input_size: Tuple[int, int] = (640, 640),
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    height, width = image.shape[:2]
    input_width, input_height = input_size
    scale = min(input_width / width, input_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    letterboxed = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    letterboxed[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    blob = cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    raw = net.forward(net.getUnconnectedOutLayersNames())
    normalized = []
    for item in raw:
        output = np.squeeze(item)
        if output.ndim == 1:
            output = output.reshape(1, -1)
        if output.shape[0] < output.shape[1] and output.shape[0] <= 256:
            output = output.T
        normalized.append(output)
    rows = np.concatenate(normalized, axis=0)
    boxes, scores, class_ids = [], [], []
    class_count = len(class_names) if class_names is not None else len(COCO_CLASSES)
    for row in rows:
        if row.shape[0] < class_count + 4:
            continue
        if row.shape[0] == class_count + 4:
            class_offset = 4
        elif row.shape[0] == class_count + 5:
            class_offset = 5
        else:
            continue
        class_scores = row[class_offset:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id] if class_offset == 4 else row[4] * class_scores[class_id])
        if score < confidence:
            continue
        center_x, center_y, box_width, box_height = (float(value) for value in row[:4])
        x = max(0, min(width - 1, int(np.floor((center_x - box_width / 2 - pad_x) / scale))))
        y = max(0, min(height - 1, int(np.floor((center_y - box_height / 2 - pad_y) / scale))))
        right = max(0, min(width, int(np.ceil((center_x + box_width / 2 - pad_x) / scale))))
        bottom = max(0, min(height, int(np.ceil((center_y + box_height / 2 - pad_y) / scale))))
        w, h = right - x, bottom - y
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, w, h])
        scores.append(score)
        class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.45)
    results = []
    for index in np.array(indices).reshape(-1) if len(indices) else []:
        x, y, w, h = boxes[int(index)]
        class_id = class_ids[int(index)]
        labels = class_names if class_names is not None else COCO_CLASSES
        results.append({"label": labels[class_id] if class_id < len(labels) else "class_{}".format(class_id), "confidence": scores[int(index)], "x": x, "y": y, "w": w, "h": h})
    return results


def infer_video_frames(
    net: cv2.dnn_Net,
    video_path: str,
    every_n_frames: int = 10,
    confidence: float = 0.35,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Run the same detector on sampled video frames and return actual frame results."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError("Could not open video: {}".format(video_path))
    results = []
    stride = max(1, int(every_n_frames))
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                results.append({"frame": frame_index, "detections": infer_image(net, frame, confidence=confidence, class_names=class_names)})
            frame_index += 1
    finally:
        capture.release()
    return results


def annotate_video(net: cv2.dnn_Net, video_path: str, output_path: str, confidence: float = 0.35) -> Dict[str, object]:
    """Run actual model detections on every frame and write a checked MP4 result."""
    source = cv2.VideoCapture(str(video_path))
    if not source.isOpened():
        raise OSError("Could not open video: {}".format(video_path))
    fps = source.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        source.release()
        raise OSError("Could not create annotated output video: {}".format(target))
    frames = 0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            detections = infer_image(net, frame, confidence=confidence)
            writer.write(draw_detections(frame, detections))
            frames += 1
    finally:
        source.release()
        writer.release()
    check = cv2.VideoCapture(str(target))
    readable = check.isOpened() and check.read()[0]
    check.release()
    if not readable:
        raise OSError("Annotated output could not be reopened: {}".format(target))
    return {"fps": fps, "frames_processed": frames, "width": width, "height": height, "output_path": str(target)}


def count_objects(detections: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for detection in detections:
        label = str(detection["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def crop_detection(image: np.ndarray, detection: Dict[str, object]) -> np.ndarray:
    x, y, width, height = (int(detection[key]) for key in ("x", "y", "w", "h"))
    crop = image[y:y + height, x:x + width]
    if crop.size == 0:
        raise ValueError("Detection box has no image pixels")
    return crop.copy()


def draw_detections(image: np.ndarray, detections: List[Dict[str, object]]) -> np.ndarray:
    output = image.copy()
    for detection in detections:
        x, y, width, height = (int(detection[key]) for key in ("x", "y", "w", "h"))
        cv2.rectangle(output, (x, y), (x + width, y + height), (40, 190, 240), 2)
        cv2.putText(output, "{} {:.2f}".format(detection["label"], detection["confidence"]), (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 190, 240), 1, cv2.LINE_AA)
    return output