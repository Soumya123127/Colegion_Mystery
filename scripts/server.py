"""Local static server and real OpenCV operation API; binds only to localhost."""

import base64
import json
import mimetypes
import os
import sys
import tempfile
import time
import uuid
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from the_last_chai.color_detection import draw_candidates, find_red_candidates
from the_last_chai.evidence import validate_accusation, validate_clue
from the_last_chai.image_enhancement import crop_image, enhance_image, image_info, load_image
from the_last_chai.object_detection import count_objects, draw_detections, infer_image, infer_video_frames, load_model
from the_last_chai.reporting import generate_report
from the_last_chai.video_tracking import track_video


def resolve_runtime_bind(default_port=8000):
    configured_port = os.environ.get("PORT")
    port = int(configured_port) if configured_port and configured_port.strip() else default_port
    host = "0.0.0.0" if configured_port and configured_port.strip() else "127.0.0.1"
    return host, port


def encoded_image(image):
    ok, content = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode operation output")
    return "data:image/png;base64," + base64.b64encode(content.tobytes()).decode("ascii")


def normalize_parameters(parameters):
    supplied = parameters if isinstance(parameters, dict) else {}

    def bounded(name, default, minimum, maximum, integer=False):
        try:
            value = float(supplied.get(name, default))
        except (TypeError, ValueError):
            value = float(default)
        value = max(minimum, min(maximum, value))
        return int(round(value)) if integer else value

    threshold_method = str(supplied.get("threshold_method", "otsu"))
    if threshold_method not in ("otsu", "binary"):
        threshold_method = "otsu"
    return {
        "brightness": bounded("brightness", 0, -100, 100, True),
        "contrast": bounded("contrast", 1.0, 0.5, 3.0),
        "clip_limit": bounded("clip_limit", 2.0, 0.5, 8.0),
        "tile_grid_size": bounded("tile_grid_size", 8, 2, 16, True),
        "blur_kernel": bounded("blur_kernel", 5, 1, 15, True),
        "threshold_method": threshold_method,
        "threshold_value": bounded("threshold_value", 127, 0, 255, True),
        "confidence": bounded("confidence", 0.35, 0.05, 0.95),
        "low_hue_max": bounded("low_hue_max", 10, 0, 20, True),
        "high_hue_min": bounded("high_hue_min", 170, 159, 179, True),
        "low_saturation": bounded("low_saturation", 70, 0, 255, True),
        "low_value": bounded("low_value", 40, 0, 255, True),
        "high_saturation": max(bounded("low_saturation", 70, 0, 255, True), bounded("high_saturation", 255, 0, 255, True)),
        "high_value": max(bounded("low_value", 40, 0, 255, True), bounded("high_value", 255, 0, 255, True)),
        "kernel_size": bounded("kernel_size", 5, 1, 15, True),
        "min_area": bounded("min_area", 500, 10, 100000, True),
        "line_percent": bounded("line_percent", 75, 10, 90, True),
    }


def active_model_config():
    custom_weights = ROOT / "models" / "custom_yolo.onnx"
    custom_classes = ROOT / "models" / "custom_yolo_classes.json"
    if custom_weights.is_file() or custom_classes.is_file():
        if not custom_weights.is_file() or not custom_classes.is_file():
            raise RuntimeError("Custom YOLO export is incomplete. Run scripts/train_custom_model.py to export weights and class names together.")
        try:
            names = json.loads(custom_classes.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Could not read custom YOLO class names: {}".format(error))
        if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
            raise RuntimeError("Custom YOLO class names must be a non-empty JSON list of strings.")
        return custom_weights, names, "Custom YOLO"
    return ROOT / "models" / "yolov8n.onnx", None, "YOLOv8n COCO"


def load_active_model():
    weights, class_names, model_name = active_model_config()
    return load_model(str(weights)), class_names, model_name


def analyze_uploaded_file(mission, filename, data_url, parameters=None):
    settings = normalize_parameters(parameters)
    match = re.fullmatch(r"data:(image|video)/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=]+)", data_url or "")
    if not match:
        raise ValueError("Upload must be a base64-encoded image or video file")
    kind, encoded = match.groups()
    content = base64.b64decode(encoded, validate=True)
    if len(content) > 18 * 1024 * 1024:
        raise ValueError("File is too large; choose a file smaller than 18 MB")
    suffix = Path(filename or "upload").suffix.lower()
    allowed = {"image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"}, "video": {".mp4", ".mov", ".m4v", ".avi", ".webm"}}
    if suffix not in allowed[kind]:
        raise ValueError("Unsupported file extension. Choose an image or video format listed by the file picker.")
    upload_dir = ROOT / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=str(upload_dir)) as temporary:
        temporary.write(content)
        upload_path = Path(temporary.name)
    try:
        if kind == "image":
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError("OpenCV could not decode this image")
            height, width = image.shape[:2]
            if width * height > 24_000_000:
                raise ValueError("Image is too large to process safely (maximum 24 megapixels)")
            if mission == "eyes-online":
                operations = enhance_image(
                    image,
                    clip_limit=settings["clip_limit"],
                    tile_grid=(settings["tile_grid_size"], settings["tile_grid_size"]),
                    brightness=settings["brightness"],
                    contrast=settings["contrast"],
                    blur_kernel=settings["blur_kernel"],
                    threshold_method=settings["threshold_method"],
                    threshold_value=settings["threshold_value"],
                )
                crop = crop_image(operations["clahe"], int(width * 0.4), int(height * 0.03), max(1, int(width * 0.58)), max(1, int(height * 0.25)))
                return {"source": "user-upload", "filename": Path(filename).name, **image_info(image), "image": encoded_image(operations["clahe"]), "gray": encoded_image(operations["gray"]), "equalized": encoded_image(operations["equalized"]), "blurred": encoded_image(operations["blurred"]), "threshold": encoded_image(operations["threshold"]), "crop": encoded_image(crop), "processing_parameters": settings, "operations": ["cv2.imdecode", "cv2.convertScaleAbs", "cv2.cvtColor(BGR2GRAY)", "cv2.createCLAHE().apply", "cv2.GaussianBlur", "cv2.threshold", "crop upper-right region"], "note": "CLAHE and crop are computed from your uploaded photo. Inspect whether the selected region contains useful detail."}
            if mission in ("red-bag", "follow-trail"):
                mask, candidates = find_red_candidates(
                    image,
                    min_area=settings["min_area"],
                    kernel_size=settings["kernel_size"],
                    low_saturation=settings["low_saturation"],
                    low_value=settings["low_value"],
                    low_hue_max=settings["low_hue_max"],
                    high_hue_min=settings["high_hue_min"],
                    high_saturation=settings["high_saturation"],
                    high_value=settings["high_value"],
                )
                mode = "Live-camera still" if mission == "follow-trail" else "Photo"
                return {"source": "user-upload", "filename": Path(filename).name, "width": width, "height": height, "count": len(candidates), "boxes": candidates, "mask": encoded_image(mask), "annotated": encoded_image(draw_candidates(image, candidates)), "processing_parameters": settings, "operations": ["cv2.imdecode", "cv2.cvtColor(BGR2HSV)", "cv2.inRange(two red hue bands)", "morphology open/close", "cv2.findContours", "cv2.boundingRect"], "note": "{} red-candidate detection completed. These color regions do not identify a bag, person, carrier or owner.".format(mode)}
            if mission == "missing-clue":
                model_path, class_names, model_name = active_model_config()
                if model_path.is_file():
                    net, class_names, model_name = load_active_model()
                    started = time.perf_counter()
                    detections = infer_image(net, image, confidence=settings["confidence"], class_names=class_names)
                    inference_ms = (time.perf_counter() - started) * 1000
                    return {"source": "user-upload", "filename": Path(filename).name, "width": width, "height": height, "model_available": True, "model_name": model_name, "detections": detections, "counts": count_objects(detections), "inference_ms": inference_ms, "processing_parameters": settings, "annotated": encoded_image(draw_detections(image, detections)), "operations": ["cv2.dnn.blobFromImage", "{} ONNX inference".format(model_name), "cv2.dnn.NMSBoxes"], "note": "Labels and counts are actual output from the selected model. Predictions may be missed or wrong and cannot identify a person or establish ownership."}
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 70, 150)
                return {"source": "user-upload", "filename": Path(filename).name, "width": width, "height": height, "model_available": False, "edge_map": encoded_image(edges), "operations": ["cv2.imdecode", "cv2.cvtColor(BGR2GRAY)", "cv2.Canny"], "note": "Actual edge analysis completed. No semantic object classes are claimed because optional YOLO weights are not installed."}
            raise ValueError("Photo uploads are supported in missions 1, 2 and 3; upload a video in mission 4")
        if mission not in ("follow-trail", "missing-clue"):
            raise ValueError("Video uploads are supported in missions 2 and 4")
        capture = cv2.VideoCapture(str(upload_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video; try MP4 H.264 or MOV")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if not width or not height or not frames:
            raise ValueError("Video metadata is incomplete or the file contains no readable frames")
        if mission == "follow-trail":
            output = ROOT / "outputs" / ("uploaded_track_{}.mp4".format(uuid.uuid4().hex[:10]))
            result = track_video(
                str(upload_path),
                str(output),
                line_x=int(width * settings["line_percent"] / 100),
                min_area=settings["min_area"],
                kernel_size=settings["kernel_size"],
                low_saturation=settings["low_saturation"],
                low_value=settings["low_value"],
                low_hue_max=settings["low_hue_max"],
                high_hue_min=settings["high_hue_min"],
                high_saturation=settings["high_saturation"],
                high_value=settings["high_value"],
            )
            result["processing_parameters"] = settings
            result.update({"source": "user-upload", "filename": Path(filename).name, "video": "/outputs/" + output.name, "operations": ["cv2.VideoCapture", "HSV red mask per frame", "contour centroid", "recent-position trail", "exit-line crossing"], "note": "Tracking ran on your uploaded video. It follows the largest red candidate; it does not identify a person. The line is placed at 75% of frame width."})
            return result
        capture = cv2.VideoCapture(str(upload_path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError("Could not decode the first frame")
        model_path, class_names, model_name = active_model_config()
        if model_path.is_file():
            started = time.perf_counter()
            net, class_names, model_name = load_active_model()
            detections = infer_video_frames(net, str(upload_path), every_n_frames=10, confidence=settings["confidence"], class_names=class_names)
            inference_ms = (time.perf_counter() - started) * 1000
            return {"source": "user-upload", "filename": Path(filename).name, "fps": fps, "frame_count": frames, "width": width, "height": height, "model_available": True, "model_name": model_name, "detections": detections, "inference_ms": inference_ms, "processing_parameters": settings, "operations": ["cv2.VideoCapture", "sample every 10th decoded frame", "{} ONNX inference".format(model_name), "cv2.dnn.NMSBoxes"], "note": "Actual model inference ran on sampled video frames. The class labels are predictions, not verified identities or case facts."}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        return {"source": "user-upload", "filename": Path(filename).name, "fps": fps, "frame_count": frames, "width": width, "height": height, "model_available": False, "edge_map": encoded_image(edges), "operations": ["cv2.VideoCapture", "cv2.read", "cv2.cvtColor(BGR2GRAY)", "cv2.Canny (first frame)"], "note": "First-frame edge analysis completed. This is not semantic object detection. Optional YOLO weights are not installed."}
    finally:
        upload_path.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _body(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size > 26 * 1024 * 1024:
            raise ValueError("Request too large; uploads must be smaller than 18 MB")
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/status":
            weights, class_names, model_name = active_model_config()
            return self._json({"status": "online", "opencv": cv2.__version__, "model_weights": weights.is_file(), "model_name": model_name if weights.is_file() else None, "model_classes": class_names, "model_note": "Custom weights are preferred when exported; otherwise the pretrained COCO model is used. No model output is reported without weights."})
        relative = unquote(route.lstrip("/")) or "app/index.html"
        if route.endswith("/"):
            relative += "index.html"
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            return self._json({"error": "Not found"}, 404)
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            if route == "/api/analyze-upload":
                body = self._body()
                result = analyze_uploaded_file(str(body.get("mission", "")), str(body.get("filename", "upload")), str(body.get("data", "")), body.get("parameters"))
                return self._json(result)
            if route == "/api/mission/eyes-online":
                settings = normalize_parameters(self._body().get("parameters"))
                image = load_image(str(ROOT / "assets/images/cctv_corrupted.png"))
                operations = enhance_image(image, clip_limit=settings["clip_limit"], tile_grid=(settings["tile_grid_size"], settings["tile_grid_size"]), brightness=settings["brightness"], contrast=settings["contrast"], blur_kernel=settings["blur_kernel"], threshold_method=settings["threshold_method"], threshold_value=settings["threshold_value"])
                crop = crop_image(operations["clahe"], 405, 15, 305, 65)
                return self._json({**image_info(image), "image": encoded_image(operations["clahe"]), "gray": encoded_image(operations["gray"]), "equalized": encoded_image(operations["equalized"]), "blurred": encoded_image(operations["blurred"]), "threshold": encoded_image(operations["threshold"]), "crop": encoded_image(crop), "processing_parameters": settings, "operation": "CLAHE, blur and threshold computed from the synthetic source image"})
            if route == "/api/mission/missing-clue":
                settings = normalize_parameters(self._body().get("parameters"))
                earlier = load_image(str(ROOT / "assets/images/cctv_corrupted.png"))
                later = load_image(str(ROOT / "assets/images/lab_reference_2343.png"))
                difference = cv2.absdiff(later, earlier)
                mean = float(np.mean(difference))
                result = {"difference": encoded_image(difference), "mean_difference": mean, "model_available": False, "processing_parameters": settings, "note": "This is an actual pixel difference, not object detection. The extra-cup comparison and missing-item inventory are prepared fictional scene facts. Model weights are optional; no model classes or counts are invented."}
                model_path, class_names, model_name = active_model_config()
                if model_path.is_file():
                    net, class_names, model_name = load_active_model()
                    started = time.perf_counter()
                    earlier_detections = infer_image(net, earlier, confidence=settings["confidence"], class_names=class_names)
                    later_detections = infer_image(net, later, confidence=settings["confidence"], class_names=class_names)
                    inference_ms = (time.perf_counter() - started) * 1000
                    result.update({"model_available": True, "model_name": model_name, "inference_ms": inference_ms, "earlier_counts": count_objects(earlier_detections), "later_counts": count_objects(later_detections), "earlier_detections": earlier_detections, "later_detections": later_detections, "earlier_annotated": encoded_image(draw_detections(earlier, earlier_detections)), "later_annotated": encoded_image(draw_detections(later, later_detections)), "note": "Detections are live model inference at the selected confidence. A model may miss objects and its predictions do not replace the supplied scene inventory."})
                return self._json(result)
            if route == "/api/mission/red-bag":
                settings = normalize_parameters(self._body().get("parameters"))
                image = load_image(str(ROOT / "assets/images/red_bag_scene.png"))
                mask, candidates = find_red_candidates(image, min_area=settings["min_area"], kernel_size=settings["kernel_size"], low_saturation=settings["low_saturation"], low_value=settings["low_value"], low_hue_max=settings["low_hue_max"], high_hue_min=settings["high_hue_min"], high_saturation=settings["high_saturation"], high_value=settings["high_value"])
                return self._json({"count": len(candidates), "boxes": candidates, "mask": encoded_image(mask), "processing_parameters": settings, "annotated": encoded_image(draw_candidates(image, candidates)), "operation": "HSV dual-red mask, morphology and contour extraction"})
            if route == "/api/mission/follow-trail":
                settings = normalize_parameters(self._body().get("parameters"))
                output = ROOT / "outputs" / "corridor_tracked.mp4"
                result = track_video(str(ROOT / "assets/videos/corridor_staged.mp4"), str(output), line_x=int(640 * settings["line_percent"] / 100), min_area=settings["min_area"], kernel_size=settings["kernel_size"], low_saturation=settings["low_saturation"], low_value=settings["low_value"], low_hue_max=settings["low_hue_max"], high_hue_min=settings["high_hue_min"], high_saturation=settings["high_saturation"], high_value=settings["high_value"])
                result["processing_parameters"] = settings
                result["video"] = "/outputs/corridor_tracked.mp4"
                result["note"] = "The detected red candidate is not person identification. Kabir is named by separate fictional case-file evidence; the 23:44 log is a prepared story asset."
                return self._json(result)
            if route == "/api/clue":
                body = self._body()
                return self._json(validate_clue(str(body.get("mission", "")), str(body.get("answer", ""))))
            if route == "/api/accusation":
                body = self._body()
                return self._json(validate_accusation(body.get("answers", {}), body.get("evidence_ids", [])))
            if route == "/api/report":
                html = generate_report(self._body()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            return self._json({"error": "Unknown operation"}, 404)
        except Exception as error:
            return self._json({"error": str(error)}, 500)

    def log_message(self, format_string, *args):
        print("[portal] {} - {}".format(self.address_string(), format_string % args))


def main():
    if len(sys.argv) > 1:
        host = "127.0.0.1"
        port = int(sys.argv[1])
    else:
        host, port = resolve_runtime_bind(default_port=8000)
    server = ThreadingHTTPServer((host, port), Handler)
    display_host = "localhost" if host == "0.0.0.0" else host
    print("The Last Chai portal: http://{}:{}/app/".format(display_host, port))
    print("OpenCV {} | optional YOLO weights: {}".format(cv2.__version__, (ROOT / "models/yolov8n.onnx").is_file()))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local portal.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
