import json
import os
from pathlib import Path
import sys
import types
import base64

import cv2
import numpy as np
import pytest

from the_last_chai.color_detection import clean_mask, find_red_candidates, red_mask
from the_last_chai.evidence import validate_accusation, validate_clue
from the_last_chai.image_enhancement import adjust_brightness_contrast, crop_image, enhance_image, image_info, load_image, save_image
from the_last_chai.object_detection import COCO_CLASSES, count_objects, crop_detection, infer_image, infer_video_frames, load_model, validate_detections
from the_last_chai.reporting import generate_report
from the_last_chai.video_tracking import line_crossed, track_video
from scripts import server as portal_server
from scripts.server import analyze_uploaded_file, normalize_parameters

ROOT = Path(__file__).resolve().parents[1]


def test_level1_image_operations(tmp_path):
    image = load_image(str(ROOT / "assets/images/cctv_corrupted.png"))
    assert image_info(image) == {"width": 720, "height": 420, "channels": 3}
    outputs = enhance_image(image)
    assert outputs["clahe"].shape == (420, 720)
    assert adjust_brightness_contrast(image, beta=30).shape == image.shape
    assert crop_image(outputs["clahe"], 405, 15, 305, 65).size > 0
    target = save_image(str(tmp_path / "enhanced.png"), outputs["clahe"])
    assert Path(target).is_file()
    assert validate_clue("eyes-online", "23:42")["correct"]


def test_level2_detection_contract_and_optional_weights():
    detections = [{"label": "cup", "confidence": 0.8, "x": 4, "y": 5, "w": 20, "h": 18}]
    assert count_objects(detections) == {"cup": 1}
    assert validate_detections(detections, 40, 40)
    assert not validate_detections([{**detections[0], "x": 30}], 40, 40)
    assert crop_detection(np.zeros((40, 40, 3), np.uint8), detections[0]).shape == (18, 20, 3)
    with pytest.raises(FileNotFoundError, match="weights are missing"):
        load_model(str(ROOT / "models/not-downloaded.onnx"))


def test_custom_yolo_class_count_uses_yolov8_output_layout():
    class FakeNetwork:
        def __init__(self, class_count, target_class):
            self.class_count = class_count
            self.target_class = target_class

        def setInput(self, _blob):
            pass

        def getUnconnectedOutLayersNames(self):
            return ["output"]

        def forward(self, output_names):
            output = np.zeros((1, 4 + self.class_count, 10), dtype=np.float32)
            output[0, :4, 0] = [320.0, 320.0, 160.0, 160.0]
            output[0, 4 + self.target_class, 0] = 0.94
            return output

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    for class_names, target_class in ((["person", "backpack", "cup", "prototype", "phone"], 3), (["prototype"], 0)):
        detections = infer_image(FakeNetwork(len(class_names), target_class), image, confidence=0.5, class_names=class_names)
        assert len(detections) == 1
        assert detections[0]["label"] == class_names[target_class]
    landscape = np.zeros((320, 640, 3), dtype=np.uint8)
    mapped = infer_image(FakeNetwork(1, 0), landscape, confidence=0.5, class_names=["prototype"])
    assert (mapped[0]["x"], mapped[0]["y"], mapped[0]["w"], mapped[0]["h"]) == (240, 80, 160, 160)


def test_custom_model_configuration_uses_exported_class_names(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "custom_yolo.onnx").write_bytes(b"test weights")
    classes_file = model_dir / "custom_yolo_classes.json"
    classes_file.write_text('["prototype"]', encoding="utf-8")
    monkeypatch.setattr(portal_server, "ROOT", tmp_path)
    weights, names, model_name = portal_server.active_model_config()
    assert weights.name == "custom_yolo.onnx"
    assert names == ["prototype"]
    assert model_name == "Custom YOLO"

    classes_file.unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        portal_server.active_model_config()


def test_custom_class_names_reach_uploaded_yolo_results(monkeypatch):
    class_names = list(COCO_CLASSES)
    class_names[COCO_CLASSES.index("cell phone")] = "workshop_phone"
    weights = ROOT / "models" / "yolov8n.onnx"
    monkeypatch.setattr(portal_server, "active_model_config", lambda: (weights, class_names, "Custom YOLO"))
    image = load_image(str(ROOT / "assets/images/real_smartphone_use.jpg"))
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    result = analyze_uploaded_file("missing-clue", "phone.jpg", data_url, {"confidence": 0.2})
    assert result["model_name"] == "Custom YOLO"
    assert "workshop_phone" in result["counts"]


def test_pretrained_onnx_inference_on_real_coco_class_photos():
    weights = ROOT / "models/yolov8n.onnx"
    if not weights.is_file():
        pytest.skip("Run python scripts/setup_model.py to enable pretrained inference test")
    net = load_model(str(weights))
    backpack_scene = load_image(str(ROOT / "assets/images/real_red_backpack_person.jpg"))
    phone_scene = load_image(str(ROOT / "assets/images/real_smartphone_use.jpg"))
    backpack_detections = infer_image(net, backpack_scene, confidence=0.2)
    custom_labels = infer_image(net, backpack_scene, confidence=0.2, class_names=["custom_person"] + COCO_CLASSES[1:])
    phone_detections = infer_image(net, phone_scene, confidence=0.2)
    assert "person" in count_objects(backpack_detections)
    assert "custom_person" in count_objects(custom_labels)
    assert "cell phone" in count_objects(phone_detections)
    assert len(infer_image(net, phone_scene, confidence=0.95)) <= len(phone_detections)
    video_results = infer_video_frames(net, str(ROOT / "assets/videos/corridor_staged.mp4"), every_n_frames=35, confidence=0.3)
    assert [result["frame"] for result in video_results] == [0, 35]


def test_level3_red_detection_including_hue_wrap():
    image = np.zeros((140, 180, 3), np.uint8)
    image[20:110, 25:95] = (0, 0, 255)
    image[35:95, 120:165] = (10, 10, 180)
    mask = red_mask(image)
    assert mask.shape == image.shape[:2]
    assert int(mask[50, 50]) == 255 and int(mask[50, 140]) == 255
    assert clean_mask(mask).shape == mask.shape
    _, candidates = find_red_candidates(image, min_area=300)
    assert candidates and candidates[0]["x"] <= 25 < candidates[0]["x"] + candidates[0]["w"]
    assert candidates[0]["w"] > 0 and candidates[0]["h"] > 0


def test_level4_actual_tracking_and_line_crossing(tmp_path):
    result = track_video(str(ROOT / "assets/videos/corridor_staged.mp4"), str(tmp_path / "tracked.mp4"), line_x=480)
    assert result["frames_processed"] == result["frame_count"] == 70
    assert result["fps"] == 10
    assert result["valid_detection_frames"] > 0
    assert result["crossing"] is not None
    assert 50 <= result["crossing"]["frame"] <= 62
    capture = cv2.VideoCapture(str(tmp_path / "tracked.mp4"))
    assert capture.isOpened() and capture.read()[0]
    capture.release()
    assert line_crossed((400, 100), (481, 100), 480)
    assert not line_crossed((400, 100), None, 480)
    with pytest.raises(OSError, match="Could not open video"):
        track_video(str(tmp_path / "missing.mp4"))


def test_tracking_challenge_clips_have_known_crossings():
    cases = [("corridor_fast.mp4", 52), ("corridor_occlusion.mp4", 83)]
    for filename, expected_frame in cases:
        result = track_video(str(ROOT / "assets/videos" / filename), line_x=480, min_area=700)
        assert result["crossing"] is not None
        assert abs(result["crossing"]["frame"] - expected_frame) <= 1
        if filename == "corridor_occlusion.mp4":
            assert result["valid_detection_frames"] < result["frames_processed"]


def test_case_report_and_accusation():
    state = {"completed": ["eyes-online"], "results": {"eyes-online": {"timestamp": "23:42", "image": "data:image/png;base64,QUJD"}}}
    report = generate_report(state)
    assert "No final accusation was submitted" in report
    assert "<img src=\"data:image/png;base64,QUJD\"" in report
    assert "23:44" in report and "data:image/png;base64,QUJD</pre>" not in report
    answer = {"culprit": "Kabir", "owner": "Aarav", "carrier": "Kabir", "incident": "23:42", "corruption": "23:44"}
    assert not validate_accusation(answer, ["ev-time"])["correct"]
    assert validate_accusation(answer, ["ev-time", "ev-scene", "ev-owner", "ev-trail", "ev-log"])["correct"]


def test_manifest_paths_exist():
    manifest = json.loads((ROOT / "data/manifests/evidence_registry.json").read_text())
    for item in manifest["evidence"]:
        for source in item["source"].split("; "):
            if source.startswith(("assets/", "data/")):
                assert (ROOT / source).is_file(), source
    dataset = json.loads((ROOT / "data/manifests/dataset_manifest.json").read_text())
    assert dataset["fictional_simulation"] is True
    assert all((ROOT / entry["filename"]).is_file() for entry in dataset["files"])
    assert all(entry["purpose"] and entry["creation_method"] and entry["license"] for entry in dataset["files"])


def test_notebook_code_cells_execute():
    import nbformat

    class PlotAxis:
        def imshow(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def axis(self, *args, **kwargs):
            pass

    class PlotShim(types.ModuleType):
        def figure(self, *args, **kwargs):
            return PlotAxis()

        def subplots(self, rows, columns, **kwargs):
            return PlotAxis(), [PlotAxis() for _ in range(columns)]

        def subplot(self, *args, **kwargs):
            return PlotAxis()

        def title(self, *args, **kwargs):
            pass

        def imshow(self, *args, **kwargs):
            pass

        def axis(self, *args, **kwargs):
            pass

        def tight_layout(self, *args, **kwargs):
            pass

    matplotlib = types.ModuleType("matplotlib")
    pyplot = PlotShim("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    original_matplotlib = sys.modules.get("matplotlib")
    original_pyplot = sys.modules.get("matplotlib.pyplot")
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        for notebook_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(notebook_path, as_version=4)
            namespace = {"__name__": "__main__"}
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    exec(compile(cell.source, str(notebook_path), "exec"), namespace)
    finally:
        os.chdir(previous)
        if original_matplotlib is None:
            sys.modules.pop("matplotlib", None)
        else:
            sys.modules["matplotlib"] = original_matplotlib
        if original_pyplot is None:
            sys.modules.pop("matplotlib.pyplot", None)
        else:
            sys.modules["matplotlib.pyplot"] = original_pyplot


def test_uploaded_image_runs_real_opencv_operations():
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    image[25:100, 35:115] = (0, 0, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    data_url = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    result = analyze_uploaded_file("red-bag", "student-evidence.png", data_url)
    assert result["source"] == "user-upload"
    assert result["count"] == 1
    assert result["boxes"][0]["x"] <= 35 < result["boxes"][0]["x"] + result["boxes"][0]["w"]
    assert result["annotated"].startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="base64-encoded"):
        analyze_uploaded_file("red-bag", "bad.png", "not-an-image")


def test_participant_parameters_change_real_processing_outputs():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
    image = cv2.merge([ramp, ramp, ramp])
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    data_url = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    normal = analyze_uploaded_file("eyes-online", "ramp.png", data_url)
    bright = analyze_uploaded_file("eyes-online", "ramp.png", data_url, {"brightness": 80})
    normal_pixels = cv2.imdecode(np.frombuffer(base64.b64decode(normal["image"].split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    bright_pixels = cv2.imdecode(np.frombuffer(base64.b64decode(bright["image"].split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert not np.array_equal(normal_pixels, bright_pixels)
    assert bright["processing_parameters"]["brightness"] == 80

    red = np.zeros((120, 120, 3), dtype=np.uint8)
    red[10:110, 10:110] = (0, 43, 255)
    ok, encoded = cv2.imencode(".png", red)
    assert ok
    red_url = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    detected = analyze_uploaded_file("red-bag", "red.png", red_url)
    suppressed = analyze_uploaded_file("red-bag", "red.png", red_url, {"low_hue_max": 0})
    upper_limited = analyze_uploaded_file("red-bag", "red.png", red_url, {"high_value": 200})
    assert detected["count"] == 1
    assert suppressed["count"] == 0
    assert upper_limited["count"] == 0


def test_parameter_bounds_and_method_validation():
    settings = normalize_parameters({"brightness": 1000, "contrast": -4, "kernel_size": 8, "threshold_method": "unexpected", "low_value": 200, "high_value": 100})
    assert settings["brightness"] == 100
    assert settings["contrast"] == 0.5
    assert settings["kernel_size"] == 8
    assert settings["threshold_method"] == "otsu"
    assert settings["high_value"] == 200


def test_uploaded_video_is_tracked_frame_by_frame():
    video_path = ROOT / "assets/videos/corridor_staged.mp4"
    data_url = "data:video/mp4;base64," + base64.b64encode(video_path.read_bytes()).decode("ascii")
    result = analyze_uploaded_file("follow-trail", "student-clip.mp4", data_url)
    assert result["source"] == "user-upload"
    assert result["filename"] == "student-clip.mp4"
    assert result["frames_processed"] == result["frame_count"] == 70
    assert result["crossing"] is not None
    assert Path(result["output_path"]).is_file()
    Path(result["output_path"]).unlink()