"""Generate original, deterministic synthetic workshop evidence with OpenCV."""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "assets" / "images"
VIDEOS = ROOT / "assets" / "videos"
REFERENCES = ROOT / "assets" / "references"


def room_scene(later=False):
    image = np.full((420, 720, 3), (30, 34, 38), dtype=np.uint8)
    cv2.rectangle(image, (0, 310), (720, 420), (43, 45, 44), -1)
    cv2.rectangle(image, (65, 56), (240, 178), (74, 77, 77), 2)
    cv2.putText(image, "ARGUS LAB", (91, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (157, 161, 157), 2)
    cv2.rectangle(image, (320, 238), (670, 255), (74, 65, 53), -1)
    cv2.rectangle(image, (340, 255), (351, 350), (57, 51, 45), -1)
    cv2.rectangle(image, (640, 255), (651, 350), (57, 51, 45), -1)
    if not later:
        cv2.rectangle(image, (355, 190), (405, 229), (76, 87, 83), -1)
        cv2.rectangle(image, (355, 190), (405, 229), (145, 151, 143), 2)
        cv2.rectangle(image, (363, 198), (380, 214), (35, 90, 91), -1)
        cv2.rectangle(image, (385, 198), (397, 206), (187, 144, 66), -1)
        cv2.putText(image, "ARGUS", (360, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (232, 224, 196), 1)
    # A deliberately low-contrast, still-readable overlay survives enhancement.
    cv2.rectangle(image, (420, 28), (696, 67), (44, 47, 48), -1)
    cv2.putText(image, "23:42  |  CAM 04", (433, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.77, (101, 105, 104), 2, cv2.LINE_AA)
    cv2.ellipse(image, (450, 215), (28, 9), 0, 0, 360, (110, 112, 109), -1)
    cv2.rectangle(image, (430, 184), (470, 215), (165, 165, 157), -1)
    cv2.ellipse(image, (450, 184), (20, 6), 0, 0, 360, (190, 189, 179), -1)
    cv2.rectangle(image, (500, 187), (552, 236), (0, 0, 205), -1)
    cv2.rectangle(image, (500, 187), (552, 236), (16, 16, 20), 2)
    cv2.ellipse(image, (526, 187), (26, 8), 0, 0, 360, (20, 20, 24), 3)
    if not later:
        cv2.rectangle(image, (567, 196), (592, 219), (75, 82, 85), -1)
        cv2.rectangle(image, (568, 197), (591, 218), (20, 24, 26), -1)
        cv2.rectangle(image, (571, 200), (588, 215), (102, 126, 132), -1)
        cv2.putText(image, "PHONE", (567, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (226, 224, 210), 1)
    cv2.rectangle(image, (571, 200), (588, 215), (102, 126, 132), -1)
    cv2.rectangle(image, (615, 208), (650, 239), (80, 80, 34), -1)
    cv2.rectangle(image, (615, 208), (650, 239), (103, 105, 95), 2)
    # Fictional lab staff silhouettes provide scene context, not facial evidence.
    cv2.circle(image, (275, 186), 22, (116, 108, 97), -1)
    cv2.rectangle(image, (250, 207), (300, 292), (100, 107, 105), -1)
    cv2.circle(image, (205, 235), 17, (112, 106, 98), -1)
    cv2.rectangle(image, (185, 252), (225, 310), (79, 88, 87), -1)
    if later:
        cv2.rectangle(image, (568, 193), (592, 220), (165, 165, 157), -1)
        cv2.ellipse(image, (580, 193), (12, 4), 0, 0, 360, (190, 189, 179), -1)
        cv2.ellipse(image, (580, 220), (12, 4), 0, 0, 360, (42, 43, 41), 2)
        cv2.putText(image, "REFERENCE SCENE 23:43", (22, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (225, 220, 207), 1)
    else:
        cv2.putText(image, "STAGED CCTV STILL 23:42", (22, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (225, 220, 207), 1)
    # Sensor noise and a mild dark veil simulate recoverable, not destroyed, footage.
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 2.2, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    veil = np.full_like(image, 0)
    image = cv2.addWeighted(image, 0.74, veil, 0.26, 0)
    return image


def portrait(name, initials, accent):
    canvas = np.full((320, 240, 3), (30, 34, 38), np.uint8)
    cv2.circle(canvas, (120, 118), 58, accent, -1)
    cv2.ellipse(canvas, (120, 286), (88, 116), 0, 180, 360, (67, 70, 69), -1)
    cv2.putText(canvas, initials, (75, 137), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (245, 241, 226), 3)
    cv2.putText(canvas, name.upper(), (27, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 226, 212), 1)
    return canvas


def write_corridor(path, speed, frame_total, line_x, hide_range=None, lighting_change=False, distraction=False):
    width, height, fps = 640, 360, 10
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4V video writer: {}".format(path))
    for frame_index in range(frame_total):
        frame = np.full((height, width, 3), (36, 39, 40), dtype=np.uint8)
        cv2.rectangle(frame, (0, 260), (width, height), (50, 51, 48), -1)
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 155, 210), 3)
        cv2.putText(frame, "EXIT LINE", (line_x - 88, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 192, 120), 2)
        cv2.putText(frame, "STAGED CORRIDOR | 23:43", (18, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (214, 213, 201), 1)
        if distraction:
            cv2.circle(frame, (380, 205), 9, (0, 0, 180), -1)
        bag_x = 30 + frame_index * speed
        hidden = hide_range is not None and hide_range[0] <= frame_index <= hide_range[1]
        if not hidden and bag_x + 78 < width:
            cv2.rectangle(frame, (bag_x, 145), (bag_x + 78, 245), (0, 0, 210), -1)
            cv2.rectangle(frame, (bag_x, 145), (bag_x + 78, 245), (12, 12, 18), 3)
            cv2.rectangle(frame, (bag_x + 18, 129), (bag_x + 60, 148), (0, 0, 175), -1)
            cv2.ellipse(frame, (bag_x + 40, 145), (20, 13), 0, 180, 360, (15, 15, 22), 3)
            cv2.circle(frame, (bag_x + 40, 93), 23, (136, 127, 116), -1)
            cv2.rectangle(frame, (bag_x + 14, 111), (bag_x + 66, 166), (101, 105, 103), -1)
        if lighting_change and 35 <= frame_index <= 55:
            frame = cv2.convertScaleAbs(frame, alpha=0.42, beta=-8)
        writer.write(frame)
    writer.release()


def image_sample_scenes(original):
    low_light = cv2.convertScaleAbs(original, alpha=0.56, beta=-7)
    cv2.imwrite(str(IMAGES / "cctv_low_light.png"), low_light)
    blurred = cv2.GaussianBlur(original, (9, 9), 0)
    rng = np.random.default_rng(108)
    noise = rng.normal(0, 9, blurred.shape).astype(np.int16)
    noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(IMAGES / "cctv_blurred_noisy.png"), noisy)

    warm = np.full((300, 420, 3), (72, 105, 132), dtype=np.uint8)
    cv2.rectangle(warm, (0, 220), (420, 300), (48, 57, 59), -1)
    cv2.rectangle(warm, (128, 98), (280, 225), (0, 0, 218), -1)
    cv2.rectangle(warm, (128, 98), (280, 225), (18, 18, 20), 4)
    cv2.ellipse(warm, (204, 101), (46, 24), 0, 180, 360, (18, 18, 22), 5)
    cv2.imwrite(str(IMAGES / "red_bag_warm_background.png"), warm)

    distractors = np.full((300, 420, 3), (142, 153, 144), dtype=np.uint8)
    cv2.rectangle(distractors, (0, 230), (420, 300), (54, 60, 57), -1)
    cv2.rectangle(distractors, (38, 170), (115, 229), (0, 0, 228), -1)
    cv2.circle(distractors, (190, 197), 34, (0, 0, 190), -1)
    cv2.rectangle(distractors, (264, 152), (340, 228), (220, 80, 10), -1)
    cv2.rectangle(distractors, (344, 168), (385, 228), (25, 170, 45), -1)
    cv2.imwrite(str(IMAGES / "red_objects_distractors.png"), distractors)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    REFERENCES.mkdir(parents=True, exist_ok=True)
    original = room_scene()
    cv2.imwrite(str(IMAGES / "cctv_corrupted.png"), original)
    image_sample_scenes(original)
    later = room_scene(later=True)
    cv2.imwrite(str(IMAGES / "lab_reference_2343.png"), later)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(str(REFERENCES / "clock_enhanced_reference.png"), enhanced)
    cv2.imwrite(str(REFERENCES / "clock_crop_reference.png"), enhanced[15:80, 405:710])
    cv2.imwrite(str(IMAGES / "red_bag_scene.png"), original)
    false_positive = original.copy()
    cv2.rectangle(false_positive, (75, 220), (126, 269), (0, 0, 170), -1)
    cv2.putText(false_positive, "RED LABEL", (30, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 205), 1)
    cv2.imwrite(str(IMAGES / "similar_colour_scene.png"), false_positive)
    partial = original.copy()
    cv2.rectangle(partial, (526, 177), (570, 242), (33, 35, 38), -1)
    cv2.imwrite(str(IMAGES / "partially_obscured_bag.png"), partial)
    for filename, name, initials, color in [
        ("aarav.png", "Aarav", "A", (35, 70, 147)),
        ("kabir.png", "Kabir", "K", (74, 84, 95)),
        ("meera.png", "Meera", "M", (75, 103, 111)),
        ("ramesh.png", "Ramesh", "R", (68, 81, 71)),
    ]:
        cv2.imwrite(str(ROOT / "assets" / "portraits" / filename), portrait(name, initials, color))
    video_path = VIDEOS / "corridor_staged.mp4"
    fps, frame_count, width, height, line_x = 10, 70, 640, 360, 480
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4V video writer")
    for frame_index in range(frame_count):
        frame = np.full((height, width, 3), (36, 39, 40), dtype=np.uint8)
        cv2.rectangle(frame, (0, 260), (width, height), (50, 51, 48), -1)
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 155, 210), 3)
        cv2.putText(frame, "EXIT LINE", (line_x - 88, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 192, 120), 2)
        cv2.putText(frame, "STAGED CORRIDOR | 23:43", (18, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (214, 213, 201), 1)
        bag_x = 110 + frame_index * 6
        cv2.rectangle(frame, (bag_x, 145), (bag_x + 78, 245), (0, 0, 210), -1)
        cv2.rectangle(frame, (bag_x, 145), (bag_x + 78, 245), (12, 12, 18), 3)
        cv2.rectangle(frame, (bag_x + 18, 129), (bag_x + 60, 148), (0, 0, 175), -1)
        cv2.ellipse(frame, (bag_x + 40, 145), (20, 13), 0, 180, 360, (15, 15, 22), 3)
        # Simplified volunteer silhouette is explicitly synthetic; the red bag is the tracked signal.
        cv2.circle(frame, (bag_x + 40, 93), 23, (136, 127, 116), -1)
        cv2.rectangle(frame, (bag_x + 14, 111), (bag_x + 66, 166), (101, 105, 103), -1)
        writer.write(frame)
    writer.release()
    write_corridor(VIDEOS / "corridor_fast.mp4", speed=8, frame_total=70, line_x=line_x)
    write_corridor(VIDEOS / "corridor_occlusion.mp4", speed=5, frame_total=90, line_x=line_x, hide_range=(25, 34), lighting_change=True, distraction=True)
    video_records = [
        {"path": "assets/videos/corridor_staged.mp4", "fps": 10, "frame_count": 70, "width": 640, "height": 360, "exit_line_x": 480, "expected_crossing_frame": 56, "challenge": "steady movement"},
        {"path": "assets/videos/corridor_fast.mp4", "fps": 10, "frame_count": 70, "width": 640, "height": 360, "exit_line_x": 480, "expected_crossing_frame": 52, "challenge": "faster movement"},
        {"path": "assets/videos/corridor_occlusion.mp4", "fps": 10, "frame_count": 90, "width": 640, "height": 360, "exit_line_x": 480, "expected_crossing_frame": 83, "challenge": "temporary disappearance, lighting shift and small red distractor"},
    ]
    manifest = {
        "fictional_simulation": True,
        "asset_method": "Original OpenCV-generated synthetic evidence; no real people or external photos.",
        "images": {"cctv": "assets/images/cctv_corrupted.png", "low_light": "assets/images/cctv_low_light.png", "blurred_noisy": "assets/images/cctv_blurred_noisy.png", "later_scene": "assets/images/lab_reference_2343.png", "bag": "assets/images/red_bag_scene.png", "warm_background_bag": "assets/images/red_bag_warm_background.png", "red_distractors": "assets/images/red_objects_distractors.png", "false_positive": "assets/images/similar_colour_scene.png", "occluded": "assets/images/partially_obscured_bag.png", "real_backpack_person": "assets/images/real_red_backpack_person.jpg", "real_smartphone": "assets/images/real_smartphone_use.jpg", "real_cup": "assets/images/real_coffee_cup.jpg"},
        "references": ["assets/references/clock_enhanced_reference.png", "assets/references/clock_crop_reference.png"],
        "portraits": ["assets/portraits/aarav.png", "assets/portraits/kabir.png", "assets/portraits/meera.png", "assets/portraits/ramesh.png"],
        "scene_ground_truth": {"early_2342": {"people": 2, "mobile_phone": 1, "chai_cups": 1, "red_backpack": 1, "secondary_bag": 1, "prototype_device": 1}, "later_2343": {"people": 2, "mobile_phone": 0, "chai_cups": 2, "red_backpack": 1, "secondary_bag": 1, "prototype_device": 0}, "notes": "Counts are manual ground truth for the original synthetic drawings, not detector inference."},
        "video": {"path": "assets/videos/corridor_staged.mp4", "fps": fps, "frame_count": frame_count, "width": width, "height": height, "exit_line_x": line_x, "expected_crossing_frame": 56, "expected_carrier": "Kabir (fictional case record; tracking only follows a red candidate)", "scripted_time": "23:43", "synthetic": True},
        "additional_videos": video_records
    }
    (ROOT / "data" / "manifests" / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    image_purposes = {
        "cctv_corrupted.png": ("Level 1 low-contrast lab still with staged 23:42 overlay; Level 2 earlier scene", ["person silhouettes", "red backpack", "phone", "chai cup", "prototype prop"], "23:42 staged overlay"),
        "cctv_low_light.png": ("Level 1 low-light enhancement challenge", ["same staged lab scene"], "23:42 staged overlay"),
        "cctv_blurred_noisy.png": ("Level 1 blur and sensor-noise challenge", ["same staged lab scene"], "23:42 staged overlay"),
        "lab_reference_2343.png": ("Level 2 later-scene comparison reference", ["person silhouettes", "red backpack", "two chai cups"], "23:43 staged reference"),
        "red_bag_scene.png": ("Level 3 primary staged red-region scene", ["red backpack", "red lab props"], "23:42 staged"),
        "red_bag_warm_background.png": ("Level 3 red backpack against a warm background", ["red backpack"], "none"),
        "red_objects_distractors.png": ("Level 3 multiple red and non-red distractors", ["red rectangular object", "red circular object", "blue object", "green object"], "none"),
        "similar_colour_scene.png": ("Level 3 false-positive red label challenge", ["red backpack", "red label"], "23:42 staged"),
        "partially_obscured_bag.png": ("Level 3 partial occlusion challenge", ["partially obscured red backpack"], "23:42 staged"),
    }
    files = []
    for name, (purpose, expected, timestamp) in image_purposes.items():
        files.append({"filename": "assets/images/" + name, "purpose": purpose, "creation_method": "Generated locally with OpenCV drawing primitives; synthetic fictional evidence.", "license": "Original project-created asset; no external license or third-party source.", "expected_objects": expected, "timestamp": timestamp, "media_type": "image"})
    external_samples = [
        ("real_red_backpack_person.jpg", "Person carrying a red backpack; real-photo YOLO class check.", "person, backpack", "ambermb (Pixabay); Wikimedia Commons file Boy-in-brown-hoodie-carrying-red-backpack-while-walking-on-207697.jpg; CC0", "https://commons.wikimedia.org/wiki/File:Boy-in-brown-hoodie-carrying-red-backpack-while-walking-on-207697.jpg"),
        ("real_smartphone_use.jpg", "Real-photo YOLO person and cell-phone class check.", "person, cell phone", "Océanos y dados; Wikimedia Commons file Smartphone Use.jpg; CC0", "https://commons.wikimedia.org/wiki/File:Smartphone_Use.jpg"),
        ("real_coffee_cup.jpg", "Real-photo YOLO cup class check.", "cup", "Edward; Wikimedia Commons file Cup of coffee in Café Butter - Prenzlauer Berg, Berlin.jpg; CC0", "https://commons.wikimedia.org/wiki/File:Cup_of_coffee_in_Caf%C3%A9_Butter_-_Prenzlauer_Berg,_Berlin.jpg"),
    ]
    for name, purpose, expected, credit, source_url in external_samples:
        files.append({"filename": "assets/images/" + name, "purpose": purpose, "creation_method": "960-pixel Wikimedia Commons thumbnail downloaded for this project.", "license": credit, "source_url": source_url, "expected_objects": expected.split(", "), "timestamp": "not applicable", "media_type": "image"})
    for reference in ("clock_enhanced_reference.png", "clock_crop_reference.png"):
        files.append({"filename": "assets/references/" + reference, "purpose": "Prepared Level 1 comparison output, not a live-processing result.", "creation_method": "Generated from cctv_corrupted.png using OpenCV CLAHE and crop.", "license": "Original project-created derivative of synthetic project evidence.", "expected_objects": ["staged timestamp overlay"], "timestamp": "23:42 staged overlay", "media_type": "image"})
    for name in ("aarav.png", "kabir.png", "meera.png", "ramesh.png"):
        files.append({"filename": "assets/portraits/" + name, "purpose": "Fictional suspect profile illustration.", "creation_method": "Generated locally with OpenCV drawing primitives and initials; not a real-person likeness.", "license": "Original project-created asset; no external license or third-party source.", "expected_objects": ["initials-based fictional portrait"], "timestamp": "not applicable", "media_type": "image"})
    for record in video_records:
        files.append({"filename": record["path"], "purpose": "Level 4 controlled tracking sample: " + record["challenge"], "creation_method": "Generated locally frame-by-frame with OpenCV drawing primitives.", "license": "Original project-created asset; no external license or third-party source.", "expected_objects": ["synthetic carrier silhouette", "red backpack"], "timestamp": "23:43 is staged story context; crossing timestamp is clip-relative", "media_type": "video", "fps": record["fps"], "frame_count": record["frame_count"], "expected_crossing_frame": record["expected_crossing_frame"], "exit_line_x": record["exit_line_x"]})
    dataset_manifest = {"schema_version": 1, "fictional_simulation": True, "model": {"path": "models/yolov8n.onnx", "required": False, "name": "Ultralytics YOLOv8n pretrained on COCO", "source_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", "license": "AGPL-3.0; review Ultralytics terms before redistribution or deployment.", "setup_command": "python -m pip install -e \".[model]\" && python scripts/setup_model.py", "reason": "ONNX weights are prepared locally and ignored by Git; setup requires network access and the optional exporter."}, "files": files}
    (ROOT / "data" / "manifests" / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n")
    print("Generated", len(list(IMAGES.glob("*.png"))), "evidence images and", video_path)


if __name__ == "__main__":
    main()