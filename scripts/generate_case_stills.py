"""Prepare four photographic, clearly simulated CCTV practice frames."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "assets" / "images"
SOURCES = IMAGES / "case_sources"
OUTPUT_SIZE = (1280, 720)

SAMPLES = [
    ("industrial_hygiene_lab_1943.png", "CAM 03 / RESEARCH LAB", None),
    ("laboratory_hallway_east.jpg", "CAM 01 / EAST HALL", (0.03, 0.20, 0.97, 0.76)),
    ("laboratory_hallway_south.jpg", "CAM 02 / WEST HALL", (0.03, 0.20, 0.97, 0.76)),
    ("nasa_research_workshop_1946.jpg", "CAM 04 / WORKSHOP", None),
]


def crop_source(image: np.ndarray, bounds: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    height, width = image.shape[:2]
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        image = image[int(height * y0):int(height * y1), int(width * x0):int(width * x1)]
    height, width = image.shape[:2]
    target_width, target_height = OUTPUT_SIZE
    target_ratio = target_width / target_height
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = int(height * target_ratio)
        x0 = (width - crop_width) // 2
        image = image[:, x0:x0 + crop_width]
    else:
        crop_height = int(width / target_ratio)
        y0 = (height - crop_height) // 2
        image = image[y0:y0 + crop_height, :]
    return cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)


def simulated_feed(image: np.ndarray, camera: str, seed: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.convertScaleAbs(gray, alpha=0.86, beta=-7)

    height, width = gray.shape
    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_axis, y_axis)
    vignette = np.clip(1.0 - 0.16 * (xx * xx + yy * yy), 0.72, 1.0)
    gray = np.clip(gray.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 2.1, gray.shape).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.45)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(frame, (0, 0), (width, 48), (9, 12, 13), -1)
    cv2.putText(frame, "26/10/2024 23:42:{:02d}  |  SIMULATED FEED".format(18 + seed % 40), (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 238, 232), 1, cv2.LINE_AA)
    cv2.putText(frame, camera, (width - 340, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (211, 215, 210), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    for index, (filename, camera, bounds) in enumerate(SAMPLES, 1):
        source_path = SOURCES / filename
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError("Missing public-domain source image: {}".format(source_path))
        frame = simulated_feed(crop_source(source, bounds), camera, index)
        target = IMAGES / "case_cctv_{:02d}.png".format(index)
        if not cv2.imwrite(str(target), frame):
            raise OSError("Could not write simulated CCTV frame: {}".format(target))
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()