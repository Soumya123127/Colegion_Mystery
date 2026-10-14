"""Small, reusable image operations used in the first investigation mission."""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(path))
    return image


def image_info(image: np.ndarray) -> Dict[str, object]:
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    height, width = image.shape[:2]
    return {"width": width, "height": height, "channels": 1 if image.ndim == 2 else image.shape[2]}


def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """Apply output = alpha * input + beta; alpha changes contrast, beta brightness."""
    return cv2.convertScaleAbs(image, alpha=float(alpha), beta=int(beta))


def enhance_image(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
    brightness: int = 0,
    contrast: float = 1.0,
    blur_kernel: int = 5,
    threshold_method: str = "otsu",
    threshold_value: int = 127,
) -> Dict[str, np.ndarray]:
    adjusted = adjust_brightness_contrast(image, alpha=contrast, beta=brightness)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY) if adjusted.ndim == 3 else adjusted.copy()
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(gray)
    kernel = max(1, int(blur_kernel))
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(clahe, (kernel, kernel), 0)
    threshold_type = cv2.THRESH_BINARY
    threshold = int(threshold_value)
    if threshold_method == "otsu":
        threshold_type |= cv2.THRESH_OTSU
        threshold = 0
    elif threshold_method != "binary":
        raise ValueError("threshold_method must be 'otsu' or 'binary'")
    _, thresholded = cv2.threshold(clahe, threshold, 255, threshold_type)
    return {"gray": gray, "equalized": equalized, "clahe": clahe, "blurred": blurred, "threshold": thresholded}


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(image_width, x0 + int(width)), min(image_height, y0 + int(height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Crop does not intersect the image")
    return image[y0:y1, x0:x1].copy()


def save_image(path: str, image: np.ndarray) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError("OpenCV could not save image: {}".format(target))
    return str(target)