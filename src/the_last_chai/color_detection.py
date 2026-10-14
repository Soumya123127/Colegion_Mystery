"""HSV red-object candidate detection for the staged backpack evidence."""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def red_mask(
    image: np.ndarray,
    low_saturation: int = 70,
    low_value: int = 40,
    low_hue_max: int = 10,
    high_hue_min: int = 170,
    high_saturation: int = 255,
    high_value: int = 255,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = max(0, min(255, int(low_saturation)))
    value = max(0, min(255, int(low_value)))
    low_hue_max = max(0, min(20, int(low_hue_max)))
    high_hue_min = max(159, min(179, int(high_hue_min)))
    high_saturation = max(saturation, min(255, int(high_saturation)))
    high_value = max(value, min(255, int(high_value)))
    lower_red = cv2.inRange(hsv, np.array([0, saturation, value]), np.array([low_hue_max, high_saturation, high_value]))
    upper_red = cv2.inRange(hsv, np.array([high_hue_min, saturation, value]), np.array([179, high_saturation, high_value]))
    return cv2.bitwise_or(lower_red, upper_red)


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    kernel = np.ones((size, size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def find_red_candidates(
    image: np.ndarray,
    min_area: int = 500,
    kernel_size: int = 5,
    low_saturation: int = 70,
    low_value: int = 40,
    low_hue_max: int = 10,
    high_hue_min: int = 170,
    high_saturation: int = 255,
    high_value: int = 255,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    mask = clean_mask(red_mask(image, low_saturation, low_value, low_hue_max, high_hue_min, high_saturation, high_value), kernel_size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        candidates.append({"x": x, "y": y, "w": width, "h": height, "area": area, "center": (x + width // 2, y + height // 2)})
    candidates.sort(key=lambda item: item["area"], reverse=True)
    return mask, candidates


def draw_candidates(image: np.ndarray, candidates: List[Dict[str, object]]) -> np.ndarray:
    output = image.copy()
    for number, item in enumerate(candidates, 1):
        x, y, width, height = (int(item[key]) for key in ("x", "y", "w", "h"))
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 220, 255), 2)
        cv2.putText(output, "red candidate {}".format(number), (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    return output