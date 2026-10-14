"""Frame-by-frame red candidate tracking with evidence-based line crossing."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .color_detection import find_red_candidates


def line_crossed(previous: Optional[Tuple[int, int]], current: Optional[Tuple[int, int]], line_x: int) -> bool:
    return previous is not None and current is not None and previous[0] < line_x <= current[0]


def track_video(
    video_path: str,
    output_path: Optional[str] = None,
    line_x: int = 420,
    min_area: int = 700,
    kernel_size: int = 5,
    low_saturation: int = 70,
    low_value: int = 40,
    low_hue_max: int = 10,
    high_hue_min: int = 170,
    high_saturation: int = 255,
    high_value: int = 255,
) -> Dict[str, object]:
    source = cv2.VideoCapture(str(video_path))
    if not source.isOpened():
        raise OSError("Could not open video: {}".format(video_path))
    fps = source.get(cv2.CAP_PROP_FPS)
    frame_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps or 10.0, (width, height))
        if not writer.isOpened():
            source.release()
            raise OSError("Could not create output video: {}".format(target))
    positions: List[Tuple[int, int]] = []
    previous = None
    crossing = None
    valid_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            _, candidates = find_red_candidates(
                frame,
                min_area=min_area,
                kernel_size=kernel_size,
                low_saturation=low_saturation,
                low_value=low_value,
                low_hue_max=low_hue_max,
                high_hue_min=high_hue_min,
                high_saturation=high_saturation,
                high_value=high_value,
            )
            current = candidates[0]["center"] if candidates else None
            if current is not None:
                current = (int(current[0]), int(current[1]))
                valid_frames += 1
                positions.append(current)
                positions = positions[-24:]
                if crossing is None and line_crossed(previous, current, line_x):
                    crossing = {"frame": frame_index, "seconds": frame_index / fps if fps else 0.0, "center": list(current)}
            annotated = frame.copy()
            cv2.line(annotated, (line_x, 0), (line_x, height), (0, 180, 255), 2)
            for point in positions:
                cv2.circle(annotated, point, 3, (255, 220, 40), -1)
            if current is not None:
                box = candidates[0]
                cv2.rectangle(annotated, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), (0, 230, 255), 2)
                cv2.circle(annotated, current, 5, (255, 255, 255), -1)
            label = "frame {} | {}".format(frame_index, "candidate" if current else "no valid candidate")
            cv2.putText(annotated, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(annotated)
            previous = current
            frame_index += 1
    finally:
        source.release()
        if writer is not None:
            writer.release()
    return {"fps": fps, "frame_count": frame_count, "frames_processed": frame_index, "width": width, "height": height, "valid_detection_frames": valid_frames, "crossing": crossing, "output_path": output_path}