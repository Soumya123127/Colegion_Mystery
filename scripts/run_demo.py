"""Run all lightweight pipelines and write honest facilitator reference outputs."""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from the_last_chai.color_detection import draw_candidates, find_red_candidates
from the_last_chai.image_enhancement import crop_image, enhance_image, load_image, save_image
from the_last_chai.video_tracking import track_video


def main():
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    still = load_image(str(ROOT / "assets/images/cctv_corrupted.png"))
    enhanced = enhance_image(still)
    save_image(str(output / "level1_enhanced.png"), enhanced["clahe"])
    save_image(str(output / "level1_clock_crop.png"), crop_image(enhanced["clahe"], 405, 15, 305, 65))
    bag_scene = load_image(str(ROOT / "assets/images/red_bag_scene.png"))
    _, candidates = find_red_candidates(bag_scene, min_area=500)
    save_image(str(output / "level3_candidates.png"), draw_candidates(bag_scene, candidates))
    tracking = track_video(str(ROOT / "assets/videos/corridor_staged.mp4"), str(output / "corridor_tracked.mp4"), line_x=480, min_area=700)
    later = load_image(str(ROOT / "assets/images/lab_reference_2343.png"))
    earlier = load_image(str(ROOT / "assets/images/cctv_corrupted.png"))
    difference = cv2.absdiff(later, earlier)
    save_image(str(output / "level2_pixel_difference.png"), difference)
    summary = {"fictional_simulation": True, "level1": {"width": still.shape[1], "height": still.shape[0], "staged_timestamp_reference": "23:42", "output": "outputs/level1_clock_crop.png"}, "level2": {"mean_pixel_difference": float(difference.mean()), "not_model_inference": True, "output": "outputs/level2_pixel_difference.png"}, "level3": {"candidate_count": len(candidates), "boxes": candidates, "output": "outputs/level3_candidates.png"}, "level4": tracking}
    (output / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
