# Participant Handbook

## Case ARG-2342

You are an investigator using the fictional ARGUS training system. All people, evidence and events are invented. The task is to learn how OpenCV changes and measures pixels, not to treat computer vision as proof of a real person's identity.

## Before starting

1. Follow [SETUP_GUIDE.md](SETUP_GUIDE.md).
2. Run `python scripts/verify_assets.py` from the project root.
3. Launch the portal and open each notebook alongside it.
4. Record what the computer reports separately from what the case file says.
5. In each mission choose a prepared sample or upload your own file; change one live parameter before rerunning.

## Mission notes

### 1. Eyes Online
Choose the staged, low-light, or blurred/noisy sample. Inspect dimensions and channels; compare grayscale, brightness, equalization, CLAHE, blur and thresholding. Use the brightness, contrast, CLAHE clip/tile, blur and threshold controls, then inspect each computed intermediate. Crop the upper-right clock and save the enhanced frame.

- Clue submission: staged overlay time, `HH:MM`.
- Try: increase brightness too far. What details clip?
- Remember: enhancement cannot recreate absent pixels and does not establish a time of death.

### 2. The Missing Clue
Compare the staged scenes, then try the real-photo person/backpack, smartphone, and coffee samples. If YOLOv8n weights are installed, change confidence and inspect live predictions; otherwise compare the scene inventory without calling it detector output. The model can miss objects and make false predictions.

- Clue: an additional cup appears in the later scene; case inventory marks the phone and prototype missing.
- Try: explain what a confidence score does and does not mean.
- Remember: a generic COCO model does not know the fictional prototype.

### 3. The Red Bag
Run both red HSV hue ranges and inspect the mask and contours. Adjust hue, saturation, value, morphology and minimum area. Try warm-background, multi-distractor, similar-color, partial-occlusion and real-photo samples.

- Clue: ownership is recorded in the fictional case file.
- Try: lower the area threshold. What false positives appear?
- Remember: a red candidate is not a person and color does not prove ownership.

### 4. Follow the Trail
Run the steady, fast, and occlusion/distractor clips. Inspect FPS, frame count, candidate center, trail and adjustable exit line. Note when and why the tracker loses its target.

- Clue: a red candidate's observed movement plus a separate case-file statement and prepared 23:44 log.
- Try: move the exit line or inspect a missing detection.
- Remember: an occluded bag can break a simple color track.

## Evidence worksheet

| Evidence | Observation | Source (visual / case file / fictional log) | Limitation |
| --- | --- | --- | --- |
| Timestamp |  |  | Staged clock overlay |
| Scene comparison |  |  | Inventory is not model inference |
| Backpack owner |  |  | Not inferred by color |
| Carrier / movement |  |  | Tracker follows color, not identity |
| System log |  |  | Prepared fictional text asset |

## Final accusation form

- Fictional culprit: __________
- Backpack owner: __________
- Backpack carrier: __________
- Scripted incident window: __________
- Prepared corruption-log time: __________
- Five independent evidence categories selected: __________
- One limitation of the CV output: __________

Submit using the portal only after discussing which conclusions depend on visual measurements and which depend on the case record. For the solution, ask the facilitator to open `FINAL_SOLUTION.md` after the reveal.
