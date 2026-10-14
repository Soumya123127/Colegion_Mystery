# Validation Record

## Verified in the local macOS workspace

- Python 3.9.6, OpenCV 4.11.0, NumPy 1.26.4, pytest, nbformat, Matplotlib, and the optional Ultralytics exporter installed in the workspace virtual environment.
- `python scripts/prepare_data.py`: deterministically generated synthetic scenes, low-light/noisy variations, initials portraits, clock references, and three controlled tracking videos. The three CC0 Wikimedia photos are retained and included in the manifest.
- `python scripts/verify_assets.py`: 18 images and 4 videos readable; all provenance records exist; declared video FPS/frame counts match; output folder writable; model weights present locally.
- `python scripts/setup_model.py`: validated an OpenCV DNN forward pass with output shape `(1, 84, 8400)`.
- `python -m pytest -q`: 13 passed. Tests cover image processing and parameter sensitivity, HSV hue wrap and parameter sensitivity, video tracking and line crossings in all controlled clips, upload handling, manifest coverage, notebook code cells, and real YOLO inference on CC0 photos and sampled video frames.
- Model results at confidence 0.2: person detected in the person/backpack photo; cell phone and person detected in the smartphone photo; coffee sample produced false positives rather than a cup label. The synthetic lab drawings returned no YOLO boxes at confidence 0.25. These outputs are detector predictions, not ground truth.
- Live API check: status reported OpenCV 4.11.0 and model available. Level 1 returned grayscale, equalized, CLAHE, blurred, threshold, and crop outputs with changed settings echoed. Level 2 ran live model inference and returned no synthetic-art boxes. Level 3 returned an actual HSV mask and candidate box. Level 4 processed all 70 frames; moving the line from 75% to 70% changed the observed crossing from frame 56 to frame 50.
- HSV upper-bound route check: setting `high_value` to 150 was echoed in the response and removed the qualifying red candidates from the staged sample (candidate count changed to zero).
- Browser check: opening, case briefing, dashboard, Level 1 mission, prepared sample analysis, slider/readout interaction, and computed intermediate-image display worked at `http://127.0.0.1:8000/app/`.
- Python compile and Pylance diagnostics reported no issues in modified source files. Node.js was unavailable, so browser JavaScript was validated through the integrated browser rather than a Node syntax check.

## Not verified or known limitations

- Browser verification did not run a complete final accusation/report through every clue; deterministic clue/accusation/report functions are covered by unit tests.
- Windows/Linux installs, other camera devices, and codecs were not tested. User-provided clips may not decode on every OpenCV build.
- The model setup extra installs PyTorch and is significantly larger than the core CPU OpenCV install. Model files remain local and ignored by Git; initial setup requires internet.
- YOLOv8n is licensed under AGPL-3.0. Review Ultralytics terms before redistribution or deployment. No custom training was performed.
- COCO models may miss small/occluded objects and make false predictions; the fictional prototype is not a COCO class.
- Computer vision does not establish fictional identity, backpack ownership, cause/time of death, or guilt. Those are separately scripted case-file facts.

## Repeat checks

```sh
python scripts/prepare_data.py
python scripts/verify_assets.py
python scripts/setup_model.py
python -m pytest -q
python scripts/run_demo.py
```
