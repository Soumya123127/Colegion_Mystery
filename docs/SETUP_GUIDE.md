# Setup Guide

## Install Python dependencies

Python 3.11+ is recommended; 3.9+ is supported. Create a virtual environment and install the small image-processing/test dependencies:

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install jupyterlab
python scripts/prepare_data.py
python scripts/verify_assets.py
```

`opencv-python-headless` is deliberate: it works without desktop GUI libraries. The notebooks use Matplotlib display; no `cv2.imshow()` is required.

## Generate and audit evidence

```sh
python scripts/prepare_data.py
python scripts/verify_assets.py
```

The script creates synthetic PNG scenes, low-light/noisy variants, initials portraits, reference crops and three MP4 tracking clips. Three CC0 Wikimedia photos are already included; they are not overwritten by generation. It may be rerun; generated output is deterministic. `asset_manifest.json` describes fictional ground truth, and `dataset_manifest.json` records purpose, source method, license and expected scene objects per file.

## Run the investigation interface

```sh
python scripts/server.py
```

Visit `http://127.0.0.1:8000/app/`. To change port: `python scripts/server.py 8001`. The server is local-only and offers endpoints for actual enhancement, scene difference, red segmentation, video tracking, clue validation and HTML report rendering. Stop it with Ctrl+C.

Each mission includes a prepared-sample selector, live parameter controls, upload input and notebook link. Changes reach the underlying OpenCV functions. Upload a photo in levels 1–3; Level 2 also accepts video for first-frame inference/edge analysis. Level 4 accepts a recorded video for red-candidate tracking. The **Open Live Camera** button requests browser permission in levels 1–3 and captures a still for the current level's algorithm; at Level 4 it runs red-candidate analysis on a still, not temporal tracking. For tracking, choose one of the three prepared clips or another recorded video. Uploads are processed by the local server only, limited to 18 MB, and temporary source files are deleted after processing. Camera access works on localhost in current Safari/Chrome when permission is granted; browser behavior may vary.

## Run notebooks and tests

```sh
jupyter lab notebooks/
python -m pytest -q
python scripts/run_demo.py
```

The notebooks load reusable package functions; they do not embed duplicate algorithm implementations. Run from the repository root. A browser print dialog saves the final HTML report as PDF.

## Prepare the Level 2 detector

Model setup is separate because the optional exporter installs PyTorch and increases setup size. To enable real CPU inference, install the model extra and run the exporter:

```sh
python -m pip install -e ".[model]"
python scripts/setup_model.py
```

This downloads the official YOLOv8n COCO checkpoint and exports a fixed 640x640 ONNX model to ignored `models/`. The script makes a real OpenCV DNN forward pass to validate the export. Supported classes include person, backpack, cup and cell phone. Synthetic art may not trigger detections; selected real photos demonstrate imperfect detections and false positives. Ultralytics terms are AGPL-3.0; see [ASSET_CREDITS.md](ASSET_CREDITS.md). Without weights, the scene comparison is clearly identified as pixel analysis, never model inference.

## Fine-tune a custom detector (organizers)

Custom YOLO training is optional and must not be a live workshop dependency. Collect and annotate permitted images of the actual prop(s), with varied lighting, distance and background. Split by complete recording or scene, not by adjacent frames, to reduce leakage. A few hundred varied images is a reasonable prototype starting point, not a guarantee of accuracy. Use CVAT, Label Studio or another YOLO-format annotation tool; keep image and matching YOLO `.txt` label files under each split.

Copy `data/custom_yolo.yaml.example` to `data/custom_yolo.yaml`, then create `data/custom_yolo/images/{train,val,test}` and matching `data/custom_yolo/labels/{train,val,test}` folders. Edit the YAML class list to match the annotations exactly; class IDs must start at zero and be consecutive. Keep test scenes separate from training and validation scenes.

After installing the `model` extra and running `scripts/setup_model.py`, train before the event:

```sh
python scripts/train_custom_model.py --data data/custom_yolo.yaml --epochs 50 --imgsz 640 --batch 16 --patience 10
```

The command fine-tunes the pretrained YOLOv8n checkpoint, validates during training, evaluates the held-out `test` split, and exports `models/custom_yolo.onnx` with its class names and test metrics. The portal automatically prefers that custom export; otherwise it uses the pretrained COCO model. Artifacts under `models/` are local and ignored by Git. Review precision, recall, mAP and false positives on real workshop evidence before relying on the detector. HSV color segmentation, image differencing, CLAHE and centroid tracking remain traditional algorithms and need no training.
