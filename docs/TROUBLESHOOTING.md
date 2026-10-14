# Troubleshooting

- **`cv2` cannot be imported:** activate `.venv`, then run `python -m pip install -r requirements.txt`. Avoid mixing system Python and environment pip.
- **`imread` returns `None`:** run commands from the repository root and check spelling/case of the relative path. `python scripts/prepare_data.py` regenerates missing synthetic assets.
- **Notebook import fails:** activate the environment, run `python -m pip install -e .`, and launch Jupyter from the repository root.
- **No `cv2.imshow()` or trackbar:** expected with the headless package. Use notebook Matplotlib displays and edit the documented threshold parameters.
- **Level 2 says weights are missing:** install the optional exporter with `python -m pip install -e ".[model]"`, then run `python scripts/setup_model.py`. If model setup is unavailable, use the prepared scene comparison; no detections are fabricated.
- **ONNX inference returns no boxes or errors:** confirm that the export is a compatible 640x640 COCO YOLOv5/YOLOv8 model and inspect its output tensor layout. Different exports require decoder adjustments.
- **No red candidate:** use the bright staged frame, check the dual hue intervals in `color_detection.py`, reduce `min_area`, and inspect lighting/occlusion.
- **False red boxes:** increase minimum contour area, saturation/value thresholds, or compare the candidate with the prepared ground truth. Red labels can be detected.
- **Video does not open:** regenerate it with `python scripts/prepare_data.py` and inspect `python scripts/verify_assets.py` output. OpenCV codec support varies.
- **VideoWriter cannot create MP4:** try another local codec/build; the pipeline raises a clear error instead of claiming a saved file. The notebook can display individual frames.
- **Portal directory returns 404:** use the documented `http://127.0.0.1:8000/app/` URL and current server script. Change port if occupied.
- **Report tab did not open:** allow pop-ups for localhost; use the browser print dialog and choose Save as PDF.
- **Windows activation:** use `.venv\Scripts\Activate.ps1` in PowerShell. Windows paths and codec behavior were not verified in this run.
- **A laptop cannot install:** pair participants and project the facilitator's working local portal. The server fallback is still real OpenCV processing, not a static answer image.
- **Upload rejected:** supported still images are PNG/JPEG/WebP/BMP and videos MP4/MOV/M4V/AVI/WebM; each file must be under 18 MB. Some browser video codecs are not included in an OpenCV build; try MP4 H.264 if decoding fails.
- **Camera does not start:** allow camera permission for localhost in Safari/Chrome and verify no other application is holding the camera. Choose a photo from disk if camera access is unavailable. Camera capture processes a still; Level 4 temporal tracking requires a recorded video upload.
