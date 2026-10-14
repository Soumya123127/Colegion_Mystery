# Asset Credits

Synthetic scenes, variation samples, corridor clips, and fictional portraits are original assets produced by `scripts/prepare_data.py` with OpenCV drawing primitives and NumPy noise. The corridor clips use deliberately simplified generated silhouettes and red rendered bags. The portraits are generated initials-based illustrations for fictional characters; they are not photographs or depictions of real individuals.

The four Level 1 photographic practice frames in `assets/images/case_cctv_01.png` through `case_cctv_04.png` are explicitly labeled simulated feeds. They are crops and tonal derivatives of public-domain archival photographs; their depicted people and places are not connected to the fictional case. Source files are kept under `assets/images/case_sources/`:

- Hallway east view: Walt Disney Studios, Process Laboratory, Historic American Buildings Survey, public domain in the United States. [Source record](https://commons.wikimedia.org/wiki/File:INTERIOR_HALLWAY_VIEW,_FACING_EAST_-_Walt_Disney_Studios,_Process_Laboratory,_Corner_of_Buena_Vista_Street_and_Alameda_Avenue,_Burbank,_Los_Angeles_County,_CA_HABS_CAL,19-BURB,1A-19.tif).
- Hallway south view: Walt Disney Studios, Process Laboratory, Historic American Buildings Survey, public domain in the United States. [Source record](https://commons.wikimedia.org/wiki/File:INTERIOR_HALLWAY_VIEW,_FACING_SOUTH_-_Walt_Disney_Studios,_Process_Laboratory,_Corner_of_Buena_Vista_Street_and_Alameda_Avenue,_Burbank,_Los_Angeles_County,_CA_HABS_CAL,19-BURB,1A-21.tif).
- Industrial hygiene laboratory: Roy Perry Jr. / U.S. Public Health Service, 1943, public domain U.S. government photograph. [Source record](https://commons.wikimedia.org/wiki/File:Industrial_Hygiene_Laboratory_interior_1943.png).
- Research workshop: NACA / NASA, Aircraft Engine Research Laboratory, 1946, public domain U.S. government photograph. [Source record](https://commons.wikimedia.org/wiki/File:Machine_Shop_Men_GPN-2000-000364.jpg).

Three realistic evaluation photographs are CC0 public-domain dedications downloaded as 960-pixel thumbnails from Wikimedia Commons:

- Red backpack carried by a person: [Wikimedia file page](https://commons.wikimedia.org/wiki/File:Boy-in-brown-hoodie-carrying-red-backpack-while-walking-on-207697.jpg). Original photographer credited on Commons as ambermb (Pixabay).
- Person using a smartphone: [Wikimedia file page](https://commons.wikimedia.org/wiki/File:Smartphone_Use.jpg). Creator: Océanos y dados.
- Coffee cup: [Wikimedia file page](https://commons.wikimedia.org/wiki/File:Cup_of_coffee_in_Caf%C3%A9_Butter_-_Prenzlauer_Berg,_Berlin.jpg). Creator: Edward.

The evaluation photos are inputs, not ground-truth claims about model output. COCO class predictions can be missed or wrong. Full filenames, purposes, source method, expected objects, and license notes are recorded in `data/manifests/dataset_manifest.json`.

The optional local detector is Ultralytics YOLOv8n pretrained on COCO. `scripts/setup_model.py` downloads `yolov8n.pt` from the official Ultralytics assets release `v8.3.0` (about 6.2 MB) and exports `models/yolov8n.onnx` (about 12.2 MB). Model files are ignored by Git. Organizers may optionally fine-tune and export a separate detector with `scripts/train_custom_model.py`; custom images, annotations and resulting weights are not included. Ultralytics identifies AGPL-3.0 as its open-source license and offers an Enterprise license; model/software redistribution or network deployment must be reviewed against the applicable terms.

The staged timestamps and `ARGUS` system log are fictional narrative assets. The timestamp overlay is rendered into the generated source still. OpenCV enhancement does not discover or authenticate the scripted time.
