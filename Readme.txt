# Rod Bubble Detection Project

This repository contains Python code for:
- Bubble detection and analysis in rod-bubble systems
- Velocity tracking across high-speed TIFF frames
- Collision probability estimation
- 2D/3D modeling of bubble setups

1. bubble-detection-pipeline.py
Purpose: GUI‑driven image loading & interactive cropping, followed by an enhanced preprocessing & contour‑based bubble detection pipeline.

Key features:

Adaptive background subtraction

CLAHE contrast enhancement

Gaussian + median filtering

Adaptive thresholding + morphology

Contour analysis with ellipse fitting & shape metrics

Exports stage‑wise images + CSV results

2. bubble-velocity-tracker.py
Purpose: Link bubble detections across frames to compute per‑bubble velocities.

Key features:

Nearest‑neighbor matching

Pixel→mm & frame rate calibration (default 6400 fps)

Exports multi‑sheet Excel with raw & summary velocity stats

3. bubble-collision-tracker.py
Purpose: Detects potential bubble collisions frame‑by‑frame.

Key features:

Distance‑based collision threshold

Velocity‑weighted collision probability

Exports Excel sheets: all collisions, high‑probability, per‑frame stats, pair analyses

4. ThreeDplanegenerator.py
Purpose: Generate n random non‑overlapping spheres in a cuboid; render a 3D view.

Key features:

Configurable box size & radius sets

Wireframe cuboid + colored sphere surfaces

Orthographic projections onto ±X, ±Y, ±Z faces

2D pixel‑coverage heatmap analysis

5. ThreeDPlaneGenerator2.py
Purpose: Alternative 3D‑plane & projection utility (e.g. simplified or experimental version).

Key features: TBD—see inline docstrings.

6. multi-perimeter-overlap-detector.py
Purpose: Advanced detection of overlapping circles in complex images.

Key features:

CLAHE + Gaussian pre‑processing

Hough Transform with multiple parameter sets

Contour‑based validation by circularity & edge density

DBSCAN clustering to merge duplicates

Partial‑circle recovery for occluded segments

7. circle-maker.py
Purpose: Simple utility to draw/test circles on images; useful for parameter tuning.

Key features:

GUI select image

Input radius/center to overlay circles

Save annotated output

8. coin-detector.py
Purpose: GUI application for coin detection & calibration.

Key features:

Hough‑based circle detection

User‑guided calibration (px/mm) via actual diameters

Dual‑panel UI: raw vs. annotated image

Pixel & real‑world unit reports