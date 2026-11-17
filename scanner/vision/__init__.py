"""Vision, shared by every puzzle: reading a puzzle from photos.

    photo -> locator (the face's corners) -> crops cut on the face -> reader -> probabilities -> the puzzle's decoder

``common``    the building blocks every puzzle's networks share: ResNet trunk, heatmap locator, letterbox,
              loaders, learning-rate schedule, temperature fitting, photo helpers
``augment``   what a real camera does to a picture (training only)
``train``     python -m scanner.vision.train <what> --puzzle <key> ...   (each puzzle's vision/train.py)
``evaluate``  python -m scanner.vision.evaluate <what> --puzzle <key> ... (each puzzle's vision/evaluate.py)

Each puzzle's own geometry, dataset, models and pipeline are in scanner/puzzles/<key>/vision/.
"""
