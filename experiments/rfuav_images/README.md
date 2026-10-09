# RFUAV spectrogram experiments

Earlier equipment-classification experiments using the public RFUAV
ResNet18 checkpoint. This directory is separate from the raw-I/Q ensemble.
Its associated O2/O3/O3+/O4 family is a specification lookup after equipment
prediction, not independent protocol-version detection.

The code, saved linear heads, fixed sample selections and historical results
are included. The 44.8 MB upstream backbone and complete image selections
are downloaded separately. No image inference result is used as evidence
that the raw-I/Q pipeline works.

## Run

Use a separate Python 3.12 environment and run from this directory:

```sh
python -m pip install -r requirements-classifier.txt
python rf_baseline.py prepare --samples-per-class 10
python rf_baseline.py evaluate
python -m unittest test_rf_baseline test_detector_model test_numeric_rf -v
```

Preparation downloads the pinned upstream checkpoint and selected public
images. It updates the local selection manifest. To classify a single
image after the checkpoint has been prepared:

```sh
python adapted_classifier.py image.jpg
python detector_model.py image.jpg
```

`adapt_head.py` fits a five-class head. `check_open_set.py` checks additional
equipment. `train_detector.py` then fits a six-class head with OTHER_SIGNAL.
Run them in that order to reproduce the experiment chain. They write local
experiment files and download additional images.

The stored six-class run accepted 49 of 50 known-equipment images correctly,
abstained on one, and accepted none of 64 other-equipment images. This is a
corpus image test, not physical-receiver or independent-session validation.
`numeric_rf.py` is a separate numeric representation and is deliberately not
a drop-in input transform for the JPEG checkpoint.
