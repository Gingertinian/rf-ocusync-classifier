# Attribution and permissions

The feature extraction, ensemble fitting, input validation, replay wrappers
and local RFUAV experiments were developed by Jeronimo Munoz Larreta.

RFUAV recordings and the optional upstream ResNet18 checkpoint are by Rui
Shi and collaborators. Source code and dataset attribution:
https://github.com/kitoweeknd/RFUAV
https://huggingface.co/datasets/kitofrank/RFUAV
`LICENSE-RFUAV.txt` contains their Apache-2.0 license text. This repository
does not claim authorship of their dataset, pretrained backbone or image
conversion pipeline.

The environmental I/Q excerpt is derived from DRFF-R2 by Haolin Zheng and
collaborators and remains under CC BY 4.0. Its authors, source, license and
conversion are identified in `DATA_SOURCES.md`.

The main `model.joblib` was fitted locally on the declared RFUAV excerpts
and generated negative controls. It does not use the RFUAV pretrained image
checkpoint. The small heads in `experiments/rfuav_images` were fitted on
features from that separately downloaded upstream checkpoint.

No confidential commissioned code, correspondence, credentials, acquisition
device identifiers or private contact details are included. Original
software has no additional license grant. Third-party terms are unaffected.
