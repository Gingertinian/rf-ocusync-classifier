import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from rf_baseline import CLASSES, LINK_FAMILY, MODEL_SHA256, ROOT, load_model


OTHER_INDEX = len(CLASSES)
DETECTOR_CLASSES = (*CLASSES, "OTHER_SIGNAL")


def decision(scores, threshold):
    if len(scores) != len(DETECTOR_CLASSES):
        raise ValueError("Wrong score count")
    values = torch.as_tensor(scores, dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any() or abs(float(values.sum()) - 1) > 1e-5:
        raise ValueError("Scores must be finite nonnegative values summing to one")
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1]")
    index = int(values.argmax())
    score = float(values[index])
    accepted = index != OTHER_INDEX and score >= threshold
    return {
        "candidate_equipment": DETECTOR_CLASSES[index],
        "accepted": accepted,
        "label": DETECTOR_CLASSES[index] if accepted else "UNKNOWN_OR_INSUFFICIENT_EVIDENCE",
        "associated_link_family_lookup": LINK_FAMILY[index] if accepted else None,
        "uncalibrated_softmax_score": score,
        "threshold": threshold,
        "direct_protocol_version_detection": False,
    }


class Detector:
    def __init__(self):
        self.backbone, self.transform = load_model()
        self.backbone.fc = nn.Identity()
        saved = torch.load(ROOT / "detector" / "model_head.pt", map_location="cpu", weights_only=True)
        if saved["base_checkpoint_sha256"] != MODEL_SHA256 or tuple(saved["classes"]) != DETECTOR_CLASSES:
            raise ValueError("Detector does not match the base checkpoint and class order")
        self.mean, self.scale = saved["feature_mean"], saved["feature_scale"]
        if self.mean.shape != (512,) or self.scale.shape != (512,):
            raise ValueError("Unexpected feature scaling shape")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("Invalid feature scaling")
        self.threshold = float(saved["threshold"])
        self.head = nn.Linear(512, len(DETECTOR_CLASSES))
        self.head.load_state_dict(saved["head"], strict=True)
        self.head.eval()

    def features(self, image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            return self.backbone(tensor)

    def classify_image(self, image):
        features = self.features(image)
        with torch.inference_mode():
            scores = self.head((features - self.mean) / self.scale).softmax(1)[0]
        result = decision(scores.tolist(), self.threshold)
        result["scores_by_equipment"] = dict(zip(DETECTOR_CLASSES, scores.tolist()))
        return result

    def classify_path(self, path):
        with Image.open(path) as image:
            return {"input": str(path), **self.classify_image(image)}


def main():
    parser = argparse.ArgumentParser(description="Offline RF equipment classifier with an unknown-signal class.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Image does not exist")
    result = Detector().classify_path(args.image)
    result["scope"] = "RFUAV spectrogram recipe only; raw I/Q and independent receivers are not yet validated"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
