import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from rf_baseline import CLASSES, LIMITS, LINK_FAMILY, MODEL_SHA256, ROOT, load_model


class AdaptedClassifier:
    def __init__(self):
        self.backbone, self.transform = load_model()
        self.backbone.fc = nn.Identity()
        saved = torch.load(ROOT / "adaptation" / "linear_head.pt", map_location="cpu", weights_only=True)
        if saved["base_checkpoint_sha256"] != MODEL_SHA256 or tuple(saved["classes"]) != CLASSES:
            raise ValueError("Adaptation does not match this backbone and class order")
        self.mean = saved["feature_mean"]
        self.scale = saved["feature_scale"]
        self.head = nn.Linear(512, len(CLASSES))
        self.head.load_state_dict(saved["head"], strict=True)
        self.head.eval()

    def features(self, path):
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            return self.backbone(tensor)

    def classify(self, path):
        features = self.features(path)
        with torch.inference_mode():
            scores = self.head((features - self.mean) / self.scale).softmax(1)[0]
        index = int(scores.argmax())
        return {
            "input": str(path),
            "candidate_equipment": CLASSES[index],
            "associated_link_family_lookup": LINK_FAMILY[index],
            "uncalibrated_softmax_score": float(scores[index]),
            "status": "candidate_only_not_an_open_set_detector",
            "device_type_prediction_not_direct_protocol_version": True,
        }


def main():
    parser = argparse.ArgumentParser(description="Locally adapted RFUAV equipment classifier, offline spectrogram input.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Image does not exist")
    classifier = AdaptedClassifier()
    print(json.dumps({"prediction": classifier.classify(args.image), "limitations": LIMITS}, indent=2))


if __name__ == "__main__":
    main()
