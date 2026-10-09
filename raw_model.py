import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from raw_features import CLASSES, RECIPE, SAMPLE_RATE, WINDOW_SAMPLES, extract, read_window


ROOT = Path(__file__).resolve().parent


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


class RawClassifier:
    def __init__(self, model_dir=ROOT):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        if digest(model_dir / "model.joblib") != self.metadata["model_sha256"]:
            raise ValueError("Model integrity check failed")
        if self.metadata["recipe"] != RECIPE or tuple(self.metadata["classes"]) != CLASSES:
            raise ValueError("Model and raw feature recipe do not match")
        self.models = joblib.load(model_dir / "model.joblib")
        self.threshold = self.metadata["threshold"]
        self.margin = self.metadata["margin"]

    def scores(self, features):
        batch = np.asarray(features, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch[None, :]
        if batch.shape[1] != 516 or not np.isfinite(batch).all():
            raise ValueError("Invalid raw feature matrix")
        all_scores = []
        for model in self.models:
            current = model.predict_proba(batch)
            ordered = np.empty((len(batch), len(CLASSES)), dtype=np.float64)
            for index, name in enumerate(CLASSES):
                ordered[:, index] = current[:, list(model.classes_).index(name)]
            all_scores.append(ordered)
        return np.mean(all_scores, axis=0)

    def decide(self, scores):
        scores = np.asarray(scores)
        indexes = np.argsort(scores)
        winner, runner_up = int(indexes[-1]), int(indexes[-2])
        accepted = winner != 3 and float(scores[winner]) >= self.threshold and float(scores[winner]-scores[runner_up]) >= self.margin
        return {
            "label": CLASSES[winner] if accepted else "UNKNOWN",
            "candidate": CLASSES[winner], "accepted": accepted,
            "score": float(scores[winner]), "scores": dict(zip(CLASSES, map(float, scores))),
            "uncalibrated_scores": True,
        }

    def classify_samples(self, samples):
        features = extract(samples)
        if not np.any(features):
            return {"label": "UNKNOWN", "candidate": "OTHER", "accepted": False,
                    "score": 1.0, "reason": "zero_or_dc_only_signal", "uncalibrated_scores": True}
        return self.decide(self.scores(features)[0])

    def classify_file(self, path, *, first_sample=0, fmt="cf32", sample_rate=SAMPLE_RATE):
        samples = read_window(path, first_sample, fmt=fmt, sample_rate=sample_rate)
        result = self.classify_samples(samples)
        result.update({"window_start_sample": first_sample, "window_samples": WINDOW_SAMPLES,
                       "sample_rate_hz": sample_rate,
                       "input": "raw_complex_iq", "protocol_packet_version_decoded": False,
                       "label_definition": "nominal OcuSync family of labelled RFUAV source transmitters"})
        return result
