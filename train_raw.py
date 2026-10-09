import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from raw_features import CLASSES, RECIPE, SAMPLE_RATE, WINDOW_SAMPLES, extract, intervals, read_window
from raw_model import ROOT, RawClassifier, digest

from capture_io import open_capture
from perturb_capture import transform


SEED = 20260930
THRESHOLD, MARGIN = 0.65, 0.15
TRAIN_CONDITIONS = ({}, {"noise_ratio_db": 30}, {"noise_ratio_db": 20}, {"noise_ratio_db": 10},
                    {"frequency_offset_hz": 500000}, {"frequency_offset_hz": -500000},
                    {"iq_gain_ratio": 0.8}, {"echo_gain": 0.1, "echo_delay_samples": 50})


def synthetic(seed, bandwidth=None, tone=False):
    rng = np.random.default_rng(seed)
    if tone:
        frequency = rng.uniform(-35e6, 35e6)
        values = np.exp(2j*np.pi*frequency*np.arange(WINDOW_SAMPLES)/SAMPLE_RATE)
        values += 0.01*(rng.standard_normal(WINDOW_SAMPLES)+1j*rng.standard_normal(WINDOW_SAMPLES))
    else:
        values = rng.standard_normal(WINDOW_SAMPLES)+1j*rng.standard_normal(WINDOW_SAMPLES)
        if bandwidth:
            transformed = np.fft.fft(values)
            frequencies = np.fft.fftfreq(WINDOW_SAMPLES, 1/SAMPLE_RATE)
            center = rng.uniform(-25e6, 25e6)
            transformed[np.abs(frequencies-center) > bandwidth/2] = 0
            values = np.fft.ifft(transformed)
    return values.astype(np.complex64)


def catalog(manifest):
    records = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    if len(records) != 3 or {record["family"] for record in records} != {"OCU2", "OCU3", "OCU4"}:
        raise ValueError("Provide exactly one recording per nominal family for this recipe")
    paths = set()
    for record in records:
        path = (manifest.parent / record["path"]).resolve()
        if path in paths:
            raise ValueError("A recording cannot have two family labels")
        paths.add(path)
        capture = open_capture(path, format="cf32", sample_rate=SAMPLE_RATE)
        checksum = digest(capture.path)
        if record.get("sha256") and record["sha256"] != checksum:
            raise ValueError("Source recording hash mismatch")
        record["path"] = str(path)
        record["sha256"] = checksum
        record["sample_count"] = capture.sample_count
        record["intervals"] = intervals(capture.sample_count)
    return records


def main():
    parser = argparse.ArgumentParser(description="Fit the raw-I/Q ensemble on three labelled source recordings")
    parser.add_argument("manifest", type=Path, help="JSON sources with path, family and optional sha256")
    parser.add_argument("--output", type=Path, default=ROOT / "training-output")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be a new or empty directory")
    sources = catalog(args.manifest.resolve())
    output.mkdir(parents=True, exist_ok=True)
    selection = {"seed": SEED, "threshold": THRESHOLD, "margin": MARGIN, "recipe": RECIPE,
                 "split_type": "temporal_disjoint_not_independent_recordings", "sources": sources,
                 "external_captures_used_in_training": False}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("Fixed raw train/validation/test intervals; three positive nominal families.", flush=True)
    started = time.perf_counter()
    features, labels = [], []
    heldout = {"validation": [], "test": []}
    for source_index, source in enumerate(sources):
        for split in ("train", "validation", "test"):
            for index in source["intervals"][split]:
                raw = read_window(source["path"], index * WINDOW_SAMPLES)
                if split == "train":
                    for condition_index, condition in enumerate(TRAIN_CONDITIONS):
                        changed, _ = transform(raw, SAMPLE_RATE, seed=SEED+source_index*1000+index*10+condition_index, **condition)
                        features.append(extract(changed)); labels.append(source["family"])
                else:
                    heldout[split].append({"source": source["path"], "first_sample": index*WINDOW_SAMPLES,
                                           "family": source["family"], "features": extract(raw)})
        print(f"Extracted {source['family']} raw features.", flush=True)
    for index in range(72):
        bandwidth = (None, 10e6, 20e6, 40e6)[index % 4]
        noise = synthetic(SEED + 100000 + index, bandwidth, tone=index % 6 == 5)
        features.append(extract(noise)); labels.append("OTHER")
    x, y = np.asarray(features, dtype=np.float32), np.asarray(labels)
    models = [
        ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, max_features=0.6, class_weight="balanced", n_jobs=4, random_state=SEED),
        RandomForestClassifier(n_estimators=350, min_samples_leaf=2, max_features=0.6, class_weight="balanced", n_jobs=4, random_state=SEED+1),
        make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=SEED)),
    ]
    for model in models:
        model.fit(x, y)
    joblib.dump(models, output / "model.joblib", compress=3)
    metadata = {"classes": CLASSES, "recipe": RECIPE, "threshold": THRESHOLD, "margin": MARGIN,
                "model_sha256": digest(output / "model.joblib"), "training_rows": len(y),
                "source_recordings": len(sources), "hardware_differentiation_and_family_are_confounded": True,
                "mode": "offline_recorded_iq_poc", "cloud_spend_usd": 0}
    (output / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    classifier = RawClassifier(output)
    result = {"metadata": metadata, "selection": selection, "splits": {}, "seconds": None}
    for split, items in heldout.items():
        scored = classifier.scores([item["features"] for item in items])
        predictions = [{**{key: value for key, value in item.items() if key != "features"}, **classifier.decide(row)} for item, row in zip(items, scored)]
        correct = sum(item["label"] == item["family"] for item in predictions)
        result["splits"][split] = {"total": len(predictions), "correct": correct,
                                   "wrong": sum(item["accepted"] and item["label"] != item["family"] for item in predictions),
                                   "abstained": sum(not item["accepted"] for item in predictions), "predictions": predictions}
        print(f"{split}: {correct}/{len(predictions)} correct nominal raw-family labels.", flush=True)
    result["seconds"] = time.perf_counter()-started
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"test": {key: value for key, value in result['splits']['test'].items() if key != 'predictions'}, "seconds": result['seconds']}), flush=True)


if __name__ == "__main__":
    main()
