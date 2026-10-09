import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from adapted_classifier import AdaptedClassifier
from rf_baseline import CLASSES, REPO, ROOT, download, list_tree, select_samples, sha256_file


SEED = 20261002
KNOWN = set(CLASSES)
SAMPLES_PER_UNKNOWN_CLASS = 2
SCORE_THRESHOLD = 0.90


def run():
    original = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))
    revision = original["revision"]
    directories = list_tree("ImageSet-AllDrones-MatlabPipeline/valid", revision)
    unknown_dirs = sorted(item["path"] for item in directories if item["type"] == "directory" and item["path"].rsplit("/", 1)[1] not in KNOWN)
    records = []
    for index, directory in enumerate(unknown_dirs):
        chosen = select_samples(list_tree(directory, revision), SAMPLES_PER_UNKNOWN_CLASS, SEED + index)
        for item in chosen:
            records.append({
                "label": directory.rsplit("/", 1)[1],
                "source_path": item["path"],
                "source_size": item["size"],
                "local_path": f"open_set/samples/{index}/" + item["path"].rsplit("/", 1)[1],
                "url": f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/" + urllib.parse.quote(item["path"], safe="/"),
            })
    destination = ROOT / "open_set"
    destination.mkdir(exist_ok=True)
    (destination / "selection.json").write_text(json.dumps({"seed": SEED, "score_threshold_fixed_before_test": SCORE_THRESHOLD, "samples": records}, indent=2), encoding="utf-8")
    print(f"Open-set check fixed: {len(records)} images from {len(unknown_dirs)} other equipment classes.", flush=True)
    def fetch(record):
        path = download(record["url"], ROOT / record["local_path"])
        if path.stat().st_size != record["source_size"]:
            raise ValueError("Image size mismatch")
        record["sha256"] = sha256_file(path)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(fetch, records))
    classifier = AdaptedClassifier()
    predictions = []
    for index, record in enumerate(records, 1):
        prediction = classifier.classify(ROOT / record["local_path"])
        prediction.update(record)
        prediction["would_be_false_positive_with_score_only_gate"] = prediction["uncalibrated_softmax_score"] >= SCORE_THRESHOLD
        predictions.append(prediction)
        print(f"{index}/{len(records)} {record['label']}: score {prediction['uncalibrated_softmax_score']:.3f}", flush=True)
    false_positives = sum(item["would_be_false_positive_with_score_only_gate"] for item in predictions)
    result = {
        "test": "Unknown-equipment samples, not Wi-Fi/background, not independent receiver sessions",
        "sample_count": len(records),
        "unknown_class_count": len(unknown_dirs),
        "score_only_threshold": SCORE_THRESHOLD,
        "score_only_false_positives": false_positives,
        "score_only_false_positive_fraction": false_positives / len(records),
        "production_ready": False,
        "purpose": "Audit whether a high closed-set equipment score alone is sufficient evidence. It is not an open-set validation.",
        "predictions": predictions,
    }
    (destination / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Score-only false positives: {false_positives}/{len(records)}", flush=True)


if __name__ == "__main__":
    run()
