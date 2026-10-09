import json
import random
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn

from adapt_head import extract
from rf_baseline import CLASSES, LIMITS, MODEL_SHA256, REPO, ROOT, download, list_tree, load_model, select_samples, sha256_file


SEED = 20261003
DETECTOR_CLASSES = (*CLASSES, "OTHER_SIGNAL")
EPOCHS = 150
MIN_SCORE = 0.90


def run():
    torch.manual_seed(SEED)
    initial = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))
    adaptation = json.loads((ROOT / "adaptation" / "selection.json").read_text(encoding="utf-8"))
    old_unknown = json.loads((ROOT / "open_set" / "selection.json").read_text(encoding="utf-8"))
    revision = initial["revision"]
    seen_paths = {item["source_path"] for item in [*initial["samples"], *adaptation["samples"], *old_unknown["samples"]]}
    known_train = [dict(item) for item in adaptation["samples"] if item["split"] == "train"]
    records = known_train.copy()
    unknown_dirs = sorted(item["path"].rsplit("/", 1)[1] for item in list_tree("ImageSet-AllDrones-MatlabPipeline/train", revision) if item["type"] == "directory" and item["path"].rsplit("/", 1)[1] not in CLASSES)
    negative_train_classes = set(random.Random(SEED).sample(unknown_dirs, len(unknown_dirs) // 2))
    for folder, split, names in (("train", "train", sorted(negative_train_classes)), ("valid", "test_unknown", unknown_dirs), ("valid", "test_known", list(CLASSES))):
        for index, name in enumerate(names):
            count = 5 if split == "train" else (2 if split == "test_unknown" else 10)
            entries = list_tree(f"ImageSet-AllDrones-MatlabPipeline/{folder}/{name}", revision)
            entries = [item for item in entries if item["path"] not in seen_paths]
            selected = select_samples(entries, count, SEED + index + (0 if split == "train" else 100 if split == "test_unknown" else 200))
            for item in selected:
                filename = item["path"].rsplit("/", 1)[1]
                records.append({
                    "split": split,
                    "label_index": CLASSES.index(name) if name in CLASSES else len(CLASSES),
                    "label": name,
                    "source_path": item["path"],
                    "source_size": item["size"],
                    "local_path": f"detector/samples/{split}/{index}/{filename}",
                    "url": f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/" + urllib.parse.quote(item["path"], safe="/"),
                    "equipment_type_seen_in_negative_training": name in negative_train_classes,
                })
    destination = ROOT / "detector"
    destination.mkdir(exist_ok=True)
    manifest = {
        "revision": revision,
        "seed": SEED,
        "unknown_types_training": sorted(negative_train_classes),
        "unknown_types_not_used_in_training": sorted(set(unknown_dirs) - negative_train_classes),
        "epochs": EPOCHS,
        "score_threshold_fixed_before_test": MIN_SCORE,
        "classes": DETECTOR_CLASSES,
        "samples": records,
    }
    (destination / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Fixed detector run: {len(known_train)} known train; {len(negative_train_classes)*5} negative train; {len(records)-len(known_train)-len(negative_train_classes)*5} fresh test.", flush=True)
    def fetch(record):
        path = download(record["url"], ROOT / record["local_path"])
        if path.stat().st_size != record["source_size"]:
            raise ValueError("Sample size mismatch")
        record["sha256"] = sha256_file(path)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(fetch, records))
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] != "train"]
    if {record["sha256"] for record in train} & {record["sha256"] for record in test}:
        raise ValueError("Training/test images overlap byte for byte")
    (destination / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model, transform = load_model()
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    x_train, y_train = extract(model, transform, train)
    x_test, y_test = extract(model, transform, test)
    feature_seconds = time.perf_counter() - started
    mean = x_train.mean(0)
    scale = x_train.std(0).clamp(min=0.05)
    train_x = (x_train - mean) / scale
    test_x = (x_test - mean) / scale
    head = nn.Linear(512, len(DETECTOR_CLASSES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.005, weight_decay=0.05)
    counts = torch.bincount(y_train, minlength=len(DETECTOR_CLASSES)).float()
    criterion = nn.CrossEntropyLoss(weight=counts.reciprocal())
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train_x), y_train)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.inference_mode():
        scores = head(test_x).softmax(1)
        train_accuracy = float((head(train_x).argmax(1) == y_train).float().mean())
    predictions = []
    for record, row in zip(test, scores):
        index = int(row.argmax())
        score = float(row[index])
        accepted = index != len(CLASSES) and score >= MIN_SCORE
        expected_known = record["split"] == "test_known"
        predictions.append({
            **record,
            "candidate": DETECTOR_CLASSES[index],
            "uncalibrated_softmax_score": score,
            "accepted_known_equipment": accepted,
            "emitted_label": DETECTOR_CLASSES[index] if accepted else "UNKNOWN_OR_INSUFFICIENT_EVIDENCE",
            "correct_known_label": expected_known and accepted and index == record["label_index"],
            "false_positive": not expected_known and accepted,
        })
    positives = [item for item in predictions if item["split"] == "test_known"]
    negatives = [item for item in predictions if item["split"] == "test_unknown"]
    heldout_negative = [item for item in negatives if not item["equipment_type_seen_in_negative_training"]]
    report = {
        "method": "Public RFUAV ResNet18 features; own six-class linear head with OTHER_SIGNAL class and abstention",
        "train_known_images": len(known_train),
        "train_negative_images": len(train)-len(known_train),
        "train_equipment_types_for_other_signal": len(negative_train_classes),
        "train_accuracy": train_accuracy,
        "test_known_images": len(positives),
        "test_known_accepted_correct": sum(item["correct_known_label"] for item in positives),
        "test_known_accepted_wrong": sum(item["accepted_known_equipment"] and not item["correct_known_label"] for item in positives),
        "test_known_abstained": sum(not item["accepted_known_equipment"] for item in positives),
        "test_other_images": len(negatives),
        "test_other_false_positives": sum(item["false_positive"] for item in negatives),
        "test_unseen_other_types": len(set(item["label"] for item in heldout_negative)),
        "test_unseen_other_images": len(heldout_negative),
        "test_unseen_other_false_positives": sum(item["false_positive"] for item in heldout_negative),
        "threshold": MIN_SCORE,
        "feature_extraction_cpu_seconds": feature_seconds,
        "cloud_spend_usd": 0,
        "raw_iq_classifier_validated": False,
        "independent_receiver_session_validated": False,
        "limitations": [*LIMITS, "This OTHER_SIGNAL class includes other equipment types, not measured Wi-Fi-only or noise-only background recordings.", "All examples are still from the same dataset; independence across real physical recording sessions is not established."],
        "predictions": predictions,
    }
    torch.save({"head": head.state_dict(), "feature_mean": mean, "feature_scale": scale, "classes": DETECTOR_CLASSES, "threshold": MIN_SCORE, "base_checkpoint_sha256": MODEL_SHA256}, destination / "model_head.pt")
    (destination / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("limitations", "predictions")}, indent=2), flush=True)


if __name__ == "__main__":
    run()
