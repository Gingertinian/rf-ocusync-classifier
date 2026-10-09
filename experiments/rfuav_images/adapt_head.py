import copy
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from torch import nn

from rf_baseline import CLASSES, LIMITS, LINK_FAMILY, MODEL_SHA256, REPO, ROOT, download, list_tree, load_model, select_samples, sha256_file


TRAIN_PER_CLASS = 40
TEST_PER_CLASS = 20
SEED = 20261001
EPOCHS = 100


def prepare_adaptation():
    previous = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))
    revision = previous["revision"]
    previous_paths = {record["source_path"] for record in previous["samples"]}
    records = []
    for split, folder, count in (("train", "train", TRAIN_PER_CLASS), ("test", "valid", TEST_PER_CLASS)):
        for label_index, label in enumerate(CLASSES):
            candidates = list_tree(f"ImageSet-AllDrones-MatlabPipeline/{folder}/{label}", revision)
            candidates = [item for item in candidates if item["path"] not in previous_paths]
            selected = select_samples(candidates, count, SEED + label_index + (100 if split == "test" else 0))
            for item in selected:
                filename = item["path"].rsplit("/", 1)[1]
                records.append({
                    "split": split,
                    "label_index": label_index,
                    "label": label,
                    "source_path": item["path"],
                    "source_size": item["size"],
                    "local_path": f"adaptation/samples/{split}/{label_index}/{filename}",
                    "url": f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/" + urllib.parse.quote(item["path"], safe="/"),
                })
    manifest = {
        "revision": revision,
        "seed": SEED,
        "train_per_class": TRAIN_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "epochs": EPOCHS,
        "frozen_backbone": True,
        "input_size": 640,
        "image_normalization": "ToTensor only, matching the upstream classifier configuration",
        "feature_normalization": "Training-set mean and standard deviation, minimum standard deviation 0.05",
        "samples": records,
    }
    destination = ROOT / "adaptation"
    destination.mkdir(exist_ok=True)
    (destination / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Adaptation selection fixed: 200 training images; 100 additional test images.", flush=True)
    def get_sample(record):
        path = download(record["url"], ROOT / record["local_path"])
        if path.stat().st_size != record["source_size"]:
            raise ValueError("Downloaded image size mismatch")
        record["sha256"] = sha256_file(path)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(get_sample, records))
    train_hashes = {record["sha256"] for record in records if record["split"] == "train"}
    test_hashes = {record["sha256"] for record in records if record["split"] == "test"}
    old_hashes = {record["sha256"] for record in previous["samples"]}
    if train_hashes & test_hashes or test_hashes & old_hashes:
        raise ValueError("Byte-identical samples overlap across evaluation splits")
    manifest["train_test_exact_hash_overlap"] = 0
    (destination / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def extract(backbone, transform, records):
    vectors, targets = [], []
    for start in range(0, len(records), 4):
        subset = records[start:start + 4]
        batch = []
        for record in subset:
            path = ROOT / record["local_path"]
            if sha256_file(path) != record["sha256"]:
                raise ValueError("Sample hash mismatch")
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
            targets.append(record["label_index"])
        with torch.inference_mode():
            vectors.append(backbone(torch.stack(batch)))
        print(f"Features: {min(start + 4, len(records))}/{len(records)}", flush=True)
    return torch.cat(vectors).clone(), torch.tensor(targets)


def run():
    torch.manual_seed(SEED)
    manifest = prepare_adaptation()
    model, transform = load_model()
    original_head = copy.deepcopy(model.fc)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train = [record for record in manifest["samples"] if record["split"] == "train"]
    test = [record for record in manifest["samples"] if record["split"] == "test"]
    started = time.perf_counter()
    x_train, y_train = extract(model, transform, train)
    x_test, y_test = extract(model, transform, test)
    feature_seconds = time.perf_counter() - started
    mean = x_train.mean(0)
    scale = x_train.std(0).clamp(min=0.05)
    norm_train = (x_train - mean) / scale
    norm_test = (x_test - mean) / scale
    head = nn.Linear(512, len(CLASSES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    started = time.perf_counter()
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(head(norm_train), y_train)
        loss.backward()
        optimizer.step()
    fit_seconds = time.perf_counter() - started
    head.eval()
    with torch.inference_mode():
        baseline_indices = original_head(x_test).argmax(1)
        adapted_scores = head(norm_test).softmax(1)
        adapted_indices = adapted_scores.argmax(1)
        train_correct = int((head(norm_train).argmax(1) == y_train).sum())
    baseline_correct = int((baseline_indices == y_test).sum())
    adapted_correct = int((adapted_indices == y_test).sum())
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    predictions = []
    for record, baseline_index, adapted_index, scores in zip(test, baseline_indices, adapted_indices, adapted_scores):
        actual = record["label_index"]
        predicted = int(adapted_index)
        confusion[actual][predicted] += 1
        predictions.append({
            **record,
            "baseline_equipment": CLASSES[int(baseline_index)],
            "adapted_equipment": CLASSES[predicted],
            "associated_link_family_lookup": LINK_FAMILY[predicted],
            "uncalibrated_softmax_score": float(scores[predicted]),
            "correct_equipment": actual == predicted,
        })
    result = {
        "method": "Fixed public RFUAV ResNet18 feature extractor; new linear head fitted locally",
        "base_checkpoint_sha256": MODEL_SHA256,
        "train_count": len(train),
        "test_count": len(test),
        "train_correct": train_correct,
        "baseline_correct_on_same_test": baseline_correct,
        "adapted_correct_on_same_test": adapted_correct,
        "equipment_accuracy": adapted_correct / len(test),
        "feature_extraction_cpu_seconds": feature_seconds,
        "head_fit_cpu_seconds": fit_seconds,
        "cloud_spend_usd": 0,
        "exact_train_test_hash_overlap": 0,
        "epochs_fixed_before_test": EPOCHS,
        "classes": CLASSES,
        "confusion_rows_actual_columns_predicted": confusion,
        "limitations": [*LIMITS, "Training and test images may be correlated slices of the same physical recording; no session-level generalization is claimed."],
        "predictions": predictions,
    }
    destination = ROOT / "adaptation"
    torch.save({"head": head.state_dict(), "feature_mean": mean, "feature_scale": scale, "classes": CLASSES, "base_checkpoint_sha256": MODEL_SHA256}, destination / "linear_head.pt")
    (destination / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("predictions", "limitations", "confusion_rows_actual_columns_predicted")}, indent=2), flush=True)


if __name__ == "__main__":
    run()
