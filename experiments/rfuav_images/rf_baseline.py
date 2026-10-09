import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = "kitofrank/RFUAV"
UPSTREAM = "https://github.com/kitoweeknd/RFUAV"
MODEL_PATH = "weight/exp1/ResNet18.pth"
MODEL_SHA256 = "ae3859b0332f705215713b9617a38e9448b2f7300cbb5c14e8cd3f2e52ef03af"
CLASSES = ("DJI AVATA2", "DJI FPV COMBO", "DJI MAVIC3 PRO", "DJI MINI3", "DJI MINI4 PRO")
LINK_FAMILY = ("O4", "O3", "O3+", "O2", "O4")
SOURCE_URLS = (
    "https://www.dji.com/avata-2/specs",
    "https://www.dji.com/newsroom/news/dji-reinvents-the-drone-flying-experience-with-the-dji-fpv",
    "https://www.dji.com/mavic-3-pro/specs",
    "https://www.dji.com/mini-3/specs",
    "https://www.dji.com/mini-4-pro/specs",
)
LIMITS = (
    "Public RFUAV ResNet18 checkpoint.",
    "Offline spectrogram images; no physical receiver or live RF test.",
    "Five known DJI equipment classes; background, Wi-Fi and unknown devices are not validated.",
    "The link family is looked up from the predicted equipment model, not decoded or independently classified.",
    "Public validation folder sampling does not establish independence from checkpoint training or recording sessions.",
    "Softmax scores are uncalibrated and are not measured correctness probabilities.",
)


def request(url):
    return urllib.request.Request(url, headers={"User-Agent": "rf-offline-evaluation/1.0"})


def json_get(url):
    with urllib.request.urlopen(request(url), timeout=45) as response:
        return json.load(response)


def list_tree(path, revision):
    quoted = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{quoted}?limit=1000"
    entries = []
    while url:
        with urllib.request.urlopen(request(url), timeout=45) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link", "")
        next_link = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = next_link.group(1) if next_link else None
    return entries


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_sha256=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (expected_sha256 is None or sha256_file(path) == expected_sha256):
        return path
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request(url), timeout=45) as response, partial.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            if expected_sha256 and sha256_file(partial) != expected_sha256:
                raise ValueError(f"Hash mismatch: {path.name}")
            partial.replace(path)
            return path
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def select_samples(entries, count, seed):
    candidates = sorted(
        (entry for entry in entries if entry["type"] == "file" and entry["path"].lower().endswith((".png", ".jpg", ".jpeg"))),
        key=lambda entry: entry["path"],
    )
    if count > len(candidates):
        raise ValueError("Not enough samples in this class")
    return random.Random(seed).sample(candidates, count)


def prepare(count, seed):
    revision = json_get(f"https://huggingface.co/api/datasets/{REPO}")["sha"]
    records = []
    for index, label in enumerate(CLASSES):
        entries = list_tree(f"ImageSet-AllDrones-MatlabPipeline/valid/{label}", revision)
        selected = select_samples(entries, count, seed + index)
        for entry in selected:
            relative = f"samples/{index}/{Path(entry['path']).name}"
            records.append({
                "label_index": index,
                "label": label,
                "source_path": entry["path"],
                "source_size": entry["size"],
                "local_path": relative,
                "url": f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/" + urllib.parse.quote(entry["path"], safe="/"),
            })
    manifest = {"repository": REPO, "revision": revision, "selection_seed": seed, "samples_per_class": count, "samples": records}
    (ROOT / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Fixed selection: {len(records)} images, revision {revision}", flush=True)
    model_url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{MODEL_PATH}"
    download(model_url, ROOT / "weights" / "ResNet18.pth", MODEL_SHA256)
    def get_sample(record):
        path = download(record["url"], ROOT / record["local_path"])
        if path.stat().st_size != record["source_size"]:
            raise ValueError(f"Wrong sample size: {path.name}")
        record["sha256"] = sha256_file(path)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(get_sample, records))
    (ROOT / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Checkpoint hash verified; samples downloaded.", flush=True)


def load_model():
    import torch
    from torchvision import models, transforms

    checkpoint = ROOT / "weights" / "ResNet18.pth"
    if sha256_file(checkpoint) != MODEL_SHA256:
        raise ValueError("Unexpected checkpoint hash")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tuple(state["fc.weight"].shape) != (len(CLASSES), 512):
        raise ValueError("Checkpoint class count does not match the published configuration")
    model = models.resnet18(weights=None, num_classes=len(CLASSES))
    model.load_state_dict(state, strict=True)
    model.eval()
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return model, transform


def predict(model, transform, path):
    import torch
    from PIL import Image

    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        scores = logits.softmax(1)[0]
    elapsed = 1000 * (time.perf_counter() - start)
    index = int(scores.argmax())
    return {
        "predicted_index": index,
        "predicted_equipment": CLASSES[index],
        "associated_link_family_lookup": LINK_FAMILY[index],
        "uncalibrated_softmax_score": float(scores[index]),
        "cpu_forward_ms": elapsed,
    }


def write_html(result):
    rows = []
    for record in result["predictions"]:
        actual = html.escape(record["label"])
        predicted = html.escape(record["predicted_equipment"])
        relative = html.escape(record["local_path"], quote=True)
        status = "OK" if record["correct_equipment"] else "ERROR"
        rows.append(f'<tr><td><a href="{relative}">{actual}</a></td><td>{predicted}</td><td>{record["associated_link_family_lookup"]}</td><td>{status}</td></tr>')
    limits = "".join(f"<li>{html.escape(item)}</li>" for item in LIMITS)
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RFUAV offline baseline</title>
<style>body{{max-width:900px;margin:40px auto;padding:0 20px;font:16px/1.5 system-ui;color:#222}}h1{{font-size:26px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:9px 8px}}a{{color:#225789}}li{{margin:6px 0}}small{{color:#555}}</style>
<h1>RFUAV offline baseline</h1>
<p>Public ResNet18 checkpoint on {result['sample_count']} fixed spectrogram samples: {result['correct_equipment_count']} correct equipment labels.</p>
<p>Link-family names below are a lookup from equipment specifications. They are not independently decoded or measured protocol versions.</p>
<table><thead><tr><th>Dataset label / recording image</th><th>Model prediction</th><th>Associated link</th><th>Equipment check</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Scope</h2><ul>{limits}</ul>
<p><a href="{UPSTREAM}">RFUAV source</a> · <a href="https://huggingface.co/datasets/{REPO}">Dataset and checkpoint</a> · <a href="results.json">Run details</a></p>
<small>Model and data: RFUAV authors, Apache-2.0. Local inference wrapper: Jeronimo Munoz Larreta.</small></html>'''
    (ROOT / "demo.html").write_text(document, encoding="utf-8")


def evaluate():
    import torch
    import torchvision

    manifest = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))
    model, transform = load_model()
    predictions = []
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    for position, record in enumerate(manifest["samples"], 1):
        path = ROOT / record["local_path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Sample changed since preparation")
        prediction = predict(model, transform, path)
        prediction.update(record)
        prediction["correct_equipment"] = prediction["predicted_index"] == record["label_index"]
        confusion[record["label_index"]][prediction["predicted_index"]] += 1
        predictions.append(prediction)
        print(f"{position:02d}/{len(manifest['samples'])} {record['label']} -> {prediction['predicted_equipment']}", flush=True)
    correct = sum(record["correct_equipment"] for record in predictions)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_source": UPSTREAM,
        "checkpoint_sha256": MODEL_SHA256,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "selection": {key: value for key, value in manifest.items() if key != "samples"},
        "classes": CLASSES,
        "equipment_to_link_family_lookup": dict(zip(CLASSES, LINK_FAMILY)),
        "link_family_sources": dict(zip(CLASSES, SOURCE_URLS)),
        "sample_count": len(predictions),
        "correct_equipment_count": correct,
        "equipment_accuracy_on_selected_samples": correct / len(predictions),
        "confusion_rows_actual_columns_predicted": confusion,
        "limitations": LIMITS,
        "predictions": predictions,
    }
    (ROOT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ("label", "predicted_equipment", "associated_link_family_lookup", "correct_equipment", "uncalibrated_softmax_score", "cpu_forward_ms", "local_path")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    write_html(result)
    print(f"Result: {correct}/{len(predictions)} equipment labels correct. Link family is a lookup, not a validated protocol classifier.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Independent offline reproduction using the public RFUAV checkpoint.")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--samples-per-class", type=int, default=10)
    preparation.add_argument("--seed", type=int, default=20260930)
    commands.add_parser("evaluate")
    inference = commands.add_parser("infer")
    inference.add_argument("image", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.samples_per_class < 1:
            parser.error("samples-per-class must be positive")
        prepare(args.samples_per_class, args.seed)
    elif args.command == "evaluate":
        evaluate()
    else:
        model, transform = load_model()
        print(json.dumps({"prediction": predict(model, transform, args.image), "limitations": LIMITS}, indent=2))


if __name__ == "__main__":
    main()
