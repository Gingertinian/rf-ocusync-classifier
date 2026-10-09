import json
import time

from detector_model import Detector
from rf_baseline import ROOT, sha256_file


def main():
    detector = Detector()
    previous = json.loads((ROOT / "detector" / "results.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    predictions = []
    for item in previous["predictions"]:
        path = ROOT / item["local_path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Test image changed since training")
        actual = detector.classify_path(path)
        if actual["accepted"] != item["accepted_known_equipment"] or actual["label"] != item["emitted_label"]:
            raise ValueError("Saved model did not reproduce its recorded prediction")
        if abs(actual["uncalibrated_softmax_score"] - item["uncalibrated_softmax_score"]) > 1e-5:
            raise ValueError("Saved model did not reproduce its recorded score")
        predictions.append({"source_path": item["source_path"], "actual_equipment": item["label"], **actual})
    result = {
        "predictions_reproduced": len(predictions),
        "head_sha256": sha256_file(ROOT / "detector" / "model_head.pt"),
        "seconds": time.perf_counter() - started,
        "all_predictions_match": True,
        "predictions": predictions,
    }
    (ROOT / "detector" / "replay_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
