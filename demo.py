import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Recorded raw-I/Q OcuSync-family proof of concept.")
    parser.add_argument("--setup", action="store_true", help="Install the pinned local CPU dependencies")
    args = parser.parse_args()
    if args.setup:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
        return
    from raw_model import RawClassifier
    cases = json.loads((ROOT / "examples.json").read_text(encoding="utf-8"))
    classifier = RawClassifier()
    correct = 0
    for case in cases:
        path = ROOT / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError("An included sample failed its checksum")
        # Expected labels are checked after prediction, never passed to the model.
        result = classifier.classify_file(path)
        passed = result["label"] == case["expected"]
        correct += passed
        print(json.dumps({"file": case["file"], "expected": case["expected"],
                          "result": result["label"], "score": round(result["score"], 4),
                          "passed": passed}), flush=True)
    print(json.dumps({"passed": correct, "cases": len(cases), "live_hardware_used": False,
                      "input": "actual_recorded_iq", "scope": "capture-specific basic nominal-family PoC"}))
    if correct != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
