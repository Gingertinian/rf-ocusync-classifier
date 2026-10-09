import argparse
import json
from pathlib import Path

from raw_features import SAMPLE_RATE, WINDOW_SAMPLES
from capture_io import open_capture
from raw_model import RawClassifier


def replay(path, *, max_windows=26, fmt="cf32", sample_rate=SAMPLE_RATE):
    if sample_rate != SAMPLE_RATE or not 1 <= max_windows <= 1000:
        raise ValueError("Use the documented 100-MS/s profile and 1-1000 windows")
    capture = open_capture(path, format=fmt, sample_rate=sample_rate)
    classifier = RawClassifier()
    total = min(max_windows, capture.sample_count // WINDOW_SAMPLES)
    if total < 1:
        raise ValueError("Recording has no complete 10-ms window")
    for index in range(total):
        result = classifier.classify_file(path, first_sample=index*WINDOW_SAMPLES, fmt=fmt, sample_rate=sample_rate)
        yield {"virtual_receiver": "recorded_USRP_X310_profile", "hardware_connected": False,
               "time_seconds": index*WINDOW_SAMPLES/sample_rate, **result}


def main():
    parser = argparse.ArgumentParser(description="Offline virtual replay of recorded SDR I/Q; no RF transmission or reception.")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--max-windows", type=int, default=26)
    parser.add_argument("--format", choices=("cf32", "ci16"), default="cf32")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args()
    for result in replay(args.capture, max_windows=args.max_windows, fmt=args.format, sample_rate=args.sample_rate):
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
