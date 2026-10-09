import argparse
import json
from pathlib import Path

from raw_features import SAMPLE_RATE
from raw_model import RawClassifier


def main():
    parser = argparse.ArgumentParser(description="OcuSync nominal-family raw-I/Q proof of concept with unknown rejection.")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--format", choices=("cf32", "ci16"), default="cf32")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help="Default documented virtual receiver profile: USRP X310 at 100 MS/s")
    args = parser.parse_args()
    try:
        result = RawClassifier().classify_file(args.capture, first_sample=args.start_sample,
                                             fmt=args.format, sample_rate=args.sample_rate)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "rejected", "error_type": type(error).__name__}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
