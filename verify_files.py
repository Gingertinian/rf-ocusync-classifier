import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    expected = json.loads((ROOT / "checksums.json").read_text(encoding="utf-8"))
    for name, checksum in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Missing or unsafe manifest path: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("Checksum mismatch: " + name)
    print(json.dumps({"verified_files": len(expected)}))


if __name__ == "__main__":
    main()
