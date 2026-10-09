import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import stft, windows

from capture_io import CaptureError, open_capture


RECIPE = {
    "name": "numeric_rf_v1", "window": "symmetric_hamming", "window_samples": 1024,
    "fft_points": 1024, "hop_samples": 512, "boundary": "none", "padding": False,
    "frequency_bins": 128, "time_bins": 128, "power_reference": "relative_to_maximum",
    "floor_below_peak_db": 60, "image_rendering": False,
    "legacy_image_checkpoint_compatible": False,
}


def representation(samples, sample_rate):
    if samples.ndim != 1 or samples.dtype.kind != "c" or not np.isfinite(samples).all():
        raise CaptureError("Expected a finite complex signal vector")
    if not 1024 <= len(samples) <= 10_000_000:
        raise CaptureError("Numeric representation requires between 1024 and ten million complex samples")
    if not np.isfinite(sample_rate) or not 0 < sample_rate <= 200e6:
        raise CaptureError("Unexpected sample rate")
    _, _, transformed = stft(samples, fs=sample_rate, window=windows.hamming(1024),
                             nperseg=1024, noverlap=512, nfft=1024,
                             return_onesided=False, boundary=None, padded=False)
    magnitude = np.abs(np.fft.fftshift(transformed, axes=0))
    del transformed
    peak = float(magnitude.max())
    if peak == 0:
        return np.zeros((128, 128), dtype=np.float32)
    magnitude /= peak
    np.maximum(magnitude, 1e-3, out=magnitude)
    np.log10(magnitude, out=magnitude)
    magnitude *= 20
    frequency_reduced = magnitude.reshape(128, 8, magnitude.shape[1]).mean(1)
    del magnitude
    edges = np.linspace(0, frequency_reduced.shape[1], 129).astype(int)
    output = np.empty((128, 128), dtype=np.float32)
    for index, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
        if last == first:
            output[:, index] = frequency_reduced[:, min(first, frequency_reduced.shape[1] - 1)]
        else:
            output[:, index] = frequency_reduced[:, first:last].mean(1)
    return (output + 60) / 60


def read_segment(capture, start_seconds=0, duration_seconds=0.1):
    if not np.isfinite(start_seconds) or start_seconds < 0 or not np.isfinite(duration_seconds) or not 0 < duration_seconds <= 0.1:
        raise CaptureError("Select a finite nonnegative start and duration in (0,100ms]")
    count = int(round(duration_seconds * capture.metadata.sample_rate))
    first = int(round(start_seconds * capture.metadata.sample_rate))
    if count > 10_000_000 or count < 1024:
        raise CaptureError("Unsupported selected sample count")
    if first + count > capture.sample_count:
        raise CaptureError("Capture does not contain the complete selected interval; padding is disabled")
    chunks = list(capture.iter_chunks(start_sample=first, max_samples=count))
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description="Numerical raw RF representation, independent of image colormaps; no classifier claim.")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--format", choices=("cf32", "ci16"))
    parser.add_argument("--sample-rate", type=float)
    parser.add_argument("--center-frequency", type=float)
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--duration-seconds", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("Choose a new .npz output file")
    capture = open_capture(args.capture, format=args.format, sample_rate=args.sample_rate,
                           center_frequency=args.center_frequency)
    signal = read_segment(capture, args.start_seconds, args.duration_seconds)
    values = representation(signal, capture.metadata.sample_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, spectrum=values)
    print(json.dumps({"output": str(args.output), "samples": len(signal), "shape": list(values.shape),
                      "sample_rate_hz": capture.metadata.sample_rate, "recipe": RECIPE,
                      "classification": "not_performed"}, indent=2))


if __name__ == "__main__":
    main()
