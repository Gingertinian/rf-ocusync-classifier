import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_io import CaptureError, open_capture


def transform(samples, sample_rate, *, noise_ratio_db=None, frequency_offset_hz=0,
              iq_gain_ratio=1, echo_gain=0, echo_delay_samples=0, seed=20260930):
    if samples.ndim != 1 or samples.dtype.kind != "c" or not len(samples) or not np.isfinite(samples).all():
        raise CaptureError("Expected nonempty finite complex I/Q")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise CaptureError("Sample rate must be positive and finite")
    if not np.isfinite(frequency_offset_hz) or abs(frequency_offset_hz) >= sample_rate / 2:
        raise CaptureError("Frequency offset exceeds the supported baseband range")
    if not np.isfinite(iq_gain_ratio) or not 0.1 <= iq_gain_ratio <= 10:
        raise CaptureError("I/Q gain ratio must be in [0.1, 10]")
    if not np.isfinite(echo_gain) or not 0 <= echo_gain <= 1:
        raise CaptureError("Echo gain must be in [0, 1]")
    if echo_gain and (isinstance(echo_delay_samples, bool) or not isinstance(echo_delay_samples, int) or not 1 <= echo_delay_samples < len(samples)):
        raise CaptureError("A nonzero echo requires a positive delay shorter than the capture")
    signal = samples.astype(np.complex128).copy()
    original_power = float(np.mean(np.abs(signal) ** 2))
    if original_power <= 0:
        raise CaptureError("Original capture has zero power")
    if echo_gain:
        signal[echo_delay_samples:] += echo_gain * signal[:-echo_delay_samples].copy()
    if iq_gain_ratio != 1:
        signal = signal.real + 1j * signal.imag * iq_gain_ratio
    if frequency_offset_hz:
        phase = 2 * np.pi * frequency_offset_hz * np.arange(len(signal), dtype=np.float64) / sample_rate
        signal *= np.exp(1j * phase)
    measured_ratio = None
    if noise_ratio_db is not None:
        if not np.isfinite(noise_ratio_db) or not -20 <= noise_ratio_db <= 80:
            raise CaptureError("Capture/noise power ratio must be finite and in [-20, 80]dB")
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
        noise *= np.sqrt(original_power / (10 ** (noise_ratio_db / 10)) / 2)
        measured_ratio = float(10 * np.log10(original_power / np.mean(np.abs(noise) ** 2)))
        signal += noise
    result = signal.astype(np.complex64)
    if not np.isfinite(result).all():
        raise CaptureError("Perturbation produced nonfinite I/Q")
    metadata = {
        "seed": seed,
        "added_frequency_offset_hz": frequency_offset_hz,
        "imaginary_component_gain_ratio": iq_gain_ratio,
        "echo_gain": echo_gain,
        "echo_delay_samples": echo_delay_samples if echo_gain else 0,
        "requested_original_capture_to_added_noise_power_db": noise_ratio_db,
        "measured_original_capture_to_added_noise_power_db": measured_ratio,
        "true_signal_to_noise_ratio_known": False,
        "real_channel_measurement": False,
    }
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description="Apply deterministic perturbations to an existing RF capture; no transmission.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="New .sigmf-data path; refuses existing files")
    parser.add_argument("--format", choices=("cf32", "ci16"))
    parser.add_argument("--sample-rate", type=float)
    parser.add_argument("--center-frequency", type=float)
    parser.add_argument("--noise-ratio-db", type=float)
    parser.add_argument("--frequency-offset-hz", type=float, default=0)
    parser.add_argument("--iq-gain-ratio", type=float, default=1)
    parser.add_argument("--echo-gain", type=float, default=0)
    parser.add_argument("--echo-delay-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260930)
    options = parser.parse_args()
    if options.output.suffix != ".sigmf-data":
        parser.error("Output must end in .sigmf-data")
    output_meta = options.output.with_suffix(".sigmf-meta")
    if options.output.exists() or output_meta.exists():
        parser.error("Refusing to replace an existing capture or metadata")
    capture = open_capture(options.input, format=options.format, sample_rate=options.sample_rate,
                           center_frequency=options.center_frequency)
    if capture.sample_count > 10_000_000:
        parser.error("Select an input of at most ten million complex samples")
    samples = np.concatenate(list(capture.iter_chunks(max_samples=10_000_000)))
    derived, metadata = transform(samples, capture.metadata.sample_rate, noise_ratio_db=options.noise_ratio_db,
                                  frequency_offset_hz=options.frequency_offset_hz, iq_gain_ratio=options.iq_gain_ratio,
                                  echo_gain=options.echo_gain, echo_delay_samples=options.echo_delay_samples,
                                  seed=options.seed)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("xb") as stream:
        stream.write(derived.view("<f4").tobytes())
    capture_meta = {"core:sample_start": 0}
    if capture.metadata.center_frequency is not None:
        capture_meta["core:frequency"] = capture.metadata.center_frequency
    sigmf = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": capture.metadata.sample_rate,
                   "core:version": "1.2.6", "core:description": "Synthetic perturbation of the specified source capture",
                   "source_sha256": hashlib.sha256(capture.path.read_bytes()).hexdigest(),
                   "derived_sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
                   "perturbation": metadata},
        "captures": [capture_meta], "annotations": [],
    }
    with output_meta.open("x", encoding="utf-8") as stream:
        json.dump(sigmf, stream, indent=2)
    print(json.dumps({"output": str(options.output), "metadata": str(output_meta), "samples": len(derived),
                      "derived_sha256": sigmf["global"]["derived_sha256"], "perturbation": metadata}, indent=2))


if __name__ == "__main__":
    main()
