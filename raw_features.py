import numpy as np
from scipy.signal import stft, windows

from capture_io import CaptureError, open_capture


SAMPLE_RATE = 100_000_000
WINDOW_SAMPLES = 1_000_000
CLASSES = ("OCU2", "OCU3", "OCU4", "OTHER")
RECIPE = {
    "sample_rate_hz": SAMPLE_RATE, "window_samples": WINDOW_SAMPLES,
    "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE,
    "fft_points": 2048, "stft_window": "symmetric_hamming",
    "hop_samples": 1024, "magnitude_db_floor_relative_to_peak": -60,
    "frequency_order": "fftshifted_low_to_high",
    "feature_input": "complex_samples_only_no_filename_or_device_metadata",
}


def extract(samples):
    if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.dtype.kind != "c":
        raise CaptureError("Expected a one-dimensional complex I/Q array")
    if len(samples) != WINDOW_SAMPLES or not np.isfinite(samples).all():
        raise CaptureError("Expected one million finite complex samples")
    values = samples.astype(np.complex128).copy()
    values -= values.mean()
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    if rms == 0:
        return np.zeros(516, dtype=np.float32)
    values /= rms
    values = values.astype(np.complex64)
    _, _, spectrum = stft(values, fs=SAMPLE_RATE, window=windows.hamming(2048),
                           nperseg=2048, noverlap=1024, nfft=2048,
                           return_onesided=False, boundary=None, padded=False)
    magnitude = np.abs(np.fft.fftshift(spectrum, axes=0))
    del spectrum
    np.maximum(magnitude, float(magnitude.max()) * 1e-3, out=magnitude)
    np.log10(magnitude, out=magnitude)
    magnitude *= 20
    magnitude -= magnitude.max()
    coarse = magnitude.reshape(128, 16, -1).mean(1)
    frequency_mean = coarse.mean(1)
    frequency_std = coarse.std(1)
    frequency_p90 = np.percentile(coarse, 90, axis=1)
    frequency_p10 = np.percentile(coarse, 10, axis=1)
    frame_energy = np.mean(np.abs(values.reshape(1000, 1000)) ** 2, axis=1)
    energy_summary = [float(np.std(frame_energy)), float(np.percentile(frame_energy, 95)),
                      float(np.percentile(frame_energy, 5)), float(np.mean(frame_energy > 2))]
    output = np.concatenate((frequency_mean / 60, frequency_std / 60,
                             frequency_p90 / 60, frequency_p10 / 60, energy_summary)).astype(np.float32)
    if output.shape != (516,) or not np.isfinite(output).all():
        raise CaptureError("Feature transform returned invalid values")
    return output


def read_window(path, first_sample=0, *, fmt="cf32", sample_rate=SAMPLE_RATE):
    if sample_rate != SAMPLE_RATE:
        raise CaptureError("This proof of concept uses the documented 100-MS/s receiver profile")
    capture = open_capture(path, format=fmt, sample_rate=sample_rate)
    if first_sample + WINDOW_SAMPLES > capture.sample_count:
        raise CaptureError("Capture does not contain a complete 10-ms window")
    values = np.concatenate(list(capture.iter_chunks(start_sample=first_sample,
                                                     max_samples=WINDOW_SAMPLES)))
    return values


def intervals(sample_count):
    complete = sample_count // WINDOW_SAMPLES
    if complete < 26:
        raise CaptureError("Training source must contain at least 260 ms")
    return {
        "train": list(range(0, 12)),
        "validation": list(range(14, 19)),
        "test": list(range(21, 26)),
        "guard_windows": [12, 13, 19, 20],
    }
