"""Passive, bounded raw I/Q file input; no radio access or class inference.

Raw cf32/ci16 files require an explicit format and sample rate. ci16 components
are scaled by 32768; floating-point components retain their original scale.
SigMF support is deliberately limited to single-channel, conforming datasets
with zero offset and at most one capture starting at sample zero. Supported
datatypes are cf32_le, cf32_be, ci16_le and ci16_be. Center frequency is optional
and remains unknown when absent. Byte alignment covers the entire file; finite
value checks cover only the selected samples.

The numeric STFT uses symmetric Hamming/Hann windows, coherent-gain scaling,
two-sided FFT-shifted frequencies, complete frames and no boundary padding.
It is not RFUAV's MATLAB image conversion or a drop-in model input transform.

CLI (stdout only):
  python -B capture_io.py inspect recording.iq --format cf32 --sample-rate 1e6
  python -B capture_io.py inspect recording.sigmf-meta --max-samples 1000000
"""

from __future__ import annotations

import argparse
import json
import math
import stat
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping

import numpy as np

DEFAULT_MAX_SAMPLES = 1_000_000
DEFAULT_CHUNK_SAMPLES = 65_536
MAX_CHUNK_SAMPLES = 262_144
MAX_ARRAY_SAMPLES = 4_000_000
MAX_SPECTROGRAM_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
_DTYPES = {
    "cf32_le": np.dtype("<f4"), "cf32_be": np.dtype(">f4"),
    "ci16_le": np.dtype("<i2"), "ci16_be": np.dtype(">i2"),
}


class CaptureError(ValueError):
    """Invalid, unsupported or truncated capture input."""


def _integer(value, name: str, minimum=0, maximum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CaptureError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise CaptureError(f"{name} is outside the supported range")
    return value


def _finite(value, name: str, *, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CaptureError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise CaptureError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


@dataclass(frozen=True)
class CaptureMetadata:
    datatype: str
    sample_rate: float
    center_frequency: float | None = None

    def __post_init__(self):
        if not isinstance(self.datatype, str) or self.datatype not in _DTYPES:
            raise CaptureError("supported datatypes: " + ", ".join(_DTYPES))
        rate = _finite(self.sample_rate, "sample_rate", positive=True)
        if rate > 1e12:
            raise CaptureError("sample_rate exceeds the SigMF limit of 1e12 samples/s")
        object.__setattr__(self, "sample_rate", rate)
        if self.center_frequency is not None:
            frequency = _finite(self.center_frequency, "center_frequency")
            if frequency < 0:
                raise CaptureError("center_frequency must be nonnegative")
            object.__setattr__(self, "center_frequency", frequency)
        if not all(math.isfinite(f) for f in self.frequency_span):
            raise CaptureError("frequency span must be finite")

    @property
    def bytes_per_sample(self) -> int:
        return 2 * _DTYPES[self.datatype].itemsize

    @property
    def frequency_span(self) -> tuple[float, float]:
        center = self.center_frequency if self.center_frequency is not None else 0.0
        return center - self.sample_rate / 2, center + self.sample_rate / 2


def parse_sigmf_metadata(document: Mapping) -> CaptureMetadata:
    """Read supported core fields; reject layouts this adapter cannot interpret."""
    if not isinstance(document, Mapping) or not isinstance(document.get("global"), Mapping):
        raise CaptureError("SigMF requires a global object")
    global_meta = document["global"]
    if "core:sample_rate" not in global_meta:
        raise CaptureError("core:sample_rate is mandatory for this adapter")
    if "core:dataset" in global_meta or global_meta.get("core:metadata_only", False):
        raise CaptureError("non-conforming or metadata-only SigMF datasets are unsupported")
    for name, expected in (("core:num_channels", 1), ("core:offset", 0), ("core:trailing_bytes", 0)):
        if _integer(global_meta.get(name, expected), name) != expected:
            raise CaptureError(f"unsupported SigMF {name}")
    extensions = global_meta.get("core:extensions", [])
    if not isinstance(extensions, list) or any(
        not isinstance(extension, Mapping) or extension.get("optional") is not True
        for extension in extensions
    ):
        raise CaptureError("required SigMF extensions are unsupported")
    captures = document.get("captures", [])
    if not isinstance(captures, list) or len(captures) > 1:
        raise CaptureError("only a single contiguous SigMF capture is supported")
    frequency = None
    if captures:
        capture = captures[0]
        if not isinstance(capture, Mapping):
            raise CaptureError("SigMF capture must be an object")
        if _integer(capture.get("core:sample_start"), "core:sample_start") != 0:
            raise CaptureError("SigMF capture must start at sample zero")
        if _integer(capture.get("core:header_bytes", 0), "core:header_bytes") != 0:
            raise CaptureError("SigMF capture headers are unsupported")
        frequency = capture.get("core:frequency")
    return CaptureMetadata(global_meta.get("core:datatype"), global_meta["core:sample_rate"], frequency)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise CaptureError(f"non-finite JSON value: {value}")


def _sample_count(byte_count: int, metadata: CaptureMetadata) -> int:
    if byte_count <= 0:
        raise CaptureError("capture is empty")
    if byte_count % metadata.bytes_per_sample:
        raise CaptureError("file contains an incomplete I/Q pair or truncated component")
    return byte_count // metadata.bytes_per_sample


def _stream_size(stream: BinaryIO) -> int:
    if not stream.seekable():
        raise CaptureError("a seekable file or binary buffer is required")
    position = stream.tell()
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(position)
    return size


def iter_iq(stream: BinaryIO, metadata: CaptureMetadata, *, start_sample=0,
            max_samples=DEFAULT_MAX_SAMPLES, chunk_samples=DEFAULT_CHUNK_SAMPLES) -> Iterator[np.ndarray]:
    """Yield complex64 arrays using bounded reads, including short-read handling.

    The stream must stay open while iterating. No unselected samples are decoded.
    Integer rail statistics can be recovered from the fixed /32768 scaling.
    """
    start = _integer(start_sample, "start_sample")
    limit = _integer(max_samples, "max_samples", 1)
    chunk = _integer(chunk_samples, "chunk_samples", 1, MAX_CHUNK_SAMPLES)
    size = _stream_size(stream)
    total = _sample_count(size, metadata)
    if start > total:
        raise CaptureError("start_sample exceeds the capture length")
    remaining = min(limit, total - start)
    stream.seek(start * metadata.bytes_per_sample)
    while remaining:
        count = min(chunk, remaining)
        wanted = count * metadata.bytes_per_sample
        payload = bytearray()
        while len(payload) < wanted:
            part = stream.read(wanted - len(payload))
            if not isinstance(part, (bytes, bytearray)) or not part:
                raise CaptureError("capture shortened or truncated while reading")
            if len(part) > wanted - len(payload):
                raise CaptureError("binary stream returned more bytes than requested")
            payload.extend(part)
        pairs = np.frombuffer(payload, dtype=_DTYPES[metadata.datatype]).reshape(count, 2)
        if not np.isfinite(pairs).all():
            raise CaptureError("selected samples contain NaN or infinity")
        samples = np.empty(count, dtype=np.complex64)
        samples.real, samples.imag = pairs[:, 0], pairs[:, 1]
        if metadata.datatype.startswith("ci16"):
            samples /= 32768.0
        remaining -= count
        yield samples
    if _stream_size(stream) != size:
        raise CaptureError("capture size changed during reading")


@dataclass(frozen=True)
class FileCapture:
    path: Path
    metadata: CaptureMetadata
    byte_count: int

    @property
    def sample_count(self) -> int:
        return _sample_count(self.byte_count, self.metadata)

    def iter_chunks(self, **options) -> Iterator[np.ndarray]:
        with self.path.open("rb") as stream:
            if _stream_size(stream) != self.byte_count:
                raise CaptureError("capture size changed since opening")
            yield from iter_iq(stream, self.metadata, **options)

    def read_samples(self, *, max_samples=DEFAULT_MAX_SAMPLES, **options) -> np.ndarray:
        """Bounded convenience read; use iter_chunks for larger recordings."""
        _integer(max_samples, "max_samples", 1, MAX_ARRAY_SAMPLES)
        chunks = list(self.iter_chunks(max_samples=max_samples, **options))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.complex64)


def open_capture(path, *, format=None, sample_rate=None, center_frequency=None, endian=None) -> FileCapture:
    """Open a raw file with explicit metadata or a standard SigMF file pair."""
    path = Path(path)
    if path.suffix in (".sigmf-meta", ".sigmf-data"):
        if any(value is not None for value in (format, sample_rate, center_frequency, endian)):
            raise CaptureError("raw metadata overrides are not allowed for SigMF")
        meta_path = path.with_suffix(".sigmf-meta")
        with meta_path.open("rb") as stream:
            encoded = stream.read(MAX_METADATA_BYTES + 1)
        if len(encoded) > MAX_METADATA_BYTES:
            raise CaptureError("SigMF metadata exceeds the size limit")
        try:
            document = json.loads(encoded.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                                  parse_constant=_invalid_constant)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise CaptureError("invalid SigMF JSON metadata") from error
        metadata = parse_sigmf_metadata(document)
        path = path.with_suffix(".sigmf-data")
    else:
        if format not in ("cf32", "ci16"):
            raise CaptureError("raw capture requires --format cf32 or ci16")
        if sample_rate is None:
            raise CaptureError("raw capture requires an explicit sample_rate")
        endian = "little" if endian is None else endian
        if endian not in ("little", "big"):
            raise CaptureError("endian must be little or big")
        metadata = CaptureMetadata(format + ("_le" if endian == "little" else "_be"),
                                   sample_rate, center_frequency)
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise CaptureError("only regular capture files are supported")
    _sample_count(status.st_size, metadata)
    return FileCapture(path, metadata, status.st_size)


def inspect_capture(capture: FileCapture, *, start_sample=0, max_samples=DEFAULT_MAX_SAMPLES,
                    chunk_samples=DEFAULT_CHUNK_SAMPLES) -> dict:
    """Summarize the selected samples; float ADC clipping is explicitly unknown."""
    count = clipped = 0
    peak = energy = 0.0
    integer = capture.metadata.datatype.startswith("ci16")
    for samples in capture.iter_chunks(start_sample=start_sample, max_samples=max_samples,
                                       chunk_samples=chunk_samples):
        i, q = samples.real.astype(np.float64), samples.imag.astype(np.float64)
        peak = max(peak, float(np.abs(i).max()), float(np.abs(q).max()))
        energy += float(np.sum(i * i + q * q))
        if integer:
            rail = (i == -1) | (i == 32767 / 32768) | (q == -1) | (q == 32767 / 32768)
            clipped += int(np.count_nonzero(rail))
        count += len(samples)
    rate = capture.metadata.sample_rate
    duration = capture.sample_count / rate
    if not math.isfinite(duration):
        raise CaptureError("capture duration exceeds finite numerical range")
    return {
        "format": capture.metadata.datatype, "sample_rate_hz": rate,
        "center_frequency_hz": capture.metadata.center_frequency,
        "nominal_frequency_span_hz": list(capture.metadata.frequency_span),
        "frequency_reference": "baseband" if capture.metadata.center_frequency is None else "absolute",
        "sample_count": capture.sample_count, "samples_inspected": count,
        "start_sample": int(start_sample), "limited": count < capture.sample_count - start_sample,
        "capture_duration_seconds": duration, "inspected_duration_seconds": count / rate,
        "component_peak": peak if count else None,
        "rms_magnitude": math.sqrt(energy / count) if count else None,
        "clipping": {"definition": "int16 rail hits; not proof of analog clipping" if integer
                     else "unknown: floating-point ADC full scale is unspecified",
                     "samples": clipped if integer else None,
                     "fraction": clipped / count if integer and count else None},
        "validation_scope": "whole-file byte alignment; selected sample values only",
    }


def compute_stft(samples: np.ndarray, sample_rate, *, n_fft=1024, hop_length=None,
                 window="hamming", center_frequency=None,
                 max_output_bytes=MAX_SPECTROGRAM_BYTES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency offsets (or absolute Hz), frame-center seconds and STFT.

    Complete windows only; no padding or implied RF band. Allocation is bounded
    before creating the frequency-by-time matrix. For longer inputs, process
    overlapping bounded selections and retain the desired cross-chunk overlap.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.dtype.kind != "c":
        raise CaptureError("STFT requires a one-dimensional complex numpy array")
    if samples.size > MAX_ARRAY_SAMPLES or not np.isfinite(samples).all():
        raise CaptureError("STFT input is too large or contains non-finite samples")
    metadata = CaptureMetadata("cf32_le", sample_rate, center_frequency)
    n_fft = _integer(n_fft, "n_fft", 2, 65_536)
    hop = n_fft // 2 if hop_length is None else _integer(hop_length, "hop_length", 1, n_fft)
    budget = _integer(max_output_bytes, "max_output_bytes", 1, MAX_SPECTROGRAM_BYTES)
    if samples.size < n_fft:
        raise CaptureError("not enough samples for one complete STFT frame")
    frames = 1 + (samples.size - n_fft) // hop
    # Includes the STFT, power/dB arrays, axes and bounded per-frame workspace.
    if n_fft * frames * 32 + (n_fft + frames) * 8 + n_fft * 64 > budget:
        raise CaptureError("STFT output exceeds the memory budget; reduce input or increase hop_length")
    windows = {"hamming": np.hamming, "hann": np.hanning, "rectangular": np.ones}
    if window not in windows:
        raise CaptureError("window must be hamming, hann or rectangular")
    weights = windows[window](n_fft)
    gain = float(weights.sum())
    if gain <= 0:
        raise CaptureError("window has zero coherent gain")
    frequencies = np.fft.fftshift(np.fft.fftfreq(n_fft)) * metadata.sample_rate
    if metadata.center_frequency is not None:
        frequencies += metadata.center_frequency
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            times = (np.arange(frames, dtype=np.float64) * hop + n_fft / 2) / metadata.sample_rate
            result = np.empty((n_fft, frames), dtype=np.complex128)
            for frame in range(frames):
                start = frame * hop
                data = samples[start:start + n_fft].astype(np.complex128) * weights
                result[:, frame] = np.fft.fftshift(np.fft.fft(data) / gain)
        except FloatingPointError as error:
            raise CaptureError("STFT exceeds finite numerical range") from error
    if not np.isfinite(result).all() or not np.isfinite(times).all():
        raise CaptureError("STFT exceeds finite numerical range")
    return frequencies, times, result


@dataclass(frozen=True)
class Spectrogram:
    frequencies_hz: np.ndarray
    times_seconds: np.ndarray
    power: np.ndarray
    power_db: np.ndarray


def compute_spectrogram(samples: np.ndarray, sample_rate, *, floor_power=1e-12, **options) -> Spectrogram:
    """Return relative power and 10*log10(power); not calibrated dBm or RGB."""
    floor = _finite(floor_power, "floor_power", positive=True)
    frequencies, times, transformed = compute_stft(samples, sample_rate, **options)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            power = transformed.real ** 2
            power += transformed.imag ** 2
            power_db = np.maximum(power, floor)
            np.log10(power_db, out=power_db)
            power_db *= 10
        except FloatingPointError as error:
            raise CaptureError("spectrogram exceeds finite numerical range") from error
    return Spectrogram(frequencies, times, power, power_db)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspection = commands.add_parser("inspect", help="inspect an existing capture; JSON to stdout")
    inspection.add_argument("path", type=Path)
    inspection.add_argument("--format", choices=("cf32", "ci16"))
    inspection.add_argument("--endian", choices=("little", "big"))
    inspection.add_argument("--sample-rate", type=float)
    inspection.add_argument("--center-frequency", type=float)
    inspection.add_argument("--start-sample", type=int, default=0)
    inspection.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    inspection.add_argument("--chunk-samples", type=int, default=DEFAULT_CHUNK_SAMPLES)
    args = parser.parse_args(argv)
    try:
        capture = open_capture(args.path, format=args.format, endian=args.endian,
                               sample_rate=args.sample_rate, center_frequency=args.center_frequency)
        result = inspect_capture(capture, start_sample=args.start_sample, max_samples=args.max_samples,
                                 chunk_samples=args.chunk_samples)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (CaptureError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
