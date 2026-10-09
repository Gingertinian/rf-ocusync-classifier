"""Synthetic unit tests for input plumbing only; no RF measurement or labels.

All fixtures stay in BytesIO. No captures, metadata, model outputs or other
files are written; run with python -B to suppress bytecode caches as well.
"""

import io
import json
import stat
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import capture_io as c


def encoded_iq(values, datatype="cf32_le"):
    endian = "<" if datatype.endswith("_le") else ">"
    kind = "f4" if datatype.startswith("cf32") else "i2"
    return np.asarray(values, dtype=endian + kind).tobytes()


class ShortReads(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.largest_request = 0

    def read(self, size=-1):
        self.largest_request = max(self.largest_request, size)
        return super().read(min(size, 3))


class SyntheticCaptureTests(unittest.TestCase):
    def test_cf32_endian_iq_order_and_chunk_boundaries(self):
        for datatype in ("cf32_le", "cf32_be"):
            with self.subTest(datatype=datatype):
                chunks = list(c.iter_iq(io.BytesIO(encoded_iq([[1, -2], [3, 4], [-5, 6]], datatype)),
                                        c.CaptureMetadata(datatype, 1e6), chunk_samples=2))
                self.assertEqual([len(chunk) for chunk in chunks], [2, 1])
                np.testing.assert_array_equal(np.concatenate(chunks), [1-2j, 3+4j, -5+6j])
                self.assertEqual(chunks[0].dtype, np.complex64)

    def test_ci16_endian_and_signed_scaling(self):
        for datatype in ("ci16_le", "ci16_be"):
            with self.subTest(datatype=datatype):
                samples = np.concatenate(list(c.iter_iq(
                    io.BytesIO(encoded_iq([[-32768, 32767], [16384, -16384]], datatype)),
                    c.CaptureMetadata(datatype, 1000))))
                np.testing.assert_array_equal(samples, [-1 + (32767/32768)*1j, .5-.5j])

    def test_short_reads_remain_bounded_and_selection_is_exact(self):
        stream = ShortReads(encoded_iq([[i, -i] for i in range(20)]))
        samples = np.concatenate(list(c.iter_iq(stream, c.CaptureMetadata("cf32_le", 1000),
                                                start_sample=3, max_samples=5, chunk_samples=2)))
        np.testing.assert_array_equal(samples, np.arange(3, 8)*(1-1j))
        self.assertLessEqual(stream.largest_request, 16)
        self.assertEqual(stream.tell(), 8*8)

    def test_whole_file_alignment_even_when_prefix_is_limited(self):
        for datatype, extra in (("cf32_le", b"x"), ("cf32_le", b"\0"*4),
                                ("ci16_le", b"x"), ("ci16_le", b"\0"*2)):
            with self.subTest(datatype=datatype, extra=len(extra)):
                data = encoded_iq([[1, 2]], datatype) + extra
                with self.assertRaisesRegex(c.CaptureError, "incomplete I/Q"):
                    list(c.iter_iq(io.BytesIO(data), c.CaptureMetadata(datatype, 1000), max_samples=1))

    def test_shortened_file_during_read_is_rejected(self):
        class TruncatesOnRead(io.BytesIO):
            def read(self, size=-1):
                self.truncate(4)
                return super().read(size)
        with self.assertRaisesRegex(c.CaptureError, "truncated"):
            list(c.iter_iq(TruncatesOnRead(encoded_iq([[1, 2]])), c.CaptureMetadata("cf32_le", 1000)))

    def test_nonfinite_selected_values_fail_without_scanning_unselected_samples(self):
        for value in (np.nan, np.inf, -np.inf):
            data = encoded_iq([[1, 2], [value, 0]])
            metadata = c.CaptureMetadata("cf32_le", 1000)
            self.assertEqual(len(list(c.iter_iq(io.BytesIO(data), metadata, max_samples=1))), 1)
            with self.assertRaisesRegex(c.CaptureError, "NaN or infinity"):
                list(c.iter_iq(io.BytesIO(data), metadata))

    def test_invalid_metadata_and_limits(self):
        for rate in (None, 0, -1, np.nan, np.inf, True, "1000", 1e13):
            with self.subTest(rate=rate), self.assertRaises(c.CaptureError):
                c.CaptureMetadata("cf32_le", rate)
        for frequency in (-1, np.nan, np.inf, True):
            with self.subTest(frequency=frequency), self.assertRaises(c.CaptureError):
                c.CaptureMetadata("cf32_le", 1000, frequency)
        for options in ({"max_samples": 0}, {"chunk_samples": c.MAX_CHUNK_SAMPLES+1},
                        {"start_sample": -1}, {"start_sample": 2}, {"max_samples": True}):
            with self.subTest(options=options), self.assertRaises(c.CaptureError):
                list(c.iter_iq(io.BytesIO(encoded_iq([[1, 2]])), c.CaptureMetadata("cf32_le", 1000), **options))
        with self.assertRaisesRegex(c.CaptureError, "empty"):
            list(c.iter_iq(io.BytesIO(), c.CaptureMetadata("cf32_le", 1000)))

    def test_inspection_rail_counts_and_unknown_float_clipping(self):
        for datatype, values, expected_clipped in (
            ("ci16_be", [[-32768, 32767], [0, 0], [100, 32767]], 2),
            ("cf32_le", [[2, -3], [0, 0], [1, 1]], None),
        ):
            data = encoded_iq(values, datatype)
            capture = c.FileCapture(Path("synthetic.iq"), c.CaptureMetadata(datatype, 1000), len(data))
            with patch.object(Path, "open", return_value=io.BytesIO(data)):
                result = c.inspect_capture(capture, chunk_samples=1)
            self.assertEqual(result["samples_inspected"], 3)
            self.assertEqual(result["clipping"]["samples"], expected_clipped)
            self.assertIsNone(result["center_frequency_hz"])
            self.assertEqual(result["frequency_reference"], "baseband")
            self.assertEqual(result["nominal_frequency_span_hz"], [-500, 500])

    def test_raw_requires_metadata_and_refuses_special_files(self):
        for options in ({}, {"format": "cf32"}, {"format": "pickle", "sample_rate": 1000}):
            with self.subTest(options=options), self.assertRaises(c.CaptureError):
                c.open_capture("synthetic.iq", **options)
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=stat.S_IFIFO, st_size=8)):
            with self.assertRaisesRegex(c.CaptureError, "regular"):
                c.open_capture("synthetic.iq", format="cf32", sample_rate=1000)

    def test_read_samples_allocation_limit_precedes_io(self):
        capture = c.FileCapture(Path("never-open.iq"), c.CaptureMetadata("cf32_le", 1000), 8)
        with patch.object(Path, "open") as opened:
            with self.assertRaises(c.CaptureError):
                capture.read_samples(max_samples=c.MAX_ARRAY_SAMPLES+1)
            opened.assert_not_called()


class SyntheticSigMFTests(unittest.TestCase):
    def metadata(self, datatype="ci16_be"):
        return {"global": {"core:datatype": datatype, "core:sample_rate": 2e6, "core:version": "1.2.6"},
                "captures": [{"core:sample_start": 0, "core:frequency": 2.4e9}], "annotations": []}

    def test_all_supported_datatypes_and_capture_frequency(self):
        for datatype in ("cf32_le", "cf32_be", "ci16_le", "ci16_be"):
            metadata = c.parse_sigmf_metadata(self.metadata(datatype))
            self.assertEqual(metadata.datatype, datatype)
            self.assertEqual(metadata.frequency_span, (2.399e9, 2.401e9))

    def test_missing_frequency_stays_unknown_and_missing_rate_fails(self):
        document = self.metadata()
        del document["captures"][0]["core:frequency"]
        self.assertIsNone(c.parse_sigmf_metadata(document).center_frequency)
        del document["global"]["core:sample_rate"]
        with self.assertRaisesRegex(c.CaptureError, "mandatory"):
            c.parse_sigmf_metadata(document)

    def test_unsupported_and_ambiguous_layouts_are_rejected(self):
        for name, value in (("core:datatype", "cf64_le"), ("core:num_channels", 2),
                            ("core:offset", 1), ("core:trailing_bytes", 2),
                            ("core:dataset", "../private.dat"),
                            ("core:extensions", [{"name": "custom", "optional": False}])):
            document = self.metadata()
            document["global"][name] = value
            with self.subTest(name=name), self.assertRaises(c.CaptureError):
                c.parse_sigmf_metadata(document)
        for captures in ([{"core:sample_start": 1}], [{"core:sample_start": 0, "core:header_bytes": 4}],
                         [{"core:sample_start": 0}, {"core:sample_start": 10}], [None]):
            document = self.metadata(); document["captures"] = captures
            with self.subTest(captures=captures), self.assertRaises(c.CaptureError):
                c.parse_sigmf_metadata(document)

    def test_sigmf_pair_read_and_explicit_override_rejection(self):
        data = encoded_iq([[16384, -16384]], "ci16_be")
        meta = json.dumps(self.metadata()).encode()
        def memory_open(path, mode="r", **kwargs):
            self.assertEqual(mode, "rb")
            return io.BytesIO(meta if path.suffix == ".sigmf-meta" else data)
        with patch.object(Path, "open", autospec=True, side_effect=memory_open), \
             patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=stat.S_IFREG, st_size=len(data))):
            for suffix in (".sigmf-meta", ".sigmf-data"):
                capture = c.open_capture("synthetic"+suffix)
                np.testing.assert_array_equal(capture.read_samples(), [.5-.5j])
        with self.assertRaisesRegex(c.CaptureError, "overrides"):
            c.open_capture("synthetic.sigmf-meta", sample_rate=1e6)

    def test_malformed_duplicate_nonfinite_and_oversized_json(self):
        for payload in (b"not json", b'{"global":{},"global":{}}', b'{"global":{"core:sample_rate":NaN}}',
                        b" "*(c.MAX_METADATA_BYTES+1)):
            with self.subTest(size=len(payload)), patch.object(Path, "open", return_value=io.BytesIO(payload)):
                with self.assertRaises(c.CaptureError):
                    c.open_capture("synthetic.sigmf-meta")


class SyntheticSTFTTests(unittest.TestCase):
    def test_signed_frequency_tone_center_and_coherent_amplitude(self):
        for frequency in (-128, 128):
            samples = np.exp(2j*np.pi*frequency*np.arange(256)/1024)
            f, t, transformed = c.compute_stft(samples, 1024, n_fft=64, hop_length=32,
                                               window="rectangular", center_frequency=1e6)
            self.assertEqual(transformed.shape, (64, 7))
            self.assertEqual(f[np.abs(transformed[:, 0]).argmax()], 1e6+frequency)
            self.assertAlmostEqual(float(np.abs(transformed[:, 0]).max()), 1)
            np.testing.assert_allclose(t, (np.arange(7)*32+32)/1024)

    def test_zero_power_db_floor_is_finite(self):
        result = c.compute_spectrogram(np.zeros(128, dtype=np.complex64), 1000, n_fft=32)
        self.assertEqual(result.power.shape, (32, 7))
        self.assertTrue(np.isfinite(result.power_db).all())
        np.testing.assert_allclose(result.power_db, -120)
        self.assertEqual(float(result.power.max()), 0)

    def test_stft_budget_and_bad_inputs_fail_before_large_allocation(self):
        samples = np.ones(128, dtype=np.complex64)
        for options in ({"max_output_bytes": 100}, {"hop_length": 0}, {"hop_length": 129},
                        {"n_fft": 1}, {"window": "invented"}):
            with self.subTest(options=options), self.assertRaises(c.CaptureError):
                c.compute_stft(samples, 1000, **options)
        for invalid in (np.ones(128), np.ones((2, 128), dtype=np.complex64),
                        np.array([complex(np.nan, 0)]*128), np.ones(8, dtype=np.complex64)):
            with self.subTest(shape=invalid.shape), self.assertRaises(c.CaptureError):
                c.compute_stft(invalid, 1000, n_fft=32)


class SyntheticCLITests(unittest.TestCase):
    def test_inspect_cli_outputs_metadata_count_span_and_clipping(self):
        data = encoded_iq([[0, 0], [32767, 0], [1, 2]], "ci16_be")
        output = io.StringIO()
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=stat.S_IFREG, st_size=len(data))), \
             patch.object(Path, "open", return_value=io.BytesIO(data)), redirect_stdout(output):
            status = c.main(["inspect", "synthetic.iq", "--format", "ci16", "--endian", "big",
                             "--sample-rate", "2000", "--center-frequency", "2400000000",
                             "--max-samples", "2", "--chunk-samples", "1"])
        result = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["format"], "ci16_be")
        self.assertEqual(result["sample_rate_hz"], 2000)
        self.assertEqual(result["nominal_frequency_span_hz"], [2399999000, 2400001000])
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["samples_inspected"], 2)
        self.assertTrue(result["limited"])
        self.assertEqual(result["clipping"]["samples"], 1)

    def test_cli_missing_rate_has_nonzero_exit_without_opening_capture(self):
        with patch.object(Path, "open") as opened, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                c.main(["inspect", "synthetic.iq", "--format", "cf32"])
            self.assertEqual(error.exception.code, 2)
            opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
