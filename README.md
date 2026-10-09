# OcuSync RF classifier

Offline classification of recorded I/Q into nominal O2, O3, O4 or UNKNOWN.
The repository includes the trained ensemble, raw-signal examples, training
code, input readers and tests. No SDR is needed for the included demo.

By Jeronimo Munoz Larreta.

## Run the demo

Use Python 3.12 and the pinned dependencies. From this directory:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -v
```

The four included files are real recorded I/Q windows: Mini 3, FPV Combo,
Avata 2 and environmental background. The demo checks their hashes, predicts
each label, then compares the prediction with the expected label.

```sh
python classify_raw.py signal_01.iq --sample-rate 100000000 --format cf32
python virtual_rx.py signal_02.iq --max-windows 1
```

Inputs are interleaved little-endian float32 I/Q (`cf32`) or int16 I/Q
(`ci16`, divided by 32768). Each inference uses one million complex samples,
or 10 ms at the supported 100 MS/s rate. Other rates are rejected.
`capture_io.py` also provides bounded readers for supported single-channel
SigMF captures.

## Model

DC removal and RMS normalization precede a two-sided STFT with a 2048-point
symmetric Hamming window and 1024-sample hop. The representation combines
128 frequency bands with four summary statistics and four temporal energy
features: 516 values per window.

An ensemble averages Extra Trees, Random Forest and scaled logistic
regression outputs. A non-OTHER winner needs score >= 0.65 and a margin
>= 0.15 to be accepted. Scores are not calibrated correctness probabilities.
The filename and expected label are not model inputs.

## Recorded results and limits

The September 30 experiment used three source recordings, one equipment
model per nominal family. Training, validation and test windows are
nonoverlapping, with 20 ms guard gaps, but belong to the same recordings.

| Condition | Correct | UNKNOWN |
| --- | ---: | ---: |
| Clean temporal holdout | 15/15 | 0/15 |
| Added noise at 10 dB capture-to-added-noise power | 14/15 | 1/15 |
| Additional FPV recording with a different bandwidth setting | 91/100 | 9/100 |
| External Mini 4 Pro, nominal O4 | 0/10 | 10/10 |

`test_results.json` retains the measured conditions and external tests.
The clean result is a small capture-specific test, not a field accuracy
estimate. Equipment, RF band and acquisition settings are confounded with
family. In particular, the Mini 4 Pro result demonstrates limited transfer
even within nominal O4. Noise ratios describe added noise relative to the
whole capture, not known field SNR.

Labels come from the documented transmitter models. Negotiated protocol
versions are not decoded. No video, control payload or private identifier is
decrypted. UNKNOWN does not establish that no drone is present.

The source profile is USRP X310 at an adopted 100 MS/s. Mini 3's available
XML belongs to adjacent pack1, not the selected pack2 recording. Applying
this model to a B210 requires a new sampling and bandwidth recipe and
evaluation on B210 recordings. Upsampling alone cannot restore uncaptured
spectrum.

## Retrain

The bundled demo windows are test examples and are too short to retrain.
Obtain the full source captures described in `DATA_SOURCES.md`, then create
a JSON manifest with exactly one recording per family:

```json
{"sources": [
  {"path": "captures/mini3.cf32", "family": "OCU2"},
  {"path": "captures/fpv.cf32", "family": "OCU3"},
  {"path": "captures/avata2.cf32", "family": "OCU4"}
]}
```

Paths are relative to the manifest. An optional `sha256` checks source
integrity. Each file needs at least 260 ms at 100 MS/s.

```sh
python train_raw.py captures.json --output training-output
```

The trainer retains the original seed, transforms, three models and split
recipe. It writes a new model, source manifest and per-window results to a
new or empty output directory. It does not overwrite the bundled model.
For new campaigns, redesign the split around independent recording sessions.

## Earlier spectrogram experiments

`experiments/rfuav_images` contains the RFUAV checkpoint baseline,
linear-head adaptation and six-class equipment classifier with OTHER_SIGNAL.
These consume images using the RFUAV recipe. Their equipment-to-link-family
lookup is different from the main raw-I/Q model. See that directory's README
for dependencies and commands.

## Data and model files

`DATA_SOURCES.md` gives attribution, original archive members, sample offsets
and licensing. `checksums.json` records every distributed file's SHA256.
Only load the bundled or other trusted joblib model files: joblib uses
pickle, which can execute code. A checksum confirms bytes, not their origin.

Original software is copyright 2026 Jeronimo Munoz Larreta. No additional
software license is granted here. Third-party data and model licenses remain
separate and are described in `NOTICE.md`. This project is not affiliated
with DJI, Ettus Research or the dataset authors.
