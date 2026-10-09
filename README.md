# OcuSync RF classifier

Offline I/Q classifier for nominal O2, O3, O4 or UNKNOWN. Includes the trained
model, four recorded examples, training code and tests. The demo runs on CPU.

By Jeronimo Munoz Larreta.

## Run the demo

With Python 3.12, from this directory:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -v
```

The examples are Mini 3, FPV Combo, Avata 2 and recorded background.
The demo checks hashes and compares labels after inference.

```sh
python classify_raw.py signal_01.iq --sample-rate 100000000 --format cf32
python virtual_rx.py signal_02.iq --max-windows 1
```

Input: interleaved little-endian float32 I/Q (`cf32`) or int16 (`ci16`,
scaled by 32768). Windows are 1000000 complex samples at 100 MS/s, or 10 ms.
Other rates are rejected. `capture_io.py` also reads supported single-channel
SigMF files.

## Model

DC removal, RMS normalization and STFT with a 2048-point Hamming window and
1024-sample hop. The 516 features describe 128 frequency bands and temporal
energy. Extra Trees, Random Forest and scaled logistic regression outputs
are averaged.

A non-OTHER winner needs score >= 0.65 and margin >= 0.15. Scores are
uncalibrated. Filenames and labels are not model inputs.

## Recorded results and limits

The September 30 test uses one recording per family. Temporal splits have
20 ms guards but share source recordings.

| Condition | Correct | UNKNOWN |
| --- | ---: | ---: |
| Clean temporal holdout | 15/15 | 0/15 |
| Added noise at 10 dB capture-to-added-noise power | 14/15 | 1/15 |
| Additional FPV recording with a different bandwidth setting | 91/100 | 9/100 |
| External Mini 4 Pro, nominal O4 | 0/10 | 10/10 |

Full conditions are in `test_results.json`. These capture-specific results
do not estimate field accuracy. Equipment, RF band and acquisition settings
are confounded with family. The Mini 4 Pro test shows limited O4 transfer.
Added-noise ratios are relative to capture power, not measured field SNR.

Labels follow transmitter specifications, not decoded protocol versions.
The model does not decrypt payloads. UNKNOWN is abstention, not absence of a
drone.

The adopted X310 profile is 100 MS/s. Mini 3's XML belongs to adjacent pack1,
not the selected pack2. B210 use needs a revised sampling/bandwidth recipe
and new evaluation. Upsampling does not restore uncaptured spectrum.

## Retrain

Use full captures from `DATA_SOURCES.md`, not the short demo windows. Create
a manifest with one recording per family:

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

The trainer writes the model, manifest and results to a new or empty output
directory. New campaigns need splits by independent recording session.

## Earlier spectrogram experiments

`experiments/rfuav_images` contains the RFUAV baseline and adapted heads.
They classify equipment from spectrogram images and look up its family.
That directory has its own dependencies and commands.

## Data and model files

Sources and licenses are in `DATA_SOURCES.md` and `NOTICE.md`. File hashes
are in `checksums.json`. Only load trusted joblib files, since pickle can
execute code.

Original software is copyright 2026 Jeronimo Munoz Larreta, with no additional
license grant. Third-party terms remain separate. No affiliation with DJI,
Ettus Research or the dataset authors.
