# Capture sources

All included I/Q samples are excerpts of public recordings. They were not
acquired with a physical receiver during this project.

## RFUAV

Rui Shi, Xiaodong Yu, Shengming Wang, Yijia Zhang, Lu Xu, Peng Pan and
Chunlai Ma. RFUAV A Benchmark Dataset for Unmanned Aerial Vehicle Detection
and Identification, 2025. https://arxiv.org/abs/2503.09033

Dataset: https://huggingface.co/datasets/kitofrank/RFUAV
Pinned revision: d91868143eb148bb5d87aa078e15ed3f8e1e7ed3
Dataset license: Apache-2.0, declared in its dataset card. The license text
is included as `LICENSE-RFUAV.txt`.

| Example | Archive and original member | Start sample |
| --- | --- | ---: |
| signal_01.iq | DJI MINI3.rar / DJI MINI3/VTSBW=10/pack2_4-5s.iq | 21000000 |
| signal_02.iq | DJI FPV COMBO.rar / DJI FPV COMBO/VTSBW=20/pack3_15-16s.iq | 21000000 |
| signal_03.iq | DJI AVATA2.rar / DJI AVATA2/VTSBW=10/pack1_10-11s.iq | 21000000 |

Each excerpt has 1000000 complex samples and preserves the source sample
values in interleaved little-endian float32 form. Source and excerpt hashes
are recorded in `examples.json`.

The full source hashes are respectively:

- Mini 3: 99c1c45271bd3b6bde7123c54ebd4507d7ad26fae35e1eb985c2ff638d6b064d
- FPV Combo: 610f1c4666a4d1c83e12bf956c2dbed7ab5894b83936353fd8bcc84520beb97c
- Avata 2: 88f7f73450d08c1568043efe0e6b51fad1d8e74c466afa3583d9e0d9f576771d

For Mini 3, 100 MS/s is adopted from corpus documentation and adjacent
pack1.xml. That sidecar is not capture-specific evidence for pack2.

## DRFF R2

Haolin Zheng, Ning Gao, Zhenghang Zhu, Zhijun Huang, Shi Jin and Michail
Matthaiou. A Multi-Scenario UAV RF Dataset with Real-World Acquisition and
Signal Processing Benchmarking, 2026. https://arxiv.org/abs/2603.00106

Dataset: https://doi.org/10.57760/sciencedb.36815
Data license: CC BY 4.0. The authors explicitly state this in sections 2,
3 and 5: https://arxiv.org/html/2603.00106v1
License: https://creativecommons.org/licenses/by/4.0/legalcode

`signal_04.iq` contains the first 1000000 samples of the environmental
background record downloaded from:
https://china.scidb.cn/download?fileId=815bc6cd2dc1b0c91ee9c2fb93c355c8

The excerpt combines the MAT record's RF0_I and RF0_Q float32 arrays as
interleaved I/Q, without resampling. Fs is 100 MS/s and the recorded center
frequency is 5745 MHz. This is a format conversion and excerpt, not an
unmodified copy of the complete MAT archive. No endorsement is implied.

## Nominal family labels

Labels follow the source equipment's published transmission specification:

- Mini 3: https://www.dji.com/mini-3/specs
- FPV: https://www.dji.com/newsroom/news/dji-reinvents-the-drone-flying-experience-with-the-dji-fpv
- Avata 2: https://www.dji.com/avata-2/specs

These labels do not establish a packet-decoded negotiated protocol version.
