# Benchmark

**Test environment:** Linux 7.0.0-38-generic | CPU: AMD Ryzen 7 6800HS with Radeon Graphics (16 threads) | RAM: 14.3 GB | Python 3.14.4 | OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4 | dot - graphviz version 14.1.2 (0) | graphviz (Python) 0.21

| Model | Passes | Nodes / edges before | Nodes / edges after | Load + build graph (s) | Passes (s) | Render SVG (s) |
|---|---|---|---|---|---|---|
| small_cnn | (none) | 27 / 29 | 27 / 29 | 0.12 | 0.000 | 0.02 |
| small_cnn | hide-constants | 27 / 29 | 16 / 17 | 0.00 | 0.000 | 0.02 |
| small_cnn | hide-constants,highlight | 27 / 29 | 16 / 17 | 0.00 | 0.000 | 0.02 |
| mini_attention | (none) | 41 / 46 | 41 / 46 | 0.00 | 0.000 | 0.02 |
| mini_attention | hide-constants | 41 / 46 | 29 / 34 | 0.00 | 0.000 | 0.03 |
| mini_attention | hide-constants,highlight | 41 / 46 | 29 / 34 | 0.00 | 0.000 | 0.02 |
| mini_attention | fold-converts,hide-constants,highlight | 41 / 46 | 21 / 26 | 0.00 | 0.000 | 0.02 |
| tinyllama | (none) | 3168 / 3741 | 3168 / 3741 | 0.13 | 0.000 | 590.47 |
| tinyllama | hide-constants | 3168 / 3741 | 1911 / 2429 | 0.14 | 0.004 | 64.38 |
| tinyllama | hide-constants,highlight | 3168 / 3741 | 1911 / 2429 | 0.14 | 0.005 | 69.96 |
| tinyllama | fold-converts,hide-constants,detach-fanout,highlight | 3168 / 3741 | 1770 / 2025 | 0.15 | 0.008 | 0.54 |
