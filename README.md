# tomobell
# Classical Emulation of Quantum State Tomography and Bell Test

## Overview

This repository contains the data-analysis code used for a classical-light emulation experiment of quantum-state tomography and a Bell-test measurement. The scripts process camera-recorded light-pulse videos, select regions of interest, extract pulse intensities, count threshold-crossing events, calculate CHSH correlation values, reconstruct effective density matrices, and generate the plots used in the paper. The repository also includes processed CSV data, reconstructed density matrices, simulation code, and saved threshold plots.

## Requirements

The scripts use Python 3 with the following packages:

```bash
pip install numpy pandas matplotlib opencv-python
```

Some scripts use OpenCV graphical windows for manual frame or ROI selection, so they should be run in a local Python environment rather than a headless terminal.

## Repository structure

```text
tomobell/
├── Important_Data/
├── Simulations/
├── saved_threshold_plots_separate/
├── jolly_good_code.py
├── make_rho.py
├── manual_qst.py
├── plot_correct_plots.py
├── plot_part_2.py
├── plot_qst.py
├── plot_rho_3d.py
├── rho_constrained_qst.npy
├── rho_constrained_qst.txt
└── README.md
```

## File descriptions

### `plot_part_2.py`

**Description:**
Interactive analysis script for the Bell-test videos. It selects or loads Alice and Bob regions of interest, extracts intensity as a function of frame number, allows manual threshold adjustment, detects peaks above threshold, counts Alice-Bob coincidences, and saves the threshold/count data.

**Inputs:**

* `videos/1.mp4` through `videos/16.mp4`
* `rois.json`, if already created; otherwise the script creates it through manual ROI selection.

**Outputs:**

* `threshold_counts.csv`, containing thresholds, peak counts, coincidence counts, Alice-only counts, Bob-only counts, and peak-frame lists for each analyzer-angle setting.

---

### `jolly_good_code.py`

**Description:**
Calculates the Bell-test count values, correlation functions, and CHSH parameter (S) from the processed coincidence-count table.

**Inputs:**

* `jolly_good_data.csv`, containing the agreement/coincidence counts for the 16 analyzer-angle settings.

**Outputs:**

* Printed values of (N_{HH}), (N_{HV}), (N_{VH}), and (N_{VV}).
* Printed correlation values (E(a,b)), (E(a,b')), (E(a',b)), and (E(a',b')).
* Printed CHSH value (S).

---

### `plot_correct_plots.py`

**Description:**
Generates publication-style threshold plots for Alice and Bob separately for all 16 Bell-test videos. Each plot shows the normalized intensity as a function of frame number, the normalized threshold line, and the detected peaks.

**Inputs:**

* `videos/1.mp4` through `videos/16.mp4`
* `jolly_good_data.csv`
* `rois.json`

**Outputs:**

* 32 threshold plots saved in `saved_threshold_plots_separate/`

  * 16 Alice plots
  * 16 Bob plots

---

### `manual_qst.py`

**Description:**
Interactive script for extracting quantum-state-tomography pulse data from a QST video. It allows manual or preloaded pulse-frame selection, selects four camera ROIs, averages intensities over the selected pulse frames, and saves one row per pulse.

**Inputs:**

* A QST video, currently set in the script as `QST_videos/partB50.mp4`
* Optionally, an old pulse-selection CSV such as `qst_pulses_only_old.csv`

**Outputs:**

* `qst_pulses_only.csv`, containing pulse index, selected frames, averaged intensities for `Alice0`, `Alice1`, `Bob0`, and `Bob1`, total intensity, and normalized intensity fractions.

---

### `make_rho.py`

**Description:**
Reconstructs an effective density matrix from QST pulse-intensity data. Each camera is normalized by its maximum intensity, each pulse is classified as one of the basis states (|HH\rangle), (|HV\rangle), (|VH\rangle), or (|VV\rangle), and an averaged state vector is used to construct a density matrix.

**Inputs:**

* `good_data.csv`, containing the four camera intensities `Alice0`, `Alice1`, `Bob0`, and `Bob1`.

**Outputs:**

* `rho_average_state_method.txt`
* `rho_average_state_method.npy`
* Printed pulse counts, reconstructed state vector, density matrix, trace, eigenvalues, and purity.

---

### `plot_qst.py`

**Description:**
Plots the normalized QST pulse intensities for Alice and Bob. For each pulse, the script displays side-by-side bars for the two polarization outputs.

**Inputs:**

* `qst_pulses_only50_wow.csv`, or another QST pulse CSV with the same columns.

**Outputs:**

* A two-panel bar plot:

  * Alice normalized intensities
  * Bob normalized intensities

---

### `plot_rho_3d.py`

**Description:**
Creates a 3D bar plot of a manually specified (4 \times 4) density matrix in the basis (|HH\rangle, |HV\rangle, |VH\rangle, |VV\rangle).

**Inputs:**

* A (4 \times 4) matrix entered directly in the script.

**Outputs:**

* A PDF file of the 3D density-matrix plot, with the filename set by `title_figure`.
* A displayed Matplotlib figure.

---

### `Simulations/S_distribution.py`

**Description:**
Monte Carlo simulation of the CHSH parameter (S). The script can simulate either an entangled-state model or a classical-mixture model, repeat the experiment many times, and plot the resulting distribution of (S).

**Inputs:**

* No external files. Simulation parameters are set inside the script:

  * `N_PAIRS`
  * `N_RUNS`
  * `STATE_MODEL`
  * Alice and Bob analyzer angles

**Outputs:**

* Printed mean, standard deviation, median, minimum, maximum, and mean absolute value of (S).
* Printed fraction of runs with (|S|>2).
* `S_distribution_20_pulses.png`

---

## Data files

### `Important_Data/jolly_good_data.csv`

**Description:**
Processed Bell-test data table containing thresholds, detected peak counts, agreement counts, and peak-frame lists for the 16 analyzer-angle combinations.

**Used by:**

* `jolly_good_code.py`
* `plot_correct_plots.py`

---

### `Important_Data/qst_pulses_only10.csv`

**Description:**
Processed QST pulse-intensity data for the 10-pulse measurement.

**Used by:**

* `make_rho.py`
* `plot_qst.py`, after updating `CSV_PATH`

---

### `Important_Data/qst_pulses_only25.csv`

**Description:**
Processed QST pulse-intensity data for the 25-pulse measurement.

**Used by:**

* `make_rho.py`
* `plot_qst.py`, after updating `CSV_PATH`

---

### `Important_Data/qst_pulses_only50.csv`

**Description:**
Processed QST pulse-intensity data for the 50-pulse measurement.

**Used by:**

* `make_rho.py`
* `plot_qst.py`, after updating `CSV_PATH`

---

### `Important_Data/rho_average_state_method10.txt` and `Important_Data/rho_average_state_method10.npy`

**Description:**
Reconstructed effective density matrix and diagnostics for the 10-pulse QST data.

**Used by:**

* The text file can be read directly.
* The NumPy file can be loaded with `numpy.load`.

---

### `Important_Data/rho_average_state_method25.txt` and `Important_Data/rho_average_state_method25.npy`

**Description:**
Reconstructed effective density matrix and diagnostics for the 25-pulse QST data.

**Used by:**

* The text file can be read directly.
* The NumPy file can be loaded with `numpy.load`.

---

### `Important_Data/rho_average_state_method50.txt` and `Important_Data/rho_average_state_method50.npy`

**Description:**
Reconstructed effective density matrix and diagnostics for the 50-pulse QST data.

**Used by:**

* The text file can be read directly.
* The NumPy file can be loaded with `numpy.load`.

---

### `Important_Data/rois.json`

**Description:**
Saved regions of interest used to extract Alice and Bob intensities from the Bell-test videos.

**Used by:**

* `plot_part_2.py`
* `plot_correct_plots.py`

---

### `rho_constrained_qst.txt` and `rho_constrained_qst.npy`

**Description:**
Saved constrained QST density-matrix reconstruction.

**Used by:**

* The text file can be read directly.
* The NumPy file can be loaded with `numpy.load`.

---

## Output folders

### `saved_threshold_plots_separate/`

**Description:**
Saved threshold plots generated from the Bell-test video analysis.

**Contents:**

* Alice intensity-threshold plots.
* Bob intensity-threshold plots.

**Generated by:**

* `plot_correct_plots.py`

---

## Reproducing the analysis

### Bell-test analysis

1. Place the 16 Bell-test videos in a folder called `videos/`, named:

```text
1.mp4, 2.mp4, ..., 16.mp4
```

2. Run the interactive threshold script:

```bash
python plot_part_2.py
```

3. Adjust thresholds using the keyboard controls shown by the script. Closing the plot saves the threshold and coincidence data.

4. Calculate the CHSH value:

```bash
python jolly_good_code.py
```

5. Generate the saved intensity-threshold plots:

```bash
python plot_correct_plots.py
```

---

### QST analysis

1. Place the QST video in the path specified by `VIDEO_PATH` in `manual_qst.py`.

2. Run:

```bash
python manual_qst.py
```

3. Use the generated pulse CSV as the input for density-matrix reconstruction. Update `CSV_PATH` in `make_rho.py` if needed, then run:

```bash
python make_rho.py
```

4. To plot the QST pulse intensities, update `CSV_PATH` in `plot_qst.py` and run:

```bash
python plot_qst.py
```

5. To plot a reconstructed density matrix as a 3D bar plot, paste the matrix into `plot_rho_3d.py` and run:

```bash
python plot_rho_3d.py
```

---

## Angle ordering for the Bell-test videos

The Bell-test scripts assume the following order of analyzer settings:

```text
1:  a,    b
2:  a,    b'
3:  a,    b_p
4:  a,    b'_p
5:  a',   b
6:  a',   b'
7:  a',   b_p
8:  a',   b'_p
9:  a_p,  b
10: a_p,  b'
11: a_p,  b_p
12: a_p,  b'_p
13: a'_p, b
14: a'_p, b'
15: a'_p, b_p
16: a'_p, b'_p
```

In the current scripts this corresponds to the angle pairs:

```text
1:  (-45°, -22.5°)
2:  (-45°,  22.5°)
3:  (-45°,  67.5°)
4:  (-45°, 112.5°)
5:  (  0°, -22.5°)
6:  (  0°,  22.5°)
7:  (  0°,  67.5°)
8:  (  0°, 112.5°)
9:  ( 45°, -22.5°)
10: ( 45°,  22.5°)
11: ( 45°,  67.5°)
12: ( 45°, 112.5°)
13: ( 90°, -22.5°)
14: ( 90°,  22.5°)
15: ( 90°,  67.5°)
16: ( 90°, 112.5°)
```

## Notes

Large raw video files are not included in this repository. The repository contains the scripts and processed data needed to reproduce the numerical analysis and figures from the extracted pulse and threshold data.

