# Onera: Interactive Sleep EEG Analysis Suite

**Onera** is an interactive Python suite for sleep EEG and polysomnography (PSG) staging and analysis. It combines a standalone local web application (`FastAPI` + HTML5 Canvas) for sleep scoring and a [Marimo](https://marimo.io/) dashboard for targeted pre-awakening brain-state analysis:

- **`Onera_SleepScorer.py`** — Standalone whole-night automated (YASA), manual, and two-scorer consensus sleep staging across major EEG/PSG formats (`.vhdr`, `.edf`, `.bdf`, `.fif`, `.set`, `.gdf`), featuring persistent zero-flash canvases, native Full-Screen mode, Light / Night / Dark (OLED) themes, tile-cached native-rate waveforms, session resuming, and multi-format export.
- **`Onera_PreAwakening.py`** — Whole-night sleep macro-architecture and targeted pre-awakening EEG analysis with recursive recording discovery and a searchable **Recording** selector, $\pm 20\text{ s}$ buffer-padded composite FIR filtering with in-memory caching, automated artifact rejection, multi-threaded Welch PSD bandpower, whole-window/time-resolved/pre-vs-post-stimulus scalp topomaps, power-domain STFT spectrograms, and master `.csv` export.

> **Looking for signal-processing parameters, scoring consensus math, or file-format schemas?** See [`METHODOLOGY.md`](METHODOLOGY.md) for the complete technical reference.

---

## 1. `Onera_SleepScorer.py` — Automated, Manual & Consensus Sleep Staging

### Overview

- **Standalone Persistent Web Workspace:** Runs locally via `FastAPI` + `Uvicorn` with zero-flash persistent canvases, native Full-Screen mode (`⛶`), cross-OS Windows/macOS/Linux support, and **Light / Night / Dark (OLED)** themes.
- **Multi-Format EEG/PSG Loading:** Opens BrainVision (`.vhdr`), EDF/EDF+ (`.edf`), BioSemi (`.bdf`), MNE FIF (`.fif`, `.fif.gz`), EEGLAB (`.set`), and GDF (`.gdf`) recordings directly, automatically loading `.vmrk`, embedded annotations, or BIDS `*_events.tsv` events.
- **Automated & Manual Scoring:** Runs automated 30-second sleep staging with YASA (single-channel or multi-electrode consensus) or lets you score recordings manually at their native sampling rate.
- **Two-Scorer Consensus Mode:** Compares two scoring files epoch-by-epoch (even with identical filenames from separate folders), auto-validates union `+MA` tags with optional `+MA` difference flagging, displays Agreement % & Cohen's $\kappa$, overlays Scorer 2's differing segments on the hypnogram, offers one-click Scorer 1 / Scorer 2 adjudication plus optional YASA 3rd opinion & confidence tie-breaking, and warns on export when unresolved conflicts default to Scorer 1.
- **Wake Overrides & Event Markers:** Automatically marks pre-`LightsOff` segments and custom start/stop intervals (e.g., dream report windows) as Wake, while displaying recording events on both the hypnogram and waveforms.
- **Fast Interactive Hypnogram & Shortcuts:** Smooth zoom, trackpad/wheel pan, click, and box-selection across the night, paired with stage buttons (`W`, `N1`, `N2`, `N3`, `S4`, `REM`, `ART`), toggleable micro-arousals (`+MA`), undo (`Ctrl`/`Cmd`+`Z`), auto-advance, `Ctrl`/`Cmd`+Click multi-select, and single-key shortcuts (`0`–`3`, `5`, `F`, `G`, `X`, `←`/`→`).
- **Tile-Cached Binary Waveform Viewer:** Streams compact Int16 waveforms on demand with offscreen epoch tile caching for smooth 60 FPS scrolling/panning (`30s`/`60s`, 1- or 2-column, amplitude scaling, dynamic channel add/reorder), alongside YASA stage probabilities and scorer consensus controls.
- **Session Resuming & 6 Export Formats:** Resumes unfinished sessions with saved edits and `+MA` tags restored, and exports scored hypnograms in **AASM** or **R&K** notation to `.Markers`, `.txt`, BIDS `*_events.tsv`, MNE `*_annotations.csv`, Rich `*_hypnogram.csv`, or `.vmrk`.

---

## 2. `Onera_PreAwakening.py` — Pre-Awakening & Whole-Night EEG Analysis

### Core Features & Methodology

- **Flexible Recording Discovery & Whole-Night Macro-Statistics:** Recursively discovers `.Markers`/`.vmrk` and raw BrainVision `.vhdr` files across arbitrary filenames and folder structures (flat, per-participant, or BIDS) in a single searchable **Recording** dropdown, automatically pairing markers with raw EEG, aligning whole-night **AASM** or **R&K** hypnograms to 24-hour clock time, computing clinical sleep metrics via `YASA`, and extracting pre-awakening stage sequences and trigger latencies.
- **Fast Buffer-Padded Composite FIR Pre-Processing:** Classifies channels (`EEG`, `EOG`, `EMG`, `ECG`, `Misc`), supports bad-channel exclusion and direct EEG re-referencing, caches buffered ($\pm 20\text{ s}$) raw awakening windows in RAM so filter adjustments skip disk I/O, and applies precomputed composite zero-phase FIR bandpass ($0.3\text{–}45\text{ Hz}$ EEG/EOG; $10\text{–}90\text{ Hz}$ EMG) + harmonic notch kernels via multi-threaded overlap-add FFT convolution, verified with before/after PSD plots.
- **Instant Threshold-Driven Artifact Rejection:** Precomputes per-epoch metrics and multi-threaded Welch PSDs once per awakening so adjusting flatline, peak-to-peak amplitude, fast voltage step ($\mu\text{V/ms}$), or chin EMG burst sliders updates diagnostics, retention rates, and top artifact-triggering channels instantaneously without re-running FFTs.
- **Standardized Welch Bandpower (PSD):** Reuses precomputed clean-epoch Welch PSDs ($4\text{-s}$ Hamming window, $0.25\text{ Hz}$ resolution) for instant switching across **Mean/Median** aggregation, **Absolute ($\mu\text{V}^2$)/Relative ($\%$)** metrics, and Clinical Sleep (*SWA, Theta, Alpha, Slow/Fast Spindles, Beta*) or Standard EEG (*Delta–Gamma*) presets.
- **Scalp Topomaps & Power-Domain STFT Spectrograms:** Generates whole-window scalp maps, smooth interactive $15\text{–}240\text{ s}$ sliding-window topographies (with precomputed Fixed Scale limits and pre-filtered timeline overlays), and split-screen pre- vs. post-stimulus maps; computes multi-channel STFT spectrograms in parallel and averages in the linear power domain before $\text{dB}$ conversion.
- **Fast Session Master Export:** Consolidates pre-awakening sleep stage metadata and channel-level clean spectral bandpower across all awakenings in a recording into a single `.csv` table.

---

## Expected File Structure

Both applications work with **any filename or folder layout** (flat directories, per-participant subfolders, spaces, dates, hyphens, or BIDS hierarchies) and present recordings in a single searchable **Recording** dropdown:

- **`Onera_SleepScorer.py`:** Recursively scans your raw data folder for `.vhdr`, `.edf`, `.bdf`, `.fif`, `.fif.gz`, `.set`, and `.gdf` files (skipping hidden folders and companion files), automatically detecting `.vmrk`, embedded annotations, or BIDS `*_events.tsv` markers.
- **`Onera_PreAwakening.py`:** Recursively scans your raw BrainVision folder (`.vhdr`, `.vmrk`, `.eeg`) and markers folder (`.Markers` or `.vmrk`, which may be the same or separate folders) and automatically pairs each scored marker file with its matching raw `.vhdr` recording (exact relative path & stem first, followed by unique stem, separator-delimited prefix, and legacy `Participant_Condition_*` fallback).

*Note: Onera automatically detects marker/header sampling rates per active recording from `.vhdr`, `.Markers`, `.vmrk`, or embedded recording headers, even when marker files were exported at a different rate than the raw PSG.*

---

## Quick Start

Onera uses [`uv`](https://docs.astral.sh/uv/), a modern Python package and project manager. It automatically handles Python versions, virtual environments, and dependencies so you don't have to.

### 1. Install `uv`
If you don't have `uv` installed, open your terminal (PowerShell on Windows, or Terminal on macOS/Linux) and run:

**Windows:**
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
**MacOS / Linux:**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **Important:** After installing `uv`, **restart PowerShell** (close the window and open a new PowerShell window) so that `uv` is recognized in your system `PATH`.

### 2. Download Onera & Sync Dependencies
Open your new terminal window and navigate to where you want to save the project (e.g., your Documents folder).
```
cd Documents
```
Choose one of the methods below:

**Option A: Using [Git](https://git-scm.com/downloads) (Recommended)**
You can download it by clicking Git.
```
git clone https://github.com/alexandru-lupu-sci/Onera.git
```
```
cd Onera
```

**Option B: Direct Download (If you don't have Git)**
1. On the right side of this GitHub page, click on the **Releases** section.
2. Download the latest **Source code (zip)**.
3. Extract the `.zip` folder, open your terminal, and navigate into it (e.g. Documents folder. Replace "/Onera" with the actual name after unzip):

```
cd Documents/Onera
```

Next, create the project's virtual environment and install all required dependencies by running:
```
uv sync
```

- **Switch to VS Code:** Once `uv sync` finishes, **stop using the external PowerShell console** and close it. Open **VS Code** (`File → Open Folder...`) and select the `Onera` folder for all remaining steps.

### 3. Set Up VS Code (One-Time Setup)
1. **Install the Python extension in VS Code:** Open the Extensions sidebar (`Ctrl + Shift + X`), search for **Python** (by Microsoft), and click **Install**.
2. **Restart VS Code:** Close and reopen VS Code so the Python extension and the `.venv` environment created by `uv sync` are fully loaded.
3. **Select the Python interpreter in VS Code:** Open the Command Palette (`Ctrl + Shift + P`), type and select **`Python: Select Interpreter`**, and choose the project's virtual environment interpreter (`.\.venv\Scripts\python.exe` on Windows or `./.venv/bin/python` on macOS/Linux, typically listed as `('.venv': venv)`).
4. Open the integrated terminal inside VS Code (`` Ctrl + ` `` or **Terminal → New Terminal**).

Note: You might be in Restricted Mode (bottom left corner). If you are, click it and select Trust.

### 4. Launch the program
From the VS Code integrated terminal, run either application:

**Sleep Scorer (Standalone Web App):**
```bash
uv run python Onera_SleepScorer.py
```

**Pre-Awakening Analysis (Marimo App):**
```bash
uv run marimo edit Onera_PreAwakening.py
```

### 5. Optional: Generate Synthetic Demo Data
To test either app immediately without your own EEG recordings, generate a synthetic 12-channel BrainVision dataset locally (`demo_data/`, ~26 MB):
```bash
uv run python generate_demo_data.py
```
- Generates a 60-minute flat recording (`Demo01_Baseline_Night`) and a 30-minute nested BIDS recording (`sub-02/ses-01/...`, paired with a `LightsOff`-cropped `.Markers` file) covering `W`, `N1`, `N2`, `N3`, `REM`, `+MA` tags, `50 Hz` line noise, mastoid drift, and awakening/stimulus triggers (use `--recordings 1`, `--sfreq`, `--seed`, or `--out <folder>` to customize).
- In **`Onera_SleepScorer.py`**, set **Raw EEG / PSG Folder** to `demo_data`; in **`Onera_PreAwakening.py`**, set both **Raw Data Path** and **Markers Path** to `demo_data`.

---
## Usage Tips

- **Toggle App View:** When using `marimo edit`, press `Ctrl + .` (or click the preview icon in the bottom-right corner) to switch between code view and the clean UI dashboard.

---

## Update to latest version

If you wish to update your local files to the latest GitHub release, run this command in your console:
```
git pull
uv sync
```

If you modified the files locally, or if the previous command gives an error, run this command:
```
git fetch origin
git reset --hard origin/main
uv sync
```
---

## Citation

If you use Onera in your research, please cite this software along with the core toolboxes it relies on. GitHub's **"Cite this repository"** button (right sidebar) reads [`CITATION.cff`](CITATION.cff) and exports APA or BibTeX.

<!-- TODO after the first Zenodo release: add the DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

* **Onera:** Lupu, A. (2026) Onera: Interactive Sleep EEG Analysis Suite (Version 1.0.0) [Computer software]. https://github.com/alexandru-lupu-sci/Onera
* **MNE-Python:** Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
* **YASA:** Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated sleep staging in Python. *eLife*, 10, e70092. [doi:10.7554/eLife.70092](https://doi.org/10.7554/eLife.70092)

---

## Author

**Alexandru Lupu**  
University of Fribourg, Switzerland  
ORCID: [0009-0005-3320-7467](https://orcid.org/0009-0005-3320-7467)  
GitHub: [@alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)
