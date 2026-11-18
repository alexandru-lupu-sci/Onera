# Onera: Interactive Sleep EEG Analysis Suite

**Onera** is an interactive Python suite for sleep EEG and polysomnography (PSG) staging and analysis, built with [Marimo](https://marimo.io/). It combines two complementary reactive applications covering the full workflow from automated whole-night sleep scoring to targeted pre-awakening brain-state analysis:

- **`Onera_SleepScorer.py`** — Automated $200\text{ Hz}$ whole-night sleep staging, multi-electrode soft-voting consensus, protocol-based Wake interval overrides, interactive hypnogram triage with $<1\text{ ms}$ pre-filtered waveform inspection, and `.Markers`/`.txt` export.
- **`Onera_PreAwakening.py`** — Whole-night sleep macro-architecture and targeted pre-awakening EEG analysis, including stage progression, $\pm 20\text{ s}$ buffer-padded FIR filtering, automated artifact rejection, Welch PSD bandpower, whole-window/time-resolved/pre-vs-post-stimulus scalp topomaps, power-domain STFT spectrograms, and master `.csv` export.

---

## 1. `Onera_SleepScorer.py` — Automated Sleep Staging & Interactive Triage

### Core Features & Methodology

- **Automated $200\text{ Hz}$ YASA Staging & Re-Referencing:** Loads BrainVision triplets (`.vhdr`, `.vmrk`, `.eeg`), automatically classifies physiological channels (`EEG`, `EOG`, `EMG`, `ECG`, `Misc`), applies optional mastoid/custom EEG re-referencing, and resamples once to $200\text{ Hz}$ before running YASA's pre-trained LightGBM classifiers on $30\text{-s}$ epochs (prioritizing submental/chin EMG over ocular or facial EMG derivations).
- **Multi-Electrode Soft-Voting Consensus:** Optionally scores across multiple EEG derivations (*Preset 1: Frontal-Central Quad `[F3, F4, C3, C4]`*, *Preset 2: Contralateral Pair `[F4, C3]`*, or *Custom EEG Channels*) and averages stage probability distributions across passes (soft voting) to determine each epoch's consensus sleep stage, confidence ($\%$), inter-electrode agreement ($\%$), and top-2 probability margin ($\Delta\%$).
- **Wake Interval Overrides:** Forces epochs whose midpoint precedes a configurable `LightsOff` anchor marker or falls within matched Start/Stop ignore marker pairs (e.g., `Start_DreamReport` $\rightarrow$ `Stop_DreamReport`) to Wake (`W`), with an option to exclude forced-Wake epochs from low-confidence flags.
- **Interactive Hypnogram Triage:** Flags epochs below a configurable confidence threshold (default $60\%$) with diagnostic tags (low agreement $<80\%$, stage transitions, narrow margin $<15\%$, N1 uncertainty). Supports horizontal scroll-zoom and pan, multi-epoch selection via individual clicks or box/lasso selection, a `Show All Epochs` toggle to inspect or edit any YASA-scored epoch alongside flagged epochs, and instant stage assignment (`W`, `N1`, `N2`, `N3`, `S4`, `REM`, `ART`) that immediately turns edited epochs green and auto-unselects them.
- **Pre-Filtered Waveform & Probability Inspector:** Pre-filters all active scoring EEG channels plus EOG ($0.3\text{–}35\text{ Hz}$) and EMG ($10\text{–}70\text{ Hz}$) once across the night and caches a $200\text{ Hz}$ `float32` $\mu\text{V}$ array for $<1\text{ ms}$ epoch slicing (displaying $\pm 37.5\text{ }\mu\text{V}$ slow-wave reference lines on EEG channels) alongside per-stage ensemble probability bar charts.
- **Export Formats:** Exports scored hypnograms as BrainVision `.Markers` files (in **AASM** or **R&K** notation, optionally retaining original `.vmrk` events) or 2-column numeric `.txt` files (`W=0, N1=1, N2=2, N3=3, S4=4/3, REM=5`).

---

## 2. `Onera_PreAwakening.py` — Pre-Awakening & Whole-Night EEG Analysis

### Core Features

- **Whole-Night Hypnograms & Macro-Statistics:** Plots whole-night sleep architecture (supporting both **AASM** and **R&K** scoring) aligned to 24-hour clock time. Automatically computes standard clinical sleep metrics (Total Sleep Time, Sleep Efficiency, WASO, Sleep Onset Latency, and stage percentages) via `YASA`.
- **Pre-Awakening Stage Progression:** Extracts the chronological sequence of sleep stages, the final sleep stage prior to awakening, the dominant stage, and the exact time elapsed between experimental triggers (e.g., auditory stimulation) and each awakening.
- **Signal Pre-Processing:** Loads BrainVision files (`.vhdr`, `.vmrk`, `.eeg`), automatically classifies channel types (`EEG`, `EOG`, `EMG`, `ECG`, and `Misc` sensors such as Resp/SCR), allows manual exclusion of detached/noisy electrodes, re-references cortical channels, and applies FIR bandpass and notch filters with before/after PSD validation plots.
- **Automated Artifact Rejection:** Epochs the pre-awakening window and removes segments containing flatlines, high peak-to-peak voltage excursions, fast voltage steps ($\mu\text{V/ms}$), or high-frequency EMG bursts. Reports per-channel artifact rates across EEG and EMG channels so faulty electrodes are easy to identify.
- **Spectral Power (PSD):** Computes absolute ($\mu\text{V}^2$) and relative ($\%$) bandpower across clinical sleep bands (*SWA, Theta, Alpha, Slow Spindles, Fast Spindles, Beta*) or standard EEG bands (*Delta, Theta, Alpha, Beta, Gamma*) using Welch's method, with selectable **Mean** or **Median** (robust) aggregation across clean epochs.
- **Whole-Window & Time-Resolved Scalp Topomaps:**
  - *Whole-Window Topography:* Displays 2D scalp maps averaged across the full pre-awakening window.
  - *Interactive Time-Resolved Topography:* Provides a sliding time window ($15\text{ s}$ to $240\text{ s}$) with a synchronized timeline showing sleep stages, experimental markers, and rejected artifact epochs. Supports both fixed colorbar scaling (to track amplitude changes over time) and dynamic scaling (to maximize spatial contrast at a specific moment).
  - *Split-Screen Pre- vs. Post-Stimulus Comparison:* Automatically locks onto the most recent experimental stimulus trigger to compare Baseline vs. Stimulation topographies side-by-side (excluding epochs straddling stimulus onset), supporting both global fixed scaling and pair-locked dynamic scaling.
- **Time-Frequency Spectrograms:** Computes multi-channel Short-Time Fourier Transform (STFT) spectrograms with external marker labels and artifact indicators.
- **Session Master Export:** Combines pre-awakening sleep stage metadata and channel-level spectral power across all awakenings in a recording into a single table for `.csv` export.

### Signal Processing Methodology

- **Channel-Specific Filtering:** Cortical EEG and EOG channels are bandpass-filtered to isolate cerebral activity (default $0.3\text{–}45\text{ Hz}$). EMG channels are filtered separately across the muscle passband ($10\text{–}90\text{ Hz}$) so that slow respiratory or sweat artifacts do not trigger false EMG rejections.
- **Edge-Artifact Prevention:** Before filtering each pre-awakening segment, Onera extracts an extra $\pm 20\text{ s}$ buffer of continuous raw signal on both sides of the window. After FIR bandpass and notch filtering (including harmonics below Nyquist), the $20\text{ s}$ buffers are cropped away so that filter edge ringing never enters the analysis window.
- **Standardized Welch Estimation:** Spectral power is estimated on clean epochs using a 4-second Hamming window ($0.25\text{ Hz}$ frequency resolution). When relative power ($\%$) is selected in the Clinical Sleep preset, each band is divided by the total broadband power ($0.5\text{–}25\text{ Hz}$); note that because *Alpha* ($8\text{–}12\text{ Hz}$) and *Slow Spindles* ($11\text{–}13\text{ Hz}$) share the $11\text{–}12\text{ Hz}$ bin, row sums across all clinical bands can slightly exceed $100\%$.
- **Power-Domain Spectrogram Averaging:** When multiple channels are selected for the spectrogram, the STFT power matrix is computed for each channel individually before averaging across channels and converting to decibels ($\text{dB}$). This avoids the phase cancellation that occurs when averaging raw time-domain signals across electrodes.

---

## Expected File Structure

Onera scans two local directories: one for raw BrainVision recordings (`.vhdr`, `.vmrk`, `.eeg`) and one for scored marker files (`.Markers` or `.vmrk`). Files should follow this naming convention:

```text
Participant_Condition_OptionalID(.vhdr / .eeg / .vmrk / .Markers)
```

- **`Participant`**: Participant code (e.g., `DS01`, `DM02`)
- **`Condition`**: Night or experimental condition (e.g., `1`, `adapt`, `Exp`)
- **`OptionalID`**: Session timestamp or identifier *(optional)*

*Note: Onera automatically detects the marker sampling rate from the `.Markers` or `.vhdr` header, even if the markers were exported at a downsampled rate in BrainVision Analyzer.*

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

### 4. Launch the pipeline
From the VS Code integrated terminal, run:
```
uv run marimo edit
```
- A browser window will open. Select the program you want (`Onera_SleepScorer.py` or `Onera_PreAwakening.py`).

---
## Usage Tips

- **Toggle App View:** When using `marimo edit`, press `Ctrl + .` (or click the preview icon in the bottom-right corner) to switch between code view and the clean UI dashboard.

---

## Citation

If you use Onera in your research, please cite this software along with the core toolboxes it relies on. GitHub's **"Cite this repository"** button (right sidebar) reads [`CITATION.cff`](CITATION.cff) and exports APA or BibTeX.

<!-- TODO after the first Zenodo release: add the DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

* **Onera:** Lupu, A. (2026). *Onera: Interactive Sleep EEG Analysis Pipeline* (Version 1.0.0) [Computer software]. University of Fribourg. https://github.com/alexandru-lupu-sci/Onera
* **MNE-Python:** Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
* **YASA:** Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated sleep staging in Python. *eLife*, 10, e70092. [doi:10.7554/eLife.70092](https://doi.org/10.7554/eLife.70092)

---

## Author

**Alexandru Lupu**  
University of Fribourg, Switzerland  
ORCID: [0009-0005-3320-7467](https://orcid.org/0009-0005-3320-7467)  
GitHub: [@alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)
