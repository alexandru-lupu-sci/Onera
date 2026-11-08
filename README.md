# Onera: Interactive Sleep EEG Analysis Pipeline

**Onera** is an interactive Python pipeline for sleep EEG and polysomnography (PSG) analysis, built with [Marimo](https://marimo.io/). It is designed to analyze brain activity in the minutes leading up to targeted awakenings (such as serial awakening and dream report paradigms), while also providing whole-night sleep architecture summaries.

---

## Core Features

- **Whole-Night Hypnograms & Macro-Statistics:** Plots whole-night sleep architecture (supporting both **AASM** and **R&K** scoring) aligned to 24-hour clock time. Automatically computes standard clinical sleep metrics (Total Sleep Time, Sleep Efficiency, WASO, Sleep Onset Latency, and stage percentages) via `YASA`.
- **Pre-Awakening Stage Progression:** Extracts the chronological sequence of sleep stages, the final sleep stage prior to awakening, the dominant stage, and the exact time elapsed between experimental triggers (e.g., auditory stimulation) and each awakening.
- **Signal Pre-Processing:** Loads BrainVision files (`.vhdr`, `.vmrk`, `.eeg`), automatically classifies channel types (EEG, EOG, EMG, ECG, Misc), allows manual exclusion of detached/noisy electrodes, re-references cortical channels, and applies FIR bandpass and notch filters with before/after PSD validation plots.
- **Automated Artifact Rejection:** Epochs the pre-awakening window and removes segments containing flatlines, high peak-to-peak voltage excursions, fast voltage steps ($\mu\text{V/ms}$), or high-frequency chin EMG bursts. Reports per-channel artifact rates so faulty electrodes are easy to identify.
- **Spectral Power (PSD):** Computes absolute ($\mu\text{V}^2$) and relative ($\%$) bandpower across clinical sleep bands (*SWA, Theta, Alpha, Slow Spindles, Fast Spindles, Beta*) or standard EEG bands (*Delta, Theta, Alpha, Beta, Gamma*) using Welch's method, with selectable **Mean** or **Median** (robust) aggregation across clean epochs.
- **Whole-Window & Time-Resolved Scalp Topomaps:**
  - *Whole-Window Topography:* Displays 2D scalp maps averaged across the full pre-awakening window.
  - *Interactive Time-Resolved Topography:* Provides a sliding time window ($15\text{ s}$ to $240\text{ s}$) with a synchronized timeline showing sleep stages, experimental markers, and rejected artifact epochs. Supports both fixed colorbar scaling (to track amplitude changes over time) and dynamic scaling (to maximize spatial contrast at a specific moment).
- **Time-Frequency Spectrograms:** Computes multi-channel Short-Time Fourier Transform (STFT) spectrograms with external marker labels and artifact indicators.
- **Session Master Export:** Combines pre-awakening sleep stage metadata and channel-level spectral power across all awakenings in a recording into a single table for `.csv` export.

---

## Signal Processing Methodology

- **Channel-Specific Filtering:** Cortical EEG and EOG channels are bandpass-filtered to isolate cerebral activity (default $0.3\text{–}45\text{ Hz}$). Chin EMG channels are filtered separately across the muscle passband ($10\text{–}90\text{ Hz}$) so that slow respiratory or sweat artifacts do not trigger false EMG rejections.
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

## Installation

### 1. Prerequisites
Install **Python 3.11** or newer.

### 2. Installation
Clone the repository:

```bash
git clone https://github.com/alexandru-lupu-sci/Onera.git
cd Onera
```
Create a new environment:
On Windows:
```
python -m venv onera_env
onera_env\Scripts\activate
```
On macOS and Linux:
```
python3 -m venv onera_env
source onera_env/bin/activate
```

And install packages:
```
pip install -r requirements.txt
```


### 3. Launching the Pipeline
To open Onera in interactive edit/dashboard mode in your browser:

```bash
marimo edit Onera_Beta-v1.17.py
```

To launch Onera directly as a clean web application (code hidden by default):

```bash
marimo run Onera_Beta-v1.17.py
```

---
## Installation for Newbies

### 1. Install Miniforge (a light and effective environment and librarie manager)

1.1 Click this link to the Miniforge GitHub page : [Install Miniforge](https://github.com/conda-forge/miniforge)

1.2 Look at the side bar on the right, look for "Releases".

1.3 Click on the latest release (e.g. Release 26.9.4-2 (Latest)).

1.4 Click on the install file that corresponds to your PC and type of CPU architecture (e.g. for Windows with x86 architecture in 64 bits (most PCs) it's the "-Windows-x86_64.exe" file). Proceed with the installation steps.

### 2. Create environment, download files, and install requirements
2.1 Open Miniforge Prompt

2.2 Create a dedicated environment for Onera:
```
conda create --name onera_env python=3.11
```
2.3 Activate your environment:
```
conda activate onera_env
```
Now instead of "(base)C:\", you should see "(onera_env)C:\".

2.4 Download the Onera files from my GitHub:
- Click the green "Code" button and click "Download ZIP".
- Unzip the folder.
- Put this folder wherever you want in your PC.

2.5 Access the Onera folder in Miniforge:
Type this code in Miniforge (make sure to replace everything after "cd" with the path to where you just put the Onera folder:
```
cd C:\Users\YourName\Documents\Onera
```

2.6 Install the packages needed to run Onera (they are all already listed in the requirements.txt). Run this code:
```
pip install -r requirements.txt
```
Whenever asked (y/n)? type "y" and click ENTER.

### 3. Run Onera
```
python -m marimo edit Onera_Beta-v1.17.py
```

## To run Onera in general

- Open Miniforge Prompt
- Activate your environemnt (cf. 2.3)
- Insert the file path to the Onera folder (cf. 2.5)
- Run Onera (cf. 3)

### Tips:
If you already have an IDE (like Spyder, PyCharm, Visual Studio,...), you can also create a new project in your IDE, set the interpreter to the Conda environment you just created. Put the Onera.py file into your project folder. Inside your project click on the Console and type "marimo edit" and click on the Onera file to launch it.

---
## Usage Tips

- **Toggle App View:** When using `marimo edit`, press `Ctrl + .` (or click the preview icon in the bottom-right corner) to switch between code view and the clean UI dashboard.
- **Reactive Updates:** Marimo automatically updates all plots and tables whenever you change a parameter. If you want to change several settings at once without waiting for intermediate recalculations, click the **Stop** (square) button in the bottom-right corner, adjust your settings, and click **Run** (play).

---

## Citation

If you use Onera in your research, please cite this repository along with the core toolboxes it relies on:

- **MNE-Python:** Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
- **YASA:** Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated sleep staging in Python. *eLife*, 10, e70092.

---

## Author

**Alexandru Lupu**  
GitHub: [@alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)
