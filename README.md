# Onera: Interactive Sleep EEG Analysis


Onera is an interactive, Python-based analytical notebook for sleep EEG and polysomnography (PSG) data. Built on the [Marimo](https://marimo.io/) framework, it provides an end-to-end pipeline for macro-sleep architecture, quantitative EEG (qEEG) spectral analysis, and pre-awakening micro-state profiling.

---

## Key Features

- **Macro-Sleep Architecture (Hypnogram):** Whole-night hypnogram plotting with automated 24-hour real-time axis synchronization and customizable, staggered event marker overlays (e.g., auditory stimulation, awakenings).
- **Pre-Awakening Stage Profiling:** Micro-state analysis isolating a configurable window (default: 10 minutes) prior to targeted awakenings, complete with stimulation duration tracking.
- **Interactive Pre-Processing:** Flexible bandpass (FIR) and powerline notch filter configurations, combined with automated mastoid re-referencing (dynamically configurable, default: `M1, M2`).
- **Automated Artifact Rejection:** Multi-criterion 4-second overlapping epoch filtering (flatline detection, peak-to-peak amplitude thresholds, voltage step spikes, and high-frequency EMG activity).
- **Quantitative Spectral Analysis (qEEG):** Absolute power spectral density computation across clinical frequency bands (SWA, Theta, Alpha, Slow/Fast Spindles, Beta) via Welch's method (`YASA`), exportable directly to CSV with participant and condition tracking.
- **Time-Frequency Analysis:** High-resolution Short-Time Fourier Transform (STFT) spectrograms with adjustable window sizes, frequency ceilings, channel averaging, and synchronized power colorbars.

---

## 📁 Expected File Nomenclature

Onera automatically scans raw data and marker directories using standard BrainVision file patterns:

```text
XXXX_YY_ZZZZ(.eeg / .vhdr / .vmrk / .Markers)
```

- **`XXXX`**: Participant identifier (e.g., `DS01`, `DM05`)
- **`YY`**: Condition or night identifier (e.g., `2`, `adapt`, `exp`)
- **`ZZZZ`**: Acquisition timestamp or session ID (not strictly required for processing)

---

## Quick Start

### 1. Prerequisites

Ensure you have Python 3.9+ installed on your system.

### 2. Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/alexandru-lupu-sci/Onera.git
cd Onera
pip install -r requirements.txt
```

### 3. Launching Onera

Run Onera as an interactive Marimo app:

```bash
marimo edit Onera.py
```

---

## 💡 Usage Tips

- **App View:** For the best visual experience, click the **App View** toggle in the bottom right corner of the Marimo interface to hide the underlying code cells.
- **Global Configuration:** At the top of the app, configure your **Awakening Marker** and **Analysis Window**. All downstream cells (epoching, spectrograms, and sleep stages) will instantly anchor and recalculate based on these settings.
- **Data Export:** The Spectral Power and Sleep Stage tables can be exported directly to `.csv` format via the download button located in the upper right corner of the table UI.

---

## 👤 Author

**Alexandru Lupu**  
GitHub: [@alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)
