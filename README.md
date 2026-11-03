# Onera: Interactive Sleep EEG Analysis Pipeline

**Onera** is an interactive, Python-based analytical pipeline designed for sleep EEG and polysomnography (PSG) data. Built on the [Marimo](https://marimo.io/) reactive notebook framework, it bridges the gap between exploratory visual quality control and rigorous, batch-ready quantitative analysis.

The pipeline specializes in **pre-awakening micro-state profiling**, isolating the exact chronological sleep stages, spatial topographies, and time-frequency dynamics occurring immediately prior to targeted awakenings (e.g., for serial awakening and dream report studies).

---

## 🔬 Scientific Rigor & Signal Processing

Onera is built to clinical and research qEEG specifications, prioritizing mathematical accuracy and artifact prevention:

- **Modality-Specific Pre-Processing:** Automatically classifies cortical EEG, EOG, chin EMG, ECG, and respiratory channels prior to re-referencing and filtering. Cortical EEG/EOG channels are bandpass-filtered to isolate cerebral activity (default - but easily customizable `0.3–40 Hz`), while surface EMG is independently filtered across the high-frequency muscle passband (`10–90 Hz`) to ensure muscle-tone artifact rejection is not confounded by slow respiratory or sweat drift.
- **Boundary Artifact Prevention:** Applies 20-second reflection padding prior to FIR bandpass and Notch (plus harmonic) filtering, cropping the padded edges afterward to eliminate filter ringing within the analysis window.
- **Standardized Welch's PSD:** Decouples the Welch FFT window (capped at a 4-second Hamming window with 50% overlap for consistent `0.25 Hz` frequency resolution) from the artifact-rejection epoch length, computing spectral estimates natively across 3D clean epochs before integrating absolute ($\mu\text{V}^2$) or relative (%) bandpower via `YASA`.
- **Phase-Immune Spectrograms:** Utilizes **Power-Domain Averaging** for multi-channel Short-Time Fourier Transforms (STFT). By computing power matrices independently per channel prior to averaging and log-transformation (dB), Onera prevents the destructive phase cancellation inherent to time-domain averaging.

---

## ⚡ Core Features

- **Macro-Sleep Architecture (Hypnograms):** Whole-night hypnogram visualization (supporting both **AASM** and **R&K** scoring conventions) with 24-hour real-time clock alignment, shaded pre-awakening windows, staggered abbreviated event tracks, and a dedicated right-hand marker legend.
- **Pre-Awakening Stage Profiling:** Chronological micro-state extraction isolating a configurable window (e.g., 10 minutes) prior to each targeted awakening, including last stage at awakening, dominant stage classification, and stimulus-to-awakening latency tracking.
- **Interactive Pre-Processing:** Automated BrainVision loading, dynamic mastoid or custom re-referencing (e.g., `M1, M2`), customizable FIR bandpass/notch filtering (`MNE-Python`), and Before/After cortical PSD validation plots.
- **Automated Artifact Rejection:** Multi-criterion epoch thresholding that adapts to the recording's native sampling rate to reject flatlines (bridged/detached electrodes), peak-to-peak voltage excursions, fast voltage steps, and high-frequency EMG bursts.
- **Quantitative Spectral Analysis & Topomaps:** Absolute ($\mu\text{V}^2$) and Relative (%) power spectral density computation across clinical sleep bands (`SWA, Theta, Alpha, Slow/Fast Spindles, Beta`) or standard EEG bands, accompanied by 10–20 scalp topomaps.
- **Batch Export Analytics:** Easy `.csv` exports for single-awakening tables as well as a **Whole-Night Batch Summary** that aggregates clean epoch counts, sleep stages, and channel-wise spectral bandpower across all awakenings simultaneously.

---

## 📂 Expected File Nomenclature

Onera automatically scans raw data and marker directories for BrainVision files (`.eeg`, `.vhdr`, `.vmrk`, and exported BrainVision Analyzer `.Markers` files):

```text
XXXX_YY_ZZZZ(.eeg / .vhdr / .vmrk / .Markers)
```

- **`XXXX`**: Participant identifier (e.g., `DS01`, `DM05`)
- **`YY`**: Condition or night identifier (e.g., `1`, `2`, `adapt`, `exp`)
- **`ZZZZ`**: Acquisition timestamp or session ID *(optional)*

---

## Download

### 1. Prerequisites
Ensure you have [**Python 3.10+**](https://www.python.org/downloads/) installed on your system.

### 2. Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/alexandru-lupu-sci/Onera.git
cd Onera
pip install -r requirements.txt
```

### 3. Launching Onera
Run Onera as an interactive Marimo application in your browser:

```bash
marimo edit Onera-v1.8.py
```

---

## 💡 Usage Tips

- **App View:** Click the **App View** toggle in the bottom-right corner (`Ctrl + .`) to hide the underlying code cells and interact purely with the dashboard UI.
- **Global Reactivity:** The pipeline is fully reactive. Modifying the **Pre-Awakening Analysis Window** or **Awakening Marker** automatically updates all downstream sleep stage tables, filters, epoching, topomaps, and spectrograms.
- **Pause Execution:** When changing multiple parameters at once (such as switching both Participant and Condition), click the **Stop** (square) button in the bottom-right corner to pause reactivity, make your selections, and press **Run** (play button) to execute the updated configuration.

---

## 📖 Citation

If you use Onera in your published research, please cite this repository alongside the core analytical toolboxes it utilizes:

- **MNE-Python:** Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
- **YASA:** Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated sleep staging in Python. *eLife*, 10, e70092.

*(A Zenodo DOI for citing the Onera pipeline directly will be provided upon release).*

---

## 👤 Author

**Alexandru Lupu**  
GitHub: [@alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)
