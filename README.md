# Onera: Interactive Sleep EEG Analysis Pipeline

**Onera** is an interactive, Python-based analytical pipeline designed for sleep EEG and polysomnography (PSG) data. Built on the [Marimo](https://marimo.io/) framework, it bridges the gap between exploratory visual quality control and rigorous, batch-ready quantitative analysis. 

The pipeline specializes in **pre-awakening micro-state profiling**, isolating the exact chronological sleep stages and spectral dynamics occurring immediately prior to targeted awakenings (e.g., for dream report studies).

---

## 🔬 Scientific Rigor & Signal Processing

Onera is built to standard clinical qEEG specifications, prioritizing mathematical accuracy and artifact prevention:
- **Phase-Immune Spectrograms:** Utilizes **Power-Domain Averaging** for multi-channel Short-Time Fourier Transforms (STFT). By calculating power matrices independently before averaging, it entirely prevents the destructive phase cancellation common in time-domain averaging.
- **Boundary Artifact Prevention:** Applies dynamic time-padding prior to FIR and Notch filtering to eliminate edge artifacts (filter ringing) from the analysis window.
- **Optimized Welch's PSD:** Frequency resolution is dynamically tied to the user's artifact rejection epoch length, natively handling spectral smoothing (overlap) prior to final absolute bandpower integration via `YASA`.

---

## ⚡ Core Features

- **Macro-Sleep Architecture (Hypnograms):** Whole-night hypnogram plotting with automated 24-hour real-time axis synchronization and staggered event marker overlays (e.g., auditory stimulations, awakenings).
- **Pre-Awakening Stage Profiling:** Chronological micro-state extraction isolating a configurable window (e.g., 10 minutes) prior to targeted awakenings, complete with automated event duration tracking.
- **Interactive Pre-Processing:** Automated BrainVision loading, dynamic mastoid re-referencing (default: `M1, M2`), and customizable bandpass/powerline filtering (`MNE-Python`).
- **Automated Artifact Rejection:** Multi-criterion overlapping epoch filtering. Dynamically adapts to any hardware sampling rate to remove flatlines, extreme peak-to-peak amplitudes, voltage step spikes, and high-frequency EMG noise.
- **Quantitative Spectral Analysis (qEEG):** Absolute power spectral density computation across clinical (SWA, Theta, Alpha, Spindles, Beta) or standard EEG frequency bands.
- **Export-Ready Analytics:** 1-click `.csv` exports for all qEEG power tables and sleep stage sequences, automatically tracked by Participant ID and Condition.

---

## 📁 Expected File Nomenclature

Onera automatically scans raw data and marker directories using standard BrainVision file patterns:

```text
XXXX_YY_ZZZZ(.eeg / .vhdr / .vmrk / .Markers)
```

- **`XXXX`**: Participant identifier (e.g., `DS01`, `DM05`)
- **`YY`**: Condition or night identifier (e.g., `1`, `2`, `adapt`, `exp`)
- **`ZZZZ`**: Acquisition timestamp or session ID (not strictly required)

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Installation
Clone this repository and install the required dependencies (which include `mne`, `yasa`, `scipy`, and `marimo`):

```bash
git clone https://github.com/alexandru-lupu-sci/Onera.git
cd Onera
pip install -r requirements.txt
```

### 3. Launching Onera
Run Onera as an interactive Marimo app in your browser:

```bash
marimo edit Onera.py
```

---

## 💡 Usage Tips

- **App View:** Click the **App View** toggle in the bottom right corner of the interface to hide the underlying code cells and interact purely with the UI.
- **Global Reactivity:** The pipeline is fully reactive. Modifying the **Analysis Window** or **Awakening Marker** at the top of the app will instantly anchor and recalculate all downstream filtering, epoching, and spectrograms.
- **Pause Execution:** If changing multiple major parameters (like switching to a new participant), click the **Stop** (square) button in the lower right to pause the reactivity, make your selections, and press **Play** (or Enter) to execute the new pipeline.

---

## 📚 Citation

If you use Onera in your published research, please cite this repository alongside the core analytical engines it utilizes:
* **MNE-Python:** Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*.
* **YASA:** Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated sleep staging in Python. *eLife*.

*(A Zenodo DOI for citing the Onera pipeline directly will be provided in future releases).*

---

## 👤 Author

**Alexandru Lupu**  
GitHub: [@alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)