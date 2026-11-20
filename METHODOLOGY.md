# Onera — Technical & Methodological Reference

This document provides the complete signal-processing, scoring consensus, event-handling, artifact-rejection, spectral-analysis, UI interaction, and file-format specifications implemented in **Onera** (`Onera_SleepScorer.py` and `Onera_PreAwakening.py`). For a high-level overview and setup instructions, see [`README.md`](README.md).

---

## Table of Contents

- [Part 1 — `Onera_SleepScorer.py`: Automated & Manual Sleep Staging](#part-1--onera_sleepscorerpy-automated--manual-sleep-staging)
  - [1.1 Recording Discovery & Multi-Format Ingestion](#11-recording-discovery--multi-format-ingestion)
  - [1.2 Unified Recording Event Extraction](#12-unified-recording-event-extraction)
  - [1.3 Physiological Channel Classification, Re-Referencing & Resampling](#13-physiological-channel-classification-re-referencing--resampling)
  - [1.4 Automated Multi-Electrode Soft-Voting Consensus (YASA)](#14-automated-multi-electrode-soft-voting-consensus-yasa)
  - [1.5 Wake Interval Overrides & Low-Confidence Diagnostic Flagging](#15-wake-interval-overrides--low-confidence-diagnostic-flagging)
  - [1.6 Display Signal Filtering, Envelopes & HTML5 Canvas Architecture](#16-display-signal-filtering-envelopes--html5-canvas-architecture)
  - [1.7 Keyboard Shortcuts, Editing State, Session Resuming & Export Schemas](#17-keyboard-shortcuts-editing-state-session-resuming--export-schemas)
- [Part 2 — `Onera_PreAwakening.py`: Pre-Awakening & Whole-Night EEG Analysis](#part-2--onera_preawakeningpy-pre-awakening--whole-night-eeg-analysis)
  - [2.1 Recording Discovery, Raw/Marker Pairing & Whole-Night Staging](#21-recording-discovery-rawmarker-pairing--whole-night-staging)
  - [2.2 Optimized Buffer-Padded Signal Pre-Processing](#22-optimized-buffer-padded-signal-pre-processing)
  - [2.3 Decoupled Non-Overlapping Artifact Rejection](#23-decoupled-non-overlapping-artifact-rejection)
  - [2.4 Welch Spectral Power (PSD) & Frequency Band Presets](#24-welch-spectral-power-psd--frequency-band-presets)
  - [2.5 Scalp Topographies & Split-Screen Event Comparison](#25-scalp-topographies--split-screen-event-comparison)
  - [2.6 Power-Domain STFT Spectrograms & Master Export](#26-power-domain-stft-spectrograms--master-export)
- [Part 3 — Synthetic Demo Data Generator (`generate_demo_data.py`)](#part-3--synthetic-demo-data-generator-generate_demo_datapy)

---

## Part 1 — `Onera_SleepScorer.py`: Automated & Manual Sleep Staging

### 1.1 Recording Discovery & Multi-Format Ingestion

`Onera_SleepScorer.py` uses `scan_recordings` (`os.walk`) to recursively scan the configured raw-data directory—supporting flat folders, nested per-participant folders, and BIDS hierarchies with arbitrary filenames—while pruning hidden folders (e.g., `.onera_sessions`, `.git`), hidden files, companion binary/header sidecars (`.eeg`, `.vmrk`, `.fdt`), and standalone MNE annotation files:

- **Supported extensions (priority order for deduplicating identical `(folder, stem)` pairs):**
  1. `.vhdr` — BrainVision header (`mne.io.read_raw_brainvision`)
  2. `.edf` — EDF / EDF+ (`mne.io.read_raw_edf(..., infer_types=False)`)
  3. `.bdf` — BioSemi BDF (`mne.io.read_raw_bdf(..., infer_types=False)`)
  4. `.fif` / `.fif.gz` — MNE FIF (`mne.io.read_raw_fif(..., allow_maxshield="yes")`)
  5. `.set` — EEGLAB (`mne.io.read_raw_eeglab`, falling back to `preload=True` if lazy header/data loading fails)
  6. `.gdf` — GDF (`mne.io.read_raw_gdf`)
- **Excluded annotation files during directory scan:** `-hypnogram.edf`, `_hypnogram.edf`, `-annot.fif`, `_annot.fif`, `-annot.fif.gz`, `_annot.fif.gz`.
- **Deduplication, Labeling & Natural Sorting:** Recordings sharing the same folder and recording stem across multiple formats are deduplicated by `(normcase(parent), stem.lower())` using the extension priority above, while identical stems in distinct subfolders remain separate entries. Top-level recordings display as `<filename>`, nested recordings display their forward-slash relative path (`sub-01/ses-01/sleep.edf`), all entries are ordered with `natural_sort_key` (numeric runs sorted numerically, e.g., `DS2` before `DS10`), and exposed via a single searchable `recording_dropdown` (`searchable=True`).
- **Chunked Disk Streaming & Single-Session Cache (`session_data_cache`):**
  - Header inspection uses `preload=False` to obtain `sfreq`, `n_times`, `meas_date`, and `ch_names` without loading unused channels.
  - Only required channels (active EEG derivations, optional EOG/EMG, and selected references) are streamed in **600-second contiguous chunks** directly into a `float32` RAM array (in Volts).
  - The cache is keyed by `(filepath, mtime_ns, size_bytes)` and retains loaded raw channels, per-derivation YASA passes `(eeg, eog, emg, refs)`, and pre-filtered `float32` display waveforms across UI threshold/override re-runs for the active recording, clearing automatically when switching sessions.

---

### 1.2 Unified Recording Event Extraction

Non-stage markers are ingested into a unified chronological event table (`Type`, `Description`, `Position`, `Length`, `Channel`, `Date`):

1. **BrainVision `.vhdr`:** Resolves `MarkerFile=` from the `.vhdr` INI header (checking both relative and basename paths inside the `.vhdr` folder, then `<stem>.vmrk`, then a case-insensitive sibling `.vmrk` stem match) and parses `Mk<N>=` / CSV entries.
2. **Non-BrainVision formats (`.edf`, `.bdf`, `.fif`, `.set`, `.gdf`):**
   - Extracts embedded `raw_hdr.annotations` (`onset`, `duration`, `description`, `ch_names`), converting onset seconds to 1-based sample indices via $\text{pos} = \max(1, \operatorname{round}(\text{onset} \cdot f_s) + 1)$.
   - Also discovers sibling BIDS event tables (`<bids_base>_events.tsv` or `<stem>_events.tsv`) when present, reading `onset`, `duration`, `trial_type`/`description`/`value`, and `status`, and deduplicating identical `(Position, Description)` pairs.
3. **Display Filtering:** `Sleep Stage`, `New Segment`, and pre-sleep biocalibration markers matching `/eyes?|teeth|blink|calib|biocal/i` (`is_calibration_marker`) are excluded from overlay markers while preserved where appropriate for Wake override checks.

---

### 1.3 Physiological Channel Classification, Re-Referencing & Resampling

- **Channel Type Classification (`classify_sleep_channels`):** Evaluates each channel name in explicit priority order to prevent multi-token collisions (e.g., `EMG_eye`):
  1. **Reference / Misc (`misc`):** User-selected reference channels or canonical mastoids/earlobes (`M1`, `M2`, `A1`, `A2`, `TP9`, `TP10`, with optional leading `EEG ` prefix stripped).
  2. **EMG (`emg`):** Contains `EMG`, `CHIN`, `SUBMENT`, `MENTALIS`, `TIB`, or `LEG`.
  3. **EOG (`eog`):** Contains `EOG`, `LOC`, `ROC`, `HEOG`, `VEOG`, or token `E1`/`E2`.
  4. **ECG (`ecg`):** Contains `ECG`, `EKG`, `HEART`, or `PULSE`.
  5. **Auxiliary Sensors / Triggers (`misc`):** Respiration, thermistor/belts, snore/mic, SpO2/pleth, position, temperature, event/trigger/status, EDA/GSR, PPG, light sensor, battery, accelerometry/gyro, CPAP pressure.
  6. **EEG (`eeg`):** Default fallback for all remaining neural channels.
- **Selective Re-Referencing (`_assemble_referenced`):**
  - When reference channels $R$ (`len(R) >= 1`) are selected, the sample-wise mean reference $\bar{r}(t) = \frac{1}{|R|} \sum_{r \in R} x_r(t)$ (accumulated in `float64`) is subtracted from `eeg`-typed channels only (`x_c(t) - \bar{r}(t)`), leaving `eog`, `emg`, `ecg`, and `misc` untouched.
  - Pre-referenced bipolar recordings (e.g., `C4-M1`, `EEG Fpz-Cz`, `E1-M2`) are detected automatically in UI defaults (leaving the Reference selector empty unless explicit reference electrodes exist).
- **Resampling Pipeline:**
  - **Automated Mode (YASA):** Signals are polyphase-resampled via `scipy.signal.resample_poly` to **$200\text{ Hz}$** (display waveform preservation up to $70\text{ Hz}$ for EMG) using exact integer ratios (`Fraction(200, int(round(sf)))`, or rational approximation with denominator $\le 1000$ for non-integer rates). A **$100\text{ Hz}$** copy is produced for YASA via `resample_poly(..., 1, 2)` when $f_s \neq 100\text{ Hz}$ (or used directly at $100\text{ Hz}$).
  - **Manual Mode:** Resampling is skipped completely; all waveforms remain at the recording's **native sampling rate**.

---

### 1.4 Automated Multi-Electrode Soft-Voting Consensus (YASA)

In Automated Mode, each selected cortical EEG derivation $m \in \{1, \dots, M\}$ is scored independently over 30-second epochs using `yasa.SleepStaging` (`eeg_name`, plus optional `eog_name` and `emg_name`). Presets automatically map canonical targets (`[F3, F4, C3, C4]` or `[F4, C3]`) to matching unipolar or bipolar labels (stripping leading `EEG ` and suffixes `-(M1|M2|A1|A2|REF|AVG|LE|AR)`).

For each epoch $t$ and canonical stage $k \in \{\text{W}, \text{N1}, \text{N2}, \text{N3}, \text{REM}\}$:

$$\bar{p}_t(k) = \frac{1}{M} \sum_{m=1}^{M} p_{t,m}(k)$$

From the soft-averaged probability distribution $\bar{p}_t$:

- **Consensus Stage:** $\hat{s}_t = \arg\max_k \bar{p}_t(k)$
- **Confidence (%):** $C_t = 100 \cdot \bar{p}_t(\hat{s}_t)$
- **Inter-Electrode Agreement (%):** $A_t = \frac{100}{M} \sum_{m=1}^{M} \mathbf{1}\{\hat{y}_{t,m} = \hat{s}_t\}$
- **Decision Margin ($\Delta\%$):** $\Delta_t = 100 \cdot \left[\bar{p}_t(\hat{s}_t) - \max_{k \neq \hat{s}_t} \bar{p}_t(k)\right]$ (against the runner-up stage $k_{\text{top2}}$)

---

### 1.5 Wake Interval Overrides & Low-Confidence Diagnostic Flagging

Every session is partitioned into $N_{\text{ep}}$ contiguous **30-second epochs**. Each epoch $t$ starts at 1-based sample $p_{\text{start}}(t) = 1 + t \cdot L_{\text{ep}}$ (with $L_{\text{ep}} = \operatorname{round}(30 \cdot f_{\text{marker}})$) and has temporal midpoint $p_{\text{mid}}(t) = p_{\text{start}}(t) + L_{\text{ep}} / 2$.

- **Pre-Sleep `LightsOff` Override:** All epochs satisfying $p_{\text{mid}}(t) < p_{\text{LightsOff}}$ are forced to `W` (Automated Mode or Manual Mode initialization).
- **Ignore Marker Interval Pairing (`_match_marker_intervals`):** Events matching selected `Start` and `Stop` descriptions are sorted chronologically and paired greedily (non-overlapping). Any epoch with $p_{\text{start\_ev}} < p_{\text{mid}}(t) < p_{\text{stop\_ev}}$ is forced to `W`. Override intervals are shaded in amber (`rgba(243, 156, 18, 0.16)`) across the hypnogram.
- **Flagging & Diagnostic Reasons (Automated Mode):**
  An epoch is flagged when $C_t < \theta$ (default $\theta = 60\%$), unless `Exclude forced Wake overrides from flags` is enabled and the epoch has an active Wake override. Flagged epochs receive structured diagnostic tags:
  - `Forced Wake (...)` (Wake override active)
  - `Low Agreement (<perc>%)` when $A_t < 80\%$
  - `Transition (<prev>→<curr>→<next>)` when neighboring stages differ
  - `Narrow Margin (<curr> vs <runner>: Δ<delta>%)` when $\Delta_t < 15\%$
  - `N1 Uncertainty` when $\hat{s}_t = \text{N1}$ or $k_{\text{top2}} = \text{N1}$ and $C_t < \theta$
  - `Low Prob (<conf>%)` fallback when $C_t < \theta$ without the above triggers

---

### 1.6 Display Signal Filtering, Envelopes & HTML5 Canvas Architecture

#### Pre-Filtered Waveform Inspector (`_prepare_cached_waveforms`)
- Signals are scaled from Volts to $\mu\text{V}$ in `float64`, zero-phase filtered with 4th-order Butterworth SOS filters (`scipy.signal.butter(..., output="sos")` + `sosfiltfilt`), and cached in `float32`:
  - **EEG & EOG:** $0.3\text{–}\min(35.0, f_N - 1.0)\text{ Hz}$
  - **EMG:** $10.0\text{–}\min(70.0, f_N - 1.0)\text{ Hz}$
- **Amplitude Y-Limits:** Estimated from the 99th percentile of $|x|$ (strided by 10):
  - EMG: $\operatorname{clip}(1.3 \cdot P_{99},\ 15,\ 150)\,\mu\text{V}$
  - EEG / EOG: $\operatorname{clip}(1.3 \cdot P_{99},\ 75,\ 220)\,\mu\text{V}$
- **Exact Min-Max Envelope Downsampling:** Each 30-second segment with $>1200$ samples is compressed into **600 temporal bins** via `np.minimum.reduceat` and `np.maximum.reduceat` (interleaved `[min_0, max_0, ..., max_599]`), preserving true peak/trough spikes in $<1\text{ ms}$ on a `<canvas>` with $\pm 37.5\,\mu\text{V}$ EEG slow-wave guides and a `1 Col` / `2 Col` toggle.

#### 60 FPS HTML5 `<canvas>` Hypnogram & Marimo State Bridge
- **Rendering & Interaction:** Uses HiDPI `devicePixelRatio` scaling, `ResizeObserver`, and `requestAnimationFrame` coalescing:
  - Initial viewport spans **240 epochs (~2 hours)** (anchored from epoch 1 if initial epoch $\le 180$, else centered on the active epoch), remaining at full-night extent once reset via double-click or `Full Night`.
  - **Wheel Zoom:** Anchored to the cursor X coordinate with `deltaMode` normalization (`deltaMode === 1` scaled $\times 16$, `deltaMode === 2` scaled by canvas height).
  - **Pan & Box Select:** Click-drag panning in `Pan` mode, box selection in `Box Select` mode or via `Shift`+drag (selecting all visible/selectable epochs inside the dragged `[xMin, xMax] × [yLow, yHigh]` box).
  - **Dynamic Ticks & Label Decluttering:** X-axis ticks adapt cleanly across step candidates `[30s, 60s, 120s, 300s, 600s, 900s, 1800s, 3600s, 7200s, 14400s]` aligned to wall-clock time (when `meas_date` is available) or elapsed time. In Automated mode, bottom-center `Epoch #` labels dynamically declutter per stage row using pixel-width spacing so denser zoom reveals every individual epoch number.
  - **High-Event Binning:** When $>250$ recording events exist, rare landmarks (`is_lights` or descriptions with $\le 25$ occurrences) stay unbinned while dense repeated markers bin into $\sim 200$ intervals (`≤ 120` thinned vertical event lines).
- **Visual Artifact (`ART`) Representation:**
  To prevent movement artifacts from appearing as false awakening spikes at the top of the hypnogram, each `ART` epoch inherits the Y-height of the **most recent preceding scored non-`ART` epoch** (falling back to the next following scored non-`ART` epoch if none precedes it, or `W` `4.0` if no scored epochs exist). Consecutive `ART` epochs render as contiguous **red (`#dc2626`) horizontal segments** (`3.0px`, minimum `2.5px` width) with white-outlined crimson square markers while keeping exported `ART`/`MT` codes untouched.
- **Two-Way Non-Cascading Marimo Bridge (`hypno_action_bridge`):**
  A hidden `mo.ui.text(value="", debounce=False)` widget wrapped in `inert` + `visibility:hidden` receives JSON payloads (`click`, `box`, `viewport`) via `CustomEvent("marimo-value-input")`. No downstream cell depends on `hypno_action_bridge.value`, so canvas interactions trigger only incremental updates to the active epoch, hypnogram, and 30-second waveform inspector without full-page flicker or Tab-focus interference.

---

### 1.7 Keyboard Shortcuts, Editing State, Session Resuming & Export Schemas

- **Keyboard Shortcuts:** Captured globally (ignoring editable inputs/editors while explicitly permitting events from `#onera_hypno_bridge`):
  `0`=`W`, `1`=`N1`, `2`=`N2`, `3`=`N3`, `5`=`REM`, `F`=`ART`, `G`=`+MA`, `←`/`→`=`Prev/Next Epoch`.
- **State Store & Undo:** Edits (`get_manual_edits`), micro-arousal flags (`get_ma_flags`), undo stacks, and hypnogram viewports (`zoom_view_state`) are keyed off the selected recording's unique relative identifier (`committed_recording_id`, paired with `scoring_mode`); `Clear All` resets only the active recording/mode's edits while preserving an imported baseline in resumed Manual sessions.
- **Sidecar Session State (`.onera_sessions/<stem>.onera.json`):** Written alongside disk exports to store `stem`, `recording_id`, `scoring_mode`, `n_epochs`, `fs`, `manual_edits`, `ma_flags`, `final_stages`, and `yasa_config` (`active_eegs`, `eog`, `emg`, `refs`, `thresh`).
- **Stage Mapping & 6 Export Formats:**

| Internal Canonical Stage | AASM Label | R&K Label | 2-Col `.txt` Code (`AASM` / `R&K`) |
|---|---|---|---|
| `W` | `W` | `W` | `0` / `0` |
| `N1` | `N1` | `S1` | `1` / `1` |
| `N2` | `N2` | `S2` | `2` / `2` |
| `N3` | `N3` | `S3` | `3` / `3` |
| `S4` | `N3` | `S4` | `3` / `4` |
| `REM` | `REM` | `REM` | `5` / `5` |
| `ART` | `ART` | `MT` | `0` / `0` |
| `UNS` | `?` | `?` | `0` / `0` |

Supported exports (all preserving `+MA` where supported and optionally retaining original non-stage events):
1. **BrainVision `.Markers` (Analyzer CSV):** Header lines (`Sampling rate: X Hz, SamplingInterval: Y ms`, `Type, Description, Position, Length, Channel`) with `Sleep Stage, <STAGE>[+MA], <pos>, <len>, All`.
2. **2-Column Numeric `.txt`:** Tab-separated `<stage_code>\t<0|1_for_MA>`.
3. **BIDS Events (`*_events.tsv`):** Columns `onset`, `duration`, `trial_type`, `micro_arousal`, `confidence`, `status`.
4. **MNE Annotations (`*_annotations.csv`):** Generated via `mne.Annotations.to_data_frame()` with `orig_time` preserved when present.
5. **Rich Hypnogram Table (`*_hypnogram.csv`):** Per-epoch table with `Epoch`, `Time_Start`, `Time_End`, `Stage`, `Stage_Int`, `Has_MA`, `Confidence_Pct`, `Agreement_Pct`, `Status`.
6. **BrainVision `.vmrk` (Standard INI):** Version 1.0 INI file referencing `DataFile=` (extracted from `.vhdr` or recording filename) and `Mk<N>=Sleep Stage,<STAGE>[+MA],<pos>,<len>,0`.

---

## Part 2 — `Onera_PreAwakening.py`: Pre-Awakening & Whole-Night EEG Analysis

### 2.1 Recording Discovery, Raw/Marker Pairing & Whole-Night Staging

- **Recursive Discovery & Deduplication (`scan_sessions`):** Recursively walks both `data_dir` (for `.vhdr` headers) and `markers_dir` (for `.Markers` and `.vmrk` files) via `os.walk` while skipping hidden folders (e.g., `.onera_sessions`, `.git`) and hidden files. Marker files sharing `(normcase(parent), stem.lower())` are deduplicated favoring `.Markers` over `.vmrk`, and companion raw `.vmrk` files referenced by a `.vhdr` that already has at least one paired `.Markers` file are pruned so combined directories remain uncluttered while preserving `.vmrk`-only sessions and multiple scorings per recording.
- **4-Tier Automatic Marker-to-Raw Pairing & Natural Sorting:** Each discovered marker file is paired with its raw `.vhdr` in strict priority order:
  1. Exact relative subfolder + stem match (keeps identical filenames in separate BIDS/subject subfolders strictly isolated).
  2. Exact stem match anywhere in `data_dir` (preferring matching parent folder name).
  3. Separator-delimited prefix match (`_`, `-`, `.`, space) in either direction (handles Analyzer/scorer export suffixes such as `<stem>_Scored.Markers` or `<stem>_AASM.Markers`).
  4. Legacy two-token `_` prefix fallback (`Participant_Condition_*`) for backward compatibility.
  Companion raw `.vmrk` files are resolved per `.vhdr` via `resolve_raw_vmrk` (`MarkerFile=` INI lookup with basename, exact sibling, and case-insensitive stem fallbacks). All recordings are ordered with `natural_sort_key` and exposed in a single searchable `recording_dropdown` (`searchable=True`).
- **Fast Directory Event Discovery & Per-Recording Sampling Rate:** Event labels across the dataset are harvested in pure Python (`_scan_marker_file_fast`, skipping `Sleep Stage` and `New Segment` lines without DataFrame construction). Switching recordings automatically re-detects the active recording's exact sampling rate (`detect_marker_fs` → `detect_vhdr_fs` → `500 Hz`) to initialize `fs_input`, alerting the user if manually overridden away from the header value.
- **Whole-Night Hypnogram & Sleep Statistics:** Aligns scored 30-second stages to the raw `.vmrk` `New Segment` 24-hour clock timestamp (applying cropped-`.Markers` anchor alignment via `get_raw_marker_offset_sec` in $O(N+M)$ time) and computes clinical macro-statistics (TST, Sleep Efficiency, WASO, SOL, stage percentages) via `yasa.Hypnogram.sleep_statistics()`.
- **Pre-Awakening Epoch Extraction (`analyze_sleep_stages`):**
  - Selects 30-second epochs whose 15-second midpoint falls inside `(pos_awake - window_size_min, pos_awake]` (falling back to start positions if empty).
  - **Post-Awakening Contamination Guard:** If the final epoch straddles the exact awakening marker (`pos_epoch < pos_awake < pos_epoch + 30*fs`) and is scored `W`, `ART`, or `UNS`, it is dropped so `Last Stage at Awakening` and stage counts reflect pre-awakening sleep rather than post-awakening report behavior.
  - Compresses consecutive identical stages (retaining up to the final 4 transitions) and computes `Dominant Stage`, stage epoch counts, and trigger-to-awakening latency against selected event markers.

---

### 2.2 Optimized Buffer-Padded Signal Pre-Processing

To prevent FIR edge ringing while minimizing disk I/O and filtering latency across multi-awakening nights:

1. **Buffer Padding & In-Memory Raw Window Cache (`raw_segment_cache`):** Each pre-awakening window $[t_{\text{start}}, t_{\text{end}}]$ is extended by **$\text{pad} = 20.0\text{ s}$** on both ends ($[\max(0, t_{\text{start}} - 20), \min(t_{\text{max}}, t_{\text{end}} + 20)]$) and read once per recording/window from disk into a non-reactive `float32` RAM cache keyed by `(resolved_vhdr_path, mtime_ns, size_bytes, sample_bounds)` (capped at `1.5 GB`, falling back to direct streaming if exceeded). Adjusting filter cutoffs, notch frequency, reference channels, or bad-channel exclusions operates entirely from RAM with zero disk re-reads, and Awakening #1's unfiltered EEG reference copy is sliced from the same buffer with no duplicate read.
2. **Channel Typing, Direct Re-Referencing & Upfront Montage:** Bad channels are excluded by index, remaining channels are classified via `classify_sleep_channels`, active EEG channels are re-referenced directly in NumPy (`seg[eeg] -= mean(seg[refs])`, matching `mne.set_eeg_reference` while reclassifying reference electrodes as `misc`), and the `standard_1020` montage is attached once to a shared template `mne.Info` outside the awakening loop.
3. **Precomputed Composite Zero-Phase FIR Filtering (`design_composite_fir_kernel` + `apply_fir_oaconvolve_mt`):**
   - For each channel group (**EEG & EOG**: $[f_{\text{lo}}, f_{\text{hi}}]$, defaults $0.3\text{–}45.0\text{ Hz}$; **EMG**: $[10.0, \min(90.0, f_N - 1.0)]\text{ Hz}$), the `firwin` bandpass kernel (`mne.filter.create_filter`) and harmonic powerline notch kernel ($k \cdot f_{\text{notch}} < f_N$, default $50.0\text{ Hz}$, `notch_widths = freqs / 200`, `trans_bandwidth = 1.0 Hz`) are designed **once per session** and convolved via `scipy.signal.oaconvolve` into a single odd-length linear-phase composite FIR kernel.
   - Each awakening segment is reflect-padded and filtered in `float64` via multi-threaded `scipy.signal.oaconvolve` (`ThreadPoolExecutor`, up to 8 workers), matching sequential MNE `Raw.filter` + `Raw.notch_filter` to machine precision on direct `float64` inputs (and within $10^{-11}\text{ V}$ quantization tolerance from the `float32` cache) while running significantly faster. Segments shorter than the kernel automatically fall back to `mne.filter.filter_data` / `mne.filter.notch_filter`.
4. **Post-Filter Cropping & Validation:** The $20\text{ s}$ buffers are sliced off directly in NumPy before wrapping each awakening into an `mne.io.RawArray`, and Awakening #1's before- vs. after-filter cortical EEG PSD is computed with multi-threaded `scipy.signal.welch` (`compute_welch_psd_mt`).

---

### 2.3 Decoupled Non-Overlapping Artifact Rejection

Artifact screening is split into two reactive stages so threshold tuning is instantaneous:

1. **Per-Epoch Metric & PSD Precomputation (`extract_epoch_metrics` + `compute_welch_psd_mt`):** Slices the selected awakening into non-overlapping fixed-length epochs (`overlap = 0.0 s`; default `4 s`, configurable `2–30 s`) without deep-copying `selected_segment`, extracts per-epoch `eeg_ptp`, `eeg_step`, and `emg_ptp` arrays in an L2-cache-friendly pass, and computes per-epoch Welch PSDs **once** (`epoch_cache_bundle`) using multi-threaded `scipy.signal.welch` (4 s Hamming windows, 50% sub-window overlap when epoch length exceeds 4 s, $0.5\text{–}\min(45.0, h_{\text{freq}})\text{ Hz}$).
2. **Vectorized Threshold Masking (`evaluate_artifact_masks`):** Evaluates all four thresholds across the precomputed `(n_epochs, n_channels)` matrices in `< 1 ms`, preserving exact rejection priority and per-channel failure diagnostics:
   - **Min Amplitude (Flatline):** Reject if any EEG channel has $\text{PTP} < V_{\text{flat}}$ (default $0.5\,\mu\text{V}$).
   - **Max Amplitude (PTP):** Reject if any EEG channel has $\text{PTP} > V_{\text{ptp}}$ (default $400\,\mu\text{V}$).
   - **Voltage Step:** Reject if any EEG channel has $|\Delta V|_{\max} > S_{\max} \cdot \frac{1000}{f_s}$ (default $S_{\max} = 50\,\mu\text{V/ms}$).
   - **EMG Activity:** Reject if any EMG channel has $\text{PTP} > V_{\text{emg}}$ (default $100\,\mu\text{V}$).

The clean epoch subset is sliced directly into `clean_psd_bundle` (zero extra FFTs for BEFORE/AFTER diagnostic plots or downstream spectral tables).

---

### 2.4 Welch Spectral Power (PSD) & Frequency Band Presets

Spectral power reuses `clean_psd_bundle` directly (computed in $\mu\text{V}^2/\text{Hz}$ via $4\text{-s}$ Hamming Welch windows at $0.25\text{ Hz}$ resolution), so toggling **Band Preset**, **Absolute vs. Relative**, or **Mean vs. Median** executes immediately without recomputing FFTs:

- Clean epoch PSDs are aggregated across epochs by **Mean** or **Median** and integrated via Simpson's rule (`yasa.bandpower_from_psd`, clipped to $h_{\text{freq}}$).

| Preset | Bands & Frequency Ranges (Hz) |
|---|---|
| **Clinical Sleep** | `SWA` ($0.5\text{–}4$), `Theta` ($4\text{–}8$), `Alpha` ($8\text{–}12$), `Slow spindles` ($11\text{–}13$), `Fast spindles` ($13\text{–}15$), `Beta` ($15\text{–}25$) |
| **Standard EEG** | `Delta` ($1\text{–}4$), `Theta` ($4\text{–}8$), `Alpha` ($8\text{–}12$), `Beta` ($12\text{–}30$), `Gamma` ($30\text{–}45$) |

- **Relative Power Note:** Relative power ($\%$) divides each band by the broadband power across the active preset's total span. In the *Clinical Sleep* preset, *Alpha* ($8\text{–}12\text{ Hz}$) and *Slow spindles* ($11\text{–}13\text{ Hz}$) share the $11\text{–}12\text{ Hz}$ bin, so row-wise sums can slightly exceed $100\%$.

---

### 2.5 Scalp Topographies & Split-Screen Event Comparison

When $\ge 3$ EEG channels possess valid 3D `standard_1020` coordinates (read directly from `clean_psd_bundle["eeg_info"]` without deep-copying epoch data):

- **Whole-Window Topomaps:** Interpolates Mean/Median absolute or relative bandpower across the full pre-awakening window (`viridis` for Absolute, `RdBu_r` for Relative).
- **Interactive Time-Resolved Sub-Windows (`topo_precomp` Cache):** Precomputes valid 10-20 channel slices, 15-window **Fixed Scale** (2nd–98th percentile) colorbar bounds, and vectorized window-filtered stage/event rows upstream of `topo_time_slider` so scrubbing through sliding windows (`15`, `30`, `60`, `120`, or `240 s`) evaluates only the active sub-window.
  - *Fixed Scale:* Locks each band's colorbar to the **2nd–98th percentile** across 15 windows spanning the awakening period.
  - *Dynamic Scale:* Auto-scales locally to maximize spatial contrast.
- **Split-Screen Pre- vs. Post-Stimulus Mode:** Locks onto the most recent selected trigger onset $t_{\text{stim}}$ before awakening to compare Baseline ($[t_{\text{stim}} - W, t_{\text{stim}}]$) vs. Stimulation ($[t_{\text{stim}}, t_{\text{stim}} + W]$), strictly excluding stimulus-straddling epochs ($t_1 \le t_{\text{stim}}$ for Baseline; $t_0 \ge t_{\text{stim}}$ for Stimulation). When *Dynamic Scale* is enabled in Split-Screen mode, Baseline and Stimulation colorbars are pair-locked for valid visual comparison.

---

### 2.6 Power-Domain STFT Spectrograms & Master Export

- **Power-Domain Time-Frequency Spectrogram:** Computes `scipy.signal.spectrogram(..., window='hamming', scaling='density')` (default `4 s` window, configurable `1–30 s`, **`87.5%` temporal overlap**) across selected EEG channels in $\mu\text{V}$ (parallelized across threads via `ThreadPoolExecutor` when multiple channels are selected), averages across channels in the **linear power domain** ($\mu\text{V}^2/\text{Hz}$) to prevent phase cancellation across electrodes, converts to decibels via $10\log_{10}(\max(S_{xx}, 10^{-12}))$, and clips visual contrast at the **5th–95th percentiles**, marking rejected artifact epochs along the top border and pre-filtered event triggers vertically.
- **Master Batch `.csv` Export:** Iterates across all detected awakenings using shared `extract_epoch_metrics`, `evaluate_artifact_masks`, and `compute_welch_psd_mt` (zero `mne.Epochs` copies) and merges macro-stage metadata (keyed by `Recording` and `Awakening`) with per-channel clean Welch bandpower tables into a single `.csv` download.

---

## Part 3 — Synthetic Demo Data Generator (`generate_demo_data.py`)

`generate_demo_data.py` synthesizes self-contained 12-channel BrainVision polysomnography fixtures (`F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `EOG1`, `EMG1`, `M1`, `M2` at `200 Hz` default) using only `numpy`, `scipy`, and the Python standard library so anyone can test both Onera pipelines locally without committing binary EEG files to Git.

- **Stage-Specific Spectral & Waveform Synthesis (`synthesize_psg`):** Combines an FFT-derived $1/f^{1.4}$ pink background with 4th-order zero-phase Butterworth band-limited carriers across slow delta ($0.5\text{–}2\text{ Hz}$), fast delta ($2\text{–}4\text{ Hz}$), theta ($4\text{–}7.5\text{ Hz}$), alpha ($8.5\text{–}11.5\text{ Hz}$), sigma ($12\text{–}14.8\text{ Hz}$), beta ($16\text{–}28\text{ Hz}$), high-frequency EMG ($20\text{–}75\text{ Hz}$), and EOG ocular components:
  - **Wake (`W`):** Posterior alpha + beta oscillations, periodic Gaussian eye blinks (`120 µV` on `EOG1`), and elevated tonic chin `EMG1`.
  - **N1:** Mixed theta/low-delta EEG with attenuated alpha and slow rolling eye movements (`0.25–0.7 Hz`).
  - **N2:** Moderate delta/theta with Gaussian-enveloped $12\text{–}14.8\text{ Hz}$ sleep spindles and biphasic K-complexes.
  - **N3:** Frontal-dominant high-amplitude slow-wave delta activity.
  - **REM:** Low-voltage theta EEG, phasic conjugate rapid eye movements (`1.2 Hz`), and muscle atonia (`EMG1`).
- **10–20 Spatial Gradients & Signal Realism:**
  - Scales slow/fast delta higher frontally (`1.25×` at `F3`/`F4`), sigma spindles centrally (`1.15×` at `C3`/`C4`), and alpha occipitally (`1.38×`/`1.35×` at `O1`/`O2`) so 2D scalp topomaps display realistic anterior–posterior topography.
  - Injects `50 Hz` line noise (`12 µV` EEG) and slow common-mode drift (`0.35 Hz` / `0.12 Hz`) shared across EEG and mastoids (`M1`, `M2`), regional post-stimulus theta/alpha modulation for Split-Screen Pre- vs. Post-Stimulus contrast, and three isolated pre-awakening artifacts (high-amplitude `F3/F4` excursion, `C3` electrode pop step, and `EMG1` jaw-clench burst).
- **BrainVision `INT_16` Export & CLI (`--out`, `--sfreq`, `--recordings`, `--seed`):** Writes multiplexed little-endian `<i2` `.eeg` files (`0.1 µV` resolution) alongside UTF-8 `.vhdr`/`.vmrk` files and BrainVision Analyzer `.Markers` CSVs:
  1. `Demo01_Baseline_Night.*`: 60-minute (`120` epochs) flat recording with `LightsOff`, 3 awakenings (`Start_DreamReport`/`Stop_DreamReport` following N3, N2, and REM periods), music triggers, and `+MA` micro-arousals.
  2. `sub-02/ses-01/sub-02_ses-01_task-sleep_eeg.*` + `_Scored.Markers`: 30-minute (`60` epochs) nested BIDS recording whose `.Markers` timeline is cropped at `LightsOff` (`+120 s`), testing nested folder discovery, suffix pairing, and cropped-marker offset synchronization.
