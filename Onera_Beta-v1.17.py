import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Onera_Beta-v1.17
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview

    **Onera** is an interactive Python pipeline for sleep EEG analysis, designed to investigate pre-awakening brain states, sleep stage transitions, and time-frequency dynamics.

    **Author:** Alexandru Lupu | **GitHub:** [alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)

    ### Core Capabilities
    - **Whole-Night Hypnograms:** Sleep architecture visualization (supporting both AASM and R&K scoring) with whole-night macro-statistics, experimental event overlays, and real-time clock alignment.
    - **Pre-Awakening Staging:** Chronological stage progression, dominant stage classification, and stimulus-to-awakening latency extraction.
    - **Signal Pre-Processing:** Automated BrainVision loading, physiological channel typing, mastoid/average/custom re-referencing, and $\pm 20\text{ s}$ buffer-padded FIR bandpass/notch filtering ([MNE-Python](https://mne.tools/)) to prevent edge ringing.
    - **Automated Artifact Rejection:** Configurable epoch-level thresholding for flatlines, peak-to-peak voltage excursions, fast voltage steps ($\mu\text{V/ms}$), and high-frequency EMG bursts, with per-channel artifact diagnostics.
    - **Spectral Power (PSD) & Topomaps:** Absolute ($\mu\text{V}^2$) and relative (%) bandpower estimation via Welch's method ([YASA](https://raphaelvallat.com/yasa/)) with 2D scalp topography.
    - **Time-Frequency Spectrograms:** Power-domain averaged Short-Time Fourier Transforms (STFT) with event markers and artifact window overlays.

    ### Expected Data Structure
    - **Raw EEG Directory:** Standard BrainVision triplets (`.vhdr`, `.vmrk`, `.eeg`).
    - **Markers Directory:** Scored BrainVision Analyzer `.Markers` (CSV-style) or `.vmrk` files containing sleep stage annotations and awakening/stimulus triggers.
    - **File Naming Convention:** `Participant_Condition_OptionalID` (e.g., `DS01_Night1_2025.vhdr` or `DS01_adapt.vhdr`).

    ### UI Navigation
    - Toggle **App View** (`Ctrl + .` or bottom-right icon) to hide code cells or to show them when in App View.
    - Use the **Table of Contents** Click `View outline` (left sidebar)
    - **Performance Tip:** Marimo is reactive and updates downstream cells automatically after a change has been done. After changing multiple settings, click **Stop** (square button) to stop the current update and use the latest settings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis Configuration

    Define the local paths to your raw EEG and marker directories. The program automatically scans for BrainVision files (`.vhdr`, `.eeg`, `.vmrk` / `.Markers`). Once paths are set, configure the core parameters that will anchor all downstream analyses (Epoching, Sleep Stages, Spectrograms, etc.).

    **Supported File Naming:**
    `XXXX_YYYY_ZZZZ`
    - **X:** Participant Code (e.g., DS01)
    - **Y:** Condition/Night (e.g., 1, adapt)
    - **Z:** Session ID/Timestamp (optional)
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    # Use a batched Markdown template to group the inputs
    paths_form = mo.md(
        """
        {data_dir}

        {markers_dir}
        """
    ).batch(
        data_dir=mo.ui.text(
            value=r"Z:\Alex\RAW_Sleep", 
            label="📂 Raw Data Path (.vhdr, .vmrk, .eeg):",
            full_width=True
        ),
        markers_dir=mo.ui.text(
            value=r"Z:\Alex\Markers", 
            label="📂 Markers Path (.Markers / .vmrk):",
            full_width=True
        )
    ).form(submit_button_label="Scan & Load Directories")

    # Display the configuration block
    mo.vstack([
        mo.md("### ⚙️ Data Paths Configuration"),
        paths_form,
        mo.md("*(Modify these paths to match your local directory structure, then click **Scan & Load Directories** to apply them)*"),
        mo.md("---")
    ])
    return Path, mo, paths_form


@app.cell
def _(Path, mo, paths_form):
    # Immediately show a loading text while scanning the directories
    mo.output.replace(
        mo.callout(
            mo.md("### ⏳ Scanning directories and loading session data...\n*Please wait, this may take a moment.*"), 
            kind="info"
        )
    )

    import os
    os.environ["MNE_USE_NUMBA"] = "false"
    import mne
    import yasa
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    import matplotlib.ticker as ticker

    # Extract values from the submitted form rather than individual inputs
    data_dir = Path(paths_form.value["data_dir"])
    markers_dir = Path(paths_form.value["markers_dir"])

    # Function to find a specific file matching the participant, condition, and file extension
    def get_file_path(directory, participant, condition, extension):
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        ext_lower = extension.lower()
        prefix_1 = f"{participant}_{condition}_".lower()
        prefix_2 = f"{participant}_{condition}{ext_lower}".lower()

        matches = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() == ext_lower and (
                f.name.lower().startswith(prefix_1) or f.name.lower() == prefix_2
            )
        ]
        if not matches:
            raise FileNotFoundError(f"No {extension} file found for '{participant}_{condition}' in {directory}")
        return sorted(matches)[0]

    # Function to read BrainVision marker files (.Markers or .vmrk) into a clean table
    def read_brainvision_markers(filepath):
        rows = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_s = line.strip()
                if not line_s or line_s.startswith(';') or line_s.startswith('['):
                    continue
                if line_s.startswith(('Sampling rate', 'Brain Vision', 'DataFile', 'MarkerFile', 'Codepage', 'Type')):
                    continue
                if re.match(r'^Mk\d+=', line_s):
                    line_s = line_s.split('=', 1)[1]
                parts = [p.strip() for p in line_s.split(',')]
                if len(parts) >= 5:
                    rows.append(parts[:5])

        df = pd.DataFrame(rows, columns=["Type", "Description", "Position", "Length", "Channel"])
        df["Position"] = pd.to_numeric(df["Position"], errors='coerce')
        df["Length"] = pd.to_numeric(df["Length"], errors='coerce')
        df["Description"] = df["Description"].astype(str).str.strip()
        df["Type"] = df["Type"].astype(str).str.strip()
        return df.dropna(subset=["Position"]).sort_values("Position").reset_index(drop=True)

    # Find all marker files (.Markers or .vmrk) in the markers folder
    all_marker_files = []
    if markers_dir.exists():
        all_marker_files = [
            f for f in markers_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ('.markers', '.vmrk')
        ]

    # Initialize sets to store unique participant IDs, conditions, marker names, and sampling rate
    unique_participants = set()
    unique_conditions = set()
    all_unique_labels = set()
    detected_fs = None

    # Loop through every marker file to extract participants, conditions, sampling rate, and event labels
    for _f in all_marker_files:
        _parts = _f.stem.split('_')
        if len(_parts) >= 2:
            unique_participants.add(_parts[0])
            unique_conditions.add(_parts[1])
        try:
            if detected_fs is None:
                with open(_f, 'r', encoding='utf-8', errors='ignore') as _fh:
                    for _ in range(5):
                        _hdr_line = _fh.readline()
                        _m_fs = re.search(r'Sampling rate:\s*([\d.]+)\s*Hz', _hdr_line, re.IGNORECASE)
                        if _m_fs:
                            detected_fs = int(round(float(_m_fs.group(1))))
                            break
            _df_temp = read_brainvision_markers(_f)
            _non_stage = _df_temp[~_df_temp["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)]
            all_unique_labels.update(_non_stage["Description"].dropna().unique())
        except Exception:
            pass

    all_unique_labels.discard("")

    # Helper function to sort participant IDs by their number (e.g., DS2 before DS10)
    def numeric_sort_key(p_id):
        numbers = re.findall(r'\d+', p_id)
        return (int(numbers[0]), p_id) if numbers else (0, p_id)

    participant_list = sorted(list(unique_participants), key=numeric_sort_key)
    condition_list = sorted(list(unique_conditions))
    detected_labels_str = ", ".join(sorted(all_unique_labels)) if all_unique_labels else "None detected"

    # If no sampling rate was found in the marker files, check the first .vhdr header file
    if detected_fs is None and data_dir.exists():
        vhdr_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() == '.vhdr']
        if vhdr_files:
            try:
                with open(vhdr_files[0], 'r', encoding='utf-8', errors='ignore') as _f_vhdr:
                    for _line in _f_vhdr:
                        if _line.startswith('SamplingInterval='):
                            _interval = float(_line.split('=')[1].strip())
                            detected_fs = int(round(1_000_000 / _interval))
                            break
            except Exception:
                pass

    if detected_fs is None:
        detected_fs = 500

    fs_input = mo.ui.number(
        value=detected_fs,
        label="Marker Sampling Rate (Hz) [Auto-detected from .Markers/.vhdr, adjustable]:"
    )

    participant_dropdown = mo.ui.dropdown(
        options=participant_list,
        value=participant_list[0] if participant_list else None,
        label="Select Participant:"
    )
    night_dropdown = mo.ui.dropdown(
        options=condition_list,
        value=condition_list[0] if condition_list else None,
        label="Select Night / Condition:"
    )

    _status_banner = (
        mo.callout(
            mo.md("⚠️ **Directory not found or empty.** Please verify the Raw Data and Markers paths above."),
            kind="warn"
        )
        if not all_marker_files else mo.md(f"**Experimental Marker Labels Detected:** `{detected_labels_str}`")
    )

    mo.vstack([
        mo.md("### 👤 Participant & Session Selection"),
        _status_banner,
        participant_dropdown,
        night_dropdown,
        mo.md("---"),
        mo.md("### ⚙️ Sampling Rate"),
        fs_input
    ])
    return (
        all_unique_labels,
        data_dir,
        fs_input,
        get_file_path,
        markers_dir,
        mne,
        night_dropdown,
        np,
        participant_dropdown,
        pd,
        plt,
        read_brainvision_markers,
        yasa,
    )


@app.cell
def _(all_unique_labels, mo):
    # Use the detected marker names, or fall back to "Start_DreamReport" if none were found
    _labels = sorted(list(all_unique_labels)) if all_unique_labels else ["Start_DreamReport"]

    # Dropdown to choose which marker label represents an awakening
    awake_marker_select = mo.ui.dropdown(
            options=_labels,
            value="Start_DreamReport" if "Start_DreamReport" in _labels else _labels[0],
            label="Select Awakening Marker:"
    )

    # Number input to set how many minutes before each awakening to analyze
    window_size_input = mo.ui.number(
            value=10, 
            step=1, 
            label="Pre-Awakening Analysis Window (Minutes):"
    )

    # Pre-select standard music markers if they exist in the dataset
    _default_events = [m for m in ["Start_Music_Happy", "Stop_Music_Sad"] if m in _labels]
    # Multiselect menu to choose optional stimulation or experimental event markers
    event_markers_select = mo.ui.multiselect(
            options=_labels,
            value=_default_events,
            label="Select Experimental Event/Stimulation Markers (Optional):"
    )

    # Display the targeting and time window controls
    mo.vstack([
            mo.md("---"),
            mo.md("### 🎯 Targeting & Window Parameters"),
            awake_marker_select,
            window_size_input,
            event_markers_select
    ])
    return awake_marker_select, event_markers_select, window_size_input


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo, window_size_input):
    mo.md(rf"""
    # Pre-Awakening Sleep Stages

    Extracts the chronological progression and dominant sleep stage for the **{window_size_input.value} minutes** prior to each targeted awakening.

    - **Resolution:** Standard 30-second epochs.
    - **Events:** Dynamically calculates elapsed time from auditory stimulation markers to the awakening.
    """)
    return


@app.cell
def _(
    awake_marker_select,
    data_dir,
    event_markers_select,
    fs_input,
    get_file_path,
    markers_dir,
    mo,
    night_dropdown,
    participant_dropdown,
    pd,
    read_brainvision_markers,
    window_size_input,
):
    import re as _re_stage

    # Get the currently selected participant and night from the dropdowns
    current_participant = participant_dropdown.value
    current_night = night_dropdown.value

    # Dictionary translating raw sleep stage labels (R&K or AASM) into standardized YASA stage names
    STAGE_MAP = {
        "W": "W", "WAKE": "W",
        "S1": "N1", "N1": "N1",
        "S2": "N2", "N2": "N2",
        "S3": "N3", "S4": "N3", "N3": "N3",
        "REM": "REM", "R": "REM",
        "MT": "ART", "MOVEMENT": "ART", "ART": "ART",
        "?": "UNS", "UNS": "UNS", "UNSCORED": "UNS"
    }

    # Helper function to detect the true sampling rate from the current session's marker or header file
    def _detect_session_fs(marker_filepath, participant, night):
        """Checks the active session's .Markers/.vmrk or .vhdr header for its exact sampling rate."""
        # 1. Check the first 10 lines of the marker file for a 'Sampling rate: X Hz' entry
        try:
            with open(marker_filepath, 'r', encoding='utf-8', errors='ignore') as _fh:
                for _ in range(10):
                    _line = _fh.readline()
                    _m = _re_stage.search(r'Sampling rate:\s*([\d.]+)\s*Hz', _line, _re_stage.IGNORECASE)
                    if _m:
                        return int(round(float(_m.group(1))))
        except Exception:
            pass

        # 2. Fallback: Check the corresponding .vhdr file for 'SamplingInterval=' (in microseconds)
        try:
            _vhdr = get_file_path(data_dir, participant, night, ".vhdr")
            with open(_vhdr, 'r', encoding='utf-8', errors='ignore') as _fv:
                for _line in _fv:
                    if _line.startswith('SamplingInterval='):
                        _interval = float(_line.split('=')[1].strip())
                        return int(round(1_000_000 / _interval))
        except Exception:
            pass

        return None

    # Function to extract pre-awakening sleep stages and event latencies for a given session
    def analyze_sleep_stages(participant, night):
        # Return empty tables if participant or night is not selected
        if not participant or not night:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

        # Locate the marker file (.Markers first, or .vmrk as fallback)
        try:
            marker_file = get_file_path(markers_dir, participant, night, ".Markers")
        except FileNotFoundError:
            marker_file = get_file_path(markers_dir, participant, night, ".vmrk")

        # Verify the sampling rate for this specific recording session
        file_fs = _detect_session_fs(marker_file, participant, night)
        fs = file_fs if file_fs is not None else fs_input.value

        # Parse all markers from the file
        df_markers = read_brainvision_markers(marker_file)

        # Filter rows corresponding to awakenings and selected experimental events
        awakenings = df_markers[df_markers["Description"] == awake_marker_select.value].copy()
        active_events = df_markers[df_markers["Description"].isin(event_markers_select.value)].copy()

        # Filter rows corresponding to scored 30-second sleep stages and map them to standard labels
        stages = df_markers[df_markers["Type"].str.contains("Sleep Stage", case=False, na=False)].copy()
        stages["Raw_Stage"] = stages["Description"].str.split('+').str[0].str.strip().str.upper()
        stages["Stage_YASA"] = stages["Raw_Stage"].map(STAGE_MAP).fillna("UNS")

        # Convert the analysis window duration, 30s epoch length, and 15s midpoint into sample points
        analysis_window_points = window_size_input.value * 60 * fs
        full_epoch_pts = 30 * fs
        half_epoch_pts = 15 * fs
        # Allow event lookback up to the analysis window (or at least 15 minutes if the user sets a very short window)
        max_event_lookback_pts = max(analysis_window_points, 15 * 60 * fs)
        report_rows = []
        prev_awake_pos = 0

        # Loop through each awakening to summarize sleep stages and preceding events
        for idx, (_, row) in enumerate(awakenings.iterrows(), 1):
            # Define the start and end sample positions of the pre-awakening window
            pos_awake = row["Position"]
            pos_start = max(0, pos_awake - analysis_window_points)

            # Find experimental events that occurred after the previous awakening and within the lookback window
            event_search_start = max(prev_awake_pos, pos_awake - max_event_lookback_pts)
            prior_events = active_events[
                (active_events["Position"] > event_search_start) &
                (active_events["Position"] <= pos_awake)
            ]
            event_type = "None"
            event_duration_str = "N/A"

            # Calculate the time elapsed between the most recent stimulus marker and the awakening
            if not prior_events.empty:
                last_event = prior_events.iloc[-1]
                duration_sec = int(round((pos_awake - last_event["Position"]) / fs))
                if duration_sec >= 0:
                    event_duration_str = f"{duration_sec // 60:02d}:{duration_sec % 60:02d}"
                    event_type = last_event["Description"]

            # Format the awakening time from recording start as HH:MM:SS
            total_seconds = int(round(pos_awake / fs))
            time_str = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"

            # Select 30-second sleep epochs whose 15-second midpoint falls inside the analysis window
            epochs_window = stages[
                (stages["Position"] + half_epoch_pts > pos_start) &
                (stages["Position"] + half_epoch_pts <= pos_awake)
            ].copy()

            # Fallback to epoch start positions if no midpoints matched
            if epochs_window.empty:
                epochs_window = stages[(stages["Position"] >= pos_start) & (stages["Position"] < pos_awake)].copy()

            # Prevent post-awakening Wake ('W') contamination:
            # If the final 30s epoch straddles the awakening marker (starts before pos_awake, ends after pos_awake)
            # and is scored as 'W' (because the participant spent >15s of that epoch awake giving a dream report),
            # drop that straddling epoch so 'Last Stage at Awakening' reflects the true pre-awakening sleep stage.
            if len(epochs_window) > 1:
                _last_ep_pos = epochs_window.iloc[-1]["Position"]
                _last_ep_stage = epochs_window.iloc[-1]["Stage_YASA"]
                if (_last_ep_pos < pos_awake < (_last_ep_pos + full_epoch_pts)) and (_last_ep_stage == "W"):
                    epochs_window = epochs_window.iloc[:-1]

            # Count how many epochs were spent in each sleep stage
            counts = epochs_window["Stage_YASA"].value_counts().to_dict()

            # Collapse consecutive identical stages to show the stage transition progression
            raw_sequence = epochs_window["Stage_YASA"].tolist()
            compressed_sequence = []
            for s in raw_sequence:
                if not compressed_sequence or compressed_sequence[-1] != s:
                    compressed_sequence.append(s)

            # Keep at most the last 4 stage transitions for clean table display
            MAX_STAGES = 4
            display_sequence = (
                ["..."] + compressed_sequence[-MAX_STAGES:]
                if len(compressed_sequence) > MAX_STAGES
                else compressed_sequence
            )

            # Store epoch counts for each physiological sleep stage
            stage_counts = {
                "W": counts.get("W", 0),
                "N1": counts.get("N1", 0),
                "N2": counts.get("N2", 0),
                "N3": counts.get("N3", 0),
                "REM": counts.get("REM", 0),
            }
            # Identify the sleep stage with the highest epoch count in the window
            max_val = max(stage_counts.values()) if stage_counts else 0
            if max_val > 0:
                top_stages = [k for k, v in stage_counts.items() if v == max_val]
                dominant_stage = " / ".join(top_stages)
            else:
                dominant_stage = "None"

            # Record the final sleep stage immediately before awakening
            Last_Stage = raw_sequence[-1] if raw_sequence else "None"
            prev_awake_pos = pos_awake

            # Append the summary metrics for this awakening to the report list
            report_rows.append({
                "Participant": participant,
                "Condition": night,
                "Awakening": idx,
                "Time (HH:MM:SS)": time_str,
                "Last Stage at Awakening": Last_Stage,
                "Dominant Stage": dominant_stage,
                "Stage Sequence": " → ".join(display_sequence),
                "Event Type": event_type,
                "Event Latency (MM:SS)": event_duration_str,
                "W (Wake)": stage_counts["W"],
                "N1": stage_counts["N1"],
                "N2": stage_counts["N2"],
                "N3": stage_counts["N3"],
                "REM": stage_counts["REM"],
                "Total Epochs": len(epochs_window)
            })

        return df_markers, stages, awakenings, pd.DataFrame(report_rows), file_fs

    # Run the sleep stage analysis and display the summary table (or a warning if the file is missing)
    try:
        df_markers, df_stages, awakenings, summary_df, _detected_session_fs = analyze_sleep_stages(current_participant, current_night)

        # Warn the user if the active session's header sampling rate differs from the UI number box
        _fs_warning = (
            mo.callout(
                mo.md(
                    f"⚠️ **Sampling Rate Mismatch:** This session's file header specifies **`{_detected_session_fs} Hz`**, "
                    f"but the Sampling Rate input box above is set to **`{fs_input.value} Hz`**. "
                    f"Please update the Sampling Rate input box above to `{_detected_session_fs}` so all downstream plots align accurately."
                ),
                kind="warn"
            )
            if (_detected_session_fs is not None and _detected_session_fs != fs_input.value)
            else mo.md("")
        )

        _stage_ui = mo.vstack([
            _fs_warning,
            mo.md(f"**Participant:** `{current_participant}` | **Night:** `{current_night}` | **Awakenings Detected:** `{len(awakenings)}`"),
            mo.ui.table(summary_df)
        ])
    except FileNotFoundError as _e:
        df_markers, df_stages, awakenings, summary_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _stage_ui = mo.callout(mo.md(f"⚠️ **File Not Found:** `{_e}`"), kind="warn")

    _stage_ui
    return (
        awakenings,
        current_night,
        current_participant,
        df_markers,
        df_stages,
        summary_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Whole-Night Architecture (Hypnogram)
    Visualizes the participant's progression through Wake, N1, N2, N3, and REM sleep. You can overlay custom physiological or experimental markers.
    """)
    return


@app.cell
def _(awake_marker_select, df_markers, event_markers_select, mo):
    # Extract all non-stage marker descriptions present in the current night's recording
    if df_markers.empty or "Type" not in df_markers.columns:
        current_night_labels = []
    else:
        non_stage_markers = df_markers[~df_markers["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)]
        current_night_labels = [
            m for m in non_stage_markers["Description"].dropna().unique().tolist()
            if str(m).strip() != ""
        ]

    # Get the currently selected awakening and event markers to use as default overlays
    _awake_val = awake_marker_select.value
    _events_val = event_markers_select.value

    default_markers = [m for m in current_night_labels if m == _awake_val or m in _events_val]

    # Create a multiselect menu to choose which markers to plot on the whole-night hypnogram
    hypno_markers_select = mo.ui.multiselect(
        options=current_night_labels,
        value=default_markers,
        label="📍 Select Markers to Display on Hypnogram:"
    )

    # Display the hypnogram marker selector
    mo.vstack([
        mo.md("### Hypnogram Configuration"),
        hypno_markers_select
    ])
    return (hypno_markers_select,)


@app.cell
def _(
    awake_marker_select,
    current_night,
    current_participant,
    data_dir,
    df_markers,
    df_stages,
    fs_input,
    get_file_path,
    hypno_markers_select,
    mo,
    pd,
    plt,
    window_size_input,
    yasa,
):
    # Import date and time utilities for formatting the real-time clock axis
    import datetime as _datetime
    from matplotlib.dates import date2num as _date2num, DateFormatter as _DateFormatter

    # Function to create short, unique abbreviations for marker labels on the hypnogram
    def _make_abbrev(name, used):
        """Creates a short, readable abbreviation (e.g., 'Stop_DreamReport' -> 'SpDR', 'Lights off' -> 'LO')."""
        # Split the marker name into words by replacing underscores and hyphens with spaces
        words = name.replace('_', ' ').replace('-', ' ').split()
        if not words:
            base = "M"
        elif len(words) == 1:
            # For a single word, use its uppercase letters or its first 3 characters
            w = words[0]
            caps = [c for c in w if c.isupper()]
            base = "".join(caps) if len(caps) >= 2 else w[:3].upper()
        else:
            # For multiple words, shorten "start" to "St", "stop" to "Sp", and take initials of other words
            parts = []
            for w in words:
                w_low = w.lower()
                if w_low == "start":
                    parts.append("St")
                elif w_low == "stop":
                    parts.append("Sp")
                else:
                    caps = [c for c in w if c.isupper()]
                    parts.append("".join(caps) if len(caps) >= 2 else w[0].upper())
            base = "".join(parts)

        # Append a number if the abbreviation is already used by another marker
        candidate = base
        counter = 2
        while candidate in used:
            candidate = f"{base}{counter}"
            counter += 1
        used.add(candidate)
        return candidate

    # Show a warning if no sleep stages are available; otherwise build the hypnogram and statistics table
    if df_stages.empty:
        _fig_out = mo.callout(mo.md("⚠️ **No sleep stage annotations found to plot hypnogram.**"), kind="warn")
    else:
        # Close previous Matplotlib figures and use a clean light background style
        plt.close('all')

        plt.style.use('default')

        # Move the awakening marker to the end of the list so it is drawn on top of other markers
        _awake_val = awake_marker_select.value
        _selected_markers = list(hypno_markers_select.value)
        if _awake_val in _selected_markers:
            _selected_markers.remove(_awake_val)
            _selected_markers.append(_awake_val)

        # Adjust figure height dynamically based on how many marker types are selected
        _num_selected = len(_selected_markers)
        _fig_height = 4.2 + (_num_selected * 0.35)

        # Create the Matplotlib figure and axis with a white background
        _fig_hypno, _ax_hypno = plt.subplots(figsize=(15, _fig_height), facecolor='white')

        # Create a YASA Hypnogram object from the scored stages and plot the hypnogram curve
        _stages_list = df_stages["Stage_YASA"].tolist()
        _hypno = yasa.Hypnogram(_stages_list)
        _hypno.plot_hypnogram(ax=_ax_hypno)

        # Calculate whole-night sleep statistics (TST, SE, WASO, etc.) using YASA
        _stats_dict = _hypno.sleep_statistics()
        _stats_df = pd.DataFrame([_stats_dict]).round(2)
        _stats_df.insert(0, "Condition", current_night)
        _stats_df.insert(0, "Participant", current_participant)

        # Compute time offsets in hours so markers align with the first scored sleep epoch
        _fs = fs_input.value
        _first_stage_pos = df_stages["Position"].iloc[0] if not df_stages.empty else 1
        _offset_hours = max(0.0, (_first_stage_pos - 1) / _fs / 3600.0)
        _win_hours = (window_size_input.value * 60.0) / 3600.0

        # Define color palette for experimental markers and track used abbreviations
        _palette = [
            '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
            '#17becf', '#e377c2', '#8c564b', '#bcbd22', '#000080'
        ]
        _used_abbrevs = {"A#"}

        # Loop through each selected marker type and draw vertical lines and labels on the hypnogram
        for _i, _marker_name in enumerate(_selected_markers):
            _is_awake = (_marker_name == _awake_val)
            _m_color = '#d62728' if _is_awake else _palette[_i % len(_palette)]
            _abbrev = "A#" if _is_awake else _make_abbrev(_marker_name, _used_abbrevs)
            _legend_label = f"[{_abbrev}] {_marker_name} (Awakening)" if _is_awake else f"[{_abbrev}] {_marker_name}"

            _specific_markers = df_markers[df_markers["Description"] == _marker_name]

            # Plot each occurrence of the current marker
            for _idx, (_, _row) in enumerate(_specific_markers.iterrows(), 1):
                _time_hours = (_row["Position"] / _fs / 3600.0) - _offset_hours

                # Draw a vertical dashed line at the marker time
                _ax_hypno.axvline(
                    x=_time_hours,
                    color=_m_color,
                    linestyle='--' if _is_awake else '-.',
                    linewidth=1.8 if _is_awake else 1.5,
                    alpha=0.9,
                    label=_legend_label if _idx == 1 else "_nolegend_"
                )

                # Shade the pre-awakening analysis window in red before each awakening
                if _is_awake:
                    _ax_hypno.axvspan(
                        max(0.0, _time_hours - _win_hours),
                        _time_hours,
                        color='#d62728',
                        alpha=0.12,
                        label=f"Pre-Awakening Window ({window_size_input.value}m)" if _idx == 1 else "_nolegend_"
                    )

                # Add a staggered text tag above the plot for each marker
                _short_text = f"A#{_idx}" if _is_awake else _abbrev

                _ax_hypno.annotate(
                    _short_text,
                    xy=(_time_hours, 1.0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, 4 + (_i * 18)),
                    textcoords='offset points',
                    color=_m_color,
                    fontweight='bold',
                    fontsize=9.5,
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75)
                )

        # Set the title with extra top padding so marker annotations above the plot have room
        _title_padding = 20 + (_num_selected * 18)
        _ax_hypno.set_title(
            f"Whole-Night Hypnogram — {current_participant} ({current_night})",
            pad=_title_padding,
            fontsize=14,
            fontweight='bold'
        )

        # Add a legend outside the right edge of the hypnogram
        _handles, _labels = _ax_hypno.get_legend_handles_labels()
        if _handles:
            _ax_hypno.legend(
                loc='upper left',
                bbox_to_anchor=(1.01, 1.0),
                frameon=True,
                facecolor='white',
                edgecolor='#cccccc',
                title="Markers & Windows",
                title_fontsize=10,
                fontsize=9.5,
                labelspacing=0.6
            )

        # Read the recording start timestamp from the raw .vmrk file to display wall-clock time
        _start_time = None
        try:
            _vmrk_file = get_file_path(data_dir, current_participant, current_night, ".vmrk")
            with open(_vmrk_file, 'r', encoding='utf-8', errors='ignore') as _f_vmrk:
                for _line in _f_vmrk:
                    if 'New Segment' in _line:
                        _parts = _line.split(',')
                        if len(_parts) >= 6:
                            _time_str = _parts[5].strip()
                            if len(_time_str) >= 14:
                                _start_time = _datetime.datetime.strptime(_time_str[:14], "%Y%m%d%H%M%S")
                                _start_time += _datetime.timedelta(hours=_offset_hours)
                        break
        except Exception:
            pass

        # If a valid start timestamp was found, add a secondary bottom axis showing 24-hour clock time
        if _start_time:
            _start_num = _date2num(_start_time)
            def _hours_to_mdates(x): return _start_num + (x / 24.0)
            def _mdates_to_hours(x): return (x - _start_num) * 24.0

            _sec_ax = _ax_hypno.secondary_xaxis('bottom', functions=(_hours_to_mdates, _mdates_to_hours))
            _sec_ax.spines['bottom'].set_position(('outward', 42))
            _sec_ax.xaxis.set_major_formatter(_DateFormatter('%H:%M'))
            _sec_ax.set_xlabel('Clock Time (24h)', fontweight='bold')

        _fig_hypno.tight_layout()
        _fine_divider = mo.md("<hr style='border: none; height: 1.5px; width: 95%; background-color: #2c3e50; margin: 34px auto;'>")

        # Combine the hypnogram plot, metric explanations, and statistics table into the output view
        _fig_out = mo.vstack([
            _fig_hypno,
            _fine_divider,
            mo.md(
                "### Whole-Night Sleep Macro-Architecture (YASA)\n\n"
                "**Clinical Metric Definitions & Typical Ranges:**\n"
                "- **TIB (Time in Bed):** Total duration (minutes) from recording start to end.\n"
                "- **SPT (Sleep Period Time):** Duration (minutes) from the first to the last epoch of sleep.\n"
                "- **WASO (Wake After Sleep Onset):** Total time (minutes) spent awake after initially falling asleep.\n"
                "- **TST (Total Sleep Time):** Total time (minutes) actually spent asleep.\n"
                "- **SE (Sleep Efficiency):** Percentage of Time in Bed spent asleep (`TST / TIB`). *Normal: ≥ 85%*\n"
                "- **SME (Sleep Maintenance Efficiency):** Percentage of Sleep Period Time spent asleep (`TST / SPT`). *Normal: ≥ 85-90%*\n"
                "- **SFI (Sleep Fragmentation Index):** Number of awakenings and shifts to N1 sleep per hour of Total Sleep Time. *Normal: Typically < 5 (higher values indicate highly broken, unrefreshing sleep).*\n"
                "- **SOL / SOL_5min (Sleep Onset Latency):** Time (minutes) to the first epoch of sleep / time to 5 continuous minutes of sleep. *Normal: 10–20 min.*"
            ),
            mo.ui.table(_stats_df)
        ])

    _fig_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo, window_size_input):
    mo.md(rf"""
    # Signal Pre-Processing

    To optimize memory, processing is only applied to the **{window_size_input.value}-minute** segments preceding each awakening.

    1. **Re-referencing:** Channels are computationally linked to the selected references (e.g., M1, M2).
    2. **Bandpass (FIR):** Isolates cortical activity and removes slow drift.
    3. **Notch Filter:** Suppresses powerline interference.
    """)
    return


@app.cell
def _(data_dir, get_file_path, mne, mo, night_dropdown, participant_dropdown):
    # Get the currently selected participant and night
    _participant = participant_dropdown.value
    _night = night_dropdown.value

    # Read the EEG header metadata (without loading the full signal) to get available channel names
    try:
        _raw_file = get_file_path(data_dir, _participant, _night, ".vhdr")
        _raw_info = mne.io.read_raw_brainvision(_raw_file, preload=False, verbose='ERROR')
        _ch_names = _raw_info.ch_names
    except Exception:
        _ch_names = ["M1", "M2"]

    # Automatically select mastoid/earlobe channels (e.g., M1, M2) as default references if present
    _default_refs = [ch for ch in ["M1", "M2", "A1", "A2", "TP9", "TP10"] if ch in _ch_names][:2]

    # Multiselect menu for choosing EEG reference channels
    ref_channels_input = mo.ui.multiselect(
        options=_ch_names,
        value=_default_refs,
        label="EEG Reference Channels (leave blank to keep native recording reference):"
    )

    # Multiselect menu for excluding noisy or detached channels prior to analysis
    bad_channels_input = mo.ui.multiselect(
        options=_ch_names,
        value=[],
        label="Exclude Bad / Detached Channels (optional — dropped before re-referencing & artifact rejection):"
    )

    # Number inputs for configuring the high-pass, low-pass, and powerline notch filters
    l_freq_input = mo.ui.number(value=0.3, step=0.1, label="EEG High-pass filter (Hz):")
    h_freq_input = mo.ui.number(value=45.0, step=1.0, label="EEG Low-pass filter (Hz):")
    notch_freq_input = mo.ui.number(value=50.0, step=10.0, label="Notch filter (Hz) [set to 0 to disable]:")

    # Display all pre-processing configuration widgets
    mo.vstack([
        mo.md("### 🎛️ Pre-Processing Filter & Channel Settings"),
        ref_channels_input,
        bad_channels_input,
        l_freq_input,
        h_freq_input,
        notch_freq_input
    ])
    return (
        bad_channels_input,
        h_freq_input,
        l_freq_input,
        notch_freq_input,
        ref_channels_input,
    )


@app.cell
def _(
    awakenings,
    bad_channels_input,
    current_night,
    current_participant,
    data_dir,
    fs_input,
    get_file_path,
    h_freq_input,
    l_freq_input,
    mne,
    mo,
    notch_freq_input,
    np,
    plt,
    ref_channels_input,
    window_size_input,
):
    # Function to assign each channel to its physiological type (EEG, EOG, EMG, ECG, or misc)
    def classify_sleep_channels(ch_names, ref_chs=None):
        """Automatically maps common sleep lab channel names to MNE channel types."""
        ref_chs = set(ref_chs or [])
        ch_types = {}
        for ch in ch_names:
            ch_up = ch.upper()
            # Mark dedicated mastoid/earlobe electrodes (or user-selected references) as 'misc'
            # so they are not treated as active scalp EEG channels.
            # Note: TP9/TP10 are only marked 'misc' if explicitly selected in ref_chs.
            if ch in ref_chs or ch_up in {'M1', 'M2', 'A1', 'A2'}:
                ch_types[ch] = 'misc'
            # Identify eye movement channels
            elif any(k in ch_up for k in ('EOG', 'LOC', 'ROC', 'HEOG', 'VEOG')):
                ch_types[ch] = 'eog'
            # Identify chin/muscle channels
            elif any(k in ch_up for k in ('EMG', 'CHIN', 'SUBMENT')):
                ch_types[ch] = 'emg'
            # Identify heart rate channels
            elif any(k in ch_up for k in ('ECG', 'EKG', 'HEART')):
                ch_types[ch] = 'ecg'
            # Identify respiration, oximetry, and body position sensors
            elif any(k in ch_up for k in ('RESP', 'FLOW', 'THOR', 'ABDO', 'SNORE', 'SPO2', 'PLETH', 'POS')):
                ch_types[ch] = 'misc'
            # Treat all remaining channels (including TP9/TP10 when not used as references) as cortical EEG
            else:
                ch_types[ch] = 'eeg'
        return ch_types

    # Convert the analysis window duration from minutes to seconds
    _marker_fs = fs_input.value
    _analysis_win = window_size_input.value * 60.0
    preprocessed_segments = []
    raw_example = None

    # Check if there are awakenings to process
    if awakenings.empty:
        output_ui_prep = mo.callout(mo.md("⚠️ **No awakenings available to preprocess.**"), kind="warn")
    else:
        # Show an informational banner while EEG segments are being loaded and filtered
        mo.output.replace(
            mo.callout(
                mo.md("### ⏳ Currently preprocessing EEG data...\n*Please wait, this may take a moment.*"), 
                kind="info"
            )
        )

        # Open the raw BrainVision file header without loading the entire night into memory
        raw_file = get_file_path(data_dir, current_participant, current_night, ".vhdr")
        raw = mne.io.read_raw_brainvision(raw_file, preload=False, verbose='ERROR')

        # Remove any bad channels selected by the user
        _bads_to_drop = [ch for ch in bad_channels_input.value if ch in raw.ch_names]
        if _bads_to_drop:
            raw.drop_channels(_bads_to_drop)

        # Keep only reference channels that are present in the recording
        _ref_channels = [ch for ch in ref_channels_input.value if ch in raw.ch_names]

        # Assign channel types, keeping reference channels as 'eeg' temporarily so MNE can re-reference to them
        _initial_types = classify_sleep_channels(raw.ch_names, ref_chs=[])
        for _r in _ref_channels:
            if _r in _initial_types:
                _initial_types[_r] = 'eeg'
        raw.set_channel_types(_initial_types, verbose='ERROR')

        # Crop and filter each pre-awakening segment while updating a progress bar
        with mo.status.progress_bar(title="Filtering Awakenings", total=len(awakenings)) as bar:
            for _idx, (_, _row) in enumerate(awakenings.iterrows()):
                bar.update(title=f"Filtering Awakening {_idx + 1} of {len(awakenings)}...")

                # Compute the start and end times (in seconds) for the pre-awakening window
                _t_end = min(_row["Position"] / _marker_fs, raw.times[-1])
                _t_start = max(0.0, _t_end - _analysis_win)

                # Save an unfiltered (but re-referenced) copy of the first awakening for the before/after comparison plot
                if raw_example is None:
                    raw_example = raw.copy().crop(tmin=_t_start, tmax=_t_end).load_data(verbose='ERROR')
                    if _ref_channels:
                        try:
                            mne.set_eeg_reference(raw_example, ref_channels=_ref_channels, verbose='ERROR')
                        except Exception:
                            pass
                        raw_example.set_channel_types({ch: 'misc' for ch in _ref_channels if ch in raw_example.ch_names}, verbose='ERROR')

                # Add 20 seconds of extra buffer on both ends before filtering to avoid edge ringing artifacts
                pad = 20.0
                _t_start_pad = max(0.0, _t_start - pad)
                _t_end_pad = min(_t_end + pad, raw.times[-1])

                # Load the buffered segment into memory
                _segment = raw.copy().crop(tmin=_t_start_pad, tmax=_t_end_pad).load_data(verbose='ERROR')

                # Re-reference the EEG channels, then reclassify the reference electrodes as 'misc'
                if _ref_channels:
                    try:
                        mne.set_eeg_reference(_segment, ref_channels=_ref_channels, verbose='ERROR')
                    except Exception:
                        pass
                    _segment.set_channel_types({ch: 'misc' for ch in _ref_channels if ch in _segment.ch_names}, verbose='ERROR')

                # Apply the FIR bandpass filter to EEG and EOG channels
                if l_freq_input.value is not None or h_freq_input.value is not None:
                    _segment.filter(
                        l_freq=l_freq_input.value,
                        h_freq=h_freq_input.value,
                        picks=['eeg', 'eog'],
                        fir_design='firwin',
                        verbose='ERROR'
                    )

                # Filter the EMG channel between 10 and 90 Hz to isolate high-frequency muscle tone
                if 'emg' in _segment:
                    _emg_hfreq = min(90.0, (_segment.info['sfreq'] / 2.0) - 1.0)
                    if _emg_hfreq > 15.0:
                        _segment.filter(l_freq=10.0, h_freq=_emg_hfreq, picks='emg', fir_design='firwin', verbose='ERROR')

                # Apply a notch filter (and its harmonics) to remove powerline noise (e.g., 50 or 60 Hz)
                if notch_freq_input.value > 0:
                    _nyq = _segment.info['sfreq'] / 2.0
                    _notches = np.arange(notch_freq_input.value, _nyq, notch_freq_input.value)
                    if len(_notches) > 0:
                        _segment.notch_filter(freqs=_notches, picks=['eeg', 'eog', 'emg'], verbose='ERROR')

                # Remove the 20-second buffer padding and attach standard 10-20 electrode coordinates
                rel_tmin = _t_start - _t_start_pad
                rel_tmax = min(_t_end - _t_start_pad, _segment.times[-1])
                _segment.crop(tmin=rel_tmin, tmax=rel_tmax)
                _segment.set_montage('standard_1020', match_case=False, on_missing='ignore', verbose='ERROR')

                # Store the cleaned segment in the list
                preprocessed_segments.append(_segment)

        # Plot the Power Spectral Density (PSD) of Awakening #1 before and after filtering
        if raw_example is not None and len(preprocessed_segments) > 0:
            plt.style.use('default')
            fig_pre, (ax_raw, ax_filt) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, facecolor='white')

            # Plot PSD before filtering
            raw_example.compute_psd(fmax=60, picks='eeg', verbose='ERROR').plot(
                axes=ax_raw, average=True, spatial_colors=False, show=False
            )
            ax_raw.set_title("Cortical EEG PSD — BEFORE Filters (Re-referenced Raw)")
            ax_raw.set_xlabel("Frequency (Hz)")
            ax_raw.set_ylabel("Power (dB)")

            # Plot PSD after filtering
            preprocessed_segments[0].compute_psd(fmax=60, picks='eeg', verbose='ERROR').plot(
                axes=ax_filt, average=True, spatial_colors=False, show=False
            )
            ax_filt.set_title("Cortical EEG PSD — AFTER Filters")
            ax_filt.set_xlabel("Frequency (Hz)")

            fig_pre.tight_layout()

            # Display the completion message and filter validation plots
            output_ui_prep = mo.vstack([
                mo.md(f"✅ **Successfully preprocessed {len(preprocessed_segments)} awakenings** (Window: {window_size_input.value} min | Raw Sampling Rate: `{raw.info['sfreq']} Hz` | Excluded Channels: `{_bads_to_drop if _bads_to_drop else 'None'}`)"),
                mo.md("### Filter Validation (Awakening #1 — Pure EEG Channels)"),
                fig_pre
            ])
        else:
            output_ui_prep = mo.md(f"✅ **Successfully preprocessed {len(preprocessed_segments)} awakenings**")

    output_ui_prep
    return (preprocessed_segments,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <div id='artifacts-section'></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo, window_size_input):
    mo.md(rf"""
    # Automated Artifact Rejection

    Segments the continuous {window_size_input.value}-minute data into overlapping epochs and applies threshold-based rejection to remove non-cerebral noise.

    - **Flatline (Min Amp):** Detects bridging or disconnected electrodes.
    - **PTP (Max Amp):** Drops epochs exceeding physiological voltage limits.
    - **Voltage Step:** Identifies unnatural, instantaneous spikes (e.g., cable movement).
    - **EMG:** Detects high-frequency muscle interference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style='display: flex; gap: 15px;'>
        <a href='#spectral-power-section'>
            <button style='background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer;'>
                Jump to Spectral Power
            </button>
        </a>
        <a href='#spectrogram-section'>
            <button style='background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer;'>
                Jump to Spectrogram
            </button>
        </a>
    </div>
    """)
    return


@app.cell
def _(mo, preprocessed_segments):
    # Slider to set the duration (in seconds) of each epoch
    epoch_length_slider = mo.ui.slider(start=2, stop=30, step=1, value=4, label="Epoch Length (Seconds)", show_value=True)

    # Sliders for configuring voltage thresholds used to detect and reject noisy epochs
    emg_slider = mo.ui.slider(start=20, stop=300, step=10, value=100, label="Max High-Freq EMG PTP (µV)", show_value=True, debounce=True)
    step_slider = mo.ui.slider(start=10, stop=100, step=5, value=50, label="Max Voltage Step (µV/ms)", show_value=True, debounce=True)
    ptp_slider = mo.ui.slider(start=100, stop=1000, step=25, value=400, label="Max EEG Amplitude PTP (µV)", show_value=True, debounce=True)
    flat_slider = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=0.5, label="Min EEG Amplitude / Flatline (µV)", show_value=True, debounce=True)

    # Dropdown to select which awakening segment to inspect in detail
    _awake_options = {f"Awakening #{i + 1}": i for i in range(len(preprocessed_segments))}
    awake_dropdown = mo.ui.dropdown(
        options=_awake_options,
        value="Awakening #1" if _awake_options else None,
        label="Select Awakening to Analyze:"
    )

    # Display the awakening selector and artifact rejection sliders
    mo.vstack([
        mo.md("### Artifact Rejection Criteria"),
        awake_dropdown,
        epoch_length_slider,
        mo.md("---"),
        emg_slider,
        step_slider,
        ptp_slider,
        flat_slider
    ])
    return (
        awake_dropdown,
        emg_slider,
        epoch_length_slider,
        flat_slider,
        ptp_slider,
        step_slider,
    )


@app.cell
def _(
    awake_dropdown,
    emg_slider,
    epoch_length_slider,
    flat_slider,
    mne,
    mo,
    np,
    plt,
    preprocessed_segments,
    ptp_slider,
    step_slider,
):
    # Get the index of the selected awakening and initialize the list of rejected time intervals
    segment_idx = awake_dropdown.value
    rejected_intervals = []

    # Show a warning if no preprocessed awakening segment is available
    if segment_idx is None or len(preprocessed_segments) == 0:
        clean_epochs = []
        selected_segment = None
        output_ui = mo.callout(mo.md("⚠️ **No awakenings detected.** Please select a valid marker or condition above."), kind="warn")
    else:
        # Copy the selected preprocessed EEG segment
        selected_segment = preprocessed_segments[segment_idx].copy()
        mne.set_log_level('ERROR')

        # Use non-overlapping (0.0s overlap) fixed-length epochs for artifact rejection.
        # Non-overlapping epochs ensure that:
        # 1. Rejected intervals on the timeline/spectrogram never shade clean data retained by an adjacent epoch.
        # 2. All clean time segments are weighted equally (no double-counting of shared overlap regions).
        # 3. Epoch counts reflect independent time windows (Welch's PSD still uses 50% sub-window overlap when epochs > 4s).
        _ep_dur = float(epoch_length_slider.value)
        _ep_overlap = 0.0

        # Divide the continuous EEG segment into non-overlapping fixed-length epochs
        epochs = mne.make_fixed_length_epochs(
            selected_segment,
            duration=_ep_dur,
            overlap=_ep_overlap,
            preload=True,
            verbose='ERROR'
        )
        # Extract the underlying 3D NumPy array (epochs x channels x time samples)
        epoch_data = epochs.get_data(copy=False)

        # Identify indices of EEG and EMG channels and initialize per-channel failure counters
        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        emg_picks = mne.pick_types(epochs.info, emg=True)
        _actual_sfreq = epochs.info['sfreq']
        _eeg_names = [epochs.ch_names[p] for p in eeg_picks]
        _chan_fail_counts = np.zeros(len(eeg_picks), dtype=int)

        # Initialize counters for clean epochs and each artifact rejection category
        drop_counts = {
            "Retained (Clean)": 0,
            "Min Amplitude (Flatline)": 0,
            "Max Amplitude (PTP)": 0,
            "Voltage Step": 0,
            "EMG Activity": 0
        }
        valid_epochs_idx = []
        _total_dur_min = selected_segment.times[-1] / 60.0
        _ep_step_sec = _ep_dur - _ep_overlap

        # Loop through every epoch and test it against the artifact rejection thresholds
        for _i, _epoch in enumerate(epoch_data):
            dropped = False
            # Convert the maximum allowed voltage step from µV/ms to µV per sample
            _max_step_uv = step_slider.value * (1000.0 / _actual_sfreq)

            if len(eeg_picks) > 0:
                # Compute peak-to-peak amplitude and maximum sample-to-sample voltage step for each EEG channel
                _ptp_vals = np.ptp(_epoch[eeg_picks], axis=1)
                _step_vals = np.max(np.abs(np.diff(_epoch[eeg_picks], axis=1)), axis=1)
                # Flag which EEG channels failed any threshold in this epoch
                _chan_bad_mask = (
                    (_ptp_vals < flat_slider.value * 1e-6) |
                    (_ptp_vals > ptp_slider.value * 1e-6) |
                    (_step_vals > _max_step_uv * 1e-6)
                )
                _chan_fail_counts += _chan_bad_mask.astype(int)

                # Reject if any EEG channel is flatlined (below minimum amplitude)
                if not dropped and np.any(_ptp_vals < flat_slider.value * 1e-6):
                    drop_counts["Min Amplitude (Flatline)"] += 1
                    dropped = True

                # Reject if any EEG channel exceeds the maximum peak-to-peak voltage
                if not dropped and np.any(_ptp_vals > ptp_slider.value * 1e-6):
                    drop_counts["Max Amplitude (PTP)"] += 1
                    dropped = True

                # Reject if any EEG channel has an abrupt voltage step/spike
                if not dropped and np.any(_step_vals > _max_step_uv * 1e-6):
                    drop_counts["Voltage Step"] += 1
                    dropped = True

            # Reject if the EMG channel exceeds the muscle activity threshold
            if not dropped and len(emg_picks) > 0:
                if np.any(np.ptp(_epoch[emg_picks], axis=1) > emg_slider.value * 1e-6):
                    drop_counts["EMG Activity"] += 1
                    dropped = True

            # Keep the epoch if it passed all checks; otherwise record its time window as rejected
            if not dropped:
                drop_counts["Retained (Clean)"] += 1
                valid_epochs_idx.append(_i)
            else:
                _t0_min = ((_i * _ep_step_sec) / 60.0) - _total_dur_min
                _t1_min = (((_i * _ep_step_sec) + _ep_dur) / 60.0) - _total_dur_min
                rejected_intervals.append((_t0_min, _t1_min))

        # Create a new MNE Epochs object containing only the clean epochs
        clean_epochs = epochs[valid_epochs_idx]

        # Plot a bar chart summarizing how many epochs were kept or rejected by each criterion
        plt.style.use('default')
        fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5), facecolor='white')
        bars = ax_bar.bar(
            drop_counts.keys(),
            drop_counts.values(),
            color=['forestgreen', 'firebrick', 'goldenrod', 'purple', 'sienna']
        )

        ax_bar.set_ylabel('Number of Epochs')
        ax_bar.set_title(f'Artifact Rejection Summary (Total: {len(epochs)} Non-Overlapping Epochs)')
        max_val = max(drop_counts.values()) if max(drop_counts.values()) > 0 else 1
        ax_bar.set_ylim(0, max_val * 1.22)
        plt.xticks(rotation=15)

        # Display the exact epoch count above each bar
        for _bar in bars:
            _yval = _bar.get_height()
            ax_bar.text(_bar.get_x() + _bar.get_width() / 2, _yval + (max_val * 0.02), int(_yval), ha='center', va='bottom')
        fig_bar.tight_layout()

        # Plot the average EEG Power Spectral Density (PSD) before artifact rejection
        fig_before, ax_before = plt.subplots(figsize=(6, 4), facecolor='white')
        if len(epochs) > 0 and len(eeg_picks) > 0:
            epochs.compute_psd(fmax=45, picks='eeg', verbose='ERROR').plot(
                axes=ax_before, average=True, spatial_colors=False, show=False
            )
            ax_before.set_title("PSD BEFORE Rejection")
            fig_before.tight_layout()
        else:
            ax_before.text(0.5, 0.5, "⚠️ No raw epochs", ha='center', va='center', fontsize=12)
            ax_before.axis('off')

        # Plot the average EEG Power Spectral Density (PSD) after artifact rejection
        fig_after, ax_after = plt.subplots(figsize=(6, 4), facecolor='white')
        if len(clean_epochs) > 0 and len(eeg_picks) > 0:
            clean_epochs.compute_psd(fmax=45, picks='eeg', verbose='ERROR').plot(
                axes=ax_after, average=True, spatial_colors=False, show=False
            )
            ax_after.set_title("PSD AFTER Rejection")
            fig_after.tight_layout()
        else:
            ax_after.text(0.5, 0.5, "⚠️ No clean epochs retained", ha='center', va='center', fontsize=12)
            ax_after.axis('off')

        # List the top 5 EEG channels that triggered the most epoch rejections
        _top_bad_str = "None"
        if len(_eeg_names) > 0 and np.any(_chan_fail_counts > 0):
            _sorted_idx = np.argsort(_chan_fail_counts)[::-1][:5]
            _top_bad_str = ", ".join(
                f"`{_eeg_names[idx]}` ({100.0 * _chan_fail_counts[idx] / max(1, len(epochs)):.0f}%)"
                for idx in _sorted_idx if _chan_fail_counts[idx] > 0
            )

        # Combine the bar chart, retention summary, and before/after PSD plots into the UI
        output_ui = mo.vstack([
            mo.md("### Artifact Rejection Statistics"),
            fig_bar,
            mo.md(
                f"**Final Clean Epochs Retained:** `{len(clean_epochs)}` / `{len(epochs)}` (`{100 * len(clean_epochs) / max(1, len(epochs)):.1f}%`)  \n"
                f"**Top Artifact-Triggering EEG Channels:** {_top_bad_str} *(Tip: If one detached electrode is rejecting most epochs, exclude it in Pre-Processing)*"
            ),
            mo.md("---"),
            mo.md("### Signal Quality Comparison"),
            mo.hstack([fig_before, fig_after])
        ])

    output_ui
    return clean_epochs, rejected_intervals, segment_idx, selected_segment


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div id='spectral-power-section'></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spectral Power (PSD) Analysis

    Computes the spectral power of the clean EEG epochs using **Welch's Method** (4-second Hamming windows, $0.25\text{ Hz}$ frequency resolution). Power spectra are computed per epoch and aggregated across clean epochs (via mean or median) to estimate bandpower.

    * **Note on Relative Power (%):** Relative bandpower divides each band's integral by the total broadband power between the lowest and highest frequencies of the selected preset. In the **Clinical Sleep** preset, *Alpha* ($8\text{–}12\text{ Hz}$) and *Slow Spindles* ($11\text{–}13\text{ Hz}$) share the $11\text{–}12\text{ Hz}$ frequency bin; consequently, row-wise relative percentages across all clinical bands can sum to slightly more than $100\%$.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    <a href='#artifacts-section'>
        <button style='background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer;'>
            ⬆️ Change Awakening number
        </button>
    </a>
    """)
    return


@app.cell
def _(mo):
    # Dropdown to choose between Clinical Sleep frequency bands or Standard EEG frequency bands
    band_preset_select = mo.ui.dropdown(
        options=[
            "Clinical Sleep (SWA, Theta, Alpha, Spindles, Beta)",
            "Standard EEG (Delta, Theta, Alpha, Beta, Gamma)"
        ],
        value="Clinical Sleep (SWA, Theta, Alpha, Spindles, Beta)",
        label="Frequency Band Preset:"
    )

    # Radio buttons to select Absolute power (µV²) or Relative power (%)
    power_norm_select = mo.ui.radio(
        options=["Absolute (µV²)", "Relative (%)"],
        value="Absolute (µV²)",
        inline=True,
        label="Power Metric:"
    )

    # Radio buttons to choose whether epochs are combined using the Mean or the Median
    psd_agg_select = mo.ui.radio(
        options=["Mean across epochs", "Median across epochs (Robust)"],
        value="Mean across epochs",
        inline=True,
        label="Epoch Aggregation:"
    )

    # Display the spectral power configuration controls
    mo.vstack([
        mo.md("### Spectral Power Configuration"),
        band_preset_select,
        power_norm_select,
        psd_agg_select
    ])
    return band_preset_select, power_norm_select, psd_agg_select


@app.cell
def _(
    band_preset_select,
    clean_epochs,
    current_night,
    current_participant,
    epoch_length_slider,
    h_freq_input,
    mo,
    np,
    pd,
    power_norm_select,
    psd_agg_select,
    segment_idx,
    yasa,
):
    # Get the upper frequency limit from the low-pass filter setting
    _max_allowed_freq = float(h_freq_input.value) if h_freq_input.value else 45.0

    # Define frequency band ranges (in Hz) based on the chosen preset
    if "Clinical" in band_preset_select.value:
        _raw_bands = [
            (0.5, 4, 'SWA'),
            (4, 8, 'Theta'),
            (8, 12, 'Alpha'),
            (11, 13, 'Slow spindles'),
            (13, 15, 'Fast spindles'),
            (15, 25, 'Beta')
        ]
    else:
        _raw_bands = [
            (1, 4, 'Delta'),
            (4, 8, 'Theta'),
            (8, 12, 'Alpha'),
            (12, 30, 'Beta'),
            (30, 45, 'Gamma')
        ]

    # Clip the frequency bands so none exceed the low-pass filter cutoff frequency
    selected_bands = [
        (f_lo, min(f_hi, _max_allowed_freq), name)
        for f_lo, f_hi, name in _raw_bands
        if f_lo < _max_allowed_freq
    ]

    # Read the selected normalization and aggregation options
    _is_relative = "Relative" in power_norm_select.value
    _use_median = "Median" in psd_agg_select.value
    psd_obj = None
    bp_df = pd.DataFrame()

    # Compute Welch's PSD and bandpower if clean epochs are available
    if len(clean_epochs) > 0:
        # Configure a 4-second Hamming window (with 50% sub-window overlap if epochs are longer than 4s)
        _actual_sfreq = clean_epochs.info['sfreq']
        _epoch_samples = int(round(epoch_length_slider.value * _actual_sfreq))
        _welch_samples = min(int(round(4.0 * _actual_sfreq)), _epoch_samples)
        _overlap_samples = int(_welch_samples * 0.50) if _welch_samples < _epoch_samples else 0

        # Compute the Power Spectral Density for each clean epoch using Welch's method
        psd_obj = clean_epochs.compute_psd(
            method='welch',
            fmin=0.5,
            fmax=min(45.0, _max_allowed_freq),
            n_fft=_welch_samples,
            n_per_seg=_welch_samples,
            n_overlap=_overlap_samples,
            picks='eeg',
            verbose='ERROR'
        )

        # Convert PSD values from V²/Hz to µV²/Hz and aggregate across epochs using mean or median
        _psd_data_uv = psd_obj.get_data() * 1e12
        _freqs = psd_obj.freqs
        _eeg_ch_names = psd_obj.ch_names
        _psd_agg = np.median(_psd_data_uv, axis=0) if _use_median else _psd_data_uv.mean(axis=0)

        # Calculate integrated power within each frequency band for every EEG channel using YASA
        _bp = yasa.bandpower_from_psd(
            _psd_agg,
            _freqs,
            ch_names=_eeg_ch_names,
            bands=selected_bands,
            relative=_is_relative
        )

        # Convert relative power proportions (0–1) into percentages (0–100%)
        if _is_relative:
            _band_cols = [b[2] for b in selected_bands]
            _bp[_band_cols] = _bp[_band_cols] * 100.0

        # Standardize the channel column name to 'Channel' and round values to 4 decimal places
        if 'Chan' in _bp.columns:
            bp_df = _bp.rename(columns={'Chan': 'Channel'}).round(4)
        else:
            _bp.index.name = 'Channel'
            bp_df = _bp.reset_index().round(4)

        # Insert session metadata columns at the beginning of the table
        bp_df.insert(0, 'Participant', current_participant)
        bp_df.insert(1, 'Condition', current_night)
        _awake_id = segment_idx + 1 if segment_idx is not None else "Unknown"
        bp_df.insert(2, 'Awakening', _awake_id)
        bp_df.insert(3, 'Metric', power_norm_select.value)

        # Display the resulting bandpower table
        output_table = mo.vstack([
            mo.md(f"### Spectral Power Analysis — {band_preset_select.value}"),
            mo.md(f"**{psd_agg_select.value} ({power_norm_select.value}) per EEG Channel:**"),
            mo.ui.table(bp_df, page_size=25)
        ])
    else:
        # Show a warning if all epochs were rejected
        output_table = mo.callout(
            mo.md("⚠️ **No clean epochs available.** Adjust the artifact rejection criteria above to retain epochs."),
            kind="warn"
        )

    output_table
    return bp_df, psd_obj, selected_bands


@app.cell
def _(
    clean_epochs,
    current_night,
    current_participant,
    mo,
    plt,
    psd_obj,
    segment_idx,
):
    # Plot the average Power Spectral Density curve across all EEG channels if PSD data exists
    if psd_obj is not None:
        # Close previous figures and apply a clean white background style
        plt.close('all')

        plt.style.use('default')

        # Create the figure and plot the channel-averaged PSD curve
        _fig_psd, _ax_psd = plt.subplots(figsize=(10, 4.5), facecolor='white')
        psd_obj.plot(axes=_ax_psd, average=True, amplitude=False, spatial_colors=False, show=False)

        # Add title and axis labels
        _ax_psd.set_title(
            f"Average Power Spectral Density (Awakening #{segment_idx + 1})\n"
            f"Participant: {current_participant} | Condition: {current_night} | Clean Epochs: {len(clean_epochs)}"
        )
        _ax_psd.set_xlabel("Frequency (Hz)")
        _fig_psd.tight_layout()
        output_graph = _fig_psd
    else:
        output_graph = mo.md("")

    output_graph
    return


@app.cell
def _(mo):
    mo.md("""
    <hr style='border: none; height: 1.5px; width: 95%; background-color: #2c3e50; margin: 34px auto;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo, window_size_input):
    mo.md(rf"""
    # Topomap average {window_size_input.value} minutes
    """)
    return


@app.cell
def _(
    bp_df,
    clean_epochs,
    mne,
    mo,
    np,
    plt,
    power_norm_select,
    segment_idx,
    selected_bands,
    window_size_input,
):
    # Plot 2D scalp topomaps for each frequency band averaged over the full pre-awakening window
    if not bp_df.empty and len(clean_epochs) > 0:
        _eeg_info = clean_epochs.copy().pick('eeg').info
        _bp_indexed = bp_df.set_index("Channel")

        # Select EEG channels that have valid 3D electrode coordinates and exist in the bandpower table
        _valid_chs = [
            i for i, ch in enumerate(_eeg_info['chs'])
            if ch['loc'] is not None
            and not np.all(np.isnan(ch['loc'][:3]))
            and np.any(ch['loc'][:3] != 0)
            and _eeg_info['ch_names'][i] in _bp_indexed.index
        ]

        # Require at least 3 channels with valid coordinates to interpolate a scalp topomap
        if len(_valid_chs) >= 3:
            _info_sub = mne.pick_info(_eeg_info, _valid_chs)
            _band_names = [b[2] for b in selected_bands]
            _n_bands = len(_band_names)

            # Create one subplot per frequency band
            _fig_topo, _axes_topo = plt.subplots(1, _n_bands, figsize=(2.6 * _n_bands, 3.2))
            if _n_bands == 1:
                _axes_topo = [_axes_topo]

            _ch_order = _info_sub.ch_names

            # Plot the scalp topography and colorbar for each frequency band
            for _ax, (_, _, _b_name) in zip(_axes_topo, selected_bands):
                _vals = _bp_indexed.loc[_ch_order, _b_name].to_numpy(dtype=float)
                _im, _ = mne.viz.plot_topomap(
                    _vals,
                    _info_sub,
                    axes=_ax,
                    show=False,
                    cmap='RdBu_r' if "Relative" in power_norm_select.value else 'viridis',
                    contours=4
                )
                _ax.set_title(_b_name, fontsize=11, fontweight='bold')
                plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)

            # Add an overall title and adjust subplot spacing
            _fig_topo.suptitle(
                f"Whole-Window Scalp Topography ({window_size_input.value} min Average) — {power_norm_select.value} (Awakening #{segment_idx + 1})",
                fontsize=13, fontweight='bold', y=1.02
            )
            _fig_topo.tight_layout()
            _topo_out = _fig_topo
        else:
            # Inform the user if fewer than 3 channels have valid 10-20 coordinates
            _topo_out = mo.callout(
                mo.md("ℹ️ **Topomap skipped:** Fewer than 3 EEG channels matched standard 10–20 coordinates."),
                kind="info"
            )
    else:
        _topo_out = mo.md("")

    _topo_out
    return


@app.cell
def _(mo):
    mo.md("""
    <hr style='border: none; height: 1.5px; width: 95%; background-color: #2c3e50; margin: 34px auto;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive Time-Resolved Scalp Topography
    """)
    return


@app.cell
def _(
    awakenings,
    df_markers,
    event_markers_select,
    fs_input,
    mo,
    np,
    segment_idx,
    selected_segment,
    window_size_input,
):
    # Determine the duration (in minutes) of the selected pre-awakening segment
    _win_min = float(window_size_input.value)
    if selected_segment is not None:
        _win_min = round(float(selected_segment.times[-1]) / 60.0, 2)

    # Find experimental markers that occurred inside the current pre-awakening window
    _detected_win_markers = []
    _default_slider_pos = round(-_win_min / 2.0, 1)

    if not awakenings.empty and segment_idx is not None and not df_markers.empty:
        _awake_pos = awakenings.iloc[segment_idx]["Position"]
        _prev_awake_pos = awakenings.iloc[segment_idx - 1]["Position"] if segment_idx > 0 else 0
        # Filter for non-stage markers between the previous awakening and current awakening
        _cand_markers = df_markers[
            (~df_markers["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)) &
            (df_markers["Position"] > _prev_awake_pos) &
            (df_markers["Position"] < _awake_pos)
        ]
        # Convert marker sample positions into minutes relative to the awakening
        for _, _m_row in _cand_markers.iterrows():
            _rel_m = (_m_row["Position"] - _awake_pos) / fs_input.value / 60.0
            if -_win_min <= _rel_m <= 0.0:
                _desc = str(_m_row["Description"]).strip()
                if _desc:
                    _detected_win_markers.append((_desc, _rel_m))
                    # Set the initial slider position to start EXACTLY at the experimental event onset
                    if _desc in event_markers_select.value:
                        _default_slider_pos = round(float(np.clip(_rel_m, -_win_min, 0.0)), 1)

    # Dropdown to select the width of the moving time sub-window
    topo_win_select = mo.ui.dropdown(
        options={
            "15 sec (High temporal resolution)": 15.0,
            "30 sec (1 Standard Sleep Epoch — Recommended)": 30.0,
            "60 sec (1-Minute Average)": 60.0,
            "120 sec (2-Minute Average)": 120.0,
            "240 sec (4-Minute Average — Full Stimulus Block)": 240.0,
        },
        value="30 sec (1 Standard Sleep Epoch — Recommended)",
        label="Integration Sub-Window Width:"
    )

    # Radio buttons to toggle between fixed colorbar limits across time or dynamic limits per time slice
    topo_scale_select = mo.ui.radio(
        options=[
            "Fixed Scale (Compare amplitude changes across the 10m window)",
            "Dynamic Scale (Maximize spatial contrast at each time point)"
        ],
        value="Fixed Scale (Compare amplitude changes across the 10m window)",
        inline=True,
        label="Colorbar Scaling Mode:"
    )

    # Slider to scrub through time (in minutes) prior to the awakening
    # debounce=True prevents Marimo from plotting intermediate topomaps while you are dragging
    topo_time_slider = mo.ui.slider(
        start=-_win_min,
        stop=0.0,
        step=0.1,
        value=_default_slider_pos,
        label=f"Scrub Sub-Window START Time (-{_win_min:.1f} to 0.0 min):",
        full_width=True,
        show_value=True,
        debounce=True
    )

    # Format a summary string listing any markers found in the current window
    _marker_summary_str = (
        " | ".join([f"`{name}` at **{t:.2f} min**" for name, t in _detected_win_markers])
        if _detected_win_markers
        else "No experimental markers detected in this pre-awakening window"
    )

    # Display the detected markers summary and sub-window controls
    mo.vstack([
        mo.md(f"**Markers in Awakening #{segment_idx + 1 if segment_idx is not None else 1} Window:** {_marker_summary_str}"),
        mo.hstack([topo_win_select, topo_scale_select], justify="start", gap=2)
    ])
    return topo_scale_select, topo_time_slider, topo_win_select


@app.cell
def _(
    awakenings,
    clean_epochs,
    df_markers,
    df_stages,
    epoch_length_slider,
    event_markers_select,
    fs_input,
    mne,
    mo,
    np,
    plt,
    power_norm_select,
    psd_agg_select,
    psd_obj,
    rejected_intervals,
    segment_idx,
    selected_bands,
    selected_segment,
    topo_scale_select,
    topo_time_slider,
    topo_win_select,
    yasa,
):
    # Import inspect to check MNE function arguments for version compatibility
    import inspect as _inspect
    # Import ticker to manually control timeline axis granularity
    import matplotlib.ticker as _ticker

    # Build the interactive topomap and timeline figure if valid PSD and epoch data exist
    if psd_obj is not None and len(clean_epochs) > 0 and selected_segment is not None:
        _eeg_info = clean_epochs.copy().pick('eeg').info
        _psd_ch_names = psd_obj.ch_names

        # Keep only EEG channels with valid 3D coordinates that are present in the PSD object
        _valid_chs = [
            i for i, ch in enumerate(_eeg_info['chs'])
            if ch['loc'] is not None
            and not np.all(np.isnan(ch['loc'][:3]))
            and np.any(ch['loc'][:3] != 0)
            and _eeg_info['ch_names'][i] in _psd_ch_names
        ]

        if len(_valid_chs) >= 3:
            # Subset channel metadata and extract per-epoch PSD data in µV²/Hz
            _info_sub = mne.pick_info(_eeg_info, _valid_chs)
            _ch_order = _info_sub.ch_names
            _ch_indices = [_psd_ch_names.index(ch) for ch in _ch_order]

            _psd_all_uv = psd_obj.get_data()[:, _ch_indices, :] * 1e12
            _freqs = psd_obj.freqs
            _sfreq = clean_epochs.info['sfreq']
            _ep_dur = float(epoch_length_slider.value)
            _total_dur_min = float(selected_segment.times[-1]) / 60.0

            # Calculate the start, end, and center time (in minutes before awakening) for every clean epoch
            _ep_start_sec = (clean_epochs.events[:, 0] - selected_segment.first_samp) / _sfreq
            _ep_t0_min = (_ep_start_sec / 60.0) - _total_dur_min
            _ep_t1_min = ((_ep_start_sec + _ep_dur) / 60.0) - _total_dur_min
            _ep_center_min = 0.5 * (_ep_t0_min + _ep_t1_min)

            # Calculate the time bounds [_t_low, _t_high] of the active sub-window.
            # The slider value (_t_sel) now represents the START of the window.
            _sub_win_min = float(topo_win_select.value) / 60.0
            _t_sel = float(topo_time_slider.value)
        
            _t_low = _t_sel
            _t_high = _t_low + _sub_win_min
        
            # Clamp to the end of the recording (0.0) if the window overshoots (e.g., slider is at 0.0)
            # This ensures the window smoothly adapts and just takes the last possible segment.
            if _t_high > 0.0:
                _t_high = 0.0
                _t_low = max(-_total_dur_min, _t_high - _sub_win_min)
            
            # Ensure we don't undershoot the beginning of the recording
            if _t_low < -_total_dur_min:
                _t_low = -_total_dur_min
                _t_high = min(0.0, _t_low + _sub_win_min)

            # Find all clean epochs whose midpoint (or any portion) falls inside the active sub-window
            _in_win_mask = (_ep_center_min >= _t_low) & (_ep_center_min <= _t_high)
            if not np.any(_in_win_mask):
                _in_win_mask = (_ep_t0_min < _t_high) & (_ep_t1_min > _t_low)

            _is_relative = "Relative" in power_norm_select.value
            _use_median = "Median" in psd_agg_select.value
            _use_fixed_scale = "Fixed" in topo_scale_select.value

            # If Fixed Scale is selected, compute global colorbar limits (2nd to 98th percentile) across the whole window
            _vlim_map = {}
            if _use_fixed_scale:
                _step_grid = np.linspace(-_total_dur_min, max(-_total_dur_min + 0.01, -_sub_win_min), 15)
                _win_psd_list = []
                for _ws in _step_grid:
                    _we = _ws + _sub_win_min
                    _m = (_ep_center_min >= _ws) & (_ep_center_min <= _we)
                    if not np.any(_m):
                        _m = (_ep_t0_min < _we) & (_ep_t1_min > _ws)
                    if np.any(_m):
                        _agg = np.median(_psd_all_uv[_m], axis=0) if _use_median else np.mean(_psd_all_uv[_m], axis=0)
                        _win_psd_list.append(_agg)
                if _win_psd_list:
                    _stacked_psd = np.vstack(_win_psd_list)
                    _bp_all_wins = yasa.bandpower_from_psd(
                        _stacked_psd, _freqs, bands=selected_bands, relative=_is_relative
                    )
                    for _, _, _b_name in selected_bands:
                        _b_vals = _bp_all_wins[_b_name].to_numpy(dtype=float) * (100.0 if _is_relative else 1.0)
                        _vlim_map[_b_name] = (
                            float(np.nanpercentile(_b_vals, 2)),
                            float(np.nanpercentile(_b_vals, 98))
                        )

            # Create a 2-row figure: Row 1 for the band topomaps, Row 2 for the synchronized timeline bar
            _n_bands = len(selected_bands)
            _fig_dyn = plt.figure(figsize=(max(12.0, 2.6 * _n_bands), 5.0))
            _gs = _fig_dyn.add_gridspec(2, _n_bands, height_ratios=[3.3, 1.0], hspace=0.45, wspace=0.25)

            _supports_vlim = 'vlim' in _inspect.signature(mne.viz.plot_topomap).parameters
            _n_clean_in_slice = int(np.sum(_in_win_mask))

            # Plot the topomaps for the active sub-window if clean epochs exist in that time range
            if _n_clean_in_slice > 0:
                _psd_slice_agg = (
                    np.median(_psd_all_uv[_in_win_mask], axis=0)
                    if _use_median else np.mean(_psd_all_uv[_in_win_mask], axis=0)
                )
                _bp_slice = yasa.bandpower_from_psd(
                    _psd_slice_agg, _freqs, ch_names=_ch_order, bands=selected_bands, relative=_is_relative
                )
                if _is_relative:
                    for _, _, _b_name in selected_bands:
                        _bp_slice[_b_name] = _bp_slice[_b_name] * 100.0
                if 'Chan' in _bp_slice.columns:
                    _bp_slice = _bp_slice.set_index('Chan')

                # Draw one topomap per frequency band in Row 1
                for _b_idx, (_, _, _b_name) in enumerate(selected_bands):
                    _ax_t = _fig_dyn.add_subplot(_gs[0, _b_idx])
                    _vals = _bp_slice.loc[_ch_order, _b_name].to_numpy(dtype=float)
                    _cmap = 'RdBu_r' if _is_relative else 'viridis'
                    _vlim = _vlim_map.get(_b_name, (None, None)) if _use_fixed_scale else (None, None)

                    _topo_kwargs = dict(axes=_ax_t, show=False, cmap=_cmap, contours=4)
                    if _supports_vlim:
                        _topo_kwargs['vlim'] = _vlim
                    else:
                        _topo_kwargs['vmin'], _topo_kwargs['vmax'] = _vlim

                    _im, _ = mne.viz.plot_topomap(_vals, _info_sub, **_topo_kwargs)
                    _ax_t.set_title(_b_name, fontsize=11, fontweight='bold')
                    plt.colorbar(_im, ax=_ax_t, fraction=0.046, pad=0.04)
            else:
                # Show a warning message on the figure if all epochs in the selected sub-window were rejected
                _ax_warn = _fig_dyn.add_subplot(_gs[0, :])
                _ax_warn.text(
                    0.5, 0.5,
                    f"⚠️ All epochs in [{_t_low:.2f}, {_t_high:.2f}] min were rejected as artifacts.\nSlide to a clean region on the timeline below.",
                    ha='center', va='center', fontsize=12, fontweight='bold', color='#c0392b'
                )
                _ax_warn.axis('off')

            # Create the synchronized timeline subplot in Row 2 showing stages, events, and artifacts
            _ax_time = _fig_dyn.add_subplot(_gs[1, :])
            _stage_colors = {
                "W": "#f39c12", "N1": "#85c1e9", "N2": "#2e86c1",
                "N3": "#1b4f72", "REM": "#8e44ad", "ART": "#e74c3c", "UNS": "#bdc3c7"
            }

            # Draw color-coded sleep stage blocks along the bottom track of the timeline
            _current_stage_label = "Unknown"
            if not awakenings.empty and segment_idx is not None and not df_stages.empty:
                _awake_pos = awakenings.iloc[segment_idx]["Position"]
                _fs = fs_input.value
                for _, _st_row in df_stages.iterrows():
                    _st_t0 = (_st_row["Position"] - _awake_pos) / _fs / 60.0
                    _st_t1 = _st_t0 + (30.0 / 60.0)
                    if _st_t1 > -_total_dur_min and _st_t0 < 0.0:
                        _c_t0 = max(-_total_dur_min, _st_t0)
                        _c_t1 = min(0.0, _st_t1)
                        _st_name = _st_row["Stage_YASA"]
                        _ax_time.axvspan(_c_t0, _c_t1, ymin=0.0, ymax=0.32, color=_stage_colors.get(_st_name, "#bdc3c7"), alpha=0.85)
                        if (_c_t1 - _c_t0) >= 0.35:
                            _ax_time.text(0.5 * (_c_t0 + _c_t1), 0.16, _st_name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                        if _st_t0 <= _t_sel <= _st_t1:
                            _current_stage_label = _st_name

            # Draw rejected artifact intervals in red along the top track of the timeline
            for _idx_art, (_rt0, _rt1) in enumerate(rejected_intervals):
                _ax_time.axvspan(_rt0, _rt1, ymin=0.78, ymax=1.0, color='#e74c3c', alpha=0.8, label="Artifact" if _idx_art == 0 else "_nolegend_")

            # Draw vertical lines and labels for experimental markers along the middle track of the timeline
            _active_stim_label = "None (Baseline)"
            if not awakenings.empty and segment_idx is not None and not df_markers.empty:
                _awake_pos = awakenings.iloc[segment_idx]["Position"]
                _prev_awake_pos = awakenings.iloc[segment_idx - 1]["Position"] if segment_idx > 0 else 0
                _win_m = df_markers[
                    (~df_markers["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)) &
                    (df_markers["Position"] > _prev_awake_pos) &
                    (df_markers["Position"] < _awake_pos)
                ]
                for _, _mr in _win_m.iterrows():
                    _m_min = (_mr["Position"] - _awake_pos) / fs_input.value / 60.0
                    if -_total_dur_min <= _m_min <= 0.0:
                        _m_desc = str(_mr["Description"]).strip()
                        _is_exp = _m_desc in event_markers_select.value
                        _m_col = '#16a085' if _is_exp else '#34495e'
                        _ax_time.axvline(_m_min, color=_m_col, linestyle='--', linewidth=2.0)
                        if _is_exp:
                            # Shade the stimulation period from marker onset until awakening
                            _ax_time.axvspan(_m_min, 0.0, ymin=0.32, ymax=0.78, color='#1abc9c', alpha=0.18)
                            if _t_sel >= _m_min:
                                _active_stim_label = _m_desc
                        _ax_time.text(
                            _m_min, 0.72, f" {_m_desc} ({_m_min:.1f}m)",
                            color=_m_col, fontweight='bold', fontsize=9, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=_m_col, alpha=0.9)
                        )

            # Highlight the currently selected time sub-window in yellow on the timeline
            _ax_time.axvspan(_t_low, _t_high, color='#f1c40f', alpha=0.45, label=f"Active Window [{_t_low:.2f}, {_t_high:.2f}] min")
            _ax_time.axvline(_t_low, color='#d35400', linewidth=1.8)
            _ax_time.axvline(_t_high, color='#d35400', linewidth=1.8)

            # Configure axes limits, ticks, and labels for the timeline bar
            _ax_time.set_xlim(-_total_dur_min, 0.0)
            _ax_time.set_ylim(0.0, 1.0)
            _ax_time.set_yticks([0.16, 0.55, 0.89])
            _ax_time.set_yticklabels(["Stage", "Events", "Artifacts"], fontsize=9, fontweight='bold')
        
            # Adaptive X-Axis Ticks
            if _total_dur_min <= 15:
                # For 10m-15m windows: Major tick every 1 minute, minor tick every 30s
                _ax_time.xaxis.set_major_locator(_ticker.MultipleLocator(1.0))
                _ax_time.xaxis.set_minor_locator(_ticker.MultipleLocator(0.5))
            elif _total_dur_min <= 30:
                # For 15m-30m windows: Major tick every 2 minutes, minor tick every 1 minute
                _ax_time.xaxis.set_major_locator(_ticker.MultipleLocator(2.0))
                _ax_time.xaxis.set_minor_locator(_ticker.MultipleLocator(1.0))
            else:
                # For very large windows: Let Matplotlib scale automatically (max 12 ticks)
                _ax_time.xaxis.set_major_locator(_ticker.MaxNLocator(nbins=12))

            _ax_time.set_xlabel("Time Relative to Awakening (Minutes) — Drag slider below to scrub", fontsize=10, fontweight='bold')
        
            # Grid lines follow the major ticks (solid) and minor ticks (dotted) for easy alignment
            _ax_time.grid(axis='x', which='major', linestyle='-', alpha=0.5)
            _ax_time.grid(axis='x', which='minor', linestyle=':', alpha=0.3)

            # Add a dynamic title showing the current time window, sleep stage, active stimulus, and clean epoch count
            _fig_dyn.suptitle(
                f"Time-Resolved Topography: [{_t_low:.2f} to {_t_high:.2f} min] | "
                f"Stage: {_current_stage_label} | Stimulus: {_active_stim_label} | Clean Epochs: {_n_clean_in_slice}",
                fontsize=12.5, fontweight='bold', y=0.98
            )

            # Stack the figure and the interactive time slider vertically
            _dyn_topo_out = mo.vstack([
                _fig_dyn,
                topo_time_slider
            ])
        else:
            _dyn_topo_out = mo.md("")
    else:
        _dyn_topo_out = mo.md("")

    _dyn_topo_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div id='spectrogram-section'></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo, window_size_input):
    mo.md(rf"""
    # Time-Frequency Spectrogram

    Visualizes frequency shifts during the {window_size_input.value} minutes before awakening using a Short-Time Fourier Transform (STFT).

    **Technical Note:** To prevent phase cancellation when multiple channels are selected, the STFT is calculated for each channel independently. The resulting matrices are then averaged together (Power-Domain Averaging) before being log-transformed (dB) into the final heatmap.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    <a href='#artifacts-section'>
        <button style='background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer;'>
            ⬆️ Change Awakening number
        </button>
    </a>
    """)
    return


@app.cell
def _(mo, selected_segment):
    # Get the list of available EEG channels from the selected segment and choose default frontal/central channels
    if selected_segment is None:
        options = []
        fallback_chan = []
    else:
        options = selected_segment.copy().pick('eeg').ch_names
        fallback_chan = [ch for ch in ['F3', 'F4', 'C3', 'C4'] if ch in options][:2] or (options[:1] if options else [])

    # Multiselect menu for choosing which EEG channels to average in the spectrogram
    tfr_chan_select = mo.ui.multiselect(
        options=options,
        value=fallback_chan,
        label="EEG Channels (Power-domain average if multiple selected):"
    )

    # Sliders and dropdown for configuring STFT window length, max frequency, and heatmap colors
    tfr_window_slider = mo.ui.slider(start=1, stop=30, step=1, value=4, label="STFT Window Length (sec):", show_value=True, debounce=True)
    tfr_fmax_slider = mo.ui.slider(start=10, stop=45, step=1, value=25, label="Max Frequency (Hz):", show_value=True, debounce=True)
    tfr_cmap_select = mo.ui.dropdown(
        options=["RdBu_r", "viridis", "inferno", "magma", "Spectral_r", "jet"],
        value="RdBu_r",
        label="Colormap (RdBu_r = YASA Default):"
    )

    # Display the spectrogram configuration controls
    mo.vstack([
        mo.md("### ⚙️ Spectrogram Settings"),
        tfr_chan_select,
        tfr_window_slider,
        tfr_fmax_slider,
        tfr_cmap_select
    ])
    return tfr_chan_select, tfr_cmap_select, tfr_fmax_slider, tfr_window_slider


@app.cell
def _(
    awakenings,
    current_night,
    current_participant,
    df_markers,
    event_markers_select,
    fs_input,
    mo,
    np,
    plt,
    rejected_intervals,
    segment_idx,
    selected_segment,
    tfr_chan_select,
    tfr_cmap_select,
    tfr_fmax_slider,
    tfr_window_slider,
):
    # Import SciPy's signal processing module to compute the Short-Time Fourier Transform (STFT)
    import scipy.signal as signal

    # Read the user's channel, window length, and frequency limit selections
    selected_chans = tfr_chan_select.value
    current_win_sec = tfr_window_slider.value
    current_fmax = tfr_fmax_slider.value

    # Show a warning if no EEG channel is selected
    if not selected_chans or selected_segment is None:
        output_tfr = mo.callout(mo.md("⚠️ **Please select at least one EEG channel.**"), kind="warn")
    else:
        # Extract the EEG signal in microvolts (µV) for the selected channels
        data_nd_uv = selected_segment.get_data(picks=selected_chans) * 1e6
        _actual_sfreq = selected_segment.info['sfreq']

        # Calculate the STFT window length in samples and set an 87.5% overlap for smooth time resolution
        nperseg = int(round(current_win_sec * _actual_sfreq))
        noverlap = int(round(nperseg * 0.875))

        # Compute the spectrogram independently for each selected EEG channel
        power_matrices = []
        for chan_idx in range(data_nd_uv.shape[0]):
            freqs, times, Sxx = signal.spectrogram(
                data_nd_uv[chan_idx, :],
                fs=_actual_sfreq,
                window='hamming',
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
            power_matrices.append(Sxx)

        # Average the spectrograms across channels in the power domain
        power_3d = np.stack(power_matrices)
        Sxx_mean = np.mean(power_3d, axis=0)

        # Restrict the spectrogram to frequencies between 0.5 Hz and the selected maximum frequency
        freq_mask = (freqs >= 0.5) & (freqs <= current_fmax)
        freqs_plot = freqs[freq_mask]
        Sxx_plot = Sxx_mean[freq_mask, :]

        # Convert power values to decibels (dB) and clip extreme values at the 5th and 95th percentiles
        Sxx_db = 10.0 * np.log10(np.maximum(Sxx_plot, 1e-12))
        vmin, vmax = np.percentile(Sxx_db, [5, 95])
        Sxx_db_clipped = np.clip(Sxx_db, vmin, vmax)

        # Create the spectrogram figure and convert time into minutes relative to the awakening
        fig_tfr, ax_tfr = plt.subplots(figsize=(12, 5.4))

        total_dur_min = selected_segment.times[-1] / 60.0
        times_rel_mins = (times / 60.0) - total_dur_min

        # Render the time-frequency heatmap
        mesh = ax_tfr.pcolormesh(
            times_rel_mins,
            freqs_plot,
            Sxx_db_clipped,
            shading='auto',
            cmap=tfr_cmap_select.value,
            vmin=vmin,
            vmax=vmax,
            rasterized=True
        )

        # Draw red bars just above the top border of the plot to indicate rejected artifact epochs
        for _idx_art, (_t0, _t1) in enumerate(rejected_intervals):
            ax_tfr.axvspan(
                _t0, _t1,
                ymin=1.005, ymax=1.04,
                color='#e74c3c', alpha=0.9,
                clip_on=False,
                label="Rejected Artifact Epoch" if _idx_art == 0 else "_nolegend_"
            )

        # Draw vertical dashed lines for experimental events and place their labels above the top border
        _num_visible_events = 0
        if not awakenings.empty and segment_idx is not None:
            _awake_pos = awakenings.iloc[segment_idx]["Position"]
            _active_ev = df_markers[df_markers["Description"].isin(event_markers_select.value)]
            for _, _ev_row in _active_ev.iterrows():
                _ev_rel_min = (_ev_row["Position"] - _awake_pos) / fs_input.value / 60.0
                if -total_dur_min <= _ev_rel_min <= 0:
                    ax_tfr.axvline(_ev_rel_min, color='black', linestyle='--', linewidth=2.0)
                    ax_tfr.axvline(_ev_rel_min, color='white', linestyle='--', linewidth=1.2)

                    # Alternate label heights if multiple markers are present so text boxes do not overlap
                    _y_offset_pts = 8 + (_num_visible_events % 2) * 18
                    ax_tfr.annotate(
                        f"{_ev_row['Description']} ({_ev_rel_min:.1f}m)",
                        xy=(_ev_rel_min, 1.0),
                        xycoords=('data', 'axes fraction'),
                        xytext=(0, _y_offset_pts),
                        textcoords='offset points',
                        color='white',
                        fontweight='bold',
                        fontsize=9.5,
                        ha='center',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#2c3e50', edgecolor='black', alpha=0.9),
                        clip_on=False
                    )
                    _num_visible_events += 1

        # Add a legend above the top-left corner if any artifact epochs were marked
        if rejected_intervals:
            ax_tfr.legend(
                loc='lower left',
                bbox_to_anchor=(0.0, 1.04),
                frameon=True,
                facecolor='white',
                edgecolor='#cccccc',
                fontsize=9.5
            )

        # Set font sizes and add the colorbar
        LABEL_SIZE = 14
        TICK_SIZE = 12
        TITLE_SIZE = 13

        cbar = fig_tfr.colorbar(mesh, ax=ax_tfr, pad=0.02)
        cbar.set_label("Power Spectral Density (dB re 1 µV²/Hz)", rotation=270, labelpad=20, fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        # Format the channel list for the plot title
        chan_label = "+".join(selected_chans) if len(selected_chans) <= 4 else f"Mean of {len(selected_chans)} EEG channels"
        _awake_num = segment_idx + 1 if segment_idx is not None else 0

        # Add title, axis labels, and axis limits
        _title_pad = 38 if (_num_visible_events <= 1) else 54
        ax_tfr.set_title(
            f"Pre-Awakening Spectrogram ({chan_label})\n"
            f"Participant: {current_participant} | Condition: {current_night} | Awakening #{_awake_num}",
            pad=_title_pad, fontsize=TITLE_SIZE, fontweight='bold'
        )
        ax_tfr.set_xlabel("Time Relative to Awakening (Minutes)", fontsize=LABEL_SIZE)
        ax_tfr.set_ylabel("Frequency (Hz)", fontsize=LABEL_SIZE)

        ax_tfr.set_xlim(times_rel_mins[0], 0.0)
        ax_tfr.set_ylim(0.5, current_fmax)
        ax_tfr.tick_params(axis='both', labelsize=TICK_SIZE)
        fig_tfr.tight_layout()

        output_tfr = fig_tfr

    output_tfr
    return


@app.cell
def _(mo):
    # Create a button to trigger the batch export table on demand so adjusting sliders above stays fast
    batch_run_button = mo.ui.run_button(
        label="🔄 Generate / Refresh Session Summary Table (All Awakenings)"
    )

    # Display the section divider, header, and run button
    mo.vstack([
        mo.md("<hr style='border: none; height: 5px; background-color: #2c3e50; margin: 100px 0;'>"),
        mo.md("# Session Summary (All Awakenings Combined)"),
        mo.md(
            "Combines pre-awakening sleep stage metadata and channel-level spectral power across **all awakenings** "
            "in this recording for direct export (e.g., to `.csv`). Click the button below once you are satisfied "
            "with your filter, artifact rejection, and spectral power settings."
        ),
        batch_run_button
    ])
    return (batch_run_button,)


@app.cell
def _(
    batch_run_button,
    emg_slider,
    epoch_length_slider,
    flat_slider,
    h_freq_input,
    mne,
    mo,
    np,
    pd,
    power_norm_select,
    preprocessed_segments,
    psd_agg_select,
    ptp_slider,
    selected_bands,
    step_slider,
    summary_df,
    yasa,
):
    # Only compute the full-night batch table when the user clicks the run button
    if not batch_run_button.value:
        _batch_ui = mo.callout(
            mo.md("ℹ️ Click **Generate / Refresh Session Summary Table** above to compute the combined export table across all awakenings."),
            kind="info"
        )
    elif len(preprocessed_segments) > 0 and not summary_df.empty:
        # Show a loading message while computing Welch PSDs across all awakenings
        mo.output.replace(
            mo.callout(
                mo.md("### ⏳ Computing spectral power across all awakenings...\n*Please wait, building master export table.*"),
                kind="info"
            )
        )

        _batch_rows = []
        # Retrieve the current epoching, filter, and spectral power parameters (using non-overlapping epochs)
        _ep_dur_b = float(epoch_length_slider.value)
        _ep_ov_b = 0.0
        _max_f_b = float(h_freq_input.value) if h_freq_input.value else 45.0
        _is_rel_b = "Relative" in power_norm_select.value
        _use_med_b = "Median" in psd_agg_select.value

        # Loop through each preprocessed awakening segment
        for _aw_i, _seg in enumerate(preprocessed_segments):
            # Segment the data into non-overlapping fixed-length epochs
            _eps = mne.make_fixed_length_epochs(_seg, duration=_ep_dur_b, overlap=_ep_ov_b, preload=True, verbose='ERROR')
            _ep_data = _eps.get_data(copy=False)
            _eeg_p = mne.pick_types(_eps.info, eeg=True)
            _emg_p = mne.pick_types(_eps.info, emg=True)
            _sf = _eps.info['sfreq']
            _max_step = step_slider.value * (1000.0 / _sf)

            # Apply the artifact rejection thresholds to identify clean epochs
            _keep = []
            for _ei, _ep in enumerate(_ep_data):
                _bad = False
                if len(_eeg_p) > 0:
                    _ptp = np.ptp(_ep[_eeg_p], axis=1)
                    _stp = np.max(np.abs(np.diff(_ep[_eeg_p], axis=1)), axis=1)
                    if np.any(_ptp < flat_slider.value * 1e-6) or np.any(_ptp > ptp_slider.value * 1e-6) or np.any(_stp > _max_step * 1e-6):
                        _bad = True
                if not _bad and len(_emg_p) > 0 and np.any(np.ptp(_ep[_emg_p], axis=1) > emg_slider.value * 1e-6):
                    _bad = True
                if not _bad:
                    _keep.append(_ei)

            # Skip this awakening if no clean epochs remain
            _clean_eps = _eps[_keep]
            if len(_clean_eps) == 0:
                continue

            # Configure the 4-second Welch window parameters (with 50% sub-window overlap when epochs > 4s)
            _ep_samps = int(round(_ep_dur_b * _sf))
            _w_samps = min(int(round(4.0 * _sf)), _ep_samps)
            _ov_samps = int(_w_samps * 0.50) if _w_samps < _ep_samps else 0

            # Compute Welch's PSD across clean epochs and aggregate using mean or median
            _psd_b = _clean_eps.compute_psd(
                method='welch', fmin=0.5, fmax=min(45.0, _max_f_b),
                n_fft=_w_samps, n_per_seg=_w_samps, n_overlap=_ov_samps,
                picks='eeg', verbose='ERROR'
            )
            _psd_uv_b = _psd_b.get_data() * 1e12
            _psd_agg_b = np.median(_psd_uv_b, axis=0) if _use_med_b else _psd_uv_b.mean(axis=0)

            # Compute bandpower for each EEG channel using YASA
            _bp_b = yasa.bandpower_from_psd(
                _psd_agg_b, _psd_b.freqs, ch_names=_psd_b.ch_names,
                bands=selected_bands, relative=_is_rel_b
            )
            if _is_rel_b:
                _b_cols = [b[2] for b in selected_bands]
                _bp_b[_b_cols] = _bp_b[_b_cols] * 100.0

            # Format the channel column
            if 'Chan' in _bp_b.columns:
                _df_aw = _bp_b.rename(columns={'Chan': 'Channel'}).round(4)
            else:
                _bp_b.index.name = 'Channel'
                _df_aw = _bp_b.reset_index().round(4)

            # Insert explicit numeric epoch retention and analysis parameter columns for clean statistical export
            _n_clean = len(_clean_eps)
            _n_total = len(_eps)
            _ret_pct = round(100.0 * _n_clean / max(1, _n_total), 1)

            _df_aw.insert(0, 'Aggregation', "Median" if _use_med_b else "Mean")
            _df_aw.insert(0, 'Metric', power_norm_select.value)
            _df_aw.insert(0, 'Retention_Pct', _ret_pct)
            _df_aw.insert(0, 'Total_Epochs', _n_total)
            _df_aw.insert(0, 'Clean_Epochs', _n_clean)
            _df_aw.insert(0, 'Epoch_Length_s', _ep_dur_b)
            _df_aw.insert(0, 'Awakening', _aw_i + 1)
            _batch_rows.append(_df_aw)

        # Merge the sleep stage summary metadata with the channel bandpower tables across all awakenings
        if _batch_rows:
            _all_bp_df = pd.concat(_batch_rows, ignore_index=True)
            _meta_cols = [
                "Participant", "Condition", "Awakening", "Time (HH:MM:SS)",
                "Last Stage at Awakening", "Dominant Stage", "Stage Sequence",
                "Event Type", "Event Latency (MM:SS)"
            ]
            master_export_df = pd.merge(summary_df[_meta_cols], _all_bp_df, on="Awakening", how="inner")
            _batch_ui = mo.vstack([
                mo.md(f"✅ **Batch Summary Ready:** `{master_export_df['Awakening'].nunique()}` awakenings exported (`{len(master_export_df)}` channel-rows)."),
                mo.ui.table(master_export_df, page_size=20)
            ])
        else:
            _batch_ui = mo.callout(mo.md("⚠️ **No clean epochs retained across awakenings for batch export.**"), kind="warn")
    else:
        _batch_ui = mo.md("")

    _batch_ui
    return


@app.cell
def _():
    # ==============================================================================
    # ONERA PIPELINE — COMPLETE FUNCTION & VARIABLE INDEX
    # ==============================================================================
    #
    # ------------------------------------------------------------------------------
    # 1. FUNCTIONS (WITH CREATION CELL & DESCRIPTION)
    # ------------------------------------------------------------------------------
    # - get_file_path(directory, participant, condition, extension)
    #     Created in: Code Cell 2 (Core Imports, File Parsers & Session Selection)
    #     Description: Searches a folder for a file matching the given participant,
    #                  condition, and extension, and returns the first sorted match.
    #
    # - read_brainvision_markers(filepath)
    #     Created in: Code Cell 2 (Core Imports, File Parsers & Session Selection)
    #     Description: Parses BrainVision Analyzer (.Markers) or standard (.vmrk)
    #                  files into a chronologically sorted pandas DataFrame.
    #
    # - numeric_sort_key(p_id)
    #     Created in: Code Cell 2 (Core Imports, File Parsers & Session Selection)
    #     Description: Extracts the numeric portion of a participant ID so IDs sort
    #                  in natural numerical order (e.g., DS2 before DS10).
    #
    # - analyze_sleep_stages(participant, night)
    #     Created in: Code Cell 4 (Pre-Awakening Sleep Stage Extraction)
    #     Description: Extracts sleep stages and experimental events preceding each
    #                  awakening and builds a pre-awakening summary DataFrame.
    #
    # - _make_abbrev(name, used)
    #     Created in: Code Cell 6 (Whole-Night Hypnogram Plot & Sleep Macro-Statistics)
    #     Description: Creates a short, unique text abbreviation (e.g., "StMH") for
    #                  marker labels displayed above the hypnogram.
    #
    # - _hours_to_mdates(x)
    #     Created in: Code Cell 6 (Whole-Night Hypnogram Plot & Sleep Macro-Statistics)
    #     Description: Converts elapsed recording hours into Matplotlib date numbers
    #                  for the hypnogram's 24-hour clock time axis.
    #
    # - _mdates_to_hours(x)
    #     Created in: Code Cell 6 (Whole-Night Hypnogram Plot & Sleep Macro-Statistics)
    #     Description: Converts Matplotlib date numbers back into elapsed recording
    #                  hours for the hypnogram's 24-hour clock time axis.
    #
    # - classify_sleep_channels(ch_names, ref_chs=None)
    #     Created in: Code Cell 8 (Signal Pre-Processing Execution & Filter Validation)
    #     Description: Automatically assigns polysomnography channel names to MNE
    #                  channel types ('eeg', 'eog', 'emg', 'ecg', or 'misc').
    #
    # ------------------------------------------------------------------------------
    # 2. SHARED / EXPORTED VARIABLES (USED ACROSS MULTIPLE CELLS)
    # ------------------------------------------------------------------------------
    # - mo                     : Marimo library module used to build interactive UI elements.
    # - Path                   : pathlib.Path class used for handling file system paths.
    # - mne                    : MNE-Python module for loading, filtering, and plotting EEG data.
    # - yasa                   : YASA module for hypnogram statistics and bandpower computation.
    # - pd                     : pandas module for creating and manipulating data tables.
    # - np                     : NumPy module for numerical array operations.
    # - plt                    : Matplotlib pyplot module for generating figures.
    # - data_dir_input         : UI text input holding the path to the raw EEG folder.
    # - markers_dir_input      : UI text input holding the path to the marker files folder.
    # - data_dir               : Path object pointing to the raw EEG directory.
    # - markers_dir            : Path object pointing to the markers directory.
    # - all_unique_labels      : Set of all non-stage marker names detected across marker files.
    # - fs_input               : UI number input for the marker sampling rate in Hz.
    # - participant_dropdown   : UI dropdown for selecting the active participant ID.
    # - night_dropdown         : UI dropdown for selecting the active night or condition.
    # - awake_marker_select    : UI dropdown for choosing the marker that defines an awakening.
    # - window_size_input      : UI number input setting the pre-awakening window length (minutes).
    # - event_markers_select   : UI multiselect for choosing experimental stimulation markers.
    # - current_participant    : String ID of the currently selected participant.
    # - current_night          : String name of the currently selected night/condition.
    # - df_markers             : DataFrame containing all markers for the current session.
    # - df_stages              : DataFrame containing only scored sleep stage markers.
    # - awakenings             : DataFrame containing only the targeted awakening markers.
    # - summary_df             : DataFrame summarizing sleep stages and events per awakening.
    # - hypno_markers_select   : UI multiselect choosing which markers to overlay on the hypnogram.
    # - ref_channels_input     : UI multiselect choosing EEG reference electrodes (e.g., M1, M2).
    # - bad_channels_input     : UI multiselect choosing bad/detached channels to drop.
    # - l_freq_input           : UI number input for the EEG high-pass filter cutoff (Hz).
    # - h_freq_input           : UI number input for the EEG low-pass filter cutoff (Hz).
    # - notch_freq_input       : UI number input for the powerline notch filter frequency (Hz).
    # - preprocessed_segments  : List of filtered MNE Raw objects (one per pre-awakening window).
    # - awake_dropdown         : UI dropdown to pick which awakening index to inspect.
    # - epoch_length_slider    : UI slider setting the epoch duration in seconds.
    # - emg_slider             : UI slider setting the max EMG peak-to-peak threshold (µV).
    # - step_slider            : UI slider setting the max EEG voltage step threshold (µV/ms).
    # - ptp_slider             : UI slider setting the max EEG peak-to-peak threshold (µV).
    # - flat_slider            : UI slider setting the min EEG flatline amplitude threshold (µV).
    # - segment_idx            : Integer index (0-based) of the currently selected awakening.
    # - selected_segment       : MNE Raw object for the currently selected pre-awakening window.
    # - clean_epochs           : MNE Epochs object containing only artifact-free epochs.
    # - rejected_intervals     : List of (start_min, end_min) tuples for rejected epochs.
    # - band_preset_select     : UI dropdown selecting Clinical Sleep or Standard EEG frequency bands.
    # - power_norm_select      : UI radio button selecting Absolute (µV²) or Relative (%) power.
    # - psd_agg_select         : UI radio button selecting Mean or Median aggregation across epochs.
    # - selected_bands         : List of (low_hz, high_hz, band_name) tuples for PSD integration.
    # - psd_obj                : MNE Spectrum object holding per-epoch Welch PSD estimates.
    # - bp_df                  : DataFrame of channel-level spectral bandpower for the selected awakening.
    # - topo_win_select        : UI dropdown setting the sub-window width (sec) for dynamic topomaps.
    # - topo_scale_select      : UI radio button toggling Fixed vs. Dynamic colorbar scaling.
    # - topo_time_slider       : UI slider scrubbing through time (minutes) before awakening.
    # - tfr_chan_select        : UI multiselect choosing EEG channels for the spectrogram.
    # - tfr_window_slider      : UI slider setting the STFT window length (seconds).
    # - tfr_fmax_slider        : UI slider setting the maximum spectrogram frequency (Hz).
    # - tfr_cmap_select        : UI dropdown selecting the spectrogram colormap.
    #
    # ------------------------------------------------------------------------------
    # 3. CELL-LOCAL VARIABLES (ORGANIZED BY PIPELINE SECTION)
    # ------------------------------------------------------------------------------
    # [Code Cell 2 — File Discovery & Session Setup]
    # - all_marker_files       : List of all .Markers and .vmrk file paths in markers_dir.
    # - unique_participants    : Set of unique participant IDs parsed from filenames.
    # - unique_conditions      : Set of unique condition names parsed from filenames.
    # - detected_fs            : Sampling rate (Hz) auto-detected from marker or .vhdr headers.
    # - participant_list       : Numerically sorted list of participant IDs.
    # - condition_list         : Alphabetically sorted list of conditions/nights.
    # - detected_labels_str    : Comma-separated string of all detected event marker names.
    # - vhdr_files             : List of .vhdr files found in data_dir.
    # - _status_banner         : UI callout or markdown text showing detected marker status.
    #
    # [Code Cell 3 — Targeting & Window Parameters]
    # - _labels                : Sorted list of available marker names for dropdown options.
    # - _default_events        : Default list of music event markers if present in _labels.
    #
    # [Code Cell 4 — Pre-Awakening Sleep Stage Extraction]
    # - STAGE_MAP              : Dictionary mapping raw stage strings to standard YASA stage codes.
    # - _stage_ui              : UI element displaying the pre-awakening stage summary table.
    #
    # [Code Cell 5 & 6 — Whole-Night Hypnogram & Sleep Statistics]
    # - non_stage_markers      : Subset of df_markers excluding sleep stages and segment headers.
    # - current_night_labels   : Unique event marker names present in the current night.
    # - default_markers        : Default markers selected for the hypnogram overlay.
    # - _fig_hypno, _ax_hypno  : Matplotlib Figure and Axes for the whole-night hypnogram.
    # - _stages_list           : Chronological list of YASA sleep stage strings for the night.
    # - _hypno                 : yasa.Hypnogram instance used for plotting and sleep statistics.
    # - _stats_dict, _stats_df : Dictionary and DataFrame of whole-night macro-architecture metrics.
    # - _offset_hours          : Time offset (hours) from recording start to the first scored stage.
    # - _win_hours             : Pre-awakening analysis window duration expressed in hours.
    # - _palette               : List of hex color codes used for plotting event markers.
    # - _used_abbrevs          : Set tracking marker abbreviations already assigned on the plot.
    # - _start_time            : datetime object representing the real-world clock start of the EEG.
    # - _sec_ax                : Secondary bottom x-axis displaying 24-hour wall-clock time.
    # - _fig_out               : Combined UI stack containing the hypnogram and statistics table.
    #
    # [Code Cell 7 & 8 — Signal Pre-Processing]
    # - _raw_file, raw_file    : Path to the current session's .vhdr BrainVision header file.
    # - _raw_info, raw         : MNE Raw BrainVision object representing the recording.
    # - _ch_names              : List of all channel names in the raw recording.
    # - _default_refs          : Default mastoid/earlobe reference channels detected in _ch_names.
    # - _bads_to_drop          : List of user-selected bad channels to remove from raw data.
    # - _ref_channels          : List of valid reference channels selected by the user.
    # - _initial_types         : Dictionary mapping channel names to initial MNE channel types.
    # - raw_example            : Unfiltered copy of Awakening #1 used for filter validation plots.
    # - pad                    : Buffer duration (20.0 seconds) added before filtering to prevent edge ringing.
    # - _segment               : Cropped and filtered MNE Raw segment for a single awakening.
    # - fig_pre, ax_raw, ax_filt: Matplotlib figure and axes comparing PSD before vs. after filtering.
    # - output_ui_prep         : UI stack showing preprocessing status and filter validation plots.
    #
    # [Code Cell 9 & 10 — Automated Artifact Rejection]
    # - _awake_options         : Dictionary mapping "Awakening #N" labels to 0-based segment indices.
    # - epochs                 : MNE Epochs object containing all fixed-length epochs before rejection.
    # - epoch_data             : 3D NumPy array of raw epoch voltages (epochs x channels x samples).
    # - eeg_picks, emg_picks   : Arrays of channel indices for cortical EEG and chin EMG channels.
    # - _chan_fail_counts      : Array counting how many epochs each EEG channel failed.
    # - drop_counts            : Dictionary counting clean epochs and rejections by artifact type.
    # - valid_epochs_idx       : List of integer indices corresponding to clean epochs.
    # - fig_bar, ax_bar, bars  : Matplotlib figure, axes, and bars for the artifact summary chart.
    # - fig_before, ax_before  : Matplotlib figure and axes for the PSD plot before artifact rejection.
    # - fig_after, ax_after    : Matplotlib figure and axes for the PSD plot after artifact rejection.
    # - _top_bad_str           : Formatted string listing the top 5 artifact-triggering EEG channels.
    # - output_ui              : UI stack displaying artifact statistics and before/after PSD plots.
    #
    # [Code Cell 13, 14 & 16 — Spectral Power (PSD) & Whole-Window Topomaps]
    # - _max_allowed_freq      : Maximum frequency (Hz) allowed by the low-pass filter cutoff.
    # - _raw_bands             : Unclipped list of frequency band tuples for the chosen preset.
    # - _psd_data_uv           : 3D array of per-epoch PSD values converted to µV²/Hz.
    # - _psd_agg               : 2D array of PSD values aggregated (mean or median) across clean epochs.
    # - _bp                    : Raw YASA bandpower DataFrame before formatting.
    # - output_table           : UI stack displaying the channel-level bandpower table.
    # - _fig_psd, _ax_psd      : Matplotlib figure and axes for the average PSD curve plot.
    # - output_graph           : UI output holding the average PSD curve figure.
    # - _eeg_info, _info_sub   : MNE Info objects for all EEG channels and channels with valid 3D coordinates.
    # - _valid_chs             : List of EEG channel indices that have valid 3D montage coordinates.
    # - _fig_topo, _axes_topo  : Matplotlib figure and axes for the whole-window scalp topomaps.
    # - _topo_out              : UI output holding the whole-window topomap figure.
    #
    # [Code Cell 18 & 19 — Interactive Time-Resolved Scalp Topography]
    # - _detected_win_markers  : List of (description, time_min) tuples for markers in the current window.
    # - _default_slider_pos    : Initial time slider position (minutes) relative to awakening.
    # - _marker_summary_str    : Formatted text listing markers inside the pre-awakening window.
    # - _psd_all_uv            : 3D array of clean epoch PSDs for channels with valid coordinates.
    # - _ep_t0_min, _ep_t1_min : Start and end times (minutes before awakening) of each clean epoch.
    # - _ep_center_min         : Midpoint time (minutes before awakening) of each clean epoch.
    # - _t_low, _t_high        : Lower and upper time bounds (minutes) of the active slider sub-window.
    # - _in_win_mask           : Boolean mask selecting clean epochs inside [_t_low, _t_high].
    # - _vlim_map              : Dictionary storing fixed (vmin, vmax) colorbar limits per band.
    # - _fig_dyn, _gs          : Matplotlib figure and GridSpec for the interactive topomap and timeline.
    # - _ax_time               : Matplotlib axes for the synchronized bottom timeline strip.
    # - _stage_colors          : Dictionary mapping sleep stages to hex colors on the timeline.
    # - _current_stage_label   : Sleep stage string at the current slider position.
    # - _active_stim_label     : Name of the active experimental stimulus at the current slider position.
    # - _dyn_topo_out          : UI stack combining the dynamic topomap figure and time slider.
    #
    # [Code Cell 21 & 22 — Time-Frequency Spectrogram]
    # - options, fallback_chan : Available EEG channel names and default channels for the spectrogram.
    # - selected_chans         : List of EEG channels currently selected for the spectrogram.
    # - current_win_sec        : STFT window duration in seconds.
    # - current_fmax           : Maximum frequency (Hz) displayed on the spectrogram y-axis.
    # - data_nd_uv             : 2D array of continuous EEG data in µV for the selected channels.
    # - nperseg, noverlap      : Number of samples per STFT segment and number of overlapping samples.
    # - power_matrices         : List of 2D spectrogram power arrays (one per selected channel).
    # - power_3d, Sxx_mean     : 3D stacked array and channel-averaged 2D power spectrogram.
    # - freqs, times           : Frequency (Hz) and time (sec) vectors returned by scipy.signal.spectrogram.
    # - freqs_plot, Sxx_plot   : Frequency-masked frequency vector and spectrogram matrix.
    # - Sxx_db, Sxx_db_clipped : Log-transformed (dB) spectrogram before and after 5th–95th percentile clipping.
    # - vmin, vmax             : 5th and 95th percentile dB values used for heatmap color scaling.
    # - fig_tfr, ax_tfr, mesh  : Matplotlib figure, axes, and QuadMesh for the spectrogram plot.
    # - times_rel_mins         : Time vector expressed in minutes relative to awakening (-window to 0).
    # - output_tfr             : UI output holding the spectrogram figure.
    #
    # [Code Cell 23 — Batch Session Summary]
    # - _batch_rows            : List of per-awakening bandpower DataFrames collected during batch loop.
    # - _all_bp_df             : Concatenated DataFrame of channel bandpowers across all awakenings.
    # - _meta_cols             : List of sleep stage and event metadata columns merged from summary_df.
    # - master_export_df       : Final merged DataFrame combining metadata and bandpower for all awakenings.
    # - _batch_ui              : UI stack displaying the combined session summary table for CSV export.
    #
    #
    # Thanks for using my pipeline, don't hesitate to star it on GitHub is it has been useful to you!
    return


if __name__ == "__main__":
    app.run()
