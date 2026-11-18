import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <style>
    button { height: auto !important; min-height: 34px !important; white-space: nowrap !important; padding: 5px 11px !important; line-height: 1.2 !important; }
    marimo-button, marimo-dropdown, marimo-number, marimo-radio { flex-shrink: 0 !important; }
    marimo-download { flex-shrink: 0 !important; }
    marimo-download a, marimo-download button { white-space: nowrap !important; height: auto !important; min-height: 34px !important; }
    .marimo-vstack { gap: 12px !important; }
    </style>

    # Onera SleepScorer

    **Author:** Alexandru Lupu | **GitHub:** [alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)

    - **Automated 200 Hz Staging:** Runs YASA's pre-trained LightGBM sleep classifiers on 30-second epochs using EEG, EOG, and EMG channels.
    - **Multi-Electrode Consensus:** Optionally scores multiple EEG derivations and aggregates predictions via soft probability averaging or hard majority voting.
    - **Wake Interval Overrides:** Forces epochs before `LightsOff` and between matched Start/Stop ignore marker pairs (e.g., `Start_DreamReport` → `Stop_DreamReport`) to Wake (`W`).
    - **Interactive Plotly Triage:** Zoom, pan, click, or box-select epochs directly on the hypnogram for instant `< 1 ms` pre-filtered waveform inspection, stage overrides, and `.Markers`/`.txt` export.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    paths_form = mo.md(
        """
        {data_dir}

        {output_dir}
        """
    ).batch(
        data_dir=mo.ui.text(
            value="K:\\",
            label="📂 Raw Data Path (.vhdr, .vmrk, .eeg):",
            full_width=True,
        ),
        output_dir=mo.ui.text(
            value="K:\\",
            label="📂 Output Directory for Scored .Markers:",
            full_width=True,
        ),
    ).form(submit_button_label="Scan & Load Directories")

    paths_form
    return Path, mo, paths_form


@app.cell
def _(Path, mo, paths_form):
    mo.stop(
        paths_form.value is None,
        mo.callout(
            mo.md("👆 **Verify your paths above and click `Scan & Load Directories` to begin.**"),
            kind="info",
        ),
    )

    import html as html_lib
    import os
    os.environ["MNE_USE_NUMBA"] = "false"
    import datetime
    import re
    import warnings
    import mne
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import plotly.graph_objects as go
    from sklearn.exceptions import InconsistentVersionWarning
    import yasa

    mne.set_log_level("ERROR")
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    data_dir = Path(paths_form.value["data_dir"].strip())
    _out_str = paths_form.value["output_dir"].strip()
    default_output_dir = Path(_out_str) if _out_str else data_dir

    def numeric_sort_key(p_id):
        """Sort participant IDs naturally (e.g., DS2 before DS10)."""
        _nums = re.findall(r"\d+", str(p_id))
        return (int(_nums[0]), str(p_id)) if _nums else (0, str(p_id))

    def parse_session_stem(stem):
        """Extract (participant, condition) from a BrainVision filename stem."""
        _parts = stem.split("_")
        if len(_parts) >= 2:
            return _parts[0], _parts[1]
        return stem, "Default"

    def get_file_path(directory, participant, condition, extension):
        """Locate a file matching participant, condition, and extension."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        _ext_lower = extension.lower()
        if condition == "Default":
            _exact = [
                _f for _f in directory.iterdir()
                if _f.is_file() and _f.suffix.lower() == _ext_lower and _f.stem.lower() == participant.lower()
            ]
            if _exact:
                return sorted(_exact)[0]
        _prefix_1 = f"{participant}_{condition}_".lower()
        _prefix_2 = f"{participant}_{condition}{_ext_lower}".lower()
        _matches = [
            _f for _f in directory.iterdir()
            if _f.is_file()
            and _f.suffix.lower() == _ext_lower
            and (_f.name.lower().startswith(_prefix_1) or _f.name.lower() == _prefix_2)
        ]
        if not _matches:
            raise FileNotFoundError(f"No {extension} file found for '{participant}_{condition}' in {directory}")
        return sorted(_matches)[0]

    def read_brainvision_markers(filepath):
        """Parse a BrainVision .vmrk or .Markers file into a chronologically sorted DataFrame."""
        _rows = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as _fh:
            for _line in _fh:
                _line_s = _line.strip()
                if not _line_s or _line_s.startswith(";") or _line_s.startswith("["):
                    continue
                if _line_s.startswith(("Sampling rate", "Brain Vision", "DataFile", "MarkerFile", "Codepage", "Type")):
                    continue
                _date_field = ""
                if re.match(r"^Mk\d+=", _line_s):
                    _line_s = _line_s.split("=", 1)[1]
                _parts = [_p.strip() for _p in _line_s.split(",")]
                if len(_parts) >= 5:
                    if len(_parts) >= 6:
                        _date_field = _parts[5]
                    _rows.append(_parts[:5] + [_date_field])

        _cols = ["Type", "Description", "Position", "Length", "Channel", "Date"]
        _df = pd.DataFrame(_rows, columns=_cols)
        if _df.empty:
            return _df
        _df["Position"] = pd.to_numeric(_df["Position"], errors="coerce")
        _df["Length"] = pd.to_numeric(_df["Length"], errors="coerce").fillna(1)
        _df["Type"] = _df["Type"].astype(str).str.strip()
        _df["Description"] = _df["Description"].astype(str).str.strip()
        _df["Channel"] = _df["Channel"].astype(str).str.strip()
        return _df.dropna(subset=["Position"]).sort_values("Position", kind="stable").reset_index(drop=True)

    def detect_vhdr_metadata(vhdr_path):
        """Read SamplingInterval and header metadata from a BrainVision .vhdr file."""
        _meta = {"sfreq": None, "marker_file": None}
        try:
            with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as _fh:
                for _line in _fh:
                    _ls = _line.strip()
                    if _ls.startswith("SamplingInterval="):
                        _val = float(_ls.split("=", 1)[1].strip())
                        if _val > 0:
                            _meta["sfreq"] = int(round(1_000_000.0 / _val))
                    elif _ls.startswith("MarkerFile="):
                        _meta["marker_file"] = _ls.split("=", 1)[1].strip()
        except Exception:
            pass
        return _meta

    def classify_sleep_channels(ch_names, ref_chs=None):
        """Classify PSG channel names into MNE physiological types ('eeg', 'eog', 'emg', 'ecg', 'misc')."""
        _ref_set = set(ref_chs or [])
        _ch_types = {}
        for _ch in ch_names:
            _up = _ch.upper()
            if _ch in _ref_set or _up in {"M1", "M2", "A1", "A2"}:
                _ch_types[_ch] = "misc"
            elif any(_k in _up for _k in ("EMG", "CHIN", "SUBMENT")):
                # Check EMG before EOG so names like 'EMG_eye' stay typed as EMG rather than EOG
                _ch_types[_ch] = "emg"
            elif any(_k in _up for _k in ("EOG", "LOC", "ROC", "HEOG", "VEOG", "E1", "E2")):
                _ch_types[_ch] = "eog"
            elif any(_k in _up for _k in ("ECG", "EKG", "HEART")):
                _ch_types[_ch] = "ecg"
            elif any(_k in _up for _k in ("RESP", "FLOW", "THOR", "ABDO", "SNORE", "SPO2", "PLETH", "POS", "SCR", "EDA", "GSR")):
                _ch_types[_ch] = "misc"
            else:
                _ch_types[_ch] = "eeg"
        return _ch_types

    def format_hms(total_seconds):
        """Format elapsed seconds into HH:MM:SS."""
        _sec = max(0, int(round(float(total_seconds))))
        return f"{_sec // 3600:02d}:{(_sec % 3600) // 60:02d}:{_sec % 60:02d}"

    def is_calibration_marker(desc):
        """Identify pre-sleep biocalibration markers (eyes, teeth, blinks) to omit from plots."""
        return bool(re.search(r"eyes?|teeth|blink|calib|biocal", str(desc), re.IGNORECASE))

    # Reactive state stores: frozen scored bundle, dynamic ignore pairs, manual edits, epoch cursor & multi-selection
    get_scored_bundle, set_scored_bundle = mo.state(None)
    get_n_extra_pairs, set_n_extra_pairs = mo.state(0, allow_self_loops=True)
    get_manual_edits, set_manual_edits = mo.state({}, allow_self_loops=True)
    get_active_epoch_state, set_active_epoch_state = mo.state(0, allow_self_loops=True)
    get_selected_epochs, set_selected_epochs = mo.state([], allow_self_loops=True)
    get_show_all_epochs, set_show_all_epochs = mo.state(False, allow_self_loops=True)
    zoom_view_state = {}

    add_ignore_pair_btn = mo.ui.button(
        label="➕\u00a0Add\u00a0Pair",
        kind="neutral",
        on_click=lambda _: set_n_extra_pairs(lambda n: n + 1),
    )
    remove_ignore_pair_btn = mo.ui.button(
        label="➖\u00a0Remove\u00a0Pair",
        kind="neutral",
        on_click=lambda _: set_n_extra_pairs(lambda n: max(0, n - 1)),
    )

    _vhdr_files = []
    if data_dir.exists():
        _vhdr_files = sorted([
            _f for _f in data_dir.iterdir()
            if _f.is_file() and _f.suffix.lower() == ".vhdr"
        ])

    _participants = set()
    _conditions = set()
    session_catalog = {}
    for _vf in _vhdr_files:
        _p_id, _cond = parse_session_stem(_vf.stem)
        _participants.add(_p_id)
        _conditions.add(_cond)
        session_catalog.setdefault((_p_id, _cond), _vf)

    participant_list = sorted(list(_participants), key=numeric_sort_key)
    condition_list = sorted(list(_conditions))

    participant_dropdown = mo.ui.dropdown(
        options=participant_list,
        value=participant_list[0] if participant_list else None,
        label="Participant:",
    )
    night_dropdown = mo.ui.dropdown(
        options=condition_list,
        value=condition_list[0] if condition_list else None,
        label="Night / Condition:",
    )

    if not data_dir.exists():
        _scan_msg = mo.callout(mo.md(f"⚠️ **Directory not found:** `{data_dir}`"), kind="warn")
    elif not _vhdr_files:
        _scan_msg = mo.callout(mo.md(f"⚠️ **No `.vhdr` files found in** `{data_dir}`"), kind="warn")
    else:
        _scan_msg = mo.md(f"*Found `{len(_vhdr_files)}` recording(s) across `{len(participant_list)}` participant(s).*")

    _scan_msg
    return (
        add_ignore_pair_btn,
        classify_sleep_channels,
        data_dir,
        datetime,
        default_output_dir,
        detect_vhdr_metadata,
        format_hms,
        get_active_epoch_state,
        get_file_path,
        get_manual_edits,
        get_n_extra_pairs,
        get_scored_bundle,
        get_selected_epochs,
        get_show_all_epochs,
        go,
        html_lib,
        is_calibration_marker,
        mne,
        night_dropdown,
        np,
        participant_dropdown,
        pd,
        plt,
        read_brainvision_markers,
        remove_ignore_pair_btn,
        session_catalog,
        set_active_epoch_state,
        set_manual_edits,
        set_scored_bundle,
        set_selected_epochs,
        set_show_all_epochs,
        ticker,
        yasa,
        zoom_view_state,
    )


@app.cell
def _(
    data_dir,
    datetime,
    detect_vhdr_metadata,
    format_hms,
    get_file_path,
    html_lib,
    is_calibration_marker,
    mne,
    mo,
    night_dropdown,
    participant_dropdown,
    pd,
    read_brainvision_markers,
    session_catalog,
):
    current_participant = participant_dropdown.value
    current_night = night_dropdown.value

    session_vhdr_path = None
    session_vmrk_path = None
    session_ch_names = []
    session_duration_sec = 0.0
    session_meas_date = None
    df_session_vmrk = pd.DataFrame(columns=["Type", "Description", "Position", "Length", "Channel", "Date"])
    session_marker_labels = []
    _detected_fs = 500
    _session_err = None

    if current_participant and current_night:
        try:
            session_vhdr_path = session_catalog.get((current_participant, current_night))
            if session_vhdr_path is None:
                session_vhdr_path = get_file_path(data_dir, current_participant, current_night, ".vhdr")

            _vhdr_meta = detect_vhdr_metadata(session_vhdr_path)
            _raw_hdr = mne.io.read_raw_brainvision(session_vhdr_path, preload=False, verbose="ERROR")
            session_ch_names = list(_raw_hdr.ch_names)
            _mne_sf = float(_raw_hdr.info["sfreq"])
            session_duration_sec = float(_raw_hdr.n_times / _mne_sf) if _mne_sf > 0 else 0.0
            _detected_fs = _vhdr_meta["sfreq"] if _vhdr_meta["sfreq"] else int(round(_mne_sf))
            session_meas_date = _raw_hdr.info.get("meas_date")

            _vmrk_candidates = []
            if _vhdr_meta.get("marker_file"):
                _vmrk_candidates.append(session_vhdr_path.parent / _vhdr_meta["marker_file"])
            _vmrk_candidates.append(session_vhdr_path.with_suffix(".vmrk"))
            for _vc in _vmrk_candidates:
                if _vc.exists() and _vc.is_file():
                    session_vmrk_path = _vc
                    break
            if session_vmrk_path is None:
                try:
                    session_vmrk_path = get_file_path(data_dir, current_participant, current_night, ".vmrk")
                except FileNotFoundError:
                    session_vmrk_path = None

            if session_vmrk_path is not None and session_vmrk_path.exists():
                df_session_vmrk = read_brainvision_markers(session_vmrk_path)
                if not df_session_vmrk.empty:
                    _non_stage = df_session_vmrk[
                        ~df_session_vmrk["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)
                    ]
                    session_marker_labels = sorted([
                        str(_m).strip()
                        for _m in _non_stage["Description"].dropna().unique()
                        if str(_m).strip() != ""
                    ])
                    if session_meas_date is None:
                        _new_segs = df_session_vmrk[
                            df_session_vmrk["Type"].str.contains("New Segment", case=False, na=False)
                        ]
                        if not _new_segs.empty:
                            _d_str = str(_new_segs.iloc[0].get("Date", "")).strip()
                            if len(_d_str) >= 14 and _d_str[:14].isdigit():
                                try:
                                    session_meas_date = datetime.datetime.strptime(_d_str[:14], "%Y%m%d%H%M%S")
                                except Exception:
                                    pass
        except Exception as _exc:
            _session_err = str(_exc)

    fs_input = mo.ui.number(
        value=_detected_fs,
        start=50,
        stop=10000,
        step=1,
        label="Sampling Rate (Hz):",
    )

    if _session_err:
        _status_card = mo.callout(mo.md(f"⚠️ **Session Error:** `{_session_err}`"), kind="warn")
    elif session_vhdr_path is None:
        _status_card = mo.callout(mo.md("⚠️ **No matching `.vhdr` file selected.**"), kind="warn")
    else:
        _n_ep_est = int(session_duration_sec // 30.0)

        def _meta_pill(label, value, bg="rgba(127, 140, 141, 0.14)", border="rgba(127, 140, 141, 0.35)"):
            return (
                f"<span style='display:inline-flex;align-items:center;gap:4px;padding:4px 10px;"
                f"border-radius:999px;font-size:12.5px;line-height:1.25;background:{bg};border:1px solid {border};'>"
                f"<span style='opacity:0.78;'>{html_lib.escape(str(label))}</span>"
                f"<strong>{html_lib.escape(str(value))}</strong></span>"
            )

        def _marker_pill(name):
            _low = name.lower().replace("_", "").replace(" ", "").replace("-", "")
            if "lightsoff" in _low or "lightoff" in _low or "lightson" in _low:
                _bg, _bd = "rgba(39, 174, 96, 0.15)", "rgba(39, 174, 96, 0.45)"
            elif is_calibration_marker(name):
                _bg, _bd = "rgba(127, 140, 141, 0.12)", "rgba(127, 140, 141, 0.30)"
            else:
                _bg, _bd = "rgba(46, 134, 193, 0.15)", "rgba(46, 134, 193, 0.45)"
            return (
                f"<span style='display:inline-flex;align-items:center;padding:3px 9px;border-radius:999px;"
                f"font-size:11.8px;font-weight:600;line-height:1.2;background:{_bg};border:1px solid {_bd};'>"
                f"{html_lib.escape(name)}</span>"
            )

        _row1 = "".join([
            _meta_pill("Recording:", session_vhdr_path.name),
            _meta_pill("Duration:", format_hms(session_duration_sec)),
            _meta_pill("Epochs:", f"{_n_ep_est} (30s)"),
            _meta_pill("Channels:", str(len(session_ch_names))),
        ])
        _marker_pills = "".join(_marker_pill(m) for m in session_marker_labels) if session_marker_labels else "<span style='opacity:0.65;font-size:12px;'>None</span>"
        _row2 = f"<span style='font-size:12.5px;opacity:0.80;font-weight:600;'>Markers:</span>{_marker_pills}"

        _status_card = mo.Html(
            "<div style='display:flex;flex-direction:column;gap:8px;padding:10px 14px;"
            "border-radius:10px;border:1px solid rgba(127,140,141,0.32);background:rgba(127,140,141,0.06);'>"
            f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;'>{_row1}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center;'>{_row2}</div>"
            "</div>"
        )

    mo.vstack([
        mo.hstack([participant_dropdown, night_dropdown, fs_input], justify="start", gap=1.2, align="end"),
        _status_card,
    ])
    return (
        current_night,
        current_participant,
        df_session_vmrk,
        fs_input,
        session_ch_names,
        session_marker_labels,
        session_meas_date,
        session_vhdr_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <hr style='border: none; height: 3px; background-color: #2c3e50; margin: 50px 0;'>
    # Configuration
    """)
    return


@app.cell
def _(
    add_ignore_pair_btn,
    classify_sleep_channels,
    get_n_extra_pairs,
    mo,
    remove_ignore_pair_btn,
    session_ch_names,
    session_marker_labels,
):
    # 1. Wake override controls (LightsOff + dynamic Start/Stop ignore pairs)
    _lo_none = "(None — Start from Sample 1)"
    _lo_options = [_lo_none] + list(session_marker_labels)
    _default_lo = _lo_none
    for _cand in session_marker_labels:
        if _cand.lower().replace("_", "").replace(" ", "").replace("-", "") in ("lightsoff", "lightoff"):
            _default_lo = _cand
            break

    lightsoff_dropdown = mo.ui.dropdown(
        options=_lo_options,
        value=_default_lo,
        label="LightsOff Anchor Marker:",
    )

    _pair_none = "(None)"
    _pair_options = [_pair_none] + list(session_marker_labels)
    _def_start_1 = "Start_DreamReport" if "Start_DreamReport" in session_marker_labels else _pair_none
    _def_stop_1 = "Stop_DreamReport" if "Stop_DreamReport" in session_marker_labels else _pair_none

    _total_pairs = 1 + int(get_n_extra_pairs())
    _start_widgets = []
    _stop_widgets = []
    for _idx in range(_total_pairs):
        _start_widgets.append(
            mo.ui.dropdown(
                options=_pair_options,
                value=_def_start_1 if _idx == 0 else _pair_none,
                label=f"Ignore Pair #{_idx + 1} Start:",
            )
        )
        _stop_widgets.append(
            mo.ui.dropdown(
                options=_pair_options,
                value=_def_stop_1 if _idx == 0 else _pair_none,
                label=f"Ignore Pair #{_idx + 1} Stop:",
            )
        )

    ignore_start_dropdowns = mo.ui.array(_start_widgets)
    ignore_stop_dropdowns = mo.ui.array(_stop_widgets)
    _pair_rows = [
        mo.hstack([ignore_start_dropdowns[_i], ignore_stop_dropdowns[_i]], justify="start", gap=1.2, align="end")
        for _i in range(_total_pairs)
    ]

    # 2. Channel selection with smart defaults (prefer EMG_chin/EMG over EMG_eye/EMG_jaw)
    _ch_list = list(session_ch_names)
    _ch_types = classify_sleep_channels(_ch_list, ref_chs=[])
    _default_refs = [_ch for _ch in ("M1", "M2", "A1", "A2") if _ch in _ch_list][:2]
    _eeg_candidates = [_ch for _ch in _ch_list if _ch_types.get(_ch) == "eeg"] or _ch_list
    _eog_candidates = [_ch for _ch in _ch_list if _ch_types.get(_ch) == "eog"]
    _emg_candidates = [_ch for _ch in _ch_list if _ch_types.get(_ch) == "emg"]

    def _pick_preferred(candidates, prefs):
        _lookup = {_c.upper(): _c for _c in candidates}
        for _p in prefs:
            if _p.upper() in _lookup:
                return _lookup[_p.upper()]
        return candidates[0] if candidates else None

    def _emg_priority(ch_name):
        _u = ch_name.upper()
        if "CHIN" in _u or "SUBMENT" in _u:
            return (0, _u)
        if _u in ("EMG", "EMG1", "EMG2") or _u.startswith("EMG_CH"):
            return (1, _u)
        if "EYE" not in _u and "JAW" not in _u and "ORBIC" not in _u and "ZYGO" not in _u:
            return (2, _u)
        return (3, _u)

    _none_opt = "(None)"
    _default_eeg = _pick_preferred(_eeg_candidates, ("C4", "C3", "F4", "F3", "Cz", "Fz"))
    _default_eog = _pick_preferred(_eog_candidates, ("EOG_H", "EOGli", "EOGre", "EOG_V", "EOG1", "EOG2", "EOG", "LOC", "ROC")) or _none_opt
    _emg_sorted = sorted(_emg_candidates, key=_emg_priority)
    _default_emg = _emg_sorted[0] if _emg_sorted else _none_opt

    ref_channels_input = mo.ui.multiselect(
        options=_ch_list,
        value=_default_refs,
        label="EEG Reference Channels:",
    )
    eeg_chan_dropdown = mo.ui.dropdown(
        options=_eeg_candidates,
        value=_default_eeg,
        label="Primary EEG Channel:",
    )
    eog_chan_dropdown = mo.ui.dropdown(
        options=[_none_opt] + _ch_list,
        value=_default_eog,
        label="EOG Channel (Optional):",
    )
    emg_chan_dropdown = mo.ui.dropdown(
        options=[_none_opt] + _ch_list,
        value=_default_emg,
        label="EMG Channel (Optional):",
    )

    # 3. Multi-electrode consensus controls
    multi_electrode_checkbox = mo.ui.checkbox(
        value=False,
        label="🧠 Enable Multi-Electrode Consensus",
    )
    electrode_preset_dropdown = mo.ui.dropdown(
        options=[
            "Preset 1: Frontal-Central Quad [F3, F4, C3, C4] (Recommended)",
            "Preset 2: Contralateral Pair [F4, C3]",
            "Custom EEG Channels",
        ],
        value="Preset 1: Frontal-Central Quad [F3, F4, C3, C4] (Recommended)",
        label="Multi-Electrode Preset:",
    )
    _default_custom = [_c for _c in _eeg_candidates if _c.upper() in {"F3", "F4", "C3", "C4"}][:4] or _eeg_candidates[:2]
    custom_eeg_multiselect = mo.ui.multiselect(
        options=_eeg_candidates,
        value=_default_custom,
        label="Custom EEG Channels:",
    )

    # 4. Confidence threshold (default 60%) & explicit scoring run button
    confidence_slider = mo.ui.slider(
        start=30,
        stop=99,
        step=1,
        value=60,
        label="🎚️ Confidence Threshold (%):",
        show_value=True,
        debounce=True,
    )
    exclude_forced_wake_checkbox = mo.ui.checkbox(
        value=True,
        label="Exclude forced Wake overrides from flags",
    )
    run_scoring_btn = mo.ui.run_button(
        label="▶\u00a0Run\u00a0Automated\u00a0Sleep\u00a0Scoring",
        kind="success",
    )

    mo.vstack([
        mo.hstack([lightsoff_dropdown, add_ignore_pair_btn, remove_ignore_pair_btn], justify="start", gap=1.2, align="end", wrap=True),
        *_pair_rows,
        mo.md("---"),
        mo.hstack([ref_channels_input, eeg_chan_dropdown, eog_chan_dropdown, emg_chan_dropdown], justify="start", gap=1.2, align="end", wrap=True),
        mo.md("---"),
        mo.hstack([multi_electrode_checkbox, electrode_preset_dropdown, custom_eeg_multiselect], justify="start", gap=1.2, align="end", wrap=True),
        mo.md("---"),
        mo.hstack([confidence_slider, exclude_forced_wake_checkbox, run_scoring_btn], justify="start", gap=1.2, align="end", wrap=True),
    ])
    return (
        confidence_slider,
        custom_eeg_multiselect,
        eeg_chan_dropdown,
        electrode_preset_dropdown,
        emg_chan_dropdown,
        eog_chan_dropdown,
        exclude_forced_wake_checkbox,
        ignore_start_dropdowns,
        ignore_stop_dropdowns,
        lightsoff_dropdown,
        multi_electrode_checkbox,
        ref_channels_input,
        run_scoring_btn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 3px; background-color: #2c3e50; margin: 50px 0;'>
    """)
    return


@app.cell
def _(
    classify_sleep_channels,
    confidence_slider,
    current_night,
    current_participant,
    custom_eeg_multiselect,
    df_session_vmrk,
    eeg_chan_dropdown,
    electrode_preset_dropdown,
    emg_chan_dropdown,
    eog_chan_dropdown,
    exclude_forced_wake_checkbox,
    format_hms,
    fs_input,
    get_scored_bundle,
    ignore_start_dropdowns,
    ignore_stop_dropdowns,
    lightsoff_dropdown,
    mne,
    mo,
    multi_electrode_checkbox,
    np,
    pd,
    ref_channels_input,
    run_scoring_btn,
    session_meas_date,
    session_vhdr_path,
    set_active_epoch_state,
    set_scored_bundle,
    set_selected_epochs,
    yasa,
):
    _STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]

    def _normalize_stage(label):
        _s = str(label).strip().upper()
        if _s in ("WAKE", "W"):
            return "W"
        if _s in ("R", "REM"):
            return "REM"
        return _s

    def _align_proba_df(proba_df):
        _renamed = proba_df.rename(columns=_normalize_stage)
        return _renamed.reindex(columns=_STAGE_ORDER, fill_value=0.0).to_numpy(dtype=float)

    def _resolve_preset_channels(preset_label, raw_ch_names, fallback_eeg):
        _preset_map = {
            "Preset 1: Frontal-Central Quad [F3, F4, C3, C4] (Recommended)": ["F3", "F4", "C3", "C4"],
            "Preset 2: Contralateral Pair [F4, C3]": ["F4", "C3"],
        }
        _targets = list(custom_eeg_multiselect.value or []) if preset_label == "Custom EEG Channels" else _preset_map.get(preset_label, [])
        _upper_lookup = {_c.upper(): _c for _c in raw_ch_names}
        _resolved = []
        for _t in _targets:
            _match = _upper_lookup.get(_t.upper())
            if _match and _match not in _resolved:
                _resolved.append(_match)
        if not _resolved and fallback_eeg in raw_ch_names:
            _resolved = [fallback_eeg]
        return _resolved

    def _match_marker_intervals(df_vmrk, start_label, stop_label):
        if df_vmrk.empty or not start_label or not stop_label or start_label == "(None)" or stop_label == "(None)":
            return []
        _events = df_vmrk[df_vmrk["Description"].isin([start_label, stop_label])].sort_values("Position", kind="stable")
        _intervals = []
        _pending_start = None
        for _, _er in _events.iterrows():
            _desc = _er["Description"]
            _pos = float(_er["Position"])
            if _desc == start_label and _pending_start is None:
                _pending_start = _pos
            elif _desc == stop_label and _pending_start is not None:
                if _pos > _pending_start:
                    _intervals.append((_pending_start, _pos))
                _pending_start = None
        return _intervals

    if not run_scoring_btn.value:
        if get_scored_bundle() is None:
            _scoring_cell_out = mo.callout(
                mo.md("👆 **Adjust settings above and click `▶ Run Automated Sleep Scoring` to begin.**"),
                kind="info",
            )
        else:
            _scoring_cell_out = mo.md("")
    else:
        _scoring_error_msg = None
        if session_vhdr_path is None or eeg_chan_dropdown.value is None:
            _scoring_error_msg = "No BrainVision recording or primary EEG channel selected."
        else:
            mo.output.replace(
                mo.callout(
                    mo.md("### ⏳ Scoring 30-second epochs at 200 Hz & caching filtered waveforms...\n*Please wait a moment.*"),
                    kind="info",
                )
            )
            try:
                _raw = mne.io.read_raw_brainvision(session_vhdr_path, preload=True, verbose="ERROR")
                _raw_sf = float(_raw.info["sfreq"])
                _fs_marker = float(fs_input.value) if fs_input.value else _raw_sf

                _ref_chs = [c for c in (ref_channels_input.value or []) if c in _raw.ch_names]
                _init_types = classify_sleep_channels(_raw.ch_names, ref_chs=[])
                for _r in _ref_chs:
                    if _r in _init_types:
                        _init_types[_r] = "eeg"
                _raw.set_channel_types(_init_types, verbose="ERROR")

                if _ref_chs:
                    try:
                        mne.set_eeg_reference(_raw, ref_channels=_ref_chs, verbose="ERROR")
                    except Exception:
                        pass
                    _raw.set_channel_types({c: "misc" for c in _ref_chs if c in _raw.ch_names}, verbose="ERROR")

                _eog_ch = None if eog_chan_dropdown.value in (None, "(None)") else eog_chan_dropdown.value
                _emg_ch = None if emg_chan_dropdown.value in (None, "(None)") else emg_chan_dropdown.value
                if _eog_ch not in _raw.ch_names:
                    _eog_ch = None
                if _emg_ch not in _raw.ch_names:
                    _emg_ch = None

                if multi_electrode_checkbox.value:
                    _active_eegs = _resolve_preset_channels(
                        electrode_preset_dropdown.value, _raw.ch_names, eeg_chan_dropdown.value
                    )
                else:
                    _active_eegs = [eeg_chan_dropdown.value]

                _active_eegs = [c for c in _active_eegs if c not in _ref_chs and c in _raw.ch_names]
                if not _active_eegs:
                    raise ValueError("No valid EEG channels available after excluding references.")

                _needed_chs = []
                for _c in _active_eegs + [_eog_ch, _emg_ch]:
                    if _c and _c in _raw.ch_names and _c not in _needed_chs:
                        _needed_chs.append(_c)

                _raw_inspector = _raw.copy().pick(_needed_chs)
                # Pre-resample ONCE to 200 Hz before multi-electrode loop for fast, high-bandwidth YASA scoring
                _raw_200 = (
                    _raw_inspector.copy().resample(200, npad="auto", verbose="ERROR")
                    if abs(_raw_sf - 200.0) > 1e-3
                    else _raw_inspector
                )

                _pass_probas = []
                _pass_preds = []
                for _eeg_ch in _active_eegs:
                    _sls = yasa.SleepStaging(_raw_200, eeg_name=_eeg_ch, eog_name=_eog_ch, emg_name=_emg_ch)
                    _hyp = _sls.predict()
                    _pass_probas.append(_align_proba_df(_hyp.proba))
                    _pass_preds.append([_normalize_stage(s) for s in _hyp.hypno.tolist()])

                _n_epochs = min(_p.shape[0] for _p in _pass_probas)
                _proba_stack = np.stack([_p[:_n_epochs] for _p in _pass_probas], axis=0)
                _avg_proba = np.mean(_proba_stack, axis=0)
                _n_passes = _proba_stack.shape[0]

                _raw_yasa_stages = []
                _agreement_pcts = []
                _confidence_pcts = []
                _runner_ups = []
                _margin_pcts = []
                _vote_summaries = []

                for _ep in range(_n_epochs):
                    _ep_probs = _avg_proba[_ep]
                    _votes = [_pass_preds[_m][_ep] for _m in range(_n_passes)]
                    _chosen = _STAGE_ORDER[int(np.argmax(_ep_probs))]

                    _chosen_idx = _STAGE_ORDER.index(_chosen)
                    _agree = 100.0 * (_votes.count(_chosen) / float(_n_passes))
                    _conf = 100.0 * float(_ep_probs[_chosen_idx])

                    _sorted_idxs = np.argsort(_ep_probs)[::-1]
                    _top2_idx = _sorted_idxs[1] if _sorted_idxs[0] == _chosen_idx else _sorted_idxs[0]
                    _runner = _STAGE_ORDER[int(_top2_idx)]
                    _margin = 100.0 * float(_ep_probs[_chosen_idx] - _ep_probs[_top2_idx])

                    _vote_counts = pd.Series(_votes).value_counts()
                    _vote_str = " | ".join(f"{k}:{v}" for k, v in _vote_counts.items())

                    _raw_yasa_stages.append(_chosen)
                    _agreement_pcts.append(round(_agree, 1))
                    _confidence_pcts.append(round(_conf, 1))
                    _runner_ups.append(_runner)
                    _margin_pcts.append(round(_margin, 1))
                    _vote_summaries.append(_vote_str)

                _epoch_len_samples = int(round(30.0 * _fs_marker))
                _pos_starts = np.array([1 + _ep * _epoch_len_samples for _ep in range(_n_epochs)], dtype=np.int64)
                _pos_mids = _pos_starts + (_epoch_len_samples / 2.0)

                _override_sources = [""] * _n_epochs
                _override_records = []

                _lo_label = lightsoff_dropdown.value
                if _lo_label and _lo_label != "(None — Start from Sample 1)" and not df_session_vmrk.empty:
                    _lo_matches = df_session_vmrk[df_session_vmrk["Description"] == _lo_label]
                    if not _lo_matches.empty:
                        _lo_pos = float(_lo_matches.iloc[0]["Position"])
                        for _ep in range(_n_epochs):
                            if _pos_mids[_ep] < _lo_pos:
                                _override_sources[_ep] = f"LightsOff (< {_lo_label})"
                        _override_records.append({"t0_sec": 0.0, "t1_sec": max(0.0, (_lo_pos - 1.0) / _fs_marker)})

                _starts_vals = list(ignore_start_dropdowns.value or [])
                _stops_vals = list(ignore_stop_dropdowns.value or [])
                for _p_i, (_st_lbl, _sp_lbl) in enumerate(zip(_starts_vals, _stops_vals), 1):
                    for _s_pos, _e_pos in _match_marker_intervals(df_session_vmrk, _st_lbl, _sp_lbl):
                        for _ep in range(_n_epochs):
                            if _s_pos < _pos_mids[_ep] < _e_pos:
                                _tag = f"Ignore Pair #{_p_i} ({_st_lbl} → {_sp_lbl})"
                                _override_sources[_ep] = _tag if not _override_sources[_ep] else f"{_override_sources[_ep]}; {_tag}"
                        _override_records.append({
                            "t0_sec": max(0.0, (_s_pos - 1.0) / _fs_marker),
                            "t1_sec": max(0.0, (_e_pos - 1.0) / _fs_marker),
                        })

                _wake_override_df = pd.DataFrame(_override_records)
                _auto_stages = ["W" if _override_sources[_ep] else _raw_yasa_stages[_ep] for _ep in range(_n_epochs)]

                _base_df = pd.DataFrame({
                    "Epoch": np.arange(_n_epochs, dtype=int),
                    "Epoch_Num": np.arange(1, _n_epochs + 1, dtype=int),
                    "Time_Start": [format_hms(_ep * 30) for _ep in range(_n_epochs)],
                    "Time_End": [format_hms((_ep + 1) * 30) for _ep in range(_n_epochs)],
                    "Position_Sample": _pos_starts,
                    "Length_Samples": _epoch_len_samples,
                    "Raw_YASA_Stage": _raw_yasa_stages,
                    "Auto_Stage": _auto_stages,
                    "Override_Source": _override_sources,
                    "Confidence_Pct": _confidence_pcts,
                    "Agreement_Pct": _agreement_pcts,
                    "RunnerUp_Stage": _runner_ups,
                    "Margin_Pct": _margin_pcts,
                    "Prob_W": np.round(_avg_proba[:, 0] * 100.0, 1),
                    "Prob_N1": np.round(_avg_proba[:, 1] * 100.0, 1),
                    "Prob_N2": np.round(_avg_proba[:, 2] * 100.0, 1),
                    "Prob_N3": np.round(_avg_proba[:, 3] * 100.0, 1),
                    "Prob_REM": np.round(_avg_proba[:, 4] * 100.0, 1),
                    "Pass_Votes": _vote_summaries,
                })

                # Pre-filter waveforms ONCE across the whole night & cache float32 µV array for <1 ms epoch slicing
                _eeg_picks = list(mne.pick_types(_raw_inspector.info, eeg=True))
                _eog_picks = list(mne.pick_types(_raw_inspector.info, eog=True))[:1]
                _emg_picks = list(mne.pick_types(_raw_inspector.info, emg=True))[:1]
                _disp_picks = _eeg_picks + _eog_picks + _emg_picks
                if not _disp_picks:
                    _disp_picks = list(range(min(3, len(_raw_inspector.ch_names))))
                _display_chs = [_raw_inspector.ch_names[i] for i in _disp_picks]

                _disp_raw = _raw_inspector.copy().pick(_display_chs)
                _nyq_nat = float(_disp_raw.info["sfreq"]) / 2.0
                _disp_eeg_eog = mne.pick_types(_disp_raw.info, eeg=True, eog=True)
                if len(_disp_eeg_eog) > 0:
                    _disp_raw.filter(l_freq=0.3, h_freq=min(35.0, _nyq_nat - 1.0), picks=_disp_eeg_eog, verbose="ERROR")
                _disp_emg = mne.pick_types(_disp_raw.info, emg=True)
                if len(_disp_emg) > 0 and _nyq_nat > 12.0:
                    _disp_raw.filter(l_freq=10.0, h_freq=min(70.0, _nyq_nat - 1.0), picks=_disp_emg, verbose="ERROR")

                _target_disp_sf = 200.0
                if abs(float(_disp_raw.info["sfreq"]) - _target_disp_sf) > 1e-3:
                    _disp_raw.resample(_target_disp_sf, npad="auto", verbose="ERROR")

                _disp_sfreq = float(_disp_raw.info["sfreq"])
                _ep_samps = int(round(30.0 * _disp_sfreq))
                _wave_uv = (_disp_raw.get_data(picks=_display_chs) * 1e6).astype(np.float32)
                _disp_types = _disp_raw.get_channel_types(picks=_display_chs)

                _ch_ylims = []
                for _ci, _ctype in enumerate(_disp_types):
                    _p99 = float(np.percentile(np.abs(_wave_uv[_ci]), 99)) if _wave_uv.shape[1] > 0 else 50.0
                    if _ctype == "emg":
                        _ch_ylims.append(max(15.0, min(150.0, _p99 * 1.3)))
                    else:
                        _ch_ylims.append(max(75.0, min(220.0, _p99 * 1.3)))

                # Freeze complete scoring snapshot so changing top UI controls never alters downstream views until re-run
                set_scored_bundle({
                    "base_epochs_df": _base_df,
                    "wake_override_intervals_df": _wake_override_df,
                    "wave_uv": _wave_uv,
                    "disp_sfreq": _disp_sfreq,
                    "ep_samps": _ep_samps,
                    "display_chs": _display_chs,
                    "display_types": _disp_types,
                    "ch_ylims": _ch_ylims,
                    "active_eegs_used": _active_eegs,
                    "committed_thresh": float(confidence_slider.value),
                    "committed_excl_wake": bool(exclude_forced_wake_checkbox.value),
                    "committed_participant": str(current_participant),
                    "committed_night": str(current_night),
                    "committed_fs": float(_fs_marker),
                    "committed_vmrk": df_session_vmrk.copy(),
                    "committed_meas_date": session_meas_date,
                    "committed_stem": session_vhdr_path.stem,
                })
                set_active_epoch_state(0)
                set_selected_epochs([])
            except Exception as _exc:
                _scoring_error_msg = str(_exc)

        if _scoring_error_msg is not None:
            _scoring_cell_out = mo.callout(mo.md(f"❌ **Scoring Error:** `{_scoring_error_msg}`"), kind="danger")
            mo.output.replace(_scoring_cell_out)
        else:
            # Explicitly replace the loading callout so it never lingers on screen
            _scoring_cell_out = mo.md("")
            mo.output.replace(_scoring_cell_out)

    _scoring_cell_out
    return


@app.cell
def _(
    get_active_epoch_state,
    get_manual_edits,
    get_scored_bundle,
    get_selected_epochs,
    get_show_all_epochs,
    mo,
    pd,
    set_active_epoch_state,
    set_manual_edits,
    set_selected_epochs,
    set_show_all_epochs,
):
    scored_bundle = get_scored_bundle()

    _RK_STAGE_MAP = {
        "W": "W",
        "N1": "S1",
        "N2": "S2",
        "N3": "S3",
        "S4": "S4",
        "REM": "REM",
        "ART": "MT",
        "UNS": "?",
    }
    _AASM_STAGE_MAP = {"S4": "N3"}

    if scored_bundle is None or scored_bundle["base_epochs_df"].empty:
        scored_epochs_df = pd.DataFrame()
        _session_key = ""
        _n_ep = 1
        _flagged_indices = []
        _session_edits = {}
    else:
        _session_key = f"{scored_bundle['committed_participant']}__{scored_bundle['committed_night']}"
        _all_edits = get_manual_edits()
        _session_edits = dict(_all_edits.get(_session_key, {}))

        _df = scored_bundle["base_epochs_df"].copy()
        _thresh = float(scored_bundle["committed_thresh"])
        _excl_wake = bool(scored_bundle["committed_excl_wake"])
        _auto_list = _df["Auto_Stage"].tolist()
        _n_ep = len(_df)

        _flag_reasons = []
        _is_flagged = []
        for _i in range(_n_ep):
            _row = _df.iloc[_i]
            _is_forced = bool(_row["Override_Source"])
            _low_conf = float(_row["Confidence_Pct"]) < _thresh
            _reasons = []

            if _is_forced:
                _reasons.append(f"Forced Wake ({_row['Override_Source']})")
            else:
                if float(_row["Agreement_Pct"]) < 80.0:
                    _reasons.append(f"Low Agreement ({_row['Agreement_Pct']:.0f}%)")
                _prev_st = _auto_list[_i - 1] if _i > 0 else _auto_list[_i]
                _next_st = _auto_list[_i + 1] if _i < _n_ep - 1 else _auto_list[_i]
                if _prev_st != _auto_list[_i] or _next_st != _auto_list[_i]:
                    _reasons.append(f"Transition ({_prev_st}→{_auto_list[_i]}→{_next_st})")
                if float(_row["Margin_Pct"]) < 15.0:
                    _reasons.append(f"Narrow Margin ({_row['Auto_Stage']} vs {_row['RunnerUp_Stage']}: Δ{_row['Margin_Pct']:.1f}%)")
                if (_row["Auto_Stage"] == "N1" or _row["RunnerUp_Stage"] == "N1") and _low_conf:
                    _reasons.append("N1 Uncertainty")
                if _low_conf and not _reasons:
                    _reasons.append(f"Low Prob ({_row['Confidence_Pct']:.1f}%)")

            _flagged = _low_conf and not (_excl_wake and _is_forced)
            _is_flagged.append(_flagged)
            _flag_reasons.append("; ".join(_reasons) if _reasons else "High Confidence")

        _df["Is_Flagged"] = _is_flagged
        _df["Flag_Reason"] = _flag_reasons

        _final_stages = []
        _is_manual = []
        _status_labels = []
        for _i in range(_n_ep):
            _auto_st = str(_df.iloc[_i]["Auto_Stage"])
            _ov = _session_edits.get(_i)
            if _ov is not None:
                _final_stages.append(_ov)
                _is_manual.append(True)
                _status_labels.append(f"Auto: {_auto_st} → Manual: {_ov}" if _ov != _auto_st else f"Manual: {_ov}")
            elif _df.iloc[_i]["Override_Source"]:
                _final_stages.append(_auto_st)
                _is_manual.append(False)
                _status_labels.append("Wake Override")
            elif _df.iloc[_i]["Is_Flagged"]:
                _final_stages.append(_auto_st)
                _is_manual.append(False)
                _status_labels.append("Flagged")
            else:
                _final_stages.append(_auto_st)
                _is_manual.append(False)
                _status_labels.append("Auto")

        _df["Final_Stage"] = _final_stages
        _df["Final_Stage_AASM"] = [_AASM_STAGE_MAP.get(_s, _s) for _s in _final_stages]
        _df["Final_Stage_RK"] = [_RK_STAGE_MAP.get(_s, _s) for _s in _final_stages]
        _df["Is_Manual_Edit"] = _is_manual
        _df["Status"] = _status_labels
        scored_epochs_df = _df
        _flagged_indices = scored_epochs_df.index[scored_epochs_df["Is_Flagged"]].tolist()

    # Exported navigation & instant stage-assignment widgets (NEVER invoke get_active_epoch_state/get_selected_epochs at eval time)
    _max_idx = max(0, _n_ep - 1)

    def _jump_single(new_idx, _mx=_max_idx):
        _clamped = max(0, min(int(new_idx), _mx))
        set_active_epoch_state(_clamped)
        set_selected_epochs([_clamped])

    def _on_prev_flag(_, _fl=_flagged_indices):
        if _fl:
            _cur = int(get_active_epoch_state())
            _tgt = max((i for i in _fl if i < _cur), default=_fl[-1])
            _jump_single(_tgt)

    def _on_next_flag(_, _fl=_flagged_indices):
        if _fl:
            _cur = int(get_active_epoch_state())
            _tgt = min((i for i in _fl if i > _cur), default=_fl[0])
            _jump_single(_tgt)

    def _on_prev_ep(_):
        _jump_single(int(get_active_epoch_state()) - 1)

    def _on_next_ep(_):
        _jump_single(int(get_active_epoch_state()) + 1)

    def _resolve_target_epochs(_mx=_max_idx):
        _sel = [max(0, min(int(i), _mx)) for i in (get_selected_epochs() or [])]
        if not _sel:
            _sel = [max(0, min(int(get_active_epoch_state()), _mx))]
        return sorted(set(_sel))

    def _set_selected_stage(new_stage, _sk=_session_key):
        if not _sk:
            return
        _targets = _resolve_target_epochs()
        if not _targets:
            return
        def _mut(prev):
            _copy = {k: dict(v) for k, v in prev.items()}
            _s_dict = _copy.setdefault(_sk, {})
            for _ep in _targets:
                _s_dict[_ep] = str(new_stage)
            return _copy
        set_manual_edits(_mut)
        set_selected_epochs([])

    def _revert_selected_stage(_, _sk=_session_key):
        if not _sk:
            return
        _targets = _resolve_target_epochs()
        if not _targets:
            return
        def _mut(prev):
            _copy = {k: dict(v) for k, v in prev.items()}
            if _sk in _copy:
                for _ep in _targets:
                    _copy[_sk].pop(_ep, None)
            return _copy
        set_manual_edits(_mut)
        set_selected_epochs([])

    def _clear_all_session_edits(_, _sk=_session_key):
        if not _sk:
            return
        def _mut(prev):
            _copy = {k: dict(v) for k, v in prev.items()}
            _copy.pop(_sk, None)
            return _copy
        set_manual_edits(_mut)
        set_selected_epochs([])

    prev_flag_btn = mo.ui.button(label="⏮️\u00a0Prev\u00a0Flag", kind="warn", disabled=(len(_flagged_indices) == 0), on_click=_on_prev_flag)
    next_flag_btn = mo.ui.button(label="⏭️\u00a0Next\u00a0Flag", kind="warn", disabled=(len(_flagged_indices) == 0), on_click=_on_next_flag)
    prev_ep_btn = mo.ui.button(label="◀\u00a0Prev\u00a0Ep", on_click=_on_prev_ep)
    next_ep_btn = mo.ui.button(label="▶\u00a0Next\u00a0Ep", on_click=_on_next_ep)
    unselect_btn = mo.ui.button(
        label="✖\u00a0Unselect",
        disabled=(len(get_selected_epochs() or []) == 0),
        on_click=lambda _: set_selected_epochs([]),
    )

    btn_set_w = mo.ui.button(label="W", on_click=lambda _: _set_selected_stage("W"))
    btn_set_n1 = mo.ui.button(label="N1", on_click=lambda _: _set_selected_stage("N1"))
    btn_set_n2 = mo.ui.button(label="N2", on_click=lambda _: _set_selected_stage("N2"))
    btn_set_n3 = mo.ui.button(label="N3", on_click=lambda _: _set_selected_stage("N3"))
    btn_set_s4 = mo.ui.button(label="S4", on_click=lambda _: _set_selected_stage("S4"))
    btn_set_rem = mo.ui.button(label="REM", on_click=lambda _: _set_selected_stage("REM"))
    btn_set_art = mo.ui.button(label="ART", on_click=lambda _: _set_selected_stage("ART"))

    revert_ep_btn = mo.ui.button(label="↩️\u00a0Revert", on_click=_revert_selected_stage)
    clear_edits_btn = mo.ui.button(label="🗑️\u00a0Clear\u00a0All", kind="danger", disabled=(len(_session_edits) == 0), on_click=_clear_all_session_edits)

    def _on_toggle_show_all(val, _fl_set=set(_flagged_indices)):
        _b = bool(val)
        set_show_all_epochs(_b)
        if not _b:
            _cur = [int(i) for i in (get_selected_epochs() or []) if int(i) in _fl_set]
            set_selected_epochs(_cur)

    show_all_epochs_checkbox = mo.ui.checkbox(
        value=bool(get_show_all_epochs()),
        label="\u00a0Show\u00a0All\u00a0Epochs",
        on_change=_on_toggle_show_all,
    )
    return (
        btn_set_art,
        btn_set_n1,
        btn_set_n2,
        btn_set_n3,
        btn_set_rem,
        btn_set_s4,
        btn_set_w,
        clear_edits_btn,
        next_ep_btn,
        next_flag_btn,
        prev_ep_btn,
        prev_flag_btn,
        revert_ep_btn,
        scored_bundle,
        scored_epochs_df,
        show_all_epochs_checkbox,
        unselect_btn,
    )


@app.cell
def _(
    btn_set_art,
    btn_set_n1,
    btn_set_n2,
    btn_set_n3,
    btn_set_rem,
    btn_set_s4,
    btn_set_w,
    clear_edits_btn,
    mo,
    next_ep_btn,
    next_flag_btn,
    prev_ep_btn,
    prev_flag_btn,
    revert_ep_btn,
    scored_bundle,
    scored_epochs_df,
    show_all_epochs_checkbox,
    unselect_btn,
):
    mo.stop(scored_epochs_df.empty or scored_bundle is None)

    mo.hstack(
        [
            prev_flag_btn,
            next_flag_btn,
            prev_ep_btn,
            next_ep_btn,
            unselect_btn,
            mo.md("| **Set Stage:**"),
            btn_set_w,
            btn_set_n1,
            btn_set_n2,
            btn_set_n3,
            btn_set_s4,
            btn_set_rem,
            btn_set_art,
            revert_ep_btn,
            clear_edits_btn,
            mo.md("|"),
            show_all_epochs_checkbox,
        ],
        justify="start",
        gap=0.55,
        align="center",
        wrap=True,
    )
    return


@app.cell
def _(
    datetime,
    get_active_epoch_state,
    get_selected_epochs,
    get_show_all_epochs,
    go,
    mo,
    np,
    scored_bundle,
    scored_epochs_df,
    set_active_epoch_state,
    set_selected_epochs,
    zoom_view_state,
):
    if scored_epochs_df.empty or scored_bundle is None:
        hypno_plot = mo.md("")
    else:
        _n_ep = len(scored_epochs_df)
        _max_ep_idx = _n_ep - 1
        _thresh = float(scored_bundle["committed_thresh"])
        _meas_date = scored_bundle["committed_meas_date"]
        _wake_ov_df = scored_bundle["wake_override_intervals_df"]
        _part = scored_bundle["committed_participant"]
        _night = scored_bundle["committed_night"]
        _ui_rev = f"{_part}_{_night}_{_thresh}"

        _stage_y_map = {"N3": 0.0, "S4": 0.0, "N2": 1.0, "N1": 2.0, "REM": 3.0, "W": 4.0, "ART": 4.2, "UNS": 4.2}
        _ep_nums = scored_epochs_df["Epoch_Num"].to_numpy(dtype=int)
        _stage_y = np.array([_stage_y_map.get(s, 4.0) for s in scored_epochs_df["Final_Stage"]], dtype=float)
        _conf_vals = scored_epochs_df["Confidence_Pct"].to_numpy(dtype=float)
        _conf_scaled = _conf_vals / 25.0  # Maps 0-100% cleanly onto 0.0-4.0
        _flag_mask = scored_epochs_df["Is_Flagged"].to_numpy(dtype=bool)
        _edit_mask = scored_epochs_df["Is_Manual_Edit"].to_numpy(dtype=bool)
        _sel_set = set(int(i) for i in (get_selected_epochs() or []))
        _show_all = bool(get_show_all_epochs())
        _vis_mask = np.ones(_n_ep, dtype=bool) if _show_all else _flag_mask

        _hover_texts = [
            (
                f"<b>Ep #{int(_r['Epoch_Num'])}</b> ({_r['Time_Start']}–{_r['Time_End']})<br>"
                f"Stage: <b>{_r['Final_Stage']}</b> ({_r['Status']})<br>"
                f"Confidence: <b>{_r['Confidence_Pct']:.1f}%</b> | Agree: {_r['Agreement_Pct']:.0f}%<br>"
                f"Note: {_r['Flag_Reason']}"
            )
            for _, _r in scored_epochs_df.iterrows()
        ]

        _fig_hyp = go.Figure()

        # 1. Hypnogram step trace (append N+1 endpoint so final epoch draws full step width)
        _step_x = np.append(_ep_nums, _ep_nums[-1] + 1)
        _step_y = np.append(_stage_y, _stage_y[-1])
        _fig_hyp.add_trace(
            go.Scatter(
                x=_step_x,
                y=_step_y,
                mode="lines",
                line=dict(shape="hv", color="#1b4f72", width=2.2),
                name="Sleep Stage",
                hoverinfo="skip",
            )
        )

        # 2. Confidence (%) trace scaled to [0, 4] + threshold line
        _fig_hyp.add_trace(
            go.Scatter(
                x=_ep_nums,
                y=_conf_scaled,
                mode="lines",
                line=dict(color="rgba(44, 62, 80, 0.35)", width=1.1),
                name="Confidence (%)",
                hoverinfo="skip",
            )
        )
        _fig_hyp.add_hline(
            y=_thresh / 25.0,
            line_dash="dot",
            line_color="rgba(231, 76, 60, 0.30)",
            line_width=0.85,
            annotation_text=f"Thresh {int(_thresh)}%",
            annotation_position="bottom right",
            annotation_font=dict(size=9.5, color="rgba(231, 76, 60, 0.50)"),
        )

        # 3. Interactive epoch markers (Flagged only, or all YASA-scored epochs when Show All Epochs is checked)
        if np.any(_vis_mask):
            _vis_orig_idxs = np.where(_vis_mask)[0]
            _vis_eps = _ep_nums[_vis_mask]
            _vis_y = _stage_y[_vis_mask]
            _vis_hover = [h for h, v in zip(_hover_texts, _vis_mask) if v]
            _vis_edit_flags = _edit_mask[_vis_mask]
            _vis_flag_flags = _flag_mask[_vis_mask]
            _vis_dot_colors = [
                "#27ae60" if bool(_ed) else ("#f1c40f" if int(_orig_i) in _sel_set else ("#e74c3c" if bool(_fl) else "#2980b9"))
                for _orig_i, _ed, _fl in zip(_vis_orig_idxs, _vis_edit_flags, _vis_flag_flags)
            ]
            _vis_dot_sizes = [
                12 if int(_orig_i) in _sel_set else (8 if (bool(_fl) or bool(_ed)) else 6.5)
                for _orig_i, _ed, _fl in zip(_vis_orig_idxs, _vis_edit_flags, _vis_flag_flags)
            ]
            _vis_text_colors = [
                "#1e8449" if bool(_ed) else ("#c0392b" if bool(_fl) else "#1b4f72")
                for _ed, _fl in zip(_vis_edit_flags, _vis_flag_flags)
            ]

            _pos_cycle = ["bottom center", "top center", "bottom right"]
            _last_pos_ep = [-10**9, -10**9, -10**9]
            _min_slot_gap = max(2, _n_ep // 150)
            _vis_texts = []
            _vis_textpos = []
            _lbl_idx = 0
            for _ep_val, _ed, _fl in zip(_vis_eps, _vis_edit_flags, _vis_flag_flags):
                _slot = _lbl_idx % 3
                if (bool(_fl) or bool(_ed)) and (int(_ep_val) - _last_pos_ep[_slot]) >= _min_slot_gap:
                    _vis_texts.append(str(int(_ep_val)))
                    _last_pos_ep[_slot] = int(_ep_val)
                    _lbl_idx += 1
                else:
                    _vis_texts.append("")
                _vis_textpos.append(_pos_cycle[_slot])

            _fig_hyp.add_trace(
                go.Scatter(
                    x=_vis_eps,
                    y=_vis_y,
                    customdata=_vis_eps,
                    mode="markers+text",
                    marker=dict(size=_vis_dot_sizes, color=_vis_dot_colors),
                    text=_vis_texts,
                    textposition=_vis_textpos,
                    textfont=dict(size=9, color=_vis_text_colors),
                    name=("All Epochs" if _show_all else "Flagged (< Thresh)"),
                    hovertext=_vis_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

        # 4. Highlight ring trace around all currently selected epoch(s)
        _sel_valid = sorted(i for i in _sel_set if 0 <= i < _n_ep)
        _sel_x = [int(_ep_nums[i]) for i in _sel_valid]
        _sel_y = [float(_stage_y[i]) for i in _sel_valid]
        _sel_ring_colors = ["#27ae60" if bool(_edit_mask[i]) else "#f39c12" for i in _sel_valid]
        _sel_ring_line_colors = ["#1e8449" if bool(_edit_mask[i]) else "#d35400" for i in _sel_valid]
        _fig_hyp.add_trace(
            go.Scatter(
                x=_sel_x,
                y=_sel_y,
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=15,
                    color=_sel_ring_colors,
                    line=dict(width=2.8, color=_sel_ring_line_colors),
                ),
                name="Selected",
                hoverinfo="skip",
                showlegend=bool(_sel_valid),
            )
        )

        # 5. Wake Override shaded regions (soft yellow/amber boxes)
        if not _wake_ov_df.empty:
            for _, _wr in _wake_ov_df.iterrows():
                _x0 = (float(_wr["t0_sec"]) / 30.0) + 1.0
                _x1 = (float(_wr["t1_sec"]) / 30.0) + 1.0
                if _x1 > _x0:
                    _fig_hyp.add_vrect(
                        x0=_x0,
                        x1=_x1,
                        fillcolor="rgba(243, 156, 18, 0.16)",
                        line_width=0,
                        layer="below",
                    )

        # Clean X-axis wall-clock/elapsed time tick labels
        _tick_step = max(1, _n_ep // 12)
        _tick_vals = sorted(set([1] + list(range(_tick_step, _n_ep + 1, _tick_step))))
        _tick_texts = []
        for _tv in _tick_vals:
            if _meas_date is not None:
                _dt = _meas_date + datetime.timedelta(seconds=(_tv - 1) * 30)
                _t_lbl = _dt.strftime("%H:%M")
            else:
                _tot_min = ((_tv - 1) * 30) // 60
                _t_lbl = f"{_tot_min // 60:02d}:{_tot_min % 60:02d}"
            _tick_texts.append(_t_lbl)

        _has_s4_all = "S4" in scored_epochs_df["Final_Stage"].values
        _zv_cur = zoom_view_state.get(_ui_rev, {})
        _active_x_range = _zv_cur.get("x_range") or [0.5, _n_ep + 1.5]
        _active_dragmode = _zv_cur.get("dragmode") or "pan"
        _fig_hyp.update_layout(
            title=dict(text=f"Interactive Hypnogram — {_part} ({_night}) [Scroll to Zoom · Drag to Pan · Click or Box-Select Flagged Epochs]", x=0.01, font=dict(size=14)),
            height=410,
            margin=dict(l=65, r=30, t=48, b=52),
            template="plotly_white",
            hovermode="closest",
            dragmode=_active_dragmode,
            clickmode="event+select",
            uirevision=_ui_rev,
            selectionrevision=f"{_ui_rev}_{_show_all}_{_sel_valid}_{int(np.sum(_edit_mask))}",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0, font=dict(size=11)),
            xaxis=dict(
                tickmode="array",
                tickvals=_tick_vals,
                ticktext=_tick_texts,
                range=_active_x_range,
                uirevision=_ui_rev,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=[0.0, 1.0, 2.0, 3.0, 4.0],
                ticktext=[("N3/S4" if _has_s4_all else "N3"), "N2", "N1", "REM", "W (100%)"],
                range=[-0.45, 4.35],
                fixedrange=True,
                uirevision=_ui_rev,
                showgrid=False,
                zeroline=False,
            ),
        )

        _selectable_set = set(range(_n_ep)) if _show_all else set(int(i) for i in np.where(_flag_mask)[0])

        def _on_plot_select(pts, _mx=_max_ep_idx, _fset=_selectable_set):
            _sel_data = getattr(hypno_plot, "_selection_data", None)
            if isinstance(_sel_data, dict):
                _zv = zoom_view_state.setdefault(_ui_rev, {})
                if not _sel_data:
                    zoom_view_state.pop(_ui_rev, None)
                else:
                    if "dragmode" in _sel_data and _sel_data["dragmode"]:
                        _zv["dragmode"] = str(_sel_data["dragmode"])
                    _x_info = _sel_data.get("xaxis")
                    if isinstance(_x_info, dict):
                        if _x_info.get("autorange"):
                            _zv.pop("x_range", None)
                        else:
                            _xr = _x_info.get("range")
                            if isinstance(_xr, (list, tuple)) and len(_xr) == 2 and all(isinstance(v, (int, float)) for v in _xr):
                                _zv["x_range"] = [float(_xr[0]), float(_xr[1])]
            if isinstance(_sel_data, dict) and "points" not in _sel_data:
                return
            if not pts:
                return
            _idxs = []
            for _p in pts:
                if not isinstance(_p, dict) or _p.get("name") == "Selected":
                    continue
                _raw_x = None
                for _key in ("x", "Epoch #", "Epoch", "customdata"):
                    if _key in _p and _p[_key] is not None:
                        _cand = _p[_key]
                        if isinstance(_cand, (list, tuple)) and _cand:
                            _cand = _cand[0]
                        if isinstance(_cand, (int, float)):
                            _raw_x = float(_cand)
                            break
                if _raw_x is not None:
                    _ep0 = max(0, min(int(round(_raw_x)) - 1, _mx))
                    if _ep0 in _fset:
                        _idxs.append(_ep0)
            _unique_idxs = sorted(set(_idxs))
            if not _unique_idxs:
                return
            _is_box_or_lasso = isinstance(_sel_data, dict) and bool(_sel_data.get("range") or _sel_data.get("lasso"))
            _cur_set = set(int(i) for i in (get_selected_epochs() or []) if 0 <= int(i) <= _mx)
            if not _is_box_or_lasso and len(_unique_idxs) == 1:
                _clicked_ep = _unique_idxs[0]
                if _clicked_ep in _cur_set:
                    _new_sel = sorted(_cur_set - {_clicked_ep})
                    set_selected_epochs(_new_sel)
                    if _new_sel and int(get_active_epoch_state()) == _clicked_ep:
                        set_active_epoch_state(_new_sel[-1])
                else:
                    _new_sel = sorted(_cur_set | {_clicked_ep})
                    set_selected_epochs(_new_sel)
                    set_active_epoch_state(_clicked_ep)
                return
            _new_sel = sorted(_cur_set | set(_unique_idxs))
            set_selected_epochs(_new_sel)
            set_active_epoch_state(_unique_idxs[0])

        hypno_plot = mo.ui.plotly(
            _fig_hyp,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["select2d", "lasso2d"],
            },
            on_change=_on_plot_select,
        )

    hypno_plot
    return


@app.cell
def _(
    format_hms,
    get_active_epoch_state,
    get_selected_epochs,
    html_lib,
    mo,
    np,
    plt,
    scored_bundle,
    scored_epochs_df,
    ticker,
):
    if scored_epochs_df.empty or scored_bundle is None:
        _inspector_ui = mo.md("")
    else:
        _part = scored_bundle["committed_participant"]
        _night = scored_bundle["committed_night"]
        _thresh = float(scored_bundle["committed_thresh"])
        _active_eegs = scored_bundle["active_eegs_used"]
        _wave_uv = scored_bundle["wave_uv"]
        _disp_sfreq = float(scored_bundle["disp_sfreq"])
        _ep_samps = int(scored_bundle["ep_samps"])
        _display_chs = scored_bundle["display_chs"]
        _display_types = scored_bundle["display_types"]
        _ch_ylims = scored_bundle["ch_ylims"]

        _n_ep = len(scored_epochs_df)
        _active_idx = max(0, min(int(get_active_epoch_state()), _n_ep - 1))
        _raw_sel = get_selected_epochs() or []
        _sel_list = sorted(set(max(0, min(int(i), _n_ep - 1)) for i in _raw_sel))
        _ep_row = scored_epochs_df.iloc[_active_idx]

        _n_flagged = int(scored_epochs_df["Is_Flagged"].sum())
        _n_edited = int(scored_epochs_df["Is_Manual_Edit"].sum())
        _mean_conf = float(scored_epochs_df["Confidence_Pct"].mean())
        _mean_agr = float(scored_epochs_df["Agreement_Pct"].mean())
        _ch_label_str = ", ".join(_active_eegs)

        def _pill(label, value, bg="rgba(127, 140, 141, 0.14)", border="rgba(127, 140, 141, 0.35)", bold=False):
            _esc_l = html_lib.escape(str(label))
            _esc_v = html_lib.escape(str(value))
            _fw = "700" if bold else "600"
            return (
                f"<span style='display:inline-flex;align-items:center;gap:4px;padding:4px 10px;"
                f"border-radius:999px;font-size:12.5px;line-height:1.25;background:{bg};"
                f"border:1px solid {border};'>"
                f"<span style='opacity:0.78;'>{_esc_l}</span>"
                f"<strong style='font-weight:{_fw};'>{_esc_v}</strong></span>"
            )

        _flag_bg = "rgba(231, 76, 60, 0.15)" if _n_flagged > 0 else "rgba(39, 174, 96, 0.14)"
        _flag_bd = "rgba(231, 76, 60, 0.45)" if _n_flagged > 0 else "rgba(39, 174, 96, 0.40)"
        _row1_html = "".join([
            _pill("Session:", f"{_part} ({_night})"),
            _pill("Epochs:", f"{_n_ep} ({format_hms(_n_ep * 30)})"),
            _pill("EEG:", _ch_label_str),
            _pill("Mean Conf:", f"{_mean_conf:.1f}%"),
            _pill("Agreement:", f"{_mean_agr:.1f}%"),
            _pill(f"Flagged (<{int(_thresh)}%):", str(_n_flagged), bg=_flag_bg, border=_flag_bd),
            _pill("Manual Edits:", str(_n_edited), bg="rgba(46, 134, 193, 0.14)", border="rgba(46, 134, 193, 0.40)"),
        ])

        _ep_num = int(_ep_row["Epoch_Num"])
        _ep_conf = float(_ep_row["Confidence_Pct"])
        _conf_low = _ep_conf < _thresh
        _conf_bg = "rgba(231, 76, 60, 0.16)" if _conf_low else "rgba(39, 174, 96, 0.16)"
        _conf_bd = "rgba(231, 76, 60, 0.48)" if _conf_low else "rgba(39, 174, 96, 0.45)"

        if _ep_row["Is_Manual_Edit"]:
            if _ep_row["Auto_Stage"] != _ep_row["Final_Stage"]:
                _stage_pill = _pill("Stage:", f"Auto: {_ep_row['Auto_Stage']} → Manual: {_ep_row['Final_Stage']}", bg="rgba(46, 134, 193, 0.16)", border="rgba(46, 134, 193, 0.48)")
            else:
                _stage_pill = _pill("Stage:", f"Manual: {_ep_row['Final_Stage']}", bg="rgba(46, 134, 193, 0.16)", border="rgba(46, 134, 193, 0.48)")
        elif _ep_row["Override_Source"]:
            _stage_pill = _pill("Stage:", f"{_ep_row['Final_Stage']} (Wake Override)", bg="rgba(243, 156, 18, 0.16)", border="rgba(243, 156, 18, 0.48)")
        else:
            _stage_pill = _pill("Stage:", f"{_ep_row['Final_Stage']} (Auto)")

        _multi_sel_pill = ""
        if len(_sel_list) > 1:
            _multi_sel_pill = _pill(
                "Selected:",
                f"{len(_sel_list)} Epochs (Ep #{_sel_list[0] + 1}–#{_sel_list[-1] + 1}) — Click a stage button above to apply to all",
                bg="rgba(142, 68, 173, 0.16)",
                border="rgba(142, 68, 173, 0.48)",
                bold=True,
            )

        _row2_html = "".join([
            _pill(
                "Inspecting",
                f"Ep #{_ep_num} ({_ep_row['Time_Start']}–{_ep_row['Time_End']})",
                bg="rgba(233, 30, 99, 0.16)",
                border="rgba(233, 30, 99, 0.50)",
                bold=True,
            ),
            _multi_sel_pill,
            _stage_pill,
            _pill("Confidence:", f"{_ep_conf:.1f}%", bg=_conf_bg, border=_conf_bd),
            _pill("Diagnostic:", str(_ep_row["Flag_Reason"])),
        ])

        _summary_box = mo.Html(
            "<div style='display:flex;flex-direction:column;gap:10px;padding:12px 14px;"
            "border-radius:10px;border:1px solid rgba(127,140,141,0.32);background:rgba(127,140,141,0.06);'>"
            f"<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:center;'>{_row1_html}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:center;'>{_row2_html}</div>"
            "</div>"
        )

        # Ultra-fast <1 ms waveform slice from pre-filtered float32 NumPy cache + Stage Probability Bar Chart
        _n_wave = max(1, len(_display_chs))
        _fig_h = max(3.6, 1.45 * _n_wave)
        plt.close("all")
        plt.style.use("default")
        _fig_insp = plt.figure(figsize=(15, _fig_h), facecolor="white")
        _gs = _fig_insp.add_gridspec(_n_wave, 2, width_ratios=[2.35, 1.0], hspace=0.22, wspace=0.18)

        _s0 = _active_idx * _ep_samps
        _s1 = min((_active_idx + 1) * _ep_samps, _wave_uv.shape[1])
        if _wave_uv.shape[1] > 0 and _s1 > _s0 and _display_chs:
            _seg_uv = _wave_uv[:, _s0:_s1]
            _sig_times = np.arange(_seg_uv.shape[1], dtype=np.float32) / _disp_sfreq
            _wave_palette = {"eeg": "#1f77b4", "eog": "#16a085", "emg": "#c0392b"}

            _share_ax = None
            for _wi, _ch_name in enumerate(_display_chs):
                _ax_w = _fig_insp.add_subplot(_gs[_wi, 0], sharex=_share_ax)
                if _share_ax is None:
                    _share_ax = _ax_w
                _ctype = _display_types[_wi]
                _ylim = float(_ch_ylims[_wi])
                _ax_w.plot(_sig_times, _seg_uv[_wi], color=_wave_palette.get(_ctype, "#34495e"), linewidth=0.75)
                _ax_w.set_ylim(-_ylim, _ylim)
                if _ctype == "eeg":
                    _ax_w.axhline(37.5, color="#7f8c8d", linestyle="--", linewidth=0.7, alpha=0.65)
                    _ax_w.axhline(-37.5, color="#7f8c8d", linestyle="--", linewidth=0.7, alpha=0.65)
                _ax_w.axhline(0.0, color="#bdc3c7", linestyle="-", linewidth=0.5, alpha=0.6)
                _ax_w.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, symmetric=True))
                _ax_w.set_ylabel(f"{_ch_name}\n(µV)", fontsize=9, fontweight="bold")
                _ax_w.tick_params(axis="y", labelsize=8)
                _ax_w.set_xlim(0.0, 30.0)
                _ax_w.grid(axis="x", linestyle=":", alpha=0.45)

                if _wi == 0:
                    _ax_w.set_title(f"30s Pre-Filtered Waveforms — Ep #{_ep_num} ({_ep_row['Time_Start']}–{_ep_row['Time_End']})", fontsize=10.5, fontweight="bold")
                if _wi < len(_display_chs) - 1:
                    plt.setp(_ax_w.get_xticklabels(), visible=False)
                else:
                    _ax_w.set_xlabel("Time Within 30s Epoch (Seconds)", fontsize=9.5, fontweight="bold")
        else:
            _ax_w = _fig_insp.add_subplot(_gs[:, 0])
            _ax_w.text(0.5, 0.5, "Signal data unavailable.", ha="center", va="center")
            _ax_w.axis("off")

        _ax_prob = _fig_insp.add_subplot(_gs[:, 1])
        _st_names = ["W", "N1", "N2", "N3", "REM"]
        _st_cols = ["#f39c12", "#5dade2", "#2e86c1", "#1b4f72", "#8e44ad"]
        _prob_vals = [float(_ep_row[f"Prob_{s}"]) for s in _st_names]
        _bars = _ax_prob.barh(_st_names[::-1], _prob_vals[::-1], color=_st_cols[::-1], edgecolor="#2c3e50", alpha=0.9)
        for _b, _pv in zip(_bars, _prob_vals[::-1]):
            _ax_prob.text(min(96.0, _pv + 1.5), _b.get_y() + _b.get_height() / 2.0, f"{_pv:.1f}%", va="center", fontsize=9, fontweight="bold")
        _ax_prob.axvline(_thresh, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8)
        _ax_prob.set_xlim(0, 108)
        _ax_prob.set_xlabel("Ensemble Probability (%)", fontsize=9.5, fontweight="bold")
        _ax_prob.set_title(
            f"Ep #{_ep_num} Probs | Votes: {_ep_row['Pass_Votes']}\n{_ep_row['Flag_Reason']}",
            fontsize=9.5,
            fontweight="bold",
        )
        _ax_prob.grid(axis="x", linestyle=":", alpha=0.45)

        _fig_insp.subplots_adjust(top=0.89, bottom=0.12, left=0.07, right=0.98)

        _inspector_ui = mo.vstack([
            _fig_insp,
            _summary_box,
        ], gap=0.75)

    _inspector_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <hr style='border: none; height: 3px; background-color: #2c3e50; margin: 50px 0;'>
    """)
    return


@app.cell
def _(mo, scored_bundle, scored_epochs_df):
    mo.stop(scored_epochs_df.empty or scored_bundle is None)

    export_txt_checkbox = mo.ui.checkbox(
        value=False,
        label="📄 Export as 2-Column Numeric .txt (Col 1: W=0, N1=1, N2=2, N3=3, REM=5 | Col 2: 0) instead of BrainVision .Markers",
    )

    mo.vstack([
        mo.md("### 💾 Export Scored Hypnogram"),
        export_txt_checkbox,
    ])
    return (export_txt_checkbox,)


@app.cell
def _(
    default_output_dir,
    export_txt_checkbox,
    mo,
    scored_bundle,
    scored_epochs_df,
):
    mo.stop(scored_epochs_df.empty or scored_bundle is None)

    _is_txt = bool(export_txt_checkbox.value)
    _suggested_name = f"{scored_bundle['committed_stem']}{'.txt' if _is_txt else '.Markers'}"

    export_dir_input = mo.ui.text(
        value=str(default_output_dir),
        label="📂 Output Directory:",
        full_width=True,
    )
    export_filename_input = mo.ui.text(
        value=_suggested_name,
        label="📄 Output Filename (.txt):" if _is_txt else "📄 Output Filename (.Markers):",
        full_width=True,
    )
    notation_radio = mo.ui.radio(
        options=[
            "AASM (W, N1, N2, N3, REM)",
            "R&K (W, S1, S2, S3, S4, REM, MT)",
        ],
        value="AASM (W, N1, N2, N3, REM)",
        label="🏷️ Stage Notation:",
        inline=True,
    )
    include_vmrk_events_checkbox = mo.ui.checkbox(
        value=True,
        label="Include original .vmrk events",
    )
    export_save_btn = mo.ui.run_button(
        label="💾\u00a0Save\u00a0Scored\u00a0File",
        kind="success",
    )

    _row2_widgets = [notation_radio, export_save_btn] if _is_txt else [notation_radio, include_vmrk_events_checkbox, export_save_btn]
    mo.vstack([
        mo.hstack([export_dir_input, export_filename_input], widths="equal"),
        mo.hstack(_row2_widgets, justify="start", gap=1.2, align="center", wrap=True),
    ])
    return (
        export_dir_input,
        export_filename_input,
        export_save_btn,
        include_vmrk_events_checkbox,
        notation_radio,
    )


@app.cell
def _(
    Path,
    export_dir_input,
    export_filename_input,
    export_save_btn,
    export_txt_checkbox,
    include_vmrk_events_checkbox,
    mo,
    notation_radio,
    scored_bundle,
    scored_epochs_df,
):
    mo.stop(scored_epochs_df.empty or scored_bundle is None)

    _fs_out = float(scored_bundle["committed_fs"])
    _vmrk_df = scored_bundle["committed_vmrk"]
    _interval_ms = 1000.0 / _fs_out
    _use_rk = notation_radio.value.startswith("R&K")
    _is_txt = bool(export_txt_checkbox.value)

    if _is_txt:
        _txt_code_map = {
            "W": 0,
            "WAKE": 0,
            "N1": 1,
            "S1": 1,
            "N2": 2,
            "S2": 2,
            "N3": 3,
            "S3": 3,
            "S4": 4 if _use_rk else 3,
            "REM": 5,
            "R": 5,
            "ART": 0,
            "MT": 0,
            "UNS": 0,
            "?": 0,
        }
        _all_lines = [
            f"{_txt_code_map.get(str(_st).strip().upper(), 0)}\t0"
            for _st in scored_epochs_df["Final_Stage"]
        ]
        _export_text = "\n".join(_all_lines) + "\n"
        _n_rows = len(_all_lines)
        _fname = export_filename_input.value.strip() or "scored_sleep.txt"
        if not _fname.lower().endswith(".txt"):
            _fname = f"{_fname}.txt"
    else:
        _stage_col = "Final_Stage_RK" if _use_rk else "Final_Stage_AASM"
        _records = []
        if include_vmrk_events_checkbox.value and not _vmrk_df.empty:
            for _, _vr in _vmrk_df.iterrows():
                _vtype = str(_vr["Type"]).strip().replace(",", ";")
                if "sleep stage" in _vtype.lower():
                    continue
                _vdesc = str(_vr["Description"]).strip().replace(",", ";")
                _vpos = int(round(float(_vr["Position"])))
                _vlen = int(round(float(_vr["Length"]))) if str(_vr["Length"]).strip() else 1
                _vchan = str(_vr["Channel"]).strip().replace(",", ";") or "All"
                _priority = 0 if "new segment" in _vtype.lower() else 1
                _records.append((_vpos, _priority, f"{_vtype}, {_vdesc}, {_vpos}, {_vlen}, {_vchan}"))
        else:
            _records.append((1, 0, "New Segment, , 1, 1, All"))

        for _, _sr in scored_epochs_df.iterrows():
            _s_pos = int(_sr["Position_Sample"])
            _s_len = int(_sr["Length_Samples"])
            _s_lbl = str(_sr[_stage_col]).strip()
            _records.append((_s_pos, 2, f"Sleep Stage, {_s_lbl}, {_s_pos}, {_s_len}, All"))

        _records.sort(key=lambda _x: (_x[0], _x[1]))

        _fs_disp = int(round(_fs_out)) if abs(_fs_out - round(_fs_out)) < 1e-6 else _fs_out
        _header_lines = [
            f"Sampling rate: {_fs_disp} Hz, SamplingInterval: {_interval_ms:g} ms",
            "Type, Description, Position, Length, Channel",
        ]
        _all_lines = _header_lines + [_r[2] for _r in _records]
        _export_text = "\n".join(_all_lines) + "\n"
        _n_rows = len(_records)
        _fname = export_filename_input.value.strip() or "scored_sleep.Markers"
        if not _fname.lower().endswith(".markers"):
            _fname = f"{_fname}.Markers"

    _target_dir = Path(export_dir_input.value.strip() or ".")
    _target_path = _target_dir / _fname

    _save_notice = mo.md("")
    if export_save_btn.value:
        try:
            _target_dir.mkdir(parents=True, exist_ok=True)
            _target_path.write_text(_export_text, encoding="utf-8")
            _save_notice = mo.callout(
                mo.md(f"✅ **Saved `{_n_rows}` rows to:** `{_target_path}`"),
                kind="success",
            )
        except Exception as _write_err:
            _save_notice = mo.callout(
                mo.md(f"❌ **Failed to save `{_target_path}`:** `{_write_err}`"),
                kind="danger",
            )

    _preview_head = "\n".join(_all_lines[:15])
    _preview_tail = "\n".join(_all_lines[-5:]) if len(_all_lines) > 20 else ""
    _preview_str = _preview_head + (f"\n...\n{_preview_tail}" if _preview_tail else "")
    _preview_lang = "text" if _is_txt else "csv"

    _download_btn = mo.download(
        data=_export_text.encode("utf-8"),
        filename=_fname,
        mimetype="text/plain",
        label="⬇️\u00a0Manual\u00a0Download",
    )

    mo.vstack([
        _save_notice,
        mo.hstack([_download_btn, mo.md(f"**Target:** `{_target_path}` (`{_n_rows}` rows)")], justify="start", gap=1.2, align="center"),
        mo.md(f"```{_preview_lang}\n{_preview_str}\n```"),
    ])
    return


if __name__ == "__main__":
    app.run()
