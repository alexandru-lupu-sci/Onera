import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Onera PreAwakening
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview

    **Onera** is an interactive Python pipeline for sleep EEG analysis, designed to investigate pre-awakening brain states, sleep stage transitions, and time-frequency dynamics.
    - Name inspired by ancient greek word Oneira (όνειρα) meaning "Dreams" : )

    **Author:** Alexandru Lupu | **GitHub:** [alexandru-lupu-sci](https://github.com/alexandru-lupu-sci)

    ### Core Capabilities
    - **Whole-Night Hypnograms:** Sleep architecture visualization (supporting both AASM and R&K scoring) with whole-night macro-statistics, experimental event overlays, and real-time clock alignment.
    - **Pre-Awakening Staging:** Chronological stage progression, dominant stage classification, and stimulus-to-awakening latency extraction.
    - **Signal Pre-Processing:** Automated BrainVision loading, physiological channel typing, mastoid/average/custom re-referencing, and $\pm 20\text{ s}$ buffer-padded composite zero-phase FIR bandpass/notch filtering ([MNE-Python](https://mne.tools/)) with in-memory awakening window caching.
    - **Automated Artifact Rejection:** Configurable epoch-level thresholding for flatlines, peak-to-peak voltage excursions, fast voltage steps ($\mu\text{V/ms}$), and high-frequency EMG bursts, with per-channel artifact diagnostics.
    - **Spectral Power (PSD) & Topomaps:** Absolute ($\mu\text{V}^2$) and relative (%) bandpower estimation via Welch's method ([YASA](https://raphaelvallat.com/yasa/)) with 2D scalp topography.
    - **Time-Frequency Spectrograms:** Power-domain averaged Short-Time Fourier Transforms (STFT) with event markers and artifact window overlays.

    ### Expected Data Structure
    - **Any Filename or Folder Layout:** Works with flat directories, nested per-participant folders, spaces, hyphens, dates, or BIDS hierarchies (no rigid `Participant_Condition` filename convention required).
    - **Raw EEG Directory:** Standard BrainVision triplets (`.vhdr`, `.vmrk`, `.eeg`) scanned recursively.
    - **Markers Directory:** Scored BrainVision Analyzer `.Markers` or `.vmrk` files scanned recursively and automatically paired with their matching raw recording (with support for cropped `.Markers` timelines via anchor-marker offset alignment).

    ### UI Navigation
    - Toggle **App View** (`Ctrl + .` or bottom-right icon) to hide code cells or to show them when in App View.
    - Use the **Table of Contents** Click `View outline` (left sidebar)
    - **Performance Tip:** Marimo is reactive and updates downstream cells automatically after a change has been done. After changing multiple settings, click **Stop** (square button) to stop the current update and automatically relaunch it with the latest settings.
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

    Define the local paths to your raw EEG and marker directories. Both folders are scanned recursively for BrainVision recordings (`.vhdr`, `.eeg`, `.vmrk`) and marker files (`.Markers`, `.vmrk`) across any filename or folder structure (flat folders, per-participant subfolders, or BIDS hierarchies). Each scored marker file is automatically paired with its matching raw recording in a single searchable **Recording** selector.
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
            value="K:\\", 
            label="📂 Raw Data Path (.vhdr, .vmrk, .eeg):",
            full_width=True
        ),
        markers_dir=mo.ui.text(
            value="K:\\", 
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
    # Pause execution cleanly on initial startup until the user clicks "Scan & Load Directories"
    mo.stop(
        paths_form.value is None,
        mo.callout(
            mo.md("👆 **Please verify your Raw Data and Markers paths above, then click `Scan & Load Directories` to start.**"),
            kind="info"
        )
    )

    # Immediately show a loading text while scanning the directories
    mo.output.replace(
        mo.callout(
            mo.md("### ⏳ Scanning directories and loading session data...\n*Please wait, this may take a moment.*"), 
            kind="info"
        )
    )

    import os
    os.environ["MNE_USE_NUMBA"] = "false"
    from concurrent.futures import ThreadPoolExecutor
    import mne
    import yasa
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from scipy import signal as scipy_signal

    # Persistent in-memory cache for buffered pre-awakening raw windows (cleared on recording switch)
    raw_segment_cache = {}
    _RE_MK_PREFIX = re.compile(r'^Mk\d+=')
    _RE_MARKER_FS = re.compile(r'Sampling rate:\s*([\d.]+)\s*Hz', re.IGNORECASE)

    # Extract paths from the submitted form (fall back to data_dir if markers_dir is left blank)
    data_dir = Path(paths_form.value["data_dir"].strip())
    _m_str = paths_form.value["markers_dir"].strip()
    markers_dir = Path(_m_str) if _m_str else data_dir

    def natural_sort_key(text):
        """Sort strings naturally with numeric segments ordered numerically (e.g., DS2 before DS10)."""
        return tuple(
            (0, int(_tok), "") if _tok.isdigit() else (1, 0, _tok.lower())
            for _tok in re.split(r"(\d+)", str(text))
            if _tok
        )

    def _norm_fs_path(path_obj):
        try:
            return os.path.normcase(str(Path(path_obj).resolve()))
        except Exception:
            return os.path.normcase(str(path_obj))

    def detect_marker_fs(filepath):
        """Read 'Sampling rate: X Hz' from the header of a .Markers or .vmrk file."""
        if filepath is None or not Path(filepath).is_file():
            return None
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as _fh:
                for _ in range(15):
                    _line = _fh.readline()
                    if not _line:
                        break
                    _m = _RE_MARKER_FS.search(_line)
                    if _m:
                        return int(round(float(_m.group(1))))
        except Exception:
            pass
        return None

    def detect_vhdr_fs(vhdr_path):
        """Read 'SamplingInterval=' (microseconds) from a BrainVision .vhdr file."""
        if vhdr_path is None or not Path(vhdr_path).is_file():
            return None
        try:
            with open(vhdr_path, 'r', encoding='utf-8', errors='ignore') as _fv:
                for _line in _fv:
                    if _line.strip().lower().startswith('samplinginterval='):
                        _interval = float(_line.split('=', 1)[1].strip())
                        if _interval > 0:
                            return int(round(1_000_000.0 / _interval))
        except Exception:
            pass
        return None

    def resolve_raw_vmrk(vhdr_path):
        """Resolve the companion .vmrk marker file for a given .vhdr header."""
        if vhdr_path is None:
            return None
        _vp = Path(vhdr_path)
        _parent = _vp.parent
        try:
            if _vp.is_file():
                with open(_vp, 'r', encoding='utf-8', errors='ignore') as _fv:
                    for _line in _fv:
                        _ls = _line.strip()
                        if _ls.lower().startswith('markerfile='):
                            _mval = _ls.split('=', 1)[1].strip()
                            if _mval:
                                for _cand in (_parent / _mval, _parent / Path(_mval).name):
                                    if _cand.is_file():
                                        return _cand
        except Exception:
            pass
        _exact = _vp.with_suffix('.vmrk')
        if _exact.is_file():
            return _exact
        try:
            _stem_low = _vp.stem.lower()
            for _f in sorted(_parent.iterdir()):
                if _f.is_file() and _f.suffix.lower() == '.vmrk' and _f.stem.lower() == _stem_low:
                    return _f
        except Exception:
            pass
        return None

    def _scan_marker_file_fast(filepath):
        """Extract sampling rate and unique non-stage marker descriptions in pure Python (no DataFrame)."""
        _fs_val = None
        _labels = set()
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as _fh:
                for _lno, _line in enumerate(_fh):
                    if _fs_val is None and _lno < 10:
                        _m_fs = _RE_MARKER_FS.search(_line)
                        if _m_fs:
                            _fs_val = int(round(float(_m_fs.group(1))))
                    _ls = _line.strip()
                    if not _ls or _ls[0] in (';', '['):
                        continue
                    if _ls.startswith(('Sampling rate', 'Brain Vision', 'DataFile', 'MarkerFile', 'Codepage', 'Type')):
                        continue
                    if _ls.startswith('Mk'):
                        _ls = _RE_MK_PREFIX.sub('', _ls, count=1)
                    _parts = _ls.split(',', 2)
                    if len(_parts) >= 2:
                        _t_low = _parts[0].lower()
                        if 'sleep stage' in _t_low or 'new segment' in _t_low:
                            continue
                        _desc = _parts[1].strip()
                        if _desc:
                            _labels.add(_desc)
        except Exception:
            pass
        return _fs_val, _labels

    # Function to read BrainVision marker files (.Markers or .vmrk) into a clean table
    def read_brainvision_markers(filepath):
        rows = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_s = line.strip()
                if not line_s or line_s[0] in (';', '['):
                    continue
                if line_s.startswith(('Sampling rate', 'Brain Vision', 'DataFile', 'MarkerFile', 'Codepage', 'Type')):
                    continue
                if line_s.startswith('Mk'):
                    line_s = _RE_MK_PREFIX.sub('', line_s, count=1)
                parts = [p.strip() for p in line_s.split(',')]
                if len(parts) >= 5:
                    rows.append(parts[:5])

        df = pd.DataFrame(rows, columns=["Type", "Description", "Position", "Length", "Channel"])
        df["Position"] = pd.to_numeric(df["Position"], errors='coerce')
        df["Length"] = pd.to_numeric(df["Length"], errors='coerce')
        df["Description"] = df["Description"].astype(str).str.strip()
        df["Type"] = df["Type"].astype(str).str.strip()
        return df.dropna(subset=["Position"]).sort_values("Position").reset_index(drop=True)

    # Function to compute the time offset (in seconds) between a cropped .Markers file (starting at LightsOff)
    # and the uncropped raw .vmrk/.eeg recording (starting at New Segment)
    def get_raw_marker_offset_sec(df_markers, raw_vmrk_path, marker_fs, raw_fs=None, marker_path=None):
        if df_markers.empty or raw_vmrk_path is None:
            return 0.0
        _vmrk_p = Path(raw_vmrk_path)
        if not _vmrk_p.is_file():
            return 0.0
        if marker_path is not None and _norm_fs_path(marker_path) == _norm_fs_path(_vmrk_p):
            return 0.0
        try:
            df_vmrk = read_brainvision_markers(_vmrk_p)
            if df_vmrk.empty:
                return 0.0
        except Exception:
            return 0.0

        _r_fs = float(raw_fs) if raw_fs is not None else float(marker_fs)
        _m_fs = float(marker_fs)
        _m_non_stage = df_markers[~df_markers["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)]
        _v_non_stage = df_vmrk[~df_vmrk["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)]
        if _m_non_stage.empty or _v_non_stage.empty:
            return 0.0

        # Build first-occurrence index for normalized raw .vmrk descriptions (O(N + M))
        _v_norms = _v_non_stage["Description"].str.lower().str.replace(r'[\s_\-]+', '', regex=True)
        _first_vmrk_pos = {}
        for _vn, _vp in zip(_v_norms, _v_non_stage["Position"]):
            if _vn and _vn not in _first_vmrk_pos:
                _first_vmrk_pos[_vn] = float(_vp)

        _m_norms = _m_non_stage["Description"].str.lower().str.replace(r'[\s_\-]+', '', regex=True)
        for _mn, _mp_pos in zip(_m_norms, _m_non_stage["Position"]):
            if not _mn:
                continue
            if _mn in _first_vmrk_pos:
                _t_m = (float(_mp_pos) - 1.0) / _m_fs
                _t_v = (_first_vmrk_pos[_mn] - 1.0) / _r_fs
                return max(0.0, float(_t_v - _t_m))
        return 0.0

    def scan_sessions(raw_root, marker_root):
        """Recursively scan raw .vhdr and .Markers/.vmrk folders, pair them automatically, and sort naturally."""
        _raw_root = Path(raw_root)
        _marker_root = Path(marker_root)

        # 1. Collect all .vhdr recordings recursively (skipping hidden folders/files)
        _vhdr_entries = []
        _vhdr_by_rel_stem = {}
        _vhdr_by_stem = {}
        if _raw_root.exists():
            for _dp, _dnames, _fnames in os.walk(_raw_root):
                _dnames[:] = sorted(d for d in _dnames if not d.startswith('.'))
                _rel_dir_path = Path(_dp).relative_to(_raw_root)
                _rel_dir_key = "" if _rel_dir_path == Path(".") else _rel_dir_path.as_posix().lower()
                for _fn in sorted(_fnames):
                    if _fn.startswith('.'):
                        continue
                    _vp = Path(_dp) / _fn
                    if _vp.suffix.lower() == '.vhdr':
                        _stem_low = _vp.stem.lower()
                        _rel_posix = _vp.relative_to(_raw_root).as_posix()
                        _ent = (_rel_dir_key, _stem_low, _vp, resolve_raw_vmrk(_vp), _rel_posix)
                        _vhdr_entries.append(_ent)
                        _vhdr_by_rel_stem[(_rel_dir_key, _stem_low)] = _ent
                        _vhdr_by_stem.setdefault(_stem_low, []).append(_ent)

        # 2. Collect all .Markers and .vmrk files recursively; deduplicate (folder, stem) favoring .Markers
        _marker_raw = []
        if _marker_root.exists():
            for _dp, _dnames, _fnames in os.walk(_marker_root):
                _dnames[:] = sorted(d for d in _dnames if not d.startswith('.'))
                for _fn in _fnames:
                    if _fn.startswith('.'):
                        continue
                    _mp = Path(_dp) / _fn
                    _sfx = _mp.suffix.lower()
                    if _sfx in ('.markers', '.vmrk'):
                        _prio = 0 if _sfx == '.markers' else 1
                        _marker_raw.append((_prio, _fn.lower(), _mp))

        _marker_dedup = {}
        for _prio, _fl, _mp in sorted(_marker_raw, key=lambda x: (x[0], x[1])):
            _dk = (os.path.normcase(str(_mp.parent)), _mp.stem.lower())
            _marker_dedup.setdefault(_dk, _mp)

        # 3. Pair each marker file to its best matching .vhdr
        _seps = ('_', '-', '.', ' ')
        def _pair_vhdr(m_path):
            if not _vhdr_entries:
                return None
            _m_rel = m_path.relative_to(_marker_root)
            _m_rel_dir = "" if _m_rel.parent == Path(".") else _m_rel.parent.as_posix().lower()
            _m_stem = m_path.stem.lower()
            _m_parent = m_path.parent.name.lower()

            # Tier 0: Exact relative folder + exact stem
            if (_m_rel_dir, _m_stem) in _vhdr_by_rel_stem:
                return _vhdr_by_rel_stem[(_m_rel_dir, _m_stem)]

            # Tier 1: Exact stem anywhere in raw_root
            if _m_stem in _vhdr_by_stem:
                _cands = _vhdr_by_stem[_m_stem]
                return min(
                    _cands,
                    key=lambda v: (0 if v[2].parent.name.lower() == _m_parent else 1, natural_sort_key(v[4]), v[4])
                )

            # Tier 2: Separator-delimited prefix match in either direction
            _t2 = [
                v for v in _vhdr_entries
                if any(_m_stem.startswith(v[1] + _s) for _s in _seps)
                or any(v[1].startswith(_m_stem + _s) for _s in _seps)
            ]
            if _t2:
                return min(
                    _t2,
                    key=lambda v: (
                        0 if v[0] == _m_rel_dir else 1,
                        -min(len(_m_stem), len(v[1])),
                        0 if v[2].parent.name.lower() == _m_parent else 1,
                        natural_sort_key(v[4]),
                        v[4],
                    )
                )

            # Tier 3: Legacy first-two underscore tokens (Participant_Condition_*)
            _m_p = _m_stem.split('_', 2)
            if len(_m_p) >= 2 and _m_p[0] and _m_p[1]:
                _t3 = [
                    v for v in _vhdr_entries
                    if len(v[1].split('_', 2)) >= 2 and v[1].split('_', 2)[:2] == _m_p[:2]
                ]
                if _t3:
                    return min(
                        _t3,
                        key=lambda v: (
                            0 if v[0] == _m_rel_dir else 1,
                            0 if v[2].parent.name.lower() == _m_parent else 1,
                            natural_sort_key(v[4]),
                            v[4],
                        )
                    )
            return None

        _paired_list = []
        _claimed_raw_vmrks = set()
        for _mp in _marker_dedup.values():
            _mv = _pair_vhdr(_mp)
            _paired_list.append((_mp, _mv))
            if _mp.suffix.lower() == '.markers' and _mv is not None and _mv[3] is not None:
                _claimed_raw_vmrks.add(_norm_fs_path(_mv[3]))

        # 4. Prune companion raw .vmrk files when a .Markers file already claims that recording
        _catalog_items = []
        for _mp, _mv in _paired_list:
            if _mp.suffix.lower() == '.vmrk' and _norm_fs_path(_mp) in _claimed_raw_vmrks:
                continue
            _rel = _mp.relative_to(_marker_root)
            _id = _mp.name if _rel.parent == Path(".") else _rel.as_posix()
            _disp = _mp.stem if _rel.parent == Path(".") else _rel.with_suffix("").as_posix()
            _catalog_items.append((
                _id,
                {
                    "id": _id,
                    "label": _disp,
                    "marker_path": _mp,
                    "vhdr_path": _mv[2] if _mv is not None else None,
                    "vmrk_path": _mv[3] if _mv is not None else None,
                },
            ))

        _catalog_items.sort(key=lambda kv: (natural_sort_key(kv[0]), kv[0]))
        return dict(_catalog_items)

    # --- Shared Fast Signal-Processing & Artifact Helpers ---
    def _get_worker_count():
        return min(os.cpu_count() or 4, 8)

    def design_composite_fir_kernel(sfreq, l_freq, h_freq, notch_freq):
        """Precompute single combined odd-length linear-phase FIR kernel (bandpass ⊛ harmonic notch)."""
        _sf = float(sfreq)
        _nyq = _sf / 2.0
        _lf = float(l_freq) if (l_freq is not None and float(l_freq) > 0) else None
        _hf = float(h_freq) if (h_freq is not None and float(h_freq) > 0 and float(h_freq) < _nyq) else None
        _h = None
        if _lf is not None or _hf is not None:
            _h = mne.filter.create_filter(None, _sf, _lf, _hf, fir_design='firwin', verbose='ERROR')

        _nf = float(notch_freq) if (notch_freq is not None and float(notch_freq) > 0) else 0.0
        if _nf > 0:
            _notches = np.arange(_nf, _nyq, _nf)
            if len(_notches) > 0:
                _nw = _notches / 200.0
                _tb = 1.0
                _lf_n = _notches + _nw / 2.0 + _tb / 2.0
                _hf_n = _notches - _nw / 2.0 - _tb / 2.0
                _valid = (_lf_n + _tb / 2.0 < _nyq) & (_hf_n - _tb / 2.0 > 0)
                if np.any(_valid):
                    _h_notch = mne.filter.create_filter(
                        None, _sf,
                        l_freq=_lf_n[_valid], h_freq=_hf_n[_valid],
                        l_trans_bandwidth=_tb / 2.0, h_trans_bandwidth=_tb / 2.0,
                        fir_design='firwin', verbose='ERROR'
                    )
                    _h = _h_notch if _h is None else scipy_signal.oaconvolve(_h, _h_notch)
        return _h

    def apply_fir_oaconvolve_mt(data_2d, kernel, sfreq, l_freq=None, h_freq=None, notch_freq=0.0):
        """Zero-phase reflect-padded overlap-add FFT filtering (MT across channels, MNE fallback for very short arrays)."""
        if data_2d.shape[0] == 0:
            return data_2d.copy()
        _sf = float(sfreq)
        if kernel is not None and data_2d.shape[1] >= len(kernel):
            _pad = len(kernel) - 1
            _half = _pad // 2
            _n_t = data_2d.shape[1]
            def _run_chunk(xc):
                _xp = np.pad(xc, ((0, 0), (_pad, _pad)), mode='reflect')
                _y = scipy_signal.oaconvolve(_xp, kernel[None, :], mode='valid', axes=-1)
                _start = _pad - _half
                return _y[:, _start:_start + _n_t]
            _nw = min(_get_worker_count(), data_2d.shape[0])
            if _nw > 1 and data_2d.shape[0] >= 4:
                _chunks = np.array_split(data_2d, _nw, axis=0)
                with ThreadPoolExecutor(max_workers=_nw) as _ex:
                    return np.vstack(list(_ex.map(_run_chunk, _chunks)))
            return _run_chunk(data_2d)

        # Safe MNE fallback when signal is shorter than filter length
        _out = np.array(data_2d, dtype=np.float64, copy=True)
        _nyq = _sf / 2.0
        _lf = float(l_freq) if (l_freq is not None and float(l_freq) > 0) else None
        _hf = float(h_freq) if (h_freq is not None and float(h_freq) > 0 and float(h_freq) < _nyq) else None
        if _lf is not None or _hf is not None:
            _out = mne.filter.filter_data(_out, _sf, _lf, _hf, fir_design='firwin', verbose='ERROR')
        _nf = float(notch_freq) if (notch_freq is not None and float(notch_freq) > 0) else 0.0
        if _nf > 0:
            _notches = np.arange(_nf, _nyq, _nf)
            _notches = _notches[_notches + (_notches / 400.0) + 1.0 < _nyq]
            if len(_notches) > 0:
                _out = mne.filter.notch_filter(_out, _sf, _notches, verbose='ERROR')
        return _out

    def compute_welch_psd_mt(data, sfreq, nperseg, noverlap, fmin=0.0, fmax=None):
        """Multi-threaded Welch PSD matching MNE Hamming window density, supporting 2D (C, T) or 3D (E, C, T)."""
        _nperseg = max(2, min(int(nperseg), int(data.shape[-1])))
        _noverlap = min(int(noverlap), _nperseg - 1)
        _n_items = data.shape[0]
        _nw = min(_get_worker_count(), _n_items)
        if _nw > 1 and _n_items >= 4:
            _chunks = np.array_split(data, _nw, axis=0)
            def _run_w(chunk):
                return scipy_signal.welch(
                    chunk, fs=sfreq, window='hamming',
                    nperseg=_nperseg, noverlap=_noverlap, axis=-1
                )
            with ThreadPoolExecutor(max_workers=_nw) as _ex:
                _res = list(_ex.map(_run_w, _chunks))
            _freqs = _res[0][0]
            _psd = np.concatenate([_r[1] for _r in _res], axis=0)
        else:
            _freqs, _psd = scipy_signal.welch(
                data, fs=sfreq, window='hamming',
                nperseg=_nperseg, noverlap=_noverlap, axis=-1
            )
        _mask = _freqs >= float(fmin)
        if fmax is not None:
            _mask = _mask & (_freqs <= float(fmax))
        return _freqs[_mask], _psd[..., _mask]

    def extract_epoch_metrics(seg_raw, ep_dur_sec):
        """Zero-copy non-overlapping epoch slicing and L2-cache-friendly PTP/step feature extraction."""
        _sf = float(seg_raw.info['sfreq'])
        _eeg_picks = mne.pick_types(seg_raw.info, eeg=True)
        _emg_picks = mne.pick_types(seg_raw.info, emg=True)
        _ep_samps = int(round(float(ep_dur_sec) * _sf))
        _n_times = int(seg_raw.n_times)
        _n_ep = (_n_times // _ep_samps) if _ep_samps >= 2 else 0
        _n_used = _n_ep * _ep_samps

        if _n_ep == 0:
            return (
                np.empty((0, len(_eeg_picks), 0), dtype=np.float64),
                np.empty((0, len(_eeg_picks)), dtype=np.float64),
                np.empty((0, len(_eeg_picks)), dtype=np.float64),
                np.empty((0, len(_emg_picks)), dtype=np.float64),
                _eeg_picks,
                _emg_picks,
                _ep_samps,
            )

        _raw_data = seg_raw.get_data()
        if len(_eeg_picks) > 0:
            _eeg_ep = np.ascontiguousarray(
                _raw_data[_eeg_picks, :_n_used].reshape(len(_eeg_picks), _n_ep, _ep_samps).transpose(1, 0, 2)
            )
            _eeg_ptp = np.empty((_n_ep, len(_eeg_picks)), dtype=np.float64)
            _eeg_step = np.empty((_n_ep, len(_eeg_picks)), dtype=np.float64)
            for _ei in range(_n_ep):
                _sl = _eeg_ep[_ei]
                _eeg_ptp[_ei] = np.ptp(_sl, axis=1)
                _eeg_step[_ei] = np.max(np.abs(np.diff(_sl, axis=1)), axis=1)
        else:
            _eeg_ep = np.empty((_n_ep, 0, _ep_samps), dtype=np.float64)
            _eeg_ptp = np.empty((_n_ep, 0), dtype=np.float64)
            _eeg_step = np.empty((_n_ep, 0), dtype=np.float64)

        if len(_emg_picks) > 0:
            _emg_ep = _raw_data[_emg_picks, :_n_used].reshape(len(_emg_picks), _n_ep, _ep_samps).transpose(1, 0, 2)
            _emg_ptp = np.ptp(_emg_ep, axis=2)
        else:
            _emg_ptp = np.empty((_n_ep, 0), dtype=np.float64)

        return _eeg_ep, _eeg_ptp, _eeg_step, _emg_ptp, _eeg_picks, _emg_picks, _ep_samps

    def evaluate_artifact_masks(eeg_ptp, eeg_step, emg_ptp, sfreq, flat_uv, ptp_uv, step_uv_ms, emg_uv):
        """Vectorized artifact thresholding preserving exact priority order and per-channel failure counts."""
        _n_ep = eeg_ptp.shape[0] if eeg_ptp.ndim == 2 else (emg_ptp.shape[0] if emg_ptp.ndim == 2 else 0)
        _n_eeg = eeg_ptp.shape[1] if eeg_ptp.ndim == 2 else 0
        _n_emg = emg_ptp.shape[1] if emg_ptp.ndim == 2 else 0
        _empty_counts = {
            "Retained (Clean)": 0,
            "Min Amplitude (Flatline)": 0,
            "Max Amplitude (PTP)": 0,
            "Voltage Step": 0,
            "EMG Activity": 0,
        }
        if _n_ep == 0:
            return (
                np.empty(0, dtype=int),
                np.empty(0, dtype=bool),
                _empty_counts,
                np.zeros(_n_eeg, dtype=int),
                np.zeros(_n_emg, dtype=int),
            )

        _flat_lim = float(flat_uv) * 1e-6
        _ptp_lim = float(ptp_uv) * 1e-6
        _step_lim = float(step_uv_ms) * (1000.0 / float(sfreq)) * 1e-6
        _emg_lim = float(emg_uv) * 1e-6

        if _n_eeg > 0:
            _flat_m = eeg_ptp < _flat_lim
            _ptp_m = eeg_ptp > _ptp_lim
            _step_m = eeg_step > _step_lim
            _chan_fail_counts = np.sum(_flat_m | _ptp_m | _step_m, axis=0, dtype=int)
            _ep_flat = np.any(_flat_m, axis=1)
            _ep_ptp = np.any(_ptp_m, axis=1)
            _ep_step = np.any(_step_m, axis=1)
        else:
            _chan_fail_counts = np.zeros(0, dtype=int)
            _ep_flat = np.zeros(_n_ep, dtype=bool)
            _ep_ptp = np.zeros(_n_ep, dtype=bool)
            _ep_step = np.zeros(_n_ep, dtype=bool)

        if _n_emg > 0:
            _emg_m = emg_ptp > _emg_lim
            _emg_fail_counts = np.sum(_emg_m, axis=0, dtype=int)
            _ep_emg = np.any(_emg_m, axis=1)
        else:
            _emg_fail_counts = np.zeros(0, dtype=int)
            _ep_emg = np.zeros(_n_ep, dtype=bool)

        _r_flat = _ep_flat
        _r_ptp = (~_r_flat) & _ep_ptp
        _r_step = (~_r_flat) & (~_r_ptp) & _ep_step
        _r_emg = (~_r_flat) & (~_r_ptp) & (~_r_step) & _ep_emg
        _rej_mask = _r_flat | _r_ptp | _r_step | _r_emg
        _valid_idx = np.flatnonzero(~_rej_mask)
        _drop_counts = {
            "Retained (Clean)": int(len(_valid_idx)),
            "Min Amplitude (Flatline)": int(np.sum(_r_flat)),
            "Max Amplitude (PTP)": int(np.sum(_r_ptp)),
            "Voltage Step": int(np.sum(_r_step)),
            "EMG Activity": int(np.sum(_r_emg)),
        }
        return _valid_idx, _rej_mask, _drop_counts, _chan_fail_counts, _emg_fail_counts

    # Scan directories and pair each marker file with its matching raw .vhdr recording
    recording_catalog = scan_sessions(data_dir, markers_dir)

    all_unique_labels = set()
    for _sess in recording_catalog.values():
        _, _lbls = _scan_marker_file_fast(_sess["marker_path"])
        all_unique_labels.update(_lbls)
    all_unique_labels.discard("")

    detected_labels_str = ", ".join(sorted(all_unique_labels)) if all_unique_labels else "None detected"
    _rec_options = list(recording_catalog.keys())

    recording_dropdown = mo.ui.dropdown(
        options=_rec_options,
        value=_rec_options[0] if _rec_options else None,
        searchable=True,
        label="Select Recording:",
    )

    _n_paired = sum(1 for _s in recording_catalog.values() if _s["vhdr_path"] is not None)
    _status_banner = (
        mo.callout(
            mo.md("⚠️ **Directory not found or empty.** Please verify the Raw Data and Markers paths above."),
            kind="warn"
        )
        if not recording_catalog
        else mo.md(
            f"**Recordings Found:** `{len(recording_catalog)}` (`{_n_paired}` paired with `.vhdr`)  \n"
            f"**Experimental Marker Labels Detected:** `{detected_labels_str}`"
        )
    )

    mo.vstack([
        mo.md("### 👤 Recording Selection"),
        _status_banner,
        recording_dropdown,
    ])
    return (
        ThreadPoolExecutor,
        all_unique_labels,
        apply_fir_oaconvolve_mt,
        compute_welch_psd_mt,
        data_dir,
        design_composite_fir_kernel,
        detect_marker_fs,
        detect_vhdr_fs,
        evaluate_artifact_masks,
        extract_epoch_metrics,
        get_raw_marker_offset_sec,
        markers_dir,
        mne,
        natural_sort_key,
        np,
        pd,
        plt,
        raw_segment_cache,
        read_brainvision_markers,
        recording_catalog,
        recording_dropdown,
        resolve_raw_vmrk,
        scipy_signal,
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


@app.cell
def _(detect_marker_fs, detect_vhdr_fs, mo, recording_catalog, recording_dropdown):
    _sel_key = recording_dropdown.value
    current_session = recording_catalog.get(_sel_key) if _sel_key else None
    current_recording_id = current_session["id"] if current_session else None
    current_recording_label = current_session["label"] if current_session else None
    current_marker_path = current_session["marker_path"] if current_session else None
    current_vhdr_path = current_session["vhdr_path"] if current_session else None
    current_raw_vmrk_path = current_session["vmrk_path"] if current_session else None

    detected_session_fs = detect_marker_fs(current_marker_path)
    if detected_session_fs is None:
        detected_session_fs = detect_vhdr_fs(current_vhdr_path)

    fs_input = mo.ui.number(
        value=detected_session_fs if detected_session_fs is not None else 500,
        label="Marker Sampling Rate (Hz) [Auto-detected for active recording, adjustable]:",
    )

    if current_session is None:
        _pair_ui = mo.md("")
    elif current_vhdr_path is not None:
        _pair_ui = mo.md(
            f"**Active Recording:** `{current_recording_id}`  \n"
            f"**Paired Raw Header:** `{current_vhdr_path.name}`"
        )
    else:
        _pair_ui = mo.callout(
            mo.md(
                f"⚠️ **No matching `.vhdr` file found** in Raw Data Path for `{current_recording_id}`. "
                "Whole-night staging & hypnogram work, but signal preprocessing requires a paired `.vhdr` recording."
            ),
            kind="warn",
        )

    mo.vstack([
        mo.md("---"),
        mo.md("### ⚙️ Active Recording & Sampling Rate"),
        _pair_ui,
        fs_input,
    ])
    return (
        current_marker_path,
        current_raw_vmrk_path,
        current_recording_id,
        current_recording_label,
        current_session,
        current_vhdr_path,
        detected_session_fs,
        fs_input,
    )


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

    - **Epoch Inclusion Rule (30s Resolution):** A scored 30-second sleep epoch is included in the pre-awakening window if its 15-second midpoint falls within $[t_{{\text{{awakening}}}} - \text{{window}},\, t_{{\text{{awakening}}}}]$.
    - **Boundary Protection:** If the final 30-second epoch straddles the awakening trigger and is scored as Wake (`W`), Movement/Artifact (`ART`), or Unscored (`UNS`)—which commonly occurs under AASM rules when the participant spends $>15\text{{ s}}$ of that epoch giving a dream report—that straddling epoch is excluded so **Last Stage at Awakening** reflects the true physiological sleep stage preceding the awakening.
    - **Stimulus Latency:** Dynamically calculates elapsed time (`MM:SS`) from the most recent experimental stimulus trigger within the pre-awakening lookback window to the awakening marker.
    """)
    return


@app.cell
def _(
    awake_marker_select,
    current_marker_path,
    current_recording_label,
    detected_session_fs,
    event_markers_select,
    fs_input,
    mo,
    pd,
    read_brainvision_markers,
    window_size_input,
):
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

    # Function to extract pre-awakening sleep stages and event latencies for a given recording
    def analyze_sleep_stages(marker_path, rec_label):
        # Return empty tables if no marker file is selected
        if marker_path is None:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if not marker_path.exists():
            raise FileNotFoundError(f"Marker file not found: {marker_path}")

        fs = float(fs_input.value)

        # Parse all markers from the file
        df_markers = read_brainvision_markers(marker_path)

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
            if len(epochs_window) > 1:
                _last_ep_pos = epochs_window.iloc[-1]["Position"]
                _last_ep_stage = epochs_window.iloc[-1]["Stage_YASA"]
                if (_last_ep_pos < pos_awake < (_last_ep_pos + full_epoch_pts)) and (_last_ep_stage in ("W", "ART", "UNS")):
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
                "Recording": rec_label,
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

        return df_markers, stages, awakenings, pd.DataFrame(report_rows)

    # Run the sleep stage analysis and display the summary table (or a warning if the file is missing)
    try:
        df_markers, df_stages, awakenings, summary_df = analyze_sleep_stages(
            current_marker_path, current_recording_label or ""
        )

        # Warn the user if the active session's header sampling rate differs from the UI number box
        _fs_warning = (
            mo.callout(
                mo.md(
                    f"⚠️ **Sampling Rate Mismatch:** This session's file header specifies **`{detected_session_fs} Hz`**, "
                    f"but the Sampling Rate input box above is set to **`{fs_input.value} Hz`**. "
                    f"Please update the Sampling Rate input box above to `{detected_session_fs}` so all downstream plots align accurately."
                ),
                kind="warn"
            )
            if (detected_session_fs is not None and detected_session_fs != fs_input.value)
            else mo.md("")
        )

        _stage_ui = mo.vstack([
            _fs_warning,
            mo.md(f"**Recording:** `{current_recording_label or 'None'}` | **Awakenings Detected:** `{len(awakenings)}`"),
            mo.ui.table(summary_df)
        ])
    except FileNotFoundError as _e:
        df_markers, df_stages, awakenings, summary_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _stage_ui = mo.callout(mo.md(f"⚠️ **File Not Found:** `{_e}`"), kind="warn")

    _stage_ui
    return (
        awakenings,
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
    current_marker_path,
    current_raw_vmrk_path,
    current_recording_label,
    df_markers,
    df_stages,
    fs_input,
    get_raw_marker_offset_sec,
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
        _stats_df.insert(0, "Recording", current_recording_label or "")

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
            f"Whole-Night Hypnogram — {current_recording_label or ''}",
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
            _vmrk_file = current_raw_vmrk_path or current_marker_path
            _raw_offset_sec_hyp = get_raw_marker_offset_sec(
                df_markers, current_raw_vmrk_path, _fs, marker_path=current_marker_path
            )
            if _vmrk_file is not None and _vmrk_file.is_file():
                with open(_vmrk_file, 'r', encoding='utf-8', errors='ignore') as _f_vmrk:
                    for _line in _f_vmrk:
                        if 'New Segment' in _line:
                            _parts = _line.split(',')
                            if len(_parts) >= 6:
                                _time_str = _parts[5].strip()
                                if len(_time_str) >= 14:
                                    _start_time = _datetime.datetime.strptime(_time_str[:14], "%Y%m%d%H%M%S")
                                    _start_time += _datetime.timedelta(hours=_offset_hours + (_raw_offset_sec_hyp / 3600.0))
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

    To optimize memory and computation speed, signal processing is applied specifically to the **{window_size_input.value}-minute** segments preceding each targeted awakening, with buffered raw segments cached in RAM so re-adjusting filter settings skips disk reads.

    1. **Timeline Synchronization & Buffer Padding:** Automatically aligns cropped `.Markers` timelines (starting at `LightsOff`) with uncropped raw `.eeg` recordings. Each segment is extracted with an extra $\pm 20\text{{ s}}$ buffer of continuous raw data on both ends prior to filtering, then cropped back after filtering so FIR filter edge ringing never contaminates the analysis window.
    2. **Channel Typing & Re-Referencing:** Channels are automatically classified into cortical EEG, EOG, EMG (chin/facial), ECG, and Misc (respiration, SCR/EDA). Bad/detached electrodes are dropped prior to linking cortical EEG channels to the selected reference electrodes (e.g., mastoids `M1`, `M2`).
    3. **Composite Bandpass & Notch Filtering (Zero-Phase FIR):** Each channel group's `firwin` bandpass and harmonic powerline notch (`scipy.signal.oaconvolve`) kernels are precomputed once per session, convolved into a single zero-phase linear-phase FIR kernel, and applied across channels via multi-threaded overlap-add FFT convolution (defaults: $0.3\text{{–}}45.0\text{{ Hz}}$ EEG/EOG, $10\text{{–}}90\text{{ Hz}}$ EMG, $50\text{{ Hz}}$ harmonics).
    """)
    return


@app.cell
def _(current_vhdr_path, mne, mo):
    # Read the EEG header metadata lazily (without loading the signal) to get available channel names
    try:
        if current_vhdr_path is None or not current_vhdr_path.is_file():
            raise FileNotFoundError("No paired .vhdr file")
        _raw_info = mne.io.read_raw_brainvision(current_vhdr_path, preload=False, verbose='ERROR')
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
    apply_fir_oaconvolve_mt,
    awakenings,
    bad_channels_input,
    compute_welch_psd_mt,
    current_marker_path,
    current_raw_vmrk_path,
    current_recording_label,
    current_vhdr_path,
    design_composite_fir_kernel,
    df_markers,
    fs_input,
    get_raw_marker_offset_sec,
    h_freq_input,
    l_freq_input,
    mne,
    mo,
    notch_freq_input,
    np,
    plt,
    raw_segment_cache,
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
            if ch in ref_chs or ch_up in {'M1', 'M2', 'A1', 'A2'}:
                ch_types[ch] = 'misc'
            elif any(k in ch_up for k in ('EOG', 'LOC', 'ROC', 'HEOG', 'VEOG')):
                ch_types[ch] = 'eog'
            elif any(k in ch_up for k in ('EMG', 'CHIN', 'SUBMENT')):
                ch_types[ch] = 'emg'
            elif any(k in ch_up for k in ('ECG', 'EKG', 'HEART')):
                ch_types[ch] = 'ecg'
            elif any(k in ch_up for k in ('RESP', 'FLOW', 'THOR', 'ABDO', 'SNORE', 'SPO2', 'PLETH', 'POS', 'SCR', 'EDA', 'GSR')):
                ch_types[ch] = 'misc'
            else:
                ch_types[ch] = 'eeg'
        return ch_types

    _marker_fs = float(fs_input.value)
    _analysis_win = float(window_size_input.value) * 60.0
    preprocessed_segments = []
    _raw_ex_eeg = None

    if awakenings.empty:
        output_ui_prep = mo.callout(mo.md("⚠️ **No awakenings available to preprocess.**"), kind="warn")
    elif current_vhdr_path is None or not current_vhdr_path.is_file():
        output_ui_prep = mo.callout(
            mo.md(
                f"⚠️ **No matching raw `.vhdr` recording found** for `{current_recording_label or 'selected recording'}`. "
                "Check your Raw Data Path and filenames."
            ),
            kind="warn"
        )
    else:
        mo.output.replace(
            mo.callout(
                mo.md("### ⏳ Currently preprocessing EEG data...\n*Please wait, this may take a moment.*"),
                kind="info"
            )
        )

        # Open header lazily without reading signals
        raw = mne.io.read_raw_brainvision(current_vhdr_path, preload=False, verbose='ERROR')
        _raw_sfreq = float(raw.info['sfreq'])
        _raw_max_time = float(raw.times[-1])

        # Detect offset between cropped .Markers and uncropped raw recording
        _raw_offset_sec = get_raw_marker_offset_sec(
            df_markers, current_raw_vmrk_path, _marker_fs, raw_fs=_raw_sfreq, marker_path=current_marker_path
        )
        _last_aw_sec = ((awakenings["Position"].max() - 1.0) / _marker_fs) + _raw_offset_sec
        if _last_aw_sec > _raw_max_time + 1.0:
            _raw_offset_sec = 0.0

        # Precompute buffered & cropped sample indices for every awakening
        _pad_sec = 20.0
        _bounds = []
        for _, _row in awakenings.iterrows():
            _t_end = max(0.0, min(((_row["Position"] - 1.0) / _marker_fs) + _raw_offset_sec, _raw_max_time))
            _t_start = max(0.0, _t_end - _analysis_win)
            _t_start_pad = max(0.0, _t_start - _pad_sec)
            _t_end_pad = min(_t_end + _pad_sec, _raw_max_time)
            _idx_t = raw.time_as_index([_t_start_pad, _t_end_pad, _t_start, _t_end])
            _s0_pad = max(0, int(_idx_t[0]))
            _s1_pad = min(int(_idx_t[1]) + 1, int(raw.n_times))
            if _s1_pad <= _s0_pad:
                _s1_pad = min(_s0_pad + 1, int(raw.n_times))
            _rel_s0 = max(0, min(int(_idx_t[2]) - _s0_pad, _s1_pad - _s0_pad - 1))
            _rel_s1 = min(max(_rel_s0 + 1, int(_idx_t[3]) + 1 - _s0_pad), _s1_pad - _s0_pad)
            _bounds.append((_s0_pad, _s1_pad, _rel_s0, _rel_s1))

        # Cache buffered raw windows in RAM (float32, capped at 1.5 GB) so filter changes skip disk I/O
        _st = current_vhdr_path.stat()
        _cache_key = (str(current_vhdr_path.resolve()), _st.st_mtime_ns, _st.st_size, tuple(_bounds))
        _est_bytes = sum((_b[1] - _b[0]) for _b in _bounds) * len(raw.ch_names) * 4
        _use_ram_cache = _est_bytes <= int(1.5 * (1024 ** 3))
        if _use_ram_cache:
            _cache_hit = _cache_key in raw_segment_cache
            if not _cache_hit:
                raw_segment_cache.clear()
                raw_segment_cache[_cache_key] = [
                    raw.get_data(start=_s0, stop=_s1).astype(np.float32)
                    for _s0, _s1, _, _ in _bounds
                ]
            _cached_bufs = raw_segment_cache[_cache_key]
        else:
            _cache_hit = False
            _cached_bufs = None

        # Channel selection & classification built ONCE
        _bads_set = set(bad_channels_input.value)
        _bads_to_drop = [ch for ch in raw.ch_names if ch in _bads_set]
        _keep_idx = [i for i, ch in enumerate(raw.ch_names) if ch not in _bads_set]
        _keep_names = [raw.ch_names[i] for i in _keep_idx]
        _ref_channels = [ch for ch in ref_channels_input.value if ch in _keep_names]
        _final_types = classify_sleep_channels(_keep_names, ref_chs=_ref_channels)

        _info_clean = mne.create_info(
            _keep_names,
            _raw_sfreq,
            ch_types=[_final_types[_c] for _c in _keep_names],
        )
        _info_clean.set_montage('standard_1020', match_case=False, on_missing='ignore', verbose='ERROR')

        _ref_set = set(_ref_channels)
        _ref_pos = [j for j, _c in enumerate(_keep_names) if _c in _ref_set]
        _eeg_pos = [j for j, _c in enumerate(_keep_names) if _final_types[_c] == 'eeg']
        _eog_pos = [j for j, _c in enumerate(_keep_names) if _final_types[_c] == 'eog']
        _emg_pos = [j for j, _c in enumerate(_keep_names) if _final_types[_c] == 'emg']
        _eeg_eog_pos = _eeg_pos + _eog_pos

        # Precompute composite FIR kernels (bandpass ⊛ harmonic notch) ONCE outside loop
        _notch_val = float(notch_freq_input.value) if notch_freq_input.value else 0.0
        _l_val = l_freq_input.value
        _h_val = h_freq_input.value
        _h_eeg = (
            design_composite_fir_kernel(_raw_sfreq, _l_val, _h_val, _notch_val)
            if _eeg_eog_pos else None
        )
        _emg_hfreq = min(90.0, (_raw_sfreq / 2.0) - 1.0)
        _emg_lo_arg = 10.0 if _emg_hfreq > 15.0 else None
        _emg_hi_arg = _emg_hfreq if _emg_hfreq > 15.0 else None
        _h_emg = (
            design_composite_fir_kernel(_raw_sfreq, _emg_lo_arg, _emg_hi_arg, _notch_val)
            if (_emg_pos and (_emg_hi_arg is not None or _notch_val > 0)) else None
        )

        with mo.status.progress_bar(title="Filtering Awakenings", total=len(_bounds)) as bar:
            for _idx, (_s0_pad, _s1_pad, _rel_s0, _rel_s1) in enumerate(_bounds):
                bar.update(title=f"Filtering Awakening {_idx + 1} of {len(_bounds)}...")

                if _cached_bufs is not None:
                    _seg_all = _cached_bufs[_idx][_keep_idx].astype(np.float64, copy=True)
                else:
                    _seg_all = raw.get_data(picks=_keep_idx, start=_s0_pad, stop=_s1_pad).astype(np.float64, copy=True)

                # Direct NumPy re-referencing across active EEG and reference electrodes
                if _ref_pos:
                    _ref_mean = np.mean(_seg_all[_ref_pos], axis=0, keepdims=True)
                    if _eeg_pos:
                        _seg_all[_eeg_pos] -= _ref_mean
                    _seg_all[_ref_pos] -= _ref_mean

                # Save unfiltered cropped EEG of Awakening #1 for filter validation PSD
                if _idx == 0 and _eeg_pos:
                    _raw_ex_eeg = _seg_all[_eeg_pos, _rel_s0:_rel_s1].copy()

                # Apply composite FIR bandpass + notch via multi-threaded oaconvolve
                if _eeg_eog_pos and (_h_eeg is not None or _l_val is not None or _h_val is not None or _notch_val > 0):
                    _seg_all[_eeg_eog_pos] = apply_fir_oaconvolve_mt(
                        _seg_all[_eeg_eog_pos], _h_eeg, _raw_sfreq,
                        l_freq=_l_val, h_freq=_h_val, notch_freq=_notch_val
                    )
                if _emg_pos and (_h_emg is not None or _emg_hi_arg is not None or _notch_val > 0):
                    _seg_all[_emg_pos] = apply_fir_oaconvolve_mt(
                        _seg_all[_emg_pos], _h_emg, _raw_sfreq,
                        l_freq=_emg_lo_arg, h_freq=_emg_hi_arg, notch_freq=_notch_val
                    )

                _cropped = np.ascontiguousarray(_seg_all[:, _rel_s0:_rel_s1])
                preprocessed_segments.append(mne.io.RawArray(_cropped, _info_clean.copy(), verbose='ERROR'))

        # Plot Power Spectral Density (PSD) of Awakening #1 before and after filtering using fast Welch
        if _raw_ex_eeg is not None and len(preprocessed_segments) > 0 and _raw_ex_eeg.shape[1] >= 4:
            plt.style.use('default')
            fig_pre, (ax_raw, ax_filt) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, facecolor='white')

            _nfft_pre = min(2048, _raw_ex_eeg.shape[1])
            _f_pre, _p_pre = compute_welch_psd_mt(_raw_ex_eeg, _raw_sfreq, nperseg=_nfft_pre, noverlap=0, fmin=0.0, fmax=60.0)
            _aft_eeg = preprocessed_segments[0].get_data(picks='eeg')
            _f_aft, _p_aft = compute_welch_psd_mt(_aft_eeg, _raw_sfreq, nperseg=min(2048, _aft_eeg.shape[1]), noverlap=0, fmin=0.0, fmax=60.0)

            _db_pre = 10.0 * np.log10(np.maximum(_p_pre * 1e12, 1e-12))
            _db_aft = 10.0 * np.log10(np.maximum(_p_aft * 1e12, 1e-12))

            _m_pre = np.mean(_db_pre, axis=0)
            ax_raw.plot(_f_pre, _m_pre, color='#1f77b4', linewidth=1.8)
            if _db_pre.shape[0] > 1:
                _s_pre = np.std(_db_pre, axis=0)
                ax_raw.fill_between(_f_pre, _m_pre - _s_pre, _m_pre + _s_pre, color='#1f77b4', alpha=0.22)
            ax_raw.set_title("Cortical EEG PSD — BEFORE Filters (Re-referenced Raw)")
            ax_raw.set_xlabel("Frequency (Hz)")
            ax_raw.set_ylabel("Power Spectral Density (dB re 1 µV²/Hz)")
            ax_raw.grid(True, linestyle=':', alpha=0.5)

            _m_aft = np.mean(_db_aft, axis=0)
            ax_filt.plot(_f_aft, _m_aft, color='#1f77b4', linewidth=1.8)
            if _db_aft.shape[0] > 1:
                _s_aft = np.std(_db_aft, axis=0)
                ax_filt.fill_between(_f_aft, _m_aft - _s_aft, _m_aft + _s_aft, color='#1f77b4', alpha=0.22)
            ax_filt.set_title("Cortical EEG PSD — AFTER Filters")
            ax_filt.set_xlabel("Frequency (Hz)")
            ax_filt.grid(True, linestyle=':', alpha=0.5)

            fig_pre.tight_layout()

            _cache_note = " | Raw Source: `RAM Cache`" if _cache_hit else ""
            output_ui_prep = mo.vstack([
                mo.md(
                    f"✅ **Successfully preprocessed {len(preprocessed_segments)} awakenings** "
                    f"(Window: {window_size_input.value} min | Raw Sampling Rate: `{_raw_sfreq} Hz` | "
                    f"Raw LightsOff Offset: `+{_raw_offset_sec:.2f} s` | "
                    f"Excluded Channels: `{_bads_to_drop if _bads_to_drop else 'None'}`{_cache_note})"
                ),
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

    Segments the continuous {window_size_input.value}-minute pre-awakening signal into **non-overlapping fixed-length epochs** and applies conservative, global threshold-based rejection. If *any* active EEG channel (or chin EMG channel) violates a threshold during an epoch, the entire epoch is rejected across all channels to preserve spatial covariance for topomaps.

    - **Flatline (Min PTP Amplitude):** Detects amplifier saturation, bridging, or detached electrodes ($\mu\text{{V}}$).
    - **Peak-to-Peak Excursion (Max PTP Amplitude):** Rejects high-amplitude movement, sweat, or slow-wave artifacts exceeding physiological limits ($\mu\text{{V}}$).
    - **Fast Voltage Step:** Identifies abrupt sample-to-sample jumps ($\mu\text{{V/ms}}$) caused by cable pops or electrode impedance spikes.
    - **High-Frequency EMG Burst:** Rejects epochs where filtered chin EMG ($10\text{{–}}90\text{{ Hz}}$) exceeds the muscle-tension threshold ($\mu\text{{V}}$).
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
    epoch_length_slider = mo.ui.slider(start=2, stop=30, step=1, value=4, label="Epoch Length (Seconds)", show_value=True, debounce=True)

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
    compute_welch_psd_mt,
    epoch_length_slider,
    extract_epoch_metrics,
    h_freq_input,
    mne,
    np,
    preprocessed_segments,
):
    # Precompute non-overlapping epoch metrics and per-epoch Welch PSD ONCE per awakening / epoch duration
    segment_idx = awake_dropdown.value
    if segment_idx is None or len(preprocessed_segments) == 0:
        selected_segment = None
        epoch_cache_bundle = None
    else:
        selected_segment = preprocessed_segments[segment_idx]
        _ep_dur = float(epoch_length_slider.value)
        _eeg_ep, _eeg_ptp, _eeg_step, _emg_ptp, _eeg_picks, _emg_picks, _ep_samps = extract_epoch_metrics(
            selected_segment, _ep_dur
        )
        _sfreq = float(selected_segment.info['sfreq'])
        _eeg_names = [selected_segment.ch_names[p] for p in _eeg_picks]
        _emg_names = [selected_segment.ch_names[p] for p in _emg_picks]
        _n_ep = int(_eeg_ptp.shape[0])
        _eeg_info = mne.pick_info(selected_segment.info, _eeg_picks) if len(_eeg_picks) > 0 else None

        _freqs = np.empty(0, dtype=np.float64)
        _all_ep_psd_uv = None
        if _n_ep > 0 and len(_eeg_picks) > 0:
            _max_f = min(45.0, float(h_freq_input.value) if h_freq_input.value else 45.0)
            _welch_samps = min(int(round(4.0 * _sfreq)), _ep_samps)
            _overlap_samps = int(_welch_samps * 0.50) if _welch_samps < _ep_samps else 0
            _freqs, _psd_v2 = compute_welch_psd_mt(
                _eeg_ep, _sfreq,
                nperseg=_welch_samps, noverlap=_overlap_samps,
                fmin=0.5, fmax=_max_f
            )
            _all_ep_psd_uv = _psd_v2 * 1e12

        epoch_cache_bundle = {
            "eeg_ptp": _eeg_ptp,
            "eeg_step": _eeg_step,
            "emg_ptp": _emg_ptp,
            "all_ep_psd_uv": _all_ep_psd_uv,
            "freqs": _freqs,
            "eeg_names": _eeg_names,
            "emg_names": _emg_names,
            "eeg_info": _eeg_info,
            "sfreq": _sfreq,
            "ep_dur": _ep_dur,
            "n_ep": _n_ep,
            "total_dur_min": float(selected_segment.times[-1]) / 60.0,
        }

    return epoch_cache_bundle, segment_idx, selected_segment


@app.cell
def _(
    emg_slider,
    epoch_cache_bundle,
    evaluate_artifact_masks,
    flat_slider,
    mo,
    np,
    plt,
    ptp_slider,
    step_slider,
):
    # Evaluate threshold sliders instantaneously against precomputed per-epoch metrics and PSDs
    rejected_intervals = []

    if epoch_cache_bundle is None:
        clean_psd_bundle = None
        output_ui = mo.callout(
            mo.md("⚠️ **No awakenings detected.** Please select a valid marker or recording above."),
            kind="warn"
        )
    else:
        _eeg_ptp = epoch_cache_bundle["eeg_ptp"]
        _eeg_step = epoch_cache_bundle["eeg_step"]
        _emg_ptp = epoch_cache_bundle["emg_ptp"]
        _all_psd_uv = epoch_cache_bundle["all_ep_psd_uv"]
        _freqs = epoch_cache_bundle["freqs"]
        _eeg_names = epoch_cache_bundle["eeg_names"]
        _emg_names = epoch_cache_bundle["emg_names"]
        _sfreq = epoch_cache_bundle["sfreq"]
        _ep_dur = epoch_cache_bundle["ep_dur"]
        _n_ep = epoch_cache_bundle["n_ep"]
        _total_dur_min = epoch_cache_bundle["total_dur_min"]

        _valid_idx, _rej_mask, drop_counts, _chan_fail_counts, _emg_fail_counts = evaluate_artifact_masks(
            _eeg_ptp, _eeg_step, _emg_ptp, _sfreq,
            flat_slider.value, ptp_slider.value, step_slider.value, emg_slider.value
        )

        for _ri in np.flatnonzero(_rej_mask):
            _t0_min = ((int(_ri) * _ep_dur) / 60.0) - _total_dur_min
            _t1_min = (((int(_ri) + 1) * _ep_dur) / 60.0) - _total_dur_min
            rejected_intervals.append((_t0_min, _t1_min))

        _clean_psd_uv = _all_psd_uv[_valid_idx] if _all_psd_uv is not None else None
        _n_clean = int(len(_valid_idx))
        clean_psd_bundle = {
            "psd_uv": _clean_psd_uv,
            "freqs": _freqs,
            "ch_names": _eeg_names,
            "eeg_info": epoch_cache_bundle["eeg_info"],
            "sfreq": _sfreq,
            "ep_start_sec": _valid_idx.astype(np.float64) * _ep_dur,
            "ep_dur": _ep_dur,
            "n_total_epochs": _n_ep,
            "n_clean": _n_clean,
        }

        plt.style.use('default')
        fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5), facecolor='white')
        bars = ax_bar.bar(
            drop_counts.keys(),
            drop_counts.values(),
            color=['forestgreen', 'firebrick', 'goldenrod', 'purple', 'sienna']
        )
        ax_bar.set_ylabel('Number of Epochs')
        ax_bar.set_title(f'Artifact Rejection Summary (Total: {_n_ep} Non-Overlapping Epochs)')
        max_val = max(drop_counts.values()) if max(drop_counts.values()) > 0 else 1
        ax_bar.set_ylim(0, max_val * 1.22)
        plt.xticks(rotation=15)
        for _bar in bars:
            _yval = _bar.get_height()
            ax_bar.text(_bar.get_x() + _bar.get_width() / 2, _yval + (max_val * 0.02), int(_yval), ha='center', va='bottom')
        fig_bar.tight_layout()

        def _plot_psd_summary(_ax, _psd_ep_uv, _title):
            if _psd_ep_uv is not None and _psd_ep_uv.shape[0] > 0 and len(_freqs) > 0:
                _ch_psd_db = 10.0 * np.log10(np.maximum(np.mean(_psd_ep_uv, axis=0), 1e-12))
                _m_db = np.mean(_ch_psd_db, axis=0)
                _ax.plot(_freqs, _m_db, color='#1f77b4', linewidth=1.8)
                if _ch_psd_db.shape[0] > 1:
                    _s_db = np.std(_ch_psd_db, axis=0)
                    _ax.fill_between(_freqs, _m_db - _s_db, _m_db + _s_db, color='#1f77b4', alpha=0.22)
                _ax.set_title(_title)
                _ax.set_xlabel("Frequency (Hz)")
                _ax.set_ylabel("Power Spectral Density (dB re 1 µV²/Hz)")
                _ax.grid(True, linestyle=':', alpha=0.5)
                return True
            return False

        fig_before, ax_before = plt.subplots(figsize=(6, 4), facecolor='white')
        if _plot_psd_summary(ax_before, _all_psd_uv, "PSD BEFORE Rejection"):
            fig_before.tight_layout()
        else:
            ax_before.text(0.5, 0.5, "⚠️ No raw epochs", ha='center', va='center', fontsize=12)
            ax_before.axis('off')

        fig_after, ax_after = plt.subplots(figsize=(6, 4), facecolor='white')
        if _plot_psd_summary(ax_after, _clean_psd_uv, "PSD AFTER Rejection"):
            fig_after.tight_layout()
        else:
            ax_after.text(0.5, 0.5, "⚠️ No clean epochs retained", ha='center', va='center', fontsize=12)
            ax_after.axis('off')

        _all_diag_names = _eeg_names + _emg_names
        _all_diag_counts = np.concatenate([_chan_fail_counts, _emg_fail_counts]) if (_eeg_names or _emg_names) else np.array([], dtype=int)
        _top_bad_str = "None"
        if len(_all_diag_names) > 0 and np.any(_all_diag_counts > 0):
            _sorted_idx = np.argsort(_all_diag_counts)[::-1][:5]
            _top_bad_str = ", ".join(
                f"`{_all_diag_names[idx]}` ({100.0 * _all_diag_counts[idx] / max(1, _n_ep):.0f}%)"
                for idx in _sorted_idx if _all_diag_counts[idx] > 0
            )

        output_ui = mo.vstack([
            mo.md("### Artifact Rejection Statistics"),
            fig_bar,
            mo.md(
                f"**Final Clean Epochs Retained:** `{_n_clean}` / `{_n_ep}` (`{100 * _n_clean / max(1, _n_ep):.1f}%`)  \n"
                f"**Top Artifact-Triggering Channels (EEG / EMG):** {_top_bad_str} *(Tip: If one detached or facial electrode is rejecting most epochs, exclude it in Pre-Processing)*"
            ),
            mo.md("---"),
            mo.md("### Signal Quality Comparison"),
            mo.hstack([fig_before, fig_after])
        ])

    output_ui
    return clean_psd_bundle, rejected_intervals


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

    Computes the power spectral density of clean, artifact-free EEG epochs using **Welch's Method** ($4\text{ s}$ Hamming windows, $0.25\text{ Hz}$ frequency resolution, with $50\%$ sub-window overlap when epoch length $>4\text{ s}$). Power spectra are calculated per epoch and aggregated across clean epochs via the **Mean** or **Median** (robust to residual outliers) before integrating bandpower via Simpson's rule (`YASA`).

    - **Clinical Sleep Preset:** SWA ($0.5\text{–}4\text{ Hz}$), Theta ($4\text{–}8\text{ Hz}$), Alpha ($8\text{–}12\text{ Hz}$), Slow Spindles ($11\text{–}13\text{ Hz}$), Fast Spindles ($13\text{–}15\text{ Hz}$), and Beta ($15\text{–}25\text{ Hz}$).
    - **Standard EEG Preset:** Delta ($1\text{–}4\text{ Hz}$), Theta ($4\text{–}8\text{ Hz}$), Alpha ($8\text{–}12\text{ Hz}$), Beta ($12\text{–}30\text{ Hz}$), and Gamma ($30\text{–}45\text{ Hz}$).
    - **Note on Relative Power (%):** Relative bandpower divides each band's absolute power by the total broadband power between the lowest and highest frequencies of the active preset. Because *Alpha* ($8\text{–}12\text{ Hz}$) and *Slow Spindles* ($11\text{–}13\text{ Hz}$) overlap in the $11\text{–}12\text{ Hz}$ bin, row-wise relative percentages in the Clinical Sleep preset can sum to slightly more than $100\%$.
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
        options=["Mean across epochs", "Median across epochs"],
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
    clean_psd_bundle,
    current_recording_label,
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
    bp_df = pd.DataFrame()

    # Integrate bandpower directly from precomputed clean epoch PSDs (zero extra FFTs)
    if clean_psd_bundle is not None and clean_psd_bundle["n_clean"] > 0 and clean_psd_bundle["psd_uv"] is not None:
        _psd_data_uv = clean_psd_bundle["psd_uv"]
        _freqs = clean_psd_bundle["freqs"]
        _eeg_ch_names = clean_psd_bundle["ch_names"]
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
        bp_df.insert(0, 'Recording', current_recording_label or '')
        _awake_id = segment_idx + 1 if segment_idx is not None else "Unknown"
        bp_df.insert(1, 'Awakening', _awake_id)
        bp_df.insert(2, 'Metric', power_norm_select.value)

        # Display the resulting bandpower table
        output_table = mo.vstack([
            mo.md(f"### Spectral Power Analysis — {band_preset_select.value}"),
            mo.md(f"**{psd_agg_select.value} ({power_norm_select.value}) per EEG Channel:**"),
            mo.ui.table(bp_df, page_size=25)
        ])
    else:
        output_table = mo.callout(
            mo.md("⚠️ **No clean epochs available.** Adjust the artifact rejection criteria above to retain epochs."),
            kind="warn"
        )

    output_table
    return bp_df, selected_bands


@app.cell
def _(
    clean_psd_bundle,
    current_recording_label,
    mo,
    np,
    plt,
    psd_agg_select,
    segment_idx,
):
    # Plot the average Power Spectral Density curve across all EEG channels from precomputed clean PSDs
    if clean_psd_bundle is not None and clean_psd_bundle["n_clean"] > 0 and clean_psd_bundle["psd_uv"] is not None:
        plt.close('all')
        plt.style.use('default')

        _psd_uv = clean_psd_bundle["psd_uv"]
        _freqs = clean_psd_bundle["freqs"]
        _use_median = "Median" in psd_agg_select.value
        _psd_ep_agg = np.median(_psd_uv, axis=0) if _use_median else np.mean(_psd_uv, axis=0)
        _ch_db = 10.0 * np.log10(np.maximum(_psd_ep_agg, 1e-12))

        _fig_psd, _ax_psd = plt.subplots(figsize=(10, 4.5), facecolor='white')
        _m_db = np.mean(_ch_db, axis=0)
        _ax_psd.plot(_freqs, _m_db, color='#1f77b4', linewidth=2.0)
        if _ch_db.shape[0] > 1:
            _s_db = np.std(_ch_db, axis=0)
            _ax_psd.fill_between(_freqs, _m_db - _s_db, _m_db + _s_db, color='#1f77b4', alpha=0.22)

        _ax_psd.set_title(
            f"Average Power Spectral Density — {psd_agg_select.value} (Awakening #{segment_idx + 1})\n"
            f"Recording: {current_recording_label or ''} | Clean Epochs: {clean_psd_bundle['n_clean']}"
        )
        _ax_psd.set_xlabel("Frequency (Hz)")
        _ax_psd.set_ylabel("Power Spectral Density (dB re 1 µV²/Hz)")
        _ax_psd.grid(True, linestyle=':', alpha=0.5)
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
    # Whole-Window Scalp Topography ({window_size_input.value}-Minute Average)

    Displays the 2D spherical-spline interpolated scalp distribution (`standard_1020` montage) of each frequency band averaged across all clean epochs in the {window_size_input.value}-minute pre-awakening window.
    """)
    return


@app.cell
def _(
    bp_df,
    clean_psd_bundle,
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
    if not bp_df.empty and clean_psd_bundle is not None and clean_psd_bundle["n_clean"] > 0 and clean_psd_bundle["eeg_info"] is not None:
        _eeg_info = clean_psd_bundle["eeg_info"]
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

    Allows you to scrub through the pre-awakening period using a configurable moving integration sub-window ($15\text{ s}$ to $240\text{ s}$) synchronized with scored sleep stages, experimental triggers, and rejected artifact windows.

    - **Sub-Window Epoch Selection:** Aggregates all clean epochs whose temporal midpoint falls inside $[\text{Start Time},\, \text{Start Time} + \text{Width}]$.
    - **Fixed vs. Dynamic Colorbar Scaling:** *Fixed Scale* locks each band's colorbar to the $2\text{nd}\text{–}98\text{th}$ percentile across the entire pre-awakening window so color changes reflect true temporal amplitude shifts. *Dynamic Scale* rescales the colorbar at each time step to highlight local spatial topography (increases contrast locally).
    - **Split-Screen Pre- vs. Post-Stimulus Mode:** Automatically locks onto the most recent experimental trigger in the window to compare Baseline ($[\text{Onset} - \text{Width},\, \text{Onset}]$) vs. Stimulation ($[\text{Onset},\, \text{Onset} + \text{Width}]$) topographies side-by-side, strictly excluding any epoch that straddles the exact stimulus onset. When *Dynamic Scale* is active in Split-Screen mode, colorbars are pair-locked across the Baseline and Stimulation rows for valid visual comparison.
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
            "Fixed Scale (Compare amplitude changes across the full window)",
            "Dynamic Scale (Maximize spatial contrast at each time point)"
        ],
        value="Fixed Scale (Compare amplitude changes across the full window)",
        inline=True,
        label="Colorbar Scaling Mode:"
    )

    # Checkbox to trigger the automatic split-screen Event-Related comparison
    compare_stim_checkbox = mo.ui.checkbox(
        label="📊 Split-Screen: Pre- vs. Post-Stimulus (Overrides Slider)",
        value=False
    )

    # Slider to scrub through time (in minutes) prior to the awakening
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
        mo.hstack([topo_win_select, topo_scale_select, compare_stim_checkbox], justify="start", gap=2)
    ])
    return (
        compare_stim_checkbox,
        topo_scale_select,
        topo_time_slider,
        topo_win_select,
    )


@app.cell
def _(
    awakenings,
    clean_psd_bundle,
    df_markers,
    df_stages,
    event_markers_select,
    fs_input,
    mne,
    np,
    power_norm_select,
    psd_agg_select,
    segment_idx,
    selected_bands,
    selected_segment,
    topo_scale_select,
    topo_win_select,
    yasa,
):
    # Precompute valid 10-20 channels, 15-step Fixed Scale limits, and window stages/markers ONCE (decoupled from slider)
    topo_precomp = None
    if (
        clean_psd_bundle is not None
        and clean_psd_bundle["n_clean"] > 0
        and clean_psd_bundle["psd_uv"] is not None
        and clean_psd_bundle["eeg_info"] is not None
        and selected_segment is not None
    ):
        _eeg_info = clean_psd_bundle["eeg_info"]
        _psd_ch_names = clean_psd_bundle["ch_names"]
        _valid_chs = [
            i for i, ch in enumerate(_eeg_info['chs'])
            if ch['loc'] is not None
            and not np.all(np.isnan(ch['loc'][:3]))
            and np.any(ch['loc'][:3] != 0)
            and _eeg_info['ch_names'][i] in _psd_ch_names
        ]

        if len(_valid_chs) >= 3:
            _info_sub = mne.pick_info(_eeg_info, _valid_chs)
            _ch_order = _info_sub.ch_names
            _ch_indices = [_psd_ch_names.index(ch) for ch in _ch_order]
            _psd_all_uv = clean_psd_bundle["psd_uv"][:, _ch_indices, :]
            _freqs = clean_psd_bundle["freqs"]
            _ep_dur = clean_psd_bundle["ep_dur"]
            _total_dur_min = float(selected_segment.times[-1]) / 60.0

            _ep_start_sec = clean_psd_bundle["ep_start_sec"]
            _ep_t0_min = (_ep_start_sec / 60.0) - _total_dur_min
            _ep_t1_min = ((_ep_start_sec + _ep_dur) / 60.0) - _total_dur_min
            _ep_center_min = 0.5 * (_ep_t0_min + _ep_t1_min)
            _sub_win_min = float(topo_win_select.value) / 60.0

            _is_relative = "Relative" in power_norm_select.value
            _use_median = "Median" in psd_agg_select.value
            _use_fixed_scale = "Fixed" in topo_scale_select.value

            # Precompute global 15-step Fixed Scale bounds only when scaling/aggregation/bands change
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
                            float(np.nanpercentile(_b_vals, 98)),
                        )

            # Pre-filter window stage rows and markers via vectorized sample bounds
            _win_stages = []
            _win_markers = []
            _target_stim_min = None
            _target_stim_name = None
            _fs = float(fs_input.value)
            if not awakenings.empty and segment_idx is not None:
                _awake_pos = float(awakenings.iloc[segment_idx]["Position"])
                _prev_awake_pos = float(awakenings.iloc[segment_idx - 1]["Position"]) if segment_idx > 0 else 0.0

                if not df_stages.empty:
                    _min_st_pos = _awake_pos - ((_total_dur_min + 0.5) * 60.0 * _fs)
                    _sub_st = df_stages[
                        (df_stages["Position"] > _min_st_pos) & (df_stages["Position"] < _awake_pos)
                    ]
                    for _, _st_row in _sub_st.iterrows():
                        _st_t0 = (float(_st_row["Position"]) - _awake_pos) / _fs / 60.0
                        _st_t1 = _st_t0 + 0.5
                        if _st_t1 > -_total_dur_min and _st_t0 < 0.0:
                            _c_t0 = max(-_total_dur_min, _st_t0)
                            _c_t1 = min(0.0, _st_t1)
                            _win_stages.append((_c_t0, _c_t1, _st_t0, _st_t1, _st_row["Stage_YASA"]))

                if not df_markers.empty:
                    _min_m_pos = max(_prev_awake_pos, _awake_pos - (_total_dur_min * 60.0 * _fs) - 1.0)
                    _win_m = df_markers[
                        (~df_markers["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)) &
                        (df_markers["Position"] > _min_m_pos) &
                        (df_markers["Position"] < _awake_pos)
                    ]
                    for _, _mr in _win_m.iterrows():
                        _m_min = (float(_mr["Position"]) - _awake_pos) / _fs / 60.0
                        if -_total_dur_min <= _m_min <= 0.0:
                            _m_desc = str(_mr["Description"]).strip()
                            _is_exp = _m_desc in event_markers_select.value
                            _win_markers.append((_m_min, _m_desc, _is_exp))
                            if _is_exp:
                                _target_stim_min = _m_min
                                _target_stim_name = _m_desc

            topo_precomp = {
                "info_sub": _info_sub,
                "ch_order": _ch_order,
                "psd_all_uv": _psd_all_uv,
                "freqs": _freqs,
                "ep_t0_min": _ep_t0_min,
                "ep_t1_min": _ep_t1_min,
                "ep_center_min": _ep_center_min,
                "total_dur_min": _total_dur_min,
                "sub_win_min": _sub_win_min,
                "is_relative": _is_relative,
                "use_median": _use_median,
                "use_fixed_scale": _use_fixed_scale,
                "vlim_map": _vlim_map,
                "win_stages": _win_stages,
                "win_markers": _win_markers,
                "target_stim_min": _target_stim_min,
                "target_stim_name": _target_stim_name,
            }

    return (topo_precomp,)


@app.cell
def _(
    compare_stim_checkbox,
    mne,
    mo,
    np,
    plt,
    rejected_intervals,
    selected_bands,
    topo_precomp,
    topo_time_slider,
    yasa,
):
    # Render active time-resolved scalp topographies using precomputed Fixed Scale bounds and timeline rows
    import inspect as _inspect
    import matplotlib.ticker as _ticker

    if topo_precomp is not None:
        _info_sub = topo_precomp["info_sub"]
        _ch_order = topo_precomp["ch_order"]
        _psd_all_uv = topo_precomp["psd_all_uv"]
        _freqs = topo_precomp["freqs"]
        _ep_t0_min = topo_precomp["ep_t0_min"]
        _ep_t1_min = topo_precomp["ep_t1_min"]
        _ep_center_min = topo_precomp["ep_center_min"]
        _total_dur_min = topo_precomp["total_dur_min"]
        _sub_win_min = topo_precomp["sub_win_min"]
        _is_relative = topo_precomp["is_relative"]
        _use_median = topo_precomp["use_median"]
        _use_fixed_scale = topo_precomp["use_fixed_scale"]
        _vlim_map = dict(topo_precomp["vlim_map"])
        _win_stages = topo_precomp["win_stages"]
        _win_markers = topo_precomp["win_markers"]
        _target_stim_min = topo_precomp["target_stim_min"]
        _target_stim_name = topo_precomp["target_stim_name"]

        _compare_mode = compare_stim_checkbox.value
        _split_fallback_notice = None

        if _compare_mode and _target_stim_min is None:
            _split_fallback_notice = mo.callout(
                mo.md(
                    "ℹ️ **No experimental stimulus marker found in this pre-awakening window.** "
                    "Displaying standard single-window slider view instead."
                ),
                kind="info"
            )

        if _compare_mode and _target_stim_min is not None:
            _t_low_pre = max(-_total_dur_min, _target_stim_min - _sub_win_min)
            _t_high_pre = _target_stim_min
            _t_low_post = _target_stim_min
            _t_high_post = min(0.0, _target_stim_min + _sub_win_min)

            _mask_pre = (_ep_center_min >= _t_low_pre) & (_ep_t1_min <= _t_high_pre)
            if not np.any(_mask_pre):
                _mask_pre = (_ep_t0_min < _t_high_pre) & (_ep_t1_min <= _t_high_pre)

            _mask_post = (_ep_t0_min >= _t_low_post) & (_ep_center_min <= _t_high_post)
            if not np.any(_mask_post):
                _mask_post = (_ep_t0_min >= _t_low_post) & (_ep_t0_min < _t_high_post)

            windows_to_plot = [
                {"label": "BASELINE", "mask": _mask_pre, "t_low": _t_low_pre, "t_high": _t_high_pre},
                {"label": "STIMULATION", "mask": _mask_post, "t_low": _t_low_post, "t_high": _t_high_post}
            ]
            _n_rows = 3
            _height_ratios = [3.0, 3.0, 1.0]
            _fig_height = 8.0
        else:
            _t_low = float(topo_time_slider.value)
            _t_high = _t_low + _sub_win_min
            if _t_high > 0.0:
                _t_high = 0.0
                _t_low = max(-_total_dur_min, _t_high - _sub_win_min)
            if _t_low < -_total_dur_min:
                _t_low = -_total_dur_min
                _t_high = min(0.0, _t_low + _sub_win_min)

            _in_win_mask = (_ep_center_min >= _t_low) & (_ep_center_min <= _t_high)
            if not np.any(_in_win_mask):
                _in_win_mask = (_ep_t0_min < _t_high) & (_ep_t1_min > _t_low)

            windows_to_plot = [
                {"label": "", "mask": _in_win_mask, "t_low": _t_low, "t_high": _t_high}
            ]
            _n_rows = 2
            _height_ratios = [3.3, 1.0]
            _fig_height = 5.0

        for _w_item in windows_to_plot:
            _m_cur = _w_item["mask"]
            _w_item["n_clean"] = int(np.sum(_m_cur))
            if _w_item["n_clean"] > 0:
                _psd_slice_agg = (
                    np.median(_psd_all_uv[_m_cur], axis=0)
                    if _use_median else np.mean(_psd_all_uv[_m_cur], axis=0)
                )
                _bp_slice = yasa.bandpower_from_psd(
                    _psd_slice_agg, _freqs, ch_names=_ch_order, bands=selected_bands, relative=_is_relative
                )
                if _is_relative:
                    for _, _, _b_name in selected_bands:
                        _bp_slice[_b_name] = _bp_slice[_b_name] * 100.0
                if 'Chan' in _bp_slice.columns:
                    _bp_slice = _bp_slice.set_index('Chan')
                _w_item["bp_df"] = _bp_slice
            else:
                _w_item["bp_df"] = None

        if not _use_fixed_scale and len(windows_to_plot) == 2:
            for _, _, _b_name in selected_bands:
                _pair_vals = [
                    _w_item["bp_df"].loc[_ch_order, _b_name].to_numpy(dtype=float)
                    for _w_item in windows_to_plot if _w_item["bp_df"] is not None
                ]
                if _pair_vals:
                    _concat_vals = np.concatenate(_pair_vals)
                    _vlim_map[_b_name] = (float(np.nanmin(_concat_vals)), float(np.nanmax(_concat_vals)))

        _n_bands = len(selected_bands)
        _fig_dyn = plt.figure(figsize=(max(12.0, 2.6 * _n_bands), _fig_height))
        _gs = _fig_dyn.add_gridspec(_n_rows, _n_bands, height_ratios=_height_ratios, hspace=0.35, wspace=0.25)
        _supports_vlim = 'vlim' in _inspect.signature(mne.viz.plot_topomap).parameters

        for _row_idx, _win_data in enumerate(windows_to_plot):
            _bp_slice = _win_data["bp_df"]
            _n_clean = _win_data["n_clean"]

            if _bp_slice is not None and _n_clean > 0:
                for _b_idx, (_, _, _b_name) in enumerate(selected_bands):
                    _ax_t = _fig_dyn.add_subplot(_gs[_row_idx, _b_idx])
                    _vals = _bp_slice.loc[_ch_order, _b_name].to_numpy(dtype=float)
                    _cmap = 'RdBu_r' if _is_relative else 'viridis'
                    _vlim = _vlim_map.get(_b_name, (None, None))

                    _topo_kwargs = dict(axes=_ax_t, show=False, cmap=_cmap, contours=4)
                    if _supports_vlim:
                        _topo_kwargs['vlim'] = _vlim
                    else:
                        _topo_kwargs['vmin'], _topo_kwargs['vmax'] = _vlim

                    _im, _ = mne.viz.plot_topomap(_vals, _info_sub, **_topo_kwargs)

                    if _row_idx == 0:
                        _ax_t.set_title(_b_name, fontsize=11, fontweight='bold')

                    if _b_idx == 0 and _win_data["label"]:
                        _ax_t.set_ylabel(
                            f"{_win_data['label']}\n(n={_n_clean} epochs)",
                            fontsize=11, fontweight='bold', labelpad=25
                        )

                    plt.colorbar(_im, ax=_ax_t, fraction=0.046, pad=0.04)
            else:
                _ax_warn = _fig_dyn.add_subplot(_gs[_row_idx, :])
                _ax_warn.text(
                    0.5, 0.5,
                    f"⚠️ All epochs in {_win_data['label'] or 'window'} [{_win_data['t_low']:.2f}, {_win_data['t_high']:.2f}] min were rejected as artifacts.",
                    ha='center', va='center', fontsize=12, fontweight='bold', color='#c0392b'
                )
                _ax_warn.axis('off')

        # Synchronized Timeline (bottom row)
        _ax_time = _fig_dyn.add_subplot(_gs[_n_rows - 1, :])
        _stage_colors = {
            "W": "#f39c12", "N1": "#85c1e9", "N2": "#2e86c1",
            "N3": "#1b4f72", "REM": "#8e44ad", "ART": "#e74c3c", "UNS": "#bdc3c7"
        }

        _stage_check_time = (
            _target_stim_min
            if (_compare_mode and _target_stim_min is not None)
            else 0.5 * (windows_to_plot[0]["t_low"] + windows_to_plot[0]["t_high"])
        )
        _current_stage_label = "Unknown"

        for _c_t0, _c_t1, _st_t0, _st_t1, _st_name in _win_stages:
            _ax_time.axvspan(_c_t0, _c_t1, ymin=0.0, ymax=0.32, color=_stage_colors.get(_st_name, "#bdc3c7"), alpha=0.85)
            if (_c_t1 - _c_t0) >= 0.35:
                _ax_time.text(0.5 * (_c_t0 + _c_t1), 0.16, _st_name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            if _st_t0 <= _stage_check_time <= _st_t1:
                _current_stage_label = _st_name

        for _idx_art, (_rt0, _rt1) in enumerate(rejected_intervals):
            _ax_time.axvspan(_rt0, _rt1, ymin=0.78, ymax=1.0, color='#e74c3c', alpha=0.8, label="Artifact" if _idx_art == 0 else "_nolegend_")

        _active_stim_label = "None (Baseline)"
        for _m_min, _m_desc, _is_exp in _win_markers:
            _m_col = '#16a085' if _is_exp else '#34495e'
            _ax_time.axvline(_m_min, color=_m_col, linestyle='--', linewidth=2.0)
            if _is_exp:
                _ax_time.axvspan(_m_min, 0.0, ymin=0.32, ymax=0.78, color='#1abc9c', alpha=0.18)
                if not (_compare_mode and _target_stim_min is not None):
                    _t_low_s = windows_to_plot[0]["t_low"]
                    _t_high_s = windows_to_plot[0]["t_high"]
                    if _t_low_s >= _m_min:
                        _active_stim_label = _m_desc
                    elif _t_high_s > _m_min:
                        _active_stim_label = f"{_m_desc} (Partial Overlap)"
            _ax_time.text(
                _m_min, 0.72, f" {_m_desc} ({_m_min:.1f}m)",
                color=_m_col, fontweight='bold', fontsize=9, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=_m_col, alpha=0.9)
            )

        for _w in windows_to_plot:
            _w_t_l = _w["t_low"]
            _w_t_h = _w["t_high"]
            _box_col = '#3498db' if "BASE" in _w["label"] else ('#e67e22' if "STIM" in _w["label"] else '#f1c40f')
            _ax_time.axvspan(_w_t_l, _w_t_h, color=_box_col, alpha=0.45)
            _ax_time.axvline(_w_t_l, color='#d35400', linewidth=1.8)
            _ax_time.axvline(_w_t_h, color='#d35400', linewidth=1.8)

        _ax_time.set_xlim(-_total_dur_min, 0.0)
        _ax_time.set_ylim(0.0, 1.0)
        _ax_time.set_yticks([0.16, 0.55, 0.89])
        _ax_time.set_yticklabels(["Stage", "Events", "Artifacts"], fontsize=9, fontweight='bold')

        if _total_dur_min <= 15:
            _ax_time.xaxis.set_major_locator(_ticker.MultipleLocator(1.0))
            _ax_time.xaxis.set_minor_locator(_ticker.MultipleLocator(0.5))
        elif _total_dur_min <= 30:
            _ax_time.xaxis.set_major_locator(_ticker.MultipleLocator(2.0))
            _ax_time.xaxis.set_minor_locator(_ticker.MultipleLocator(1.0))
        else:
            _ax_time.xaxis.set_major_locator(_ticker.MaxNLocator(nbins=12))

        _ax_time.set_xlabel("Time Relative to Awakening (Minutes)", fontsize=10, fontweight='bold')
        _ax_time.grid(axis='x', which='major', linestyle='-', alpha=0.5)
        _ax_time.grid(axis='x', which='minor', linestyle=':', alpha=0.3)

        if _compare_mode and _target_stim_min is not None:
            _scale_note = "Global Window Scale" if _use_fixed_scale else "Pair-Locked Pre/Post Scale"
            _fig_dyn.suptitle(
                f"Split-Screen Comparison: Pre- vs. Post-Stimulus ({_target_stim_name}) | "
                f"Stage at Onset: {_current_stage_label} | Sub-Window: {_sub_win_min*60:.0f}s | {_scale_note}",
                fontsize=12.5, fontweight='bold', y=0.98
            )
        else:
            _clean_eps_count = windows_to_plot[0]["n_clean"]
            _fig_dyn.suptitle(
                f"Time-Resolved Topography: [{windows_to_plot[0]['t_low']:.2f} to {windows_to_plot[0]['t_high']:.2f} min] | "
                f"Stage: {_current_stage_label} | Stimulus: {_active_stim_label} | Clean Epochs: {_clean_eps_count}",
                fontsize=12.5, fontweight='bold', y=0.98
            )

        _ui_items = [_split_item for _split_item in [_split_fallback_notice, _fig_dyn, topo_time_slider] if _split_item is not None]
        _dyn_topo_out = mo.vstack(_ui_items)
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

    Visualizes continuous time-frequency power dynamics across the {window_size_input.value} minutes preceding awakening using a Short-Time Fourier Transform (STFT; Hamming window with $87.5\%$ temporal overlap).

    - **Power-Domain Channel Averaging:** When multiple channels are selected, the STFT power density ($\mu\text{{V}}^2/\text{{Hz}}$) is computed for each channel independently and averaged in the linear power domain before log-transformation ($10\log_{{10}}$) into decibels ($\text{{dB}}$). This prevents phase cancellation across electrodes.
    - **Visual Contrast:** Colorbar limits are clipped at the $5\text{{th}}$ and $95\text{{th}}$ percentiles of the displayed time-frequency matrix, with rejected artifact epochs marked in red along the top border.
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
def _(mne, mo, selected_segment):
    # Get the list of available EEG channels from the selected segment and choose default frontal/central channels
    if selected_segment is None:
        options = []
        fallback_chan = []
    else:
        options = [selected_segment.ch_names[i] for i in mne.pick_types(selected_segment.info, eeg=True)]
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
    ThreadPoolExecutor,
    awakenings,
    current_recording_label,
    df_markers,
    event_markers_select,
    fs_input,
    mo,
    np,
    plt,
    rejected_intervals,
    scipy_signal,
    segment_idx,
    selected_segment,
    tfr_chan_select,
    tfr_cmap_select,
    tfr_fmax_slider,
    tfr_window_slider,
):
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
        _actual_sfreq = float(selected_segment.info['sfreq'])

        # Calculate STFT window length and 87.5% overlap
        nperseg = min(int(round(current_win_sec * _actual_sfreq)), int(data_nd_uv.shape[1]))
        noverlap = min(int(round(nperseg * 0.875)), max(0, nperseg - 1))

        # Compute spectrogram independently per channel (parallelized when multiple channels selected)
        if data_nd_uv.shape[0] > 1:
            def _run_spec(_row):
                return scipy_signal.spectrogram(
                    _row,
                    fs=_actual_sfreq,
                    window='hamming',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scaling='density',
                )
            with ThreadPoolExecutor(max_workers=min(data_nd_uv.shape[0], 8)) as _ex:
                _spec_res = list(_ex.map(_run_spec, data_nd_uv))
            freqs, times, _ = _spec_res[0]
            Sxx_mean = np.mean(np.stack([_r[2] for _r in _spec_res]), axis=0)
        else:
            freqs, times, Sxx_mean = scipy_signal.spectrogram(
                data_nd_uv[0],
                fs=_actual_sfreq,
                window='hamming',
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density',
            )

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

        total_dur_min = float(selected_segment.times[-1]) / 60.0
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

        # Draw vertical dashed lines for experimental events filtered to the awakening window
        _num_visible_events = 0
        if not awakenings.empty and segment_idx is not None and not df_markers.empty:
            _fs_tfr = float(fs_input.value)
            _awake_pos = float(awakenings.iloc[segment_idx]["Position"])
            _min_ev_pos = _awake_pos - (total_dur_min * 60.0 * _fs_tfr)
            _active_ev = df_markers[
                (df_markers["Description"].isin(event_markers_select.value)) &
                (df_markers["Position"] >= _min_ev_pos) &
                (df_markers["Position"] <= _awake_pos)
            ]
            for _, _ev_row in _active_ev.iterrows():
                _ev_rel_min = (float(_ev_row["Position"]) - _awake_pos) / _fs_tfr / 60.0
                if -total_dur_min <= _ev_rel_min <= 0:
                    ax_tfr.axvline(_ev_rel_min, color='black', linestyle='--', linewidth=2.0)
                    ax_tfr.axvline(_ev_rel_min, color='white', linestyle='--', linewidth=1.2)

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
            f"Recording: {current_recording_label or ''} | Awakening #{_awake_num}",
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
    compute_welch_psd_mt,
    emg_slider,
    epoch_length_slider,
    evaluate_artifact_masks,
    extract_epoch_metrics,
    flat_slider,
    h_freq_input,
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
        mo.output.replace(
            mo.callout(
                mo.md("### ⏳ Computing spectral power across all awakenings...\n*Please wait, building master export table.*"),
                kind="info"
            )
        )

        _batch_rows = []
        _ep_dur_b = float(epoch_length_slider.value)
        _max_f_b = float(h_freq_input.value) if h_freq_input.value else 45.0
        _is_rel_b = "Relative" in power_norm_select.value
        _use_med_b = "Median" in psd_agg_select.value

        # Zero-copy vectorized epoching, masking, and MT Welch PSD across all awakenings
        for _aw_i, _seg in enumerate(preprocessed_segments):
            _eeg_ep_b, _eeg_ptp_b, _eeg_step_b, _emg_ptp_b, _eeg_p, _emg_p, _ep_samps = extract_epoch_metrics(
                _seg, _ep_dur_b
            )
            _sf = float(_seg.info['sfreq'])
            _n_total = int(_eeg_ptp_b.shape[0])
            if _n_total == 0 or len(_eeg_p) == 0:
                continue

            _valid_b, _, _, _, _ = evaluate_artifact_masks(
                _eeg_ptp_b, _eeg_step_b, _emg_ptp_b, _sf,
                flat_slider.value, ptp_slider.value, step_slider.value, emg_slider.value
            )
            _n_clean = int(len(_valid_b))
            if _n_clean == 0:
                continue

            _w_samps = min(int(round(4.0 * _sf)), _ep_samps)
            _ov_samps = int(_w_samps * 0.50) if _w_samps < _ep_samps else 0
            _freqs_b, _psd_v2_b = compute_welch_psd_mt(
                _eeg_ep_b[_valid_b], _sf,
                nperseg=_w_samps, noverlap=_ov_samps,
                fmin=0.5, fmax=min(45.0, _max_f_b)
            )
            _psd_uv_b = _psd_v2_b * 1e12
            _psd_agg_b = np.median(_psd_uv_b, axis=0) if _use_med_b else _psd_uv_b.mean(axis=0)

            _eeg_names_b = [_seg.ch_names[p] for p in _eeg_p]
            _bp_b = yasa.bandpower_from_psd(
                _psd_agg_b, _freqs_b, ch_names=_eeg_names_b,
                bands=selected_bands, relative=_is_rel_b
            )
            if _is_rel_b:
                _b_cols = [b[2] for b in selected_bands]
                _bp_b[_b_cols] = _bp_b[_b_cols] * 100.0

            if 'Chan' in _bp_b.columns:
                _df_aw = _bp_b.rename(columns={'Chan': 'Channel'}).round(4)
            else:
                _bp_b.index.name = 'Channel'
                _df_aw = _bp_b.reset_index().round(4)

            _ret_pct = round(100.0 * _n_clean / max(1, _n_total), 1)
            _df_aw.insert(0, 'Aggregation', "Median" if _use_med_b else "Mean")
            _df_aw.insert(0, 'Metric', power_norm_select.value)
            _df_aw.insert(0, 'Retention_Pct', _ret_pct)
            _df_aw.insert(0, 'Total_Epochs', _n_total)
            _df_aw.insert(0, 'Clean_Epochs', _n_clean)
            _df_aw.insert(0, 'Epoch_Length_s', _ep_dur_b)
            _df_aw.insert(0, 'Awakening', _aw_i + 1)
            _batch_rows.append(_df_aw)

        if _batch_rows:
            _all_bp_df = pd.concat(_batch_rows, ignore_index=True)
            _meta_cols = [
                "Recording", "Awakening", "Time (HH:MM:SS)",
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
    # - natural_sort_key(text)
    #     Created in: Code Cell 2 (Core Imports, Recursive Discovery & Shared Helpers)
    #     Description: Numeric-aware sorting key for arbitrary filenames and subfolder paths.
    #
    # - detect_marker_fs(filepath) / detect_vhdr_fs(vhdr_path)
    #     Created in: Code Cell 2 (Core Imports, Recursive Discovery & Shared Helpers)
    #     Description: Extracts exact sampling rate from .Markers/.vmrk or .vhdr headers.
    #
    # - resolve_raw_vmrk(vhdr_path)
    #     Created in: Code Cell 2 (Core Imports, Recursive Discovery & Shared Helpers)
    #     Description: Resolves the companion raw .vmrk path referenced by a .vhdr file.
    #
    # - read_brainvision_markers(filepath) / _scan_marker_file_fast(filepath)
    #     Created in: Code Cell 2 (Core Imports, Recursive Discovery & Shared Helpers)
    #     Description: Parses BrainVision .Markers/.vmrk files (full table or fast pure-Python scan).
    #
    # - scan_sessions(raw_root, marker_root)
    #     Created in: Code Cell 2 (Core Imports, Recursive Discovery & Shared Helpers)
    #     Description: Recursively scans raw .vhdr and .Markers/.vmrk files, deduplicates
    #                  and pairs sessions across any filename/folder layout, and sorts naturally.
    #
    # - design_composite_fir_kernel(...) / apply_fir_oaconvolve_mt(...)
    #     Created in: Code Cell 2 (Core Imports, Recursive Discovery & Shared Helpers)
    #     Description: Precomputes combined bandpass ⊛ harmonic notch linear-phase FIR kernels
    #                  and applies zero-phase reflect-padded FFT convolution across threads.
    #
    # - compute_welch_psd_mt(...) / extract_epoch_metrics(...) / evaluate_artifact_masks(...)
    #     Created in: Code Cell 2 (Core Imports, Recursive Discovery & Shared Helpers)
    #     Description: Zero-copy non-overlapping epoch slicing, L2-friendly PTP/step metrics,
    #                  vectorized artifact masking, and multi-threaded Welch PSD computation.
    #
    # - analyze_sleep_stages(marker_path, rec_label)
    #     Created in: Code Cell 4 (Pre-Awakening Sleep Stage Extraction)
    #     Description: Extracts sleep stages and experimental events preceding each
    #                  awakening and builds a pre-awakening summary DataFrame.
    #
    # - _make_abbrev(name, used) / _hours_to_mdates(x) / _mdates_to_hours(x)
    #     Created in: Code Cell 6 (Whole-Night Hypnogram Plot & Sleep Macro-Statistics)
    #     Description: Marker label abbreviation and 24-hour clock axis helpers.
    #
    # - classify_sleep_channels(ch_names, ref_chs=None)
    #     Created in: Code Cell 8 (Signal Pre-Processing Execution & Filter Validation)
    #     Description: Automatically assigns polysomnography channel names to MNE
    #                  channel types ('eeg', 'eog', 'emg', 'ecg', or 'misc').
    #
    # ------------------------------------------------------------------------------
    # 2. SHARED / EXPORTED VARIABLES (USED ACROSS MULTIPLE CELLS)
    # ------------------------------------------------------------------------------
    # - recording_catalog      : Dict mapping recording ID -> {id, label, marker_path, vhdr_path, vmrk_path}.
    # - recording_dropdown     : Searchable UI dropdown selecting the active recording.
    # - current_session        : Active session metadata dict from recording_catalog.
    # - current_recording_id   : Relative marker file identifier of the active recording.
    # - current_recording_label: Display label (stem / relative stem) of the active recording.
    # - current_marker_path    : Path to the active .Markers or .vmrk file.
    # - current_vhdr_path      : Path to the paired raw .vhdr header (or None if unpaired).
    # - current_raw_vmrk_path  : Path to the paired raw companion .vmrk file (or None).
    # - detected_session_fs    : Auto-detected sampling rate (Hz) for the active recording.
    # - fs_input               : UI number input initialized to detected_session_fs.
    # - raw_segment_cache      : Non-reactive RAM cache (float32, 1.5 GB cap) of buffered awakening windows.
    # - all_unique_labels      : Set of all non-stage marker names detected across recordings.
    # - awake_marker_select    : UI dropdown for choosing the marker that defines an awakening.
    # - window_size_input      : UI number input setting the pre-awakening window length (minutes).
    # - event_markers_select   : UI multiselect for choosing experimental stimulation markers.
    # - df_markers / df_stages / awakenings / summary_df : Session marker & pre-awakening tables.
    # - hypno_markers_select   : UI multiselect choosing which markers to overlay on the hypnogram.
    # - ref_channels_input / bad_channels_input / l_freq_input / h_freq_input / notch_freq_input
    # - preprocessed_segments  : List of filtered MNE Raw objects (one per pre-awakening window).
    # - awake_dropdown         : UI dropdown to pick which awakening index to inspect.
    # - epoch_length_slider / emg_slider / step_slider / ptp_slider / flat_slider
    # - epoch_cache_bundle     : Precomputed epoch PTP/step metrics and all-epoch Welch PSD array.
    # - clean_psd_bundle       : Clean epoch PSD subset + channel metadata used instantaneously downstream.
    # - segment_idx / selected_segment / rejected_intervals
    # - band_preset_select / power_norm_select / psd_agg_select / selected_bands / bp_df
    # - topo_win_select / topo_scale_select / topo_time_slider / compare_stim_checkbox / topo_precomp
    # - tfr_chan_select / tfr_window_slider / tfr_fmax_slider / tfr_cmap_select / batch_run_button
    #
    #
    # Thanks for using my pipeline, don't hesitate to star it on GitHub if it has been useful to you!
    return


if __name__ == "__main__":
    app.run()
