"""Thread-safe session state, single-session cache, background scoring runner, and binary waveform encoder."""

from __future__ import annotations

import datetime
import html as html_lib
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from .eeg_io import (
    EMG_BINS_PER_EP,
    MAX_SMOOTH_PTS_PER_EP,
    SMOOTH_RESAMPLE_PTS,
    STAGE_ORDER,
    WAVE_PALETTE,
    align_proba_df,
    build_export_content,
    build_hypnogram_events,
    build_scored_epochs_df,
    build_sidecar_payload,
    classify_sleep_channels,
    compute_channel_defaults,
    compute_inter_rater_metrics,
    compute_wake_overrides,
    detect_vhdr_metadata,
    extract_recording_events,
    extract_vmrk_events,
    fast_resample_to_100,
    format_epoch_overlay,
    format_hms,
    get_recording_stem,
    normalize_export_fmt_key,
    normalize_stage,
    parse_scoring_upload,
    prepare_display_waveforms,
    read_raw_eeg,
    resolve_export_filename,
    resolve_preset_channels,
    scan_recordings,
)


def _default_start_dir() -> str:
    """Return universal default directory across Windows ('C:\\') and macOS/Linux ('~')."""
    if sys.platform == "win32":
        return "C:\\"
    return "~"


def _normalize_user_path(raw_str: str | None, fallback: str = ".") -> Path:
    """Strip surrounding copy-paste quotes and expand user '~' for cross-OS POSIX/Windows paths."""
    cleaned = str(raw_str or "").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        cleaned = fallback
    return Path(cleaned).expanduser()


class SessionManager:
    """Manages active recording metadata, caches, scoring jobs, edits, and exports."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs_lock = threading.Lock()
        default_dir = _default_start_dir()
        self.data_dir: str = default_dir
        self.output_dir: str = default_dir
        self.scanned: bool = False
        self.recording_catalog: dict[str, Path] = {}
        self.selected_recording: dict[str, object] | None = None

        self.data_cache: dict[str, object] = {
            "file_key": None,
            "sfreq": None,
            "n_times": 0,
            "meas_date": None,
            "all_ch_names": [],
            "channels": {},
            "yasa_passes": {},
            "wave_cache_key": None,
            "wave_cache": None,
            "disp_channels": {},
        }

        self.scored_bundle: dict[str, object] | None = None
        self.manual_edits: dict[str, dict[int, str]] = {}
        self.ma_flags: dict[str, set[int]] = {}
        self.ma_reviewed: dict[str, set[int]] = {}
        self.undo_history: dict[str, list[tuple[str, dict]]] = {}
        self.jobs: dict[str, dict[str, object]] = {}

    def _clear_data_cache_locked(self) -> None:
        self.data_cache["channels"].clear()  # type: ignore[union-attr]
        self.data_cache["yasa_passes"].clear()  # type: ignore[union-attr]
        self.data_cache["disp_channels"].clear()  # type: ignore[union-attr]
        self.data_cache["wave_cache_key"] = None
        self.data_cache["wave_cache"] = None

    def get_initial_state(self) -> dict[str, object]:
        with self._lock:
            has_bundle = self.scored_bundle is not None and not self.scored_bundle["base_epochs_df"].empty  # type: ignore[union-attr]
            return {
                "data_dir": self.data_dir,
                "output_dir": self.output_dir,
                "scanned": self.scanned,
                "recordings": list(self.recording_catalog.keys()),
                "selected_recording": self._serialize_selected_recording(self.selected_recording) if self.selected_recording else None,
                "has_session": has_bundle,
            }

    def scan_directories(self, data_dir_str: str, output_dir_str: str) -> dict[str, object]:
        data_dir = _normalize_user_path(data_dir_str, ".")
        output_dir = _normalize_user_path(output_dir_str, str(data_dir)) if str(output_dir_str or "").strip() else data_dir

        exists = data_dir.exists()
        catalog = scan_recordings(data_dir) if exists else {}

        warning = None
        if not exists:
            warning = f"Directory not found: {data_dir}"
        elif not catalog:
            warning = (
                "No supported EEG/PSG recordings (.vhdr, .edf, .bdf, .fif, .set, .gdf) "
                f"found in {data_dir}"
            )

        with self._lock:
            self.data_dir = str(data_dir)
            self.output_dir = str(output_dir)
            self.scanned = True
            self.recording_catalog = catalog
            if not catalog:
                self.selected_recording = None

        return {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "recordings": list(catalog.keys()),
            "count": len(catalog),
            "warning": warning,
        }

    def _serialize_selected_recording(self, rec: dict[str, object]) -> dict[str, object]:
        return {
            "recording_key": rec["recording_key"],
            "recording_id": rec["recording_id"],
            "recording_label": rec["recording_label"],
            "filename": Path(str(rec["eeg_path"])).name,
            "duration_sec": rec["duration_sec"],
            "duration_hms": format_hms(float(rec["duration_sec"])),  # type: ignore[arg-type]
            "n_epochs_est": int(float(rec["duration_sec"]) // 30.0),  # type: ignore[arg-type]
            "n_channels": len(rec["ch_names"]),  # type: ignore[arg-type]
            "sfreq_detected": rec["sfreq_detected"],
            "ch_names": rec["ch_names"],
            "marker_labels": rec["marker_labels"],
            "defaults": rec["defaults"],
        }

    def select_recording(self, rec_key: str) -> dict[str, object]:
        with self._lock:
            eeg_path = self.recording_catalog.get(rec_key)
            data_dir = Path(self.data_dir)
        if eeg_path is None:
            raise ValueError(f"Recording not found in catalog: {rec_key}")
        if not eeg_path.exists():
            raise FileNotFoundError(f"Recording not found: {eeg_path}")

        try:
            rel_rec = eeg_path.relative_to(data_dir)
            rec_id = rel_rec.as_posix()
            stem_rec = get_recording_stem(eeg_path)
            rec_label = stem_rec if rel_rec.parent == Path(".") else f"{rel_rec.parent.as_posix()}/{stem_rec}"
        except ValueError:
            rec_id = eeg_path.name
            rec_label = get_recording_stem(eeg_path)

        raw_hdr = read_raw_eeg(eeg_path, preload=False)
        ch_names = list(raw_hdr.ch_names)
        mne_sf = float(raw_hdr.info["sfreq"])
        duration_sec = float(raw_hdr.n_times / mne_sf) if mne_sf > 0 else 0.0
        if eeg_path.suffix.lower() == ".vhdr":
            vhdr_meta = detect_vhdr_metadata(eeg_path)
            detected_fs = int(vhdr_meta["sfreq"]) if vhdr_meta.get("sfreq") else int(round(mne_sf))
        else:
            detected_fs = int(round(mne_sf)) if mne_sf > 0 else 500
        meas_date = raw_hdr.info.get("meas_date")

        df_vmrk, events_path = extract_recording_events(eeg_path, raw_hdr, detected_fs)
        marker_labels: list[str] = []
        if not df_vmrk.empty:
            non_stage = df_vmrk[
                ~df_vmrk["Type"].str.contains("Sleep Stage|New Segment", case=False, na=False)
            ]
            marker_labels = sorted(
                str(m).strip()
                for m in non_stage["Description"].dropna().unique()
                if str(m).strip() != ""
            )
            if meas_date is None:
                new_segs = df_vmrk[df_vmrk["Type"].str.contains("New Segment", case=False, na=False)]
                if not new_segs.empty:
                    d_str = str(new_segs.iloc[0].get("Date", "")).strip()
                    if len(d_str) >= 14 and d_str[:14].isdigit():
                        try:
                            meas_date = datetime.datetime.strptime(d_str[:14], "%Y%m%d%H%M%S")
                        except Exception:
                            pass

        defaults = compute_channel_defaults(ch_names, marker_labels)
        rec_info: dict[str, object] = {
            "recording_key": rec_key,
            "eeg_path": eeg_path,
            "recording_id": rec_id,
            "recording_label": rec_label,
            "ch_names": ch_names,
            "duration_sec": duration_sec,
            "sfreq_detected": detected_fs,
            "meas_date": meas_date,
            "df_vmrk": df_vmrk,
            "marker_labels": marker_labels,
            "events_path": events_path,
            "defaults": defaults,
        }
        with self._lock:
            self.selected_recording = rec_info
        return self._serialize_selected_recording(rec_info)

    def ensure_display_channel(self, ch_name: str) -> dict[str, object] | None:
        """Lazily load, re-reference, filter, and cache a display channel at native sampling rate."""
        if not ch_name:
            return None
        with self._lock:
            bundle = self.scored_bundle
            if not bundle:
                return None
            refs = list(bundle.get("committed_refs") or [])  # type: ignore[arg-type]
            ref_key = tuple(sorted(refs))
            disp_sf = float(bundle.get("disp_sfreq") or 200.0)  # type: ignore[arg-type]
            overrides = bundle.get("committed_type_overrides") or {}
            ctype = str(
                overrides.get(ch_name)  # type: ignore[union-attr]
                or classify_sleep_channels([ch_name], ref_chs=refs).get(ch_name, "eeg")
            ).lower()
            fk = bundle.get("committed_file_key")
            ck = (ch_name, ctype, round(disp_sf, 3), ref_key)
            disp_cache: dict = self.data_cache.setdefault("disp_channels", {})  # type: ignore[assignment]
            if self.data_cache.get("file_key") == fk and ck in disp_cache:
                w1, y1 = disp_cache[ck]
                enc = max(float(y1), 300.0) if ctype in ("eeg", "eog") else max(float(y1), 1e-12)
                return {
                    "name": ch_name,
                    "type": ctype,
                    "unit": "(µV)" if ctype in ("eeg", "eog", "emg", "ecg") else "(raw)",
                    "color": WAVE_PALETTE.get(ctype, "#5c6b7a"),
                    "ylim": float(y1),
                    "encodeMax": float(enc),
                    "wave": w1,
                }

            base_chs = list(bundle.get("display_chs") or [])  # type: ignore[arg-type]
            if ch_name in base_chs:
                bi = base_chs.index(ch_name)
                w_uv_arr = bundle.get("wave_uv")
                ylims_arr = bundle.get("ch_ylims") or []
                types_arr = bundle.get("display_types") or []
                if w_uv_arr is not None and bi < len(w_uv_arr):  # type: ignore[arg-type]
                    b_type = str(types_arr[bi]).lower() if bi < len(types_arr) else ctype  # type: ignore[index]
                    b_ylim = float(ylims_arr[bi]) if bi < len(ylims_arr) else 100.0  # type: ignore[index]
                    w1 = w_uv_arr[bi]  # type: ignore[index]
                    if fk is not None and self.data_cache.get("file_key") == fk:
                        disp_cache[(ch_name, b_type, round(disp_sf, 3), ref_key)] = (w1, b_ylim)
                    enc = max(b_ylim, 300.0) if b_type in ("eeg", "eog") else max(b_ylim, 1e-12)
                    return {
                        "name": ch_name,
                        "type": b_type,
                        "unit": "(µV)" if b_type in ("eeg", "eog", "emg", "ecg") else "(raw)",
                        "color": WAVE_PALETTE.get(b_type, "#5c6b7a"),
                        "ylim": b_ylim,
                        "encodeMax": float(enc),
                        "wave": w1,
                    }

            eeg_path_str = bundle.get("committed_eeg_path")
            if not eeg_path_str:
                return None
            eeg_path = Path(str(eeg_path_str))
            if not eeg_path.exists():
                return None

            if self.data_cache.get("file_key") != fk or int(self.data_cache.get("n_times") or 0) <= 0:  # type: ignore[arg-type]
                self._clear_data_cache_locked()
                hdr = read_raw_eeg(eeg_path, preload=False)
                self.data_cache["file_key"] = fk
                self.data_cache["sfreq"] = float(hdr.info["sfreq"])
                self.data_cache["n_times"] = int(hdr.n_times)
                self.data_cache["meas_date"] = hdr.info.get("meas_date")
                self.data_cache["all_ch_names"] = list(hdr.ch_names)

            raw_sf = float(bundle.get("committed_raw_sfreq") or self.data_cache.get("sfreq") or disp_sf)  # type: ignore[arg-type]
            n_times = int(self.data_cache.get("n_times") or 0)  # type: ignore[arg-type]
            all_names = list(self.data_cache.get("all_ch_names") or [])  # type: ignore[arg-type]
            raw_channels: dict = self.data_cache["channels"]  # type: ignore[assignment]
            if n_times <= 0 or (ch_name not in all_names and ch_name not in raw_channels):
                return None

            need_raw = [ch_name] + (refs if ctype == "eeg" else [])
            need_raw = list(dict.fromkeys(c for c in need_raw if not all_names or c in all_names))
            miss = [c for c in need_raw if c not in raw_channels]
            if miss:
                raw = read_raw_eeg(eeg_path, preload=False)
                buf = np.empty((len(miss), n_times), dtype=np.float32)
                chunk = max(1, int(round(600.0 * raw_sf)))
                for s in range(0, n_times, chunk):
                    e = min(s + chunk, n_times)
                    buf[:, s:e] = raw.get_data(picks=miss, start=s, stop=e).astype(np.float32, copy=False)
                for mi, mch in enumerate(miss):
                    raw_channels[mch] = buf[mi]

            if ch_name not in raw_channels:
                return None

            src = raw_channels[ch_name]
            data_1 = np.empty((1, n_times), dtype=np.float32)
            loaded_refs = [rc for rc in refs if rc in raw_channels]
            if ctype == "eeg" and loaded_refs:
                ref_acc = np.zeros(n_times, dtype=np.float64)
                for rc in loaded_refs:
                    ref_acc += raw_channels[rc]
                ref_sig = (ref_acc / float(len(loaded_refs))).astype(np.float32)
                np.subtract(src, ref_sig, out=data_1[0])
            else:
                data_1[0] = src

            if abs(disp_sf - raw_sf) > 0.05 and raw_sf > 0.0:
                from fractions import Fraction

                if abs(disp_sf - round(disp_sf)) < 1e-3 and abs(raw_sf - round(raw_sf)) < 1e-3:
                    frac = Fraction(int(round(disp_sf)), int(round(raw_sf)))
                else:
                    frac = Fraction(disp_sf / raw_sf).limit_denominator(1000)
                data_1 = scipy_signal.resample_poly(data_1, frac.numerator, frac.denominator, axis=-1).astype(
                    np.float32, copy=False
                )

            w_arr, _, _, _, y_arr = prepare_display_waveforms(data_1, disp_sf, [ctype], ch_names=[ch_name])
            w1 = w_arr[0]
            y1 = float(y_arr[0])
            disp_cache[ck] = (w1, y1)
            enc = max(y1, 300.0) if ctype in ("eeg", "eog") else max(y1, 1e-12)
            return {
                "name": ch_name,
                "type": ctype,
                "unit": "(µV)" if ctype in ("eeg", "eog", "emg", "ecg") else "(raw)",
                "color": WAVE_PALETTE.get(ctype, "#5c6b7a"),
                "ylim": y1,
                "encodeMax": float(enc),
                "wave": w1,
            }

    def start_scoring_job(
        self,
        config: dict[str, object],
        resume_bytes: bytes | None = None,
        resume_filename: str | None = None,
        scorer1_bytes: bytes | None = None,
        scorer1_filename: str | None = None,
        scorer2_bytes: bytes | None = None,
        scorer2_filename: str | None = None,
    ) -> str:
        with self._lock:
            rec = dict(self.selected_recording) if self.selected_recording else None
            output_dir = _normalize_user_path(self.output_dir, ".")

        mode = str(config.get("mode") or "auto").lower()
        is_consensus_run = mode == "consensus"
        consensus_with_yasa = is_consensus_run and bool(config.get("consensus_with_yasa", False))
        is_resume_run = mode == "resume"
        resume_with_yasa = is_resume_run and bool(config.get("resume_with_yasa", False))
        is_yasa_run = (mode == "auto") or resume_with_yasa or consensus_with_yasa
        is_manual_channels = (
            (mode == "manual")
            or (is_resume_run and not resume_with_yasa)
            or (is_consensus_run and not consensus_with_yasa)
        )

        if rec is None:
            raise ValueError("No EEG/PSG recording selected.")
        if is_consensus_run and (
            not scorer1_bytes or not scorer1_filename or not scorer2_bytes or not scorer2_filename
        ):
            raise ValueError(
                "Please select both Scorer 1 and Scorer 2 scoring files before starting Consensus Mode."
            )
        if is_resume_run and (not resume_bytes or not resume_filename):
            raise ValueError(
                "Please select a saved scoring file (.Markers, .txt, .tsv, .csv, .vmrk, or .onera.json) "
                "before clicking Resume Saved Scoring."
            )
        if is_yasa_run and not config.get("primary_eeg"):
            raise ValueError("No primary EEG channel selected.")

        if is_consensus_run:
            bar_title = (
                "Comparing Two Scorers + YASA 3rd Opinion"
                if consensus_with_yasa
                else "Comparing Two Scorers (Consensus Mode)"
            )
        elif is_resume_run:
            bar_title = "Resuming Saved Scoring with YASA" if resume_with_yasa else "Resuming Saved Sleep Scoring"
        else:
            bar_title = "Preparing Manual Sleep Scoring" if is_manual_channels else "Running Automated Sleep Scoring"

        job_id = uuid.uuid4().hex[:12]
        with self._jobs_lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "pct": 0,
                "title": bar_title,
                "subtitle": "Checking recording & cache...",
                "error": None,
            }

        thread = threading.Thread(
            target=self._run_scoring_job,
            args=(
                job_id,
                config,
                rec,
                output_dir,
                resume_bytes,
                resume_filename,
                scorer1_bytes,
                scorer1_filename,
                scorer2_bytes,
                scorer2_filename,
            ),
            daemon=True,
        )
        thread.start()
        return job_id

    def get_job_status(self, job_id: str) -> dict[str, object] | None:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            return dict(job) if job is not None else None

    def _update_job(self, job_id: str, **fields: object) -> None:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def _run_scoring_job(
        self,
        job_id: str,
        config: dict[str, object],
        rec: dict[str, object],
        output_dir: Path,
        resume_bytes: bytes | None,
        resume_filename: str | None,
        scorer1_bytes: bytes | None = None,
        scorer1_filename: str | None = None,
        scorer2_bytes: bytes | None = None,
        scorer2_filename: str | None = None,
    ) -> None:
        t_init_start = time.perf_counter()
        prog = [0]

        def _advance_bar(target_pct: int, subtitle: str) -> None:
            tgt = max(prog[0], min(100, int(target_pct)))
            prog[0] = tgt
            self._update_job(job_id, pct=tgt, subtitle=subtitle)

        try:
            mode = str(config.get("mode") or "auto").lower()
            is_consensus_run = mode == "consensus"
            consensus_with_yasa = is_consensus_run and bool(config.get("consensus_with_yasa", False))
            is_resume_run = mode == "resume"
            resume_with_yasa = is_resume_run and bool(config.get("resume_with_yasa", False))
            is_yasa_run = (mode == "auto") or resume_with_yasa or consensus_with_yasa
            is_manual_run = (mode == "manual") or (is_resume_run and not resume_with_yasa)
            scoring_mode = "consensus" if is_consensus_run else ("manual" if is_manual_run else "auto")

            eeg_path: Path = rec["eeg_path"]  # type: ignore[assignment]
            recording_id = str(rec["recording_id"])
            recording_label = str(rec["recording_label"])
            df_vmrk: pd.DataFrame = rec["df_vmrk"]  # type: ignore[assignment]
            meas_date = rec.get("meas_date")

            with self._lock:
                # 1. Single-session header & chunked disk streaming (0% -> 50%)
                t0 = time.perf_counter()
                rec_stat = eeg_path.stat()
                file_key = f"{eeg_path}|{rec_stat.st_mtime_ns}|{rec_stat.st_size}"
                if self.data_cache["file_key"] != file_key:
                    self._clear_data_cache_locked()
                    raw_hdr = read_raw_eeg(eeg_path, preload=False)
                    self.data_cache["file_key"] = file_key
                    self.data_cache["sfreq"] = float(raw_hdr.info["sfreq"])
                    self.data_cache["n_times"] = int(raw_hdr.n_times)
                    self.data_cache["meas_date"] = raw_hdr.info.get("meas_date")
                    self.data_cache["all_ch_names"] = list(raw_hdr.ch_names)

                raw_sf = float(self.data_cache["sfreq"])  # type: ignore[arg-type]
                n_times = int(self.data_cache["n_times"])  # type: ignore[arg-type]
                raw_meas_date = self.data_cache["meas_date"]
                all_ch_names = list(self.data_cache["all_ch_names"])  # type: ignore[arg-type]
                fs_val = config.get("fs")
                fs_marker = float(fs_val) if fs_val else raw_sf  # type: ignore[arg-type]
                ref_chs = [c for c in (config.get("refs") or []) if c in all_ch_names]  # type: ignore[union-attr]
                ref_key = tuple(sorted(ref_chs))
                eog_ch: str | None = None
                emg_ch: str | None = None

                if not is_yasa_run:
                    raw_manual_chs = config.get("manual_chs") or config.get("visual_chs") or []
                    display_chs = [
                        c
                        for c in raw_manual_chs  # type: ignore[union-attr]
                        if c in all_ch_names and c not in ref_chs
                    ]
                    if not display_chs:
                        display_chs = [c for c in all_ch_names if c not in ref_chs][:4] or list(all_ch_names[:4])
                    if not display_chs:
                        raise ValueError("No valid display channels selected.")
                    ch_order_rank = {"eeg": 0, "eog": 1, "emg": 2, "ecg": 3, "misc": 4}
                    man_types_preview = classify_sleep_channels(display_chs, ref_chs=ref_chs)
                    display_chs = sorted(
                        list(dict.fromkeys(display_chs)),
                        key=lambda c: ch_order_rank.get(man_types_preview.get(c, "eeg"), 4),
                    )
                    pick_chs = list(dict.fromkeys(display_chs + ref_chs))
                else:
                    primary_eeg = str(config.get("primary_eeg") or "")
                    raw_eog = config.get("eog")
                    raw_emg = config.get("emg")
                    eog_ch = None if raw_eog in (None, "", "(None)") else str(raw_eog)
                    emg_ch = None if raw_emg in (None, "", "(None)") else str(raw_emg)
                    if eog_ch not in all_ch_names:
                        eog_ch = None
                    if emg_ch not in all_ch_names:
                        emg_ch = None

                    if bool(config.get("multi_electrode", False)):
                        active_eegs = resolve_preset_channels(
                            str(config.get("electrode_preset") or ""),
                            all_ch_names,
                            primary_eeg,
                            custom_eegs=list(config.get("custom_eegs") or []),  # type: ignore[arg-type]
                        )
                    else:
                        active_eegs = [primary_eeg]

                    active_eegs = [c for c in active_eegs if c not in ref_chs and c in all_ch_names]
                    if not active_eegs:
                        raise ValueError("No valid EEG channels available after excluding references.")

                    yasa_chs = list(
                        dict.fromkeys(
                            c for c in active_eegs + [eog_ch, emg_ch] if c and c in all_ch_names
                        )
                    )
                    extra_vis_chs = [
                        c
                        for c in (config.get("visual_chs") or [])  # type: ignore[union-attr]
                        if c in all_ch_names and c not in ref_chs and c not in yasa_chs
                    ]
                    needed_chs = list(dict.fromkeys(yasa_chs + extra_vis_chs))
                    pick_chs = list(dict.fromkeys(needed_chs + ref_chs))

                raw_channels: dict = self.data_cache["channels"]  # type: ignore[assignment]
                missing_chs = [c for c in pick_chs if c not in raw_channels]
                if missing_chs:
                    raw = read_raw_eeg(eeg_path, preload=False)
                    buf = np.empty((len(missing_chs), n_times), dtype=np.float32)
                    chunk_samps = max(1, int(round(600.0 * raw_sf)))
                    total_mins = max(1, int(round(n_times / (raw_sf * 60.0))))
                    for s in range(0, n_times, chunk_samps):
                        e = min(s + chunk_samps, n_times)
                        buf[:, s:e] = raw.get_data(picks=missing_chs, start=s, stop=e).astype(
                            np.float32, copy=False
                        )
                        pct = int(round(50.0 * e / max(1, n_times)))
                        if pct - prog[0] >= 5 or e >= n_times:
                            cur_min = int(round(e / (raw_sf * 60.0)))
                            _advance_bar(pct, f"Reading {len(missing_chs)} ch ({cur_min}/{total_mins} min)...")
                    for mi, mch in enumerate(missing_chs):
                        raw_channels[mch] = buf[mi]
                else:
                    _advance_bar(50, "Loaded raw channels from cache")
                t_load = time.perf_counter() - t0

                def _assemble_referenced(
                    ch_list: list[str], type_overrides: dict[str, str] | None = None
                ) -> tuple[np.ndarray, list[str]]:
                    types_map = classify_sleep_channels(ch_list, ref_chs=ref_chs)
                    if type_overrides:
                        types_map.update(type_overrides)
                    ch_types_out = [types_map.get(c, "eeg") for c in ch_list]
                    ref_sig = None
                    if ref_chs:
                        ref_acc = np.zeros(n_times, dtype=np.float64)
                        for rc in ref_chs:
                            ref_acc += raw_channels[rc]
                        ref_sig = (ref_acc / float(len(ref_chs))).astype(np.float32)
                    out = np.empty((len(ch_list), n_times), dtype=np.float32)
                    for ci, (ch, ct) in enumerate(zip(ch_list, ch_types_out)):
                        src = raw_channels[ch]
                        if ref_sig is not None and ct == "eeg":
                            np.subtract(src, ref_sig, out=out[ci])
                        else:
                            out[ci] = src
                    return out, ch_types_out

                def _on_filt_ch(ci: int, n_tot: int, cname: str) -> None:
                    _advance_bar(
                        90 + int(round(10.0 * (ci + 1) / max(1, n_tot))),
                        f"Filtering {cname} ({ci + 1}/{n_tot})...",
                    )

                thresh = float(config.get("confidence_thresh") or 60.0)
                excl_wake = bool(config.get("exclude_forced_wake", True))
                lightsoff_label = str(config.get("lightsoff") or "")
                ignore_pairs_raw = config.get("ignore_pairs") or []
                ignore_pairs: list[tuple[str, str]] = [
                    (str(p[0]), str(p[1])) for p in ignore_pairs_raw if isinstance(p, (list, tuple)) and len(p) >= 2  # type: ignore[union-attr]
                ]

                auto_type_overrides: dict[str, str] = {}
                resume_meta: dict[str, object] | None = None
                resume_base_ma: list[int] = []
                resume_init_edits: dict[int, str] | None = None
                consensus_meta: dict[str, object] | None = None

                # Compute waveforms + optional YASA predictions
                raw_yasa_stages: list[str] = []
                agreement_pcts: list[float] = []
                confidence_pcts: list[float] = []
                runner_ups: list[str] = []
                margin_pcts: list[float] = []
                vote_summaries: list[str] = []
                avg_proba: np.ndarray | None = None

                if not is_yasa_run:
                    t_res0 = time.perf_counter()
                    n_epochs = int(n_times // int(round(30.0 * raw_sf)))
                    if n_epochs <= 0:
                        raise ValueError("Recording is shorter than one 30-second epoch.")

                    wave_cache_key = ("manual", tuple(display_chs), ref_key)
                    wave_cache_hit = (
                        self.data_cache["wave_cache_key"] == wave_cache_key
                        and self.data_cache["wave_cache"] is not None
                    )
                    disp_data_v = None
                    disp_types_in = None
                    if not wave_cache_hit:
                        disp_data_v, disp_types_in = _assemble_referenced(display_chs)
                    _advance_bar(
                        65, "Re-referenced channels" if not wave_cache_hit else "Reusing cached waveforms"
                    )
                    t_resample = time.perf_counter() - t_res0
                    _advance_bar(
                        90,
                        "Comparing scorer files..." if is_consensus_run else "Skipping YASA (Manual mode)",
                    )
                    t_yasa = 0.0

                    t_flt0 = time.perf_counter()
                    if wave_cache_hit:
                        wave_uv, disp_sfreq, ep_samps, disp_types, ch_ylims = self.data_cache["wave_cache"]  # type: ignore[misc]
                        _advance_bar(100, "Display waveforms ready (cached)")
                    else:
                        wave_uv, disp_sfreq, ep_samps, disp_types, ch_ylims = prepare_display_waveforms(
                            disp_data_v, raw_sf, disp_types_in, ch_names=display_chs, on_channel_done=_on_filt_ch  # type: ignore[arg-type]
                        )
                        self.data_cache["wave_cache_key"] = wave_cache_key
                        self.data_cache["wave_cache"] = (wave_uv, disp_sfreq, ep_samps, disp_types, ch_ylims)
                    t_filter = time.perf_counter() - t_flt0
                    active_eegs = [c for c, t in zip(display_chs, disp_types) if t == "eeg"] or list(display_chs)
                else:
                    auto_type_overrides = {c: "eeg" for c in active_eegs}
                    if eog_ch:
                        auto_type_overrides[eog_ch] = "eog"
                    if emg_ch:
                        auto_type_overrides[emg_ch] = "emg"

                    needed_types_map = classify_sleep_channels(needed_chs, ref_chs=ref_chs)
                    needed_types_map.update(auto_type_overrides)
                    needed_types_preview = [needed_types_map.get(c, "eeg") for c in needed_chs]
                    ch_order_rank = {"eeg": 0, "eog": 1, "emg": 2, "ecg": 3, "misc": 4}
                    disp_picks = sorted(
                        range(len(needed_chs)),
                        key=lambda i: ch_order_rank.get(needed_types_preview[i], 4),
                    )
                    if not disp_picks:
                        disp_picks = list(range(min(3, len(needed_chs))))
                    display_chs = [needed_chs[i] for i in disp_picks]

                    disp_types_preview = tuple(needed_types_preview[i] for i in disp_picks)
                    wave_cache_key = ("auto", tuple(display_chs), disp_types_preview, ref_key)
                    wave_cache_hit = (
                        self.data_cache["wave_cache_key"] == wave_cache_key
                        and self.data_cache["wave_cache"] is not None
                    )
                    yasa_passes: dict = self.data_cache["yasa_passes"]  # type: ignore[assignment]
                    missing_yasa_eegs = [
                        e for e in active_eegs if (e, eog_ch, emg_ch, ref_key) not in yasa_passes
                    ]

                    t_res0 = time.perf_counter()
                    disp_data_v = None
                    needed_types = needed_types_preview
                    raw_100 = None
                    if not wave_cache_hit or missing_yasa_eegs:
                        needed_data_v, needed_types = _assemble_referenced(needed_chs, type_overrides=auto_type_overrides)
                        if not wave_cache_hit:
                            disp_data_v = needed_data_v[disp_picks]
                        if missing_yasa_eegs:
                            _advance_bar(58, "Preparing 100 Hz copy for YASA...")
                            n_yasa = len(yasa_chs)
                            data_100, _ = fast_resample_to_100(needed_data_v[:n_yasa], raw_sf)
                            import mne

                            info_100 = mne.create_info(
                                ch_names=yasa_chs, sfreq=100.0, ch_types=needed_types[:n_yasa]
                            )
                            if raw_meas_date is not None:
                                info_100.set_meas_date(raw_meas_date)  # type: ignore[arg-type]
                            raw_100 = mne.io.RawArray(
                                data_100.astype(np.float64, copy=False), info_100, verbose="ERROR"
                            )
                            del data_100
                        del needed_data_v
                    _advance_bar(
                        65,
                        "Prepared 100 Hz copy for YASA (native display rate)"
                        if (not wave_cache_hit or missing_yasa_eegs)
                        else "Reusing cached YASA & waveforms",
                    )
                    t_resample = time.perf_counter() - t_res0

                    t_yasa0 = time.perf_counter()
                    pass_probas = []
                    pass_preds = []
                    n_passes_tot = len(active_eegs)
                    if missing_yasa_eegs:
                        import yasa
                    for pi, eeg_ch in enumerate(active_eegs):
                        y_key = (eeg_ch, eog_ch, emg_ch, ref_key)
                        cached_pass = yasa_passes.get(y_key)
                        if cached_pass is not None:
                            p_arr, p_list = cached_pass
                            _advance_bar(
                                65 + int(round(25.0 * (pi + 1) / n_passes_tot)),
                                f"YASA pass {pi + 1}/{n_passes_tot} ({eeg_ch}) [cached]",
                            )
                        else:
                            _advance_bar(prog[0], f"YASA scoring {pi + 1}/{n_passes_tot}: {eeg_ch}...")
                            sls = yasa.SleepStaging(raw_100, eeg_name=eeg_ch, eog_name=eog_ch, emg_name=emg_ch)
                            hyp = sls.predict()
                            p_arr = align_proba_df(hyp.proba)
                            p_list = [normalize_stage(s) for s in hyp.hypno.tolist()]
                            yasa_passes[y_key] = (p_arr, p_list)
                            _advance_bar(
                                65 + int(round(25.0 * (pi + 1) / n_passes_tot)),
                                f"Scored YASA pass {pi + 1}/{n_passes_tot}: {eeg_ch}",
                            )
                        pass_probas.append(p_arr)
                        pass_preds.append(p_list)
                    if raw_100 is not None:
                        del raw_100
                    t_yasa = time.perf_counter() - t_yasa0

                    n_epochs = min(p.shape[0] for p in pass_probas)
                    proba_stack = np.stack([p[:n_epochs] for p in pass_probas], axis=0)
                    avg_proba = np.mean(proba_stack, axis=0)
                    n_passes = proba_stack.shape[0]

                    for ep in range(n_epochs):
                        ep_probs = avg_proba[ep]
                        votes = [pass_preds[m][ep] for m in range(n_passes)]
                        chosen = STAGE_ORDER[int(np.argmax(ep_probs))]
                        chosen_idx = STAGE_ORDER.index(chosen)
                        agree = 100.0 * (votes.count(chosen) / float(n_passes))
                        conf = 100.0 * float(ep_probs[chosen_idx])
                        sorted_idxs = np.argsort(ep_probs)[::-1]
                        top2_idx = sorted_idxs[1] if sorted_idxs[0] == chosen_idx else sorted_idxs[0]
                        runner = STAGE_ORDER[int(top2_idx)]
                        margin = 100.0 * float(ep_probs[chosen_idx] - ep_probs[top2_idx])
                        vote_counts = pd.Series(votes).value_counts()
                        vote_str = " | ".join(f"{k}:{v}" for k, v in vote_counts.items())

                        raw_yasa_stages.append(chosen)
                        agreement_pcts.append(round(agree, 1))
                        confidence_pcts.append(round(conf, 1))
                        runner_ups.append(runner)
                        margin_pcts.append(round(margin, 1))
                        vote_summaries.append(vote_str)

                    t_flt0 = time.perf_counter()
                    if wave_cache_hit:
                        wave_uv, disp_sfreq, ep_samps, disp_types, ch_ylims = self.data_cache["wave_cache"]  # type: ignore[misc]
                        _advance_bar(100, "Display waveforms ready (cached)")
                    else:
                        disp_types_in = [needed_types[i] for i in disp_picks]
                        wave_uv, disp_sfreq, ep_samps, disp_types, ch_ylims = prepare_display_waveforms(
                            disp_data_v, raw_sf, disp_types_in, ch_names=display_chs, on_channel_done=_on_filt_ch  # type: ignore[arg-type]
                        )
                        del disp_data_v
                        self.data_cache["wave_cache_key"] = wave_cache_key
                        self.data_cache["wave_cache"] = (wave_uv, disp_sfreq, ep_samps, disp_types, ch_ylims)
                    t_filter = time.perf_counter() - t_flt0

                epoch_len_samples, pos_starts, override_sources, wake_override_df = compute_wake_overrides(
                    df_vmrk, n_epochs, fs_marker, lightsoff_label, ignore_pairs
                )
                epoch_events, plot_events = extract_vmrk_events(df_vmrk, n_epochs, fs_marker, meas_date)  # type: ignore[arg-type]
                time_starts = [format_hms(ep * 30) for ep in range(n_epochs)]
                time_ends = [format_hms((ep + 1) * 30) for ep in range(n_epochs)]

                if is_consensus_run:
                    s1_stages, s1_ma, s1_meta, _, _ = parse_scoring_upload(
                        str(scorer1_filename),
                        scorer1_bytes or b"",
                        n_epochs,
                        fs_marker,
                        output_dir,
                        eeg_path,
                        lookup_sidecar=False,
                    )
                    s2_stages, s2_ma, s2_meta, _, _ = parse_scoring_upload(
                        str(scorer2_filename),
                        scorer2_bytes or b"",
                        n_epochs,
                        fs_marker,
                        output_dir,
                        eeg_path,
                        lookup_sidecar=False,
                    )
                    s1_label = str(config.get("scorer1_label") or "").strip()[:32] or "Scorer 1"
                    s2_label = str(config.get("scorer2_label") or "").strip()[:32] or "Scorer 2"
                    flag_ma_diffs = bool(config.get("flag_ma_diffs", False))
                    auto_resolve_yasa = bool(config.get("auto_resolve_yasa", False)) and consensus_with_yasa
                    irr = compute_inter_rater_metrics(s1_stages, s2_stages, s1_ma, s2_ma)

                    resume_base_ma = sorted(s1_ma | s2_ma)
                    auto_stages: list[str] = []
                    is_flagged: list[bool] = []
                    flag_reasons: list[str] = []
                    base_status: list[str] = []
                    base_hover: list[str] = []
                    stage_diff_list: list[bool] = []
                    ma_diff_list: list[bool] = []
                    yasa_resolved_list: list[bool] = []
                    s1_ma_list: list[bool] = []
                    s2_ma_list: list[bool] = []

                    for i in range(n_epochs):
                        s1_st = s1_stages[i]
                        s2_st = s2_stages[i]
                        s1_has_ma = i in s1_ma
                        s2_has_ma = i in s2_ma
                        has_ma_i = s1_has_ma or s2_has_ma
                        st_diff = s1_st != s2_st
                        ma_diff = s1_has_ma != s2_has_ma
                        s1_ma_list.append(s1_has_ma)
                        s2_ma_list.append(s2_has_ma)
                        stage_diff_list.append(st_diff)
                        ma_diff_list.append(ma_diff)

                        y_st = raw_yasa_stages[i] if consensus_with_yasa else ""
                        y_conf = float(confidence_pcts[i]) if consensus_with_yasa else float("nan")
                        y_tie_break = False

                        if not st_diff:
                            init_st = s1_st
                        elif (
                            auto_resolve_yasa
                            and np.isfinite(y_conf)
                            and y_conf >= thresh
                            and y_st in (s1_st, s2_st)
                            and y_st != "UNS"
                        ):
                            init_st = y_st
                            y_tie_break = True
                        else:
                            init_st = s1_st

                        yasa_resolved_list.append(y_tie_break)
                        flagged_i = (st_diff and not y_tie_break) or (flag_ma_diffs and ma_diff)
                        reasons_c: list[str] = []
                        if st_diff and y_tie_break:
                            win_lbl = s1_label if init_st == s1_st else s2_label
                            st_lbl = "YASA Tie-Break"
                            reasons_c.append(f"YASA Tie-Break → {win_lbl} ({y_conf:.1f}%)")
                            if flag_ma_diffs and ma_diff:
                                reasons_c.append("+MA diff")
                        elif st_diff:
                            st_lbl = "Disagreement"
                            reasons_c.append(f"{s1_label}: {s1_st} vs {s2_label}: {s2_st}")
                            if consensus_with_yasa and np.isfinite(y_conf):
                                if y_conf < thresh:
                                    reasons_c.append(f"YASA: {y_st} ({y_conf:.1f}% < {thresh:.0f}%)")
                                elif y_st not in (s1_st, s2_st):
                                    reasons_c.append(f"YASA disagrees: {y_st} ({y_conf:.1f}%)")
                                else:
                                    reasons_c.append(f"YASA: {y_st} ({y_conf:.1f}%)")
                            if ma_diff:
                                reasons_c.append("+MA diff")
                        elif flag_ma_diffs and ma_diff:
                            st_lbl = "+MA Diff"
                            ma_holder = s1_label if s1_has_ma else s2_label
                            reasons_c.append(f"Stage Agreed ({s1_st}); +MA by {ma_holder}")
                        else:
                            st_lbl = "Agreed"
                            reasons_c.append(f"Agreed ({s1_st})")
                            if ma_diff:
                                ma_holder = s1_label if s1_has_ma else s2_label
                                reasons_c.append(f"+MA kept ({ma_holder})")

                        if override_sources[i]:
                            reasons_c.append(override_sources[i])
                        reason_str = "; ".join(reasons_c)
                        _, hv_html = format_epoch_overlay(
                            idx=i,
                            auto_stage=init_st,
                            final_stage=init_st,
                            is_manual_edit=False,
                            has_ma=False,
                            base_status=st_lbl,
                            base_hover="",
                            time_start=time_starts[i],
                            time_end=time_ends[i],
                            conf_pct=y_conf,
                            agr_pct=float(agreement_pcts[i]) if consensus_with_yasa else float("nan"),
                            flag_reason=reason_str,
                            is_manual_mode=False,
                            is_consensus_mode=True,
                            s1_label=s1_label,
                            s2_label=s2_label,
                            s1_stage=s1_st,
                            s2_stage=s2_st,
                            s1_ma=s1_has_ma,
                            s2_ma=s2_has_ma,
                            has_yasa=consensus_with_yasa,
                            yasa_stage=y_st,
                        )
                        auto_stages.append(init_st)
                        is_flagged.append(flagged_i)
                        flag_reasons.append(reason_str)
                        base_status.append(st_lbl)
                        base_hover.append(hv_html)

                    nan_col = np.full(n_epochs, np.nan, dtype=float)
                    base_df = pd.DataFrame({
                        "Epoch": np.arange(n_epochs, dtype=int),
                        "Epoch_Num": np.arange(1, n_epochs + 1, dtype=int),
                        "Time_Start": time_starts,
                        "Time_End": time_ends,
                        "Position_Sample": pos_starts,
                        "Length_Samples": epoch_len_samples,
                        "Raw_YASA_Stage": raw_yasa_stages if consensus_with_yasa else [""] * n_epochs,
                        "Auto_Stage": auto_stages,
                        "Override_Source": override_sources,
                        "Confidence_Pct": confidence_pcts if consensus_with_yasa else nan_col,
                        "Agreement_Pct": agreement_pcts if consensus_with_yasa else nan_col.copy(),
                        "RunnerUp_Stage": runner_ups if consensus_with_yasa else [""] * n_epochs,
                        "Margin_Pct": margin_pcts if consensus_with_yasa else nan_col.copy(),
                        "Prob_W": np.round(avg_proba[:, 0] * 100.0, 1) if avg_proba is not None else nan_col.copy(),
                        "Prob_N1": np.round(avg_proba[:, 1] * 100.0, 1) if avg_proba is not None else nan_col.copy(),
                        "Prob_N2": np.round(avg_proba[:, 2] * 100.0, 1) if avg_proba is not None else nan_col.copy(),
                        "Prob_N3": np.round(avg_proba[:, 3] * 100.0, 1) if avg_proba is not None else nan_col.copy(),
                        "Prob_REM": np.round(avg_proba[:, 4] * 100.0, 1) if avg_proba is not None else nan_col.copy(),
                        "Pass_Votes": vote_summaries if consensus_with_yasa else [""] * n_epochs,
                        "Scorer1_Stage": s1_stages,
                        "Scorer2_Stage": s2_stages,
                        "Scorer1_MA": s1_ma_list,
                        "Scorer2_MA": s2_ma_list,
                        "Stage_Diff": stage_diff_list,
                        "MA_Diff": ma_diff_list,
                        "YASA_Resolved": yasa_resolved_list,
                        "Is_Flagged": is_flagged,
                        "Flag_Reason": flag_reasons,
                        "Base_Status": base_status,
                        "Base_Hover": base_hover,
                    })
                    consensus_meta = {
                        "scorer1_label": s1_label,
                        "scorer2_label": s2_label,
                        "scorer1_file": s1_meta,
                        "scorer2_file": s2_meta,
                        "flag_ma_diffs": flag_ma_diffs,
                        "has_yasa": consensus_with_yasa,
                        "auto_resolve_yasa": auto_resolve_yasa,
                        "n_yasa_resolved": sum(1 for yr in yasa_resolved_list if yr),
                        "irr": irr,
                        "stage_diff_indices": [i for i, d in enumerate(stage_diff_list) if d],
                        "ma_diff_indices": [i for i, d in enumerate(ma_diff_list) if d],
                        "yasa_resolved_indices": [i for i, r in enumerate(yasa_resolved_list) if r],
                    }
                    init_active = next((i for i, f in enumerate(is_flagged) if f), 0)
                    init_selected = [init_active]
                elif is_manual_run:
                    if is_resume_run:
                        imp_stages, imp_ma, resume_meta, _, _ = parse_scoring_upload(
                            str(resume_filename), resume_bytes or b"", n_epochs, fs_marker, output_dir, eeg_path
                        )
                        auto_stages = list(imp_stages)
                        resume_base_ma = sorted(imp_ma)
                        resume_init_edits = {}
                    else:
                        auto_stages = ["W" if override_sources[i] else "UNS" for i in range(n_epochs)]

                    is_flagged = [False] * n_epochs
                    flag_reasons = []
                    base_status = []
                    base_hover = []
                    for i in range(n_epochs):
                        ov_src = override_sources[i]
                        if is_resume_run:
                            if auto_stages[i] != "UNS":
                                st_lbl = "Imported"
                                reason_str = f"Imported ({ov_src})" if ov_src else "Imported"
                            else:
                                st_lbl = "Unscored"
                                reason_str = f"Unscored ({ov_src})" if ov_src else "Unscored"
                        else:
                            is_forced = bool(ov_src)
                            st_lbl = "Wake Override" if is_forced else "Unscored"
                            reason_str = f"Forced Wake ({ov_src})" if is_forced else "Unscored"
                        flag_reasons.append(reason_str)
                        base_status.append(st_lbl)
                        base_hover.append(
                            f"<b>Ep #{i + 1}</b> ({time_starts[i]}–{time_ends[i]})<br>"
                            f"Stage: <b>{auto_stages[i]}</b> ({html_lib.escape(str(st_lbl))})<br>"
                            f"Note: {html_lib.escape(str(reason_str))}"
                        )

                    nan_col = np.full(n_epochs, np.nan, dtype=float)
                    base_df = pd.DataFrame({
                        "Epoch": np.arange(n_epochs, dtype=int),
                        "Epoch_Num": np.arange(1, n_epochs + 1, dtype=int),
                        "Time_Start": time_starts,
                        "Time_End": time_ends,
                        "Position_Sample": pos_starts,
                        "Length_Samples": epoch_len_samples,
                        "Raw_YASA_Stage": auto_stages,
                        "Auto_Stage": auto_stages,
                        "Override_Source": override_sources,
                        "Confidence_Pct": nan_col,
                        "Agreement_Pct": nan_col.copy(),
                        "RunnerUp_Stage": [""] * n_epochs,
                        "Margin_Pct": nan_col.copy(),
                        "Prob_W": nan_col.copy(),
                        "Prob_N1": nan_col.copy(),
                        "Prob_N2": nan_col.copy(),
                        "Prob_N3": nan_col.copy(),
                        "Prob_REM": nan_col.copy(),
                        "Pass_Votes": [""] * n_epochs,
                        "Is_Flagged": is_flagged,
                        "Flag_Reason": flag_reasons,
                        "Base_Status": base_status,
                        "Base_Hover": base_hover,
                    })

                    init_active = next((i for i, st in enumerate(auto_stages) if st == "UNS"), 0)
                    init_selected = [init_active]
                else:
                    auto_stages = ["W" if override_sources[ep] else raw_yasa_stages[ep] for ep in range(n_epochs)]
                    flag_reasons = []
                    is_flagged = []
                    base_status = []
                    base_hover = []

                    for i in range(n_epochs):
                        ov_src = override_sources[i]
                        is_forced = bool(ov_src)
                        conf_i = confidence_pcts[i]
                        agr_i = agreement_pcts[i]
                        auto_i = auto_stages[i]
                        run_i = runner_ups[i]
                        mar_i = margin_pcts[i]
                        low_conf = conf_i < thresh
                        reasons: list[str] = []

                        if is_forced:
                            reasons.append(f"Forced Wake ({ov_src})")
                        else:
                            if agr_i < 80.0:
                                reasons.append(f"Low Agreement ({agr_i:.0f}%)")
                            prev_st = auto_stages[i - 1] if i > 0 else auto_i
                            next_st = auto_stages[i + 1] if i < n_epochs - 1 else auto_i
                            if prev_st != auto_i or next_st != auto_i:
                                reasons.append(f"Transition ({prev_st}→{auto_i}→{next_st})")
                            if mar_i < 15.0:
                                reasons.append(f"Narrow Margin ({auto_i} vs {run_i}: Δ{mar_i:.1f}%)")
                            if (auto_i == "N1" or run_i == "N1") and low_conf:
                                reasons.append("N1 Uncertainty")
                            if low_conf and not reasons:
                                reasons.append(f"Low Prob ({conf_i:.1f}%)")

                        flagged = low_conf and not (excl_wake and is_forced)
                        reason_str = "; ".join(reasons) if reasons else "High Confidence"
                        st_lbl = "Wake Override" if is_forced else ("Flagged" if flagged else "Auto")
                        is_flagged.append(flagged)
                        flag_reasons.append(reason_str)
                        base_status.append(st_lbl)
                        base_hover.append(
                            f"<b>Ep #{i + 1}</b> ({time_starts[i]}–{time_ends[i]})<br>"
                            f"Stage: <b>{auto_i}</b> ({html_lib.escape(str(st_lbl))})<br>"
                            f"Confidence: <b>{conf_i:.1f}%</b> | Agree: {agr_i:.0f}%<br>"
                            f"Note: {html_lib.escape(str(reason_str))}"
                        )

                    assert avg_proba is not None
                    base_df = pd.DataFrame({
                        "Epoch": np.arange(n_epochs, dtype=int),
                        "Epoch_Num": np.arange(1, n_epochs + 1, dtype=int),
                        "Time_Start": time_starts,
                        "Time_End": time_ends,
                        "Position_Sample": pos_starts,
                        "Length_Samples": epoch_len_samples,
                        "Raw_YASA_Stage": raw_yasa_stages,
                        "Auto_Stage": auto_stages,
                        "Override_Source": override_sources,
                        "Confidence_Pct": confidence_pcts,
                        "Agreement_Pct": agreement_pcts,
                        "RunnerUp_Stage": runner_ups,
                        "Margin_Pct": margin_pcts,
                        "Prob_W": np.round(avg_proba[:, 0] * 100.0, 1),
                        "Prob_N1": np.round(avg_proba[:, 1] * 100.0, 1),
                        "Prob_N2": np.round(avg_proba[:, 2] * 100.0, 1),
                        "Prob_N3": np.round(avg_proba[:, 3] * 100.0, 1),
                        "Prob_REM": np.round(avg_proba[:, 4] * 100.0, 1),
                        "Pass_Votes": vote_summaries,
                        "Is_Flagged": is_flagged,
                        "Flag_Reason": flag_reasons,
                        "Base_Status": base_status,
                        "Base_Hover": base_hover,
                    })

                    if is_resume_run:
                        imp_stages, imp_ma, resume_meta, sidecar, is_direct_json = parse_scoring_upload(
                            str(resume_filename), resume_bytes or b"", n_epochs, fs_marker, output_dir, eeg_path
                        )
                        warn_parts = [str(resume_meta["warning"])] if resume_meta.get("warning") else []
                        if is_direct_json and sidecar is not None:
                            if sidecar.get("scoring_mode") == "manual" and sidecar.get("final_stages") is not None:
                                resume_init_edits = {
                                    i: str(s) for i, s in enumerate(sidecar["final_stages"]) if s != "UNS"  # type: ignore[arg-type]
                                }
                                resume_init_edits.update(dict(sidecar["manual_edits"]))  # type: ignore[arg-type]
                            else:
                                resume_init_edits = dict(sidecar["manual_edits"])  # type: ignore[arg-type]
                            resume_base_ma = sorted(imp_ma)
                        else:
                            if sidecar is not None:
                                if sidecar.get("scoring_mode") == "manual" and sidecar.get("final_stages") is not None:
                                    resume_init_edits = {
                                        i: str(s) for i, s in enumerate(sidecar["final_stages"]) if s != "UNS"  # type: ignore[arg-type]
                                    }
                                    resume_init_edits.update(dict(sidecar["manual_edits"]))  # type: ignore[arg-type]
                                else:
                                    resume_init_edits = dict(sidecar["manual_edits"])  # type: ignore[arg-type]
                                for i, imp_st in enumerate(imp_stages):
                                    if imp_st == "UNS":
                                        continue
                                    if i in resume_init_edits and imp_st != resume_init_edits[i]:
                                        resume_init_edits[i] = imp_st
                                    elif i not in resume_init_edits and imp_st != auto_stages[i]:
                                        resume_init_edits[i] = imp_st
                                resume_base_ma = sorted(imp_ma)
                            else:
                                resume_init_edits = {
                                    i: imp_st
                                    for i, imp_st in enumerate(imp_stages)
                                    if imp_st != "UNS" and imp_st != auto_stages[i]
                                }
                                resume_base_ma = sorted(imp_ma)
                                txt_clause = (
                                    ", and .txt unscored 0 rows are treated as W"
                                    if resume_meta.get("format") == ".txt"
                                    else ""
                                )
                                warn_parts.append(
                                    "No .onera.json sidecar found: only stages differing from YASA were marked edited "
                                    f"(previously confirmed same-stage flagged epochs remain flagged{txt_clause})"
                                )

                        if sidecar is not None and sidecar.get("scoring_mode", "auto") != "manual":
                            saved_cfg = dict(sidecar.get("yasa_config") or {})  # type: ignore[arg-type]
                            cfg_diffs: list[str] = []
                            if sorted(saved_cfg.get("active_eegs") or []) != sorted(active_eegs):
                                cfg_diffs.append("EEG")
                            if saved_cfg.get("eog") != eog_ch:
                                cfg_diffs.append("EOG")
                            if saved_cfg.get("emg") != emg_ch:
                                cfg_diffs.append("EMG")
                            if sorted(saved_cfg.get("refs") or []) != sorted(ref_chs):
                                cfg_diffs.append("Refs")
                            if cfg_diffs:
                                warn_parts.append(
                                    f"YASA channels/refs differ from saved session ({', '.join(cfg_diffs)})"
                                )

                        resume_meta["n_overrides"] = len(resume_init_edits or {})
                        resume_meta["warning"] = "; ".join(warn_parts)
                        init_active_found = next(
                            (i for i, f in enumerate(is_flagged) if f and i not in (resume_init_edits or {})),
                            None,
                        )
                        if init_active_found is None:
                            init_active = next((i for i, f in enumerate(is_flagged) if f), 0)
                        else:
                            init_active = init_active_found
                        init_selected = [init_active]
                    else:
                        init_active = next((i for i, f in enumerate(is_flagged) if f), 0)
                        init_selected = [init_active]

                t_total = time.perf_counter() - t_init_start
                init_timings = {
                    "load_s": round(t_load, 2),
                    "resample_s": round(t_resample, 2),
                    "yasa_s": round(t_yasa, 2),
                    "filter_s": round(t_filter, 2),
                    "total_s": round(t_total, 2),
                }

                committed_stem = get_recording_stem(eeg_path)
                if eeg_path.suffix.lower() == ".vhdr":
                    vhdr_df = detect_vhdr_metadata(eeg_path).get("data_file")
                    committed_data_file = str(vhdr_df) if vhdr_df else f"{committed_stem}.eeg"
                else:
                    committed_data_file = eeg_path.name

                disp_ch_cache: dict = self.data_cache.setdefault("disp_channels", {})  # type: ignore[assignment]
                for di, (dch, dt) in enumerate(zip(display_chs, disp_types)):
                    disp_ch_cache[(dch, str(dt).lower(), round(float(disp_sfreq), 3), ref_key)] = (
                        wave_uv[di],
                        float(ch_ylims[di]),
                    )

                session_key = f"{recording_id}__{scoring_mode}"
                if is_consensus_run:
                    self.undo_history.pop(session_key, None)
                    self.manual_edits.pop(session_key, None)
                    self.ma_reviewed.pop(session_key, None)
                    if resume_base_ma:
                        self.ma_flags[session_key] = set(int(i) for i in resume_base_ma)
                    else:
                        self.ma_flags.pop(session_key, None)
                elif is_resume_run:
                    self.undo_history.pop(session_key, None)
                    self.ma_reviewed.pop(session_key, None)
                    if resume_init_edits:
                        self.manual_edits[session_key] = {int(k): str(v) for k, v in resume_init_edits.items()}
                    else:
                        self.manual_edits.pop(session_key, None)
                    if resume_base_ma:
                        self.ma_flags[session_key] = set(int(i) for i in resume_base_ma)
                    else:
                        self.ma_flags.pop(session_key, None)

                self.scored_bundle = {
                    "session_key": session_key,
                    "scoring_mode": scoring_mode,
                    "resumed_from": resume_meta,
                    "consensus_meta": consensus_meta,
                    "base_ma_indices": resume_base_ma,
                    "init_timings": init_timings,
                    "epoch_events": epoch_events,
                    "plot_events": plot_events,
                    "base_epochs_df": base_df,
                    "wake_override_intervals_df": wake_override_df,
                    "wave_uv": wave_uv,
                    "disp_sfreq": float(disp_sfreq),
                    "ep_samps": int(ep_samps),
                    "display_chs": display_chs,
                    "display_types": disp_types,
                    "ch_ylims": ch_ylims,
                    "all_recording_chs": list(all_ch_names),
                    "committed_eeg_path": str(eeg_path),
                    "committed_raw_sfreq": float(raw_sf),
                    "committed_file_key": str(file_key),
                    "committed_type_overrides": dict(auto_type_overrides),
                    "active_eegs_used": active_eegs,
                    "committed_eog": eog_ch,
                    "committed_emg": emg_ch,
                    "committed_refs": list(ref_chs),
                    "flagged_indices": [i for i, f in enumerate(is_flagged) if f],
                    "committed_thresh": thresh,
                    "committed_excl_wake": excl_wake,
                    "committed_recording_id": recording_id,
                    "committed_recording_label": recording_label or committed_stem,
                    "committed_fs": float(fs_marker),
                    "committed_vmrk": df_vmrk.copy(),
                    "committed_meas_date": meas_date,
                    "committed_stem": committed_stem,
                    "committed_data_file": committed_data_file,
                    "init_active_ep": int(init_active),
                    "init_selected_eps": list(init_selected),
                    "default_output_dir": str(output_dir),
                }

            self._update_job(job_id, status="done", pct=100, subtitle="Ready")
        except Exception as exc:
            self._update_job(job_id, status="error", error=str(exc), subtitle=f"Error: {exc}")

    def _active_context_locked(self) -> tuple[dict[str, object], str, dict[int, str], set[int]]:
        bundle = self.scored_bundle
        if bundle is None or bundle["base_epochs_df"].empty:  # type: ignore[union-attr]
            raise ValueError("No active scoring session.")
        sk = str(bundle["session_key"])
        edits = self.manual_edits.get(sk, {})
        mas = self.ma_flags.get(sk, set())
        return bundle, sk, edits, mas

    def _compute_consensus_counts_locked(
        self,
        bundle: dict[str, object],
        edits: dict[int, str],
        ma_rev: set[int] | None = None,
    ) -> dict[str, object] | None:
        if bundle.get("scoring_mode") != "consensus":
            return None
        base_df: pd.DataFrame = bundle["base_epochs_df"]  # type: ignore[assignment]
        cmeta: dict[str, object] = dict(bundle.get("consensus_meta") or {})  # type: ignore[arg-type]
        flagged = list(bundle.get("flagged_indices") or [])  # type: ignore[arg-type]
        stage_diffs = base_df["Stage_Diff"].tolist() if "Stage_Diff" in base_df.columns else [False] * len(base_df)
        ma_diffs = base_df["MA_Diff"].tolist() if "MA_Diff" in base_df.columns else [False] * len(base_df)
        flag_ma_diffs = bool(cmeta.get("flag_ma_diffs", False))
        rev_set = ma_rev or set()

        unresolved_indices: list[int] = []
        n_unresolved_stage = 0
        for idx in flagged:
            st_unres = bool(stage_diffs[idx]) and (idx not in edits)
            ma_unres = bool(flag_ma_diffs and ma_diffs[idx]) and (idx not in rev_set) and (idx not in edits)
            if st_unres or ma_unres:
                unresolved_indices.append(idx)
                if st_unres:
                    n_unresolved_stage += 1
        unres_set = set(unresolved_indices)
        resolved_indices = [idx for idx in flagged if idx not in unres_set]
        return {
            "total_flagged": len(flagged),
            "n_resolved": len(resolved_indices),
            "n_unresolved": len(unresolved_indices),
            "n_unresolved_stage": n_unresolved_stage,
            "unresolved_indices": unresolved_indices,
            "resolved_indices": resolved_indices,
        }

    def get_serialized_session(self) -> dict[str, object]:
        with self._lock:
            bundle, sk, edits, mas = self._active_context_locked()
            scored_df = build_scored_epochs_df(bundle, edits, mas)
            n_ep = len(scored_df)
            scoring_mode = str(bundle.get("scoring_mode", "auto"))
            is_manual_mode = scoring_mode == "manual"
            is_consensus_mode = scoring_mode == "consensus"
            cons_meta = dict(bundle.get("consensus_meta") or {}) if is_consensus_mode else None  # type: ignore[arg-type]
            has_yasa = (not is_manual_mode) and (not is_consensus_mode or bool((cons_meta or {}).get("has_yasa", False)))
            display_chs = list(bundle["display_chs"])  # type: ignore[arg-type]
            display_types = list(bundle["display_types"])  # type: ignore[arg-type]
            ch_ylims = list(bundle["ch_ylims"])  # type: ignore[arg-type]

            channel_meta: dict[str, dict[str, object]] = {}
            for cname, ctype, cyl in zip(display_chs, display_types, ch_ylims):
                ct = str(ctype).lower()
                yl = float(cyl)
                enc = max(yl, 300.0) if ct in ("eeg", "eog") else max(yl, 1e-12)
                channel_meta[cname] = {
                    "name": cname,
                    "type": ct,
                    "unit": "(µV)" if ct in ("eeg", "eog", "emg", "ecg") else "(raw)",
                    "color": WAVE_PALETTE.get(ct, "#5c6b7a"),
                    "ylim": yl,
                    "encodeMax": float(enc),
                }

            wake_ov_df: pd.DataFrame = bundle["wake_override_intervals_df"]  # type: ignore[assignment]
            wake_overrides: list[list[float]] = []
            if not wake_ov_df.empty:
                for _, wr in wake_ov_df.iterrows():
                    wx0 = (float(wr["t0_sec"]) / 30.0) + 1.0
                    wx1 = (float(wr["t1_sec"]) / 30.0) + 1.0
                    if wx1 > wx0:
                        wake_overrides.append([round(wx0, 3), round(wx1, 3)])

            plot_events: list[dict[str, object]] = bundle.get("plot_events", [])  # type: ignore[assignment]
            m_ev, l_ev = build_hypnogram_events(plot_events, n_ep)
            ep_events_raw: dict[int, list[tuple[float, str]]] = bundle.get("epoch_events", {})  # type: ignore[assignment]
            epoch_events = {str(k): [[float(s), str(d)] for s, d in v] for k, v in ep_events_raw.items()}

            meas_date = bundle.get("committed_meas_date")
            meas_start_sec = (
                float(meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second)  # type: ignore[union-attr]
                + float(meas_date.microsecond) * 1e-6  # type: ignore[union-attr]
                if meas_date is not None
                else None
            )
            ep0_clock_sec = 0
            try:
                ts0 = str(scored_df.iloc[0]["Time_Start"]).strip().split(":")
                if len(ts0) >= 3:
                    ep0_clock_sec = int(ts0[0]) * 3600 + int(ts0[1]) * 60 + int(float(ts0[2]))
            except Exception:
                ep0_clock_sec = 0

            def _clean_float_list(col_name: str) -> list[float | None]:
                vals = scored_df[col_name].tolist()
                return [None if (v is None or pd.isna(v) or not np.isfinite(float(v))) else float(v) for v in vals]

            probs_list: list[list[float]] = []
            if has_yasa:
                pw = scored_df["Prob_W"].tolist()
                pn1 = scored_df["Prob_N1"].tolist()
                pn2 = scored_df["Prob_N2"].tolist()
                pn3 = scored_df["Prob_N3"].tolist()
                pr = scored_df["Prob_REM"].tolist()
                for i in range(n_ep):
                    probs_list.append([
                        0.0 if pd.isna(pw[i]) else float(pw[i]),
                        0.0 if pd.isna(pn1[i]) else float(pn1[i]),
                        0.0 if pd.isna(pn2[i]) else float(pn2[i]),
                        0.0 if pd.isna(pn3[i]) else float(pn3[i]),
                        0.0 if pd.isna(pr[i]) else float(pr[i]),
                    ])

            final_stages = scored_df["Final_Stage"].tolist()
            flagged_indices = list(bundle["flagged_indices"])  # type: ignore[arg-type]
            jump_indices = [i for i, st in enumerate(final_stages) if st == "UNS"] if is_manual_mode else flagged_indices
            can_undo = bool(self.undo_history.get(sk))

            consensus_payload: dict[str, object] | None = None
            if is_consensus_mode and cons_meta is not None:
                ma_rev = self.ma_reviewed.get(sk, set())
                counts = self._compute_consensus_counts_locked(bundle, edits, ma_rev)
                consensus_payload = {
                    **cons_meta,
                    "counts": counts,
                    "scorer1_stage": scored_df["Scorer1_Stage"].tolist(),
                    "scorer2_stage": scored_df["Scorer2_Stage"].tolist(),
                    "scorer1_ma": [bool(x) for x in scored_df["Scorer1_MA"].tolist()],
                    "scorer2_ma": [bool(x) for x in scored_df["Scorer2_MA"].tolist()],
                    "stage_diff": [bool(x) for x in scored_df["Stage_Diff"].tolist()],
                    "ma_diff": [bool(x) for x in scored_df["MA_Diff"].tolist()],
                    "yasa_resolved": [bool(x) for x in scored_df["YASA_Resolved"].tolist()],
                }

            return {
                "session_key": sk,
                "recording_id": bundle["committed_recording_id"],
                "recording_label": bundle["committed_recording_label"],
                "stem": bundle["committed_stem"],
                "scoring_mode": scoring_mode,
                "is_manual_mode": is_manual_mode,
                "is_consensus_mode": is_consensus_mode,
                "has_yasa": has_yasa,
                "consensus": consensus_payload,
                "n_ep": n_ep,
                "disp_sfreq": bundle["disp_sfreq"],
                "ep_samps": bundle["ep_samps"],
                "total_sec": float(n_ep) * 30.0,
                "fs_marker": bundle["committed_fs"],
                "thresh": bundle["committed_thresh"],
                "excl_wake": bundle["committed_excl_wake"],
                "active_eegs": bundle["active_eegs_used"],
                "display_chs": display_chs,
                "all_recording_chs": bundle["all_recording_chs"],
                "channel_meta": channel_meta,
                "init_timings": bundle["init_timings"],
                "resumed_from": bundle["resumed_from"],
                "init_active_ep": bundle["init_active_ep"],
                "init_selected_eps": bundle["init_selected_eps"],
                "can_undo": can_undo,
                "default_output_dir": bundle.get("default_output_dir") or self.output_dir,
                "ep0_clock_sec": ep0_clock_sec,
                "meas_start_sec": meas_start_sec,
                "flagged_indices": flagged_indices,
                "jump_indices": jump_indices,
                "wake_overrides": wake_overrides,
                "hypno_events": {"mEv": m_ev, "lEv": l_ev},
                "epoch_events": epoch_events,
                "has_s4": "S4" in final_stages,
                "epochs": {
                    "time_start": scored_df["Time_Start"].tolist(),
                    "time_end": scored_df["Time_End"].tolist(),
                    "raw_yasa_stage": scored_df["Raw_YASA_Stage"].tolist(),
                    "auto_stage": scored_df["Auto_Stage"].tolist(),
                    "final_stage": final_stages,
                    "override_source": scored_df["Override_Source"].tolist(),
                    "confidence_pct": _clean_float_list("Confidence_Pct"),
                    "agreement_pct": _clean_float_list("Agreement_Pct"),
                    "runner_up": scored_df["RunnerUp_Stage"].tolist(),
                    "margin_pct": _clean_float_list("Margin_Pct"),
                    "pass_votes": scored_df["Pass_Votes"].tolist(),
                    "is_flagged": [bool(x) for x in scored_df["Is_Flagged"].tolist()],
                    "flag_reason": scored_df["Flag_Reason"].tolist(),
                    "base_status": scored_df["Base_Status"].tolist(),
                    "is_manual_edit": [bool(x) for x in scored_df["Is_Manual_Edit"].tolist()],
                    "has_ma": [bool(x) for x in scored_df["Has_MA"].tolist()],
                    "status": scored_df["Status"].tolist(),
                    "hover_text": scored_df["Hover_Text"].tolist(),
                    "probs": probs_list,
                },
            }

    def _build_patch_rows_locked(
        self,
        bundle: dict[str, object],
        edits: dict[int, str],
        mas: set[int],
        indices: list[int],
    ) -> dict[str, object]:
        base_df: pd.DataFrame = bundle["base_epochs_df"]  # type: ignore[assignment]
        n_ep = len(base_df)
        scoring_mode = str(bundle.get("scoring_mode", "auto"))
        is_manual_mode = scoring_mode == "manual"
        is_consensus_mode = scoring_mode == "consensus"
        cmeta: dict[str, object] = dict(bundle.get("consensus_meta") or {}) if is_consensus_mode else {}  # type: ignore[arg-type]
        s1_lbl = str(cmeta.get("scorer1_label") or "Scorer 1")
        s2_lbl = str(cmeta.get("scorer2_label") or "Scorer 2")
        has_yasa = bool(cmeta.get("has_yasa", not is_manual_mode))

        patches: list[dict[str, object]] = []
        for idx in sorted(set(indices)):
            if not (0 <= idx < n_ep):
                continue
            row = base_df.iloc[idx]
            auto_st = str(row["Auto_Stage"])
            is_ed = idx in edits
            final_st = str(edits[idx]) if is_ed else auto_st
            has_ma = idx in mas
            st_str, hover = format_epoch_overlay(
                idx=idx,
                auto_stage=auto_st,
                final_stage=final_st,
                is_manual_edit=is_ed,
                has_ma=has_ma,
                base_status=str(row["Base_Status"]),
                base_hover=str(row["Base_Hover"]),
                time_start=str(row["Time_Start"]),
                time_end=str(row["Time_End"]),
                conf_pct=float(row["Confidence_Pct"]),
                agr_pct=float(row["Agreement_Pct"]),
                flag_reason=str(row["Flag_Reason"]),
                is_manual_mode=is_manual_mode,
                is_consensus_mode=is_consensus_mode,
                s1_label=s1_lbl,
                s2_label=s2_lbl,
                s1_stage=str(row["Scorer1_Stage"]) if is_consensus_mode else "",
                s2_stage=str(row["Scorer2_Stage"]) if is_consensus_mode else "",
                s1_ma=bool(row["Scorer1_MA"]) if is_consensus_mode else False,
                s2_ma=bool(row["Scorer2_MA"]) if is_consensus_mode else False,
                has_yasa=has_yasa,
                yasa_stage=str(row["Raw_YASA_Stage"]) if "Raw_YASA_Stage" in row else "",
            )
            patches.append({
                "idx": idx,
                "final_stage": final_st,
                "is_manual_edit": is_ed,
                "has_ma": has_ma,
                "status": st_str,
                "hover_text": hover,
            })

        auto_all = base_df["Auto_Stage"].tolist()
        final_all = [edits.get(i, auto_all[i]) for i in range(n_ep)]
        has_s4 = "S4" in final_all
        jump_indices = (
            [i for i, st in enumerate(final_all) if st == "UNS"]
            if is_manual_mode
            else list(bundle["flagged_indices"])  # type: ignore[arg-type]
        )
        sk = str(bundle["session_key"])
        ma_rev = self.ma_reviewed.get(sk, set())
        cons_counts = self._compute_consensus_counts_locked(bundle, edits, ma_rev)
        return {
            "patches": patches,
            "can_undo": bool(self.undo_history.get(sk)),
            "has_s4": has_s4,
            "n_edited": len(edits),
            "n_ma": len(mas),
            "n_unscored": sum(1 for st in final_all if st == "UNS"),
            "jump_indices": jump_indices,
            "consensus_counts": cons_counts,
        }

    def apply_stage_edit(self, epochs: list[int], stage: str) -> dict[str, object]:
        canon = normalize_stage(stage)
        if canon not in ("W", "N1", "N2", "N3", "S4", "REM", "ART"):
            raise ValueError(f"Unsupported stage: {stage}")
        with self._lock:
            bundle, sk, _, _ = self._active_context_locked()
            n_ep = len(bundle["base_epochs_df"])  # type: ignore[arg-type]
            targets = sorted({int(ep) for ep in epochs if 0 <= int(ep) < n_ep})
            if not targets:
                raise ValueError("No valid epoch indices selected.")
            cur_edits = self.manual_edits.setdefault(sk, {})
            undo_snap = ("stage", {ep: cur_edits.get(ep, None) for ep in targets})
            self.undo_history.setdefault(sk, []).append(undo_snap)
            for ep in targets:
                cur_edits[ep] = canon
            mas = self.ma_flags.get(sk, set())
            return self._build_patch_rows_locked(bundle, cur_edits, mas, targets)

    def accept_consensus_scorer(self, epochs: list[int], scorer_num: int) -> dict[str, object]:
        if scorer_num not in (1, 2):
            raise ValueError(f"Scorer must be 1 or 2, got: {scorer_num}")
        with self._lock:
            bundle, sk, _, _ = self._active_context_locked()
            if bundle.get("scoring_mode") != "consensus":
                raise ValueError("Accept Scorer action is only valid in Two-Scorer Consensus Mode.")
            base_df: pd.DataFrame = bundle["base_epochs_df"]  # type: ignore[assignment]
            n_ep = len(base_df)
            targets = sorted({int(ep) for ep in epochs if 0 <= int(ep) < n_ep})
            if not targets:
                raise ValueError("No valid epoch indices selected.")

            cmeta: dict[str, object] = dict(bundle.get("consensus_meta") or {})  # type: ignore[arg-type]
            flag_ma_diffs = bool(cmeta.get("flag_ma_diffs", False))
            s_col = "Scorer1_Stage" if scorer_num == 1 else "Scorer2_Stage"
            ma_col = "Scorer1_MA" if scorer_num == 1 else "Scorer2_MA"
            s_vals = base_df[s_col].tolist()
            ma_vals = base_df[ma_col].tolist()

            cur_edits = self.manual_edits.setdefault(sk, {})
            cur_ma = self.ma_flags.setdefault(sk, set())
            cur_rev = self.ma_reviewed.setdefault(sk, set())

            undo_snap = (
                "consensus",
                {ep: (cur_edits.get(ep, None), (ep in cur_ma), (ep in cur_rev)) for ep in targets},
            )
            self.undo_history.setdefault(sk, []).append(undo_snap)

            for ep in targets:
                cur_edits[ep] = str(s_vals[ep])
                if flag_ma_diffs:
                    if bool(ma_vals[ep]):
                        cur_ma.add(ep)
                    else:
                        cur_ma.discard(ep)
                cur_rev.add(ep)

            if not cur_ma:
                self.ma_flags.pop(sk, None)
            return self._build_patch_rows_locked(bundle, cur_edits, cur_ma, targets)

    def toggle_ma(self, epochs: list[int]) -> dict[str, object]:
        with self._lock:
            bundle, sk, edits, _ = self._active_context_locked()
            n_ep = len(bundle["base_epochs_df"])  # type: ignore[arg-type]
            targets = sorted({int(ep) for ep in epochs if 0 <= int(ep) < n_ep})
            if not targets:
                raise ValueError("No valid epoch indices selected.")
            is_cons = bundle.get("scoring_mode") == "consensus"
            cur_ma = self.ma_flags.setdefault(sk, set())
            all_have_ma = all(ep in cur_ma for ep in targets)
            if is_cons:
                cur_rev = self.ma_reviewed.setdefault(sk, set())
                undo_snap = ("ma_cons", {ep: ((ep in cur_ma), (ep in cur_rev)) for ep in targets})
            else:
                undo_snap = ("ma", {ep: (ep in cur_ma) for ep in targets})
            self.undo_history.setdefault(sk, []).append(undo_snap)
            if all_have_ma:
                for ep in targets:
                    cur_ma.discard(ep)
            else:
                for ep in targets:
                    cur_ma.add(ep)
            if is_cons:
                cur_rev = self.ma_reviewed.setdefault(sk, set())
                for ep in targets:
                    cur_rev.add(ep)
            if not cur_ma:
                self.ma_flags.pop(sk, None)
            res = self._build_patch_rows_locked(bundle, edits, cur_ma, targets)
            return res

    def undo_last_edit(self) -> dict[str, object]:
        with self._lock:
            bundle, sk, _, _ = self._active_context_locked()
            stack = self.undo_history.get(sk)
            if not stack:
                return {"restored_epochs": [], "patches": [], "can_undo": False}
            kind, snap = stack.pop()
            if kind == "consensus":
                cur_edits = self.manual_edits.setdefault(sk, {})
                cur_ma = self.ma_flags.setdefault(sk, set())
                cur_rev = self.ma_reviewed.setdefault(sk, set())
                for ep, (prev_ed, prev_ma, prev_rev) in snap.items():
                    idx = int(ep)
                    if prev_ed is None:
                        cur_edits.pop(idx, None)
                    else:
                        cur_edits[idx] = str(prev_ed)
                    if prev_ma:
                        cur_ma.add(idx)
                    else:
                        cur_ma.discard(idx)
                    if prev_rev:
                        cur_rev.add(idx)
                    else:
                        cur_rev.discard(idx)
                if not cur_edits:
                    self.manual_edits.pop(sk, None)
                if not cur_ma:
                    self.ma_flags.pop(sk, None)
                if not cur_rev:
                    self.ma_reviewed.pop(sk, None)
            elif kind == "ma_cons":
                cur_ma = self.ma_flags.setdefault(sk, set())
                cur_rev = self.ma_reviewed.setdefault(sk, set())
                for ep, (prev_ma, prev_rev) in snap.items():
                    idx = int(ep)
                    if prev_ma:
                        cur_ma.add(idx)
                    else:
                        cur_ma.discard(idx)
                    if prev_rev:
                        cur_rev.add(idx)
                    else:
                        cur_rev.discard(idx)
                if not cur_ma:
                    self.ma_flags.pop(sk, None)
                if not cur_rev:
                    self.ma_reviewed.pop(sk, None)
            elif kind == "ma":
                cur_ma = self.ma_flags.setdefault(sk, set())
                for ep, prev_flag in snap.items():
                    if prev_flag:
                        cur_ma.add(int(ep))
                    else:
                        cur_ma.discard(int(ep))
                if not cur_ma:
                    self.ma_flags.pop(sk, None)
            else:
                cur_edits = self.manual_edits.setdefault(sk, {})
                for ep, prev_val in snap.items():
                    if prev_val is None:
                        cur_edits.pop(int(ep), None)
                    else:
                        cur_edits[int(ep)] = str(prev_val)
                if not cur_edits:
                    self.manual_edits.pop(sk, None)
            restored = sorted(int(ep) for ep in snap.keys())
            edits = self.manual_edits.get(sk, {})
            mas = self.ma_flags.get(sk, set())
            res = self._build_patch_rows_locked(bundle, edits, mas, restored)
            res["restored_epochs"] = restored
            return res

    def clear_all_edits(self) -> dict[str, object]:
        with self._lock:
            bundle, sk, _, _ = self._active_context_locked()
            scoring_mode = str(bundle.get("scoring_mode", "auto"))
            is_manual_mode = scoring_mode == "manual"
            base_ma = tuple(bundle.get("base_ma_indices") or ())  # type: ignore[arg-type]
            is_res = bundle.get("resumed_from") is not None
            self.undo_history.pop(sk, None)
            self.manual_edits.pop(sk, None)
            self.ma_reviewed.pop(sk, None)
            if (scoring_mode == "consensus" or (is_manual_mode and is_res)) and base_ma:
                self.ma_flags[sk] = set(int(i) for i in base_ma)
            else:
                self.ma_flags.pop(sk, None)
            n_ep = len(bundle["base_epochs_df"])  # type: ignore[arg-type]
            edits = self.manual_edits.get(sk, {})
            mas = self.ma_flags.get(sk, set())
            return self._build_patch_rows_locked(bundle, edits, mas, list(range(n_ep)))

    def build_waveforms_binary(self, start_ep: int, end_ep: int, channels: list[str]) -> bytes:
        """Encode [start_ep, end_ep) waveforms for requested channels into a 4-byte-aligned Int16 binary frame."""
        with self._lock:
            bundle = self.scored_bundle
            if bundle is None or bundle["base_epochs_df"].empty:  # type: ignore[union-attr]
                raise ValueError("No active scoring session.")
            n_ep = int(len(bundle["base_epochs_df"]))  # type: ignore[arg-type]
            ep_samps = int(bundle["ep_samps"])  # type: ignore[arg-type]

        s_ep = max(0, int(start_ep))
        e_ep = min(n_ep, int(end_ep))
        if e_ep <= s_ep:
            raise ValueError(f"Empty or invalid epoch range [{start_ep}, {end_ep}).")
        if (e_ep - s_ep) > 32:
            e_ep = s_ep + 32
        n_batch = e_ep - s_ep

        ch_list = [str(c) for c in channels if str(c)]
        if not ch_list:
            raise ValueError("No channels requested.")

        ch_meta_list: list[dict[str, object]] = []
        ch_bytes_list: list[bytes] = []
        byte_offset = 0

        for ch_name in ch_list:
            loaded = self.ensure_display_channel(ch_name)
            if loaded is None:
                continue
            w1d: np.ndarray = loaded["wave"]  # type: ignore[assignment]
            ylim = float(loaded["ylim"])  # type: ignore[arg-type]
            ctype = str(loaded["type"])
            encode_max = float(loaded["encodeMax"])  # type: ignore[arg-type]
            w_len = int(w1d.shape[0])
            scale_q = 32767.0 / max(1e-12, encode_max)

            is_emg_minmax = ctype == "emg" and ep_samps >= EMG_BINS_PER_EP * 2
            if is_emg_minmax:
                pts_per_ep = EMG_BINS_PER_EP * 2
                edges = np.linspace(0, ep_samps, EMG_BINS_PER_EP + 1, dtype=int)
                out_q = np.empty((n_batch, pts_per_ep), dtype="<i2")
                for bi, ep in enumerate(range(s_ep, e_ep)):
                    s0 = ep * ep_samps
                    s1 = s0 + ep_samps
                    if w_len == 0 or s0 >= w_len:
                        seg = np.zeros(ep_samps, dtype=np.float32)
                    elif s1 <= w_len:
                        seg = w1d[s0:s1]
                    else:
                        seg = np.zeros(ep_samps, dtype=np.float32)
                        avail = max(0, w_len - s0)
                        seg[:avail] = w1d[s0 : s0 + avail]
                    if not np.all(np.isfinite(seg)):
                        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
                    ds = np.empty(pts_per_ep, dtype=np.float64)
                    ds[0::2] = np.minimum.reduceat(seg, edges[:-1])
                    ds[1::2] = np.maximum.reduceat(seg, edges[:-1])
                    out_q[bi] = np.clip(np.round(ds * scale_q), -32767, 32767).astype("<i2")
            else:
                need_resample = ep_samps > MAX_SMOOTH_PTS_PER_EP
                pts_per_ep = (SMOOTH_RESAMPLE_PTS + 1) if need_resample else (ep_samps + 1)
                x_new = np.linspace(0.0, float(ep_samps), SMOOTH_RESAMPLE_PTS + 1) if need_resample else None
                x_old = np.arange(ep_samps + 1, dtype=np.float64) if need_resample else None
                out_q = np.empty((n_batch, pts_per_ep), dtype="<i2")
                need_len = ep_samps + 1
                for bi, ep in enumerate(range(s_ep, e_ep)):
                    s0 = ep * ep_samps
                    s1 = s0 + need_len
                    if w_len == 0 or s0 >= w_len:
                        seg = np.zeros(need_len, dtype=np.float32)
                    elif s1 <= w_len:
                        seg = w1d[s0:s1]
                    else:
                        seg = np.zeros(need_len, dtype=np.float32)
                        avail = max(0, w_len - s0)
                        seg[:avail] = w1d[s0 : s0 + avail]
                        if avail > 0 and avail < need_len:
                            seg[avail:] = seg[avail - 1]
                    if not np.all(np.isfinite(seg)):
                        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
                    if need_resample:
                        vals = np.interp(x_new, x_old, seg)  # type: ignore[arg-type]
                    else:
                        vals = seg.astype(np.float64, copy=False)
                    out_q[bi] = np.clip(np.round(vals * scale_q), -32767, 32767).astype("<i2")

            raw_bytes = out_q.tobytes()
            ch_meta_list.append({
                "name": ch_name,
                "type": ctype,
                "ylim": ylim,
                "encodeMax": encode_max,
                "isMinMax": is_emg_minmax,
                "ptsPerEp": pts_per_ep,
                "byteOffset": byte_offset,
            })
            ch_bytes_list.append(raw_bytes)
            byte_offset += len(raw_bytes)
            # Keep every channel block 4-byte aligned for clean Int16Array views
            rem = len(raw_bytes) % 4
            if rem != 0:
                pad_n = 4 - rem
                ch_bytes_list.append(b"\x00" * pad_n)
                byte_offset += pad_n

        header_json = json.dumps(
            {
                "startEp": s_ep,
                "endEp": e_ep,
                "channels": ch_meta_list,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        pad_hdr = (-len(header_json)) % 4
        if pad_hdr:
            header_json += b" " * pad_hdr

        return struct.pack("<I", len(header_json)) + header_json + b"".join(ch_bytes_list)

    def _prepare_export(
        self, req: dict[str, object]
    ) -> tuple[str, int, int, str, Path, Path, str, dict[str, object], pd.DataFrame, dict[int, str], set[int], dict[str, object] | None]:
        with self._lock:
            bundle, sk, edits, mas = self._active_context_locked()
            scored_df = build_scored_epochs_df(bundle, edits, mas)
            edits_copy = dict(edits)
            mas_copy = set(mas)
            ma_rev_copy = set(self.ma_reviewed.get(sk, set()))
            cons_counts = self._compute_consensus_counts_locked(bundle, edits_copy, ma_rev_copy)

        fmt_key = normalize_export_fmt_key(str(req.get("fmt_key") or req.get("format") or "markers"))
        use_rk = bool(req.get("use_rk", False))
        include_events = bool(req.get("include_events", True))
        stem = str(bundle["committed_stem"])
        fname = resolve_export_filename(fmt_key, str(req.get("filename") or ""), stem)
        target_dir = _normalize_user_path(str(req.get("output_dir") or ""), ".")
        target_path = target_dir / fname

        export_text, n_rows, _, n_unscored, preview_str = build_export_content(
            fmt_key=fmt_key,
            use_rk=use_rk,
            include_events=include_events,
            scored_bundle=bundle,
            scored_df=scored_df,
        )
        return (
            export_text,
            n_rows,
            n_unscored,
            preview_str,
            target_dir,
            target_path,
            fname,
            bundle,
            scored_df,
            edits_copy,
            mas_copy,
            cons_counts,
        )

    def preview_export(self, req: dict[str, object]) -> dict[str, object]:
        _, n_rows, n_unscored, preview_str, _, target_path, fname, bundle, _, _, _, cons_counts = self._prepare_export(req)
        fmt_key = normalize_export_fmt_key(str(req.get("fmt_key") or req.get("format") or "markers"))
        cmeta = dict(bundle.get("consensus_meta") or {})  # type: ignore[arg-type]
        return {
            "fmt_key": fmt_key,
            "filename": fname,
            "target_path": str(target_path),
            "n_rows": n_rows,
            "n_unscored": n_unscored,
            "n_unresolved_conflicts": int(cons_counts["n_unresolved"]) if cons_counts else 0,
            "n_unresolved_stage_conflicts": int(cons_counts["n_unresolved_stage"]) if cons_counts else 0,
            "s1_label": str(cmeta.get("scorer1_label") or "Scorer 1") if cmeta else "Scorer 1",
            "preview": preview_str,
        }

    def save_export(self, req: dict[str, object]) -> dict[str, object]:
        (
            export_text,
            n_rows,
            n_unscored,
            preview_str,
            target_dir,
            target_path,
            fname,
            bundle,
            scored_df,
            edits_copy,
            mas_copy,
            cons_counts,
        ) = self._prepare_export(req)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(export_text, encoding="utf-8")

        sidecar_path_str = ""
        try:
            sidecar_payload = build_sidecar_payload(
                scored_bundle=bundle,
                scored_df=scored_df,
                manual_edits=edits_copy,
                ma_flags=mas_copy,
                target_stem=target_path.stem,
            )
            sidecar_dir = target_dir / ".onera_sessions"
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = sidecar_dir / f"{target_path.stem}.onera.json"
            sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")
            sidecar_path_str = f".onera_sessions/{sidecar_path.name}"
        except Exception:
            pass

        cmeta = dict(bundle.get("consensus_meta") or {})  # type: ignore[arg-type]
        return {
            "filename": fname,
            "target_path": str(target_path),
            "n_rows": n_rows,
            "n_unscored": n_unscored,
            "n_unresolved_conflicts": int(cons_counts["n_unresolved"]) if cons_counts else 0,
            "n_unresolved_stage_conflicts": int(cons_counts["n_unresolved_stage"]) if cons_counts else 0,
            "s1_label": str(cmeta.get("scorer1_label") or "Scorer 1") if cmeta else "Scorer 1",
            "preview": preview_str,
            "sidecar_relpath": sidecar_path_str,
        }

    def download_export(self, req: dict[str, object]) -> tuple[bytes, str]:
        export_text, _, _, _, _, _, fname, _, _, _, _, _ = self._prepare_export(req)
        return export_text.encode("utf-8"), fname
