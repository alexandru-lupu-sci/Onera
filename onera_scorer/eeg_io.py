"""Pure EEG/PSG I/O, marker parsing, signal processing, and multi-format export helpers."""

from __future__ import annotations

import csv
import datetime
from fractions import Fraction
import html as html_lib
import io
import json
import os
from pathlib import Path
import re
import warnings

os.environ.setdefault("MNE_USE_NUMBA", "false")

import mne
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.exceptions import InconsistentVersionWarning

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

SUPPORTED_EEG_EXTS = (".vhdr", ".edf", ".bdf", ".fif", ".set", ".gdf")
STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]
AASM_STAGE_MAP = {"S4": "N3", "UNS": "?"}
RK_STAGE_MAP = {
    "W": "W",
    "N1": "S1",
    "N2": "S2",
    "N3": "S3",
    "S4": "S4",
    "REM": "REM",
    "ART": "MT",
    "UNS": "?",
}
WAVE_PALETTE = {
    "eeg": "#2f63ab",
    "eog": "#268074",
    "emg": "#b54854",
    "ecg": "#6c52a8",
    "misc": "#475569",
}
MULTI_ELECTRODE_PRESETS = {
    "Preset 1: Frontal-Central Quad [F3, F4, C3, C4] (Recommended)": ["F3", "F4", "C3", "C4"],
    "Preset 2: Contralateral Pair [F4, C3]": ["F4", "C3"],
}
MAX_SMOOTH_PTS_PER_EP = 7680
SMOOTH_RESAMPLE_PTS = 6000
EMG_BINS_PER_EP = 1800

_STAGE_ALIASES = {
    "W": "W",
    "WAKE": "W",
    "0": "W",
    "N1": "N1",
    "S1": "N1",
    "1": "N1",
    "N2": "N2",
    "S2": "N2",
    "2": "N2",
    "N3": "N3",
    "S3": "N3",
    "3": "N3",
    "S4": "S4",
    "4": "S4",
    "REM": "REM",
    "R": "REM",
    "5": "REM",
    "ART": "ART",
    "MT": "ART",
    "MOVEMENT": "ART",
    "MOVEMENT TIME": "ART",
    "?": "UNS",
    "UNS": "UNS",
    "UNSCORED": "UNS",
    "-1": "UNS",
    "NAN": "UNS",
    "": "UNS",
}


def get_recording_stem(filepath: str | Path) -> str:
    """Return filename stem cleanly, stripping compound .fif.gz extension when present."""
    name = Path(filepath).name
    if name.lower().endswith(".fif.gz"):
        return name[:-7]
    return Path(filepath).stem


def read_raw_eeg(filepath: str | Path, preload: bool = False):
    """Unified MNE raw reader across BrainVision, EDF/EDF+, BioSemi BDF, FIF, EEGLAB, and GDF."""
    p = Path(filepath)
    low = p.name.lower()
    if low.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(p, preload=preload, verbose="ERROR")
    if low.endswith(".edf"):
        return mne.io.read_raw_edf(p, infer_types=False, preload=preload, verbose="ERROR")
    if low.endswith(".bdf"):
        return mne.io.read_raw_bdf(p, infer_types=False, preload=preload, verbose="ERROR")
    if low.endswith((".fif", ".fif.gz")):
        return mne.io.read_raw_fif(p, allow_maxshield="yes", preload=preload, verbose="ERROR")
    if low.endswith(".set"):
        try:
            return mne.io.read_raw_eeglab(p, preload=preload, verbose="ERROR")
        except Exception:
            if not preload:
                return mne.io.read_raw_eeglab(p, preload=True, verbose="ERROR")
            raise
    if low.endswith(".gdf"):
        return mne.io.read_raw_gdf(p, preload=preload, verbose="ERROR")
    return mne.io.read_raw(p, preload=preload, verbose="ERROR")


def natural_sort_key(text: str) -> tuple:
    """Sort strings naturally with numeric segments ordered numerically."""
    return tuple(
        (0, int(tok), "") if tok.isdigit() else (1, 0, tok.lower())
        for tok in re.split(r"(\d+)", str(text))
        if tok
    )


def scan_recordings(root_dir: str | Path) -> dict[str, Path]:
    """Recursively scan root_dir for EEG/PSG recordings, deduplicate per (folder, stem), and sort naturally."""
    ext_priority = {".vhdr": 0, ".edf": 1, ".bdf": 2, ".fif": 3, ".set": 4, ".gdf": 5}
    skip_annot_suffixes = (
        "-hypnogram.edf",
        "_hypnogram.edf",
        "-annot.fif",
        "_annot.fif",
        "-annot.fif.gz",
        "_annot.fif.gz",
    )
    root = Path(root_dir)
    if not root.exists():
        return {}

    def _rec_sort_key(p: Path) -> tuple[int, str]:
        low = p.name.lower()
        ext = ".fif" if low.endswith(".fif.gz") else p.suffix.lower()
        return (ext_priority.get(ext, 9), low)

    raw_candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in filenames:
            if fn.startswith("."):
                continue
            low = fn.lower()
            if low.endswith(skip_annot_suffixes):
                continue
            fp = Path(dirpath) / fn
            if low.endswith(".fif.gz") or fp.suffix.lower() in SUPPORTED_EEG_EXTS:
                raw_candidates.append(fp)

    dedup: dict[tuple[str, str], Path] = {}
    for rf in sorted(raw_candidates, key=_rec_sort_key):
        dkey = (os.path.normcase(str(rf.parent)), get_recording_stem(rf).lower())
        dedup.setdefault(dkey, rf)

    catalog_items: list[tuple[str, Path]] = []
    for rp in dedup.values():
        try:
            rel = rp.relative_to(root)
            label = rp.name if rel.parent == Path(".") else rel.as_posix()
        except ValueError:
            label = rp.name
        catalog_items.append((label, rp))

    catalog_items.sort(key=lambda kv: (natural_sort_key(kv[0]), kv[0]))
    return dict(catalog_items)


def detect_vhdr_metadata(vhdr_path: str | Path) -> dict[str, object]:
    """Read SamplingInterval and header metadata from a BrainVision .vhdr file."""
    meta: dict[str, object] = {"sfreq": None, "marker_file": None, "data_file": None}
    try:
        with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                ls = line.strip()
                low = ls.lower()
                if low.startswith("samplinginterval="):
                    val = float(ls.split("=", 1)[1].strip())
                    if val > 0:
                        meta["sfreq"] = int(round(1_000_000.0 / val))
                elif low.startswith("markerfile="):
                    meta["marker_file"] = ls.split("=", 1)[1].strip()
                elif low.startswith("datafile="):
                    df_val = ls.split("=", 1)[1].strip()
                    if df_val:
                        meta["data_file"] = re.split(r"[/\\]", df_val)[-1].strip()
    except Exception:
        pass
    return meta


def read_brainvision_markers(filepath: str | Path) -> pd.DataFrame:
    """Parse a BrainVision .vmrk or .Markers file into a chronologically sorted DataFrame."""
    rows: list[list[str]] = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line_s = line.strip()
            if not line_s or line_s.startswith(";") or line_s.startswith("["):
                continue
            if line_s.startswith(("Sampling rate", "Brain Vision", "DataFile", "MarkerFile", "Codepage", "Type")):
                continue
            date_field = ""
            if re.match(r"^Mk\d+=", line_s):
                line_s = line_s.split("=", 1)[1]
            parts = [p.strip() for p in line_s.split(",")]
            if len(parts) >= 5:
                if len(parts) >= 6:
                    date_field = parts[5]
                rows.append(parts[:5] + [date_field])

    cols = ["Type", "Description", "Position", "Length", "Channel", "Date"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(1)
    df["Type"] = df["Type"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Channel"] = df["Channel"].astype(str).str.strip()
    return df.dropna(subset=["Position"]).sort_values("Position", kind="stable").reset_index(drop=True)


def extract_recording_events(eeg_path: str | Path, raw_hdr, sfreq: float) -> tuple[pd.DataFrame, Path | None]:
    """Extract BrainVision .vmrk or embedded MNE annotations + BIDS *_events.tsv into unified event table."""
    p = Path(eeg_path)
    sf = float(sfreq) if sfreq and float(sfreq) > 0 else float(raw_hdr.info["sfreq"])
    if p.suffix.lower() == ".vhdr":
        vhdr_meta = detect_vhdr_metadata(p)
        vmrk_path = None
        vmrk_cands: list[Path] = []
        if vhdr_meta.get("marker_file"):
            mf = str(vhdr_meta["marker_file"])
            vmrk_cands.append(p.parent / mf)
            vmrk_cands.append(p.parent / Path(re.split(r"[/\\]", mf)[-1]))
        vmrk_cands.append(p.with_suffix(".vmrk"))
        for vc in vmrk_cands:
            if vc.exists() and vc.is_file():
                vmrk_path = vc
                break
        if vmrk_path is None and p.parent.exists():
            stem_low = p.stem.lower()
            for f in sorted(p.parent.iterdir()):
                if f.is_file() and f.suffix.lower() == ".vmrk" and f.stem.lower() == stem_low:
                    vmrk_path = f
                    break
        if vmrk_path is not None and vmrk_path.exists():
            return read_brainvision_markers(vmrk_path), vmrk_path
        return pd.DataFrame(columns=["Type", "Description", "Position", "Length", "Channel", "Date"]), None

    rows: list[list[object]] = []
    seen: set[tuple[int, str]] = set()

    def _add_row(rtype: str, desc: str, pos: float, length: float, chan: str = "All") -> None:
        d = str(desc or "").strip()
        if not d:
            return
        ip = max(1, int(round(pos)))
        il = max(1, int(round(length)))
        k = (ip, d)
        if k in seen:
            return
        seen.add(k)
        rows.append([str(rtype), d, ip, il, str(chan or "All"), ""])

    for ann in (getattr(raw_hdr, "annotations", None) or []):
        desc = str(ann.get("description", "")).strip()
        if not desc:
            continue
        onset = float(ann.get("onset", 0.0) or 0.0)
        dur = float(ann.get("duration", 0.0) or 0.0)
        pos = max(1, int(round(onset * sf)) + 1)
        length = max(1, int(round(dur * sf)))
        rtype = "Sleep Stage" if desc.lower().startswith("sleep stage") else "Annotation"
        chs = ann.get("ch_names") or ()
        ch_str = ";".join(str(c) for c in chs) if chs else "All"
        _add_row(rtype, desc, pos, length, ch_str)

    events_source = None
    rec_stem = get_recording_stem(p)
    bids_base = re.sub(r"_(eeg|ieeg|meg|psg)$", "", rec_stem, flags=re.IGNORECASE)
    tsv_cands: list[Path] = []
    for cand_name in (f"{bids_base}_events.tsv", f"{rec_stem}_events.tsv"):
        cp = p.parent / cand_name
        if cp not in tsv_cands:
            tsv_cands.append(cp)
    for tp in tsv_cands:
        if tp.is_file():
            events_source = tp
            try:
                tdf = pd.read_csv(tp, sep="\t", dtype=str, keep_default_na=False)
                tcols = {str(c).strip().lower(): c for c in tdf.columns}
                on_c = tcols.get("onset")
                dur_c = tcols.get("duration")
                desc_c = tcols.get("trial_type") or tcols.get("description") or tcols.get("value")
                stat_c = tcols.get("status")
                if on_c and desc_c:
                    for _, tr in tdf.iterrows():
                        dval = str(tr.get(desc_c, "")).strip()
                        if not dval or dval.lower() == "n/a":
                            continue
                        try:
                            ons = float(str(tr.get(on_c, "0")).strip())
                        except Exception:
                            continue
                        try:
                            durs = float(str(tr.get(dur_c, "0")).strip()) if dur_c else 0.0
                        except Exception:
                            durs = 0.0
                        st_val = str(tr.get(stat_c, "")).strip().lower() if stat_c else ""
                        is_stage_row = dval.lower().startswith("sleep stage") or st_val in (
                            "auto",
                            "manual",
                            "wake override",
                            "flagged",
                            "imported",
                            "unscored",
                        )
                        rtype = "Sleep Stage" if is_stage_row else "Event"
                        pos = max(1, int(round(ons * sf)) + 1)
                        length = max(1, int(round(durs * sf)))
                        _add_row(rtype, dval, pos, length, "All")
            except Exception:
                pass
            break

    cols = ["Type", "Description", "Position", "Length", "Channel", "Date"]
    out_df = pd.DataFrame(rows, columns=cols)
    if out_df.empty:
        return out_df, events_source
    out_df["Position"] = pd.to_numeric(out_df["Position"], errors="coerce")
    out_df["Length"] = pd.to_numeric(out_df["Length"], errors="coerce").fillna(1)
    return out_df.dropna(subset=["Position"]).sort_values("Position", kind="stable").reset_index(drop=True), events_source


def _parse_stage_and_ma(token: object) -> tuple[str | None, bool]:
    s = str(token or "").strip()
    s = re.sub(r"^Sleep\s*stage\s*[:\s]+", "", s, flags=re.IGNORECASE).strip()
    has_ma = bool(re.search(r"\s*\+\s*MA\s*$", s, re.IGNORECASE))
    if has_ma:
        s = re.sub(r"\s*\+\s*MA\s*$", "", s, flags=re.IGNORECASE).strip()
    canon = _STAGE_ALIASES.get(s.upper())
    return canon, has_ma


def parse_saved_scoring_file(
    filename: str,
    raw_bytes: bytes,
    n_epochs: int,
    default_fs: float,
) -> tuple[list[str], set[int], dict[str, object]]:
    """Parse saved .Markers/.vmrk, 2-col .txt, BIDS *_events.tsv, MNE *_annotations.csv, or Rich *_hypnogram.csv."""
    text = bytes(raw_bytes).decode("utf-8", errors="ignore").lstrip("\ufeff")
    fn = str(filename or "uploaded_file")
    fn_low = fn.lower()

    stages = ["UNS"] * int(n_epochs)
    ma_indices: set[int] = set()
    lines = text.splitlines()
    first_nonempty = next((ln.strip() for ln in lines if ln.strip()), "")
    first_low = first_nonempty.lower()
    text_low = text.lower()

    is_bids_tsv = fn_low.endswith(".tsv") or (
        "\t" in first_nonempty and "onset" in first_low and ("trial_type" in first_low or "description" in first_low)
    )
    is_csv_table = (not is_bids_tsv) and (
        fn_low.endswith(".csv")
        or (
            "," in first_nonempty
            and (
                ("epoch" in first_low and "stage" in first_low)
                or ("onset" in first_low and "description" in first_low)
            )
        )
    )
    is_markers = (not is_bids_tsv) and (not is_csv_table) and (
        fn_low.endswith((".markers", ".vmrk"))
        or "sleep stage" in text_low
        or "[marker infos]" in text_low
        or "type, description" in text_low
    )

    n_rows_found = 0
    n_out_of_range = 0

    if is_bids_tsv:
        fmt = ".tsv"
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        field_map = {str(f).strip().lower(): f for f in (reader.fieldnames or [])}
        onset_col = field_map.get("onset")
        stage_col = field_map.get("trial_type") or field_map.get("description") or field_map.get("stage")
        ma_col = field_map.get("micro_arousal") or field_map.get("has_ma") or field_map.get("ma")
        status_col = field_map.get("status")
        for row in reader:
            if status_col and str(row.get(status_col, "")).strip().lower() == "event":
                continue
            if not stage_col:
                continue
            st_canon, has_ma0 = _parse_stage_and_ma(row.get(stage_col, ""))
            if st_canon is None:
                continue
            ep_idx = n_rows_found
            if onset_col:
                try:
                    ep_idx = int(round(float(row.get(onset_col, "")) / 30.0))
                except Exception:
                    pass
            n_rows_found += 1
            has_ma1 = bool(ma_col and str(row.get(ma_col, "")).strip().upper() in ("1", "TRUE", "YES", "MA", "+MA"))
            has_ma = bool(has_ma0 or has_ma1)
            if 0 <= ep_idx < n_epochs:
                stages[ep_idx] = st_canon
                if has_ma:
                    ma_indices.add(ep_idx)
                else:
                    ma_indices.discard(ep_idx)
            else:
                n_out_of_range += 1
        if n_rows_found == 0:
            raise ValueError("No valid sleep stage rows found in the uploaded .tsv file.")

    elif is_csv_table:
        fmt = ".csv"
        reader = csv.DictReader(io.StringIO(text))
        field_map = {str(f).strip().lower(): f for f in (reader.fieldnames or [])}
        rows_list = list(reader)

        if "epoch" in field_map and ("stage" in field_map or "stage_int" in field_map):
            ep_col = field_map["epoch"]
            st_col = field_map.get("stage") or field_map.get("stage_int")
            ma_col = field_map.get("has_ma") or field_map.get("micro_arousal") or field_map.get("ma")
            for ri, row in enumerate(rows_list):
                st_canon, has_ma0 = _parse_stage_and_ma(row.get(st_col, ""))
                if st_canon is None:
                    continue
                try:
                    ep_idx = int(float(str(row.get(ep_col, ri + 1)).strip())) - 1
                except Exception:
                    ep_idx = ri
                n_rows_found += 1
                has_ma1 = bool(ma_col and str(row.get(ma_col, "")).strip().upper() in ("1", "TRUE", "YES", "MA", "+MA"))
                has_ma = bool(has_ma0 or has_ma1)
                if 0 <= ep_idx < n_epochs:
                    stages[ep_idx] = st_canon
                    if has_ma:
                        ma_indices.add(ep_idx)
                    else:
                        ma_indices.discard(ep_idx)
                else:
                    n_out_of_range += 1
        elif "onset" in field_map and ("description" in field_map or "trial_type" in field_map):
            on_col = field_map["onset"]
            desc_col = field_map.get("description") or field_map.get("trial_type")
            parsed_onsets: list[tuple[str, object]] = []
            for row in rows_list:
                raw_on = str(row.get(on_col, "")).strip()
                try:
                    parsed_onsets.append(("float", float(raw_on)))
                except Exception:
                    try:
                        parsed_onsets.append(("ts", datetime.datetime.fromisoformat(raw_on.replace("Z", "+00:00"))))
                    except Exception:
                        parsed_onsets.append(("none", 0.0))
            ts_vals = [v for k, v in parsed_onsets if k == "ts"]
            t0 = min(ts_vals) if ts_vals else None
            for row, (okind, oval) in zip(rows_list, parsed_onsets):
                st_canon, has_ma = _parse_stage_and_ma(row.get(desc_col, ""))
                if st_canon is None or okind == "none":
                    continue
                sec = float(oval) if okind == "float" else (oval - t0).total_seconds()  # type: ignore[operator]
                ep_idx = int(round(sec / 30.0))
                n_rows_found += 1
                if 0 <= ep_idx < n_epochs:
                    stages[ep_idx] = st_canon
                    if has_ma:
                        ma_indices.add(ep_idx)
                    else:
                        ma_indices.discard(ep_idx)
                else:
                    n_out_of_range += 1
        if n_rows_found == 0:
            raise ValueError("No valid sleep stage rows found in the uploaded .csv file.")

    elif is_markers:
        fmt = ".vmrk" if fn_low.endswith(".vmrk") else ".Markers"
        markers_fs = float(default_fs)
        for hl in lines[:30]:
            hls = hl.strip()
            m_hz = re.search(r"Sampling\s*rate\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", hls, re.IGNORECASE)
            if m_hz:
                val_hz = float(m_hz.group(1))
                if val_hz > 0:
                    markers_fs = val_hz
                    break
            m_ms = re.search(r"SamplingInterval\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", hls, re.IGNORECASE)
            if m_ms:
                val_ms = float(m_ms.group(1))
                if val_ms > 0:
                    markers_fs = 1000.0 / val_ms
                    break
            if hls.startswith("SamplingInterval="):
                try:
                    val_us = float(hls.split("=", 1)[1].strip())
                    if val_us > 0:
                        markers_fs = 1_000_000.0 / val_us
                        break
                except Exception:
                    pass

        for line in lines:
            ls = line.strip()
            if not ls or ls.startswith((";", "[")):
                continue
            if ls.startswith(("Sampling rate", "Brain Vision", "DataFile", "MarkerFile", "Codepage", "Type")):
                continue
            if re.match(r"^Mk\d+=", ls):
                ls = ls.split("=", 1)[1]
            parts = [p.strip().replace(r"\1", ",") for p in ls.split(",")]
            if len(parts) < 4 or "sleep stage" not in parts[0].lower():
                continue
            try:
                pos = float(parts[2])
            except Exception:
                continue
            n_rows_found += 1
            ep_idx = int(round((pos - 1.0) / (30.0 * markers_fs)))
            if 0 <= ep_idx < n_epochs:
                st_canon, has_ma = _parse_stage_and_ma(parts[1])
                if st_canon is not None:
                    stages[ep_idx] = st_canon
                if has_ma:
                    ma_indices.add(ep_idx)
                else:
                    ma_indices.discard(ep_idx)
            else:
                n_out_of_range += 1

        if n_rows_found == 0:
            raise ValueError("No 'Sleep Stage' entries found in the uploaded .Markers/.vmrk file.")
    else:
        fmt = ".txt"
        row_idx = 0
        first_seen = False
        for line in lines:
            ls = line.strip()
            if not ls or ls.startswith(("#", ";", "//")):
                continue
            toks = [t for t in re.split(r"[\t,\s]+", ls) if t]
            if not toks:
                continue
            st_canon, has_ma0 = _parse_stage_and_ma(toks[0])
            if not first_seen:
                first_seen = True
                if st_canon is None:
                    continue
            if st_canon is None:
                continue
            has_ma1 = len(toks) >= 2 and toks[1].strip().upper() in ("1", "TRUE", "YES", "MA", "+MA")
            has_ma = bool(has_ma0 or has_ma1)
            n_rows_found += 1
            if 0 <= row_idx < n_epochs:
                stages[row_idx] = st_canon
                if has_ma:
                    ma_indices.add(row_idx)
            else:
                n_out_of_range += 1
            row_idx += 1

        if n_rows_found == 0:
            raise ValueError("No valid sleep stage rows found in the uploaded .txt file.")

    warns: list[str] = []
    if n_out_of_range > 0:
        warns.append(f"{n_out_of_range} epoch(s) exceeded the {n_epochs}-epoch recording length and were ignored")
    if fmt in (".txt", ".tsv", ".csv") and n_rows_found < n_epochs:
        warns.append(f"File had {n_rows_found}/{n_epochs} rows (remaining marked UNS)")

    n_scored = sum(1 for s in stages if s != "UNS")
    meta = {
        "filename": fn,
        "format": fmt,
        "n_rows_found": n_rows_found,
        "n_scored": n_scored,
        "n_unscored": int(n_epochs) - n_scored,
        "n_ma": len(ma_indices),
        "warning": "; ".join(warns),
    }
    return stages, ma_indices, meta


def parse_onera_session_json(raw_bytes: bytes, n_epochs: int) -> dict[str, object]:
    """Parse an Onera .onera.json companion session file into normalized edits, MA flags, and config."""
    text = bytes(raw_bytes).decode("utf-8", errors="ignore").lstrip("\ufeff")
    try:
        data = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid .onera.json session file: {exc}")
    if not isinstance(data, dict) or ("manual_edits" not in data and "final_stages" not in data):
        raise ValueError("Invalid .onera.json session file.")

    n_ep = int(n_epochs)
    out_of_range = 0
    manual_edits: dict[int, str] = {}
    for k, v in dict(data.get("manual_edits") or {}).items():
        try:
            idx = int(k)
        except Exception:
            continue
        canon = _STAGE_ALIASES.get(str(v or "").strip().upper())
        if canon is None:
            continue
        if 0 <= idx < n_ep:
            manual_edits[idx] = canon
        else:
            out_of_range += 1

    ma_flags: set[int] = set()
    for mi in list(data.get("ma_flags") or []):
        try:
            idx = int(mi)
        except Exception:
            continue
        if 0 <= idx < n_ep:
            ma_flags.add(idx)
        else:
            out_of_range += 1

    raw_finals = data.get("final_stages")
    final_stages: list[str] | None = None
    if isinstance(raw_finals, list):
        norm_finals = [
            _STAGE_ALIASES.get(str(s or "").strip().upper(), "UNS")
            for s in raw_finals[:n_ep]
        ]
        if len(norm_finals) < n_ep:
            norm_finals.extend(["UNS"] * (n_ep - len(norm_finals)))
        final_stages = norm_finals

    saved_n_raw = data.get("n_epochs")
    saved_n = int(saved_n_raw) if isinstance(saved_n_raw, (int, float)) else n_ep
    warns: list[str] = []
    if out_of_range > 0 or saved_n > n_ep:
        warns.append(f"Saved session had {saved_n} epochs vs {n_ep} in recording; extra indices ignored")

    return {
        "manual_edits": manual_edits,
        "ma_flags": ma_flags,
        "final_stages": final_stages,
        "yasa_config": dict(data.get("yasa_config") or {}),
        "scoring_mode": str(data.get("scoring_mode") or "auto"),
        "stem": str(data.get("stem") or ""),
        "n_epochs": saved_n,
        "warning": "; ".join(warns),
    }


def parse_scoring_upload(
    filename: str,
    raw_bytes: bytes,
    n_epochs: int,
    fs_marker: float,
    output_dir: Path | None = None,
    eeg_path: Path | None = None,
    lookup_sidecar: bool = True,
) -> tuple[list[str], set[int], dict[str, object], dict[str, object] | None, bool]:
    """Parse any supported scoring upload plus optional companion .onera.json sidecar."""
    up_name = str(filename or "uploaded_scoring")
    if up_name.lower().endswith(".json"):
        sidecar = parse_onera_session_json(raw_bytes or b"", n_epochs)
        if sidecar.get("final_stages") is not None:
            stages = list(sidecar["final_stages"])  # type: ignore[arg-type]
        else:
            stages = ["UNS"] * n_epochs
        for ek, ev in dict(sidecar.get("manual_edits") or {}).items():  # type: ignore[arg-type]
            if 0 <= int(ek) < n_epochs:
                stages[int(ek)] = str(ev)
        ma_set = set(int(i) for i in (sidecar.get("ma_flags") or set()))  # type: ignore[arg-type]
        n_sc = sum(1 for s in stages if s != "UNS")
        meta: dict[str, object] = {
            "filename": up_name,
            "format": ".onera.json",
            "sidecar_used": True,
            "n_rows_found": n_epochs,
            "n_scored": n_sc,
            "n_unscored": n_epochs - n_sc,
            "n_ma": len(ma_set),
            "warning": str(sidecar.get("warning") or ""),
        }
        return stages, ma_set, meta, sidecar, True

    imp_stages, imp_ma, meta = parse_saved_scoring_file(up_name, raw_bytes or b"", n_epochs, fs_marker)
    sidecar = None
    if lookup_sidecar:
        up_stem = Path(up_name).stem
        cand_dirs: list[Path] = []
        if output_dir is not None:
            cand_dirs.append(output_dir / ".onera_sessions")
        if eeg_path is not None:
            cand_dirs.append(eeg_path.parent / ".onera_sessions")
        seen_dirs: set[str] = set()
        for sdir in cand_dirs:
            skey = str(sdir)
            if skey in seen_dirs:
                continue
            seen_dirs.add(skey)
            cand_path = sdir / f"{up_stem}.onera.json"
            if cand_path.is_file():
                try:
                    sidecar = parse_onera_session_json(cand_path.read_bytes(), n_epochs)
                    break
                except Exception:
                    pass

    warn_m = [str(meta["warning"])] if meta.get("warning") else []
    if sidecar is not None:
        meta["sidecar_used"] = True
        if meta.get("format") == ".txt" and sidecar.get("final_stages") is not None:
            sc_finals: list[str] = sidecar["final_stages"]  # type: ignore[assignment]
            for si in range(min(n_epochs, len(sc_finals))):
                if imp_stages[si] == "W" and sc_finals[si] == "UNS":
                    imp_stages[si] = "UNS"
            n_sc = sum(1 for s in imp_stages if s != "UNS")
            meta["n_scored"] = n_sc
            meta["n_unscored"] = n_epochs - n_sc
        if sidecar.get("warning"):
            warn_m.append(str(sidecar["warning"]))
    else:
        meta["sidecar_used"] = False
    meta["warning"] = "; ".join(warn_m)
    return imp_stages, set(imp_ma), meta, sidecar, False


def compute_inter_rater_metrics(
    stages1: list[str],
    stages2: list[str],
    ma1: set[int],
    ma2: set[int],
) -> dict[str, object]:
    """Compute overall agreement percentage and Cohen's kappa across scored epochs of two raters."""
    n_ep = min(len(stages1), len(stages2))
    pairs: list[tuple[str, str]] = []
    n_disagree = 0
    n_ma_diff = 0
    n_unscored_either = 0
    for i in range(n_ep):
        s1 = stages1[i]
        s2 = stages2[i]
        if (i in ma1) != (i in ma2):
            n_ma_diff += 1
        if s1 == "UNS" or s2 == "UNS":
            n_unscored_either += 1
            if s1 != s2:
                n_disagree += 1
            continue
        pairs.append((s1, s2))
        if s1 != s2:
            n_disagree += 1

    n_both = len(pairs)
    n_agree_both = sum(1 for a, b in pairs if a == b)
    if n_both <= 0:
        return {
            "n_epochs": n_ep,
            "n_both_scored": 0,
            "n_agree_both": 0,
            "n_stage_disagree": n_disagree,
            "n_ma_diff": n_ma_diff,
            "n_unscored_either": n_unscored_either,
            "agree_pct": None,
            "kappa": None,
        }

    po = n_agree_both / float(n_both)
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    pe = 0.0
    for lbl in labels:
        p1 = sum(1 for a, _ in pairs if a == lbl) / float(n_both)
        p2 = sum(1 for _, b in pairs if b == lbl) / float(n_both)
        pe += p1 * p2
    if pe >= 1.0 - 1e-12:
        kappa = 1.0 if po >= 1.0 - 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "n_epochs": n_ep,
        "n_both_scored": n_both,
        "n_agree_both": n_agree_both,
        "n_stage_disagree": n_disagree,
        "n_ma_diff": n_ma_diff,
        "n_unscored_either": n_unscored_either,
        "agree_pct": round(po * 100.0, 1),
        "kappa": round(float(kappa), 3),
    }


def classify_sleep_channels(ch_names: list[str], ref_chs: list[str] | None = None) -> dict[str, str]:
    """Classify PSG channel names into MNE physiological types ('eeg', 'eog', 'emg', 'ecg', 'misc')."""
    ref_set = set(ref_chs or [])
    ch_types: dict[str, str] = {}
    for ch in ch_names:
        up = ch.upper().strip()
        up_bare = re.sub(r"^EEG\s+", "", up).strip()
        if ch in ref_set or up_bare in {"M1", "M2", "A1", "A2", "TP9", "TP10"}:
            ch_types[ch] = "misc"
        elif any(k in up for k in ("EMG", "CHIN", "SUBMENT", "MENTALIS", "TIB", "LEG")):
            # Check EMG before EOG so names like 'EMG_eye' stay typed as EMG rather than EOG
            ch_types[ch] = "emg"
        elif any(k in up for k in ("EOG", "LOC", "ROC", "HEOG", "VEOG")) or bool(
            re.search(r"(?:^|[\s_\-])E[12](?:$|[\s_\-])", up)
        ):
            ch_types[ch] = "eog"
        elif any(k in up for k in ("ECG", "EKG", "HEART", "PULSE")):
            ch_types[ch] = "ecg"
        elif any(
            k in up
            for k in (
                "RESP",
                "FLOW",
                "NASAL",
                "ORAL",
                "THERM",
                "THOR",
                "ABDO",
                "BELT",
                "EFFORT",
                "SNORE",
                "MIC",
                "SOUND",
                "SPO2",
                "SAO2",
                "OXIM",
                "PLETH",
                "POS",
                "BODY",
                "TEMP",
                "RECTAL",
                "EVENT",
                "MARKER",
                "STATUS",
                "STIM",
                "TRIG",
                "SCR",
                "EDA",
                "GSR",
                "PPG",
                "LIGHT",
                "BATT",
                "ACC",
                "GYRO",
                "CPAP",
                "PRESS",
            )
        ):
            ch_types[ch] = "misc"
        else:
            ch_types[ch] = "eeg"
    return ch_types


def _norm_ch_token(ch_name: str) -> tuple[str, str]:
    u = re.sub(r"^(?:EEG|EOG|EMG)\s+", "", str(ch_name).upper().strip()).strip()
    base = re.sub(r"-(?:M[12]|A[12]|REF|AVG|LE|AR)$", "", u).strip()
    return u, base


def _pick_preferred(candidates: list[str], prefs: tuple[str, ...]) -> str | None:
    exact = {c.upper().strip(): c for c in candidates}
    norm_exact: dict[str, str] = {}
    norm_base: dict[str, str] = {}
    for c in candidates:
        u, base = _norm_ch_token(c)
        norm_exact.setdefault(u, c)
        norm_base.setdefault(base, c)
    for p in prefs:
        pu = p.upper().strip()
        if pu in exact:
            return exact[pu]
    for p in prefs:
        pu, pb = _norm_ch_token(p)
        if pu in norm_exact:
            return norm_exact[pu]
        if pb in norm_base:
            return norm_base[pb]
    for p in prefs:
        pu = p.upper().strip()
        for c in candidates:
            if pu in c.upper():
                return c
    return candidates[0] if candidates else None


def _emg_priority(ch_name: str) -> tuple[int, str]:
    u = ch_name.upper()
    if "CHIN" in u or "SUBMENT" in u or "MENTALIS" in u:
        return (0, u)
    if u in ("EMG", "EMG1", "EMG2") or u.startswith("EMG_CH"):
        return (1, u)
    if "EYE" not in u and "JAW" not in u and "ORBIC" not in u and "ZYGO" not in u and "LEG" not in u and "TIB" not in u:
        return (2, u)
    return (3, u)


def compute_channel_defaults(ch_names: list[str], marker_labels: list[str]) -> dict[str, object]:
    """Compute smart Wake Override, reference, primary YASA, visual, and manual channel defaults."""
    default_lo = ""
    for cand in marker_labels:
        if cand.lower().replace("_", "").replace(" ", "").replace("-", "") in ("lightsoff", "lightoff"):
            default_lo = cand
            break

    def_start_1 = "Start_DreamReport" if "Start_DreamReport" in marker_labels else ""
    def_stop_1 = "Stop_DreamReport" if "Stop_DreamReport" in marker_labels else ""

    ch_list = list(ch_names)
    ch_types = classify_sleep_channels(ch_list, ref_chs=[])
    eeg_candidates = [ch for ch in ch_list if ch_types.get(ch) == "eeg"] or ch_list
    eog_candidates = [ch for ch in ch_list if ch_types.get(ch) == "eog"]
    emg_candidates = [ch for ch in ch_list if ch_types.get(ch) == "emg"]

    standalone_mastoids = {"M1", "M2", "A1", "A2"}
    all_bipolar_eeg = bool(eeg_candidates) and all("-" in c for c in eeg_candidates)
    default_refs = (
        []
        if all_bipolar_eeg
        else [ch for ch in ch_list if ch.strip().upper() in standalone_mastoids][:2]
    )

    eeg_prefs = ("C4", "C4-M1", "C4-A1", "C3", "C3-M2", "C3-A2", "Fpz-Cz", "F4", "F4-M1", "F3", "F3-M2", "Pz-Oz", "Cz", "Fz")
    sec_eeg_prefs = ("C3", "C3-M2", "C3-A2", "C4", "C4-M1", "Pz-Oz", "F3", "F3-M2", "F4", "F4-M1", "Cz", "Fz")
    eog_prefs = ("EOG_H", "EOGli", "EOGre", "EOG_V", "EOG horizontal", "E1-M2", "E2-M1", "E1", "E2", "EOG1", "EOG2", "EOG", "LOC", "ROC")

    default_eeg = _pick_preferred(eeg_candidates, eeg_prefs)
    default_eog = _pick_preferred(eog_candidates, eog_prefs) or ""
    emg_sorted = sorted(emg_candidates, key=_emg_priority)
    default_emg = emg_sorted[0] if emg_sorted else ""

    non_ref_chs = [c for c in ch_list if c not in default_refs]
    secondary_eeg = _pick_preferred(
        [c for c in eeg_candidates if c != default_eeg and c not in default_refs],
        sec_eeg_prefs,
    )

    default_manual_chs: list[str] = []
    for mc in (default_eeg, secondary_eeg, default_eog, default_emg):
        if mc and mc not in default_refs and mc not in default_manual_chs:
            default_manual_chs.append(mc)
    if not default_manual_chs:
        default_manual_chs = non_ref_chs[:4] or ch_list[:4]

    default_visual_chs: list[str] = []
    vis_non_ref_eegs = [c for c in eeg_candidates if c not in default_refs]
    for site in ("F3", "F4", "C3", "C4"):
        for c in vis_non_ref_eegs:
            if _norm_ch_token(c)[1] == site and c not in default_visual_chs:
                default_visual_chs.append(c)
                break
    if not default_visual_chs:
        for c in (default_eeg, secondary_eeg):
            if c and c in vis_non_ref_eegs and c not in default_visual_chs:
                default_visual_chs.append(c)
    for c in [ec for ec in eog_candidates if ec not in default_refs][:2]:
        if c not in default_visual_chs:
            default_visual_chs.append(c)
    for c in [mc for mc in emg_sorted if mc not in default_refs][:2]:
        if c not in default_visual_chs:
            default_visual_chs.append(c)
    if not default_visual_chs:
        default_visual_chs = non_ref_chs[:4] or ch_list[:4]

    default_custom = [c for c in eeg_candidates if _norm_ch_token(c)[1] in {"F3", "F4", "C3", "C4"}][:4] or eeg_candidates[:2]

    return {
        "lightsoff": default_lo,
        "ignore_start_1": def_start_1,
        "ignore_stop_1": def_stop_1,
        "refs": default_refs,
        "primary_eeg": default_eeg or (eeg_candidates[0] if eeg_candidates else ""),
        "eog": default_eog,
        "emg": default_emg,
        "visual_chs": default_visual_chs,
        "manual_chs": default_manual_chs,
        "custom_eegs": default_custom,
        "eeg_candidates": eeg_candidates,
        "non_ref_chs": non_ref_chs or ch_list,
    }


def resolve_preset_channels(
    preset_label: str,
    raw_ch_names: list[str],
    fallback_eeg: str | None,
    custom_eegs: list[str] | None = None,
) -> list[str]:
    """Resolve multi-electrode preset or custom selection onto concrete recording channel names."""
    if preset_label == "Custom EEG Channels":
        targets = list(custom_eegs or [])
    else:
        targets = MULTI_ELECTRODE_PRESETS.get(preset_label, [])
    upper_lookup = {c.upper().strip(): c for c in raw_ch_names}
    norm_lookup: dict[str, str] = {}
    for c in raw_ch_names:
        _, nb = _norm_ch_token(c)
        norm_lookup.setdefault(nb, c)
    resolved: list[str] = []
    for t in targets:
        tu = str(t).upper().strip()
        match = upper_lookup.get(tu) or norm_lookup.get(tu)
        if match and match not in resolved:
            resolved.append(match)
    if not resolved and fallback_eeg and fallback_eeg in raw_ch_names:
        resolved = [fallback_eeg]
    return resolved


def format_hms(total_seconds: float) -> str:
    """Format elapsed seconds into HH:MM:SS."""
    sec = max(0, int(round(float(total_seconds))))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def is_calibration_marker(desc: str) -> bool:
    """Identify pre-sleep biocalibration markers (eyes, teeth, blinks) to omit from plots."""
    return bool(re.search(r"eyes?|teeth|blink|calib|biocal", str(desc), re.IGNORECASE))


def normalize_stage(label: object) -> str:
    """Normalize YASA sleep stage predictions into canonical Onera stage codes."""
    s = str(label).strip().upper()
    if s in ("WAKE", "W"):
        return "W"
    if s in ("R", "REM"):
        return "REM"
    return s


def align_proba_df(proba_df: pd.DataFrame) -> np.ndarray:
    """Align YASA probability DataFrame columns to STAGE_ORDER."""
    renamed = proba_df.rename(columns=normalize_stage)
    return renamed.reindex(columns=STAGE_ORDER, fill_value=0.0).to_numpy(dtype=float)


def match_marker_intervals(df_vmrk: pd.DataFrame, start_label: str, stop_label: str) -> list[tuple[float, float]]:
    """Pair sequential Start/Stop markers chronologically into sample-position intervals."""
    if df_vmrk.empty or not start_label or not stop_label or start_label == "(None)" or stop_label == "(None)":
        return []
    events = df_vmrk[df_vmrk["Description"].isin([start_label, stop_label])].sort_values("Position", kind="stable")
    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for _, er in events.iterrows():
        desc = er["Description"]
        pos = float(er["Position"])
        if desc == start_label and pending_start is None:
            pending_start = pos
        elif desc == stop_label and pending_start is not None:
            if pos > pending_start:
                intervals.append((pending_start, pos))
            pending_start = None
    return intervals


def compute_wake_overrides(
    df_vmrk: pd.DataFrame,
    n_epochs: int,
    fs_marker: float,
    lightsoff_label: str | None,
    ignore_pairs: list[tuple[str, str]],
) -> tuple[int, np.ndarray, list[str], pd.DataFrame]:
    """Compute per-epoch forced Wake override sources and shaded interval records."""
    ep_len_samps = int(round(30.0 * fs_marker))
    p_starts = np.array([1 + ep * ep_len_samps for ep in range(n_epochs)], dtype=np.int64)
    p_mids = p_starts + (ep_len_samps / 2.0)
    ov_sources = [""] * n_epochs
    ov_records: list[dict[str, float]] = []

    if lightsoff_label and lightsoff_label != "(None — Start from Sample 1)" and not df_vmrk.empty:
        lo_matches = df_vmrk[df_vmrk["Description"] == lightsoff_label]
        if not lo_matches.empty:
            lo_pos = float(lo_matches.iloc[0]["Position"])
            for ep in range(n_epochs):
                if p_mids[ep] < lo_pos:
                    ov_sources[ep] = f"LightsOff (< {lightsoff_label})"
            ov_records.append({"t0_sec": 0.0, "t1_sec": max(0.0, (lo_pos - 1.0) / fs_marker)})

    for p_i, (st_lbl, sp_lbl) in enumerate(ignore_pairs, 1):
        for s_pos, e_pos in match_marker_intervals(df_vmrk, st_lbl, sp_lbl):
            for ep in range(n_epochs):
                if s_pos < p_mids[ep] < e_pos:
                    tag = f"Ignore Pair #{p_i} ({st_lbl} → {sp_lbl})"
                    ov_sources[ep] = tag if not ov_sources[ep] else f"{ov_sources[ep]}; {tag}"
            ov_records.append({
                "t0_sec": max(0.0, (s_pos - 1.0) / fs_marker),
                "t1_sec": max(0.0, (e_pos - 1.0) / fs_marker),
            })

    return ep_len_samps, p_starts, ov_sources, pd.DataFrame(ov_records)


def extract_vmrk_events(
    df_vmrk: pd.DataFrame,
    n_epochs: int,
    fs_marker: float,
    meas_date: datetime.datetime | None,
) -> tuple[dict[int, list[tuple[float, str]]], list[dict[str, object]]]:
    """Extract per-epoch and hypnogram recording events (excluding calibration & sleep stages)."""
    ep_events: dict[int, list[tuple[float, str]]] = {}
    pl_events: list[dict[str, object]] = []
    if df_vmrk.empty or n_epochs <= 0:
        return ep_events, pl_events
    max_sec = float(n_epochs) * 30.0
    for _, vr in df_vmrk.iterrows():
        vtype = str(vr["Type"]).strip()
        vtype_low = vtype.lower()
        if "sleep stage" in vtype_low or "new segment" in vtype_low:
            continue
        vdesc = str(vr["Description"]).strip()
        if not vdesc or is_calibration_marker(vdesc):
            continue
        t_sec = max(0.0, (float(vr["Position"]) - 1.0) / fs_marker)
        ep_idx = int(t_sec // 30.0)
        sec_in_ep = min(30.0, round(t_sec - (ep_idx * 30.0), 2))
        if 0 <= ep_idx < n_epochs:
            ep_events.setdefault(ep_idx, []).append((sec_in_ep, vdesc))
        if 0.0 <= t_sec <= max_sec:
            ts_str = (
                (meas_date + datetime.timedelta(seconds=t_sec)).strftime("%H:%M:%S")
                if meas_date is not None
                else format_hms(t_sec)
            )
            pl_events.append({
                "x_ep": (t_sec / 30.0) + 1.0,
                "ep_num": min(n_epochs, max(1, ep_idx + 1)),
                "t_sec": t_sec,
                "sec_in_ep": sec_in_ep,
                "desc": vdesc,
                "ts_str": ts_str,
                "is_lights": ("lights" in vdesc.lower()),
            })
    return ep_events, pl_events


def fast_resample_to_100(data_v: np.ndarray, orig_sf: float) -> tuple[np.ndarray, float]:
    """Resample multi-channel array directly to 100 Hz using exact rational polyphase filtering."""
    sf = float(orig_sf)
    if abs(sf - 100.0) <= 1e-3:
        return data_v, sf
    if abs(sf - round(sf)) < 1e-3:
        frac = Fraction(100, int(round(sf)))
    else:
        frac = Fraction(100.0 / sf).limit_denominator(1000)
    resampled = scipy_signal.resample_poly(data_v, frac.numerator, frac.denominator, axis=-1)
    return resampled, 100.0


def prepare_display_waveforms(
    data_v: np.ndarray,
    sfreq: float,
    ch_types: list[str],
    ch_names: list[str] | None = None,
    on_channel_done=None,
) -> tuple[np.ndarray, float, int, list[str], list[float]]:
    """Zero-phase SOS display filtering at native sampling rate + robust percentile auto-scaling."""
    sf = float(sfreq)
    nyq = sf / 2.0
    types_list = [str(t).lower() for t in ch_types]
    n_ch = len(types_list)
    n_s = int(data_v.shape[1]) if n_ch > 0 else 0
    w_uv = np.empty((n_ch, n_s), dtype=np.float32)

    sos_eeg = None
    sos_emg = None
    sos_ecg = None
    if n_s > 30:
        if nyq > 1.5:
            hi = min(35.0, nyq - 1.0)
            if hi > 0.3:
                sos_eeg = scipy_signal.butter(4, [0.3 / nyq, hi / nyq], btype="bandpass", output="sos")
            sos_ecg = scipy_signal.butter(4, 0.5 / nyq, btype="highpass", output="sos")
        if nyq > 12.0:
            hi_emg = min(70.0, nyq - 1.0)
            if hi_emg > 10.0:
                sos_emg = scipy_signal.butter(4, [10.0 / nyq, hi_emg / nyq], btype="bandpass", output="sos")

    ep_samps_out = int(round(30.0 * sf))
    stride = 10 if n_s >= 100 else 1
    ylims: list[float] = []
    for ci, ctype in enumerate(types_list):
        raw64 = np.asarray(data_v[ci], dtype=np.float64)
        if ctype in ("eeg", "eog"):
            ch64 = raw64 * 1e6
            if sos_eeg is not None:
                ch64 = scipy_signal.sosfiltfilt(sos_eeg, ch64)
        elif ctype == "emg":
            ch64 = raw64 * 1e6
            if sos_emg is not None:
                ch64 = scipy_signal.sosfiltfilt(sos_emg, ch64)
        elif ctype == "ecg":
            ch64 = raw64 * 1e6
            if sos_ecg is not None:
                ch64 = scipy_signal.sosfiltfilt(sos_ecg, ch64)
        else:
            fin = np.isfinite(raw64)
            med = float(np.median(raw64[fin])) if np.any(fin) else 0.0
            ch64 = np.where(fin, raw64 - med, 0.0)
        w_uv[ci] = ch64.astype(np.float32, copy=False)
        del raw64, ch64

        if n_s > 0:
            abs_s = np.abs(w_uv[ci, ::stride])
            fin_abs = abs_s[np.isfinite(abs_s)]
        else:
            fin_abs = np.empty(0, dtype=np.float32)

        if ctype == "emg":
            p88 = float(np.percentile(fin_abs, 88)) if fin_abs.size > 0 else 15.0
            ylims.append(max(15.0, min(40.0, p88 * 1.3)))
        elif ctype == "ecg":
            p99 = float(np.percentile(fin_abs, 99)) if fin_abs.size > 0 else 200.0
            ylims.append(max(100.0, min(5000.0, p99 * 1.3)))
        elif ctype not in ("eeg", "eog"):
            p99 = float(np.percentile(fin_abs, 99)) if fin_abs.size > 0 else 0.0
            ylims.append(max(1e-9, p99 * 1.3) if p99 > 0.0 else 1.0)
        else:
            p92 = float(np.percentile(fin_abs, 92)) if fin_abs.size > 0 else 50.0
            ylims.append(max(50.0, min(85.0, p92 * 1.25)))

        if on_channel_done is not None:
            cname = ch_names[ci] if ch_names and ci < len(ch_names) else f"Ch {ci + 1}"
            on_channel_done(ci, n_ch, cname)

    return w_uv, sf, ep_samps_out, types_list, ylims


def format_epoch_overlay(
    idx: int,
    auto_stage: str,
    final_stage: str,
    is_manual_edit: bool,
    has_ma: bool,
    base_status: str,
    base_hover: str,
    time_start: str,
    time_end: str,
    conf_pct: float,
    agr_pct: float,
    flag_reason: str,
    is_manual_mode: bool,
    is_consensus_mode: bool = False,
    s1_label: str = "Scorer 1",
    s2_label: str = "Scorer 2",
    s1_stage: str = "",
    s2_stage: str = "",
    s1_ma: bool = False,
    s2_ma: bool = False,
    has_yasa: bool = True,
    yasa_stage: str = "",
) -> tuple[str, str]:
    """Compute canonical Status and Hover_Text for an epoch given edits and +MA state."""
    conf_finite = conf_pct is not None and pd.notna(conf_pct) and np.isfinite(float(conf_pct))
    agr_finite = agr_pct is not None and pd.notna(agr_pct) and np.isfinite(float(agr_pct))

    if is_consensus_mode:
        if is_manual_edit:
            st_str = (
                f"Resolved: {final_stage}"
                if s1_stage != s2_stage
                else f"Override: {auto_stage} → {final_stage}"
            )
        else:
            st_str = base_status
        s1_disp = f"{s1_stage}{'+MA' if s1_ma else ''}"
        s2_disp = f"{s2_stage}{'+MA' if s2_ma else ''}"
        yasa_line = (
            f"YASA: <b>{html_lib.escape(str(yasa_stage))}</b> ({float(conf_pct):.1f}%)<br>"
            if has_yasa and yasa_stage and conf_finite
            else ""
        )
        hover = (
            f"<b>Ep #{idx + 1}</b> ({time_start}–{time_end})<br>"
            f"Stage: <b>{final_stage}</b> ({html_lib.escape(str(st_str))})<br>"
            f"{html_lib.escape(s1_label)}: <b>{html_lib.escape(s1_disp)}</b> | "
            f"{html_lib.escape(s2_label)}: <b>{html_lib.escape(s2_disp)}</b><br>"
            f"{yasa_line}"
            f"Note: {html_lib.escape(str(flag_reason))}"
        )
    elif is_manual_edit:
        if is_manual_mode:
            st_str = (
                f"Manual: {final_stage}"
                if (auto_stage == "UNS" or final_stage == auto_stage)
                else f"Override: {auto_stage} → Manual: {final_stage}"
            )
            hover = (
                f"<b>Ep #{idx + 1}</b> ({time_start}–{time_end})<br>"
                f"Stage: <b>{final_stage}</b> ({html_lib.escape(str(st_str))})<br>"
                f"Note: {html_lib.escape(str(flag_reason))}"
            )
        else:
            st_str = f"Auto: {auto_stage} → Manual: {final_stage}" if final_stage != auto_stage else f"Manual: {final_stage}"
            conf_line = (
                f"Confidence: <b>{float(conf_pct):.1f}%</b> | Agree: {float(agr_pct):.0f}%<br>"
                if conf_finite and agr_finite
                else ""
            )
            hover = (
                f"<b>Ep #{idx + 1}</b> ({time_start}–{time_end})<br>"
                f"Stage: <b>{final_stage}</b> ({html_lib.escape(str(st_str))})<br>"
                f"{conf_line}"
                f"Note: {html_lib.escape(str(flag_reason))}"
            )
    else:
        st_str = base_status
        hover = base_hover

    if has_ma:
        hover = hover.replace(f"Stage: <b>{final_stage}</b>", f"Stage: <b>{final_stage} +MA</b>", 1)
    return st_str, hover


def build_scored_epochs_df(
    bundle: dict[str, object],
    manual_edits: dict[int, str],
    ma_flags: set[int],
) -> pd.DataFrame:
    """Materialize the active scored epochs DataFrame from frozen bundle + edits + MA flags."""
    base_df: pd.DataFrame = bundle["base_epochs_df"]  # type: ignore[assignment]
    if base_df.empty:
        return pd.DataFrame()
    scoring_mode = str(bundle.get("scoring_mode", "auto"))
    is_manual_mode = scoring_mode == "manual"
    is_consensus_mode = scoring_mode == "consensus"
    cons_meta: dict[str, object] = dict(bundle.get("consensus_meta") or {}) if is_consensus_mode else {}  # type: ignore[arg-type]
    s1_lbl = str(cons_meta.get("scorer1_label") or "Scorer 1")
    s2_lbl = str(cons_meta.get("scorer2_label") or "Scorer 2")
    has_yasa = bool(cons_meta.get("has_yasa", not is_manual_mode))

    df = base_df.copy()
    n_ep = len(df)
    session_ma = {int(i) for i in (ma_flags or set()) if 0 <= int(i) < n_ep}
    auto_stages = df["Auto_Stage"].tolist()
    raw_yasa_stages = df["Raw_YASA_Stage"].tolist() if "Raw_YASA_Stage" in df.columns else [""] * n_ep
    s1_stages = df["Scorer1_Stage"].tolist() if "Scorer1_Stage" in df.columns else [""] * n_ep
    s2_stages = df["Scorer2_Stage"].tolist() if "Scorer2_Stage" in df.columns else [""] * n_ep
    s1_mas = df["Scorer1_MA"].tolist() if "Scorer1_MA" in df.columns else [False] * n_ep
    s2_mas = df["Scorer2_MA"].tolist() if "Scorer2_MA" in df.columns else [False] * n_ep
    final_stages = list(auto_stages)
    is_manual = [False] * n_ep
    base_statuses = df["Base_Status"].tolist()
    base_hovers = df["Base_Hover"].tolist()
    status_labels = list(base_statuses)
    hover_texts = list(base_hovers)

    t_starts = df["Time_Start"].tolist()
    t_ends = df["Time_End"].tolist()
    confs = df["Confidence_Pct"].tolist()
    agrs = df["Agreement_Pct"].tolist()
    reasons_col = df["Flag_Reason"].tolist()

    for ep_i, ov in (manual_edits or {}).items():
        idx = int(ep_i)
        if 0 <= idx < n_ep:
            final_stages[idx] = str(ov)
            is_manual[idx] = True

    for idx in range(n_ep):
        if is_manual[idx] or idx in session_ma:
            st_str, hv = format_epoch_overlay(
                idx=idx,
                auto_stage=auto_stages[idx],
                final_stage=final_stages[idx],
                is_manual_edit=is_manual[idx],
                has_ma=(idx in session_ma),
                base_status=base_statuses[idx],
                base_hover=base_hovers[idx],
                time_start=t_starts[idx],
                time_end=t_ends[idx],
                conf_pct=confs[idx],
                agr_pct=agrs[idx],
                flag_reason=reasons_col[idx],
                is_manual_mode=is_manual_mode,
                is_consensus_mode=is_consensus_mode,
                s1_label=s1_lbl,
                s2_label=s2_lbl,
                s1_stage=str(s1_stages[idx]),
                s2_stage=str(s2_stages[idx]),
                s1_ma=bool(s1_mas[idx]),
                s2_ma=bool(s2_mas[idx]),
                has_yasa=has_yasa,
                yasa_stage=str(raw_yasa_stages[idx]),
            )
            status_labels[idx] = st_str
            hover_texts[idx] = hv

    df["Final_Stage"] = final_stages
    df["Final_Stage_AASM"] = [AASM_STAGE_MAP.get(s, s) for s in final_stages]
    df["Final_Stage_RK"] = [RK_STAGE_MAP.get(s, s) for s in final_stages]
    df["Is_Manual_Edit"] = is_manual
    df["Has_MA"] = [(i in session_ma) for i in range(n_ep)]
    df["Status"] = status_labels
    df["Hover_Text"] = hover_texts
    return df


def build_hypnogram_events(
    plot_events: list[dict[str, object]],
    n_epochs: int,
) -> tuple[list[list[object]], list[list[object]]]:
    """Prepare bounded/binned recording event lists for 60 FPS hypnogram canvas rendering."""
    marker_events: list[dict[str, object]] = []
    line_events: list[dict[str, object]] = []
    if not plot_events:
        return [], []

    if len(plot_events) <= 250:
        for ev in plot_events:
            ed_esc = html_lib.escape(str(ev["desc"]))
            marker_events.append({
                "x_ep": float(ev["x_ep"]),  # type: ignore[arg-type]
                "is_lights": bool(ev.get("is_lights")),
                "is_landmark": bool(ev.get("is_lights")),
                "hover": f"<b>{ed_esc}</b><br>Epoch #{ev['ep_num']} (+{ev['sec_in_ep']:.1f}s)<br>Time: {html_lib.escape(str(ev['ts_str']))}",
            })
    else:
        desc_counts: dict[str, int] = {}
        for ev in plot_events:
            d = str(ev.get("desc", ""))
            desc_counts[d] = desc_counts.get(d, 0) + 1
        ev_bin_w = max(1.0, float(n_epochs) / 200.0)
        binned: dict[tuple[str, int], list[object]] = {}
        for ev in plot_events:
            d = str(ev.get("desc", ""))
            cnt_d = desc_counts.get(d, 0)
            is_l = bool(ev.get("is_lights"))
            if is_l or cnt_d <= 25:
                ed_esc = html_lib.escape(d)
                marker_events.append({
                    "x_ep": float(ev["x_ep"]),  # type: ignore[arg-type]
                    "is_lights": is_l,
                    "is_landmark": bool(is_l or cnt_d <= 10),
                    "hover": f"<b>{ed_esc}</b><br>Epoch #{ev['ep_num']} (+{ev['sec_in_ep']:.1f}s)<br>Time: {html_lib.escape(str(ev['ts_str']))}",
                })
            else:
                bk = (d, int(float(ev["x_ep"]) // ev_bin_w))  # type: ignore[arg-type]
                entry = binned.get(bk)
                if entry is None:
                    binned[bk] = [dict(ev), 1]
                else:
                    entry[1] = int(entry[1]) + 1
        for ev0, cnt in binned.values():  # type: ignore[misc]
            d_esc = html_lib.escape(str(ev0.get("desc", "")))
            ts_esc = html_lib.escape(str(ev0["ts_str"]))
            hv = (
                f"<b>{d_esc}</b> (×{cnt} near Ep #{ev0['ep_num']})<br>First: {ts_esc}"
                if cnt > 1
                else f"<b>{d_esc}</b><br>Epoch #{ev0['ep_num']} (+{ev0['sec_in_ep']:.1f}s)<br>Time: {ts_esc}"
            )
            marker_events.append({
                "x_ep": float(ev0["x_ep"]),
                "is_lights": bool(ev0.get("is_lights")),
                "is_landmark": False,
                "hover": hv,
            })
        marker_events.sort(key=lambda e: float(e["x_ep"]))

    if len(marker_events) <= 120:
        line_events = marker_events
    else:
        line_gap = max(1.0, float(n_epochs) / 120.0)
        last_lx = -1e9
        for mev in marker_events:
            if mev["is_landmark"] or (float(mev["x_ep"]) - last_lx) >= line_gap:
                line_events.append(mev)
                if not mev["is_landmark"]:
                    last_lx = float(mev["x_ep"])

    m_ev_out = [[round(float(e["x_ep"]), 3), (1 if e["is_lights"] else 0), e["hover"]] for e in marker_events]
    l_ev_out = [[round(float(e["x_ep"]), 3), (1 if e["is_lights"] else 0)] for e in line_events]
    return m_ev_out, l_ev_out


def normalize_export_fmt_key(fmt_choice: str) -> str:
    """Map UI export format label or key to canonical format key."""
    s = str(fmt_choice or "").strip()
    low = s.lower()
    if low in ("markers", "txt", "bids_tsv", "mne_csv", "rich_csv", "vmrk"):
        return low
    if "2-column" in low:
        return "txt"
    if "bids" in low:
        return "bids_tsv"
    if "mne annotations" in low:
        return "mne_csv"
    if "rich hypnogram" in low:
        return "rich_csv"
    if ".vmrk" in low:
        return "vmrk"
    return "markers"


def suggest_export_filename(fmt_key: str, stem: str) -> tuple[str, bool]:
    """Return suggested filename and whether the format supports embedding original events."""
    bids_base = re.sub(r"_(eeg|ieeg|meg|psg)$", "", stem, flags=re.IGNORECASE)
    if fmt_key == "txt":
        return f"{stem}.txt", False
    if fmt_key == "bids_tsv":
        return f"{bids_base}_events.tsv", True
    if fmt_key == "mne_csv":
        return f"{stem}_annotations.csv", True
    if fmt_key == "rich_csv":
        return f"{stem}_hypnogram.csv", False
    if fmt_key == "vmrk":
        return f"{stem}.vmrk", True
    return f"{stem}.Markers", True


def resolve_export_filename(fmt_key: str, raw_name: str, stem: str) -> str:
    """Ensure the output filename carries the appropriate extension for fmt_key."""
    fname = (raw_name or "").strip() or stem
    fl = fname.lower()
    if fmt_key == "txt" and not fl.endswith(".txt"):
        return f"{fname}.txt"
    if fmt_key == "bids_tsv" and not fl.endswith(".tsv"):
        return f"{fname}_events.tsv"
    if fmt_key == "mne_csv" and not fl.endswith(".csv"):
        return f"{fname}_annotations.csv"
    if fmt_key == "rich_csv" and not fl.endswith(".csv"):
        return f"{fname}_hypnogram.csv"
    if fmt_key == "vmrk" and not fl.endswith(".vmrk"):
        return f"{fname}.vmrk"
    if fmt_key == "markers" and not fl.endswith(".markers"):
        return f"{fname}.Markers"
    return fname


def build_export_content(
    fmt_key: str,
    use_rk: bool,
    include_events: bool,
    scored_bundle: dict[str, object],
    scored_df: pd.DataFrame,
) -> tuple[str, int, list[str], int, str]:
    """Generate serialized sleep-staging export text, row count, unscored count, and preview snippet."""
    fs_out = float(scored_bundle["committed_fs"])  # type: ignore[arg-type]
    vmrk_df: pd.DataFrame = scored_bundle["committed_vmrk"]  # type: ignore[assignment]
    interval_ms = 1000.0 / fs_out
    has_ma_list = scored_df["Has_MA"].tolist() if "Has_MA" in scored_df.columns else [False] * len(scored_df)
    stage_col = "Final_Stage_RK" if use_rk else "Final_Stage_AASM"
    txt_code_map = {
        "W": 0,
        "WAKE": 0,
        "N1": 1,
        "S1": 1,
        "N2": 2,
        "S2": 2,
        "N3": 3,
        "S3": 3,
        "S4": 4 if use_rk else 3,
        "REM": 5,
        "R": 5,
        "ART": 0,
        "MT": 0,
        "UNS": 0,
        "?": 0,
    }

    def _iter_orig_events() -> list[tuple[str, str, int, int, str]]:
        if not include_events or vmrk_df.empty:
            return []
        out: list[tuple[str, str, int, int, str]] = []
        for vt, vd, vp, vl, vc in zip(
            vmrk_df["Type"].tolist(),
            vmrk_df["Description"].tolist(),
            vmrk_df["Position"].tolist(),
            vmrk_df["Length"].tolist(),
            vmrk_df["Channel"].tolist(),
        ):
            vtype = str(vt).strip()
            if "sleep stage" in vtype.lower() or "new segment" in vtype.lower():
                continue
            vdesc = str(vd).strip()
            if not vdesc:
                continue
            vpos = max(1, int(round(float(vp))))
            vlen = max(1, int(round(float(vl)))) if str(vl).strip() else 1
            vchan = str(vc).strip() or "All"
            out.append((vtype, vdesc, vpos, vlen, vchan))
        return out

    if fmt_key == "txt":
        all_lines = [
            f"{txt_code_map.get(str(st).strip().upper(), 0)}\t{1 if bool(ma) else 0}"
            for st, ma in zip(scored_df["Final_Stage"].tolist(), has_ma_list)
        ]
        export_text = "\n".join(all_lines) + "\n"
        n_rows = len(all_lines)
    elif fmt_key == "bids_tsv":
        bids_rows: list[tuple[float, int, str]] = []
        for _, vdesc, vpos, vlen, _ in _iter_orig_events():
            ons = max(0.0, (float(vpos) - 1.0) / fs_out)
            dur = max(0.0, float(vlen) / fs_out)
            clean_d = vdesc.replace("\t", " ").replace("\n", " ")
            bids_rows.append((ons, 0, f"{ons:.3f}\t{dur:.3f}\t{clean_d}\tn/a\tn/a\tEvent"))
        confs = scored_df["Confidence_Pct"].tolist()
        stats = scored_df["Status"].tolist()
        for s_pos, s_len, s_lbl, s_ma, cf, st_lbl in zip(
            scored_df["Position_Sample"].tolist(),
            scored_df["Length_Samples"].tolist(),
            scored_df[stage_col].tolist(),
            has_ma_list,
            confs,
            stats,
        ):
            ons = max(0.0, (float(s_pos) - 1.0) / fs_out)
            dur = float(s_len) / fs_out
            cf_str = f"{float(cf):.1f}" if pd.notna(cf) and np.isfinite(float(cf)) else "n/a"
            st_clean = str(st_lbl).replace("\t", " ").replace("\n", " ").strip()
            ma_int = 1 if bool(s_ma) else 0
            bids_rows.append((ons, 1, f"{ons:.3f}\t{dur:.3f}\t{str(s_lbl).strip()}\t{ma_int}\t{cf_str}\t{st_clean}"))
        bids_rows.sort(key=lambda x: (x[0], x[1]))
        all_lines = ["onset\tduration\ttrial_type\tmicro_arousal\tconfidence\tstatus"] + [r[2] for r in bids_rows]
        export_text = "\n".join(all_lines) + "\n"
        n_rows = len(bids_rows)
    elif fmt_key == "mne_csv":
        ann_tuples: list[tuple[float, int, float, str]] = []
        for _, vdesc, vpos, vlen, _ in _iter_orig_events():
            ons = max(0.0, (float(vpos) - 1.0) / fs_out)
            dur = max(0.0, float(vlen) / fs_out)
            ann_tuples.append((ons, 0, dur, vdesc))
        for s_pos, s_len, s_lbl, s_ma in zip(
            scored_df["Position_Sample"].tolist(),
            scored_df["Length_Samples"].tolist(),
            scored_df[stage_col].tolist(),
            has_ma_list,
        ):
            ons = max(0.0, (float(s_pos) - 1.0) / fs_out)
            dur = float(s_len) / fs_out
            lbl_out = f"{str(s_lbl).strip()}+MA" if bool(s_ma) else str(s_lbl).strip()
            ann_tuples.append((ons, 1, dur, lbl_out))
        ann_tuples.sort(key=lambda x: (x[0], x[1]))
        orig_time = scored_bundle.get("committed_meas_date")
        if orig_time is not None and getattr(orig_time, "tzinfo", None) is None:
            orig_time = orig_time.replace(tzinfo=datetime.timezone.utc)  # type: ignore[union-attr]
        ann_obj = mne.Annotations(
            onset=[t[0] for t in ann_tuples],
            duration=[t[2] for t in ann_tuples],
            description=[t[3] for t in ann_tuples],
            orig_time=orig_time,  # type: ignore[arg-type]
        )
        export_text = ann_obj.to_data_frame().to_csv(index=False)
        all_lines = export_text.splitlines()
        n_rows = len(ann_tuples)
    elif fmt_key == "rich_csv":
        rich_df = pd.DataFrame({
            "Epoch": scored_df["Epoch_Num"].tolist(),
            "Time_Start": scored_df["Time_Start"].tolist(),
            "Time_End": scored_df["Time_End"].tolist(),
            "Stage": [str(s).strip() for s in scored_df[stage_col].tolist()],
            "Stage_Int": [txt_code_map.get(str(s).strip().upper(), 0) for s in scored_df["Final_Stage"].tolist()],
            "Has_MA": [bool(m) for m in has_ma_list],
            "Confidence_Pct": scored_df["Confidence_Pct"].tolist(),
            "Agreement_Pct": scored_df["Agreement_Pct"].tolist(),
            "Status": scored_df["Status"].tolist(),
        })
        export_text = rich_df.to_csv(index=False)
        all_lines = export_text.splitlines()
        n_rows = len(rich_df)
    elif fmt_key == "vmrk":
        records: list[tuple[int, int, str]] = []
        seg_date = ""
        meas_date = scored_bundle.get("committed_meas_date")
        if meas_date is not None:
            try:
                seg_date = meas_date.strftime("%Y%m%d%H%M%S000000")  # type: ignore[union-attr]
            except Exception:
                seg_date = ""
        has_seg = False
        if include_events and not vmrk_df.empty:
            date_col = vmrk_df["Date"].tolist() if "Date" in vmrk_df.columns else [""] * len(vmrk_df)
            for vt, vd, vp, vl, vc, vdt in zip(
                vmrk_df["Type"].tolist(),
                vmrk_df["Description"].tolist(),
                vmrk_df["Position"].tolist(),
                vmrk_df["Length"].tolist(),
                vmrk_df["Channel"].tolist(),
                date_col,
            ):
                vt_s = str(vt).strip().replace(",", r"\1")
                if "sleep stage" in vt_s.lower():
                    continue
                vd_s = str(vd).strip().replace(",", r"\1")
                vp_i = max(1, int(round(float(vp))))
                vl_i = max(1, int(round(float(vl)))) if str(vl).strip() else 1
                vc_s = str(vc).strip()
                ch_num = vc_s if vc_s.isdigit() else "0"
                prio = 0 if "new segment" in vt_s.lower() else 1
                if prio == 0:
                    has_seg = True
                dt_suffix = f",{str(vdt).strip()}" if str(vdt).strip() else ""
                records.append((vp_i, prio, f"{vt_s},{vd_s},{vp_i},{vl_i},{ch_num}{dt_suffix}"))
        if not has_seg:
            dt_suffix = f",{seg_date}" if seg_date else ""
            records.append((1, 0, f"New Segment,,1,1,0{dt_suffix}"))
        for s_pos, s_len, s_lbl, s_ma in zip(
            scored_df["Position_Sample"].tolist(),
            scored_df["Length_Samples"].tolist(),
            scored_df[stage_col].tolist(),
            has_ma_list,
        ):
            ip = int(s_pos)
            lbl_out = (f"{str(s_lbl).strip()}+MA" if bool(s_ma) else str(s_lbl).strip()).replace(",", r"\1")
            records.append((ip, 2, f"Sleep Stage,{lbl_out},{ip},{int(s_len)},0"))
        records.sort(key=lambda x: (x[0], x[1]))
        fs_disp = int(round(fs_out)) if abs(fs_out - round(fs_out)) < 1e-6 else fs_out
        data_ref = str(scored_bundle.get("committed_data_file") or f"{scored_bundle['committed_stem']}.eeg")
        if data_ref.lower().endswith(".vhdr"):
            data_ref = f"{data_ref[:-5]}.eeg"
        header_lines = [
            "Brain Vision Data Exchange Marker File, Version 1.0",
            f"; Sampling rate: {fs_disp} Hz, SamplingInterval: {interval_ms:g} ms",
            "",
            "[Common Infos]",
            "Codepage=UTF-8",
            f"DataFile={data_ref}",
            "",
            "[Marker Infos]",
            "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,",
            "; <Size in data points>, <Channel number (0 = marker is related to all channels)>",
        ]
        mk_lines = [f"Mk{mi}={r[2]}" for mi, r in enumerate(records, 1)]
        all_lines = header_lines + mk_lines
        export_text = "\n".join(all_lines) + "\n"
        n_rows = len(records)
    else:
        records = []
        if include_events and not vmrk_df.empty:
            for vt, vd, vp, vl, vc in zip(
                vmrk_df["Type"].tolist(),
                vmrk_df["Description"].tolist(),
                vmrk_df["Position"].tolist(),
                vmrk_df["Length"].tolist(),
                vmrk_df["Channel"].tolist(),
            ):
                vtype = str(vt).strip().replace(",", ";")
                if "sleep stage" in vtype.lower():
                    continue
                vdesc = str(vd).strip().replace(",", ";")
                vpos = int(round(float(vp)))
                vlen = int(round(float(vl))) if str(vl).strip() else 1
                vchan = str(vc).strip().replace(",", ";") or "All"
                priority = 0 if "new segment" in vtype.lower() else 1
                records.append((vpos, priority, f"{vtype}, {vdesc}, {vpos}, {vlen}, {vchan}"))
        else:
            records.append((1, 0, "New Segment, , 1, 1, All"))

        for s_pos, s_len, s_lbl, s_ma in zip(
            scored_df["Position_Sample"].tolist(),
            scored_df["Length_Samples"].tolist(),
            scored_df[stage_col].tolist(),
            has_ma_list,
        ):
            ip = int(s_pos)
            lbl_out = f"{str(s_lbl).strip()}+MA" if bool(s_ma) else str(s_lbl).strip()
            records.append((ip, 2, f"Sleep Stage, {lbl_out}, {ip}, {int(s_len)}, All"))

        records.sort(key=lambda x: (x[0], x[1]))
        fs_disp = int(round(fs_out)) if abs(fs_out - round(fs_out)) < 1e-6 else fs_out
        header_lines = [
            f"Sampling rate: {fs_disp} Hz, SamplingInterval: {interval_ms:g} ms",
            "Type, Description, Position, Length, Channel",
        ]
        all_lines = header_lines + [r[2] for r in records]
        export_text = "\n".join(all_lines) + "\n"
        n_rows = len(records)

    n_unscored = int((scored_df["Final_Stage"] == "UNS").sum())
    preview_head = "\n".join(all_lines[:15])
    preview_tail = "\n".join(all_lines[-5:]) if len(all_lines) > 20 else ""
    preview_str = preview_head + (f"\n...\n{preview_tail}" if preview_tail else "")
    return export_text, n_rows, all_lines, n_unscored, preview_str


def build_sidecar_payload(
    scored_bundle: dict[str, object],
    scored_df: pd.DataFrame,
    manual_edits: dict[int, str],
    ma_flags: set[int],
    target_stem: str,
) -> dict[str, object]:
    """Construct .onera.json companion sidecar payload preserving edits, +MA tags, and YASA config."""
    exp_mode = str(scored_bundle.get("scoring_mode", "auto"))
    fs_out = float(scored_bundle["committed_fs"])  # type: ignore[arg-type]
    payload: dict[str, object] = {
        "stem": target_stem,
        "recording_id": str(scored_bundle["committed_recording_id"]),
        "scoring_mode": exp_mode,
        "n_epochs": int(len(scored_df)),
        "fs": fs_out,
        "manual_edits": {
            str(int(k)): str(v)
            for k, v in sorted((manual_edits or {}).items(), key=lambda kv: int(kv[0]))
        },
        "ma_flags": sorted(int(i) for i in (ma_flags or set())),
        "final_stages": [str(s) for s in scored_df["Final_Stage"].tolist()],
        "yasa_config": {
            "active_eegs": list(scored_bundle.get("active_eegs_used") or []),  # type: ignore[arg-type]
            "eog": scored_bundle.get("committed_eog"),
            "emg": scored_bundle.get("committed_emg"),
            "refs": list(scored_bundle.get("committed_refs") or []),  # type: ignore[arg-type]
            "thresh": float(scored_bundle.get("committed_thresh", 60.0)),  # type: ignore[arg-type]
        },
    }
    if exp_mode == "consensus":
        payload["consensus_config"] = dict(scored_bundle.get("consensus_meta") or {})  # type: ignore[arg-type]
    return payload
