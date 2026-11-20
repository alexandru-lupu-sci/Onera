"""Generate synthetic 12-channel BrainVision polysomnography demo recordings for Onera.

Creates realistic multi-stage sleep EEG/EOG/EMG recordings (.vhdr, .eeg, .vmrk)
paired with scored .Markers files covering Wake, N1, N2, N3, and REM sleep,
50 Hz line noise, common-mode mastoid drift, LightsOff / dream-report / stimulus
events, micro-arousal (+MA) annotations, and controlled epoch artifacts so every
workflow in Onera_SleepScorer.py and Onera_PreAwakening.py can be tested locally.

Usage:
    uv run python generate_demo_data.py
    uv run python generate_demo_data.py --out demo_data --recordings 2 --sfreq 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
from scipy import signal


CH_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "EOG1", "EMG1", "M1", "M2"]

# Spatial topography weights across the 8 cortical EEG derivations:
# (sdelta/fdelta, theta, alpha, sigma, beta)
SPATIAL_WEIGHTS = {
    "F3": (1.25, 1.10, 0.72, 0.90, 1.05),
    "F4": (1.22, 1.08, 0.74, 0.88, 1.05),
    "C3": (1.00, 1.00, 0.95, 1.15, 1.00),
    "C4": (1.02, 1.00, 0.95, 1.12, 1.00),
    "P3": (0.85, 0.95, 1.15, 1.10, 0.95),
    "P4": (0.85, 0.95, 1.15, 1.08, 0.95),
    "O1": (0.70, 0.90, 1.38, 0.65, 0.92),
    "O2": (0.72, 0.90, 1.35, 0.65, 0.92),
}


def _pink_noise(rng: np.random.Generator, n: int, sfreq: float, beta: float = 1.35) -> np.ndarray:
    """Synthesize unit-variance 1/f^beta noise in the frequency domain."""
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    spec = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    scale = np.zeros_like(freqs)
    scale[1:] = 1.0 / (freqs[1:] ** (beta / 2.0))
    y = np.fft.irfft(spec * scale, n=n)
    return y / (np.std(y) + 1e-12)


def _band_noise(
    rng: np.random.Generator, n: int, sfreq: float, lo: float, hi: float
) -> np.ndarray:
    """Synthesize unit-variance band-limited noise via 4th-order zero-phase Butterworth filter."""
    nyq = sfreq / 2.0
    hi_eff = min(hi, nyq - 1.0)
    sos = signal.butter(4, [lo, hi_eff], btype="bandpass", fs=sfreq, output="sos")
    x = signal.sosfiltfilt(sos, rng.standard_normal(n))
    return x / (np.std(x) + 1e-12)


def synthesize_psg(
    stage_plan: list[str],
    stim_intervals_sec: list[tuple[float, float]],
    artifact_times_sec: dict[str, float],
    sfreq: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate 12-channel PSG voltage data (in microvolts, shape 12 x n_times)."""
    rng = np.random.default_rng(seed)
    ep_samps = int(round(30.0 * sfreq))
    n_times = len(stage_plan) * ep_samps
    t = np.arange(n_times, dtype=np.float64) / sfreq

    # Shared frequency-band carriers (uV normalized to unit std)
    n_sdelta = _band_noise(rng, n_times, sfreq, 0.5, 2.0)
    n_fdelta = _band_noise(rng, n_times, sfreq, 2.0, 4.0)
    n_theta = _band_noise(rng, n_times, sfreq, 4.0, 7.5)
    n_alpha = _band_noise(rng, n_times, sfreq, 8.5, 11.5)
    n_sigma = _band_noise(rng, n_times, sfreq, 12.0, 14.8)
    n_beta = _band_noise(rng, n_times, sfreq, 16.0, 28.0)
    n_emg = _band_noise(rng, n_times, sfreq, 20.0, 75.0)
    n_eog_slow = _band_noise(rng, n_times, sfreq, 0.25, 0.7)
    n_eog_rem = _band_noise(rng, n_times, sfreq, 0.5, 2.0)
    n_pink_shared = _pink_noise(rng, n_times, sfreq, beta=1.4)

    # Channel envelopes along epoch grid
    n_eeg = 8
    eeg_base = np.zeros((n_eeg, n_times), dtype=np.float64)
    eog = np.zeros(n_times, dtype=np.float64)
    emg = np.zeros(n_times, dtype=np.float64)

    t_ep = np.arange(ep_samps, dtype=np.float64) / sfreq
    step_blink = max(1, int(round(4.5 * sfreq)))
    step_spindle = max(1, int(round(5.5 * sfreq)))
    step_rem = max(1, int(round(5.5 * sfreq)))

    for ei, st in enumerate(stage_plan):
        s0 = ei * ep_samps
        s1 = s0 + ep_samps
        sl = slice(s0, s1)

        if st == "W":
            a_sd, a_fd, a_th, a_al, a_sg, a_be, a_pk = 2.0, 2.0, 5.0, 18.0, 2.0, 10.0, 4.0
            blinks = np.zeros(ep_samps, dtype=np.float64)
            for bt in np.arange(2.0, 28.0, 4.5):
                blinks += 120.0 * np.exp(-0.5 * ((t_ep - bt) / 0.15) ** 2)
            eog[sl] = blinks + 8.0 * n_alpha[sl] + 6.0 * n_eog_slow[sl]
            emg[sl] = 28.0 * n_emg[sl] + 4.0 * n_pink_shared[sl]
        elif st == "N1":
            a_sd, a_fd, a_th, a_al, a_sg, a_be, a_pk = 7.0, 5.0, 10.0, 3.0, 1.5, 1.5, 4.0
            eog[sl] = 28.0 * n_eog_slow[sl] + 4.0 * n_theta[sl]
            emg[sl] = 7.5 * n_emg[sl] + 2.0 * n_pink_shared[sl]
        elif st == "N2":
            a_sd, a_fd, a_th, a_al, a_sg, a_be, a_pk = 14.0, 9.0, 9.0, 2.0, 5.0, 1.2, 5.0
            sp_env = np.zeros(ep_samps, dtype=np.float64)
            for st_s in np.arange(2.0, 28.0, 5.5):
                sp_env += np.exp(-0.5 * ((t_ep - st_s) / 0.40) ** 2)
            kc_center = 10.0 + (ei % 3) * 4.0
            kt = t_ep - kc_center
            kc = -65.0 * np.exp(-0.5 * (kt / 0.14) ** 2) + 35.0 * np.exp(-0.5 * ((kt - 0.38) / 0.18) ** 2)
            spindle_burst = 18.0 * sp_env * n_sigma[sl]
            eog[sl] = 6.0 * n_sdelta[sl] + 4.0 * n_eog_slow[sl]
            emg[sl] = 4.5 * n_emg[sl] + 1.2 * n_pink_shared[sl]
        elif st == "N3":
            a_sd, a_fd, a_th, a_al, a_sg, a_be, a_pk = 56.0, 18.0, 10.0, 2.0, 3.0, 1.0, 8.0
            eog[sl] = 20.0 * n_sdelta[sl] + 4.0 * n_eog_slow[sl]
            emg[sl] = 3.5 * n_emg[sl] + 1.0 * n_pink_shared[sl]
        else:  # REM
            a_sd, a_fd, a_th, a_al, a_sg, a_be, a_pk = 4.0, 5.0, 10.0, 2.5, 1.0, 1.5, 4.0
            rems = np.zeros(ep_samps, dtype=np.float64)
            for rt0 in np.arange(2.5, 27.5, 5.5):
                rt = t_ep - rt0
                sign = 1.0 if int(round(rt0 * 10 + ei)) % 2 == 0 else -1.0
                rems += sign * 45.0 * np.exp(-0.5 * (rt / 0.18) ** 2) * np.sin(2.0 * np.pi * 1.2 * rt)
            eog[sl] = rems + 14.0 * n_eog_rem[sl] + 4.0 * n_theta[sl]
            emg[sl] = 0.7 * n_emg[sl] + 0.3 * n_pink_shared[sl]

        for ci, ch in enumerate(CH_NAMES[:n_eeg]):
            w_del, w_th, w_al, w_sg, w_be = SPATIAL_WEIGHTS[ch]
            sig = (
                (w_del * a_sd) * n_sdelta[sl]
                + (w_del * a_fd) * n_fdelta[sl]
                + (w_th * a_th) * n_theta[sl]
                + (w_al * a_al) * n_alpha[sl]
                + (w_sg * a_sg) * n_sigma[sl]
                + (w_be * a_be) * n_beta[sl]
                + a_pk * n_pink_shared[sl]
            )
            if st == "N2":
                sig = sig + (w_sg * spindle_burst) + (0.9 * w_del * kc)
            eeg_base[ci, sl] = sig

    # Add subtle independent pink sensor noise per EEG electrode
    for ci in range(n_eeg):
        eeg_base[ci] += 1.6 * _pink_noise(rng, n_times, sfreq, beta=1.2)

    # Apply stimulus-related spectral modulation inside stimulation intervals
    # (frontal-central theta & posterior alpha enhancement for clear Split-Screen contrast)
    for t_s0, t_s1 in stim_intervals_sec:
        i0 = max(0, int(round(t_s0 * sfreq)))
        i1 = min(n_times, int(round(t_s1 * sfreq)))
        if i1 > i0:
            fade = np.hanning(max(4, min(i1 - i0, int(round(4.0 * sfreq)))))
            half_f = len(fade) // 2
            env = np.ones(i1 - i0, dtype=np.float64)
            env[:half_f] = fade[:half_f]
            env[-half_f:] = fade[-half_f:]
            sl_stim = slice(i0, i1)
            for ci, ch in enumerate(CH_NAMES[:n_eeg]):
                if ch.startswith(("F", "C")):
                    eeg_base[ci, sl_stim] += env * 9.0 * n_theta[sl_stim]
                if ch.startswith(("P", "O")):
                    eeg_base[ci, sl_stim] += env * 7.5 * n_alpha[sl_stim]

    # Inject controlled transient artifacts inside pre-awakening windows
    if "movement" in artifact_times_sec:
        a_t = artifact_times_sec["movement"]
        c_idx = int(round(a_t * sfreq))
        half_w = int(round(1.2 * sfreq))
        i0, i1 = max(0, c_idx - half_w), min(n_times, c_idx + half_w)
        tt = np.linspace(-1.0, 1.0, i1 - i0)
        burst = 290.0 * np.exp(-2.5 * tt ** 2) * np.sin(2.0 * np.pi * 2.2 * tt)
        eeg_base[0, i0:i1] += burst
        eeg_base[1, i0:i1] += 0.85 * burst

    if "step" in artifact_times_sec:
        s_idx = min(n_times - 10, max(10, int(round(artifact_times_sec["step"] * sfreq))))
        pop = 260.0 * np.exp(-np.arange(int(round(0.4 * sfreq))) / max(1.0, 0.08 * sfreq))
        end_p = min(n_times, s_idx + len(pop))
        eeg_base[2, s_idx:end_p] += pop[: end_p - s_idx]

    if "emg_burst" in artifact_times_sec:
        e_idx = int(round(artifact_times_sec["emg_burst"] * sfreq))
        half_w = int(round(1.0 * sfreq))
        i0, i1 = max(0, e_idx - half_w), min(n_times, e_idx + half_w)
        env_e = np.hanning(i1 - i0)
        emg[i0:i1] += env_e * 95.0 * n_emg[i0:i1]

    # Add common-mode mastoid drift + 50 Hz line noise (demonstrates re-referencing & notch filtering)
    drift = 20.0 * np.sin(2.0 * np.pi * 0.35 * t) + 10.0 * np.sin(2.0 * np.pi * 0.12 * t)
    line_hum = 12.0 * np.sin(2.0 * np.pi * 50.0 * t)
    m1 = drift + 1.0 * _pink_noise(rng, n_times, sfreq, beta=1.0)
    m2 = drift + 1.0 * _pink_noise(rng, n_times, sfreq, beta=1.0)

    out_uv = np.zeros((len(CH_NAMES), n_times), dtype=np.float64)
    out_uv[:n_eeg] = eeg_base + drift[None, :] + line_hum[None, :]
    out_uv[8] = eog + 0.5 * line_hum
    out_uv[9] = emg + 0.6 * line_hum
    out_uv[10] = m1
    out_uv[11] = m2
    return out_uv


def write_brainvision_and_markers(
    out_dir: Path,
    raw_stem: str,
    markers_stem: str,
    sfreq: float,
    data_uv: np.ndarray,
    stage_plan: list[str],
    ma_epochs: set[int],
    exp_events_sec: list[tuple[str, float]],
    crop_offset_sec: float = 0.0,
    start_timestamp: str = "20260115230000000000",
) -> dict[str, Path]:
    """Write INT_16 BrainVision triplet (.vhdr/.eeg/.vmrk) and paired .Markers CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eeg_path = out_dir / f"{raw_stem}.eeg"
    vmrk_path = out_dir / f"{raw_stem}.vmrk"
    vhdr_path = out_dir / f"{raw_stem}.vhdr"
    markers_path = out_dir / f"{markers_stem}.Markers"

    resolution_uv = 0.1
    scaled = np.clip(np.round(data_uv / resolution_uv), -32767, 32767).astype("<i2")
    scaled.T.tofile(eeg_path)

    interval_us = int(round(1_000_000.0 / sfreq))
    ch_lines = "\n".join(
        f"Ch{i + 1}={ch},,{resolution_uv},µV" for i, ch in enumerate(CH_NAMES)
    )
    vhdr_txt = (
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={eeg_path.name}\n"
        f"MarkerFile={vmrk_path.name}\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={len(CH_NAMES)}\n"
        f"SamplingInterval={interval_us}\n"
        "\n[Binary Infos]\n"
        "BinaryFormat=INT_16\n"
        "\n[Channel Infos]\n"
        f"{ch_lines}\n"
    )
    vhdr_path.write_text(vhdr_txt, encoding="utf-8")

    # Sort experimental events chronologically for .vmrk and .Markers
    sorted_events = sorted(exp_events_sec, key=lambda x: x[1])
    vmrk_lines = [
        "Brain Vision Data Exchange Marker File, Version 1.0",
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={eeg_path.name}",
        "",
        "[Marker Infos]",
        f"Mk1=New Segment,,1,1,0,{start_timestamp}",
    ]
    for idx, (desc, t_sec) in enumerate(sorted_events, start=2):
        pos = 1 + int(round(t_sec * sfreq))
        vmrk_lines.append(f"Mk{idx}=Comment,{desc},{pos},1,0")
    vmrk_path.write_text("\n".join(vmrk_lines) + "\n", encoding="utf-8")

    # Build .Markers timeline (uncropped if crop_offset_sec == 0, or LightsOff-cropped otherwise)
    ep_sec = 30.0
    ep_samps = int(round(ep_sec * sfreq))
    marker_rows: list[tuple[int, str, str, int, int]] = []
    for desc, t_sec in sorted_events:
        if t_sec < crop_offset_sec - 1e-6:
            continue
        pos = 1 + int(round((t_sec - crop_offset_sec) * sfreq))
        marker_rows.append((pos, "Comment", desc, pos, 1))

    for ei, st in enumerate(stage_plan):
        ep_t_sec = ei * ep_sec
        if ep_t_sec < crop_offset_sec - 1e-6:
            continue
        pos = 1 + int(round((ep_t_sec - crop_offset_sec) * sfreq))
        label = f"{st}+MA" if ei in ma_epochs else st
        marker_rows.append((pos, "Sleep Stage", label, pos, ep_samps))

    marker_rows.sort(key=lambda r: (r[0], 0 if r[1] != "Sleep Stage" else 1))
    interval_ms = 1000.0 / sfreq
    mk_lines = [
        f"Sampling rate: {int(round(sfreq))} Hz, SamplingInterval: {interval_ms:.3f} ms",
        "Type, Description, Position, Length, Channel",
    ]
    for _, mtype, desc, pos, length in marker_rows:
        mk_lines.append(f"{mtype}, {desc}, {pos}, {length}, All")
    markers_path.write_text("\n".join(mk_lines) + "\n", encoding="utf-8")

    return {
        "vhdr": vhdr_path,
        "vmrk": vmrk_path,
        "eeg": eeg_path,
        "markers": markers_path,
    }


def generate_demo_dataset(
    out_dir: str | Path = "demo_data",
    sfreq: int = 200,
    recordings: int = 2,
    seed: int = 42,
) -> list[dict[str, Path]]:
    """Generate 1 or 2 synthetic demo sleep recordings ready for Onera."""
    if sfreq < 100:
        raise ValueError("--sfreq must be >= 100 Hz so 50 Hz notch and 0.3-45 Hz bandpass stay valid.")
    root = Path(out_dir)
    created: list[dict[str, Path]] = []

    # Recording 1 (flat, 60 min = 120 epochs): W, N1, N2, N3, REM + 3 awakenings & stim events
    plan_rec1 = (
        ["W"] * 6
        + ["N1"] * 6
        + ["N2"] * 18
        + ["N3"] * 12
        + ["W"] * 3
        + ["N1"] * 4
        + ["N2"] * 25
        + ["W"] * 3
        + ["N1"] * 4
        + ["N2"] * 6
        + ["REM"] * 23
        + ["W"] * 10
    )
    assert len(plan_rec1) == 120
    events_rec1 = [
        ("LightsOff", 120.0),
        ("Start_Music_Happy", 1020.0),
        ("Start_DreamReport", 1260.0),
        ("Stop_DreamReport", 1350.0),
        ("Stop_Music_Sad", 1980.0),
        ("Start_DreamReport", 2220.0),
        ("Stop_DreamReport", 2310.0),
        ("Start_Music_Happy", 3060.0),
        ("Start_DreamReport", 3300.0),
        ("Stop_DreamReport", 3390.0),
    ]
    stims_rec1 = [(1020.0, 1260.0), (1980.0, 2220.0), (3060.0, 3300.0)]
    arts_rec1 = {"movement": 735.0, "step": 825.0, "emg_burst": 920.0}
    data_rec1 = synthesize_psg(plan_rec1, stims_rec1, arts_rec1, sfreq=float(sfreq), seed=seed)
    paths_1 = write_brainvision_and_markers(
        root,
        raw_stem="Demo01_Baseline_Night",
        markers_stem="Demo01_Baseline_Night",
        sfreq=float(sfreq),
        data_uv=data_rec1,
        stage_plan=plan_rec1,
        ma_epochs={18, 58},
        exp_events_sec=events_rec1,
        crop_offset_sec=0.0,
        start_timestamp="20260115230000000000",
    )
    created.append(paths_1)

    if recordings >= 2:
        # Recording 2 (nested BIDS layout, 30 min = 60 epochs): cropped .Markers (+120s LightsOff offset)
        plan_rec2 = (
            ["W"] * 6
            + ["N1"] * 3
            + ["N2"] * 13
            + ["N3"] * 8
            + ["W"] * 2
            + ["N1"] * 3
            + ["N2"] * 9
            + ["REM"] * 10
            + ["W"] * 6
        )
        assert len(plan_rec2) == 60
        events_rec2 = [
            ("LightsOff", 120.0),
            ("Start_Music_Happy", 750.0),
            ("Start_DreamReport", 900.0),
            ("Stop_DreamReport", 960.0),
            ("Stop_Music_Sad", 1470.0),
            ("Start_DreamReport", 1620.0),
            ("Stop_DreamReport", 1710.0),
        ]
        stims_rec2 = [(750.0, 900.0), (1470.0, 1620.0)]
        arts_rec2 = {"movement": 450.0, "emg_burst": 1280.0}
        data_rec2 = synthesize_psg(plan_rec2, stims_rec2, arts_rec2, sfreq=float(sfreq), seed=seed + 101)
        paths_2 = write_brainvision_and_markers(
            root / "sub-02" / "ses-01",
            raw_stem="sub-02_ses-01_task-sleep_eeg",
            markers_stem="sub-02_ses-01_task-sleep_eeg_Scored",
            sfreq=float(sfreq),
            data_uv=data_rec2,
            stage_plan=plan_rec2,
            ma_epochs={14, 40},
            exp_events_sec=events_rec2,
            crop_offset_sec=120.0,
            start_timestamp="20260116233000000000",
        )
        created.append(paths_2)

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic BrainVision + .Markers sleep EEG demo data for Onera."
    )
    parser.add_argument(
        "--out",
        default="demo_data",
        help="Output folder path (default: demo_data)",
    )
    parser.add_argument(
        "--sfreq",
        type=int,
        default=200,
        help="Sampling frequency in Hz (default: 200, must be >= 100)",
    )
    parser.add_argument(
        "--recordings",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of demo recordings to generate (1 = flat 60m, 2 = flat 60m + nested BIDS 30m)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic waveforms (default: 42)",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    sessions = generate_demo_dataset(
        out_dir=out_path,
        sfreq=args.sfreq,
        recordings=args.recordings,
        seed=args.seed,
    )

    total_bytes = sum(p.stat().st_size for s in sessions for p in s.values())
    print(f"Generated {len(sessions)} demo recording(s) in: {out_path}")
    for i, s in enumerate(sessions, start=1):
        mb = s["eeg"].stat().st_size / (1024 * 1024)
        print(f"  [{i}] {s['vhdr']} ({mb:.1f} MB) | markers: {s['markers'].name}")
    print(f"Total disk size: {total_bytes / (1024 * 1024):.1f} MB")
    print("\nQuick test:")
    print(f"  uv run marimo edit Onera_SleepScorer.py   -> set Raw Data Path to: {out_path}")
    print(f"  uv run marimo edit Onera_PreAwakening.py -> set both paths to:      {out_path}")


if __name__ == "__main__":
    main()
