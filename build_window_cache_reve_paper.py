#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""build_window_cache_reve_paper.py

REVE-paper-aligned window cache builder (ds004504 derivatives).

Paper-aligned preprocessing (as per your quoted section):
- Keep recordings >= 10 seconds (we use MIN_TOTAL_SEC=10)
- Resample to 200 Hz
- Band-pass filter 0.5–99.5 Hz
- Convert to float32
- Convert amplitude to microvolts (MNE gives Volts)
- Recording/session-level Z-score normalization (per-channel stats over the whole recording)
- Clip to ±15 standard deviations

Plus (critical for REVE+pos in LOSO):
- Enforce a single canonical channel order across all subjects
  (locked from first successfully processed subject after preprocessing)

Output format (compatible with your pipeline):
- OUT_DIR/sub-XXX_windows.pkl
- payload = {"subject": str, "X": np.ndarray (n_win, n_ch, win_samp), "meta": dict}
"""

import os
import sys
import glob
import pickle
import logging
from typing import List, Optional

import numpy as np
import mne

# =========================
# CONFIG (Edit these)
# =========================
DERIVATIVES_ROOT = r"D:\ACADEMICS\datasets\alz-ftd-ctl\ds004504\derivatives"
OUT_DIR          = r"D:\REPOS\alz-ftd-ctl-reve\cache\windows"

# Windowing (used to generate multiple segments per subject)
WIN_SEC   = 2.0
STEP_SEC  = 1.0
MAX_WINDOWS = None  # e.g. 2000 or None

# Paper-aligned preprocessing
TARGET_SFREQ = 200.0
BANDPASS = (0.5, 99.5)
NOTCH    = None
REREF    = None

# Keep >= 10 seconds (paper); you can raise this if you want
MIN_TOTAL_SEC = 10.0

# Normalization & clipping (paper)
ZCLIP_STD = 15.0

# Behavior
OVERWRITE = True
VERBOSE   = True

# Canonical channel behavior
STRICT_CANONICAL = True  # recommended: skip subjects missing any canonical channel
CHANNEL_RENAME_MAP = {
    # Example mappings if needed:
    # "T3": "T7",
    # "T4": "T8",
    # "T5": "P7",
    # "T6": "P8",
}

SUPPORTED_EXTS = (".set", ".fif", ".edf", ".bdf")


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("build_window_cache_reve_paper")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def find_eeg_file(subject_dir: str) -> Optional[str]:
    for ext in SUPPORTED_EXTS:
        files = glob.glob(os.path.join(subject_dir, "**", f"*{ext}"), recursive=True)
        if files:
            return files[0]
    return None


def load_raw(path: str) -> mne.io.BaseRaw:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if ext == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if ext == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if ext == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    raise RuntimeError(f"Unsupported EEG format: {path}")


def apply_preprocessing(raw: mne.io.BaseRaw, logger: logging.Logger) -> mne.io.BaseRaw:
    raw.pick("eeg")

    # Optional channel renaming
    if CHANNEL_RENAME_MAP:
        present = {ch: CHANNEL_RENAME_MAP[ch] for ch in raw.ch_names if ch in CHANNEL_RENAME_MAP}
        if present:
            raw.rename_channels(present)
            logger.debug(f"Renamed channels: {present}")

    # Paper: band-pass 0.5–99.5 Hz
    if BANDPASS is not None:
        raw.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1], verbose="ERROR")

    if NOTCH is not None:
        raw.notch_filter(NOTCH, verbose="ERROR")

    if REREF is not None:
        rr = REREF.lower()
        if rr == "average":
            raw.set_eeg_reference("average", verbose="ERROR")
        else:
            logger.warning(f"Unknown reref='{REREF}'. Skipping reref.")

    # Paper: resample to 200 Hz
    if abs(float(raw.info["sfreq"]) - TARGET_SFREQ) > 1e-6:
        raw.resample(TARGET_SFREQ, npad="auto", verbose="ERROR")

    return raw


def reorder_to_canonical(
    data: np.ndarray,
    ch_names: List[str],
    canonical: List[str],
    subject: str,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    """Reorder (n_ch, n_samp) data to canonical channel list."""
    idx = {c.strip(): i for i, c in enumerate(ch_names)}
    canonical_clean = [c.strip() for c in canonical]

    missing = [c for c in canonical_clean if c not in idx]
    if missing:
        msg = f"{subject} missing channels: {missing}"
        if STRICT_CANONICAL:
            logger.warning("[SKIP] " + msg)
            return None
        else:
            logger.warning("[FILL-ZERO] " + msg)

    out = np.zeros((len(canonical_clean), data.shape[1]), dtype=np.float32)
    for i, c in enumerate(canonical_clean):
        if c in idx:
            out[i] = data[idx[c]]
    return out


def recording_zscore_and_clip(x: np.ndarray, clip_std: float) -> np.ndarray:
    """Paper: Z-score using stats across the recording session, then clip to ±15 std.

    x: (n_ch, n_samp) in microvolts.
    Returns float32.
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    z = (x - mu) / sd
    if clip_std is not None:
        z = np.clip(z, -float(clip_std), float(clip_std))
    return z.astype(np.float32, copy=False)


def sliding_window(x: np.ndarray, win_samp: int, step_samp: int) -> np.ndarray:
    """x: (n_ch, n_samp) -> (n_win, n_ch, win_samp)"""
    n_ch, n_samp = x.shape
    if n_samp < win_samp:
        return np.empty((0, n_ch, win_samp), dtype=np.float32)

    n_win = 1 + (n_samp - win_samp) // step_samp
    out = np.empty((n_win, n_ch, win_samp), dtype=np.float32)
    for i in range(n_win):
        s = i * step_samp
        out[i] = x[:, s : s + win_samp]
    return out


def main():
    logger = setup_logger(VERBOSE)
    os.makedirs(OUT_DIR, exist_ok=True)

    subjects = sorted([d for d in os.listdir(DERIVATIVES_ROOT) if d.startswith("sub-")])
    logger.info(f"Found {len(subjects)} subjects under derivatives.")

    canonical_ch_names: Optional[List[str]] = None
    ok, skipped = 0, 0

    for sub in subjects:
        subj_dir = os.path.join(DERIVATIVES_ROOT, sub)
        out_path = os.path.join(OUT_DIR, f"{sub}_windows.pkl")

        if os.path.exists(out_path) and not OVERWRITE:
            logger.info(f"[SKIP] cache exists: {sub}")
            skipped += 1
            continue

        eeg_file = find_eeg_file(subj_dir)
        if eeg_file is None:
            logger.warning(f"[SKIP] no EEG file: {sub}")
            skipped += 1
            continue

        try:
            raw = load_raw(eeg_file)
            raw = apply_preprocessing(raw, logger)
        except Exception as e:
            logger.warning(f"[SKIP] load/preprocess failed: {sub} | {e}")
            skipped += 1
            continue

        sfreq = float(raw.info["sfreq"])  # should be TARGET_SFREQ
        data_v = raw.get_data().astype(np.float32)  # Volts
        duration_sec = data_v.shape[1] / sfreq

        if duration_sec < MIN_TOTAL_SEC:
            logger.warning(f"[SKIP] too short: {sub} ({duration_sec:.1f}s)")
            skipped += 1
            continue

        # Lock canonical list from first valid subject (after preprocessing)
        if canonical_ch_names is None:
            canonical_ch_names = list(raw.ch_names)
            logger.info(f"[CANONICAL] locked from {sub}: {len(canonical_ch_names)} channels")
            logger.info(f"[CANONICAL] {canonical_ch_names}")

        # Reorder to canonical
        data_v = reorder_to_canonical(data_v, raw.ch_names, canonical_ch_names, sub, logger)
        if data_v is None:
            skipped += 1
            continue

        # Convert V -> µV (paper keeps >100µV; we do not drop them)
        data_uv = data_v * 1e6

        # Recording-level z-score + clip to ±15 std
        data_norm = recording_zscore_and_clip(data_uv, ZCLIP_STD)

        win_samp = int(WIN_SEC * sfreq)
        step_samp = int(STEP_SEC * sfreq)

        X = sliding_window(data_norm, win_samp, step_samp)
        if X.shape[0] == 0:
            logger.warning(f"[SKIP] insufficient samples for 1 window: {sub}")
            skipped += 1
            continue

        if MAX_WINDOWS is not None and X.shape[0] > int(MAX_WINDOWS):
            X = X[: int(MAX_WINDOWS)]

        payload = {
            "subject": sub,
            "X": X.astype(np.float32, copy=False),
            "meta": {
                "sfreq": sfreq,
                "target_sfreq": TARGET_SFREQ,
                "bandpass": BANDPASS,
                "win_sec": WIN_SEC,
                "step_sec": STEP_SEC,
                "win_samp": win_samp,
                "step_samp": step_samp,
                "n_channels": int(X.shape[1]),
                "ch_names": list(canonical_ch_names),
                "duration_sec": float(duration_sec),
                "source_file": eeg_file,
                "strict_canonical": bool(STRICT_CANONICAL),
                "zscore_scope": "recording_per_channel",
                "clip_std": float(ZCLIP_STD),
                "units": "zscore_clipped_(V->uV)",
            },
        }

        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"[OK] {sub} | windows={X.shape[0]} | ch={X.shape[1]}")
        ok += 1

    logger.info(f"DONE | ok={ok} | skipped={skipped}")
    if canonical_ch_names is None:
        logger.error("No subject processed successfully. Check DERIVATIVES_ROOT and file patterns.")


if __name__ == "__main__":
    main()
