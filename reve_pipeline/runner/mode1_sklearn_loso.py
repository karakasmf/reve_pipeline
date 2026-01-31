#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from reve_pipeline.common.paths import WINDOW_DIR, RESULTS_ROOT
from reve_pipeline.common.labels import load_labels
from reve_pipeline.common.cache_io import list_pkls, load_window_pkl, subsample
from reve_pipeline.common.pooling import pool_subject
from reve_pipeline.common.metrics import compute_metrics, compute_confusion_matrix
from reve_pipeline.common.reve_embed import load_reve, make_pos, extract_window_embeddings


# =========================
# QUICK TEST CONFIG (A-C)
# =========================
from common.paths import PARTICIPANTS_TXT
with open(PARTICIPANTS_TXT) as f:
    p = f.read()

TARGET_LABELS = ["A", "C"]     # şimdilik sadece A-C
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REVE_MODEL_NAME = "brain-bzh/reve-base"
REVE_POS_NAME   = "brain-bzh/reve-positions"

BATCH_SIZE = 256
MAX_TRAIN_WINDOWS_PER_SUBJ = 300
MAX_TEST_WINDOWS_PER_SUBJ  = None

POOLING = "mean"          # mean | trimmed_mean
TRIM_RATIO = 0.1

# Logistic Regression params
LOGREG_C = 1.0
LOGREG_MAX_ITER = 2000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_logreg_pipeline(n_classes: int) -> Pipeline:
    # Paper-aligned linear probe
    if n_classes == 2:
        logreg = LogisticRegression(
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
    else:
        logreg = LogisticRegression(
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
            random_state=SEED,
        )

    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", logreg),
    ])


@dataclass
class RunMeta:
    run_id: str
    created_at: str
    phase: str
    probe_type: str
    target_labels: List[str]
    class_order_used: List[str]
    seed: int
    device: str
    reve_model_name: str
    reve_pos_name: str
    batch_size: int
    max_train_windows_per_subj: Optional[int]
    max_test_windows_per_subj: Optional[int]
    pooling: str
    trim_ratio: float
    logreg_c: float
    logreg_max_iter: int
    window_dir: str
    participants_txt: str
    n_subjects: int
    n_channels: int
    win_samp: int


def main():
    set_seed(SEED)

    tag = "-".join(TARGET_LABELS)
    run_id = datetime.now().strftime("run_%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / tag / run_id
    ensure_dir(str(run_dir))

    # labels
    label_dict, label_map, id_to_label = load_labels(PARTICIPANTS_TXT, target_labels=TARGET_LABELS)

    # subjects from cache
    pkls = list_pkls(WINDOW_DIR)
    subj_items: List[Tuple[str, str, int]] = []
    for p in pkls:
        sid = os.path.basename(p).split("_windows.pkl")[0]
        if sid in label_dict:
            subj_items.append((sid, p, int(label_dict[sid])))
    subj_items = sorted(subj_items, key=lambda x: x[0])

    if not subj_items:
        raise ValueError(f"[{tag}] No cached subjects match TARGET_LABELS. Check cache dir and participants.txt")

    # canonical channel names & shape
    _, X0, meta0 = load_window_pkl(subj_items[0][1])
    ch_names = meta0.get("ch_names")
    if not ch_names:
        raise ValueError("Cache meta['ch_names'] missing. Rebuild cache with builder script.")
    n_ch = int(X0.shape[1])
    win_samp = int(X0.shape[2])

    print("=" * 78)
    print(f"[{tag}] DEVICE: {DEVICE}")
    if DEVICE == "cuda":
        print("[cuda] torch:", torch.__version__, "| cuda:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
    print(f"[{tag}] WINDOW_DIR: {WINDOW_DIR}")
    print(f"[{tag}] Subjects: {len(subj_items)}")
    print(f"[{tag}] CLASS_ORDER_USED: {[id_to_label[i] for i in range(len(id_to_label))]}")
    print("=" * 78, flush=True)

    # load REVE frozen
    reve, pos_bank = load_reve(REVE_MODEL_NAME, REVE_POS_NAME, DEVICE)
    pos = make_pos(pos_bank, ch_names, DEVICE)

    n_classes = len(id_to_label)
    clf = build_logreg_pipeline(n_classes)

    fold_rows: List[Dict] = []
    y_true_all, y_pred_all = [], []

    for fold_idx, (test_sid, test_pkl, y_test) in enumerate(subj_items, start=1):
        t0 = time.time()

        print(f"[{tag}] Fold {fold_idx:03d}/{len(subj_items)} test={test_sid} y_true={id_to_label[int(y_test)]}", flush=True)

        train_items = [(sid, pkl, y) for (sid, pkl, y) in subj_items if sid != test_sid]

        X_train_subj, y_train_subj = [], []
        for sid, pkl, y in train_items:
            _, X, meta = load_window_pkl(pkl)
            if X.shape[1] != n_ch or X.shape[2] != win_samp:
                raise ValueError(f"[{tag}] Shape mismatch for {sid}: {X.shape}, expected (*,{n_ch},{win_samp})")

            X = subsample(X, MAX_TRAIN_WINDOWS_PER_SUBJ, SEED + fold_idx)
            w_emb = extract_window_embeddings(reve, pos, X, batch_size=BATCH_SIZE, device=DEVICE)
            s_emb = pool_subject(w_emb, method=POOLING, trim_ratio=TRIM_RATIO)

            X_train_subj.append(s_emb)
            y_train_subj.append(int(y))

        X_train_subj = np.stack(X_train_subj, axis=0).astype(np.float32)
        y_train_subj = np.asarray(y_train_subj, dtype=int)

        # fit sklearn probe
        clf.fit(X_train_subj, y_train_subj)

        # test subject embedding
        _, X_test, _ = load_window_pkl(test_pkl)
        X_test = subsample(X_test, MAX_TEST_WINDOWS_PER_SUBJ, SEED + 1000 + fold_idx)
        w_emb_test = extract_window_embeddings(reve, pos, X_test, batch_size=BATCH_SIZE, device=DEVICE)
        s_emb_test = pool_subject(w_emb_test, method=POOLING, trim_ratio=TRIM_RATIO).reshape(1, -1).astype(np.float32)

        prob = clf.predict_proba(s_emb_test)[0]
        pred = int(np.argmax(prob))

        dt = time.time() - t0

        prob_str = ", ".join([f"{id_to_label[i]}={prob[i]:.4f}" for i in range(len(prob))])
        print(f"[{tag}]   y_pred={id_to_label[pred]} prob=[{prob_str}] runtime={dt:.1f}s", flush=True)

        y_true_all.append(int(y_test))
        y_pred_all.append(int(pred))

        row = {
            "fold_idx": fold_idx,
            "test_subject": test_sid,
            "y_true": int(y_test),
            "y_pred": int(pred),
            "y_true_label": id_to_label[int(y_test)],
            "y_pred_label": id_to_label[int(pred)],
            "runtime_sec": float(dt),
            "n_train_subjects": len(train_items),
            "n_test_windows": int(X_test.shape[0]),
        }
        for i in range(len(prob)):
            row[f"prob_{id_to_label[i]}"] = float(prob[i])
        fold_rows.append(row)

    y_true_all = np.asarray(y_true_all, dtype=int)
    y_pred_all = np.asarray(y_pred_all, dtype=int)

    metrics = compute_metrics(y_true_all, y_pred_all, n_classes=n_classes)
    cm = compute_confusion_matrix(y_true_all, y_pred_all, n_classes=n_classes)

    # save fold_results.csv
    fold_csv = os.path.join(str(run_dir), "fold_results.csv")
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    # save confusion matrix
    np.save(os.path.join(str(run_dir), "confusion_matrix.npy"), cm)
    cm_csv = os.path.join(str(run_dir), "confusion_matrix.csv")
    with open(cm_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [id_to_label[i] for i in range(n_classes)])
        for i in range(n_classes):
            w.writerow([id_to_label[i]] + list(cm[i].tolist()))

    # meta
    meta = RunMeta(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        phase="phase1",
        probe_type="sklearn",
        target_labels=TARGET_LABELS,
        class_order_used=[id_to_label[i] for i in range(n_classes)],
        seed=SEED,
        device=DEVICE,
        reve_model_name=REVE_MODEL_NAME,
        reve_pos_name=REVE_POS_NAME,
        batch_size=BATCH_SIZE,
        max_train_windows_per_subj=MAX_TRAIN_WINDOWS_PER_SUBJ,
        max_test_windows_per_subj=MAX_TEST_WINDOWS_PER_SUBJ,
        pooling=POOLING,
        trim_ratio=TRIM_RATIO,
        logreg_c=LOGREG_C,
        logreg_max_iter=LOGREG_MAX_ITER,
        window_dir=str(WINDOW_DIR),
        participants_txt=PARTICIPANTS_TXT,
        n_subjects=len(subj_items),
        n_channels=n_ch,
        win_samp=win_samp,
    )
    out = {"meta": asdict(meta), "global_metrics": metrics}
    with open(os.path.join(str(run_dir), "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print(f"[{tag}] DONE -> {run_dir}")
    print(f"[{tag}] Global metrics: {metrics}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
