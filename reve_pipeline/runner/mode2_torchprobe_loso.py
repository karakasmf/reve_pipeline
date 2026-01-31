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
import torch.nn as nn
import torch.nn.functional as F

from reve_pipeline.common.paths import WINDOW_DIR, RESULTS_ROOT
from reve_pipeline.common.labels import load_labels
from reve_pipeline.common.cache_io import list_pkls, load_window_pkl, subsample
from reve_pipeline.common.pooling import pool_subject
from reve_pipeline.common.metrics import compute_metrics, compute_confusion_matrix
from reve_pipeline.common.reve_embed import load_reve, make_pos, extract_window_embeddings


# =========================
# QUICK TEST CONFIG (A-C)
# =========================
reve_pipeline..paths import PARTICIPANTS_TXT
with open(PARTICIPANTS_TXT) as f:
    p = f.read()

TARGET_LABEL_SETS = [
    ["A", "C"],
    ["A", "F"],
    ["A", "F", "C"],
]
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REVE_MODEL_NAME = "brain-bzh/reve-base"
REVE_POS_NAME   = "brain-bzh/reve-positions"

BATCH_SIZE = 256
MAX_TRAIN_WINDOWS_PER_SUBJ = 300
MAX_TEST_WINDOWS_PER_SUBJ  = None

POOLING = "mean"          # mean | trimmed_mean
TRIM_RATIO = 0.1

# Torch probe
EPOCHS = 30               # hızlı kontrol için 30 (istersen 20)
LR = 1e-3
WEIGHT_DECAY = 0.0
PATIENCE = 8              # val varsa erken durdurma


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


class TorchLinearProbe(nn.Module):
    def __init__(self, d: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    n_classes: int,
    log_csv_path: str,
) -> TorchLinearProbe:
    X_train_t = torch.from_numpy(X_train).to(DEVICE, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(DEVICE, dtype=torch.long)

    if X_val is not None and y_val is not None and len(y_val) > 0:
        X_val_t = torch.from_numpy(X_val).to(DEVICE, dtype=torch.float32)
        y_val_t = torch.from_numpy(y_val).to(DEVICE, dtype=torch.long)
    else:
        X_val_t, y_val_t = None, None

    d = X_train.shape[1]
    model = TorchLinearProbe(d, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = None
    best_state = None
    bad = 0

    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        logits = model(X_train_t)
        loss = F.cross_entropy(logits, y_train_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            train_acc = (pred == y_train_t).float().mean().item()
            train_loss = float(loss.item())

            val_loss, val_acc = "", ""
            if X_val_t is not None:
                model.eval()
                v_logits = model(X_val_t)
                v_loss = F.cross_entropy(v_logits, y_val_t).item()
                v_pred = torch.argmax(v_logits, dim=1)
                v_acc = (v_pred == y_val_t).float().mean().item()
                val_loss, val_acc = float(v_loss), float(v_acc)

        lr_now = float(opt.param_groups[0]["lr"])
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr_now])

        # early stopping
        if X_val_t is not None:
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


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
    epochs: int
    lr: float
    weight_decay: float
    window_dir: str
    participants_txt: str
    n_subjects: int
    n_channels: int
    win_samp: int


def main(TARGET_LABELS):
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

        # small val split (optional but useful for stability)
        rng = np.random.default_rng(SEED + fold_idx)
        idx = np.arange(len(y_train_subj))
        rng.shuffle(idx)
        split = max(1, int(0.8 * len(idx)))
        tr_idx, va_idx = idx[:split], idx[split:]

        X_tr, y_tr = X_train_subj[tr_idx], y_train_subj[tr_idx]
        X_va, y_va = (X_train_subj[va_idx], y_train_subj[va_idx]) if len(va_idx) > 0 else (None, None)

        log_path = os.path.join(str(run_dir), f"training_log_fold{fold_idx:03d}.csv")
        probe = train_probe(X_tr, y_tr, X_va, y_va, n_classes=n_classes, log_csv_path=log_path)

        # test subject embedding
        _, X_test, _ = load_window_pkl(test_pkl)
        X_test = subsample(X_test, MAX_TEST_WINDOWS_PER_SUBJ, SEED + 1000 + fold_idx)
        w_emb_test = extract_window_embeddings(reve, pos, X_test, batch_size=BATCH_SIZE, device=DEVICE)
        s_emb_test = pool_subject(w_emb_test, method=POOLING, trim_ratio=TRIM_RATIO).reshape(1, -1).astype(np.float32)

        probe.eval()
        with torch.no_grad():
            xt = torch.from_numpy(s_emb_test).to(DEVICE, dtype=torch.float32)
            logits = probe(xt)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
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
        probe_type="torch",
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
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
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
    for TARGET_LABELS in TARGET_LABEL_SETS:
        print("=" * 70)
        print("Running TARGET_LABELS =", TARGET_LABELS)
        print("=" * 70)
        main(TARGET_LABELS)

