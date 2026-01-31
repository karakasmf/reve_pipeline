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
from reve_pipeline.common.metrics import compute_metrics, compute_confusion_matrix
from reve_pipeline.common.reve_embed import load_reve, make_pos
from reve_pipeline.common.c2c_attention import C2CAttention


# =========================================================
# CONFIG (A-C quick test)
# =========================================================
from common.paths import PARTICIPANTS_TXT
with open(PARTICIPANTS_TXT) as f:
    p = f.read()
TARGET_LABELS = ["A", "C"]
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REVE_MODEL_NAME = "brain-bzh/reve-base"
REVE_POS_NAME   = "brain-bzh/reve-positions"

# Cache sampling (memory + speed control)
MAX_TRAIN_WINDOWS_PER_SUBJ = 300        # cache'ten okununca train subj için üst sınır
MAX_TEST_WINDOWS_PER_SUBJ  = None       # test subj tüm pencereler (istersen 500 yap)

# End-to-end training hyperparams
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6

# Subject-batched training:
SUBJECT_BATCH = 6                       # bir step'te kaç subject
WINDOWS_PER_SUBJECT_STEP = 32           # her subject'ten step başına kaç window örnekle
STEPS_PER_EPOCH = None                  # None => tüm train subjectler üzerinde 1 geçiş

# C2C parameters
C2C_D = 64
C2C_DROPOUT = 0.1

# Misc
PRINT_EVERY = 5                         # epoch bazlı print
TRAIN_LOG_ONE_FILE = True               # tek training_log.csv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _expand_pos(pos: torch.Tensor, batch_size: int) -> torch.Tensor:
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    if pos.size(0) == 1:
        return pos.expand(batch_size, -1, -1)
    if pos.size(0) != batch_size:
        return pos[:1].expand(batch_size, -1, -1)
    return pos


class End2EndPhase2Model(nn.Module):
    """
    Trainable parts: C2C + classifier head
    Frozen: REVE encoder (params requires_grad=False, but autograd flows through ops to inputs)
    """
    def __init__(self, n_channels: int, n_classes: int, c2c_d: int, c2c_dropout: float, emb_dim: int):
        super().__init__()
        self.c2c = C2CAttention(n_channels=n_channels, d=c2c_d, dropout=c2c_dropout)
        self.norm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward_subject_batch(
        self,
        reve,                 # frozen model
        pos: torch.Tensor,    # (1,C,3) or (C,3)
        x_bkct: torch.Tensor, # (B,K,C,T)
    ) -> torch.Tensor:
        """
        Returns logits: (B, n_classes)
        """
        B, K, C, T = x_bkct.shape
        x = x_bkct.reshape(B * K, C, T)

        # trainable C2C
        x = self.c2c(x)

        # frozen REVE forward (no detach!)
        pb = _expand_pos(pos, x.shape[0])
        feats = reve(x, pb)                    # shape depends on model; we flatten
        emb = feats.reshape(feats.shape[0], -1)  # (B*K, D)

        emb = emb.reshape(B, K, -1).mean(dim=1)  # subject pooling in-torch (B, D)

        emb = self.norm(emb)
        logits = self.fc(emb)
        return logits


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
    max_train_windows_per_subj: Optional[int]
    max_test_windows_per_subj: Optional[int]
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    subject_batch: int
    windows_per_subject_step: int
    c2c_d: int
    c2c_dropout: float
    window_dir: str
    participants_txt: str
    n_subjects: int
    n_channels: int
    win_samp: int
    emb_dim: int


def _prepare_subject_cache(subj_items, seed, max_train_windows):
    """
    Returns dict: sid -> np.ndarray (n_win, C, T) float32
    """
    cache = {}
    for sid, pkl, _y in subj_items:
        _sid, X, _meta = load_window_pkl(pkl)
        X = subsample(X, max_train_windows, seed=seed)
        cache[sid] = X.astype(np.float32, copy=False)
    return cache


def _iter_subject_batches(train_subjects: List[Tuple[str, int]], subject_batch: int, rng: np.random.Generator):
    """
    train_subjects: list of (sid, y)
    yields list of (sid,y) of length <= subject_batch
    """
    idx = np.arange(len(train_subjects))
    rng.shuffle(idx)
    for i in range(0, len(idx), subject_batch):
        batch_idx = idx[i:i+subject_batch]
        yield [train_subjects[j] for j in batch_idx]


def train_one_fold_end2end(
    fold_idx: int,
    tag: str,
    run_dir: str,
    reve,
    pos: torch.Tensor,
    model: End2EndPhase2Model,
    train_subjects: List[Tuple[str, int]],
    train_cache: Dict[str, np.ndarray],
    n_classes: int,
    global_train_log_path: Optional[str] = None,
):
    """
    Trains model (C2C + head) end-to-end for one LOSO fold.
    Returns nothing; model weights updated in-place.
    """
    # class-balanced CE
    ys = np.array([y for (_sid, y) in train_subjects], dtype=int)
    counts = np.bincount(ys, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.clip(counts, 1, None))
    class_w = torch.tensor(w, device=DEVICE, dtype=torch.float32)

    crit = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = None
    best_state = None
    bad = 0

    # simple internal "val": use a few subjects (10%) from train, deterministic per fold
    rng = np.random.default_rng(SEED + 10000 + fold_idx)
    idx = np.arange(len(train_subjects))
    rng.shuffle(idx)
    n_val = max(2, int(0.1 * len(train_subjects)))
    val_idx = set(idx[:n_val].tolist())
    val_subjects = [train_subjects[i] for i in range(len(train_subjects)) if i in val_idx]
    tr_subjects  = [train_subjects[i] for i in range(len(train_subjects)) if i not in val_idx]

    def _run_epoch(subjects: List[Tuple[str, int]], train_mode: bool):
        model.train(train_mode)
        total_loss, total_correct, total_n = 0.0, 0, 0

        rng_local = np.random.default_rng(SEED + fold_idx + (0 if train_mode else 777))
        batches = list(_iter_subject_batches(subjects, SUBJECT_BATCH, rng_local))

        if STEPS_PER_EPOCH is not None:
            batches = batches[:STEPS_PER_EPOCH]

        for batch in batches:
            B = len(batch)
            # build (B,K,C,T) by sampling windows per subject
            X_list = []
            y_list = []
            for sid, y in batch:
                Xs = train_cache[sid]
                nwin = Xs.shape[0]
                k = min(WINDOWS_PER_SUBJECT_STEP, nwin)
                win_idx = rng_local.choice(nwin, size=k, replace=False) if nwin > k else np.arange(nwin)
                X_list.append(Xs[win_idx])  # (k,C,T)
                y_list.append(y)

            # pad K to max_k for stacking
            max_k = max(x.shape[0] for x in X_list)
            X_bkct = np.zeros((B, max_k, X_list[0].shape[1], X_list[0].shape[2]), dtype=np.float32)
            mask = np.zeros((B, max_k), dtype=np.float32)

            for i, Xi in enumerate(X_list):
                kk = Xi.shape[0]
                X_bkct[i, :kk] = Xi
                mask[i, :kk] = 1.0

            x = torch.from_numpy(X_bkct).to(DEVICE, dtype=torch.float32)
            y = torch.tensor(y_list, device=DEVICE, dtype=torch.long)

            # forward: we used mean over K; but padded zeros could bias.
            # We'll do masked mean manually:
            B, K, C, T = x.shape
            x_flat = x.reshape(B*K, C, T)
            x_flat = model.c2c(x_flat)
            pb = _expand_pos(pos, x_flat.shape[0])
            feats = reve(x_flat, pb)
            emb = feats.reshape(feats.shape[0], -1)  # (B*K,D)
            D = emb.shape[1]
            emb = emb.reshape(B, K, D)

            m = torch.from_numpy(mask).to(DEVICE, dtype=torch.float32).unsqueeze(-1)  # (B,K,1)
            emb_sum = (emb * m).sum(dim=1)
            denom = m.sum(dim=1).clamp_min(1.0)
            subj_emb = emb_sum / denom  # (B,D)

            subj_emb = model.norm(subj_emb)
            logits = model.fc(subj_emb)

            loss = crit(logits, y)

            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                total_correct += int((pred == y).sum().item())
                total_n += int(y.shape[0])
                total_loss += float(loss.item()) * int(y.shape[0])

        avg_loss = total_loss / max(1, total_n)
        avg_acc = total_correct / max(1, total_n)
        return avg_loss, avg_acc

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(tr_subjects, train_mode=True)
        va_loss, va_acc = _run_epoch(val_subjects, train_mode=False)

        # log
        if global_train_log_path:
            with open(global_train_log_path, "a", newline="", encoding="utf-8") as f:
                wri = csv.writer(f)
                wri.writerow([fold_idx, epoch, tr_loss, tr_acc, va_loss, va_acc, LR])

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"[{tag}] Fold {fold_idx:03d} | epoch {epoch:03d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | va_loss={va_loss:.4f} va_acc={va_acc:.3f}",
                  flush=True)

        # early stopping on val loss
        if best_val is None or va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def infer_subject(
    tag: str,
    reve,
    pos: torch.Tensor,
    model: End2EndPhase2Model,
    X: np.ndarray,
    max_test_windows: Optional[int],
):
    X = subsample(X.astype(np.float32, copy=False), max_test_windows, seed=SEED + 999)
    # Use all selected windows; pooled embedding in-torch with masking (no need mask here)
    x = torch.from_numpy(X).to(DEVICE, dtype=torch.float32)  # (N,C,T)
    # apply c2c then REVE
    x = model.c2c(x)
    pb = _expand_pos(pos, x.shape[0])
    feats = reve(x, pb)
    emb = feats.reshape(feats.shape[0], -1)  # (N,D)
    subj_emb = emb.mean(dim=0, keepdim=True)  # (1,D)
    subj_emb = model.norm(subj_emb)
    logits = model.fc(subj_emb)
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred = int(np.argmax(prob))
    return pred, prob


def main():
    set_seed(SEED)

    tag = "-".join(TARGET_LABELS)
    run_id = datetime.now().strftime("run_%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / tag / ("end2end_phase2_" + run_id)
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

    # shape / channels
    _, X0, meta0 = load_window_pkl(subj_items[0][1])
    ch_names = meta0.get("ch_names")
    if not ch_names:
        raise ValueError("Cache meta['ch_names'] missing. Rebuild cache with builder script.")
    n_ch = int(X0.shape[1])
    win_samp = int(X0.shape[2])

    print("=" * 78)
    print(f"[{tag}] END2END PHASE-2 | DEVICE: {DEVICE}")
    if DEVICE == "cuda":
        print("[cuda] torch:", torch.__version__, "| cuda:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
    print(f"[{tag}] WINDOW_DIR: {WINDOW_DIR}")
    print(f"[{tag}] Subjects: {len(subj_items)}")
    print(f"[{tag}] CLASS_ORDER_USED: {[id_to_label[i] for i in range(len(id_to_label))]}")
    print(f"[{tag}] TRAIN: subj_batch={SUBJECT_BATCH} win_per_subj_step={WINDOWS_PER_SUBJECT_STEP} epochs={EPOCHS} patience={PATIENCE}")
    print(f"[{tag}] C2C: d={C2C_D} dropout={C2C_DROPOUT}")
    print("=" * 78, flush=True)

    # load REVE frozen (params frozen but allow grad flow to inputs)
    reve, pos_bank = load_reve(REVE_MODEL_NAME, REVE_POS_NAME, DEVICE)
    pos = make_pos(pos_bank, ch_names, DEVICE)

    # freeze REVE params (already frozen in load_reve), keep eval mode
    reve.eval()

    # determine embedding dim by one forward (no detach from graph needed; use no_grad for probe init only)
    with torch.no_grad():
        x_tmp = torch.from_numpy(X0[:2].astype(np.float32)).to(DEVICE)  # (2,C,T)
        pb_tmp = _expand_pos(pos, x_tmp.shape[0])
        f_tmp = reve(x_tmp, pb_tmp)
        emb_dim = int(f_tmp.reshape(f_tmp.shape[0], -1).shape[1])

    n_classes = len(id_to_label)

    # global training log (single file)
    global_train_log = None
    if TRAIN_LOG_ONE_FILE:
        global_train_log = os.path.join(str(run_dir), "training_log.csv")
        with open(global_train_log, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["fold_idx", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    fold_rows: List[Dict] = []
    y_true_all, y_pred_all = [], []

    # Preload ALL subjects once (speed) – they are numpy arrays already.
    # For training we use per-fold train_cache (excluding test) but reading once is fine.
    all_cache = _prepare_subject_cache(subj_items, seed=SEED, max_train_windows=MAX_TRAIN_WINDOWS_PER_SUBJ)

    for fold_idx, (test_sid, test_pkl, y_test) in enumerate(subj_items, start=1):
        t0 = time.time()
        print(f"[{tag}] Fold {fold_idx:03d}/{len(subj_items)} test={test_sid} y_true={id_to_label[int(y_test)]}", flush=True)

        train_items = [(sid, pkl, y) for (sid, pkl, y) in subj_items if sid != test_sid]
        train_subjects = [(sid, y) for (sid, _p, y) in train_items]

        # build per-fold cache dict from preloaded
        train_cache = {sid: all_cache[sid] for (sid, _y) in train_subjects}

        # init fresh model per fold (LOSO)
        model = End2EndPhase2Model(
            n_channels=n_ch,
            n_classes=n_classes,
            c2c_d=C2C_D,
            c2c_dropout=C2C_DROPOUT,
            emb_dim=emb_dim,
        ).to(DEVICE)

        # train end-to-end (C2C + head) for this fold
        train_one_fold_end2end(
            fold_idx=fold_idx,
            tag=tag,
            run_dir=str(run_dir),
            reve=reve,
            pos=pos,
            model=model,
            train_subjects=train_subjects,
            train_cache=train_cache,
            n_classes=n_classes,
            global_train_log_path=global_train_log,
        )

        # inference on test subject (using its full cache from disk for correctness)
        _sid, X_test, _m = load_window_pkl(test_pkl)
        pred, prob = infer_subject(tag, reve, pos, model, X_test, max_test_windows=MAX_TEST_WINDOWS_PER_SUBJ)

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
        wri = csv.writer(f)
        wri.writerow([""] + [id_to_label[i] for i in range(n_classes)])
        for i in range(n_classes):
            wri.writerow([id_to_label[i]] + list(cm[i].tolist()))

    # meta
    meta = RunMeta(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        phase="phase2_end2end",
        probe_type="torch_end2end",
        target_labels=TARGET_LABELS,
        class_order_used=[id_to_label[i] for i in range(n_classes)],
        seed=SEED,
        device=DEVICE,
        reve_model_name=REVE_MODEL_NAME,
        reve_pos_name=REVE_POS_NAME,
        max_train_windows_per_subj=MAX_TRAIN_WINDOWS_PER_SUBJ,
        max_test_windows_per_subj=MAX_TEST_WINDOWS_PER_SUBJ,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        subject_batch=SUBJECT_BATCH,
        windows_per_subject_step=WINDOWS_PER_SUBJECT_STEP,
        c2c_d=C2C_D,
        c2c_dropout=C2C_DROPOUT,
        window_dir=str(WINDOW_DIR),
        participants_txt=PARTICIPANTS_TXT,
        n_subjects=len(subj_items),
        n_channels=n_ch,
        win_samp=win_samp,
        emb_dim=emb_dim,
    )
    out = {"meta": asdict(meta), "global_metrics": metrics}
    with open(os.path.join(str(run_dir), "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print(f"[{tag}] END2END PHASE-2 DONE -> {run_dir}")
    print(f"[{tag}] Global metrics: {metrics}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
