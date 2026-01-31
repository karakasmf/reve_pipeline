from typing import Dict, Tuple
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    out = {"acc": acc, "bal_acc": bal_acc, "mcc": mcc}

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    out.update({
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=sup) if sup.sum() > 0 else 0.0),
    })

    if n_classes == 2:
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        sens = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
        spec = 0.0 if (tn + fp) == 0 else float(tn / (tn + fp))
        out.update({"sens": sens, "spec": spec})

    return out

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(int)
