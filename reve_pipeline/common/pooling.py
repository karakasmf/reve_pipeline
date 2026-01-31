import numpy as np

def pool_subject(embs: np.ndarray, method: str = "mean", trim_ratio: float = 0.1) -> np.ndarray:
    """
    embs: (n_win, d)
    returns: (d,)
    """
    if method == "mean":
        return embs.mean(axis=0)

    if method == "trimmed_mean":
        med = np.median(embs, axis=0)
        d = np.linalg.norm(embs - med[None, :], axis=1)
        k = int((1.0 - 2.0 * trim_ratio) * len(d))
        if k <= 1:
            return embs.mean(axis=0)
        keep_idx = np.argsort(d)[:k]
        return embs[keep_idx].mean(axis=0)

    raise ValueError(f"Unknown pooling method: {method}")
