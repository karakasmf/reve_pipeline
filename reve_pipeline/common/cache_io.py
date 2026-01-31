import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

def list_pkls(window_dir) -> List[str]:
    pkls = sorted(glob.glob(os.path.join(str(window_dir), "sub-*_windows.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No window cache found in: {window_dir}")
    return pkls

def load_window_pkl(pkl_path: str) -> Tuple[str, np.ndarray, Dict]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj["subject"], obj["X"], obj.get("meta", {})

def subsample(X: np.ndarray, max_n: Optional[int], seed: int) -> np.ndarray:
    if max_n is None or X.shape[0] <= max_n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx]
