from typing import List

import numpy as np
import torch
from transformers import AutoModel

def load_reve(reve_model_name: str, reve_pos_name: str, device: str):
    model = AutoModel.from_pretrained(reve_model_name, trust_remote_code=True)
    pos_bank = AutoModel.from_pretrained(reve_pos_name, trust_remote_code=True)

    for p in model.parameters():
        p.requires_grad = False

    model.to(device).eval()
    pos_bank.to(device).eval()
    return model, pos_bank

def make_pos(pos_bank, ch_names: List[str], device: str) -> torch.Tensor:
    cleaned = [c.strip() for c in ch_names]
    pos = pos_bank(cleaned)
    return pos.to(device)

def _batch_pos(pos: torch.Tensor, batch_size: int) -> torch.Tensor:
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    if pos.size(0) == 1:
        return pos.expand(batch_size, -1, -1)
    if pos.size(0) != batch_size:
        return pos[:1].expand(batch_size, -1, -1)
    return pos

@torch.no_grad()
def extract_window_embeddings(model, pos: torch.Tensor, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """
    X: (n_win, n_ch, win_samp) float32
    returns: (n_win, d) float32
    """
    embs = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device, dtype=torch.float32)
        pb = _batch_pos(pos, xb.shape[0])
        feats = model(xb, pb)  # trust_remote_code output
        flat = feats.reshape(feats.shape[0], -1).detach().cpu().numpy().astype(np.float32, copy=False)
        embs.append(flat)
    return np.concatenate(embs, axis=0)
