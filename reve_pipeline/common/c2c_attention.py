import torch
import torch.nn as nn
import torch.nn.functional as F

class C2CAttention(nn.Module):
    """
    Channel-to-Channel attention for EEG windows.
    Input:  x of shape (B, C, T)
    Output: y of shape (B, C, T)

    We compute channel descriptors by temporal mean, then do a self-attention across channels:
      q,k,v: (B, C, d)
      attn:  (B, C, C)
      out:   (B, C, d) -> projected gate per channel -> mix channels via (B,C,C) @ x

    To keep it simple + stable:
      - attention produces mixing weights over channels
      - we apply mixing on the raw x (per timepoint) with residual connection
    """
    def __init__(self, n_channels: int, d: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.d = d

        self.q = nn.Linear(1, d, bias=True)
        self.k = nn.Linear(1, d, bias=True)
        self.v = nn.Linear(1, d, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.scale = d ** 0.5

        # residual strength (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        B, C, T = x.shape
        assert C == self.n_channels, f"C2CAttention expected C={self.n_channels}, got {C}"

        # channel descriptor: mean over time -> (B, C, 1)
        desc = x.mean(dim=2, keepdim=True)

        q = self.q(desc)  # (B, C, d)
        k = self.k(desc)  # (B, C, d)
        v = self.v(desc)  # (B, C, d)

        # attention logits: (B, C, C)
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / self.scale
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # channel mixing: apply attn to x across channel dim for each time point
        # attn: (B, C, C), x: (B, C, T) -> mixed: (B, C, T)
        mixed = torch.matmul(attn, x)

        # residual: y = x + alpha * mixed
        y = x + self.alpha * mixed
        return y
