"""Sequence models: baselines B3-B6 and PI-PatchSurv (CLAUDE.md "Proposed model" §5).

Every model maps
    x   (B, L, C)  last L minutes of scaled channels + mask channels + RI*
    phi (B, F)     physics features at the window end (RI* level/trend, T-JUS-CKP, P-TPT stats)
to an embedding z (B, D); a shared head turns z into P(event within H) for each horizon.

Two heads (so the "-survival head" ablation is one flag):
  "surv"  discrete-time hazards h_k = σ(w_kᵀz + b_k), F(k) = 1 - Π_{j<=k}(1 - h_j)
          -> monotone in k by construction (P(within 24 h) can never be < P(within 6 h)).
  "bce"   one independent logit per horizon (what B1/B2 do) -> may be non-monotone.
Baselines B3-B6 use "bce" + sequence only, matching B1/B2; PI-PatchSurv adds physics + survival.
All models are small (< 1 M params): M4 laptop, 24 real events -> big models would just memorise.
"""

from __future__ import annotations

import torch
import torch.nn as nn

D = 64


class Head(nn.Module):
    def __init__(self, kind: str, n_bins: int):
        super().__init__()
        self.kind = kind
        # surv: K hazards; bce: K-1 horizons (the open last bin has no finite horizon)
        self.lin = nn.Linear(D, n_bins if kind == "surv" else n_bins - 1)

    def forward(self, z):
        """Returns logits; use .cdf() for P(event within horizon_j), j = 0..K-2."""
        return self.lin(z)

    def cdf(self, logits):
        if self.kind == "bce":
            return torch.sigmoid(logits)
        surv = torch.cumprod(1 - torch.sigmoid(logits), dim=1)   # S(k) = Π_{j<=k} (1 - h_j)
        return (1 - surv)[:, :-1]                                  # F at the finite edges


class RNN(nn.Module):
    """B3 LSTM / B4 GRU: classic recurrent baselines from the 3W literature."""

    def __init__(self, c_in, cell="lstm"):
        super().__init__()
        self.proj = nn.Linear(c_in, D)
        self.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(D, D, num_layers=2, batch_first=True, dropout=0.1)

    def forward(self, x, phi):
        out, _ = self.rnn(self.proj(x))
        return out[:, -1]


class CNNLSTM(nn.Module):
    """B5: strided convs shorten 360 steps -> 90 (local shape), LSTM models the slower trend."""

    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, D, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(D, D, 5, stride=2, padding=2), nn.GELU())
        self.rnn = nn.LSTM(D, D, batch_first=True)

    def forward(self, x, phi):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(h)
        return out[:, -1]


class TCN(nn.Module):
    """B6: causal dilated convs (kernel 3); receptive field 1 + 2*(1+2+...+128) = 511 min >= 360 window."""

    def __init__(self, c_in, dilations=(1, 2, 4, 8, 16, 32, 64, 128)):
        super().__init__()
        self.inp = nn.Conv1d(c_in, D, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(D, D, 3, dilation=d) for d in dilations)
        self.dil = dilations

    def forward(self, x, phi):
        h = self.inp(x.transpose(1, 2))
        for conv, d in zip(self.blocks, self.dil):
            # left-pad only -> causal: output at t sees inputs <= t
            h = h + nn.functional.gelu(conv(nn.functional.pad(h, (2 * d, 0))))
        return h[:, :, -1]


class TransLSTM(nn.Module):
    """B7 (Lai 2026 style): Transformer encoder for global context, then LSTM for temporal order.
    WHY avg-pool 4 first: full attention over 360 steps is slow on the M4; 4-min steps keep the trend."""

    def __init__(self, c_in):
        super().__init__()
        self.pool = nn.AvgPool1d(4)
        self.proj = nn.Linear(c_in, D)
        self.pos = nn.Parameter(torch.zeros(1, 90, D))
        layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=128, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.lstm = nn.LSTM(D, D, batch_first=True)

    def forward(self, x, phi):
        h = self.proj(self.pool(x.transpose(1, 2)).transpose(1, 2))
        out, _ = self.lstm(self.enc(h + self.pos[:, :h.shape[1]]))
        return out[:, -1]


class ITransformer(nn.Module):
    """B8i iTransformer (Liu et al. 2024): one token per VARIATE (its whole 360-min series + its mask
    embedded linearly), attention across variates -> learns cross-sensor relations, not time steps."""

    def __init__(self, c_in, L=360):
        super().__init__()
        self.n_val = (c_in + 1) // 2           # same value/mask layout as PIPatchSurv
        self.embed = nn.Linear(2 * L, D)
        self.chan = nn.Parameter(torch.zeros(1, self.n_val, D))
        layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=128, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(self.n_val * D, D)

    def forward(self, x, phi):
        C = self.n_val
        m = x[..., C:]
        tok = torch.cat([x[..., :C], torch.cat([m, m[..., :1]], -1)], 1).transpose(1, 2)   # (B, C, 2L)
        h = self.enc(self.embed(tok) + self.chan)
        return self.out(h.flatten(1))


class PIPatchSurv(nn.Module):
    """Physics-informed patch Transformer.
    Sequence branch: channel-independent patches (P=16, stride 8) -> shared Transformer encoder
    -> per-channel mean over patches -> flatten channels -> z_s.
    Physics branch: MLP(2x64, GELU, dropout) on phi -> z_p.
    Gated fusion: g = σ(W_g[z_s; z_p]), z = g*z_s + (1-g)*z_p  (g shows how much physics is used).
    use_physics=False / patch=1 give the "-physics" and "-patching" ablations."""

    def __init__(self, c_in, f_phys, L=360, patch=16, stride=8, use_physics=True):
        super().__init__()
        self.patch, self.stride, self.use_physics = patch, stride, use_physics
        n_patch = (L - patch) // stride + 1
        # x = [9 channels + RI*] values then 9 masks. WHY fold each mask into its value's patch token
        # (2P inputs): keeps "this value was imputed" info but halves the channel count (19 -> 10)
        # -> ~2x faster channel-independent attention on the M4. RI* reuses the P-MON-CKP mask.
        self.n_val = (c_in + 1) // 2
        c_in = self.n_val
        self.embed = nn.Linear(2 * patch, D)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, D))
        self.chan = nn.Parameter(torch.zeros(1, c_in, D))      # tells the shared encoder which channel
        layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=128, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.to_zs = nn.Linear(c_in * D, D)
        if use_physics:
            self.phys = nn.Sequential(nn.Linear(f_phys, D), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(D, D), nn.GELU(), nn.Dropout(0.1))
            self.gate = nn.Linear(2 * D, D)
        self.last_gate = None

    def forward(self, x, phi):
        B, L, _ = x.shape
        C = self.n_val
        m = x[..., C:]
        x = torch.stack([x[..., :C], torch.cat([m, m[..., :1]], -1)], -1)          # (B, L, C, 2)
        p = x.permute(0, 2, 3, 1).unfold(3, self.patch, self.stride)               # (B, C, 2, n, P)
        p = p.permute(0, 1, 3, 2, 4).reshape(B * C, -1, 2 * self.patch)            # value+mask per token
        h = self.embed(p) + self.pos                                               # channel-independent
        h = self.enc(h).mean(1).reshape(B, C, D) + self.chan
        zs = self.to_zs(h.reshape(B, C * D))
        if not self.use_physics:
            return zs
        zp = self.phys(phi)
        g = torch.sigmoid(self.gate(torch.cat([zs, zp], 1)))
        self.last_gate = g.detach().mean().item()
        return g * zs + (1 - g) * zp


class Model(nn.Module):
    def __init__(self, body: nn.Module, head: str, n_bins: int):
        super().__init__()
        self.body, self.head = body, Head(head, n_bins)

    def forward(self, x, phi):
        return self.head(self.body(x, phi))


def build(name: str, c_in: int, f_phys: int, n_bins: int, L: int = 360) -> Model:
    bodies = {
        "B3": lambda: RNN(c_in, "lstm"), "B4": lambda: RNN(c_in, "gru"),
        "B5": lambda: CNNLSTM(c_in), "B6": lambda: TCN(c_in), "B7": lambda: TransLSTM(c_in),
        # B8 PatchTST = the same channel-independent patch encoder as PI-PatchSurv, no physics, BCE head
        "B8": lambda: PIPatchSurv(c_in, f_phys, L, use_physics=False), "B8i": lambda: ITransformer(c_in, L),
        "PIPS": lambda: PIPatchSurv(c_in, f_phys, L),
        # E2 ablations of PI-PatchSurv
        "PIPS-phys": lambda: PIPatchSurv(c_in, f_phys, L, use_physics=False),
        "PIPS-patch": lambda: PIPatchSurv(c_in, f_phys, L, patch=8, stride=8),
        # E2 "-patching": every minute is its own token (patch 1), everything else as PIPS
        "PIPS-nopatch": lambda: PIPatchSurv(c_in, f_phys, L, patch=1, stride=1),
    }
    head = "surv" if name.startswith("PIPS") and name != "PIPS-surv" else "bce"
    body = bodies.get(name, bodies["PIPS"])()   # "PIPS-surv", "PIPS-mono" reuse the full body
    return Model(body, head, n_bins)
