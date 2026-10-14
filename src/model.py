"""Fetal-ECG-FM: Multi-scale CNN + Transformer foundation model.

Designed for self-supervised pre-training on fetal ECG signals.
Architecture:
  Input: (batch, n_ch, n_samp)  -- multi-channel abdominal recording
  → Multi-scale 1D CNN (3 parallel branches)
  → Channel fusion (1x1 conv)
  → Transformer encoder (temporal)
  → Global average pooling
  → Projection head (for contrastive learning)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    import warnings
    warnings.warn("PyTorch not available; model code will not run.")


class MultiScaleConv(nn.Module):
    """Parallel 1D convolutions at multiple scales."""
    def __init__(self, in_ch, out_ch=64, pool=32, dec=4):
        super().__init__()
        branch = out_ch // 3
        self.out_ch = branch * 3  # actual output width (may be < out_ch)
        # Decimate first (anti-aliased average) so the large-kernel convolutions
        # run at 1/dec the sample rate. This is a large speed-up on 10 s windows
        # at 1 kHz; the multi-scale kernels below are sized in the decimated rate
        # so they cover the same physical durations as before.
        self.dec = nn.AvgPool1d(kernel_size=dec, stride=dec)
        # At 1 kHz / dec=4 (250 Hz): 11 -> ~44 ms (QRS), 31 -> ~124 ms (T-wave),
        # 101 -> ~404 ms (baseline/HRV).
        self.conv_s = nn.Conv1d(in_ch, branch, kernel_size=11, padding=5, bias=False)
        self.conv_m = nn.Conv1d(in_ch, branch, kernel_size=31, padding=15, bias=False)
        self.conv_l = nn.Conv1d(in_ch, branch, kernel_size=101, padding=50, bias=False)
        self.bn = nn.BatchNorm1d(self.out_ch)
        # Remaining pooling so total downsampling = dec * (pool//dec) = pool.
        self.pool = nn.MaxPool1d(kernel_size=pool // dec, stride=pool // dec)

    def forward(self, x):
        # x: (B, C, T)
        x = self.dec(x)
        s = F.relu(self.conv_s(x))
        m = F.relu(self.conv_m(x))
        l = F.relu(self.conv_l(x))
        out = torch.cat([s, m, l], dim=1)
        out = self.bn(out)
        out = self.pool(out)
        return out  # (B, out_ch, T // pool)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder for temporal modeling."""
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x


class FetalECGFM(nn.Module):
    """Full foundation model."""
    def __init__(self, n_input_channels=4, d_model=256, nhead=8, num_layers=4,
                 proj_dim=64, dropout=0.1, pool=32, use_transformer=True,
                 input_norm="instance"):
        super().__init__()
        self.use_transformer = use_transformer
        self.input_norm = input_norm
        # Multi-scale CNN front-end
        self.frontend = MultiScaleConv(n_input_channels, out_ch=128, pool=pool)
        # Channel mixer: 1x1 conv to project to d_model
        self.channel_mixer = nn.Conv1d(self.frontend.out_ch, d_model, kernel_size=1)
        # Temporal transformer (optional; CNN-only ablation skips it)
        self.transformer = TransformerEncoder(
            d_model, nhead, num_layers, dim_feedforward=512, dropout=dropout
        ) if use_transformer else None
        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def _normalise(self, x):
        """x: (B, C, T). instance = per-window per-channel; none = identity."""
        if self.input_norm == "none":
            return x
        if self.input_norm == "layernorm":
            mu = x.mean(dim=(1, 2), keepdim=True)
            sd = x.std(dim=(1, 2), keepdim=True)
            return (x - mu) / (sd + 1e-5)
        # default: per-window, per-channel instance normalisation
        return (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-5)

    def token_features(self, x):
        """Return per-token transformer features (B, T', d_model).

        T' is the temporal length after front-end pooling; used by the
        downstream QRS-detection head to localise beats.
        """
        # Per-window, per-channel instance normalisation: makes the encoder
        # amplitude-invariant (abdominal fECG amplitude varies with electrode
        # placement and gestation) and well-scaled for masked reconstruction.
        x = self._normalise(x)
        x = self.frontend(x)
        x = self.channel_mixer(x)
        x = x.transpose(1, 2)
        if self.transformer is None:
            return x
        return self.transformer(x)

    def forward(self, x, return_embedding=False):
        """x: (B, C, T)

        Returns:
          z: (B, proj_dim) -- normalized projection for contrastive loss
          emb: (B, d_model) -- pooled embedding for downstream tasks
        """
        tok = self.token_features(x)  # (B, T', d_model)
        # Global average pooling
        emb = tok.mean(dim=1)  # (B, d_model)
        # Projection
        z = self.proj(emb)
        z = F.normalize(z, dim=1)
        if return_embedding:
            return z, emb
        return z

    def project_unnorm(self, x):
        """Un-normalised pooled projection (for VICReg, which needs raw variance)."""
        emb = self.token_features(x).mean(dim=1)
        return self.proj(emb)


def make_token_mask(n_batch, n_tok, mask_ratio, device, seed_gen=None):
    """Boolean (B, n_tok) mask; True = masked. Each row masks ~mask_ratio tokens."""
    k = max(1, int(round(mask_ratio * n_tok)))
    mask = torch.zeros(n_batch, n_tok, dtype=torch.bool, device=device)
    for b in range(n_batch):
        idx = torch.randperm(n_tok, device=device)[:k]
        mask[b, idx] = True
    return mask


class SignalDecoder(nn.Module):
    """Lightweight decoder for masked signal modelling (MAE-style).

    Maps per-token encoder features (B, T', d_model) back to the input signal
    (B, C, T). It upsamples the token features to the input length and applies
    two convolutions. Used only during pre-training; the encoder alone is saved
    for downstream use, so the downstream interface is unchanged.
    """
    def __init__(self, d_model=256, out_ch=4, target_len=10000):
        super().__init__()
        self.target_len = target_len
        self.net = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(64, out_ch, kernel_size=15, padding=7),
        )

    def forward(self, tok):
        # tok: (B, T', d_model) -> (B, d_model, T')
        x = tok.transpose(1, 2)
        x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        return self.net(x)  # (B, C, T)


def vicreg_loss(z_a, z_b, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0,
                return_parts=False):
    """VICReg loss (Bardes, Ponce & LeCun, 2022): invariance + variance +
    covariance. Explicitly prevents representation collapse by forcing each
    embedding dimension to keep unit variance and to be decorrelated.

    Default coefficients follow Bardes et al. (λ=25, μ=25, ν=1).
    z_a, z_b: (B, D) embeddings of two views (NOT L2-normalised).
    """
    B, D = z_a.shape
    # invariance: match the two views
    sim = F.mse_loss(z_a, z_b)
    # variance: hinge on per-dimension std -> keep it >= 1
    std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
    std_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
    std = torch.mean(F.relu(1.0 - std_a)) + torch.mean(F.relu(1.0 - std_b))
    # covariance: off-diagonal covariance -> 0 (decorrelate dimensions)
    za = z_a - z_a.mean(dim=0)
    zb = z_b - z_b.mean(dim=0)
    cov_a = (za.T @ za) / (B - 1)
    cov_b = (zb.T @ zb) / (B - 1)
    off = lambda C: (C - torch.diag(torch.diag(C))).pow(2).sum() / D
    cov = off(cov_a) + off(cov_b)
    loss = sim_coeff * sim + std_coeff * std + cov_coeff * cov
    if return_parts:
        return loss, {
            "sim": float(sim.detach()),
            "std": float(std.detach()),
            "cov": float(cov.detach()),
            "std_mean": float(0.5 * (std_a.mean() + std_b.mean()).detach()),
            "lambda": sim_coeff, "mu": std_coeff, "nu": cov_coeff,
        }
    return loss


class CrossModalProjector(nn.Module):
    """Separate encoder branch for Doppler ultrasound signals (NInFEA).

    Doppler is a 1D velocity envelope; we process with a simple CNN → projection.
    """
    def __init__(self, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=51, padding=25),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )

    def forward(self, x):
        """x: (B, 1, T) Doppler velocity envelope."""
        x = self.net(x).squeeze(-1)  # (B, 64)
        z = self.proj(x)
        return F.normalize(z, dim=1)


# --------------------------------------------------------------------------- #
# Contrastive losses
# --------------------------------------------------------------------------- #

def info_nce_loss(z_i, z_j, temperature=0.1):
    """Standard InfoNCE for paired views (z_i, z_j).

    z_i, z_j: (B, D) normalized embeddings.
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    # Positives: i ↔ j for each sample
    pos = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ])
    # Cross-entropy over softmax of similarities
    loss = F.cross_entropy(sim, pos)
    return loss


def gestation_aware_nce(z_i, z_j, ga_i, ga_j, temperature=0.1, transition_weeks=2.0, k=1.0):
    """Gestation-aware contrastive loss with soft *negative* weighting.

    This is a weighted InfoNCE. The positive pair (the two augmented views of
    the same window) always receives full weight. Each negative term in the
    denominator is weighted by

        w_ij = sigmoid(k * (|GA_i - GA_j| - transition_weeks)) in (0, 1),

    so negatives at a similar gestational age (small |ΔGA|) are down-weighted
    (they are plausibly similar and should not be pushed apart), while negatives
    far apart in GA act as full negatives. Pairs where either sample has no GA
    label (ga < 0, e.g. CinC 2013 / FECGSYNDB) get full negative weight.

    ga_i, ga_j: (B,) gestational ages in weeks (use a negative value for unknown).
    """
    B = z_i.shape[0]
    device = z_i.device
    z = torch.cat([z_i, z_j], dim=0)          # (2B, D)
    ga = torch.cat([ga_i, ga_j])              # (2B,)
    N = 2 * B
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    pos = torch.cat([
        torch.arange(B, N, device=device),
        torch.arange(0, B, device=device),
    ])
    rows = torch.arange(N, device=device)

    # Negative weights from GA distance; unknown-GA pairs -> full weight (1).
    ga_dist = torch.abs(ga.unsqueeze(1) - ga.unsqueeze(0))
    w = torch.sigmoid(k * (ga_dist - transition_weeks))
    unknown = (ga < 0)
    w = torch.where(unknown.unsqueeze(1) | unknown.unsqueeze(0), torch.ones_like(w), w)

    eye = torch.eye(N, dtype=torch.bool, device=device)
    w = w.masked_fill(eye, 0.0)               # no self term
    w[rows, pos] = 1.0                        # positive always full weight

    # Numerically stable weighted log-softmax over each row.
    sim = sim.masked_fill(eye, -9e15)
    m = sim.max(dim=1, keepdim=True).values
    exp_w = torch.exp(sim - m) * w
    log_denom = torch.log(exp_w.sum(dim=1) + 1e-12) + m.squeeze(1)
    log_pos = sim[rows, pos] - log_denom
    return -log_pos.mean()


def cross_modal_nce(z_ecg, z_doppler):
    """Contrastive alignment between ECG and Doppler embeddings.

    z_ecg, z_doppler: (B, D) from the same recordings.
    """
    batch_size = z_ecg.shape[0]
    z = torch.cat([z_ecg, z_doppler], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / 0.1
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ])
    return F.cross_entropy(sim, pos)


if __name__ == "__main__" and TORCH_AVAILABLE:
    # Smoke test
    model = FetalECGFM(n_input_channels=4, d_model=256)
    x = torch.randn(2, 4, 10000)  # batch=2, 4 ch, 10 s @ 1kHz
    z, emb = model(x, return_embedding=True)
    print("Input:", x.shape, "→ Embedding:", emb.shape, "→ Projection:", z.shape)
    assert z.shape == (2, 64)
    print("Model OK")
