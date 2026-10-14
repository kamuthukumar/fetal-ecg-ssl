"""Training script for Fetal-ECG-FM self-supervised pre-training.

Usage (after installing PyTorch):
  cd code
  .venv/bin/python src/train_foundation.py --epochs 100 --batch_size 16 --lr 1e-4

The script will:
  1. Load all datasets (ADfECGDB, NInFEA, CinC 2013, FECGSYNDB, NI-FECG DB, NIFEA ARR)
  2. Apply physics-informed augmentations to create two views per sample
  3. Pre-train FetalECGFM with gestation-aware contrastive loss
  4. Save model checkpoint and training logs
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Path setup
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from data_loader import load_all_records, DATA_ROOT
from augmentations import DEFAULT_AUGMENT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--window_sec", type=float, default=10.0, help="Window length in seconds")
    p.add_argument("--target_fs", type=float, default=1000.0)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain_datasets", type=str, default="NInFEA,NI_FECG_DB",
                   help="Comma-separated datasets used for SSL pre-training. "
                        "CinC2013 and ADfECGDB are held out for downstream eval.")
    p.add_argument("--simclr", action="store_true",
                   help="Ablation: generic augmentations + plain InfoNCE (no physics, no GA).")
    p.add_argument("--mode", type=str, default=None,
                   choices=["contrastive", "simclr", "vicreg", "mae"],
                   help="Pre-training objective. Overrides --simclr. "
                        "contrastive=physics augs + GA-aware NCE; "
                        "vicreg=variance-covariance anti-collapse; "
                        "mae=masked signal reconstruction.")
    p.add_argument("--mask_ratio", type=float, default=0.5, help="MAE token mask ratio.")
    p.add_argument("--input_norm", type=str, default="instance",
                   choices=["instance", "none", "layernorm"],
                   help="Input normalisation inside the encoder.")
    p.add_argument("--no_transformer", action="store_true",
                   help="CNN-only ablation: skip the Transformer encoder.")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "results" / "foundation"))
    return p.parse_args()


def _window_signal(signal, fs, window_sec=10.0, stride_sec=5.0):
    """Cut a long signal into overlapping windows."""
    n_samp = int(window_sec * fs)
    stride = int(stride_sec * fs)
    if signal.shape[1] < n_samp:
        # Pad
        pad = n_samp - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="constant")
    windows = []
    for start in range(0, signal.shape[1] - n_samp + 1, stride):
        windows.append(signal[:, start:start + n_samp])
    if not windows:
        windows.append(signal[:, :n_samp])
    return windows


def make_dataset(records, window_sec=10.0, target_fs=1000.0):
    """Create a list of (window_tensor, ga_weeks, dataset_name, subject) tuples."""
    items = []
    for rec in records:
        abdo = rec["abdominal"]
        fs = rec["fs"]
        # Resample to target_fs if needed (already done in loader, but double-check)
        if fs != target_fs:
            from scipy.signal import resample
            n_target = int(abdo.shape[1] * target_fs / fs)
            abdo = resample(abdo, n_target, axis=1)
            fs = target_fs
        wins = _window_signal(abdo, fs, window_sec, stride_sec=window_sec / 2)
        for w in wins:
            items.append({
                "signal": w.astype(np.float32),
                "ga_weeks": rec.get("ga_weeks"),
                "dataset": rec["dataset"],
                "subject": rec.get("subject"),
                "record": rec["record"],
            })
    return items


def collate_batch(items, augment, device):
    """Create a batch with two augmented views."""
    # Stack signals
    max_ch = max(it["signal"].shape[0] for it in items)
    max_len = max(it["signal"].shape[1] for it in items)

    # Pad to common shape
    batch = []
    for it in items:
        sig = it["signal"]
        # Pad channels
        if sig.shape[0] < max_ch:
            sig = np.pad(sig, ((0, max_ch - sig.shape[0]), (0, 0)), mode="constant")
        # Pad length
        if sig.shape[1] < max_len:
            sig = np.pad(sig, ((0, 0), (0, max_len - sig.shape[1])), mode="constant")
        batch.append(sig)

    batch = np.stack(batch, axis=0)  # (B, C, T)

    # Create two augmented views
    view1 = np.stack([augment(batch[i], 1000.0, it["ga_weeks"]) for i, it in enumerate(items)], axis=0)
    view2 = np.stack([augment(batch[i], 1000.0, it["ga_weeks"]) for i, it in enumerate(items)], axis=0)

    # Convert to torch tensors (float32: MPS does not support float64)
    import torch
    v1 = torch.from_numpy(view1.astype(np.float32)).to(device)
    v2 = torch.from_numpy(view2.astype(np.float32)).to(device)
    ga = torch.tensor([it["ga_weeks"] if it["ga_weeks"] is not None else -1.0 for it in items], dtype=torch.float32, device=device)
    return v1, v2, ga


def train_one_epoch(model, dataloader, optimizer, augment, device, temperature,
                    epoch, batch_size=16, use_gestation=True):
    import torch
    from model import gestation_aware_nce, info_nce_loss
    model.train()
    total_loss = 0.0
    n = 0
    for i in range(0, len(dataloader), batch_size):
        batch = dataloader[i:i + batch_size]
        if len(batch) < 2:
            continue
        v1, v2, ga = collate_batch(batch, augment, device)
        z1 = model(v1)
        z2 = model(v2)
        # Gestation-aware loss when GA is available for >50% of the batch and
        # not running the SimCLR ablation; otherwise plain InfoNCE.
        has_ga = (ga >= 0).float().mean() > 0.5
        if use_gestation and has_ga:
            loss = gestation_aware_nce(z1, z2, ga, ga, temperature=temperature)
        else:
            loss = info_nce_loss(z1, z2, temperature=temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _collate_raw(items, device):
    """Stack raw (un-augmented) windows to a tensor (B, C, T)."""
    import torch
    x = np.stack([it["signal"] for it in items], axis=0).astype(np.float32)
    return torch.from_numpy(x).to(device)


def train_one_epoch_mae(model, decoder, dataloader, optimizer, device, epoch,
                        batch_size, mask_ratio, pool=32):
    """Masked signal modelling: mask spans of the input, reconstruct them.

    Forces per-token encoder features to retain local morphology, which is what
    a downstream QRS-detection probe needs (unlike instance-contrastive learning,
    which optimises window-level invariance and tends to collapse)."""
    import torch
    from model import make_token_mask
    model.train(); decoder.train()
    total, n = 0.0, 0
    for i in range(0, len(dataloader), batch_size):
        batch = dataloader[i:i + batch_size]
        if len(batch) < 2:
            continue
        import torch.nn.functional as F
        x = _collate_raw(batch, device)                 # (B, C, T)
        B, C, T = x.shape
        # instance-normalise (scale-free objective, matches encoder input norm)
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-5)
        n_tok = T // pool                                       # encoder tokens
        tmask = make_token_mask(B, n_tok, mask_ratio, device)   # (B, n_tok)

        def expand(mask, factor, length):
            m = mask.repeat_interleave(factor, dim=1)
            if m.shape[1] < length:
                m = torch.cat([m, m[:, -1:].expand(B, length - m.shape[1])], dim=1)
            return m[:, :length]

        # mask the encoder input at full resolution
        smask_in = expand(tmask, pool, T).unsqueeze(1)          # (B,1,T)
        x_in = x.masked_fill(smask_in, 0.0)
        tok = model.token_features(x_in)                        # (B, n_tok, d)
        recon = decoder(tok)                                    # (B, C, T/4)
        # reconstruction target + mask at the decoder's (1/4) resolution
        Tt = recon.shape[2]
        target = F.avg_pool1d(x, kernel_size=T // Tt) if T % Tt == 0 else \
            F.interpolate(x, size=Tt, mode="linear", align_corners=False)
        smask_t = expand(tmask, Tt // n_tok if Tt % n_tok == 0 else max(1, Tt // n_tok),
                         Tt).unsqueeze(1).expand(B, C, Tt)
        loss = ((recon - target) ** 2 * smask_t).sum() / (smask_t.sum() + 1e-8)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def train_one_epoch_vicreg(model, dataloader, optimizer, augment, device, epoch, batch_size):
    """VICReg: variance-invariance-covariance regularisation on two views."""
    import torch
    from model import vicreg_loss
    model.train()
    total, n = 0.0, 0
    parts_acc = {"sim": 0.0, "std": 0.0, "cov": 0.0, "std_mean": 0.0}
    for i in range(0, len(dataloader), batch_size):
        batch = dataloader[i:i + batch_size]
        if len(batch) < 2:
            continue
        v1, v2, _ = collate_batch(batch, augment, device)
        z1 = model.project_unnorm(v1)
        z2 = model.project_unnorm(v2)
        loss, parts = vicreg_loss(z1, z2, return_parts=True)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item(); n += 1
        for k in parts_acc:
            parts_acc[k] += parts[k]
    n = max(n, 1)
    return total / n, {k: v / n for k, v in parts_acc.items()}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check PyTorch
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(args.seed)
        except Exception:
            pass

    # Load data (CinC2013 and ADfECGDB are held out for downstream evaluation)
    pretrain_datasets = [d.strip() for d in args.pretrain_datasets.split(",") if d.strip()]
    print(f"Loading pre-training datasets: {pretrain_datasets}")
    records = load_all_records(datasets=pretrain_datasets)
    pretrain_items = make_dataset(records, args.window_sec, args.target_fs)
    print(f"Pre-train: {len(pretrain_items)} windows from {len(records)} records")

    # Resolve mode (--mode overrides the legacy --simclr flag).
    mode = args.mode or ("simclr" if args.simclr else "contrastive")

    # Model — all recordings are standardised to 4 channels by the loader.
    from model import FetalECGFM, SignalDecoder
    from augmentations import GENERIC_AUGMENT
    n_ch = 4
    model = FetalECGFM(
        n_input_channels=n_ch, d_model=args.d_model, proj_dim=args.proj_dim,
        use_transformer=not args.no_transformer, input_norm=args.input_norm,
    ).to(device)
    decoder = None
    params = list(model.parameters())
    if mode == "mae":
        # reconstruct at 1/4 resolution (250 Hz): cheaper and above fECG bandwidth
        recon_len = int(args.window_sec * args.target_fs) // 4
        decoder = SignalDecoder(d_model=args.d_model, out_ch=n_ch,
                                target_len=recon_len).to(device)
        params = params + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Reported SimCLR and VICReg use the generic jitter/scale view generator.
    # Physiological augmentations + GA-weighted InfoNCE are used only in
    # --mode contrastive (reported as an explicit ablation). Masked modelling
    # reconstructs unaugmented windows (masking is the corruption).
    augment = GENERIC_AUGMENT if mode in ("simclr", "vicreg") else DEFAULT_AUGMENT
    labels = {"contrastive": "Fetal-ECG-FM (physics augs, GA-aware NCE)",
              "simclr": "SimCLR (generic augs, InfoNCE)",
              "vicreg": "VICReg (generic augs, variance-covariance)",
              "mae": f"MAE (masked signal modelling, mask_ratio={args.mask_ratio})"}
    print(f"Mode: {labels[mode]}  seed={args.seed}  norm={args.input_norm}  "
          f"transformer={not args.no_transformer}")
    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(pretrain_items)
        vic_parts = None
        if mode == "mae":
            loss = train_one_epoch_mae(model, decoder, pretrain_items, optimizer, device,
                                       epoch, args.batch_size, args.mask_ratio)
        elif mode == "vicreg":
            loss, vic_parts = train_one_epoch_vicreg(
                model, pretrain_items, optimizer, augment, device, epoch, args.batch_size)
        else:
            loss = train_one_epoch(model, pretrain_items, optimizer, augment, device,
                                   args.temperature, epoch, batch_size=args.batch_size,
                                   use_gestation=(mode == "contrastive"))
        scheduler.step()
        rec = {"epoch": epoch, "loss": loss}
        if vic_parts:
            rec.update(vic_parts)
        history.append(rec)
        extra = ""
        if vic_parts:
            extra = (f"  sim={vic_parts['sim']:.3f} std={vic_parts['std']:.3f} "
                     f"cov={vic_parts['cov']:.3f} std_mean={vic_parts['std_mean']:.3f}")
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}{extra}")
        if epoch % 10 == 0:
            ckpt = out_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt)
            print(f"  Saved {ckpt}")

    # Save final model + history
    final_ckpt = out_dir / "final_model.pt"
    torch.save(model.state_dict(), final_ckpt)
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Training complete. Final model saved to {final_ckpt}")

    # Save config
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["n_input_channels"] = n_ch
    config["pool"] = 32
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
