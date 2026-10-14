"""Score the second-round pre-training runs (train_revision2.sh).

  mae_nonorm  : masked modelling without instance normalisation (Reviewer 2, Q3)
  vicreg_b48  : VICReg with batch 48 instead of 24 (Reviewer 3, R10)
  mae_setb    : masked modelling with CinC set-b as unlabelled data (R12)

Each is compared with random initialisation and with its matched-budget
counterpart (rev_equal_mae / rev_equal_vicreg) on the same 75 CinC records.
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
OUT = ROOT / "results" / "revision"

RUNS = {
    "mae_nonorm": dict(ckpt="rev_mae_nonorm", ref="rev_equal_mae", input_norm="none"),
    "mae_nonorm_s7": dict(ckpt="rev_mae_nonorm_s7", ref="rev_mae_s7", input_norm="none"),
    "mae_nonorm_s123": dict(ckpt="rev_mae_nonorm_s123", ref="rev_mae_s123", input_norm="none"),
    "vicreg_b48": dict(ckpt="rev_vicreg_b48", ref="rev_equal_vicreg", input_norm="instance"),
    "mae_setb": dict(ckpt="rev_mae_setb", ref="rev_equal_mae", input_norm="instance"),
}


def _paired(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    rng = np.random.default_rng(42)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return dict(delta=float(d.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), p=float(wilcoxon(a, b).pvalue))


def _eff_rank(ckpt, input_norm, device):
    import torch
    from model import FetalECGFM
    from data_loader import load_all_records
    from eval_structure import record_embeddings
    from run_all_evals import effective_rank
    model = FetalECGFM(n_input_channels=4, d_model=256, proj_dim=64, pool=32,
                       input_norm=input_norm).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()
    recs = load_all_records(datasets=["NInFEA", "CinC2013", "NI_FECG_DB", "NIFEA_ARR"])
    X, _ = record_embeddings(model, recs, device)
    X = X[np.all(np.isfinite(X), axis=1)]
    return effective_rank(X)


def main(device_str="mps"):
    import torch
    import eval_qrs as eq
    device = torch.device(device_str)
    rnd = json.loads((ROOT / "results/benchmark/qrs_random.json").read_text())["per_record"]
    path = OUT / "round2_ckpts.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    orig = eq.make_encoder
    for name, cfg in RUNS.items():
        ck = ROOT / "results" / cfg["ckpt"] / "final_model.pt"
        ref_ck = ROOT / "results" / cfg["ref"] / "final_model.pt"
        if not ck.exists() or name in res:
            continue
        eq.make_encoder = lambda kind, ckpt, device, _n=cfg["input_norm"]: orig(kind, ckpt, device, input_norm=_n)
        try:
            r = eq.run("ssl", str(ck), device_str, str(OUT / "new_ckpts"))
        finally:
            eq.make_encoder = orig
        (OUT / "new_ckpts" / f"qrs_{name}.json").write_text(json.dumps(r, indent=2))
        ref_path = OUT / "new_ckpts" / f"qrs_{cfg['ref'].replace('rev_', '')}.json"
        ref = (json.loads(ref_path.read_text()) if ref_path.exists()
               else eq.run("ssl", str(ref_ck), device_str, None))["per_record"]
        recs = sorted(r["per_record"])
        pr = [r["per_record"][k] for k in recs]
        base = rnd
        if cfg["input_norm"] == "none":
            base = json.loads((OUT / "new_ckpts" / "qrs_random_nonorm.json").read_text())["per_record"]
        res[name] = dict(
            f1_mean=r["f1_mean"], f1_std=r["f1_std"],
            vs_random=_paired([base[k] for k in recs], pr),
            vs_matched=_paired([ref[k] for k in recs], pr),
            matched_f1=float(np.mean([ref[k] for k in recs])),
            eff_rank=_eff_rank(ck, cfg["input_norm"], device),
            matched_eff_rank=_eff_rank(ref_ck, "instance", device),
        )
        hist = ROOT / "results" / cfg["ckpt"] / "training_history.json"
        if hist.exists():
            h = json.loads(hist.read_text())
            res[name]["history"] = [h[i] for i in (0, 9, 24, len(h) - 1) if i < len(h)]
        path.write_text(json.dumps(res, indent=2))
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
