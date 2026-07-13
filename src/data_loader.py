"""Unified data loader for all fetal ECG datasets.

Handles ADfECGDB, NInFEA, CinC 2013, FECGSYNDB, NI-FECG DB, and NIFEA Arrhythmia DB.
Standardizes to: (n_channels, n_samples) @ 1000 Hz, band-pass 1-90 Hz, 50 Hz notch.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# --------------------------------------------------------------------------- #
# Pre-processing primitives
# --------------------------------------------------------------------------- #
from fecg_extraction import bandpass, notch  # reuse existing filters


def standardize_channels(sig, fs, target_fs=1000.0):
    """Resample to target_fs and ensure 2-D (n_chan, n_samp)."""
    sig = np.atleast_2d(sig)
    if sig.shape[0] > sig.shape[1]:
        sig = sig.T
    if fs != target_fs:
        n_target = int(sig.shape[1] * target_fs / fs)
        sig = resample(sig, n_target, axis=1)
    return sig, target_fs


def preprocess(sig, fs, target_fs=1000.0):
    """Full pre-processing: resample → bandpass 1-90 Hz → notch 50 Hz."""
    sig, fs = standardize_channels(sig, fs, target_fs)
    out = []
    for ch in sig:
        x = notch(bandpass(ch, fs, 1.0, 90.0), fs, f0=50.0, q=30.0)
        out.append(x)
    arr = np.array(out)
    # Some raw records contain flat/saturated channels or missing samples that
    # make the zero-phase filters emit NaN/Inf; replace with zeros so downstream
    # feature extraction stays finite.
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), fs


def select_channels(sig, fs, n_keep=4):
    """Standardise every recording to exactly ``n_keep`` channels.

    For recordings with more channels (e.g. NInFEA's 24 abdominal leads) we
    keep the ``n_keep`` channels with the highest fetal-band kurtosis, a simple
    proxy for the presence of well-defined fetal QRS complexes (spiky, high
    excess-kurtosis morphology). For recordings with fewer channels we tile the
    existing channels up to ``n_keep``. This keeps the model input dimension
    fixed across heterogeneous hardware instead of zero-padding to the widest
    montage.
    """
    from scipy.stats import kurtosis
    n_ch = sig.shape[0]
    if n_ch == n_keep:
        return sig
    if n_ch > n_keep:
        # kurtosis in the fetal QRS band (~10-45 Hz)
        try:
            band = np.array([bandpass(ch, fs, 10.0, 45.0) for ch in sig])
        except Exception:
            band = sig
        k = kurtosis(band, axis=1, fisher=True, nan_policy="omit")
        k = np.nan_to_num(k, nan=-np.inf)
        idx = np.argsort(k)[::-1][:n_keep]
        idx = np.sort(idx)
        return sig[idx]
    # n_ch < n_keep: tile channels round-robin
    reps = int(np.ceil(n_keep / n_ch))
    return np.tile(sig, (reps, 1))[:n_keep]


# --------------------------------------------------------------------------- #
# Dataset-specific loaders
# --------------------------------------------------------------------------- #

def load_adfecgdb(record_id="r01"):
    """Load ADfECGDB EDF record. Returns (abdominal, scalp, fs, fetal_beats)."""
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib required for ADfECGDB; pip install pyedflib")
    p = DATA_ROOT / "adfecgdb" / f"{record_id}.edf"
    if not p.exists():
        return None
    f = pyedflib.EdfReader(str(p))
    n = f.signals_in_file
    labels = [f.getLabel(i) for i in range(n)]
    sigs = [f.readSignal(i) for i in range(n)]
    fs = f.getSampleFrequency(0)
    f.close()
    # scalp is usually last channel; abdominals are first 4
    scalp_idx = labels.index("Scalp") if "Scalp" in labels else -1
    abdo = np.array(sigs[:4]) if n >= 4 else np.array(sigs[:-1])
    scalp = np.array(sigs[scalp_idx]) if scalp_idx >= 0 else None
    # simple R-peak detection on scalp for reference (not used in SSL)
    return dict(abdominal=abdo, scalp=scalp, fs=fs, record=record_id,
                ga_weeks=None, subject=record_id, dataset="ADfECGDB")


def load_ninfea(record_id=1):
    """Load NInFEA WFDB record. Returns pre-processed signals + metadata."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb required; pip install wfdb")
    from ninfea_labels import GA_WEEKS, SUBJECT
    p = DATA_ROOT / "ninfea" / str(record_id)
    if not p.with_suffix(".hea").exists():
        return None
    rec = wfdb.rdrecord(str(p))
    sig, fs = rec.p_signal.T, rec.fs  # (n_samp, n_chan) → (n_chan, n_samp)
    # NInFEA has 24 abdominal + 3 thoracic channels; keep the 24 abdominal,
    # then reduce to the 4 highest fetal-band-kurtosis leads for a fixed montage.
    abdo = sig[:24]
    abdo, fs = preprocess(abdo, fs)
    abdo = select_channels(abdo, fs, n_keep=4)
    return dict(
        abdominal=abdo, fs=fs, record=record_id,
        ga_weeks=GA_WEEKS.get(record_id, None),
        subject=SUBJECT.get(record_id, None),
        dataset="NInFEA",
    )


def load_cinc2013(record_id="a01"):
    """Load CinC 2013 set-a record."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb required; pip install wfdb")
    p = DATA_ROOT / "cinc2013" / record_id
    if not p.with_suffix(".hea").exists():
        return None
    rec = wfdb.rdrecord(str(p))
    sig, fs = rec.p_signal.T, rec.fs
    abdo, fs = preprocess(sig[:4], fs)
    # load reference annotations if available
    fqrs = []
    try:
        ann = wfdb.rdann(str(p), "fqrs")
        fqrs = ann.sample
    except Exception:
        pass
    return dict(
        abdominal=abdo, fs=fs, record=record_id,
        fetal_beats=fqrs, dataset="CinC2013",
        ga_weeks=None, subject=None,
    )


def load_fecgsyndb(record_id=1):
    """Load FECGSYNDB synthetic record.

    FECGSYNDB stores .mat files with keys: 'fECG' (fetal), 'mECG' (maternal),
    'noise', 'fRef', etc. We return the mixed abdominal signal + fetal reference.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy required")
    # FECGSYNDB structure: subfolders by SNR and noise type
    root = DATA_ROOT / "fecgsyndb"
    if not root.exists():
        return None
    # heuristic: find first .mat file matching pattern
    mats = list(root.rglob("*.mat"))
    if not mats:
        return None
    idx = (record_id - 1) % len(mats)
    mat = loadmat(str(mats[idx]), squeeze_me=True)
    # typical keys: 'ecg', 'fecg', 'mecg', 'noise', 'fRef', 'mRef'
    keys = [k for k in mat.keys() if not k.startswith("__")]
    # Try common key names
    sig = None
    for k in ("ecg", "ECG", "mixture", "abdominal"):
        if k in mat:
            sig = mat[k]
            break
    if sig is None and keys:
        # fallback: first numeric array with >1 element
        for k in keys:
            v = mat[k]
            if isinstance(v, np.ndarray) and v.size > 100:
                sig = v
                break
    if sig is None:
        return None
    sig = np.atleast_2d(sig)
    if sig.shape[0] > sig.shape[1]:
        sig = sig.T
    fs = 250.0  # FECGSYNDB standard
    abdo, fs = preprocess(sig, fs)
    abdo = select_channels(abdo, fs, n_keep=4)
    return dict(
        abdominal=abdo, fs=fs, record=f"fecgsyn_{idx}",
        dataset="FECGSYNDB", ga_weeks=None, subject=None, synthetic=True,
    )


def load_nifecgdb(record_id=1):
    """Load Non-Invasive Fetal ECG Database (single-subject longitudinal, 55 recs).

    Records are EDF files (ecgca*.edf), each with 1 direct (scalp/abdominal
    reference) + several abdominal channels. We keep the abdominal channels.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib required for nifecgdb")
    root = DATA_ROOT / "nifecgdb"
    if not root.exists():
        return None
    edfs = sorted(root.glob("*.edf"))
    if not edfs:
        return None
    idx = (record_id - 1) % len(edfs)
    path = edfs[idx]
    try:
        f = pyedflib.EdfReader(str(path))
    except Exception:
        return None
    n = f.signals_in_file
    labels = [f.getLabel(i) for i in range(n)]
    sigs = [f.readSignal(i) for i in range(n)]
    fs = f.getSampleFrequency(0)
    f.close()
    # Drop a direct/reference channel if labelled; keep abdominal leads.
    keep = [i for i, lab in enumerate(labels) if "direct" not in lab.lower()]
    sig = np.array([sigs[i] for i in keep]) if keep else np.array(sigs)
    abdo, fs = preprocess(sig, fs)
    abdo = select_channels(abdo, fs, n_keep=4)
    return dict(
        abdominal=abdo, fs=fs, record=path.stem,
        dataset="NI_FECG_DB", ga_weeks=None, subject=0,  # single subject
    )


def load_nifeadb(record_id=1):
    """Load Non-Invasive Fetal ECG Arrhythmia Database."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb required")
    root = DATA_ROOT / "nifeadb"
    if not root.exists():
        return None
    hea_files = sorted(root.glob("*.hea"))
    if not hea_files:
        return None
    idx = (record_id - 1) % len(hea_files)
    stem = hea_files[idx].with_suffix("")
    rec = wfdb.rdrecord(str(stem))
    sig, fs = rec.p_signal.T, rec.fs
    abdo, fs = preprocess(sig, fs)
    abdo = select_channels(abdo, fs, n_keep=4)
    return dict(
        abdominal=abdo, fs=fs, record=stem.name,
        dataset="NIFEA_ARR", ga_weeks=None, subject=None,
    )


# --------------------------------------------------------------------------- #
# Master loader
# --------------------------------------------------------------------------- #

LOADER_MAP = {
    "ADfECGDB": load_adfecgdb,
    "NInFEA": load_ninfea,
    "CinC2013": load_cinc2013,
    "FECGSYNDB": load_fecgsyndb,
    "NI_FECG_DB": load_nifecgdb,
    "NIFEA_ARR": load_nifeadb,
}


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"{fn.__name__}{a}: {e}")
        return None


def load_all_records(datasets=None, use_cache=True):
    """Load available records. ``datasets`` optionally restricts the sources.

    If a preprocessed cache exists (see build_cache.py) it is used, which is
    dramatically faster than re-filtering the raw recordings. Any record that
    fails to load (e.g. a partially downloaded file) is skipped, not raised.
    """
    if use_cache:
        try:
            from build_cache import load_cached
            cached = load_cached(datasets)
            if cached:
                print(f"Loaded {len(cached)} records from cache"
                      f"{' (' + ','.join(sorted(set(datasets))) + ')' if datasets else ''}.")
                return cached
        except Exception:
            pass

    want = set(datasets) if datasets is not None else None

    def enabled(name):
        return want is None or name in want

    records = []
    if enabled("ADfECGDB"):
        for rid in ["r01", "r04", "r07", "r08", "r10"]:
            r = _safe(load_adfecgdb, rid)
            if r: records.append(r)
    if enabled("NInFEA"):
        for i in range(1, 61):
            r = _safe(load_ninfea, i)
            if r: records.append(r)
    if enabled("CinC2013"):
        for i in range(1, 76):
            r = _safe(load_cinc2013, f"a{i:02d}")
            if r: records.append(r)
    if enabled("FECGSYNDB"):
        for i in range(1, 51):
            r = _safe(load_fecgsyndb, i)
            if r: records.append(r)
    if enabled("NI_FECG_DB"):
        n = len(sorted((DATA_ROOT / "nifecgdb").glob("*.edf"))) if (DATA_ROOT / "nifecgdb").exists() else 0
        for i in range(1, n + 1):
            r = _safe(load_nifecgdb, i)
            if r: records.append(r)
    if enabled("NIFEA_ARR"):
        n = len(sorted((DATA_ROOT / "nifeadb").glob("*.hea"))) if (DATA_ROOT / "nifeadb").exists() else 0
        for i in range(1, n + 1):
            r = _safe(load_nifeadb, i)
            if r: records.append(r)
    print(f"Loaded {len(records)} records total"
          f"{' from ' + ','.join(sorted(want)) if want else ''}.")
    return records


if __name__ == "__main__":
    recs = load_all_records()
    df = pd.DataFrame([
        {"dataset": r["dataset"], "record": r["record"], "fs": r["fs"],
         "n_chan": r["abdominal"].shape[0], "n_samp": r["abdominal"].shape[1],
         "ga_weeks": r.get("ga_weeks"), "subject": r.get("subject")}
        for r in recs
    ])
    print(df.groupby("dataset").size())
    print(df[["dataset", "ga_weeks"]].dropna().head(10))
