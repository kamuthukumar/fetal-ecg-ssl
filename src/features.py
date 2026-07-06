"""Physiologically-interpretable features from a fetal R-peak series + fetal-band ECG.

All features are computed from quantities we can actually measure from the
extracted fetal ECG (R-peak timing + local morphology). No feature depends on
ground-truth GA. Features:

  HRV / rhythm:
    mean_fhr, sdnn, rmssd, pnn20, cv_rr,
    sampen (sample entropy of RR), dfa_alpha1 (short-scale DFA),
    lf_hf (spectral LF/HF of RR)
  Morphology (from fetal-band waveform around detected beats):
    qrs_amp_med, qrs_amp_cv, qrs_width_med, beat_template_corr
"""
import numpy as np
from scipy.signal import periodogram


def _rr_series(peaks, fs):
    if len(peaks) < 4:
        return np.array([])
    rr = np.diff(np.sort(peaks)) / fs * 1000.0     # ms
    # remove physiologically impossible RR (fetal HR 90-200 bpm -> 300-667 ms)
    rr = rr[(rr > 250) & (rr < 700)]
    return rr


def _sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, float)
    n = len(x)
    if n < m + 2:
        return np.nan
    r *= np.std(x) + 1e-12

    def _phi(mm):
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(d <= r) - 1
        return count

    a = _phi(m + 1)
    b = _phi(m)
    if b == 0 or a == 0:
        return np.nan
    return -np.log(a / b)


def _dfa_alpha1(x, scales=range(4, 17)):
    x = np.asarray(x, float)
    if len(x) < 20:
        return np.nan
    y = np.cumsum(x - np.mean(x))
    F = []
    used = []
    for s in scales:
        if s >= len(y):
            break
        n_seg = len(y) // s
        if n_seg < 1:
            continue
        rms = []
        for j in range(n_seg):
            seg = y[j * s:(j + 1) * s]
            t = np.arange(s)
            coef = np.polyfit(t, seg, 1)
            fit = np.polyval(coef, t)
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        F.append(np.mean(rms))
        used.append(s)
    if len(F) < 3:
        return np.nan
    coef = np.polyfit(np.log(used), np.log(F), 1)
    return coef[0]


def _lf_hf(rr_ms):
    """LF/HF from RR series resampled to 4 Hz."""
    if len(rr_ms) < 16:
        return np.nan
    t = np.cumsum(rr_ms) / 1000.0
    fs_i = 4.0
    tn = np.arange(t[0], t[-1], 1 / fs_i)
    if len(tn) < 16:
        return np.nan
    rri = np.interp(tn, t, rr_ms)
    rri = rri - np.mean(rri)
    f, p = periodogram(rri, fs=fs_i)
    lf = np.trapezoid(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapezoid(p[(f >= 0.15) & (f < 1.0)], f[(f >= 0.15) & (f < 1.0)])
    return lf / hf if hf > 0 else np.nan


def _morphology(sig, peaks, fs, half_ms=40):
    w = int(half_ms * fs / 1000)
    beats, amps, widths = [], [], []
    for p in peaks:
        a, b = p - w, p + w
        if a < 0 or b >= len(sig):
            continue
        seg = sig[a:b] - np.median(sig[a:b])
        beats.append(seg)
        amps.append(np.max(np.abs(seg)))
        # width at half-max of |seg|
        env = np.abs(seg)
        thr = 0.5 * env.max()
        widths.append(np.sum(env >= thr) / fs * 1000.0)
    if len(beats) < 3:
        return dict(qrs_amp_med=np.nan, qrs_amp_cv=np.nan,
                    qrs_width_med=np.nan, beat_template_corr=np.nan)
    L = min(len(b) for b in beats)
    beats = np.array([b[:L] for b in beats])
    template = np.median(beats, axis=0)
    corrs = [np.corrcoef(b, template)[0, 1] for b in beats]
    return dict(
        qrs_amp_med=float(np.median(amps)),
        qrs_amp_cv=float(np.std(amps) / (np.mean(amps) + 1e-12)),
        qrs_width_med=float(np.median(widths)),
        beat_template_corr=float(np.nanmean(corrs)),
    )


FEATURE_NAMES = [
    "mean_fhr", "sdnn", "rmssd", "pnn20", "cv_rr",
    "sampen", "dfa_alpha1", "lf_hf",
    "qrs_amp_med", "qrs_amp_cv", "qrs_width_med", "beat_template_corr",
]


def compute_features(peaks, fetal_sig, fs):
    rr = _rr_series(peaks, fs)
    feats = {n: np.nan for n in FEATURE_NAMES}
    if len(rr) >= 4:
        feats["mean_fhr"] = 60000.0 / np.mean(rr)
        feats["sdnn"] = np.std(rr)
        feats["rmssd"] = np.sqrt(np.mean(np.diff(rr) ** 2))
        feats["pnn20"] = np.mean(np.abs(np.diff(rr)) > 20) * 100
        feats["cv_rr"] = np.std(rr) / (np.mean(rr) + 1e-12)
        feats["sampen"] = _sample_entropy(rr)
        feats["dfa_alpha1"] = _dfa_alpha1(rr)
        feats["lf_hf"] = _lf_hf(rr)
    feats.update(_morphology(fetal_sig, peaks, fs))
    return feats
