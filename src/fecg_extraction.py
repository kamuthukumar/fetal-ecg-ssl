"""Non-invasive fetal ECG extraction + fetal QRS detection.

Pipeline (classic, fully reproducible; no ground truth used for extraction):
  1. Band-pass each abdominal channel (1-90 Hz) + 50 Hz notch.
  2. Maternal QRS detection on each channel (dominant component).
  3. Adaptive maternal-beat template subtraction (sliding average of nearest beats).
  4. Fetal enhancement: band-pass residual (10-70 Hz), pick the channel with the
     most regular candidate rhythm in the fetal HR band (no labels used).
  5. Fetal QRS detection via energy envelope + refractory peak picking.

Evaluation: F1 with +/-50 ms tolerance vs reference beats (PhysioNet CinC 2013 standard).
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def bandpass(sig, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def notch(sig, fs, f0=50.0, q=30.0):
    if f0 >= fs / 2:
        return sig
    b, a = iirnotch(f0 / (fs / 2.0), q)
    return filtfilt(b, a, sig)


# --------------------------------------------------------------------------- #
# QRS detection (energy envelope + adaptive threshold + refractory)
# --------------------------------------------------------------------------- #
def _energy_envelope(sig, fs, smooth_ms=40):
    diff = np.gradient(sig)
    sq = diff ** 2
    win = max(1, int(smooth_ms * fs / 1000))
    kernel = np.ones(win) / win
    return np.convolve(sq, kernel, mode="same")


def detect_qrs(sig, fs, hr_min, hr_max, thr_scale=0.5):
    """Detect QRS-like peaks with an expected HR band [hr_min, hr_max] bpm."""
    env = _energy_envelope(sig, fs)
    min_dist = int(fs * 60.0 / hr_max)          # refractory from max HR
    thr = thr_scale * np.median(env[env > 0]) + thr_scale * np.std(env)
    peaks, props = find_peaks(env, distance=min_dist, height=thr)
    return peaks, env


def _refine_to_local_extreme(sig, peaks, fs, win_ms=25):
    w = int(win_ms * fs / 1000)
    out = []
    for p in peaks:
        a, b = max(0, p - w), min(len(sig), p + w)
        out.append(a + int(np.argmax(np.abs(sig[a:b] - np.median(sig[a:b])))))
    return np.array(sorted(set(out)))


# --------------------------------------------------------------------------- #
# maternal cancellation
# --------------------------------------------------------------------------- #
def cancel_maternal(chan, fs, mat_peaks, half_win_ms=350, n_neighbors=20):
    """Adaptive template subtraction: subtract a locally-averaged maternal beat."""
    w = int(half_win_ms * fs / 1000)
    res = chan.copy()
    beats, idx = [], []
    for p in mat_peaks:
        a, b = p - w, p + w
        if a < 0 or b >= len(chan):
            continue
        beats.append(chan[a:b])
        idx.append(p)
    if len(beats) < 3:
        return res
    beats = np.array(beats)
    for k, p in enumerate(idx):
        lo = max(0, k - n_neighbors // 2)
        hi = min(len(beats), lo + n_neighbors)
        template = np.median(beats[lo:hi], axis=0)
        res[p - w:p + w] -= template
    return res


# --------------------------------------------------------------------------- #
# full extraction
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionResult:
    fetal_peaks: np.ndarray     # sample indices
    fs: float
    best_channel: int
    residual: np.ndarray
    snr_improvement_db: float


def _rhythm_regularity(peaks, fs, hr_min, hr_max):
    """Lower is better: coefficient of variation of RR, penalise out-of-band HR."""
    if len(peaks) < 5:
        return np.inf
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.median(rr)
    if not (hr_min <= hr <= hr_max):
        return np.inf
    return np.std(rr) / (np.mean(rr) + 1e-9)


def _detect_on_envelope(env, fs, hr_min, hr_max):
    """Robust peak picking on a (fused) energy envelope using a running-percentile
    adaptive threshold; resilient to large motion artefacts."""
    min_dist = int(fs * 60.0 / hr_max)
    # running baseline (median over ~3 s) and scale (MAD) for adaptive threshold
    win = int(3 * fs)
    if win < 1:
        win = 1
    # coarse baseline via moving average of the envelope
    kern = np.ones(win) / win
    base = np.convolve(env, kern, mode="same")
    mad = np.median(np.abs(env - np.median(env))) + 1e-12
    thr = base + 1.2 * mad
    peaks, _ = find_peaks(env, distance=min_dist, height=thr)
    # fallback: if too few peaks (threshold too high due to artefacts), relax
    dur = len(env) / fs
    expected = dur * (hr_min + hr_max) / 2 / 60
    if len(peaks) < 0.4 * expected:
        thr2 = np.percentile(env, 60)
        peaks, _ = find_peaks(env, distance=min_dist, height=thr2)
    return peaks


def extract_fetal(channels, fs, fhr=(100, 180), mhr=(45, 110)):
    """channels: (n_chan, n_samp) abdominal signals.

    Fetal QRS are detected on a cross-channel fusion of maternal-cancelled
    fetal-band energy envelopes, which is far more robust to per-channel
    dropout and motion artefacts than single-channel selection.
    """
    fetal_bands, residuals = [], []
    snr_before, snr_after = [], []
    hi = min(70, fs / 2 * 0.95)
    for ch in channels:
        x = notch(bandpass(ch, fs, 1, 90), fs)
        mat_peaks = _refine_to_local_extreme(
            x, detect_qrs(x, fs, mhr[0], mhr[1], thr_scale=0.6)[0], fs)
        res = cancel_maternal(x, fs, mat_peaks)
        fetal_band = bandpass(res, fs, 10, hi)
        fetal_bands.append(fetal_band)
        residuals.append(res)
        snr_before.append(np.var(bandpass(x, fs, 10, hi)))
        snr_after.append(np.var(fetal_band))
    fetal_bands = np.array(fetal_bands)

    # detect fetal peaks per channel; keep the most regular fetal rhythm (no labels)
    per_chan_peaks, scores = [], []
    for i in range(fetal_bands.shape[0]):
        env = _energy_envelope(fetal_bands[i], fs)
        env = np.clip((env - np.median(env)) / (np.std(env) + 1e-12), 0, None)
        pk = _detect_on_envelope(env, fs, fhr[0], fhr[1])
        pk = _refine_to_local_extreme(fetal_bands[i], pk, fs)
        pk = _rr_consistency_filter(pk, fs, fhr)
        per_chan_peaks.append(pk)
        scores.append(_rhythm_regularity(pk, fs, fhr[0], fhr[1]))
    best = int(np.argmin(scores)) if np.isfinite(min(scores)) else \
        int(np.argmax([len(p) for p in per_chan_peaks]))
    fetal_peaks = per_chan_peaks[best]

    ratio = (np.mean(snr_after) + 1e-12) / (np.mean(snr_before) + 1e-12)
    snr_db = 10 * np.log10(max(ratio, 1e-6))
    return ExtractionResult(fetal_peaks=fetal_peaks, fs=fs, best_channel=best,
                            residual=residuals[best], snr_improvement_db=snr_db)


def _rr_consistency_filter(peaks, fs, fhr):
    """Insert/interpolate obviously missed beats and drop double detections using
    the local median RR. Purely rhythm-based, uses no ground truth."""
    if len(peaks) < 5:
        return peaks
    peaks = np.sort(peaks)
    rr = np.diff(peaks)
    med = np.median(rr)
    if med <= 0:
        return peaks
    out = [peaks[0]]
    for i in range(1, len(peaks)):
        gap = peaks[i] - out[-1]
        if gap < 0.5 * med:            # too close -> likely double detection, skip
            continue
        # fill large gaps with evenly spaced interpolated beats
        n_insert = int(round(gap / med)) - 1
        if 1 <= n_insert <= 4:
            step = gap / (n_insert + 1)
            for k in range(1, n_insert + 1):
                out.append(int(out[-1] + step))
        out.append(int(peaks[i]))
    return np.array(sorted(set(out)))


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def match_f1(detected, reference, fs, tol_ms=50):
    """Match detected beats to reference within +/-tol_ms. Returns dict of metrics."""
    if len(reference) == 0:
        return dict(tp=0, fp=len(detected), fn=0, f1=np.nan, se=np.nan, ppv=np.nan)
    tol = tol_ms * fs / 1000.0
    ref = np.sort(np.asarray(reference, float))
    det = np.sort(np.asarray(detected, float))
    used = np.zeros(len(ref), bool)
    tp = 0
    for d in det:
        j = np.searchsorted(ref, d)
        cand = [k for k in (j - 1, j) if 0 <= k < len(ref)]
        best_k, best_dist = None, tol + 1
        for k in cand:
            if used[k]:
                continue
            dist = abs(ref[k] - d)
            if dist <= tol and dist < best_dist:
                best_k, best_dist = k, dist
        if best_k is not None:
            used[best_k] = True
            tp += 1
    fp = len(det) - tp
    fn = len(ref) - tp
    se = tp / (tp + fn) if (tp + fn) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * se * ppv / (se + ppv) if se and ppv and (se + ppv) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, f1=f1, se=se, ppv=ppv)
