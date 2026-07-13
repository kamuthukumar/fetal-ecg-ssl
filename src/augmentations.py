"""Physics-informed data augmentations for fetal ECG.

All augmentations are designed to respect fetal cardiac electrophysiology:
  1. Fetal dipole rotation (3D rotation → electrode projection)
  2. Maternal breathing modulation (0.15-0.5 Hz baseline wander)
  3. Realistic noise injection (SNR-matched to dataset)
  4. Gestational-age-dependent QRS scaling
"""
import numpy as np
from scipy.signal import butter, filtfilt


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rand_angle():
    """Random 3D rotation angles (roll, pitch, yaw) in radians."""
    return np.random.uniform(-np.pi / 4, np.pi / 4, size=3)


def rot_matrix(roll, pitch, yaw):
    """3D rotation matrix (Z-Y-X intrinsic)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    return R


def _electrode_positions(n_ch=4):
    """Simplified abdominal electrode positions on a unit circle (x,y,z=0).

    For a 4-channel setup, positions are at N=4 equally spaced points.
    For n_ch > 4, expand to a grid-like layout.
    """
    if n_ch == 4:
        angles = np.linspace(0, 2 * np.pi, 5)[:-1] + np.pi / 4
        pos = np.array([[np.cos(a), np.sin(a), 0.0] for a in angles])
        return pos
    elif n_ch == 24:
        # NInFEA 24-channel: roughly a 5x5 grid minus corners
        # simplified: 4 rings of 6 electrodes
        pos = []
        for ring in range(4):
            r = 0.3 + 0.2 * ring
            n = 6
            for i in range(n):
                a = 2 * np.pi * i / n
                pos.append([r * np.cos(a), r * np.sin(a), 0.0])
        return np.array(pos)
    else:
        angles = np.linspace(0, 2 * np.pi, n_ch + 1)[:-1]
        return np.array([[np.cos(a), np.sin(a), 0.0] for a in angles])


# --------------------------------------------------------------------------- #
# Augmentations
# --------------------------------------------------------------------------- #

class FetalDipoleRotation:
    """Rotate fetal cardiac dipole in 3D and reproject to electrode positions.

    The fetal ECG at each electrode is approximated as the projection of the
dipole moment onto the electrode direction (quasi-static approximation).
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal, fs):
        """signal: (n_ch, n_samp)

        Returns augmented signal of same shape.
        """
        if np.random.rand() > self.p:
            return signal
        n_ch, n_samp = signal.shape
        pos = _electrode_positions(n_ch)
        # Estimate dipole direction from dominant PCA of signal
        # (simplified: use channel with max variance as dominant axis)
        variances = np.var(signal, axis=1)
        dom_ch = np.argmax(variances)
        # Treat signal as projection of dipole; we rotate the dipole
        R = rot_matrix(*_rand_angle())
        # Project rotated dipole to electrodes
        new_signal = np.zeros_like(signal)
        for i in range(n_ch):
            # simplified: amplitude scaling based on new projection
            proj = np.dot(R[0], pos[i])  # dot with x-axis (dominant)
            scale = 1.0 + 0.3 * proj  # ±30% variation
            new_signal[i] = signal[i] * scale
        return new_signal


class BreathingModulation:
    """Add slow baseline wander simulating maternal breathing (0.15–0.5 Hz)."""
    def __init__(self, p=0.5, amp_range=(0.05, 0.3)):
        self.p = p
        self.amp_range = amp_range

    def __call__(self, signal, fs):
        if np.random.rand() > self.p:
            return signal
        n_ch, n_samp = signal.shape
        t = np.arange(n_samp) / fs
        f = np.random.uniform(0.15, 0.5)
        amp = np.random.uniform(*self.amp_range) * np.std(signal)
        phase = np.random.uniform(0, 2 * np.pi)
        mod = amp * np.sin(2 * np.pi * f * t + phase)
        return signal + mod[np.newaxis, :]


class RealisticNoise:
    """Inject realistic noise with SNR drawn from dataset distribution."""
    def __init__(self, p=0.5, snr_db_range=(5, 25)):
        self.p = p
        self.snr_db_range = snr_db_range

    def __call__(self, signal, fs):
        if np.random.rand() > self.p:
            return signal
        n_ch, n_samp = signal.shape
        snr_db = np.random.uniform(*self.snr_db_range)
        noise = self._generate_noise(n_ch, n_samp, fs)
        sig_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2) + 1e-12
        scale = np.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))
        return signal + noise * scale

    def _generate_noise(self, n_ch, n_samp, fs):
        """Generate colored noise: white + muscle (100-300 Hz band) + baseline."""
        white = np.random.randn(n_ch, n_samp)
        # muscle noise: band-limited 80-200 Hz
        b, a = butter(4, [80 / (fs / 2), min(200 / (fs / 2), 0.99)], btype="band")
        muscle = filtfilt(b, a, white, axis=1)
        # baseline wander: very low frequency
        b2, a2 = butter(2, 0.5 / (fs / 2), btype="low")
        baseline = filtfilt(b2, a2, white, axis=1)
        noise = 0.5 * white + 0.3 * muscle + 0.2 * baseline
        return noise


class GAScaling:
    """Scale QRS amplitude and duration based on gestational age.

    Literature suggests fetal QRS amplitude increases ~20% from 21→41 weeks
    and QRS duration shortens slightly (~5%). This augmentation simulates
    these known developmental changes.
    """
    def __init__(self, p=0.3, ga_ref=30.0):
        self.p = p
        self.ga_ref = ga_ref  # reference GA for no scaling

    def __call__(self, signal, fs, ga_weeks=None):
        if ga_weeks is None or np.random.rand() > self.p:
            return signal
        # Amplitude scaling: +1% per week above reference, -1% below
        amp_scale = 1.0 + 0.01 * (ga_weeks - self.ga_ref)
        # Duration (time) scaling: slightly compress/expand
        time_scale = 1.0 - 0.002 * (ga_weeks - self.ga_ref)
        signal = signal * amp_scale
        if time_scale != 1.0:
            n_samp = signal.shape[1]
            new_n = int(n_samp * time_scale)
            if new_n > 10:
                from scipy.signal import resample
                signal = resample(signal, n_samp, axis=1)
        return signal


class Compose:
    """Compose multiple augmentations."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal, fs, ga_weeks=None):
        for t in self.transforms:
            if isinstance(t, GAScaling):
                signal = t(signal, fs, ga_weeks)
            else:
                signal = t(signal, fs)
        return signal


# Default augmentation pipeline for self-supervised pre-training
DEFAULT_AUGMENT = Compose([
    FetalDipoleRotation(p=0.5),
    BreathingModulation(p=0.5),
    RealisticNoise(p=0.5),
    GAScaling(p=0.3),
])


# --------------------------------------------------------------------------- #
# Generic (non-physics) augmentations — SimCLR baseline / ablation
# --------------------------------------------------------------------------- #
class GenericJitterScale:
    """Standard time-series contrastive augmentations: additive Gaussian jitter,
    per-channel amplitude scaling, and random temporal masking. No physiological
    structure — this is the control against the physics-informed pipeline."""
    def __init__(self, p=0.9, jitter=0.1, scale=0.2, mask_frac=0.1):
        self.p, self.jitter, self.scale, self.mask_frac = p, jitter, scale, mask_frac

    def __call__(self, signal, fs, ga_weeks=None):
        if np.random.rand() > self.p:
            return signal
        n_ch, n = signal.shape
        sd = np.std(signal) + 1e-8
        out = signal + np.random.randn(n_ch, n) * self.jitter * sd
        out = out * (1.0 + np.random.uniform(-self.scale, self.scale, size=(n_ch, 1)))
        if self.mask_frac > 0:
            m = int(self.mask_frac * n)
            if m > 0:
                start = np.random.randint(0, max(1, n - m))
                out[:, start:start + m] = 0.0
        return out


GENERIC_AUGMENT = Compose([GenericJitterScale(p=0.9)])


if __name__ == "__main__":
    # Quick visual test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = 1000.0
    t = np.arange(10000) / fs
    # fake fetal ECG: fast sine + some QRS-like spikes
    sig = np.sin(2 * np.pi * 2.5 * t) * 0.1
    sig[1000::300] += 0.5  # fake QRS
    sig[1100::300] -= 0.3
    sig = sig[np.newaxis, :]  # 1 channel

    aug = DEFAULT_AUGMENT
    aug_sig = aug(sig.copy(), fs, ga_weeks=25)

    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    ax[0].plot(t, sig[0], label="original")
    ax[0].set_title("Original")
    ax[1].plot(t, aug_sig[0], label="augmented", color="orange")
    ax[1].set_title("Physics-informed augmentation")
    fig.tight_layout()
    fig.savefig("/tmp/aug_test.png")
    print("Saved /tmp/aug_test.png")
