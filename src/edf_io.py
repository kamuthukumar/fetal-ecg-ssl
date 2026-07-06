"""Minimal, dependency-free EDF / EDF+ reader for ADfECGDB.

Returns per-signal waveforms at their true sampling rate and parses the
EDF+ 'EDF Annotations' channel to recover fetal R-peak onset times (seconds).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EdfRecord:
    fs: float
    labels: list
    signals: dict          # label -> np.ndarray (physical units)
    ann_times: np.ndarray  # annotation onsets in seconds
    duration_s: float


def _read_ascii(f, n):
    return f.read(n).decode("ascii", errors="replace").strip()


def read_edf(path):
    path = Path(path)
    with open(path, "rb") as f:
        f.read(8)
        f.read(80); f.read(80); f.read(8); f.read(8); f.read(8); f.read(44)
        n_records = int(_read_ascii(f, 8))
        rec_dur = float(_read_ascii(f, 8))
        n_sig = int(_read_ascii(f, 4))
        labels = [_read_ascii(f, 16) for _ in range(n_sig)]
        for _ in range(n_sig): f.read(80)             # transducer
        phys_dim = [_read_ascii(f, 8) for _ in range(n_sig)]
        phys_min = [float(_read_ascii(f, 8)) for _ in range(n_sig)]
        phys_max = [float(_read_ascii(f, 8)) for _ in range(n_sig)]
        dig_min = [float(_read_ascii(f, 8)) for _ in range(n_sig)]
        dig_max = [float(_read_ascii(f, 8)) for _ in range(n_sig)]
        for _ in range(n_sig): f.read(80)             # prefilter
        spr = [int(_read_ascii(f, 8)) for _ in range(n_sig)]
        for _ in range(n_sig): f.read(32)             # reserved
        data = f.read()

    ann_idx = next((i for i, l in enumerate(labels) if "Annotation" in l), None)
    scale = [(phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i]) if dig_max[i] != dig_min[i] else 1.0
             for i in range(n_sig)]
    offset = [phys_min[i] - scale[i] * dig_min[i] for i in range(n_sig)]

    sig_buffers = {labels[i]: [] for i in range(n_sig)}
    ann_bytes = bytearray()
    record_stride = sum(spr) * 2
    pos = 0
    for _ in range(n_records):
        for i in range(n_sig):
            nbytes = spr[i] * 2
            chunk = data[pos:pos + nbytes]
            pos += nbytes
            if i == ann_idx:
                ann_bytes += chunk
            else:
                vals = np.frombuffer(chunk, dtype="<i2").astype(np.float64)
                sig_buffers[labels[i]].append(vals * scale[i] + offset[i])

    signals = {}
    fs = None
    for i in range(n_sig):
        if i == ann_idx:
            continue
        signals[labels[i]] = np.concatenate(sig_buffers[labels[i]])
        fs = spr[i] / rec_dur

    ann_times = _parse_edf_annotations(ann_bytes)
    return EdfRecord(fs=fs, labels=[l for i, l in enumerate(labels) if i != ann_idx],
                     signals=signals, ann_times=ann_times,
                     duration_s=n_records * rec_dur)


def _parse_edf_annotations(raw: bytes) -> np.ndarray:
    """Parse TALs from an EDF+ annotation stream. Return onset times (s).

    A TAL: [onset]\x15[duration]\x14[annotation]\x14\x00 . The first TAL of each
    record is a timekeeping TAL (onset then \x14\x14\x00). We keep every onset
    that carries a non-empty annotation text (i.e. an R-peak marker)."""
    times = []
    for tal in raw.split(b"\x00"):
        if not tal.strip(b"\x14\x15 "):
            continue
        # onset is up to first \x14 or \x15
        # split off duration if present
        head = tal.split(b"\x14", 1)
        onset_dur = head[0]
        text = head[1] if len(head) > 1 else b""
        if b"\x15" in onset_dur:
            onset_b, _dur = onset_dur.split(b"\x15", 1)
        else:
            onset_b = onset_dur
        try:
            onset = float(onset_b.decode("ascii").strip())
        except ValueError:
            continue
        # keep only TALs that actually carry annotation text (skip pure timekeeping)
        label = text.split(b"\x14")[0].decode("ascii", errors="replace").strip()
        if label != "":
            times.append(onset)
    return np.array(sorted(set(times)))


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[1] / "data" / "adfecgdb"
    for edf in sorted(data.glob("*.edf")):
        r = read_edf(edf)
        rr = np.diff(r.ann_times)
        fhr = 60 / np.median(rr) if len(rr) else float("nan")
        print(f"{edf.name}: fs={r.fs:.0f}Hz dur={r.duration_s:.0f}s "
              f"chans={r.labels} n_fQRS={len(r.ann_times)} "
              f"median_FHR={fhr:.0f}bpm")
