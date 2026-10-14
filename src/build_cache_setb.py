"""Expanded cache: the fixed 68-record pre-training cache plus all other usable
open recordings available locally (Reviewer 3, R12).

* CinC 2013 set-b (100 one-minute records b00--b99, same hardware as the set-a
  benchmark, no public fetal QRS reference) as unlabelled data.
* Every further NI-FECG EDF that is complete and readable (files that are
  truncated are skipped).

The original cache is left untouched.

    python src/build_cache_setb.py
    FECG_CACHE=results/cache/records_setb.pkl python src/train_foundation.py \
        --pretrain_datasets NInFEA,NI_FECG_DB,CinC2013b ...
"""
import pickle
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

BASE = ROOT / "results" / "cache" / "records.pkl"
OUT = ROOT / "results" / "cache" / "records_setb.pkl"


def main():
    import numpy as np
    import pyedflib
    import wfdb
    from data_loader import DATA_ROOT, preprocess, select_channels
    with open(BASE, "rb") as fh:
        recs = pickle.load(fh)
    have = {r["record"] for r in recs}
    added = 0
    for n in range(100):
        p = DATA_ROOT / "cinc2013b" / f"b{n:02d}"
        if not (p.with_suffix(".hea").exists() and p.with_suffix(".dat").exists()):
            continue
        try:
            r = wfdb.rdrecord(str(p))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        abdo, fs = preprocess(r.p_signal.T[:4], r.fs)
        recs.append(dict(abdominal=np.asarray(abdo, np.float32), fs=fs, record=p.name,
                         dataset="CinC2013b", ga_weeks=None, subject=None, fetal_beats=[]))
        added += 1
    extra_nifecg = []
    for p in sorted((DATA_ROOT / "nifecgdb").glob("*.edf")):
        if p.stem in have:
            continue
        try:
            f = pyedflib.EdfReader(str(p))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {str(e)[:60]}")
            continue
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        sigs = [f.readSignal(i) for i in range(f.signals_in_file)]
        fs = f.getSampleFrequency(0)
        f.close()
        keep = [i for i, lab in enumerate(labels) if "direct" not in lab.lower()]
        abdo, fs = preprocess(np.array([sigs[i] for i in keep]), fs)
        abdo = select_channels(abdo, fs, n_keep=4)
        recs.append(dict(abdominal=np.asarray(abdo, np.float32), fs=fs, record=p.stem,
                         dataset="NI_FECG_DB", ga_weeks=None, subject=0))
        extra_nifecg.append(p.stem)
    with open(OUT, "wb") as fh:
        pickle.dump(recs, fh)
    from collections import Counter
    print(f"added {added} set-b records and NI-FECG {extra_nifecg} -> {OUT}")
    print(dict(Counter(r["dataset"] for r in recs)))


if __name__ == "__main__":
    main()
