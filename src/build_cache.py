"""Pre-compute and cache preprocessed 4-channel records for fast reuse.

Loading + filtering + channel-selecting the raw PhysioNet records (especially
NInFEA's 24-channel 2 kHz recordings) is the slow part of every experiment.
This script does it once and pickles the result so pre-training, the SimCLR
ablation, the random baseline, and all evaluations reload in seconds.

    python src/build_cache.py            # cache every available dataset
"""
import os
import pickle
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

CACHE = Path(os.environ.get("FECG_CACHE", ROOT / "results" / "cache" / "records.pkl"))


def build():
    from data_loader import load_all_records
    records = load_all_records(use_cache=False)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    # keep only the fields downstream code needs (drop scalp etc. to save space)
    slim = []
    for r in records:
        slim.append(dict(
            abdominal=r["abdominal"], fs=r["fs"], record=r["record"],
            dataset=r["dataset"], ga_weeks=r.get("ga_weeks"),
            subject=r.get("subject"), fetal_beats=r.get("fetal_beats", []),
        ))
    with open(CACHE, "wb") as fh:
        pickle.dump(slim, fh)
    from collections import Counter
    print(f"Cached {len(slim)} records -> {CACHE}")
    print(dict(Counter(r["dataset"] for r in slim)))


def load_cached(datasets=None):
    """Load records from cache, optionally filtered by dataset. Returns None if
    the cache is missing so callers can fall back to live loading."""
    if not CACHE.exists():
        return None
    with open(CACHE, "rb") as fh:
        recs = pickle.load(fh)
    if datasets is not None:
        want = set(datasets)
        recs = [r for r in recs if r["dataset"] in want]
    return recs


if __name__ == "__main__":
    build()
