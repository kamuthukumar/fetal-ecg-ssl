"""Download the three open-access PhysioNet datasets used in this study.

Uses direct HTTPS file downloads (robust across wfdb versions).

  ADfECGDB   - 5 EDF records, 4 abdominal + 1 direct scalp fetal ECG, 38-41 wk,
               cardiologist-verified fetal beat annotations (in the EDF+ stream).
  NInFEA     - 60 WFDB records, 24 abdominal + 3 thoracic channels, 2048 Hz,
               21-27 wk. Per-record gestational age is taken from the companion
               paper (see src/ninfea_labels.py).
  CinC 2013  - 75 set-a records, 4 abdominal channels, 1 kHz, expert .fqrs
               fetal QRS reference.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SPECS = {
    "adfecgdb": dict(
        url=f"{BASE}/adfecgdb/1.0.0",
        records=[f"r{n:02d}" for n in (1, 4, 7, 8, 10)],
        exts=["edf"],
    ),
    "cinc2013": dict(
        url=f"{BASE}/challenge-2013/1.0.0/set-a",
        records=[f"a{n:02d}" for n in range(1, 76)],
        exts=["dat", "hea", "fqrs"],
    ),
    "ninfea": dict(
        url=f"{BASE}/ninfea/1.0.0/wfdb_format_ecg_and_respiration",
        records=[str(n) for n in range(1, 61)],
        exts=["dat", "hea"],
    ),
    # Non-Invasive Fetal ECG Database: 55 EDF recordings from a single subject
    # across 21-40 wk (longitudinal). Record names read from the RECORDS file.
    "nifecgdb": dict(
        url=f"{BASE}/nifecgdb/1.0.0",
        records="@RECORDS",
        exts=[""],  # names in RECORDS already include the .edf suffix
    ),
    # Non-Invasive Fetal ECG Arrhythmia Database: 26 WFDB records (12 arrhythmic).
    "nifeadb": dict(
        url=f"{BASE}/nifeadb/1.0.0",
        records="@RECORDS",
        exts=["dat", "hea"],
    ),
}


def _record_list(spec):
    """Resolve a spec's record list, fetching the RECORDS file if requested."""
    recs = spec["records"]
    if recs == "@RECORDS":
        import urllib.request
        with urllib.request.urlopen(f"{spec['url']}/RECORDS", timeout=60) as fh:
            return [ln.strip() for ln in fh.read().decode().splitlines() if ln.strip()]
    return recs


def _fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:  # noqa: BLE001
        print(f"  WARN could not fetch {url}: {e}", file=sys.stderr)


def download(name):
    spec = SPECS[name]
    dest = DATA_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    records = _record_list(spec)
    print(f"[{name}] {len(records)} records -> {dest}")
    for r in records:
        for ext in spec["exts"]:
            fname = r if ext == "" else f"{r}.{ext}"
            _fetch(f"{spec['url']}/{fname}", dest / Path(fname).name)
    print(f"[{name}] done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*", default=list(SPECS))
    args = p.parse_args()
    for t in (args.datasets or list(SPECS)):
        if t not in SPECS:
            print(f"unknown dataset {t}; choices={list(SPECS)}", file=sys.stderr)
            continue
        download(t)
