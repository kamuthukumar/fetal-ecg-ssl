"""Per-record gestational age (GA) labels for NInFEA.

Source: Sulas et al. (2021) 'A non-invasive multimodal foetal ECG-Doppler
dataset for antenatal cardiology research', Scientific Data 8:30, Table 3
('Clinical information about mothers and foetuses'). GA given as weeks(+days);
converted here to decimal weeks (days/7). Records sharing a signal-group row
come from the SAME participant (subject id) and share the GA -> used to prevent
subject leakage in cross-validation.
"""

# (record_range, weeks, days, subject_id)
_GROUPS = [
    ((1, 3), 27, 0), ((4, 5), 25, 0), ((6, 7), 21, 1), ((8, 8), 22, 4),
    ((9, 9), 24, 0), ((10, 11), 24, 4), ((12, 12), 25, 4), ((13, 14), 21, 5),
    ((15, 16), 25, 2), ((17, 17), 24, 0), ((18, 18), 26, 6), ((19, 19), 22, 4),
    ((20, 21), 26, 2), ((22, 22), 25, 1), ((23, 23), 24, 1), ((24, 24), 26, 6),
    ((25, 25), 26, 3), ((26, 26), 24, 1), ((27, 27), 27, 1), ((28, 29), 26, 6),
    ((30, 30), 22, 3), ((31, 33), 27, 5), ((34, 34), 25, 0), ((35, 36), 27, 0),
    ((37, 38), 25, 1), ((39, 39), 24, 0), ((40, 41), 21, 3), ((42, 42), 24, 6),
    ((43, 44), 23, 4), ((45, 45), 23, 0), ((46, 46), 24, 4), ((47, 48), 21, 1),
    ((49, 50), 22, 0), ((51, 52), 24, 2), ((53, 54), 21, 0), ((55, 56), 25, 0),
    ((57, 57), 23, 6), ((58, 58), 23, 0), ((59, 60), 27, 3),
]

GA_WEEKS = {}     # record_id (int) -> GA in decimal weeks
SUBJECT = {}      # record_id (int) -> subject id (int)

for subj, ((lo, hi), wk, dd) in enumerate(_GROUPS):
    ga = wk + dd / 7.0
    for r in range(lo, hi + 1):
        GA_WEEKS[r] = ga
        SUBJECT[r] = subj

if __name__ == "__main__":
    import numpy as np
    vals = np.array(list(GA_WEEKS.values()))
    print(f"{len(GA_WEEKS)} records, {len(set(SUBJECT.values()))} subjects")
    print(f"GA range {vals.min():.2f}-{vals.max():.2f} wk, mean {vals.mean():.2f}, std {vals.std():.2f}")
    for r in sorted(GA_WEEKS):
        print(f"  rec {r:2d}: GA={GA_WEEKS[r]:.2f} wk  subj={SUBJECT[r]}")
