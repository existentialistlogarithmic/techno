#!/usr/bin/env python3
"""Per-bus audibility: how far each bus sits above everything else, measured
in the band that bus lives in and only while it is actually playing.

A bus that is less than a few dB above the rest of the mix in its own band is
buried, whatever its fader says."""
import sys
import numpy as np
from scipy.signal import welch

sys.path.insert(0, ".")
from technogen.track import TARGETS
from technogen.mixer import balance, BANDS
from technogen.dsp import SR
from render import load_buses

BAND_HZ = {"low": (25, 150), "mid": (250, 9000), "full": (25, 16000)}
FOCUS = {"drums": (5000, 16000), "lead": (900, 6000), "voice": (300, 3000),
         "bass": (100, 900), "fx": (400, 9000), "pad": (250, 3000),
         "kick": (35, 110), "rumble": (35, 150), "sub": (30, 100),
         "kickfar": (30, 400), "air": (60, 900)}


def band_blocks(x, lo, hi, sr=SR, win=0.4):
    """Per-block band energy, so sparse parts are judged when they play."""
    from scipy.signal import butter, sosfiltfilt
    mono = x.mean(axis=0) if x.ndim > 1 else x
    sos = butter(4, [max(20, lo) / (sr / 2), min(hi, sr / 2 - 100) / (sr / 2)],
                 btype="band", output="sos")
    y = sosfiltfilt(sos, mono)
    w = max(1, int(win * sr))
    nb = len(y) // w
    return np.mean(y[:nb * w].reshape(nb, w) ** 2, axis=1) + 1e-20


def main(path, b0=40, b1=72, bpm=150.0):
    buses = load_buses(path)
    gains = balance(buses, TARGETS, reference="kick", verbose=False)
    bar = 60.0 / bpm * 4
    i0, i1 = int(b0 * bar * SR), int(b1 * bar * SR)
    scaled = {nm: buses[nm][:, i0:i1] * g for nm, g in gains.items()}
    total = sum(scaled.values())
    print(f"bars {b0}-{b1}:  bus level vs. the rest of the mix, in its own band")
    print("  bus          band(Hz)      solo    rest   delta  duty")
    rows = []
    for nm, x in scaled.items():
        lo, hi = FOCUS.get(nm, BAND_HZ[TARGETS[nm][1]])
        bs = band_blocks(x, lo, hi)
        br = band_blocks(total - x, lo, hi)
        if bs.max() < 1e-16:
            continue                                  # bus silent in this section
        active = bs >= np.quantile(bs, 0.70)          # only where it plays
        e_solo, e_rest = bs[active].mean(), br[active].mean()
        duty = float(np.mean(bs > bs.max() * 0.02))
        rows.append((10 * np.log10(e_solo / e_rest), nm, lo, hi, e_solo, e_rest, duty))
    for d, nm, lo, hi, es, er, duty in sorted(rows, reverse=True):
        flag = "  <-- buried" if d < -8 else ("  <-- dominant" if d > 8 else "")
        print(f"  {nm:<10s} {lo:5d}-{hi:<6d} {10*np.log10(es):+7.1f} {10*np.log10(er):+7.1f} "
              f"{d:+7.1f}   {duty*100:3.0f}%{flag}")


if __name__ == "__main__":
    p = sys.argv[1]
    a = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    b = int(sys.argv[3]) if len(sys.argv) > 3 else 72
    main(p, a, b)
