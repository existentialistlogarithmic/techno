#!/usr/bin/env python3
"""Third-octave levels of a section against reference levels for a loud
club master. Relative tilt can look fine while the whole midrange is missing,
so check absolute levels too."""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

# rough dBFS RMS per third-octave for a loud techno/EDM master
REF = {31.5: -16, 40: -12, 50: -10, 63: -11, 80: -13, 100: -14, 125: -16, 160: -18,
       200: -19, 250: -20, 315: -21, 400: -21, 500: -22, 630: -23, 800: -23,
       1000: -24, 1250: -25, 1600: -25, 2000: -26, 2500: -27, 3150: -27,
       4000: -28, 5000: -29, 6300: -30, 8000: -32, 10000: -34, 12500: -36, 16000: -40}


def main(path, b0=104, b1=136, bpm=150.0):
    sr, data = wavfile.read(path)
    x = data.astype(np.float64) / 32768.0
    mono = x.mean(axis=1) if x.ndim > 1 else x
    bar = 60.0 / bpm * 4
    seg = mono[int(b0 * bar * sr):int(b1 * bar * sr)]
    f, P = welch(seg, sr, nperseg=16384)
    print(f"bars {b0}-{b1}   third-octave level vs. loud-master reference")
    print("   Hz     level     ref    diff")
    diffs = []
    for fc, ref in sorted(REF.items()):
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        e = P[(f >= lo) & (f < hi)].sum()
        lvl = 10 * np.log10(e + 1e-20)
        d = lvl - ref
        diffs.append(d)
        bar_txt = ("+" * min(18, int(d)) if d > 0 else "-" * min(18, int(-d)))
        print(f"{fc:7.0f}  {lvl:+7.1f} {ref:+7.1f} {d:+7.1f}  {bar_txt}")
    print(f"\nmean offset {np.mean(diffs):+.1f} dB   spread (std) {np.std(diffs):.1f} dB")
    print("spread under ~3 dB means the tonal balance tracks the reference curve")


if __name__ == "__main__":
    a = sys.argv[1]
    main(a, int(sys.argv[2]) if len(sys.argv) > 2 else 104,
         int(sys.argv[3]) if len(sys.argv) > 3 else 136)
