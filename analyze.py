#!/usr/bin/env python3
"""Inspect a rendered mix: section levels, spectral balance, dynamics."""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

SECTIONS = [("intro", 0, 16), ("build1", 16, 32), ("pre", 32, 40), ("DROP1", 40, 72),
            ("break", 72, 88), ("build2", 88, 104), ("DROP2", 104, 136), ("outro", 136, 152)]


def main(path, bpm=150.0):
    sr, data = wavfile.read(path)
    x = data.astype(np.float64) / 32768.0
    if x.ndim == 1:
        x = x[:, None]
    mono = x.mean(axis=1)
    bar = 60.0 / bpm * 4
    print(f"{path}  {len(mono)/sr:.1f}s  {sr}Hz  {x.shape[1]}ch")
    print(f"peak {20*np.log10(np.max(np.abs(x))+1e-9):+.2f} dBFS   "
          f"rms {20*np.log10(np.sqrt(np.mean(mono**2))+1e-9):+.2f} dBFS   "
          f"crest {20*np.log10(np.max(np.abs(mono))/np.sqrt(np.mean(mono**2))):.1f} dB")

    print("\nsection      bars      rms      peak    crest   low%   mid%   high%  corr")
    for name, b0, b1 in SECTIONS:
        i0, i1 = int(b0*bar*sr), min(len(mono), int(b1*bar*sr))
        seg = mono[i0:i1]
        if len(seg) < sr:
            continue
        r = np.sqrt(np.mean(seg**2)); pk = np.max(np.abs(seg))
        f, P = welch(seg, sr, nperseg=8192)
        tot = P.sum()
        lo = P[(f < 150)].sum()/tot; mid = P[(f >= 150) & (f < 2500)].sum()/tot
        hi = P[f >= 2500].sum()/tot
        corr = np.corrcoef(x[i0:i1, 0], x[i0:i1, 1])[0, 1] if x.shape[1] > 1 else 1.0
        print(f"{name:10s} {b0:4d}-{b1:<4d} {20*np.log10(r):+6.2f}  {20*np.log10(pk):+6.2f}  "
              f"{20*np.log10(pk/r):5.1f}  {lo*100:5.1f}  {mid*100:5.1f}  {hi*100:5.1f}  {corr:+.2f}")

    # spectral tilt of the whole thing
    f, P = welch(mono, sr, nperseg=16384)
    print("\noctave band energy (dB rel. peak band):")
    edges = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20000]
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        bands.append(P[(f >= a) & (f < b)].sum())
    mx = max(bands)
    for (a, b), v in zip(zip(edges[:-1], edges[1:]), bands):
        d = 10*np.log10(v/mx+1e-12)      # welch returns power: 10*log10, not 20
        print(f"  {a:5d}-{b:<5d} {d:+6.1f}  " + "#" * max(0, int(40 + d*1.3)))

    # how squashed: short-term crest over 400ms windows
    w = int(0.4*sr)
    cr = []
    for i in range(0, len(mono)-w, w):
        seg = mono[i:i+w]
        r = np.sqrt(np.mean(seg**2))
        if r > 1e-4:
            cr.append(20*np.log10(np.max(np.abs(seg))/r))
    print(f"\nshort-term crest: min {min(cr):.1f}  median {np.median(cr):.1f}  max {max(cr):.1f} dB")
    clipped = int(np.sum(np.abs(x) >= 0.999))
    print(f"samples at full scale: {clipped}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "concrete_cathedral.wav")
