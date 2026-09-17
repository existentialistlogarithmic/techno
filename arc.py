#!/usr/bin/env python3
"""Does the track actually go somewhere? Level and brightness, bar by bar."""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

MARKS = {0: "intro", 16: "build 1", 32: "pre-drop", 40: "DROP 1", 72: "breakdown",
         88: "build 2", 104: "DROP 2", 136: "outro"}


def main(path, bpm=150.0, group=4):
    sr, data = wavfile.read(path)
    x = data.astype(np.float64) / 32768.0
    mono = x.mean(axis=1) if x.ndim > 1 else x
    bar = int(60.0 / bpm * 4 * sr)
    nbars = len(mono) // bar
    print(f"bars  level(dBFS)  centroid(Hz)  sub  low  mid  high")
    for b0 in range(0, nbars, group):
        seg = mono[b0 * bar:min(len(mono), (b0 + group) * bar)]
        if len(seg) < sr:
            continue
        r = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        f, P = welch(seg, sr, nperseg=8192)
        cen = float(np.sum(f * P) / np.sum(P))
        tot = P.sum()
        b = [P[(f < 60)].sum() / tot, P[(f >= 60) & (f < 300)].sum() / tot,
             P[(f >= 300) & (f < 3000)].sum() / tot, P[f >= 3000].sum() / tot]
        bars_txt = "".join("#" * max(0, int(v * 20)) + "." * (5 - min(5, int(v * 20))) + "|" for v in b)
        tag = ""
        for m, name in MARKS.items():
            if b0 <= m < b0 + group:
                tag = "  <<< " + name
        meter = "#" * max(0, int((r + 30) * 1.3))
        print(f"{b0:4d}  {r:+7.2f} {meter:<30s} {cen:7.0f}  {bars_txt}{tag}")


if __name__ == "__main__":
    main(sys.argv[1])
