#!/usr/bin/env python3
"""Render the track to a WAV file.

usage:
    python3 render.py [output.wav]
    python3 render.py out.wav --cache-buses DIR   # save buses for fast re-mixing
    python3 render.py out.wav --from-cache DIR    # re-balance/master only
    python3 render.py out.wav --mp3              # also write an mp3
"""

import os
import sys
import time

import numpy as np

from technogen.track import build, process_buses, finalize
from technogen.mixer import to_wav, to_mp3

BUS_ORDER = ["kick", "kickfar", "sub", "rumble", "drums", "bass", "lead",
             "voice", "fx", "pad", "air"]


def save_buses(path, buses):
    os.makedirs(path, exist_ok=True)
    for nm, b in buses.items():
        np.save(os.path.join(path, nm + ".npy"), b.astype(np.float32))


def load_buses(path):
    out = {}
    for nm in BUS_ORDER:
        f = os.path.join(path, nm + ".npy")
        if os.path.exists(f):
            out[nm] = np.load(f).astype(np.float64)
    return out


def main():
    args = [a for a in sys.argv[1:]]
    out = args[0] if args and not args[0].startswith("-") else "concrete_cathedral.wav"
    cache = from_cache = None
    if "--cache-buses" in args:
        cache = args[args.index("--cache-buses") + 1]
    if "--from-cache" in args:
        from_cache = args[args.index("--from-cache") + 1]

    t0 = time.time()
    if from_cache:
        print("[1/3] loading cached buses from", from_cache)
        buses = load_buses(from_cache)
    else:
        print("[1/3] arranging")
        s, p = build()
        print(f"[2/3] processing ({s.dur:.1f}s = {int(s.dur // 60)}:{int(s.dur % 60):02d}, "
              f"{len(s.kick_times)} kicks)")
        buses = process_buses(s, p)
        if cache:
            save_buses(cache, buses)
            print("      buses cached to", cache)

    print("[3/3] mixdown ->", out)
    mix = finalize(buses)
    to_wav(out, mix)
    if "--mp3" in args:
        mp3 = out.rsplit(".", 1)[0] + ".mp3"
        if to_mp3(mp3, mix):
            print("      also wrote", mp3)
    peak = float(np.max(np.abs(mix)))
    rms = float(np.sqrt(np.mean(mix ** 2)))
    print(f"done in {time.time() - t0:.1f}s | peak {20 * np.log10(peak):.2f} dBFS | "
          f"rms {20 * np.log10(rms):.2f} dBFS")


if __name__ == "__main__":
    main()
