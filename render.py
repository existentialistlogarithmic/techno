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
from technogen.mixer import to_wav, to_mp3, balance
from technogen.track import TARGETS

from technogen.track import BUSES
BUS_ORDER = BUSES + ["rumble"]


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


# Buses grouped into the stems a person actually wants on a mixer channel.
STEM_GROUPS = [
    ("01_kick", ["kick", "kickfar"]),
    ("02_sub_and_rumble", ["sub", "rumble"]),
    ("03_drums", ["drums"]),
    ("04_industrial_metal", ["metal"]),
    ("05_bass_acid", ["bass"]),
    ("06_leads", ["lead"]),
    ("07_strings_and_voice", ["strings", "solo", "speech", "pad"]),
    ("08_texture_and_fx", ["texture", "fx", "air"]),
]


def save_stems(buses, outdir, as_mp3=True, bitrate=320):
    """Pre-master stems: balanced as in the mix, but without the master chain,
    so they sum back to the mix and you can treat them yourself."""
    os.makedirs(outdir, exist_ok=True)
    gains = balance(buses, TARGETS, reference="kick", verbose=False)
    total = None
    scaled = {}
    for nm, g in gains.items():
        scaled[nm] = buses[nm] * g
        total = scaled[nm] if total is None else total + scaled[nm]
    # one common gain for the whole set, so the balance between stems is kept
    head = 10 ** (-6.0 / 20) / max(1e-9, float(np.max(np.abs(total))))
    written = []
    for name, members in STEM_GROUPS:
        mix = None
        for nm in members:
            if nm in scaled:
                mix = scaled[nm] if mix is None else mix + scaled[nm]
        if mix is None:
            continue
        mix = mix * head
        path = os.path.join(outdir, name + (".mp3" if as_mp3 else ".wav"))
        if as_mp3:
            to_mp3(path, mix, bitrate=bitrate)
        else:
            to_wav(path, mix)
        written.append(path)
    return written


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

    if "--stems" in args:
        d = args[args.index("--stems") + 1]
        fmt_wav = "--stems-wav" in args
        print(f"[stems] writing to {d}/ as {'wav' if fmt_wav else 'mp3'}")
        for f in save_stems(buses, d, as_mp3=not fmt_wav):
            print("        ", f)

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
