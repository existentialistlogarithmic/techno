#!/usr/bin/env python3
"""Generate a placement guide: which file from the sound pack goes where.

    python3 export_placement.py [out.txt]

Reads the real arrangement by recording every place() call, so the guide
cannot drift from the track.
"""

import sys
from collections import defaultdict

from technogen import track as TR
from technogen.mixer import Session
from export_sounds import LAYOUT, NOTES

SECTIONS = [(0, 15, "LAMENT"), (15, 16, "BOOM BOOM"), (16, 32, "THE MACHINE"),
            (32, 48, "GROOVE"), (48, 64, "BUILD 1"), (64, 72, "TENSION"),
            (72, 104, "DROP 1"), (104, 112, "INTERLUDE"), (112, 128, "BUILD 2"),
            (128, 160, "DROP 2"), (160, 176, "BREAKDOWN"), (176, 184, "BUILD 3"),
            (184, 216, "DROP 3"), (216, 224, "OUTRO")]


def filename_for(key):
    if key in LAYOUT:
        folder, name = LAYOUT[key]
        return f"{folder}/{name}.wav"
    for pre, folder, stem in (("hoovS", "04_synths", "hoover_stab"),
                              ("hoov", "04_synths", "hoover_long"),
                              ("stab", "04_synths", "saw_stab")):
        if key.startswith(pre) and key[len(pre):].isdigit():
            m = int(key[len(pre):])
            if m in NOTES:
                return f"{folder}/{stem}_{NOTES[m]}.wav"
    return None


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "PLACEMENT_GUIDE.txt"
    print("reading the arrangement...")
    p = TR.build_palette()
    by_id = {id(v): k for k, v in p.items()}
    # build() makes its own palette; pin ours so the recorded arrays are the
    # same objects we can look names up from
    real_palette = TR.build_palette
    TR.build_palette = lambda *a, **k: p

    events = []
    original = Session.place

    def spy(self, name, x, bar, step=0.0, gain=1.0, pan_=0.0, reverse=False):
        key = by_id.get(id(x))
        events.append((float(bar), float(step), name, key, float(gain)))
        return original(self, name, x, bar, step, gain, pan_, reverse)

    Session.place = spy
    try:
        TR.build(verbose=False)
    finally:
        Session.place = original
        TR.build_palette = real_palette

    # key -> {bar: set(steps)}
    plays = defaultdict(lambda: defaultdict(set))
    derived = defaultdict(list)
    for bar, step, bus, key, gain in events:
        if key is None:
            derived[int(bar)].append((bus, gain))
        else:
            plays[key][int(bar)].add(round(step, 1))

    lines = []
    A = lines.append
    A("CONCRETE CATHEDRAL - what to place, and when")
    A("=" * 52)
    A("")
    A("150 BPM, F# minor.  Bars are Ableton bar numbers (bar 1 = the start),")
    A("so a section listed as bars 16-31 starts on Ableton's bar 17.")
    A("Steps are sixteenths within the bar: 1 5 9 13 are the four beats.")
    A("")

    for b0, b1, title in SECTIONS:
        A("")
        A("-" * 52)
        t = b0 * 1.6
        A(f"{title}   bars {b0}-{b1 - 1}   (Ableton bar {b0 + 1}, {int(t // 60)}:{t % 60:04.1f})")
        A("-" * 52)
        rows = []
        for key, bars in plays.items():
            hit = sorted(b for b in bars if b0 <= b < b1)
            if not hit:
                continue
            fn = filename_for(key)
            if fn is None:
                continue
            steps = sorted({s for b in hit for s in bars[b]})
            span = len(hit)
            if span >= max(4, (b1 - b0) // 2):
                st = " ".join(str(int(s) + 1) for s in steps[:16])
                rows.append((0, f"  {fn:46s} every bar, steps {st}"))
            elif span > 1:
                rows.append((1, f"  {fn:46s} bars {hit[0]}-{hit[-1]} ({span}x)"))
            else:
                b = hit[0]
                st = int(sorted(bars[b])[0]) + 1
                rows.append((2, f"  {fn:46s} bar {b}, step {st}"))
        if not rows:
            A("  (silence)")
        for _, r in sorted(rows):
            A(r)
    A("")
    A("")
    A("=" * 52)
    A("NOTES")
    A("=" * 52)
    A("- 'every bar' entries are the loop. Build one bar, then duplicate.")
    A("- single-bar entries are one-shots: place them by hand, once.")
    A("- Warp OFF for one-shots. Warp ON (Beats) only for *_150bpm files.")
    A("- A few parts are processed versions of other samples (the distorted")
    A("  violin, the corrupted strings, the tape-braked groove bar). Those")
    A("  ship as their own files: violin_distorted_lead, strings_corrupted_*,")
    A("  groove_1bar_150bpm.")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
