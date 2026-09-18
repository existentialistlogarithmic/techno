#!/usr/bin/env python3
"""Export the whole arrangement as one multitrack MIDI file.

    python3 export_arrangement.py [out.mid]

Drag the result into Ableton: you get one named MIDI track per part, with
every hit laid out across all 224 bars. Drop the matching sample into each
track's Simpler or Drum Rack (arrangement_map.txt says which note is which
file) and the whole track is in front of you, movable.
"""

import sys
from collections import defaultdict

from technogen import track as TR
from technogen.mixer import Session
from export_sounds import LAYOUT, NOTES
from export_midi import write_midi, PPQ, STEP, acid_to_midi, _vlq
import struct

# which palette keys land on which Ableton track, in order
TRACKS = [
    ("1 KICK",        lambda k: k.startswith("kick") or k == "sub"),
    ("2 HATS PERC",   lambda k: k.startswith(("hat", "ohat")) or k in
                      ("clap", "clap_tight", "snare", "snare_t", "rim", "ride",
                       "tom_h", "tom_l", "shaker")),
    ("3 METAL",       lambda k: k.startswith(("metal_", "anvil", "clang", "chain",
                                              "steam", "scrape", "hammer"))),
    ("4 LOOPS",       lambda k: k.startswith(("conveyor", "machine", "rust")) or
                      k == "groove_bar"),
    ("5 TEXTURE",     lambda k: k.startswith(("drone", "bed"))),
    ("6 BASS ACID",   lambda k: False),          # filled from the acid patterns
    ("7 LEAD",        lambda k: k.startswith(("hoov", "scr_", "horn", "stab"))),
    ("8 STRINGS",     lambda k: k.startswith(("violin", "strings", "cello", "pad_"))),
    ("9 VOICE",       lambda k: k.startswith(("moan_", "breath_"))),
    ("10 SPEECH",     lambda k: k.startswith(("say_", "forgive_", "sins_"))),
    ("11 FX BOOMS",   lambda k: k in ("boom", "boom2", "impact", "rev_swell",
                                      "sub_drop", "noise_fall", "feedback",
                                      "feedback_hi") or k.startswith("riser")),
]


def filename_for(key):
    if key in LAYOUT:
        f, n = LAYOUT[key]
        return f"{f}/{n}.wav"
    for pre, folder, stem in (("hoovS", "04_synths", "hoover_stab"),
                              ("hoov", "04_synths", "hoover_long"),
                              ("stab", "04_synths", "saw_stab")):
        if key.startswith(pre) and key[len(pre):].isdigit():
            m = int(key[len(pre):])
            if m in NOTES:
                return f"{folder}/{stem}_{NOTES[m]}.wav"
    return None


def write_multitrack(path, tracks, bpm=150.0, ppq=PPQ):
    """tracks: [(name, [(start, dur, note, vel), ...]), ...] -> type 1 MIDI."""
    chunks = []
    # track 0 carries tempo and time signature
    t0 = bytearray(b"\x00\xFF\x03" + _vlq(len("tempo")) + b"tempo")
    t0 += b"\x00\xFF\x51\x03" + struct.pack(">I", int(round(60_000_000 / bpm)))[1:]
    t0 += b"\x00\xFF\x58\x04\x04\x02\x18\x08"
    t0 += b"\x00\xFF\x2F\x00"
    chunks.append(bytes(t0))

    for name, notes in tracks:
        ev = []
        for start, dur, note, vel in notes:
            ev.append((int(start), 0, 0x90, int(note), max(1, min(127, int(vel)))))
            ev.append((int(start) + max(1, int(dur)), 1, 0x80, int(note), 0))
        ev.sort(key=lambda e: (e[0], e[1]))
        b = bytearray(b"\x00\xFF\x03" + _vlq(len(name)) + name.encode())
        prev = 0
        for start, _, status, note, vel in ev:
            b += _vlq(start - prev) + bytes([status, note, vel])
            prev = start
        b += b"\x00\xFF\x2F\x00"
        chunks.append(bytes(b))

    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), ppq))
        for c in chunks:
            f.write(b"MTrk" + struct.pack(">I", len(c)) + c)


def _record():
    """Run build() with place() and lay_acid() instrumented, and return the
    notes grouped per Ableton track."""
    p = TR.build_palette()
    by_id = {id(v): k for k, v in p.items()}
    real_palette = TR.build_palette
    TR.build_palette = lambda *a, **k: p

    events, acid_spans = [], []
    orig_place, orig_acid = Session.place, TR.lay_acid

    def spy(self, name, x, bar, step=0.0, gain=1.0, pan_=0.0, reverse=False):
        events.append((float(bar), float(step), by_id.get(id(x)), float(gain),
                       len(x) if hasattr(x, "__len__") else 0))
        return orig_place(self, name, x, bar, step, gain, pan_, reverse)

    def spy_acid(s, pp, b0, b1, pattern, gain=0.5, **kw):
        for nm in ("ACID_A", "ACID_B", "ACID_C", "ACID_SEX"):
            if pattern is getattr(TR, nm):
                acid_spans.append((b0, b1, nm))
                break
        return orig_acid(s, pp, b0, b1, pattern, gain, **kw)

    Session.place, TR.lay_acid = spy, spy_acid
    try:
        TR.build(verbose=False)
    finally:
        Session.place, TR.lay_acid = orig_place, orig_acid
        TR.build_palette = real_palette

    note_of, used, track_of = {}, defaultdict(int), {}
    for key in sorted({e[2] for e in events if e[2]}):
        if filename_for(key) is None:
            continue
        for tname, test in TRACKS:
            if test(key):
                track_of[key] = tname
                note_of[key] = 36 + used[tname]
                used[tname] += 1
                break

    track_notes = defaultdict(list)
    for bar, step, key, gain, nsamp in events:
        if key not in note_of:
            continue
        start = int(round(bar * 16 * STEP + step * STEP))
        dur = max(STEP // 2, int(nsamp / 44100.0 * (PPQ * 150 / 60.0)))
        dur = min(dur, 8 * 16 * STEP)
        vel = max(20, min(127, int(30 + 97 * min(1.0, gain))))
        track_notes[track_of[key]].append((start, dur, note_of[key], vel))

    for b0, b1, nm in acid_spans:
        pat = getattr(TR, nm)
        for b in range(b0, b1):
            base, pos = b * 16 * STEP, 0
            for midi, steps, acc, slide in pat:
                d = steps * STEP
                if midi is not None:
                    track_notes["6 BASS ACID"].append(
                        (base + pos, int(d * (1.6 if slide else 0.85)), midi,
                         118 if acc else 86))
                pos += d
    return track_notes, note_of, track_of


def build_events():
    """[(track name, [(start_beat, dur_beat, key, vel), ...]), ...] + bars."""
    notes, _, _ = _record()
    out = []
    for tname, _t in TRACKS:
        out.append((tname, [(s / PPQ, d / PPQ, k, v)
                            for s, d, k, v in sorted(notes[tname])]))
    last = max((s + d for _, rows in out for s, d, _, _ in rows), default=0)
    return out, int(last / 4) + 8


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "CONCRETE_CATHEDRAL_arrangement.mid"
    print("reading the arrangement...")
    track_notes, note_of, track_of = _record()

    tracks = [(n, sorted(track_notes[n])) for n, _ in TRACKS if track_notes[n]]
    write_multitrack(out, tracks, TR.BPM)
    total = sum(len(v) for _, v in tracks)
    print(f"wrote {out}: {len(tracks)} tracks, {total} notes")

    lines = ["CONCRETE CATHEDRAL - arrangement MIDI",
             "=" * 46, "",
             "One MIDI track per part, 150 BPM.",
             "Drop the listed sample on the listed note in a Drum Rack",
             "(or Simpler set to the right note) and the part plays itself.",
             ""]
    for tname, _ in TRACKS:
        keys = [k for k in note_of if track_of.get(k) == tname]
        if not keys:
            if tname == "6 BASS ACID":
                lines += [f"[{tname}]", "  real pitches - point at a bass synth,",
                          "  or the acid_*_1bar_150bpm.wav loops", ""]
            continue
        lines.append(f"[{tname}]")
        for k in sorted(keys, key=lambda k: note_of[k]):
            lines.append(f"  note {note_of[k]:3d}  {filename_for(k)}")
        lines.append("")
    open(out.replace(".mid", "_map.txt"), "w").write("\n".join(lines) + "\n")
    print("wrote", out.replace(".mid", "_map.txt"))


if __name__ == "__main__":
    main()
