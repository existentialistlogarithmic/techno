#!/usr/bin/env python3
"""Export the track's patterns as MIDI files.

    python3 export_midi.py [outdir]

Drum notes follow the General MIDI drum map where one exists, so the parts
land on sensible pads. midi_map.txt lists which note triggers which sample
from the sound pack.
"""

import os
import struct
import sys

from technogen import track as TR

PPQ = 480                      # ticks per quarter note
STEP = PPQ // 4                # one sixteenth


# ------------------------------------------------------------- midi writing

def _vlq(n):
    """Variable-length quantity, the format MIDI uses for delta times."""
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def write_midi(path, notes, bpm=150.0, name="pattern", ppq=PPQ):
    """notes: [(start_tick, duration_tick, midi_note, velocity), ...]"""
    ev = []
    for start, dur, note, vel in notes:
        ev.append((start, 0, 0x90, note, max(1, min(127, int(vel)))))
        ev.append((start + max(1, dur), 1, 0x80, note, 0))
    ev.sort(key=lambda e: (e[0], e[1]))

    body = bytearray()
    body += b"\x00\xFF\x03" + _vlq(len(name)) + name.encode()
    us = int(round(60_000_000 / bpm))
    body += b"\x00\xFF\x51\x03" + struct.pack(">I", us)[1:]
    body += b"\x00\xFF\x58\x04\x04\x02\x18\x08"          # 4/4

    prev = 0
    for start, _, status, note, vel in ev:
        body += _vlq(start - prev) + bytes([status, note, vel])
        prev = start
    body += b"\x00\xFF\x2F\x00"

    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq))
        f.write(b"MTrk" + struct.pack(">I", len(body)) + bytes(body))


# ------------------------------------------------------------- drum mapping

DRUM = {"kick": 36, "sub": 35, "kick_roll": 33, "rim": 37, "clap": 39,
        "hat_closed": 42, "hat_tip": 44, "hat_open": 46, "snare": 38,
        "tom_low": 41, "tom_high": 45, "ride": 51, "shaker": 70,
        "metal_plate_mid": 47, "metal_can_high": 48, "anvil_high": 49,
        "anvil_low": 50, "metal_plate_low": 52}

SAMPLE_FOR = {
    36: "01_drums/kick_main.wav", 35: "01_drums/kick_sub_layer.wav",
    33: "01_drums/kick_short_roll.wav", 37: "01_drums/rim.wav",
    39: "01_drums/clap.wav", 42: "01_drums/hat_closed.wav",
    44: "01_drums/hat_tip.wav", 46: "01_drums/hat_open.wav",
    38: "01_drums/snare.wav", 41: "01_drums/tom_low.wav",
    45: "01_drums/tom_high.wav", 51: "01_drums/ride.wav",
    70: "01_drums/shaker.wav", 47: "02_industrial/metal_plate_mid.wav",
    48: "02_industrial/metal_can_high.wav", 49: "02_industrial/anvil_high.wav",
    50: "02_industrial/anvil_low.wav", 52: "02_industrial/metal_plate_low.wav",
}


def drum_pattern(bars=8):
    """The main beat, exactly as the track plays it."""
    n = []
    for b in range(bars):
        base = b * 16 * STEP
        for st in (0, 4, 8, 12):
            n.append((base + st * STEP, STEP, DRUM["kick"], 118 if st == 0 else 114))
            n.append((base + st * STEP, STEP * 2, DRUM["sub"], 96))
        for st in range(16):
            vel = 104 if st % 4 == 2 else (78 if st % 4 == 0 else 66)
            n.append((base + st * STEP, STEP // 2, DRUM["hat_closed"], vel))
        for st in (2, 6, 10, 14):
            n.append((base + st * STEP, STEP, DRUM["hat_open"], 92))
        for st in (1, 7, 9, 15):
            n.append((base + st * STEP, STEP // 2, DRUM["hat_tip"], 44))
        for st in (3, 11, 15):
            n.append((base + st * STEP, STEP // 2, DRUM["rim"], 40))
        if b % 2 == 1:
            n.append((base + 8 * STEP, STEP, DRUM["clap"], 96))
        if b % 4 == 3:
            n.append((base + 14 * STEP, STEP, DRUM["kick_roll"], 70))
        if b % 8 == 7:
            n.append((base + 10 * STEP, STEP, DRUM["tom_low"], 78))
            n.append((base + 13 * STEP, STEP, DRUM["tom_high"], 72))
        if b == bars - 1:
            for i, st in enumerate((12, 13, 14, 15)):
                n.append((base + st * STEP, STEP, DRUM["kick_roll"], 64 + 14 * i))
    return n


def metal_pattern(bars=8):
    name = {"metal_a": "metal_plate_mid", "metal_b": "metal_can_high",
            "anvil": "anvil_high", "anvil_lo": "anvil_low",
            "metal_c": "metal_plate_low"}
    n = []
    for b in range(bars):
        base = b * 16 * STEP
        pat = TR.METAL_PAT2 if b % 2 else TR.METAL_PAT
        for st, key, g in pat:
            n.append((base + st * STEP, STEP, DRUM[name[key]], int(30 + 95 * g)))
    return n


def acid_to_midi(seq, bars=4):
    """The 303 line: (note, steps, accent, slide) -> midi."""
    n = []
    one = sum(s[1] for s in seq) * STEP
    for b in range(bars):
        pos = b * one
        for midi, steps, acc, slide in seq:
            dur = steps * STEP
            if midi is not None:
                n.append((pos, int(dur * (1.6 if slide else 0.85)), midi,
                          118 if acc else 86))
            pos += dur
    return n


def riff_to_midi(riff, bars=8, dur_steps=6, vel=100):
    return [(rb * 16 * STEP + rs * STEP, dur_steps * STEP, m, vel)
            for rb, rs, m in riff for _ in (0,)] * 1 if riff else []


def lead_riff_to_midi(riff):
    return [(rb * 16 * STEP + rs * STEP, int(ln * 16 * STEP / 2), m, 104)
            for rb, rs, m, ln in riff]


def melody_to_midi(notes, beat_unit=2.0, vel=100):
    """The lament, written in half-time."""
    n, pos = [], 0
    for midi, beats in notes:
        dur = int(beats * beat_unit * PPQ)
        if midi is not None:
            n.append((pos, int(dur * 1.05), int(round(midi)), vel))
        pos += dur
    return n


def chords_to_midi(seq, vel=80):
    n, pos = [], 0
    for voicing, beats in seq:
        dur = int(beats * PPQ)
        for m in voicing:
            n.append((pos, int(dur * 0.98), m, vel))
        pos += dur
    return n


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "midi"
    os.makedirs(out, exist_ok=True)
    w = lambda f, notes, nm: write_midi(os.path.join(out, f), notes, TR.BPM, nm)

    w("01_main_beat_8bars.mid", drum_pattern(8), "main beat")
    w("02_metal_pattern_8bars.mid", metal_pattern(8), "industrial metal")
    w("03_full_drums_8bars.mid", drum_pattern(8) + metal_pattern(8), "drums + metal")
    w("04_acid_A.mid", acid_to_midi(TR.ACID_A), "acid A")
    w("05_acid_B.mid", acid_to_midi(TR.ACID_B), "acid B")
    w("06_acid_C.mid", acid_to_midi(TR.ACID_C), "acid C")
    w("07_acid_slow.mid", acid_to_midi(TR.ACID_SEX), "acid slow")
    w("08_hoover_riff_1.mid", riff_to_midi(TR.HOOVER_RIFF), "hoover riff 1")
    w("09_hoover_riff_2.mid", riff_to_midi(TR.HOOVER_RIFF2), "hoover riff 2")
    w("10_screech_lead_riff.mid", lead_riff_to_midi(TR.LEAD_RIFF), "screech lead")
    w("11_violin_cry_phrase_1.mid", melody_to_midi(TR.CRY_A), "violin phrase 1")
    w("12_violin_cry_phrase_2.mid", melody_to_midi(TR.CRY_B), "violin phrase 2")
    w("13_string_chords.mid", chords_to_midi(TR.LAMENT_CHORDS), "string chords")
    w("14_cello_bass.mid", melody_to_midi(TR.LAMENT_BASS, beat_unit=1.0, vel=88),
      "cello bass")

    lines = ["CONCRETE CATHEDRAL - MIDI patterns",
             "=" * 34, "", "150 BPM, F# minor. Drag into any DAW.", "",
             "Drum note -> sample from the sound pack:", ""]
    for note in sorted(SAMPLE_FOR):
        lines.append(f"  {note:3d}   {SAMPLE_FOR[note]}")
    lines += ["", "Pitched parts (acid, hoover, screech, violin, strings) carry",
              "real pitches - point them at the matching sample or your own synth.",
              "Hoover and saw stab samples are named by note, so they map directly.",
              "", "01-03 are the beat. 04-07 are basslines. 08-10 are leads.",
              "11-14 are the lament.", ""]
    open(os.path.join(out, "midi_map.txt"), "w").write("\n".join(lines))
    print(f"wrote {len(os.listdir(out))} files to {out}/")


if __name__ == "__main__":
    main()
