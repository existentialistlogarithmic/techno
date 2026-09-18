#!/usr/bin/env python3
"""Export every sound used in the track as an isolated WAV.

    python3 export_sounds.py [outdir]

Each file is peak-normalised to -1 dBFS, 44.1 kHz. Tempo-locked loops carry
'_150bpm' in the name; everything else is a one-shot.
"""

import os
import shutil
import sys

import numpy as np

from technogen import instruments as I
from technogen import strings as ST
from technogen import track as TR
from technogen.dsp import SR, normalize
from technogen.mixer import to_wav

# palette key -> (folder, filename). Anything not listed is skipped.
LAYOUT = {
    # ---- drums
    "kick": ("01_drums", "kick_main"),
    "kick_hard": ("01_drums", "kick_hard"),
    "kick_max": ("01_drums", "kick_hardest"),
    "kick_soft": ("01_drums", "kick_soft_distant"),
    "kick_short": ("01_drums", "kick_short_roll"),
    "sub": ("01_drums", "kick_sub_layer"),
    "hat": ("01_drums", "hat_closed"),
    "hat2": ("01_drums", "hat_closed_bright"),
    "hat_tip": ("01_drums", "hat_tip"),
    "ohat": ("01_drums", "hat_open"),
    "ohat_long": ("01_drums", "hat_open_long"),
    "clap": ("01_drums", "clap"),
    "clap_tight": ("01_drums", "clap_tight"),
    "snare": ("01_drums", "snare"),
    "snare_t": ("01_drums", "snare_tight"),
    "rim": ("01_drums", "rim"),
    "ride": ("01_drums", "ride"),
    "tom_h": ("01_drums", "tom_high"),
    "tom_l": ("01_drums", "tom_low"),
    "shaker": ("01_drums", "shaker"),
    # ---- industrial
    "metal_a": ("02_industrial", "metal_plate_mid"),
    "metal_b": ("02_industrial", "metal_can_high"),
    "metal_c": ("02_industrial", "metal_plate_low"),
    "anvil": ("02_industrial", "anvil_high"),
    "anvil_lo": ("02_industrial", "anvil_low"),
    "clang": ("02_industrial", "clang_big"),
    "clang_hi": ("02_industrial", "clang_high"),
    "chain": ("02_industrial", "chain_long"),
    "chain_s": ("02_industrial", "chain_short"),
    "steam": ("02_industrial", "steam_long"),
    "steam_s": ("02_industrial", "steam_short"),
    "scrape": ("02_industrial", "metal_scrape"),
    "scrape_l": ("02_industrial", "metal_scrape_long"),
    "hammer": ("02_industrial", "hammer"),
    "hammer_lo": ("02_industrial", "hammer_low"),
    "boom": ("02_industrial", "BOOM_1"),
    "boom2": ("02_industrial", "BOOM_2_bigger"),
    "feedback": ("02_industrial", "feedback_tone"),
    "feedback_hi": ("02_industrial", "feedback_tone_high"),
    "machine": ("02_industrial", "machine_loop_150bpm"),
    "machine_lo": ("02_industrial", "machine_loop_low_150bpm"),
    "conveyor": ("02_industrial", "conveyor_loop_150bpm"),
    "conveyor2": ("02_industrial", "conveyor_loop_high_150bpm"),
    # ---- textures
    "drone_lo": ("03_textures", "drone_low"),
    "drone_mid": ("03_textures", "drone_mid"),
    "bed_a": ("03_textures", "noise_bed_dark"),
    "bed_b": ("03_textures", "noise_bed_bright"),
    "rust_a": ("03_textures", "metal_debris_sparse_150bpm"),
    "rust_b": ("03_textures", "metal_debris_dense_150bpm"),
    # ---- synths
    "scr_up": ("04_synths", "screech_up"),
    "scr_dn": ("04_synths", "screech_down"),
    "scr_ud": ("04_synths", "screech_updown"),
    "scr_st": ("04_synths", "screech_stab_up"),
    "scr_st2": ("04_synths", "screech_stab_down"),
    "scr_long": ("04_synths", "screech_long"),
    "scr_evil": ("04_synths", "screech_evil"),
    "horn": ("04_synths", "war_horn_F#1"),
    "horn_hi": ("04_synths", "war_horn_C#2"),
    "pad_a": ("04_synths", "pad_a"),
    "pad_b": ("04_synths", "pad_b"),
    "pad_sex": ("04_synths", "pad_bright"),
    "pad_dark": ("04_synths", "pad_dark"),
    # ---- strings
    "violin_a": ("05_strings", "violin_cry_phrase_1"),
    "violin_b": ("05_strings", "violin_cry_phrase_2_climax"),
    "violin_b_low": ("05_strings", "violin_cry_phrase_2_octave_down"),
    "violin_a2": ("05_strings", "violin_cry_phrase_1_breakdown"),
    "violin_dead": ("05_strings", "violin_distorted_lead"),
    "strings_a": ("05_strings", "string_section_chords_1"),
    "strings_b": ("05_strings", "string_section_chords_2"),
    "strings_dead": ("05_strings", "string_section_distorted"),
    "cello": ("05_strings", "cello_lament_bass"),
    "strings_rot0": ("05_strings", "strings_corrupted_25pct"),
    "strings_rot1": ("05_strings", "strings_corrupted_50pct"),
    "strings_rot2": ("05_strings", "strings_corrupted_75pct"),
    "strings_rot3": ("05_strings", "strings_corrupted_100pct"),
    # ---- fx
    "impact": ("06_fx", "impact"),
    "rev_swell": ("06_fx", "reverse_swell"),
    "sub_drop": ("06_fx", "sub_drop"),
    "noise_fall": ("06_fx", "noise_fall"),
    "riser_n": ("06_fx", "noise_riser_4bar"),
    "riser_n2": ("06_fx", "noise_riser_8bar"),
    "riser_s": ("06_fx", "noise_riser_2bar"),
    # ---- speech
    "say_close": ("07_speech", "confession_close"),
    "say_whisper": ("07_speech", "confession_whispered"),
    "say_choir": ("07_speech", "confession_vocoded_choir"),
    "say_radio": ("07_speech", "confession_radio"),
    "forgive_close": ("07_speech", "forgive_me_close"),
    "forgive_whisper": ("07_speech", "forgive_me_whispered"),
    "sins_close": ("07_speech", "for_all_my_sins_close"),
    "sins_choir": ("07_speech", "for_all_my_sins_choir"),
    # ---- loops
    "groove_bar": ("08_loops", "groove_1bar_150bpm"),
}

# midi -> note name, for the pitched one-shots
NOTES = {42: "F#2", 45: "A2", 47: "B2", 49: "C#3", 52: "E3", 54: "F#3",
         57: "A3", 59: "B3", 61: "C#4"}


# these are 16-bar beds; 8 bars is a usable loop and half the file
TRIM_BARS = {"machine": 8, "machine_lo": 8, "bed_a": 8, "bed_b": 8,
             "drone_lo": 8, "drone_mid": 8}
BAR = 60.0 / 150.0 * 4


def write(path, x, peak_db=-1.0, trim_bars=None):
    """Mono, 16-bit, peak-normalised. The sources are mono - writing them as
    dual-mono would double the pack for nothing."""
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=0)
    if trim_bars:
        x = x[:int(trim_bars * BAR * SR)]
    m = np.max(np.abs(x))
    if m < 1e-9:
        return False
    x = x * (10 ** (peak_db / 20.0) / m)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from scipy.io import wavfile
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return True


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "sounds"
    if os.path.exists(out):
        shutil.rmtree(out)
    print("building the palette...")
    p = TR.build_palette()

    n = 0
    for key, (folder, name) in LAYOUT.items():
        if key not in p:
            print("  missing:", key)
            continue
        if write(os.path.join(out, folder, name + ".wav"), p[key],
                 trim_bars=TRIM_BARS.get(key)):
            n += 1

    # pitched one-shots, named by note
    for midi, note in NOTES.items():
        for src, folder, stem in (("hoov", "04_synths", "hoover_long"),
                                  ("hoovS", "04_synths", "hoover_stab"),
                                  ("stab", "04_synths", "saw_stab")):
            k = f"{src}{midi}"
            if k in p and write(os.path.join(out, folder, f"{stem}_{note}.wav"), p[k]):
                n += 1

    # the acid lines, rendered as one-bar loops
    for name, pat, kw in (
            ("acid_A", TR.ACID_A, dict(cutoff=520, env_mod=3600, res=0.85, drive=7)),
            ("acid_B", TR.ACID_B, dict(cutoff=620, env_mod=4200, res=0.87, drive=8)),
            ("acid_C", TR.ACID_C, dict(cutoff=700, env_mod=4800, res=0.90, drive=10)),
            ("acid_slow", TR.ACID_SEX, dict(cutoff=330, env_mod=2100, res=0.86,
                                            drive=4, decay=0.30))):
        line = I.acid(pat, TR.BPM, **kw)
        if write(os.path.join(out, "08_loops", f"{name}_1bar_150bpm.wav"), line):
            n += 1

    # raw single notes of the crying violin, for playing your own melody
    for midi, note in ((66, "F#4"), (69, "A4"), (71, "B4"), (73, "C#5"),
                       (74, "D5"), (76, "E5"), (78, "F#5"), (81, "A5")):
        x = ST.bowed(midi, 2.6, vel=0.85, seed=midi, **TR.CRY)
        if write(os.path.join(out, "05_strings", f"violin_note_{note}.wav"), x):
            n += 1

    open(os.path.join(out, "README.txt"), "w").write(README)
    print(f"wrote {n} files to {out}/")
    return out


README = """CONCRETE CATHEDRAL - isolated sounds
====================================

Every sound used in the track, rendered on its own.
44.1 kHz, 16-bit, mono, each peak-normalised to -1 dBFS.

Track key: F# minor.  Tempo: 150 BPM.
Files with '150bpm' in the name are tempo-locked loops; everything else
is a one-shot you can trigger freely.

01_drums        kick variants, hats, claps, snares, toms, percussion
02_industrial   struck metal, anvils, chains, steam, machines, the BOOMs
03_textures     drones and noise beds to sit under everything
04_synths       screeches, hoovers (by note), war horns, saw stabs, pads
05_strings      the crying violin - phrases, single notes, and the section
06_fx           impacts, risers, sub drops, reverse swells
07_speech       the spoken line in four treatments
08_loops        one-bar acid lines and a full groove bar

Notes
-----
- The hoover and saw stabs are named by note (F#2 ... C#4) so you can
  build your own riff without repitching.
- 05_strings/violin_note_*.wav are single sustained notes with the full
  crying articulation (portamento, growing vibrato, swell, sob), so you
  can sequence your own melody.
- The BOOMs are mostly sub-bass energy. Check them on something with
  real low end; on laptop speakers they will sound like nothing.
- Everything here was synthesised from scratch in numpy/scipy. Source:
  see the repo this came from.
"""

if __name__ == "__main__":
    main()
