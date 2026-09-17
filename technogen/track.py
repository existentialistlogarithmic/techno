"""The arrangement: 'CONCRETE CATHEDRAL' - 150 BPM, F# minor.

A lament played by strings, then taken apart by the machine.

    bars           section
    000-023        lament                 solo violin over a string section
    024-031        the turn               the strings are put through the machine
    032-047        industrial groove      struck metal, the kick still small
    048-063        build 1
    064-071        pre-drop
    072-103        DROP 1
    104-111        transition             tape brake, sub drop, a ghost of the theme
    112-127        dark mid-section
    128-143        build 2
    144-175        DROP 2
    176-191        breakdown              the violin comes back
    192-199        build 3
    200-223        DROP 3                 the theme returns as a distorted lead
    224-233        outro
"""

import numpy as np

from .dsp import (SR, biquad, sweep, reverb, delay, widen, tanh_drive, drive_os,
                  soft_clip, normalize, env_curve, env_ar, noise, n_samples, db,
                  pitch_shift_naive, tape_stop, gate, fit, supersaw, transient_shape)
from . import instruments as I
from . import texture as T
from . import strings as ST
from .mixer import Session, send_reverb, send_delay, master, balance

BPM = 150.0
BARS = 234
RNG = np.random.default_rng(2024)

BUSES = ["kick", "kickfar", "sub", "rumble", "drums", "metal", "bass", "lead",
         "voice", "scream", "speech", "breath", "strings", "solo", "fx", "pad", "air"]

TARGETS = {
    "kick":    (0.0, "low"),
    "kickfar": (-11.0, "low"),
    "sub":     (-7.5, "low"),
    "rumble":  (-8.5, "low"),
    "drums":   (1.0, "mid"),
    "metal":   (-0.5, "mid"),
    "bass":    (-0.5, "mid"),
    "lead":    (3.0, "mid"),
    "voice":   (-0.5, "mid"),
    "scream":  (-2.0, "mid"),
    "speech":  (1.0, "mid"),
    "breath":  (-7.0, "mid"),
    "strings": (6.5, "mid"),
    "solo":    (9.0, "mid"),
    "fx":      (-1.0, "mid"),
    "pad":     (-5.0, "mid"),
    "air":     (-16.0, "mid"),
}

LOW_WEIGHT = [(0, 0.02), (24, 0.06), (28, 0.14), (32, 0.48), (40, 0.52), (48, 0.44),
              (56, 0.52), (63.9, 0.58), (64, 0.34), (71.9, 0.20), (72, 1.0),
              (103.9, 1.0), (104, 0.08), (112, 0.42), (120, 0.50), (128, 0.54),
              (136, 0.64), (143.9, 0.68), (144, 1.0), (175.9, 1.0), (176, 0.04),
              (184, 0.22), (192, 0.48), (199.9, 0.58), (200, 1.0), (223.9, 1.0),
              (224, 0.74), (229, 0.54), (232, 0.24), (234, 0.08)]

KICK_GAIN = [(0, 0.35), (28, 0.45), (32, 0.80), (40, 0.84), (48, 0.76), (56, 0.82),
             (63.9, 0.86), (64, 0.88), (71.9, 0.88), (72, 1.0), (103.9, 1.0),
             (104, 0.45), (112, 0.74), (120, 0.80), (128, 0.84), (136, 0.90),
             (143.9, 0.92), (144, 1.0), (175.9, 1.0), (176, 0.5), (184, 0.7),
             (192, 0.86), (199.9, 0.92), (200, 1.0), (223.9, 1.0), (224, 0.94),
             (230, 0.78), (234, 0.35)]

KICK_TONE = [(0, 1200), (28, 1500), (32, 2200), (40, 2900), (48, 3600), (56, 4400),
             (63.9, 4800), (64, 5200), (71.9, 5200), (72, 20000), (103.9, 20000),
             (104, 2000), (112, 5200), (120, 6500), (128, 7500), (136, 9000),
             (143.9, 10000), (144, 20000), (175.9, 20000), (176, 2400), (184, 4200),
             (192, 6000), (199.9, 7000), (200, 20000), (227, 20000), (231, 8000),
             (234, 2000)]

# ------------------------------------------------------------- the lament

# F# minor. Slow: written in half-time, so one melody beat is two real beats.
LAMENT_A = [(73, 3), (71, 1), (69, 4),
            (71, 2), (69, 2), (68, 4),
            (66, 4), (69, 2), (68, 2),
            (66, 5), (None, 3)]

LAMENT_B = [(78, 3), (76, 1), (74, 4),
            (73, 2), (71, 2), (69, 4),
            (68, 4), (66, 4),
            (64, 6), (None, 2)]

# i - VI - III - VII, two bars each
LAMENT_CHORDS = [([42, 54, 57, 61], 8), ([38, 50, 57, 62], 8),
                 ([45, 52, 57, 64], 8), ([40, 52, 56, 64], 8)]

# the descending lament tetrachord, the oldest sad gesture there is
LAMENT_BASS = [(42, 8), (40, 8), (38, 8), (37, 8)]

def build_palette():
    p = {}
    # --- drums
    p["kick"] = I.kick(0.62, tune=45.0, punch=1.0, drive=7.5, decay=0.135, dirt=0.7)
    p["kick_hard"] = I.kick(0.62, tune=45.0, punch=1.15, drive=10.0, decay=0.15, dirt=0.95)
    p["kick_max"] = I.kick(0.66, tune=44.0, punch=1.25, drive=12.0, decay=0.16, dirt=1.0)
    p["kick_soft"] = I.kick(0.55, tune=45.0, punch=0.8, drive=3.5, decay=0.12, dirt=0.25)
    p["kick_short"] = I.kick(0.3, tune=46.0, punch=1.0, drive=8.0, decay=0.07, dirt=0.6)
    p["sub"] = I.kick_sub(0.75, tune=45.0, decay=0.2)

    p["hat"] = I.hat(0.05, tone=9200, decay=0.012, metal=0.35)
    p["hat2"] = I.hat(0.05, tone=11000, decay=0.009, metal=0.55)
    p["hat_tip"] = I.hat(0.035, tone=10500, decay=0.006, metal=0.25)
    p["ohat"] = I.hat(0.30, tone=8200, decay=0.075, metal=0.45)
    p["ohat_long"] = I.hat(0.55, tone=7600, decay=0.16, metal=0.5)
    p["clap"] = I.clap(0.5, tone=1600, body=0.35)
    p["clap_tight"] = I.clap(0.3, tone=2100, spread=0.007, body=0.2)
    p["snare"] = I.snare(0.32, tune=196, snap=0.78)
    p["snare_t"] = I.snare(0.16, tune=230, snap=0.85)
    p["rim"] = I.rim(0.12, 1750)
    p["ride"] = I.ride(1.0, tone=5400, decay=0.36)
    p["tom_h"] = I.tom(0.35, tune=150, decay=0.1)
    p["tom_l"] = I.tom(0.45, tune=96, decay=0.15)
    p["shaker"] = I.hat(0.07, tone=6800, decay=0.020, metal=0.05)

    # --- industrial
    p["metal_a"] = T.metal(0.9, 520, T.PLATE_RATIOS, decay=0.30, seed=1)
    p["metal_b"] = T.metal(0.55, 780, T.CAN_RATIOS, decay=0.16, seed=2)
    p["metal_c"] = T.metal(1.4, 330, T.PLATE_RATIOS, decay=0.55, seed=3)
    p["anvil"] = T.anvil(0.7, 1150, seed=4)
    p["anvil_lo"] = T.anvil(0.9, 640, seed=5)
    p["clang"] = T.clang(2.6, 155, seed=6)
    p["clang_hi"] = T.clang(1.8, 260, seed=7)
    p["chain"] = T.chain(1.1, 18, seed=8)
    p["chain_s"] = T.chain(0.55, 9, seed=9)
    p["steam"] = T.steam(1.7, seed=10)
    p["steam_s"] = T.steam(0.8, seed=11, pitch=1.3)
    p["scrape"] = T.scrape(0.9, seed=12)
    p["scrape_l"] = T.scrape(1.6, seed=13, f_from=500, f_to=2600)
    p["machine"] = T.machine(25.6, rate=6.25, seed=14)
    p["machine_lo"] = T.machine(25.6, rate=3.125, seed=15, tune=62.0)
    p["conveyor"] = T.conveyor(6.4, BPM, seed=16)
    p["conveyor2"] = T.conveyor(6.4, BPM, seed=17, tune=210.0)

    # --- synths
    for m in (42, 45, 47, 49, 52, 54, 57, 59, 61):
        p[f"hoov{m}"] = I.hoover(m, 1.5, detune=27, sweep_from=1.5, glide=0.12,
                                 cutoff=(600, 5600), res=0.74, drive=5.5)
        p[f"hoovS{m}"] = I.hoover(m, 0.42, detune=22, sweep_from=1.28, glide=0.05,
                                  cutoff=(900, 6200), res=0.68, drive=6.5)
        p[f"stab{m}"] = I.stab(m, 0.26, detune=20, cutoff=3000, res=0.72, drive=6.0)

    p["scr_up"] = I.screech(1.6, 520, 5200, 6.5, 0.93, 9, "up", seed=1)
    p["scr_dn"] = I.screech(1.2, 4600, 420, 9.0, 0.9, 8, "down", seed=2)
    p["scr_ud"] = I.screech(2.2, 700, 6400, 5.0, 0.95, 11, "updown", seed=3)
    p["scr_st"] = I.screech(0.34, 1500, 4800, 14.0, 0.9, 10, "up", seed=4)
    p["scr_st2"] = I.screech(0.28, 5200, 1800, 16.0, 0.9, 10, "down", seed=5)
    p["scr_long"] = I.screech(3.2, 380, 7200, 4.0, 0.96, 12, "up", seed=6)
    p["scr_evil"] = I.screech(2.6, 240, 8200, 3.2, 0.97, 13, "up", seed=8)

    p["pad_a"] = I.pad([42, 49, 54, 57], 12.8, detune=16, cutoff=1300, drive=1.5)
    p["pad_b"] = I.pad([40, 47, 52, 59], 12.8, detune=18, cutoff=1100, drive=1.5)
    p["pad_sex"] = I.pad([42, 49, 54, 61], 12.8, detune=22, cutoff=1900, drive=1.3)
    p["pad_dark"] = I.pad([30, 37, 42, 49], 19.2, detune=12, cutoff=750, drive=1.2)

    # --- FX
    p["riser_n"] = I.riser_noise(6.4, 260, 12000, q=2.2, swell=1.8, seed=4)
    p["riser_n2"] = I.riser_noise(12.8, 180, 13000, q=2.0, swell=2.4, seed=8)
    p["riser_t"] = I.riser_tone(6.4, 34, 76, detune=22, drive=3.5)
    p["riser_t2"] = I.riser_tone(12.8, 30, 80, detune=26, drive=4.0)
    p["down"] = I.downlifter(2.4, 2600, 55)
    p["impact"] = I.impact(3.0, tune=44)
    p["rev_swell"] = I.reverse_swell(3.2, seed=9)
    p["rev_swell2"] = I.reverse_swell(1.6, seed=10, bright=9000)
    p["zap"] = I.zap(0.35)
    p["sub_drop"] = T.sub_drop(3.4, 120, 24)
    p["noise_fall"] = T.noise_fall(2.6, 9000, 220, seed=3)

    # --- voices
    p["moan_a"] = I.moan(2.6, root=54, seed=0, breath=0.30, up=True)
    p["moan_b"] = I.moan(3.2, root=49, seed=1, breath=0.36, up=False)
    p["moan_c"] = I.moan(2.0, root=57, seed=2, breath=0.26, up=True)
    p["moan_close"] = I.moan(2.2, root=56, seed=4, breath=0.42, up=True)
    p["chat_a"] = I.vocal_chatter(3.0, syllables=7, root=45, seed=1)
    p["chat_b"] = I.vocal_chatter(2.2, syllables=5, root=50, seed=4)
    p["whisper"] = I.whisper(4.0, seed=3)
    p["whisper2"] = I.whisper(2.6, seed=6, tone=1.15)
    p["vstab"] = I.vocal_stab(0.40, root=57, vowel="eh", drive=7)
    p["vstab2"] = I.vocal_stab(0.32, root=62, vowel="ah", drive=8)
    p["vstab_low"] = pitch_shift_naive(I.vocal_stab(0.5, root=50, vowel="oh", drive=5), 1.45)

    p["sigh_a"] = T.sigh(1.9, root=57, seed=1)
    p["sigh_b"] = T.sigh(2.3, root=52, seed=2, breathiness=0.62)
    p["breath_out"] = T.breath(1.1, "out", seed=1, intensity=0.7)
    p["breath_in"] = T.breath(0.9, "in", seed=2, intensity=0.8)
    p["breath_short"] = T.breath(0.45, "out", seed=3, intensity=0.9, tone=1.15)

    p["help_far"] = I.scream_help(1.9, pitch=0.85, seed=7, drive=5.0, effort=0.85)
    p["help_mid"] = I.scream_help(1.6, pitch=1.0, seed=0, drive=6.5, effort=1.0)
    p["help_panic"] = I.scream_help(1.25, pitch=1.22, seed=3, drive=8.0, effort=1.2)
    p["cry"] = I.scream(1.7, f0=(230, 560, 300),
                        vowel_path=((0.0, "eh"), (0.5, "ah"), (1.0, "ah")),
                        roughness=0.65, breath=0.5, drive=6.5, seed=5)
    p["cry_short"] = I.scream(0.85, f0=(320, 620, 380),
                              vowel_path=((0.0, "ah"), (1.0, "eh")),
                              roughness=0.7, breath=0.45, drive=7.5, seed=9, effort=1.15)

    # --- speech
    sp = T.speech_layers("assets/confession.wav", shift=1.055, seed=1)
    p["say_close"], p["say_whisper"] = sp["close"], sp["whisper"]
    p["say_choir"], p["say_radio"] = sp["choir"], sp["radio"]
    fg = T.speech_layers("assets/forgive.wav", shift=1.08, seed=2)
    p["forgive_close"], p["forgive_whisper"] = fg["close"], fg["whisper"]
    sn = T.speech_layers("assets/sins.wav", shift=1.02, seed=3)
    p["sins_close"], p["sins_choir"] = sn["close"], sn["choir"]

    # --- rave
    p["siren"] = T.siren(4.8, 380, 1500, rate=0.42, drive=6)
    p["siren_s"] = T.siren(2.0, 500, 2100, rate=0.9, drive=7)
    p["horn"] = T.war_horn(42, 3.0, drive=7, growl=0.5)
    p["horn_hi"] = T.war_horn(49, 2.4, drive=8, growl=0.6)
    p["whoosh_up"] = T.whoosh(2.4, True, seed=1)
    p["whoosh_up_l"] = T.whoosh(4.8, True, seed=2)
    p["whoosh_dn"] = T.whoosh(2.2, False, seed=3)

    # --- the lament
    p["violin_a"] = ST.phrase(LAMENT_A, BPM, beat_unit=2.0, vel=0.72, seed=1)
    p["violin_b"] = ST.phrase(LAMENT_B, BPM, beat_unit=2.0, vel=0.78, seed=2)
    p["violin_a2"] = ST.phrase(LAMENT_A, BPM, beat_unit=2.0, vel=0.80, seed=5,
                               bright=1.15)
    p["strings_a"] = ST.chords(LAMENT_CHORDS, BPM, players=3, vel=0.55, seed=3)
    p["strings_b"] = ST.chords(LAMENT_CHORDS, BPM, players=4, vel=0.65, seed=4)
    p["cello"] = ST.phrase(LAMENT_BASS, BPM, beat_unit=1.0, vel=0.62,
                           body=ST.CELLO_BODY, seed=6, vib_depth=0.004)
    # the theme after the machine has had it
    p["violin_dead"] = ST.desecrate(p["violin_a2"], 1.0, cutoff=3000, drive=8)
    p["strings_dead"] = ST.desecrate(p["strings_b"], 1.0, cutoff=2200, drive=7)
    for i, amt in enumerate((0.25, 0.5, 0.75, 1.0)):
        p[f"strings_rot{i}"] = ST.desecrate(p["strings_a"], amt,
                                            cutoff=3400 - 700 * i, drive=4 + 2 * i)

    # a bar of the groove, pre-mixed, so the transition can brake it like tape
    p["groove_bar"] = _groove_bar(p)
    return p


def _groove_bar(p):
    step = 60.0 / BPM / 4.0
    n = n_samples(step * 16)
    out = np.zeros(n)
    for st in (0, 4, 8, 12):
        i = n_samples(st * step)
        k = p["kick"][:min(len(p["kick"]), n - i)]
        out[i:i + len(k)] += k
    for st in range(0, 16, 2):
        i = n_samples(st * step)
        h = p["hat"] if st % 4 else p["hat2"]
        h = h[:min(len(h), n - i)]
        out[i:i + len(h)] += 0.4 * h
    line = I.acid(ACID_B, BPM, cutoff=620, env_mod=4200, res=0.87, drive=8)
    out += 0.5 * fit(line, n)
    return normalize(out, 0.95)


# ---------------------------------------------------------------- patterns

ACID_A = [(30, 2, 1, 0), (None, 1, 0, 0), (30, 1, 0, 0), (42, 1, 0, 1),
          (30, 1, 0, 0), (None, 1, 0, 0), (33, 1, 0, 0),
          (30, 2, 1, 0), (None, 1, 0, 0), (37, 1, 0, 1), (30, 1, 0, 0),
          (35, 1, 0, 0), (30, 1, 0, 0), (42, 1, 1, 0)]

ACID_B = [(30, 1, 1, 0), (30, 1, 0, 0), (42, 1, 0, 1), (30, 1, 0, 0),
          (37, 1, 1, 0), (30, 1, 0, 0), (30, 1, 0, 0), (45, 1, 0, 1),
          (30, 2, 1, 0), (35, 1, 0, 0), (30, 1, 0, 0),
          (33, 1, 0, 0), (30, 1, 0, 0), (49, 1, 1, 1), (30, 1, 0, 0)]

ACID_C = [(30, 2, 1, 0), (42, 1, 0, 1), (30, 1, 0, 0), (30, 1, 0, 0), (45, 1, 1, 0),
          (30, 1, 0, 0), (49, 1, 0, 1),
          (30, 1, 1, 0), (30, 1, 0, 0), (54, 1, 1, 0), (42, 1, 0, 1), (30, 1, 0, 0),
          (37, 1, 0, 0), (30, 1, 0, 0), (52, 1, 1, 1)]

# slower, rounder, more space - the mid-section line
ACID_SEX = [(30, 3, 1, 0), (None, 1, 0, 0), (42, 2, 0, 1), (None, 2, 0, 0),
            (37, 2, 1, 0), (None, 1, 0, 0), (35, 1, 0, 1),
            (30, 2, 0, 0), (49, 1, 1, 1), (None, 1, 0, 0)]

HOOVER_RIFF = [(0, 0, 54), (0, 12, 49), (2, 0, 47), (2, 10, 45),
               (4, 0, 54), (4, 8, 57), (6, 0, 52), (6, 6, 49)]

HOOVER_RIFF2 = [(0, 0, 57), (1, 8, 54), (2, 0, 61), (3, 4, 57),
                (4, 0, 59), (5, 8, 54), (6, 0, 52), (7, 2, 49)]

LEAD_RIFF = [(0, 0, 54, 1.5), (1, 8, 57, 0.75), (2, 0, 52, 1.0), (3, 4, 49, 0.75),
             (4, 0, 57, 1.5), (5, 8, 61, 0.75), (6, 0, 59, 1.0), (7, 2, 54, 1.25)]

METAL_PAT = [(0, "metal_a", 0.7), (3, "metal_b", 0.45), (6, "metal_b", 0.4),
             (7, "anvil", 0.35), (10, "metal_a", 0.5), (11, "metal_b", 0.4),
             (14, "anvil_lo", 0.45)]

METAL_PAT2 = [(2, "metal_b", 0.5), (5, "anvil", 0.4), (6, "metal_a", 0.55),
              (9, "metal_b", 0.45), (12, "anvil_lo", 0.5), (13, "metal_b", 0.35),
              (15, "metal_c", 0.4)]


# ---------------------------------------------------------------- layers

def far(x, hz, rolloff=2):
    """Distance cue: air and walls eat the top end long before the level."""
    return biquad(x, "lp", hz, 0.7, stages=rolloff)


def lay_kicks(s, p, b0, b1, key="kick", gain=1.0, ghosts=False, rolls=True,
              sub_gain=1.0, every=4, bus="kick"):
    for b in range(b0, b1):
        for st in range(0, 16, every):
            s.place(bus, p[key], b, st, gain=gain * (1.0 if st == 0 else 0.97))
            s.mark_kick(b, st)
            if sub_gain:
                s.place("sub", p["sub"], b, st, gain=0.85 * sub_gain)
        if ghosts and (b - b0) % 4 == 3:
            s.place(bus, p["kick_short"], b, 14, gain=gain * 0.55)
            s.mark_kick(b, 14)
        if rolls and (b - b0) % 16 == 15:
            for st in (12, 13, 14, 15):
                s.place(bus, p["kick_short"], b, st, gain=gain * (0.5 + 0.13 * (st - 12)))
                s.mark_kick(b, st)


def lay_hats(s, p, b0, b1, density=16, open_off=True, gain=0.5, swing=0.0,
             jitter=0.0012, tips=False):
    for b in range(b0, b1):
        for st in range(0, 16, max(1, 16 // density)):
            acc = 1.0 if st % 4 == 2 else (0.72 if st % 4 == 0 else 0.6)
            k = "hat2" if st % 8 == 4 else "hat"
            off = swing if st % 2 else 0.0
            s.place("drums", p[k], b, st + off + RNG.normal(0, jitter) / s.step,
                    gain=gain * acc * RNG.uniform(0.85, 1.05),
                    pan_=RNG.uniform(-0.28, 0.28))
        if tips:
            for st in (1, 7, 9, 15):
                s.place("drums", p["hat_tip"], b, st + swing, gain=gain * 0.26,
                        pan_=RNG.uniform(-0.5, 0.5))
        if open_off:
            for st in (2, 6, 10, 14):
                s.place("drums", p["ohat"], b, st, gain=gain * 0.85,
                        pan_=RNG.uniform(-0.15, 0.15))


def lay_shaker(s, p, b0, b1, gain=0.3, swing=0.06):
    for b in range(b0, b1):
        for st in range(0, 16):
            off = swing if st % 2 else 0.0
            v = 0.55 if st % 4 == 0 else (1.0 if st % 4 == 2 else 0.4)
            s.place("drums", p["shaker"], b, st + off, gain=gain * v,
                    pan_=RNG.uniform(-0.45, 0.45))


def lay_perc(s, p, b0, b1, gain=1.0, claps=True, rides=False, rims=True, tight=False):
    for b in range(b0, b1):
        i = b - b0
        if claps and i % 2 == 1:
            s.place("drums", p["clap_tight" if tight else "clap"], b, 8,
                    gain=0.52 * gain, pan_=0.08)
        if rims:
            for st in (3, 11, 15):
                if (i + st) % 3:
                    s.place("drums", p["rim"], b, st, gain=0.22 * gain,
                            pan_=RNG.uniform(-0.6, 0.6))
        if rides:
            for st in range(0, 16, 2):
                s.place("drums", p["ride"], b, st, gain=0.16 * gain,
                        pan_=RNG.uniform(-0.4, 0.4))
        if i % 8 == 7:
            s.place("drums", p["tom_l"], b, 10, gain=0.4 * gain, pan_=-0.3)
            s.place("drums", p["tom_h"], b, 13, gain=0.36 * gain, pan_=0.35)


def lay_metal(s, p, b0, b1, gain=0.5, alt=True, chains=True):
    for b in range(b0, b1):
        i = b - b0
        pat = METAL_PAT2 if (alt and i % 2) else METAL_PAT
        for st, key, g in pat:
            s.place("metal", p[key], b, st, gain=gain * g * RNG.uniform(0.85, 1.1),
                    pan_=RNG.uniform(-0.55, 0.55))
        if chains and i % 4 == 3:
            s.place("metal", p["chain_s"], b, 12, gain=gain * 0.5, pan_=RNG.uniform(-0.6, 0.6))
        if i % 8 == 7:
            s.place("metal", p["scrape"], b, 8, gain=gain * 0.45, pan_=-0.4)


def snare_roll(s, p, b0, bars=2, gain=0.8):
    total_steps = bars * 16
    t = 0.0
    while t < total_steps:
        frac = t / total_steps
        key = "snare_t" if frac > 0.45 else "snare"
        g = gain * (0.45 + 0.75 * frac) * RNG.uniform(0.9, 1.05)
        s.place("drums", p[key], b0 + int(t // 16), t % 16, gain=g,
                pan_=RNG.uniform(-0.2, 0.2))
        t += max(0.25, 4 - 3 * frac)


def lay_acid(s, p, b0, b1, pattern, gain=0.5, **kw):
    line = I.acid(pattern, BPM, **kw)
    for b in range(b0, b1):
        s.place("bass", line, b, 0, gain=gain)


def lay_hoover(s, p, b0, b1, gain=0.5, riff=HOOVER_RIFF, stabs=True):
    for b in range(b0, b1):
        i = (b - b0) % 8
        for rb, rs, m in riff:
            if rb == i:
                s.place("lead", p[f"hoov{m}"], b, rs, gain=gain,
                        pan_=RNG.uniform(-0.12, 0.12))
        if stabs and i % 2 == 1:
            m = 54 if i % 4 == 1 else 49
            s.place("lead", p[f"hoovS{m}"], b, 14, gain=gain * 0.7, pan_=0.2)


def lay_lead_screech(s, p, b0, b1, gain=0.45):
    keys = ["scr_st", "scr_st2"]
    for b in range(b0, b1):
        i = (b - b0) % 8
        for rb, rs, m, ln in LEAD_RIFF:
            if rb == i:
                sc = I.screech(ln * s.bar / 2, base=I.note_hz(m) * 1.6,
                               top=I.note_hz(m) * 5.2, wobble=5.5 + (m % 5),
                               res=0.94, drive=10, direction="updown", seed=m)
                s.place("lead", sc, b, rs, gain=gain, pan_=RNG.uniform(-0.25, 0.25))
        if i % 2 == 0:
            s.place("lead", p[keys[i // 2 % 2]], b, 6, gain=gain * 0.65, pan_=-0.3)
            s.place("lead", p[keys[(i // 2 + 1) % 2]], b, 14, gain=gain * 0.65, pan_=0.3)


def lay_gated_pad(s, p, key, b0, bars, gain=0.4, pattern=(1, 0, .7, 0, 1, 0, .5, .8)):
    """Pulsing pad - the slink under the mid-section."""
    x = gate(p[key], s.step * 2, list(pattern), smooth=0.012, floor=0.06)
    s.place("pad", x, b0, 0, gain=gain)


def fill(s, p, b, kind="metal"):
    """Phrase-end punctuation so the ear knows where it is."""
    if kind == "metal":
        s.place("metal", p["anvil"], b, 12, gain=0.5, pan_=-0.3)
        s.place("metal", p["metal_b"], b, 14, gain=0.45, pan_=0.35)
        s.place("metal", p["chain_s"], b, 15, gain=0.4, pan_=0.1)
    elif kind == "tom":
        for k, st in (("tom_l", 10), ("tom_h", 12), ("tom_l", 13), ("tom_h", 14)):
            s.place("drums", p[k], b, st, gain=0.45, pan_=RNG.uniform(-0.4, 0.4))
    elif kind == "rev":
        s.place("fx", p["rev_swell2"], b, 8, gain=0.34, pan_=RNG.uniform(-0.3, 0.3))
    elif kind == "steam":
        s.place("metal", p["steam_s"], b, 12, gain=0.4, pan_=RNG.uniform(-0.5, 0.5))


# ---------------------------------------------------------------- sections


# ---------------------------------------------------------- transition kit

def approach(s, p, bar, power=1.0, bars=2, steam=True):
    """Lead into a section start at `bar`: rise, then a gap to fall into."""
    s.place("fx", p["rev_swell"], bar - bars, 0, gain=0.34 * power)
    s.place("fx", p["whoosh_up"], bar - 1.5, 0, gain=0.40 * power)
    if steam:
        s.place("metal", p["steam"], bar - 1, 8, gain=0.34 * power)
    s.place("breath", p["breath_in"], bar - 1, 12, gain=0.42 * power, pan_=0.2)


def enter(s, p, bar, power=1.0, horn=False, siren=False):
    """Mark a section start so the ear knows something changed."""
    s.place("fx", p["impact"], bar, 0, gain=0.70 * power)
    s.place("metal", p["clang"], bar, 0, gain=0.50 * power)
    s.place("fx", p["whoosh_dn"], bar, 0, gain=0.34 * power)
    if horn:
        s.place("lead", p["horn"], bar, 0, gain=0.34 * power)
    if siren:
        s.place("fx", p["siren"], bar, 0, gain=0.26 * power)


def exit_(s, p, bar, power=1.0, drop=True):
    """Close a section: downlifter, and optionally pull the floor out."""
    s.place("fx", p["down"], bar, 12, gain=0.30 * power)
    if drop:
        s.place("fx", p["sub_drop"], bar, 12, gain=0.40 * power)


# ---------------------------------------------------------------- sections

def build(verbose=True):
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    log("rendering sound palette (strings are slow)...")
    p = build_palette()
    s = Session(BPM, BARS, tail=8.0)

    # --------------------------------------------------------- 000-023 lament
    log("lament")
    s.place("air", I.room_tone(s.dur, level=0.075), 0, 0, gain=1.0)
    s.place("metal", p["machine_lo"], 0, 0, gain=0.30)
    s.place("strings", p["strings_a"], 0, 0, gain=1.45)
    s.place("strings", p["cello"], 0, 0, gain=1.15)
    s.place("solo", p["violin_a"], 4, 0, gain=1.0, pan_=-0.08)
    s.place("strings", p["strings_a"], 8, 0, gain=1.35)
    s.place("strings", p["cello"], 8, 0, gain=1.05)
    s.place("speech", p["say_close"], 12, 4, gain=0.78)
    s.place("speech", p["say_whisper"], 12, 4.3, gain=0.40, pan_=-0.55)
    s.place("speech", far(p["say_radio"], 3000), 12, 4, gain=0.16, pan_=0.3)
    s.place("strings", p["strings_a"], 16, 0, gain=1.40)
    s.place("strings", p["cello"], 16, 0, gain=1.10)
    s.place("solo", p["violin_b"], 16, 0, gain=1.0, pan_=0.06)
    s.place("metal", far(p["machine"], 1300), 12, 0, gain=0.42)
    s.place("scream", far(p["help_far"], 1100), 10, 6, gain=0.26, pan_=-0.45)
    s.place("metal", p["scrape_l"], 19, 12, gain=0.42, pan_=-0.5)
    s.place("breath", p["breath_out"], 21, 8, gain=0.40, pan_=0.3)

    # ----------------------------------------------------------- 024-031 turn
    log("the turn: the machine takes the theme")
    for i, b in enumerate((24, 26, 28, 30)):
        s.place("strings", p[f"strings_rot{i}"], b, 0, gain=0.55 + 0.12 * i)
    s.place("solo", ST.desecrate(p["violin_a"], 0.55, 3200, 5), 24, 0, gain=0.9)
    s.place("metal", p["machine"], 24, 0, gain=0.55)
    s.place("metal", p["machine"], 28, 0, gain=0.75)
    s.place("scream", far(p["cry"], 2400), 25, 6, gain=0.46, pan_=0.4)
    s.place("metal", p["steam"], 26, 8, gain=0.55, pan_=-0.3)
    s.place("metal", p["chain"], 27, 2, gain=0.6, pan_=0.45)
    for b in range(26, 32):
        for st in ((0, 8) if b < 29 else (0, 4, 8, 12)):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.22 + 0.035 * (b - 26))
            s.mark_kick(b, st)
    s.place("fx", p["whoosh_up_l"], 29, 0, gain=0.42)
    s.place("lead", p["horn"], 30, 0, gain=0.42)
    approach(s, p, 32, power=1.0)

    # --------------------------------------------- 032-047 industrial groove
    log("industrial groove")
    enter(s, p, 32, power=0.9)
    lay_kicks(s, p, 32, 48, key="kick", gain=0.88, ghosts=True, rolls=False)
    for b in range(32, 48, 4):
        s.place("metal", p["conveyor"], b, 0, gain=0.62)
    lay_metal(s, p, 32, 48, gain=0.95)
    lay_hats(s, p, 32, 40, density=4, open_off=False, gain=0.30)
    lay_hats(s, p, 40, 48, density=8, gain=0.38)
    lay_perc(s, p, 36, 48, gain=0.6, claps=True)
    s.place("metal", p["machine"], 32, 0, gain=0.75)
    s.place("strings", p["strings_dead"], 40, 0, gain=0.20)
    s.place("breath", p["breath_out"], 35, 12, gain=0.45, pan_=-0.3)
    s.place("speech", p["forgive_whisper"], 43, 8, gain=0.5, pan_=-0.35)
    for b in (39, 47):
        fill(s, p, b, "metal")
    approach(s, p, 48, power=0.7, steam=False)

    # ------------------------------------------------------- 048-063 build 1
    log("build 1")
    lay_kicks(s, p, 48, 64, key="kick", gain=0.94, ghosts=True, rolls=False)
    lay_hats(s, p, 48, 56, density=8, gain=0.40)
    lay_hats(s, p, 56, 64, density=16, gain=0.48, tips=True)
    lay_perc(s, p, 48, 64, gain=0.8)
    lay_metal(s, p, 48, 64, gain=0.55)
    lay_acid(s, p, 52, 64, ACID_A, gain=0.40, cutoff=400, env_mod=2600, res=0.80, drive=5)
    s.place("voice", p["vstab_low"], 55, 12, gain=0.3, pan_=-0.2)
    s.place("fx", p["riser_n2"], 56, 0, gain=0.26)
    s.place("lead", p["scr_up"], 62, 8, gain=0.26, pan_=-0.25)
    for b in (55, 63):
        fill(s, p, b, "tom")

    # ----------------------------------------------------- 064-071 pre-drop
    log("pre-drop")
    lay_kicks(s, p, 64, 68, key="kick", gain=1.0, ghosts=True, rolls=False)
    lay_hats(s, p, 64, 68, density=16, gain=0.48)
    lay_perc(s, p, 64, 68, gain=0.8)
    lay_acid(s, p, 64, 70, ACID_A, gain=0.46, cutoff=540, env_mod=3400, res=0.84, drive=6)
    s.place("fx", p["riser_n"], 68, 0, gain=0.42)
    s.place("fx", p["riser_t"], 68, 0, gain=0.30)
    s.place("fx", p["whoosh_up_l"], 69, 0, gain=0.40)
    snare_roll(s, p, 68, bars=3, gain=0.62)
    s.place("lead", p["scr_long"], 69, 0, gain=0.34, pan_=0.1)
    s.place("metal", p["steam"], 70, 0, gain=0.38)
    s.place("fx", p["rev_swell"], 70, 0, gain=0.40)
    exit_(s, p, 71, power=0.7, drop=False)

    # ------------------------------------------------------- 072-103 DROP 1
    log("DROP 1")
    enter(s, p, 72, power=1.0, horn=True)
    lay_kicks(s, p, 72, 104, key="kick", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 72, 104, density=16, gain=0.5, tips=True)
    lay_perc(s, p, 72, 104, gain=1.0, claps=True)
    lay_metal(s, p, 72, 104, gain=0.5)
    lay_acid(s, p, 72, 88, ACID_A, gain=0.5, cutoff=520, env_mod=3600, res=0.85, drive=7)
    lay_acid(s, p, 88, 104, ACID_B, gain=0.52, cutoff=600, env_mod=4200, res=0.87, drive=8)
    lay_hoover(s, p, 72, 104, gain=0.34)
    for b in range(72, 104, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.30, pan_=RNG.uniform(-0.3, 0.3))
        fill(s, p, b + 7, "metal")
    for b in range(76, 104, 8):
        s.place("voice", p["vstab"], b, 12, gain=0.34, pan_=0.15)
        s.place("voice", p["vstab2"], b + 2, 6, gain=0.26, pan_=-0.25)
    s.place("fx", p["siren"], 87, 0, gain=0.24)
    s.place("lead", p["scr_ud"], 87, 8, gain=0.34, pan_=-0.15)
    s.place("speech", p["sins_choir"], 92, 0, gain=0.42)
    s.place("fx", p["impact"], 88, 0, gain=0.4)
    s.place("strings", p["strings_dead"], 96, 0, gain=0.18)
    s.place("breath", p["breath_short"], 95, 14, gain=0.4, pan_=0.4)
    exit_(s, p, 103, power=1.0)

    # ---------------------------------------------------- 104-111 transition
    log("transition")
    s.place("fx", T.reverse_tail(p["clang"], 2.8), 102, 8, gain=0.5)
    s.place("drums", tape_stop(p["groove_bar"], start=0.12, end_ratio=0.035,
                               curve=1.7, max_stretch=2.6), 104, 0, gain=0.85)
    s.place("fx", p["sub_drop"], 104, 0, gain=0.62)
    s.place("fx", p["noise_fall"], 104, 2, gain=0.34)
    s.place("metal", p["steam"], 105, 6, gain=0.40, pan_=-0.35)
    s.place("breath", p["breath_out"], 105, 8, gain=0.62, pan_=0.25)
    s.place("solo", ST.desecrate(p["violin_a"], 0.35, 4000, 4), 106, 0, gain=0.30)
    s.place("voice", p["moan_close"], 106, 8, gain=0.48, pan_=-0.2)
    s.place("pad", p["pad_sex"], 106, 0, gain=0.70)
    s.place("breath", p["sigh_b"], 108, 4, gain=0.55, pan_=0.3)
    s.place("speech", p["forgive_close"], 110, 0, gain=0.58)
    s.place("breath", p["breath_in"], 111, 12, gain=0.6, pan_=-0.15)
    for b in range(108, 112):
        for st in (0, 8):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.42)
            s.mark_kick(b, st)

    # ----------------------------------------------------- 112-127 dark mid
    log("dark mid-section")
    lay_kicks(s, p, 112, 128, key="kick", gain=0.9, ghosts=False, rolls=False)
    lay_shaker(s, p, 112, 128, gain=0.30, swing=0.055)
    lay_hats(s, p, 112, 120, density=8, open_off=True, gain=0.32, swing=0.05)
    lay_hats(s, p, 120, 128, density=16, gain=0.40, swing=0.045, tips=True)
    lay_perc(s, p, 114, 128, gain=0.55, claps=True, tight=True)
    lay_acid(s, p, 112, 128, ACID_SEX, gain=0.48, cutoff=330, env_mod=2100,
             res=0.86, drive=4, decay=0.30)
    lay_gated_pad(s, p, "pad_sex", 112, 8, gain=0.85)
    lay_gated_pad(s, p, "pad_sex", 120, 8, gain=0.80,
                  pattern=(1, 0, .6, .8, 0, 1, .4, 0))
    s.place("strings", ST.desecrate(p["strings_a"], 0.5, 2600, 5), 116, 0, gain=0.26)
    for b, st, key, g, pn in [(113, 8, "moan_close", 0.44, -0.3), (116, 0, "sigh_a", 0.46, 0.35),
                              (118, 12, "breath_short", 0.5, -0.4), (121, 4, "moan_c", 0.40, 0.25),
                              (124, 0, "sigh_b", 0.44, -0.2), (126, 8, "moan_a", 0.42, 0.4)]:
        s.place("breath" if key.startswith(("sigh", "breath")) else "voice",
                p[key], b, st, gain=g, pan_=pn)
    s.place("metal", p["conveyor2"], 120, 0, gain=0.5)
    s.place("metal", p["conveyor2"], 124, 0, gain=0.5)
    s.place("speech", p["say_whisper"], 122, 0, gain=0.34, pan_=0.5)
    for b in (119, 127):
        fill(s, p, b, "rev")

    # -------------------------------------------------------- 128-143 build 2
    log("build 2")
    lay_kicks(s, p, 128, 140, key="kick", gain=0.96, ghosts=True, rolls=False)
    lay_hats(s, p, 128, 136, density=8, gain=0.40)
    lay_hats(s, p, 136, 144, density=16, gain=0.50, tips=True)
    lay_perc(s, p, 128, 144, gain=0.85)
    lay_metal(s, p, 128, 144, gain=0.5)
    lay_acid(s, p, 128, 144, ACID_B, gain=0.46, cutoff=460, env_mod=3800, res=0.86, drive=7)
    lay_hoover(s, p, 136, 140, gain=0.26, stabs=False)
    s.place("fx", p["riser_n2"], 136, 0, gain=0.34)
    s.place("fx", p["riser_t2"], 136, 0, gain=0.26)
    s.place("fx", p["riser_n"], 140, 0, gain=0.46)
    s.place("fx", p["whoosh_up_l"], 141, 0, gain=0.42)
    snare_roll(s, p, 140, bars=3, gain=0.7)
    s.place("lead", p["scr_long"], 141, 0, gain=0.36, pan_=-0.1)
    s.place("voice", p["vstab"], 139, 8, gain=0.34)
    s.place("metal", p["steam"], 142, 0, gain=0.38)
    s.place("fx", p["rev_swell"], 142, 0, gain=0.44)
    exit_(s, p, 143, power=0.8, drop=False)

    # ------------------------------------------------------- 144-175 DROP 2
    log("DROP 2")
    enter(s, p, 144, power=1.1, horn=True, siren=True)
    lay_kicks(s, p, 144, 176, key="kick_hard", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 144, 176, density=16, gain=0.54, tips=True)
    lay_perc(s, p, 144, 176, gain=1.0, claps=True, rides=True)
    lay_metal(s, p, 144, 176, gain=0.55)
    lay_acid(s, p, 144, 160, ACID_B, gain=0.54, cutoff=640, env_mod=4400, res=0.88, drive=9)
    lay_acid(s, p, 160, 176, ACID_C, gain=0.56, cutoff=700, env_mod=4800, res=0.90, drive=10)
    lay_hoover(s, p, 144, 160, gain=0.34)
    lay_hoover(s, p, 160, 176, gain=0.34, riff=HOOVER_RIFF2)
    lay_lead_screech(s, p, 152, 168, gain=0.34)
    for b in range(144, 176, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.32, pan_=RNG.uniform(-0.3, 0.3))
        s.place("voice", p["vstab2"], b + 3, 12, gain=0.28, pan_=RNG.uniform(-0.3, 0.3))
        fill(s, p, b + 7, "metal")
    s.place("speech", p["sins_choir"], 158, 0, gain=0.5)
    s.place("strings", p["strings_dead"], 164, 0, gain=0.20)
    s.place("lead", p["horn_hi"], 168, 0, gain=0.34)
    s.place("scream", far(p["cry_short"], 7000), 167, 14, gain=0.5, pan_=0.25)
    s.place("fx", p["siren_s"], 171, 8, gain=0.26)
    s.place("fx", p["impact"], 160, 0, gain=0.45)
    exit_(s, p, 175, power=1.0)

    # ----------------------------------------------------- 176-191 breakdown
    log("breakdown: the violin returns")
    s.place("fx", T.reverse_tail(p["impact"], 3.2), 174, 8, gain=0.45)
    s.place("pad", p["pad_dark"], 176, 0, gain=0.42)
    s.place("strings", p["strings_b"], 176, 0, gain=1.40)
    s.place("strings", p["cello"], 176, 0, gain=1.10)
    s.place("solo", p["violin_a2"], 178, 0, gain=1.0, pan_=-0.05)
    s.place("speech", p["say_choir"], 184, 0, gain=0.52)
    s.place("strings", p["strings_b"], 184, 0, gain=1.30)
    s.place("voice", p["moan_b"], 182, 8, gain=0.38, pan_=0.26)
    s.place("breath", p["sigh_b"], 187, 0, gain=0.44, pan_=-0.3)
    s.place("scream", far(p["help_far"], 1700), 186, 4, gain=0.40, pan_=0.45)
    s.place("metal", p["machine_lo"], 176, 0, gain=0.5)
    for b in range(186, 192):
        s.place("drums", p["rim"], b, 6, gain=0.16, pan_=RNG.uniform(-0.6, 0.6))
        for st in (0, 8):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.4 + 0.035 * (b - 186))
            s.mark_kick(b, st)

    # -------------------------------------------------------- 192-199 build 3
    log("build 3")
    lay_kicks(s, p, 192, 198, key="kick", gain=0.94, ghosts=True, rolls=False)
    lay_hats(s, p, 192, 200, density=16, gain=0.48, tips=True)
    lay_perc(s, p, 192, 200, gain=0.8)
    lay_metal(s, p, 192, 200, gain=0.5)
    lay_acid(s, p, 192, 199, ACID_C, gain=0.5, cutoff=600, env_mod=4200, res=0.88, drive=8)
    s.place("strings", ST.desecrate(p["strings_b"], 0.6, 2800, 6), 192, 0, gain=0.26)
    s.place("fx", p["riser_n"], 196, 0, gain=0.48)
    s.place("fx", p["riser_t"], 196, 0, gain=0.36)
    s.place("fx", p["whoosh_up_l"], 197, 0, gain=0.46)
    snare_roll(s, p, 196, bars=3, gain=0.76)
    s.place("lead", p["scr_evil"], 197, 0, gain=0.38, pan_=0.1)
    s.place("metal", p["steam"], 198, 0, gain=0.40)
    s.place("fx", p["rev_swell"], 198, 0, gain=0.46)
    exit_(s, p, 199, power=0.9, drop=False)

    # -------------------------------------------------------- 200-223 DROP 3
    log("DROP 3: the theme as a weapon")
    enter(s, p, 200, power=1.2, horn=True, siren=True)
    lay_kicks(s, p, 200, 224, key="kick_max", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 200, 224, density=16, gain=0.56, tips=True)
    lay_perc(s, p, 200, 224, gain=1.0, claps=True, rides=True)
    lay_metal(s, p, 200, 224, gain=0.6)
    lay_acid(s, p, 200, 224, ACID_C, gain=0.58, cutoff=760, env_mod=5000, res=0.91, drive=11)
    lay_hoover(s, p, 200, 216, gain=0.34, riff=HOOVER_RIFF2)
    # the lament, played by the machine
    s.place("solo", p["violin_dead"], 204, 0, gain=0.42)
    s.place("strings", p["strings_dead"], 204, 0, gain=0.26)
    s.place("solo", p["violin_dead"], 216, 0, gain=0.38)
    lay_lead_screech(s, p, 208, 216, gain=0.34)
    for b in range(200, 224, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.34, pan_=RNG.uniform(-0.3, 0.3))
        fill(s, p, b + 7, "metal")
    s.place("speech", p["sins_choir"], 212, 0, gain=0.5)
    s.place("fx", p["siren"], 215, 0, gain=0.26)
    s.place("lead", p["horn_hi"], 216, 0, gain=0.36)
    s.place("scream", far(p["help_panic"], 8000), 222, 10, gain=0.5, pan_=-0.25)
    exit_(s, p, 223, power=1.0)

    # --------------------------------------------------------- 224-233 outro
    log("outro")
    lay_kicks(s, p, 224, 230, key="kick", gain=0.95, ghosts=True, rolls=False)
    lay_hats(s, p, 224, 229, density=16, gain=0.42)
    lay_perc(s, p, 224, 229, gain=0.65, claps=True)
    lay_metal(s, p, 224, 231, gain=0.45)
    lay_acid(s, p, 224, 229, ACID_A, gain=0.36, cutoff=420, env_mod=2600, res=0.82, drive=5)
    s.place("metal", p["machine"], 228, 0, gain=0.7)
    s.place("speech", p["say_whisper"], 229, 0, gain=0.46, pan_=-0.3)
    s.place("solo", p["violin_a"], 230, 0, gain=0.95, pan_=-0.05)
    s.place("strings", p["strings_a"], 230, 0, gain=0.95)
    s.place("fx", p["impact"], 230, 0, gain=0.45)
    s.place("speech", p["forgive_close"], 232, 4, gain=0.55)
    s.place("breath", p["breath_out"], 232, 8, gain=0.42, pan_=0.3)

    return s, p


# ---------------------------------------------------------------- mixdown

def process_buses(s, p, verbose=True):
    """Everything up to the balance stage: sends, space, dirt, sidechain."""
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    n = s.n
    for nm in BUSES:
        s.bus(nm)

    log("rumble bus")
    s.buses["rumble"] = I.rumble_from(s.buses["sub"][0], rt60=2.0, cut=200.0, drive=2.8)

    log("arrangement automation")
    low_w = s.ramp(LOW_WEIGHT)
    s.buses["rumble"] *= low_w
    s.buses["sub"] *= low_w
    s.buses["kick"] *= s.ramp(KICK_GAIN)
    s.buses["kick"] = sweep(s.buses["kick"], "lp", s.ramp(KICK_TONE), 0.8, block=512)
    s.buses["kick"] = s.buses["kick"] * (0.55 + 0.45 * low_w) + \
        biquad(s.buses["kick"], "hp", 110, 0.7) * (1.0 - low_w) * 0.45

    log("distant kick")
    s.buses["kickfar"] = biquad(s.buses["kickfar"], "lp", 420, 0.9)
    s.buses["kickfar"] = send_reverb(s.buses["kickfar"], 0.55, rt60=3.0, damp=0.8,
                                     hp=90, seed=21)

    log("strings: a hall, then the machine room")
    # The section sits back in a long hall; the solo line stays in front of it,
    # which is the whole difference between accompaniment and melody.
    s.buses["strings"] = send_reverb(s.buses["strings"], 0.85, rt60=4.6, damp=0.72,
                                     hp=150, predelay=0.032, seed=141)
    s.buses["strings"] = widen(s.buses["strings"], 0.8, 21.0)
    s.buses["strings"] = biquad(s.buses["strings"], "hp", 90, 0.7)
    s.buses["strings"] = s.buses["strings"] - 0.18 * biquad(s.buses["strings"],
                                                            "bp", 330, 0.8)

    s.buses["solo"] = send_delay(s.buses["solo"], 0.16, s.step * 6, feedback=0.30,
                                 damp=4000)
    s.buses["solo"] = send_reverb(s.buses["solo"], 0.52, rt60=3.4, damp=0.65,
                                  hp=200, predelay=0.048, seed=151, width=0.8)
    s.buses["solo"] = widen(s.buses["solo"], 0.35, 12.0)
    s.buses["solo"] = biquad(s.buses["solo"], "hp", 170, 0.7)
    s.buses["solo"] = s.buses["solo"] + 0.16 * biquad(s.buses["solo"], "bp", 2600, 0.8)

    log("drums")
    s.buses["drums"] = drive_os(s.buses["drums"], 2.0, os=2)
    s.buses["drums"] = transient_shape(s.buses["drums"], attack=0.5, sustain=0.95)
    s.buses["drums"] = send_reverb(s.buses["drums"], 0.28, rt60=1.5, damp=0.6, hp=400,
                                   predelay=0.012, seed=31)
    s.buses["drums"] = widen(s.buses["drums"], 0.35, 9.0)

    log("industrial metal")
    s.buses["metal"] = drive_os(s.buses["metal"], 1.8, os=2)
    s.buses["metal"] = send_delay(s.buses["metal"], 0.18, s.step * 3, feedback=0.30,
                                  damp=6000)
    s.buses["metal"] = send_reverb(s.buses["metal"], 0.62, rt60=3.4, damp=0.62, hp=260,
                                   predelay=0.022, seed=111)
    s.buses["metal"] = widen(s.buses["metal"], 0.7, 17.0)
    s.buses["metal"] = biquad(s.buses["metal"], "hp", 180, 0.7)
    s.buses["metal"] = s.buses["metal"] + 0.30 * biquad(s.buses["metal"], "bp", 900, 0.7)
    s.buses["metal"] = s.buses["metal"] - 0.30 * biquad(s.buses["metal"], "hp", 11000, 0.7)

    log("bass")
    s.buses["bass"] = drive_os(s.buses["bass"], 1.8, os=2)
    s.buses["bass"] = send_delay(s.buses["bass"], 0.22, s.step * 3, feedback=0.30)
    s.buses["bass"] = send_reverb(s.buses["bass"], 0.16, rt60=1.4, damp=0.7, hp=300, seed=41)
    s.buses["bass"] = biquad(s.buses["bass"], "hp", 60, 0.7)

    log("lead")
    s.buses["lead"] = drive_os(s.buses["lead"], 2.2, os=2)
    s.buses["lead"] = send_delay(s.buses["lead"], 0.34, s.step * 3, feedback=0.42, damp=4200)
    s.buses["lead"] = send_reverb(s.buses["lead"], 0.55, rt60=2.8, damp=0.5, hp=240,
                                  predelay=0.026, seed=51)
    s.buses["lead"] = widen(s.buses["lead"], 0.55, 13.0)
    s.buses["lead"] = biquad(s.buses["lead"], "hp", 170, 0.7)

    log("voices, screams, speech, breath")
    s.buses["voice"] = send_delay(s.buses["voice"], 0.30, s.step * 6, feedback=0.40, damp=3200)
    s.buses["voice"] = send_reverb(s.buses["voice"], 0.85, rt60=3.6, damp=0.62, hp=180,
                                   predelay=0.034, seed=61)
    s.buses["voice"] = widen(s.buses["voice"], 0.5, 15.0)
    s.buses["voice"] = biquad(s.buses["voice"], "hp", 130, 0.7)
    s.buses["voice"] = drive_os(s.buses["voice"], 1.5, os=2)

    s.buses["scream"] = drive_os(s.buses["scream"], 1.6, os=2)
    s.buses["scream"] = send_delay(s.buses["scream"], 0.34, s.step * 6, feedback=0.46,
                                   damp=2800)
    s.buses["scream"] = send_reverb(s.buses["scream"], 1.15, rt60=4.4, damp=0.6, hp=210,
                                    predelay=0.045, seed=101)
    s.buses["scream"] = widen(s.buses["scream"], 0.45, 15.0)
    s.buses["scream"] = biquad(s.buses["scream"], "hp", 160, 0.7)

    # Long predelay keeps the words in front of the reverb instead of inside it.
    s.buses["speech"] = send_delay(s.buses["speech"], 0.20, s.step * 6, feedback=0.34,
                                   damp=3600)
    s.buses["speech"] = send_reverb(s.buses["speech"], 0.62, rt60=5.0, damp=0.68, hp=190,
                                    predelay=0.075, seed=121, width=0.85)
    s.buses["speech"] = widen(s.buses["speech"], 0.3, 11.0)
    s.buses["speech"] = biquad(s.buses["speech"], "hp", 105, 0.7)
    s.buses["speech"] = s.buses["speech"] + 0.18 * biquad(s.buses["speech"], "bp", 2400, 0.7)

    s.buses["breath"] = send_reverb(s.buses["breath"], 0.22, rt60=1.8, damp=0.72, hp=300,
                                    predelay=0.018, seed=131)
    s.buses["breath"] = widen(s.buses["breath"], 0.85, 23.0)
    s.buses["breath"] = biquad(s.buses["breath"], "hp", 220, 0.7)
    s.buses["breath"] = s.buses["breath"] + 0.12 * biquad(s.buses["breath"], "hp", 6500, 0.7)

    s.buses["fx"] = send_reverb(s.buses["fx"], 0.55, rt60=3.2, damp=0.55, hp=160, seed=71)
    s.buses["fx"] = widen(s.buses["fx"], 0.6, 17.0)

    s.buses["pad"] = send_reverb(s.buses["pad"], 0.9, rt60=4.2, damp=0.7, hp=200,
                                 predelay=0.04, seed=81)
    s.buses["pad"] = widen(s.buses["pad"], 0.7, 19.0)

    s.buses["air"] = send_reverb(s.buses["air"], 0.6, rt60=3.4, damp=0.8, hp=80, seed=91)

    log("intro filter")
    i1 = s.i(32)
    open_env = np.full(n, 18000.0)          # wide open after the intro
    open_env[:i1] = np.interp(np.arange(i1), [0, s.i(24), s.i(30), i1],
                              [700.0, 1400.0, 4000.0, 18000.0])
    for nm in ("kickfar", "drums", "air"):
        s.buses[nm] = sweep(s.buses[nm], "lp", open_env, 0.8, block=512)

    log("sidechain")
    deep = s.duck_envelope(depth=0.88, attack=0.003, hold=0.03, release=0.155)
    mid = s.duck_envelope(depth=0.62, attack=0.004, hold=0.02, release=0.13)
    light = s.duck_envelope(depth=0.34, attack=0.005, hold=0.012, release=0.10)
    s.apply_duck(["rumble", "sub"], deep)
    s.apply_duck(["bass", "pad", "metal", "strings"], mid)
    s.apply_duck(["lead", "voice", "scream", "speech", "breath", "solo", "fx",
                  "drums", "air"], light)
    return s.buses


def finalize(buses, verbose=True):
    if verbose:
        print("  balancing buses", flush=True)
    n = next(iter(buses.values())).shape[1]
    mix = np.zeros((2, n))
    for nm, g in balance(buses, TARGETS, reference="kick", verbose=verbose).items():
        mix += g * buses[nm]
    if verbose:
        print("  master chain", flush=True)
    return master(mix, headroom_db=-0.8, verbose=verbose)


def mixdown(s, p, verbose=True):
    return finalize(process_buses(s, p, verbose=verbose), verbose=verbose)
