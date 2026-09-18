"""The arrangement: 'CONCRETE CATHEDRAL' - 150 BPM, F# minor.

Dark industrial hard techno. A string lament, then the machine.

    bars           section
    000-015        lament
    016-031        the turn           strings fed through the machine
    032-047        groove             kick, metal, no melody
    048-063        build 1
    064-071        tension
    072-103        DROP 1
    104-111        interlude
    112-127        build 2
    128-159        DROP 2
    160-175        breakdown          the violin returns
    176-183        build 3
    184-215        DROP 3
    216-223        outro

Discipline over variety: each drop runs one locked pattern, changes on the
8, and takes a fill on the bar before the change. Nothing happens once.
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
BARS = 224
RNG = np.random.default_rng(2024)

BUSES = ["kick", "kickfar", "sub", "rumble", "drums", "metal", "texture", "bass",
         "lead", "speech", "strings", "solo", "fx", "pad", "air"]

TARGETS = {
    "kick":    (0.0, "low"),
    "kickfar": (-11.0, "low"),
    "sub":     (-7.5, "low"),
    "rumble":  (-8.0, "low"),
    "drums":   (1.0, "mid"),
    "metal":   (0.5, "mid"),
    "texture": (-5.5, "mid"),
    "bass":    (-0.5, "mid"),
    "lead":    (3.0, "mid"),
    "speech":  (1.0, "mid"),
    "strings": (6.5, "mid"),
    "solo":    (9.0, "mid"),
    "fx":      (-1.5, "mid"),
    "pad":     (-5.0, "mid"),
    "air":     (-17.0, "mid"),
}

LOW_WEIGHT = [(0, 0.02), (13.9, 0.04), (15, 0.30), (16, 0.74), (31.9, 0.76),
              (32, 0.58), (47.9, 0.62), (48, 0.56), (56, 0.62), (63.9, 0.66),
              (64, 0.40), (71.9, 0.24), (72, 1.0),
              (103.9, 1.0), (104, 0.30), (112, 0.52), (120, 0.62), (127.9, 0.66),
              (128, 1.0), (159.9, 1.0), (160, 0.05), (168, 0.20), (176, 0.50),
              (183.9, 0.60), (184, 1.0), (215.9, 1.0), (216, 0.76), (220, 0.52),
              (224, 0.14)]

KICK_GAIN = [(0, 0.35), (15, 0.5), (16, 0.94), (31.9, 0.94), (32, 0.84),
             (47.9, 0.86), (48, 0.82), (56, 0.88), (63.9, 0.92), (64, 0.92),
             (71.9, 0.92), (72, 1.0), (103.9, 1.0),
             (104, 0.62), (112, 0.82), (120, 0.90), (127.9, 0.92), (128, 1.0),
             (159.9, 1.0), (160, 0.5), (168, 0.72), (176, 0.88), (183.9, 0.92),
             (184, 1.0), (215.9, 1.0), (216, 0.95), (221, 0.8), (224, 0.4)]

KICK_TONE = [(0, 1200), (15, 3500), (16, 12000), (31.9, 12000), (32, 8000),
             (47.9, 9000), (48, 6500), (56, 7500), (63.9, 8000), (64, 6000),
             (71.9, 6000), (72, 20000), (103.9, 20000),
             (104, 2600), (112, 5600), (120, 8000), (127.9, 9500), (128, 20000),
             (159.9, 20000), (160, 2400), (168, 4400), (176, 6500), (183.9, 7500),
             (184, 20000), (219, 20000), (222, 7000), (224, 1800)]


# ------------------------------------------------------------- the lament

# Written in half-time so it breathes at 150 BPM. The falling semitone pairs
# (D-C#, E-D) are the sigh figure; the second phrase climbs instead of falling,
# which is what turns grief into panic.
CRY_A = [(78, 3), (76, 1), (74, 4), (73, 2), (74, 2), (73, 2)]
CRY_B = [(73, 2), (74, 2), (76, 3), (74, 1), (78, 3), (76, 1), (81, 2)]

# how the solo line is played: slides between notes, vibrato that widens on a
# held note, a leaning bow, and a catch in the tone
CRY = dict(porta=0.22, swell=0.55, sob=0.45, vib_growth=1.35, strain=0.8,
           vib_depth=0.0082, vib_rate=6.0, vib_delay=0.16, bow_noise=0.13,
           attack=0.10, release=0.38)

LAMENT_CHORDS = [([42, 54, 57, 61], 7), ([38, 50, 57, 62], 7),
                 ([45, 52, 57, 64], 7), ([40, 52, 56, 64], 7)]

LAMENT_BASS = [(42, 7), (40, 7), (38, 7), (37, 7)]

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
    p["impact"] = I.impact(3.0, tune=44)
    p["rev_swell"] = I.reverse_swell(3.2, seed=9)
    p["sub_drop"] = T.sub_drop(3.4, 120, 24)
    p["noise_fall"] = T.noise_fall(2.6, 9000, 220, seed=3)

    # --- speech
    sp = T.speech_layers("assets/confession.wav", shift=1.055, seed=1)
    p["say_close"], p["say_whisper"] = sp["close"], sp["whisper"]
    p["say_choir"], p["say_radio"] = sp["choir"], sp["radio"]
    fg = T.speech_layers("assets/forgive.wav", shift=1.08, seed=2)
    p["forgive_close"], p["forgive_whisper"] = fg["close"], fg["whisper"]
    sn = T.speech_layers("assets/sins.wav", shift=1.02, seed=3)
    p["sins_close"], p["sins_choir"] = sn["close"], sn["choir"]

    # --- industrial textures
    p["drone_lo"] = T.drone(25.6, 41.0, drive=5.0, seed=1)
    p["drone_mid"] = T.drone(25.6, 61.5, drive=6.0, seed=2, movement=0.07)
    p["bed_a"] = T.noise_bed(25.6, seed=1, lo=160, hi=4200, grit=0.35)
    p["bed_b"] = T.noise_bed(25.6, seed=2, lo=500, hi=9000, motion=0.07, grit=0.5)
    p["rust_a"] = T.rust(6.4, BPM, seed=1, density=0.45)
    p["rust_b"] = T.rust(6.4, BPM, seed=2, density=0.65)
    p["boom"] = T.boom(4.2, 42, seed=1, drive=6.0, size=1.0)
    p["boom2"] = T.boom(5.0, 39, seed=2, drive=7.0, size=1.15)
    p["hammer"] = T.hammer(1.8, 98, seed=1)
    p["hammer_lo"] = T.hammer(2.4, 66, seed=2, drive=8)
    p["feedback"] = T.feedback_tone(2.4, 1850, seed=1)
    p["feedback_hi"] = T.feedback_tone(1.6, 3100, seed=2, drive=9)
    p["riser_n"] = T.noise_riser(6.4, seed=1)
    p["riser_n2"] = T.noise_riser(12.8, seed=2, f_lo=220, f_hi=8000)
    p["riser_s"] = T.noise_riser(3.2, seed=3, f_lo=400, f_hi=9000)

    # --- rave
    p["horn"] = T.war_horn(42, 3.0, drive=7, growl=0.5)
    p["horn_hi"] = T.war_horn(49, 2.4, drive=8, growl=0.6)

    # --- the lament
    p["violin_a"] = ST.phrase(CRY_A, BPM, beat_unit=2.0, vel=0.74, seed=1,
                              legato=1.28, rubato=0.55, **CRY)
    p["violin_b"] = ST.phrase(CRY_B, BPM, beat_unit=2.0, vel=0.86, seed=2,
                              legato=1.28, rubato=0.5, **{**CRY, "sob": 0.55,
                                                          "strain": 1.0})
    # an octave below, doubling the second phrase so the climax has weight
    p["violin_b_low"] = ST.phrase([(m - 12 if m else None, b) for m, b in CRY_B],
                                  BPM, beat_unit=2.0, vel=0.62, seed=9,
                                  legato=1.28, rubato=0.5, **CRY)
    p["violin_a2"] = ST.phrase(CRY_A, BPM, beat_unit=2.0, vel=0.84, seed=5,
                               legato=1.28, rubato=0.5, bright=1.15, **CRY)
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
    elif kind == "feedback":
        s.place("fx", p["feedback_hi"], b, 8, gain=0.26, pan_=RNG.uniform(-0.3, 0.3))
    elif kind == "steam":
        s.place("metal", p["steam_s"], b, 12, gain=0.4, pan_=RNG.uniform(-0.5, 0.5))


# ---------------------------------------------------------------- sections



# ---------------------------------------------------------- transition kit

def approach(s, p, bar, power=1.0, bars=4):
    """Lead into `bar` with noise and metal. No pitched swoops."""
    s.place("fx", p["riser_n"], bar - bars, 0, gain=0.40 * power)
    s.place("metal", p["steam"], bar - 1, 8, gain=0.34 * power)
    s.place("texture", p["rust_b"], bar - 2, 0, gain=0.5 * power)


def enter(s, p, bar, power=1.0, horn=False):
    """Mark a section start: something heavy hits."""
    s.place("fx", p["impact"], bar, 0, gain=0.62 * power)
    s.place("metal", p["hammer"], bar, 0, gain=0.66 * power)
    s.place("metal", p["clang"], bar, 0, gain=0.42 * power)
    if horn:
        s.place("lead", p["horn"], bar, 0, gain=0.30 * power)


def exit_(s, p, bar, power=1.0):
    s.place("fx", p["sub_drop"], bar, 12, gain=0.32 * power)


def drop_core(s, p, b0, b1, kick_key="kick", acid=None, acid_kw=None, riff=None,
              rides=False, screech_lead=None, hoover_gain=0.34, metal_gain=0.55):
    """One locked pattern. Every drop is built from this so they are the same
    thing at different intensities, which is what makes a set feel composed."""
    acid_kw = acid_kw or {}
    lay_kicks(s, p, b0, b1, key=kick_key, gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, b0, b1, density=16, gain=0.52, tips=True)
    lay_perc(s, p, b0, b1, gain=1.0, claps=True, rides=rides)
    lay_metal(s, p, b0, b1, gain=metal_gain)
    lay_hoover(s, p, b0, b1, gain=hoover_gain, riff=riff or HOOVER_RIFF)
    half = b0 + (b1 - b0) // 2
    lay_acid(s, p, b0, half, acid[0], gain=0.52, **acid_kw)
    lay_acid(s, p, half, b1, acid[1], gain=0.54, **acid_kw)
    if screech_lead:
        lay_lead_screech(s, p, screech_lead[0], screech_lead[1], gain=0.32)
    for b in range(b0, b1, 4):
        s.place("texture", p["rust_a" if (b // 4) % 2 else "rust_b"], b, 0, gain=0.42)
    # a fill on the bar before every eight, and nothing else one-off
    for b in range(b0 + 7, b1, 8):
        fill(s, p, b, "metal")
        s.place("lead", p["scr_up"], b, 8, gain=0.28, pan_=RNG.uniform(-0.25, 0.25))
    for b in range(b0 + 15, b1, 16):
        s.place("metal", p["hammer_lo"], b, 12, gain=0.40)


# ---------------------------------------------------------------- sections

def build(verbose=True):
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    log("rendering sound palette...")
    p = build_palette()
    s = Session(BPM, BARS, tail=8.0)

    ACID_KW = dict(cutoff=620, env_mod=4200, res=0.87, drive=8)

    # room and machine under the whole record
    s.place("air", I.room_tone(s.dur, level=0.07), 0, 0, gain=1.0)
    for b in range(0, BARS, 16):
        s.place("texture", p["bed_a" if (b // 16) % 2 else "bed_b"], b, 0, gain=0.34)

    # ------------------------------------------------- 000-014 lament (crying)
    log("lament")
    s.place("strings", p["strings_a"], 0, 0, gain=1.45)
    s.place("strings", p["cello"], 0, 0, gain=1.15)
    s.place("solo", p["violin_a"], 0, 0, gain=1.00, pan_=-0.06)
    s.place("speech", p["say_close"], 3, 4, gain=0.78)
    s.place("speech", p["say_whisper"], 3, 4.3, gain=0.32, pan_=-0.5)
    s.place("texture", p["drone_lo"], 5, 0, gain=0.42)
    # second phrase climbs instead of falling, doubled an octave down
    s.place("strings", p["strings_a"], 7, 0, gain=1.55)
    s.place("strings", p["cello"], 7, 0, gain=1.20)
    s.place("solo", p["violin_b"], 7, 0, gain=1.05, pan_=-0.04)
    s.place("solo", p["violin_b_low"], 7, 0, gain=0.48, pan_=0.12)
    s.place("metal", p["machine_lo"], 9, 0, gain=0.38)
    s.place("fx", p["riser_n2"], 8, 0, gain=0.26)
    # the room inhales before the hit
    s.place("fx", T.reverse_tail(p["boom"], 3.0), 13, 0, gain=0.46)
    s.place("metal", p["steam"], 14, 8, gain=0.36)

    # ------------------------------------------------------ 015 the two hits
    log("BOOM BOOM")
    s.place("fx", p["boom"], 15, 0, gain=1.00)
    s.place("metal", p["clang"], 15, 0, gain=0.55)
    s.place("fx", p["boom2"], 15, 8, gain=1.00)
    s.place("metal", p["hammer_lo"], 15, 8, gain=0.60)

    # ---------------------------------------------------- 016-031 the machine
    log("the machine")
    enter(s, p, 16, power=1.2, horn=True)
    lay_kicks(s, p, 16, 32, key="kick_hard", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 16, 32, density=16, gain=0.50, tips=True)
    lay_perc(s, p, 16, 32, gain=0.9, claps=True)
    lay_metal(s, p, 16, 32, gain=0.90)
    for b in range(16, 32, 4):
        s.place("metal", p["conveyor"], b, 0, gain=0.58)
        s.place("texture", p["rust_b"], b, 0, gain=0.44)
    # the lament, already coming apart, rides over the top of it
    for i, bb in enumerate((16, 20, 24, 28)):
        s.place("strings", p[f"strings_rot{i}"], bb, 0, gain=0.58 + 0.13 * i)
    s.place("solo", ST.desecrate(p["violin_b"], 0.85, 2800, 7), 24, 0, gain=0.52)
    s.place("metal", p["machine"], 16, 0, gain=0.72)
    s.place("texture", p["drone_lo"], 16, 0, gain=0.52)
    for b in range(23, 32, 8):
        fill(s, p, b, "metal")
        s.place("lead", p["scr_up"], b, 8, gain=0.26, pan_=RNG.uniform(-0.25, 0.25))
    s.place("metal", p["hammer_lo"], 31, 12, gain=0.42)

    # --------------------------------------------------------- 032-047 groove
    log("groove")
    lay_kicks(s, p, 32, 48, key="kick", gain=0.96, ghosts=True, rolls=False)
    lay_hats(s, p, 32, 40, density=8, gain=0.40)
    lay_hats(s, p, 40, 48, density=16, gain=0.46, tips=True)
    lay_perc(s, p, 32, 48, gain=0.75, claps=True)
    lay_metal(s, p, 32, 48, gain=0.85)
    lay_acid(s, p, 36, 48, ACID_A, gain=0.42, cutoff=420, env_mod=2800, res=0.82, drive=6)
    for b in range(32, 48, 4):
        s.place("metal", p["conveyor"], b, 0, gain=0.50)
        s.place("texture", p["rust_a"], b, 0, gain=0.40)
    s.place("metal", p["machine"], 40, 0, gain=0.62)
    s.place("texture", p["drone_mid"], 40, 0, gain=0.42)
    for b in (39, 47):
        fill(s, p, b, "metal")

    # -------------------------------------------------------- 048-063 build 1
    log("build 1")
    lay_kicks(s, p, 48, 64, key="kick", gain=0.96, ghosts=True, rolls=False)
    lay_hats(s, p, 48, 56, density=8, gain=0.42)
    lay_hats(s, p, 56, 64, density=16, gain=0.50, tips=True)
    lay_perc(s, p, 48, 64, gain=0.85)
    lay_metal(s, p, 48, 64, gain=0.6)
    lay_acid(s, p, 52, 64, ACID_A, gain=0.42, cutoff=420, env_mod=2800, res=0.82, drive=6)
    for b in range(48, 64, 4):
        s.place("texture", p["rust_b"], b, 0, gain=0.42)
    s.place("texture", p["drone_mid"], 56, 0, gain=0.40)
    for b in (55, 63):
        fill(s, p, b, "tom")

    # ------------------------------------------------------- 064-071 tension
    log("tension")
    lay_kicks(s, p, 64, 68, key="kick", gain=1.0, ghosts=True, rolls=False)
    lay_hats(s, p, 64, 68, density=16, gain=0.50)
    lay_perc(s, p, 64, 68, gain=0.8)
    lay_acid(s, p, 64, 70, ACID_A, gain=0.48, cutoff=560, env_mod=3600, res=0.85, drive=7)
    s.place("fx", p["riser_n"], 68, 0, gain=0.46)
    s.place("fx", p["riser_n2"], 64, 0, gain=0.30)
    snare_roll(s, p, 68, bars=3, gain=0.64)
    s.place("metal", p["steam"], 70, 0, gain=0.40)
    s.place("fx", p["feedback"], 70, 8, gain=0.28)
    exit_(s, p, 71, power=0.7)

    # ------------------------------------------------------- 072-103 DROP 1
    log("DROP 1")
    enter(s, p, 72, power=1.0, horn=True)
    drop_core(s, p, 72, 104, "kick", acid=(ACID_A, ACID_B), acid_kw=ACID_KW,
              hoover_gain=0.34, metal_gain=0.5)
    s.place("speech", p["sins_choir"], 88, 0, gain=0.40)
    s.place("texture", p["drone_lo"], 88, 0, gain=0.34)
    exit_(s, p, 103, power=1.0)

    # ---------------------------------------------------- 104-111 interlude
    log("interlude")
    s.place("drums", tape_stop(p["groove_bar"], start=0.12, end_ratio=0.035,
                               curve=1.7, max_stretch=2.6), 104, 0, gain=0.8)
    s.place("fx", p["sub_drop"], 104, 0, gain=0.55)
    s.place("texture", p["drone_lo"], 104, 0, gain=0.40)
    s.place("metal", p["machine"], 104, 0, gain=0.44)
    s.place("metal", p["steam"], 105, 6, gain=0.40, pan_=-0.35)
    for b in range(106, 112):
        for st in (0, 8):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.44)
            s.mark_kick(b, st)
        s.place("metal", p["metal_b"], b, 6, gain=0.34, pan_=RNG.uniform(-0.5, 0.5))
        s.place("texture", p["rust_a"], b, 0, gain=0.30)
    s.place("fx", p["feedback"], 109, 0, gain=0.26, pan_=-0.2)
    s.place("speech", p["forgive_close"], 110, 0, gain=0.52)
    approach(s, p, 112, power=0.8)

    # -------------------------------------------------------- 112-127 build 2
    log("build 2")
    lay_kicks(s, p, 112, 124, key="kick", gain=0.98, ghosts=True, rolls=False)
    lay_hats(s, p, 112, 120, density=8, gain=0.42)
    lay_hats(s, p, 120, 128, density=16, gain=0.52, tips=True)
    lay_perc(s, p, 112, 128, gain=0.9)
    lay_metal(s, p, 112, 128, gain=0.6)
    lay_acid(s, p, 112, 128, ACID_B, gain=0.48, cutoff=480, env_mod=3900, res=0.86, drive=7)
    lay_hoover(s, p, 120, 124, gain=0.26, stabs=False)
    for b in range(112, 128, 4):
        s.place("texture", p["rust_b"], b, 0, gain=0.44)
    s.place("fx", p["riser_n2"], 120, 0, gain=0.36)
    s.place("fx", p["riser_n"], 124, 0, gain=0.48)
    snare_roll(s, p, 124, bars=3, gain=0.72)
    s.place("metal", p["steam"], 126, 0, gain=0.40)
    exit_(s, p, 127, power=0.8)

    # ------------------------------------------------------- 128-159 DROP 2
    log("DROP 2")
    enter(s, p, 128, power=1.1, horn=True)
    drop_core(s, p, 128, 160, "kick_hard", acid=(ACID_B, ACID_C), acid_kw=ACID_KW,
              riff=HOOVER_RIFF2, rides=True, screech_lead=(136, 152),
              hoover_gain=0.34, metal_gain=0.58)
    s.place("speech", p["sins_choir"], 144, 0, gain=0.44)
    s.place("lead", p["horn_hi"], 152, 0, gain=0.30)
    s.place("texture", p["drone_mid"], 144, 0, gain=0.34)
    exit_(s, p, 159, power=1.0)

    # ----------------------------------------------------- 160-175 breakdown
    log("breakdown")
    s.place("pad", p["pad_dark"], 160, 0, gain=0.40)
    s.place("strings", p["strings_b"], 160, 0, gain=1.40)
    s.place("strings", p["cello"], 160, 0, gain=1.10)
    s.place("solo", p["violin_a2"], 162, 0, gain=1.0, pan_=-0.05)
    s.place("texture", p["drone_lo"], 160, 0, gain=0.30)
    s.place("metal", p["machine_lo"], 160, 0, gain=0.28)
    s.place("speech", p["say_choir"], 168, 0, gain=0.46)
    s.place("strings", p["strings_b"], 168, 0, gain=1.25)
    for b in range(170, 176):
        for st in (0, 8):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.40 + 0.035 * (b - 170))
            s.mark_kick(b, st)
        s.place("texture", p["rust_a"], b, 0, gain=0.34)
    approach(s, p, 176, power=0.9)

    # -------------------------------------------------------- 176-183 build 3
    log("build 3")
    lay_kicks(s, p, 176, 182, key="kick", gain=0.96, ghosts=True, rolls=False)
    lay_hats(s, p, 176, 184, density=16, gain=0.50, tips=True)
    lay_perc(s, p, 176, 184, gain=0.85)
    lay_metal(s, p, 176, 184, gain=0.6)
    lay_acid(s, p, 176, 183, ACID_C, gain=0.52, cutoff=620, env_mod=4300, res=0.88, drive=8)
    s.place("fx", p["riser_n"], 180, 0, gain=0.50)
    snare_roll(s, p, 180, bars=3, gain=0.78)
    s.place("metal", p["steam"], 182, 0, gain=0.42)
    s.place("fx", p["feedback_hi"], 182, 8, gain=0.26)
    exit_(s, p, 183, power=0.9)

    # -------------------------------------------------------- 184-215 DROP 3
    log("DROP 3")
    enter(s, p, 184, power=1.2, horn=True)
    drop_core(s, p, 184, 216, "kick_max", acid=(ACID_C, ACID_C), acid_kw=ACID_KW,
              riff=HOOVER_RIFF2, rides=True, screech_lead=(192, 208),
              hoover_gain=0.36, metal_gain=0.62)
    # the lament, played by the machine
    s.place("solo", p["violin_dead"], 188, 0, gain=0.40)
    s.place("strings", p["strings_dead"], 188, 0, gain=0.24)
    s.place("solo", p["violin_dead"], 204, 0, gain=0.36)
    s.place("speech", p["sins_choir"], 200, 0, gain=0.44)
    s.place("lead", p["horn_hi"], 208, 0, gain=0.32)
    exit_(s, p, 215, power=1.0)

    # --------------------------------------------------------- 216-223 outro
    log("outro")
    lay_kicks(s, p, 216, 221, key="kick", gain=0.95, ghosts=True, rolls=False)
    lay_hats(s, p, 216, 220, density=16, gain=0.42)
    lay_perc(s, p, 216, 220, gain=0.6, claps=True)
    lay_metal(s, p, 216, 222, gain=0.5)
    lay_acid(s, p, 216, 220, ACID_A, gain=0.36, cutoff=440, env_mod=2700, res=0.82, drive=5)
    s.place("metal", p["machine"], 218, 0, gain=0.70)
    s.place("texture", p["drone_lo"], 216, 0, gain=0.55)
    s.place("fx", p["impact"], 220, 0, gain=0.45)
    s.place("solo", p["violin_a"], 220, 0, gain=0.90, pan_=-0.05)
    s.place("strings", p["strings_a"], 220, 0, gain=0.85)
    s.place("speech", p["forgive_close"], 222, 4, gain=0.50)

    return s, p


# ---------------------------------------------------------------- mixdown

def process_buses(s, p, verbose=True):
    """Sends, space, dirt, sidechain.

    Everything is drier than a pop mix on purpose. Hard techno lives close to
    the speaker; long tails on every bus is what makes a busy arrangement turn
    to mud. Only the rumble and the strings get a real room.
    """
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    n = s.n
    for nm in BUSES:
        s.bus(nm)

    log("rumble")
    s.buses["rumble"] = I.rumble_from(s.buses["sub"][0], rt60=2.0, cut=200.0, drive=2.8)

    log("automation")
    low_w = s.ramp(LOW_WEIGHT)
    s.buses["rumble"] *= low_w
    s.buses["sub"] *= low_w
    s.buses["kick"] *= s.ramp(KICK_GAIN)
    s.buses["kick"] = sweep(s.buses["kick"], "lp", s.ramp(KICK_TONE), 0.8, block=512)
    s.buses["kick"] = s.buses["kick"] * (0.55 + 0.45 * low_w) + \
        biquad(s.buses["kick"], "hp", 110, 0.7) * (1.0 - low_w) * 0.45

    s.buses["kickfar"] = biquad(s.buses["kickfar"], "lp", 420, 0.9)
    s.buses["kickfar"] = send_reverb(s.buses["kickfar"], 0.42, rt60=2.6, damp=0.82,
                                     hp=90, seed=21)

    log("strings")
    s.buses["strings"] = send_reverb(s.buses["strings"], 0.60, rt60=3.8, damp=0.75,
                                     hp=150, predelay=0.030, seed=141)
    s.buses["strings"] = widen(s.buses["strings"], 0.7, 18.0)
    s.buses["strings"] = biquad(s.buses["strings"], "hp", 95, 0.7)
    s.buses["strings"] = s.buses["strings"] - 0.18 * biquad(s.buses["strings"],
                                                            "bp", 330, 0.8)
    s.buses["solo"] = send_reverb(s.buses["solo"], 0.34, rt60=2.8, damp=0.70,
                                  hp=200, predelay=0.042, seed=151, width=0.8)
    s.buses["solo"] = widen(s.buses["solo"], 0.30, 11.0)
    s.buses["solo"] = biquad(s.buses["solo"], "hp", 175, 0.7)
    s.buses["solo"] = s.buses["solo"] + 0.16 * biquad(s.buses["solo"], "bp", 2600, 0.8)

    log("drums")
    s.buses["drums"] = drive_os(s.buses["drums"], 2.1, os=2)
    s.buses["drums"] = transient_shape(s.buses["drums"], attack=0.6, sustain=0.9)
    s.buses["drums"] = send_reverb(s.buses["drums"], 0.14, rt60=1.1, damp=0.68, hp=500,
                                   predelay=0.008, seed=31)
    s.buses["drums"] = widen(s.buses["drums"], 0.30, 8.0)

    log("metal")
    s.buses["metal"] = drive_os(s.buses["metal"], 2.0, os=2)
    s.buses["metal"] = send_delay(s.buses["metal"], 0.14, s.step * 3, feedback=0.26,
                                  damp=5200)
    s.buses["metal"] = send_reverb(s.buses["metal"], 0.30, rt60=2.2, damp=0.70, hp=300,
                                   predelay=0.016, seed=111)
    s.buses["metal"] = widen(s.buses["metal"], 0.55, 14.0)
    s.buses["metal"] = biquad(s.buses["metal"], "hp", 190, 0.7)
    s.buses["metal"] = s.buses["metal"] + 0.28 * biquad(s.buses["metal"], "bp", 900, 0.7)
    s.buses["metal"] = s.buses["metal"] - 0.30 * biquad(s.buses["metal"], "hp", 11000, 0.7)

    log("texture")
    s.buses["texture"] = drive_os(s.buses["texture"], 1.6, os=2)
    s.buses["texture"] = send_reverb(s.buses["texture"], 0.24, rt60=2.4, damp=0.78,
                                     hp=200, seed=161)
    s.buses["texture"] = widen(s.buses["texture"], 0.75, 20.0)
    s.buses["texture"] = biquad(s.buses["texture"], "hp", 60, 0.7)

    log("bass and lead")
    s.buses["bass"] = drive_os(s.buses["bass"], 1.9, os=2)
    s.buses["bass"] = send_delay(s.buses["bass"], 0.16, s.step * 3, feedback=0.24)
    s.buses["bass"] = biquad(s.buses["bass"], "hp", 60, 0.7)

    s.buses["lead"] = drive_os(s.buses["lead"], 2.3, os=2)
    s.buses["lead"] = send_delay(s.buses["lead"], 0.24, s.step * 3, feedback=0.34, damp=4000)
    s.buses["lead"] = send_reverb(s.buses["lead"], 0.28, rt60=1.9, damp=0.60, hp=280,
                                  predelay=0.018, seed=51)
    s.buses["lead"] = widen(s.buses["lead"], 0.45, 11.0)
    s.buses["lead"] = biquad(s.buses["lead"], "hp", 175, 0.7)

    log("speech")
    s.buses["speech"] = send_reverb(s.buses["speech"], 0.44, rt60=4.0, damp=0.72, hp=190,
                                    predelay=0.070, seed=121, width=0.85)
    s.buses["speech"] = widen(s.buses["speech"], 0.28, 10.0)
    s.buses["speech"] = biquad(s.buses["speech"], "hp", 110, 0.7)
    s.buses["speech"] = s.buses["speech"] + 0.16 * biquad(s.buses["speech"], "bp", 2400, 0.7)

    s.buses["fx"] = send_reverb(s.buses["fx"], 0.32, rt60=2.4, damp=0.62, hp=180, seed=71)
    s.buses["fx"] = widen(s.buses["fx"], 0.5, 14.0)
    s.buses["pad"] = send_reverb(s.buses["pad"], 0.60, rt60=3.4, damp=0.74, hp=220,
                                 predelay=0.034, seed=81)
    s.buses["pad"] = widen(s.buses["pad"], 0.6, 16.0)
    s.buses["air"] = send_reverb(s.buses["air"], 0.45, rt60=2.8, damp=0.82, hp=80, seed=91)

    log("intro filter")
    i1 = s.i(32)
    open_env = np.full(n, 18000.0)
    open_env[:i1] = np.interp(np.arange(i1), [0, s.i(20), s.i(30), i1],
                              [600.0, 1300.0, 4000.0, 18000.0])
    for nm in ("kickfar", "drums", "air"):
        s.buses[nm] = sweep(s.buses[nm], "lp", open_env, 0.8, block=512)

    log("sidechain")
    deep = s.duck_envelope(depth=0.88, attack=0.003, hold=0.03, release=0.155)
    mid = s.duck_envelope(depth=0.62, attack=0.004, hold=0.02, release=0.13)
    light = s.duck_envelope(depth=0.34, attack=0.005, hold=0.012, release=0.10)
    s.apply_duck(["rumble", "sub"], deep)
    s.apply_duck(["bass", "pad", "metal", "strings", "texture"], mid)
    s.apply_duck(["lead", "speech", "solo", "fx", "drums", "air"], light)
    return s.buses


def finalize(buses, verbose=True):
    if verbose:
        print("  balancing", flush=True)
    n = next(iter(buses.values())).shape[1]
    mix = np.zeros((2, n))
    for nm, g in balance(buses, TARGETS, reference="kick", verbose=verbose).items():
        mix += g * buses[nm]
    if verbose:
        print("  master", flush=True)
    return master(mix, headroom_db=-0.8, verbose=verbose)


def mixdown(s, p, verbose=True):
    return finalize(process_buses(s, p, verbose=verbose), verbose=verbose)
