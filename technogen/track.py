"""The arrangement: 'CONCRETE CATHEDRAL' - 150 BPM hard techno, F# minor.

    bars           section
    000-015        intro / confession
    016-031        industrial groove
    032-047        build 1
    048-055        pre-drop
    056-087        DROP 1
    088-095        transition
    096-111        sultry mid-section
    112-127        build 2
    128-159        DROP 2
    160-171        breakdown
    172-179        build 3
    180-195        DROP 3
    196-207        outro
"""

import numpy as np

from .dsp import (SR, biquad, sweep, reverb, delay, widen, tanh_drive, drive_os,
                  soft_clip, normalize, env_curve, env_ar, noise, n_samples, db,
                  pitch_shift_naive, tape_stop, gate, fit, supersaw, transient_shape)
from . import instruments as I
from . import texture as T
from .mixer import Session, send_reverb, send_delay, master, balance

BPM = 150.0
BARS = 208
RNG = np.random.default_rng(2024)

BUSES = ["kick", "kickfar", "sub", "rumble", "drums", "metal", "bass", "lead",
         "voice", "scream", "speech", "breath", "fx", "pad", "air"]

# Where each bus sits relative to the kick, and the band it is measured in.
TARGETS = {
    "kick":    (0.0, "low"),
    "kickfar": (-11.0, "low"),
    "sub":     (-7.5, "low"),
    "rumble":  (-8.5, "low"),
    "drums":   (1.0, "mid"),
    "metal":   (1.5, "mid"),
    "bass":    (1.0, "mid"),
    "lead":    (5.0, "mid"),
    "voice":   (0.5, "mid"),
    "scream":  (-2.0, "mid"),
    "speech":  (2.0, "mid"),
    "breath":  (-7.0, "mid"),
    "fx":      (0.0, "mid"),
    "pad":     (-5.0, "mid"),
    "air":     (-16.0, "mid"),
}

# Automation in bars. Holding the low end back through a build is what makes
# the next drop land; without it more elements just means more noise.
LOW_WEIGHT = [(0, 0.10), (16, 0.32), (24, 0.40), (32, 0.36), (40, 0.48),
              (47.9, 0.56), (48, 0.32), (55.9, 0.20), (56, 1.0), (87.9, 1.0),
              (88, 0.10), (96, 0.40), (104, 0.48), (112, 0.52), (120, 0.62),
              (127.9, 0.66), (128, 1.0), (159.9, 1.0), (160, 0.10), (168, 0.26),
              (172, 0.46), (179.9, 0.56), (180, 1.0), (195.9, 1.0), (196, 0.72),
              (202, 0.56), (206, 0.30), (208, 0.12)]

KICK_GAIN = [(0, 0.5), (16, 0.66), (24, 0.74), (32, 0.72), (40, 0.82), (47.9, 0.86),
             (48, 0.88), (55.9, 0.88), (56, 1.0), (87.9, 1.0), (88, 0.5), (96, 0.74),
             (104, 0.80), (112, 0.84), (120, 0.90), (127.9, 0.92), (128, 1.0),
             (159.9, 1.0), (160, 0.6), (172, 0.84), (179.9, 0.9), (180, 1.0),
             (195.9, 1.0), (196, 0.94), (204, 0.8), (208, 0.4)]

KICK_TONE = [(0, 1500), (16, 1800), (24, 2600), (32, 3200), (40, 4200), (47.9, 4800),
             (48, 5200), (55.9, 5200), (56, 20000), (87.9, 20000), (88, 2200),
             (96, 6000), (112, 7000), (120, 9000), (127.9, 10000), (128, 20000),
             (159.9, 20000), (160, 2600), (172, 5000), (179.9, 6000), (180, 20000),
             (199, 20000), (204, 9000), (208, 2200)]


# ---------------------------------------------------------------- palette

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

def build(verbose=True):
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    log("rendering sound palette...")
    p = build_palette()
    s = Session(BPM, BARS, tail=7.0)

    # ------------------------------------------------ 000-015 intro/confession
    log("intro / confession")
    s.place("air", I.room_tone(s.dur, level=0.09), 0, 0, gain=1.0)
    s.place("metal", p["machine_lo"], 0, 0, gain=0.55)
    s.place("metal", far(p["machine"], 1400), 4, 0, gain=0.62)
    s.place("metal", far(p["machine"], 2200), 12, 0, gain=0.70)

    s.place("speech", p["say_close"], 1, 4, gain=0.88, pan_=0.0)
    s.place("speech", p["say_whisper"], 1, 4.35, gain=0.45, pan_=-0.55)
    s.place("speech", p["say_whisper"], 1, 3.7, gain=0.38, pan_=0.55)
    s.place("speech", far(p["say_radio"], 3200), 1, 4, gain=0.18, pan_=0.25)

    s.place("scream", far(p["help_far"], 1200), 6, 10, gain=0.34, pan_=-0.45)
    s.place("scream", far(p["cry"], 2100), 9, 6, gain=0.44, pan_=0.42)
    s.place("scream", far(p["help_mid"], 4200), 12, 8, gain=0.62, pan_=-0.18)
    s.place("scream", far(p["help_panic"], 6800), 14, 10, gain=0.70, pan_=0.22)

    s.place("metal", p["scrape_l"], 3, 12, gain=0.85, pan_=-0.5)
    s.place("metal", p["chain"], 5, 2, gain=0.80, pan_=0.45)
    s.place("metal", p["steam"], 7, 8, gain=0.90, pan_=-0.3)
    s.place("metal", p["clang"], 8, 0, gain=1.15, pan_=0.1)
    s.place("metal", p["chain"], 11, 6, gain=0.75, pan_=-0.4)
    s.place("metal", p["steam_s"], 13, 4, gain=0.85, pan_=0.5)
    for b in range(4, 16):
        for st in range(0, 16, 8 if b < 8 else 4):
            s.place("kickfar", p["kick_soft"], b, st,
                    gain=min(0.55, 0.16 + 0.035 * (b - 4)))
            s.mark_kick(b, st)
    s.place("fx", p["rev_swell"], 14, 8, gain=0.26)
    s.place("breath", p["breath_in"], 15, 12, gain=0.5, pan_=0.2)

    # ----------------------------------------------- 016-031 industrial groove
    log("industrial groove")
    lay_kicks(s, p, 16, 32, key="kick", gain=0.86, ghosts=True, rolls=False)
    for b in range(16, 32, 4):
        s.place("metal", p["conveyor"], b, 0, gain=0.62)
    lay_metal(s, p, 16, 32, gain=0.95)
    lay_hats(s, p, 16, 24, density=4, open_off=False, gain=0.30)
    lay_hats(s, p, 24, 32, density=8, gain=0.38)
    lay_perc(s, p, 20, 32, gain=0.6, claps=True)
    s.place("metal", p["machine"], 16, 0, gain=0.75)
    s.place("breath", p["breath_out"], 19, 12, gain=0.45, pan_=-0.3)
    s.place("voice", p["whisper"], 22, 0, gain=0.22, pan_=0.45)
    s.place("speech", p["forgive_whisper"], 27, 8, gain=0.5, pan_=-0.35)
    s.place("breath", p["sigh_a"], 29, 4, gain=0.42, pan_=0.3)
    for b in (23, 31):
        fill(s, p, b, "metal")

    # ------------------------------------------------------- 032-047 build 1
    log("build 1")
    lay_kicks(s, p, 32, 48, key="kick", gain=0.94, ghosts=True, rolls=False)
    lay_hats(s, p, 32, 40, density=8, gain=0.40)
    lay_hats(s, p, 40, 48, density=16, gain=0.48, tips=True)
    lay_perc(s, p, 32, 48, gain=0.8)
    lay_metal(s, p, 32, 48, gain=0.45)
    lay_acid(s, p, 36, 48, ACID_A, gain=0.40, cutoff=400, env_mod=2600, res=0.80, drive=5)
    s.place("voice", p["vstab_low"], 39, 12, gain=0.3, pan_=-0.2)
    s.place("fx", p["riser_n2"], 40, 0, gain=0.26)
    s.place("lead", p["scr_up"], 46, 8, gain=0.26, pan_=-0.25)
    for b in (39, 47):
        fill(s, p, b, "tom")

    # ----------------------------------------------------- 048-055 pre-drop
    log("pre-drop")
    lay_kicks(s, p, 48, 52, key="kick", gain=1.0, ghosts=True, rolls=False)
    lay_hats(s, p, 48, 52, density=16, gain=0.48)
    lay_perc(s, p, 48, 52, gain=0.8)
    lay_acid(s, p, 48, 54, ACID_A, gain=0.46, cutoff=540, env_mod=3400, res=0.84, drive=6)
    s.place("fx", p["riser_n"], 52, 0, gain=0.42)
    s.place("fx", p["riser_t"], 52, 0, gain=0.30)
    snare_roll(s, p, 52, bars=3, gain=0.62)
    s.place("lead", p["scr_long"], 53, 0, gain=0.34, pan_=0.1)
    s.place("metal", p["steam"], 54, 0, gain=0.34)
    s.place("fx", p["rev_swell"], 54, 0, gain=0.38)
    s.place("fx", p["down"], 55, 12, gain=0.18)

    # ------------------------------------------------------- 056-087 DROP 1
    log("DROP 1")
    s.place("fx", p["impact"], 56, 0, gain=0.72)
    s.place("metal", p["clang"], 56, 0, gain=0.5)
    lay_kicks(s, p, 56, 88, key="kick", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 56, 88, density=16, gain=0.5, tips=True)
    lay_perc(s, p, 56, 88, gain=1.0, claps=True)
    lay_metal(s, p, 56, 88, gain=0.5)
    lay_acid(s, p, 56, 72, ACID_A, gain=0.5, cutoff=520, env_mod=3600, res=0.85, drive=7)
    lay_acid(s, p, 72, 88, ACID_B, gain=0.52, cutoff=600, env_mod=4200, res=0.87, drive=8)
    lay_hoover(s, p, 56, 88, gain=0.34)
    for b in range(56, 88, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.30, pan_=RNG.uniform(-0.3, 0.3))
        fill(s, p, b + 7, "metal")
    for b in range(60, 88, 8):
        s.place("voice", p["vstab"], b, 12, gain=0.34, pan_=0.15)
        s.place("voice", p["vstab2"], b + 2, 6, gain=0.26, pan_=-0.25)
    s.place("lead", p["scr_ud"], 71, 8, gain=0.34, pan_=-0.15)
    s.place("speech", p["sins_choir"], 76, 0, gain=0.42, pan_=0.0)
    s.place("fx", p["impact"], 72, 0, gain=0.4)
    s.place("breath", p["breath_short"], 79, 14, gain=0.4, pan_=0.4)

    # ---------------------------------------------------- 088-095 transition
    log("transition")
    s.place("fx", T.reverse_tail(p["clang"], 2.8), 86, 8, gain=0.5)
    s.place("drums", tape_stop(p["groove_bar"], start=0.12, end_ratio=0.035,
                               curve=1.7, max_stretch=2.6), 88, 0, gain=0.85)
    s.place("fx", p["sub_drop"], 88, 0, gain=0.62)
    s.place("fx", p["noise_fall"], 88, 2, gain=0.34)
    s.place("metal", p["steam"], 89, 6, gain=0.34, pan_=-0.35)
    s.place("breath", p["breath_out"], 89, 8, gain=0.62, pan_=0.25)
    s.place("voice", p["moan_close"], 90, 0, gain=0.52, pan_=-0.2)
    s.place("pad", p["pad_sex"], 90, 0, gain=0.70)
    s.place("breath", p["sigh_b"], 92, 4, gain=0.55, pan_=0.3)
    s.place("voice", p["moan_a"], 93, 8, gain=0.44, pan_=0.35)
    s.place("speech", p["forgive_close"], 94, 0, gain=0.6, pan_=0.0)
    s.place("fx", p["rev_swell"], 94, 8, gain=0.34)
    s.place("breath", p["breath_in"], 95, 12, gain=0.6, pan_=-0.15)
    for b in range(92, 96):
        for st in (0, 8):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.42)
            s.mark_kick(b, st)

    # --------------------------------------------------- 096-111 sultry mid
    log("sultry mid-section")
    lay_kicks(s, p, 96, 112, key="kick", gain=0.9, ghosts=False, rolls=False)
    lay_shaker(s, p, 96, 112, gain=0.30, swing=0.055)
    lay_hats(s, p, 96, 104, density=8, open_off=True, gain=0.32, swing=0.05)
    lay_hats(s, p, 104, 112, density=16, gain=0.40, swing=0.045, tips=True)
    lay_perc(s, p, 98, 112, gain=0.55, claps=True, tight=True, rims=True)
    lay_acid(s, p, 96, 112, ACID_SEX, gain=0.48, cutoff=330, env_mod=2100,
             res=0.86, drive=4, decay=0.30)
    lay_gated_pad(s, p, "pad_sex", 96, 8, gain=0.85)
    lay_gated_pad(s, p, "pad_sex", 104, 8, gain=0.80,
                  pattern=(1, 0, .6, .8, 0, 1, .4, 0))
    for b, st, key, g, pn in [(97, 8, "moan_close", 0.44, -0.3), (100, 0, "sigh_a", 0.46, 0.35),
                              (102, 12, "breath_short", 0.5, -0.4), (105, 4, "moan_c", 0.40, 0.25),
                              (108, 0, "sigh_b", 0.44, -0.2), (110, 8, "moan_a", 0.42, 0.4)]:
        s.place("breath" if key.startswith(("sigh", "breath")) else "voice",
                p[key], b, st, gain=g, pan_=pn)
    s.place("metal", p["conveyor2"], 104, 0, gain=0.5)
    s.place("metal", p["conveyor2"], 108, 0, gain=0.5)
    s.place("speech", p["say_whisper"], 106, 0, gain=0.34, pan_=0.5)
    for b in (103, 111):
        fill(s, p, b, "rev")

    # -------------------------------------------------------- 112-127 build 2
    log("build 2")
    lay_kicks(s, p, 112, 124, key="kick", gain=0.96, ghosts=True, rolls=False)
    lay_hats(s, p, 112, 120, density=8, gain=0.40)
    lay_hats(s, p, 120, 128, density=16, gain=0.50, tips=True)
    lay_perc(s, p, 112, 128, gain=0.85)
    lay_metal(s, p, 112, 128, gain=0.5)
    lay_acid(s, p, 112, 128, ACID_B, gain=0.46, cutoff=460, env_mod=3800, res=0.86, drive=7)
    lay_hoover(s, p, 120, 124, gain=0.26, stabs=False)
    s.place("fx", p["riser_n2"], 120, 0, gain=0.34)
    s.place("fx", p["riser_t2"], 120, 0, gain=0.26)
    s.place("fx", p["riser_n"], 124, 0, gain=0.46)
    s.place("fx", p["riser_t"], 124, 0, gain=0.34)
    snare_roll(s, p, 124, bars=3, gain=0.7)
    s.place("lead", p["scr_long"], 125, 0, gain=0.36, pan_=-0.1)
    s.place("voice", p["vstab"], 123, 8, gain=0.34)
    s.place("metal", p["steam"], 126, 0, gain=0.36)
    s.place("fx", p["rev_swell"], 126, 0, gain=0.42)

    # ------------------------------------------------------- 128-159 DROP 2
    log("DROP 2")
    s.place("fx", p["impact"], 128, 0, gain=0.8)
    s.place("metal", p["clang"], 128, 0, gain=0.55)
    lay_kicks(s, p, 128, 160, key="kick_hard", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 128, 160, density=16, gain=0.54, tips=True)
    lay_perc(s, p, 128, 160, gain=1.0, claps=True, rides=True)
    lay_metal(s, p, 128, 160, gain=0.55)
    lay_acid(s, p, 128, 144, ACID_B, gain=0.54, cutoff=640, env_mod=4400, res=0.88, drive=9)
    lay_acid(s, p, 144, 160, ACID_C, gain=0.56, cutoff=700, env_mod=4800, res=0.90, drive=10)
    lay_hoover(s, p, 128, 144, gain=0.34)
    lay_hoover(s, p, 144, 160, gain=0.34, riff=HOOVER_RIFF2)
    lay_lead_screech(s, p, 136, 152, gain=0.34)
    for b in range(128, 160, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.32, pan_=RNG.uniform(-0.3, 0.3))
        s.place("voice", p["vstab2"], b + 3, 12, gain=0.28, pan_=RNG.uniform(-0.3, 0.3))
        fill(s, p, b + 7, "metal")
    s.place("speech", p["sins_choir"], 142, 0, gain=0.5)
    s.place("voice", p["vstab_low"], 135, 8, gain=0.34, pan_=-0.2)
    s.place("scream", far(p["cry_short"], 7000), 151, 14, gain=0.5, pan_=0.25)
    s.place("lead", p["scr_ud"], 151, 8, gain=0.36, pan_=0.15)
    s.place("fx", p["impact"], 144, 0, gain=0.45)
    s.place("fx", p["down"], 159, 12, gain=0.32)

    # ----------------------------------------------------- 160-171 breakdown
    log("breakdown")
    s.place("pad", p["pad_dark"], 160, 0, gain=0.42)
    s.place("speech", p["say_choir"], 161, 0, gain=0.58)
    s.place("speech", p["say_whisper"], 161, 0.5, gain=0.40, pan_=-0.5)
    s.place("voice", p["moan_b"], 163, 8, gain=0.42, pan_=0.26)
    s.place("breath", p["sigh_b"], 165, 0, gain=0.46, pan_=-0.3)
    s.place("scream", far(p["help_far"], 1700), 166, 4, gain=0.42, pan_=0.45)
    s.place("voice", p["moan_c"], 167, 8, gain=0.38, pan_=-0.25)
    s.place("pad", p["pad_b"], 166, 0, gain=0.34)
    s.place("voice", p["whisper2"], 168, 4, gain=0.3, pan_=0.4)
    s.place("metal", p["machine_lo"], 160, 0, gain=0.6)
    s.place("metal", p["chain"], 169, 2, gain=0.7, pan_=-0.4)
    for b in range(166, 172):
        s.place("drums", p["rim"], b, 6, gain=0.16, pan_=RNG.uniform(-0.6, 0.6))
        for st in (0, 8):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.4 + 0.035 * (b - 166))
            s.mark_kick(b, st)

    # -------------------------------------------------------- 172-179 build 3
    log("build 3")
    lay_kicks(s, p, 172, 178, key="kick", gain=0.94, ghosts=True, rolls=False)
    lay_hats(s, p, 172, 180, density=16, gain=0.48, tips=True)
    lay_perc(s, p, 172, 180, gain=0.8)
    lay_metal(s, p, 172, 180, gain=0.5)
    lay_acid(s, p, 172, 179, ACID_C, gain=0.5, cutoff=600, env_mod=4200, res=0.88, drive=8)
    s.place("fx", p["riser_n"], 176, 0, gain=0.48)
    s.place("fx", p["riser_t"], 176, 0, gain=0.36)
    snare_roll(s, p, 176, bars=3, gain=0.76)
    s.place("lead", p["scr_evil"], 177, 0, gain=0.38, pan_=0.1)
    s.place("metal", p["steam"], 178, 0, gain=0.38)
    s.place("fx", p["rev_swell"], 178, 0, gain=0.44)
    s.place("fx", p["down"], 179, 12, gain=0.2)

    # -------------------------------------------------------- 180-195 DROP 3
    log("DROP 3")
    s.place("fx", p["impact"], 180, 0, gain=0.85)
    s.place("metal", p["clang"], 180, 0, gain=0.6)
    lay_kicks(s, p, 180, 196, key="kick_max", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 180, 196, density=16, gain=0.56, tips=True)
    lay_perc(s, p, 180, 196, gain=1.0, claps=True, rides=True)
    lay_metal(s, p, 180, 196, gain=0.6)
    lay_acid(s, p, 180, 196, ACID_C, gain=0.58, cutoff=760, env_mod=5000, res=0.91, drive=11)
    lay_hoover(s, p, 180, 196, gain=0.36, riff=HOOVER_RIFF2)
    lay_lead_screech(s, p, 184, 196, gain=0.36)
    for b in range(180, 196, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.34, pan_=RNG.uniform(-0.3, 0.3))
        fill(s, p, b + 7, "metal")
    s.place("speech", p["sins_choir"], 188, 0, gain=0.5)
    s.place("voice", p["vstab"], 183, 12, gain=0.34)
    s.place("voice", p["vstab2"], 191, 6, gain=0.3)
    s.place("scream", far(p["help_panic"], 8000), 194, 10, gain=0.5, pan_=-0.25)

    # --------------------------------------------------------- 196-207 outro
    log("outro")
    lay_kicks(s, p, 196, 204, key="kick", gain=0.95, ghosts=True, rolls=False)
    lay_hats(s, p, 196, 202, density=16, gain=0.42)
    lay_perc(s, p, 196, 202, gain=0.65, claps=True)
    lay_metal(s, p, 196, 204, gain=0.45)
    lay_acid(s, p, 196, 202, ACID_A, gain=0.36, cutoff=420, env_mod=2600, res=0.82, drive=5)
    s.place("metal", p["machine"], 200, 0, gain=0.7)
    s.place("speech", p["say_whisper"], 202, 0, gain=0.46, pan_=-0.3)
    s.place("speech", p["forgive_close"], 204, 4, gain=0.6)
    s.place("fx", p["impact"], 204, 0, gain=0.5)
    s.place("breath", p["breath_out"], 205, 8, gain=0.44, pan_=0.3)
    s.place("scream", far(p["help_far"], 1100), 205, 2, gain=0.36, pan_=-0.35)
    s.place("metal", p["clang_hi"], 206, 0, gain=0.3, pan_=0.2)
    s.place("fx", p["rev_swell"], 206, 0, gain=0.2)

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

    log("distant kick + intro filter")
    s.buses["kickfar"] = biquad(s.buses["kickfar"], "lp", 420, 0.9)
    s.buses["kickfar"] = send_reverb(s.buses["kickfar"], 0.55, rt60=3.0, damp=0.8,
                                     hp=90, seed=21)
    open_env = np.ones(n)
    i1 = s.i(16)
    open_env[:i1] = np.interp(np.arange(i1), [0, s.i(8), i1], [420.0, 1100.0, 18000.0])
    open_env[i1:] = 18000.0
    for nm in ("kickfar", "drums", "air"):
        s.buses[nm] = sweep(s.buses[nm], "lp", open_env, 0.8, block=512)

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

    log("voices")
    s.buses["voice"] = send_delay(s.buses["voice"], 0.30, s.step * 6, feedback=0.40, damp=3200)
    s.buses["voice"] = send_reverb(s.buses["voice"], 0.85, rt60=3.6, damp=0.62, hp=180,
                                   predelay=0.034, seed=61)
    s.buses["voice"] = widen(s.buses["voice"], 0.5, 15.0)
    s.buses["voice"] = biquad(s.buses["voice"], "hp", 130, 0.7)
    s.buses["voice"] = drive_os(s.buses["voice"], 1.5, os=2)

    log("screams")
    s.buses["scream"] = drive_os(s.buses["scream"], 1.6, os=2)
    s.buses["scream"] = send_delay(s.buses["scream"], 0.34, s.step * 6, feedback=0.46,
                                   damp=2800)
    s.buses["scream"] = send_reverb(s.buses["scream"], 1.15, rt60=4.4, damp=0.6, hp=210,
                                    predelay=0.045, seed=101)
    s.buses["scream"] = widen(s.buses["scream"], 0.45, 15.0)
    s.buses["scream"] = biquad(s.buses["scream"], "hp", 160, 0.7)

    log("speech: a voice in a large stone room")
    # Long predelay keeps the words in front of the reverb instead of inside
    # it - the difference between a cathedral and a bathroom.
    s.buses["speech"] = send_delay(s.buses["speech"], 0.20, s.step * 6, feedback=0.34,
                                   damp=3600)
    s.buses["speech"] = send_reverb(s.buses["speech"], 0.62, rt60=5.0, damp=0.68, hp=190,
                                    predelay=0.075, seed=121, width=0.85)
    s.buses["speech"] = widen(s.buses["speech"], 0.3, 11.0)
    s.buses["speech"] = biquad(s.buses["speech"], "hp", 105, 0.7)
    s.buses["speech"] = s.buses["speech"] + 0.18 * biquad(s.buses["speech"], "bp", 2400, 0.7)

    log("breath: close and wide")
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

    log("sidechain")
    deep = s.duck_envelope(depth=0.88, attack=0.003, hold=0.03, release=0.155)
    mid = s.duck_envelope(depth=0.62, attack=0.004, hold=0.02, release=0.13)
    light = s.duck_envelope(depth=0.34, attack=0.005, hold=0.012, release=0.10)
    s.apply_duck(["rumble", "sub"], deep)
    s.apply_duck(["bass", "pad", "metal"], mid)
    s.apply_duck(["lead", "voice", "scream", "speech", "breath", "fx", "drums", "air"],
                 light)
    return s.buses


def finalize(buses, verbose=True):
    """Balance, sum and master. Cheap enough to re-run while tuning."""
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
