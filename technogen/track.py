"""The arrangement: 'CONCRETE CATHEDRAL' - 150 BPM hard techno, F# minor.

Sections
  000-015  intro      warehouse room, distant voices, muffled kick
  016-031  build 1    kick opens up, acid enters, tension rises
  032-039  pre-drop   roll, riser, cut to silence
  040-071  DROP 1     kick + rumble + hoover + screeches
  072-087  breakdown  pads, moans, whispers, reverb wash
  088-103  build 2    everything climbs, snare roll, hard cut
  104-135  DROP 2     harder, faster, lead screech riff
  136-151  outro      strip back, filter down, last impact
"""

import numpy as np

from .dsp import (SR, biquad, sweep, reverb, delay, widen, tanh_drive, soft_clip,
                  normalize, env_curve, noise, n_samples, db, pitch_shift_naive)
from . import instruments as I
from .mixer import Session, send_reverb, send_delay, master, balance

BPM = 150.0
BARS = 152
ROOT = 42          # F#2
RNG = np.random.default_rng(2024)


# ------------------------------------------------------------------ palette

def build_palette():
    p = {}
    p["kick"] = I.kick(0.62, tune=45.0, punch=1.0, drive=7.5, decay=0.135, dirt=0.7)
    p["kick_hard"] = I.kick(0.62, tune=45.0, punch=1.15, drive=10.0, decay=0.15, dirt=0.95)
    p["kick_soft"] = I.kick(0.55, tune=45.0, punch=0.8, drive=3.5, decay=0.12, dirt=0.25)
    p["kick_short"] = I.kick(0.3, tune=46.0, punch=1.0, drive=8.0, decay=0.07, dirt=0.6)
    p["sub"] = I.kick_sub(0.75, tune=45.0, decay=0.2)

    p["hat"] = I.hat(0.05, tone=9200, decay=0.012, metal=0.35)
    p["hat2"] = I.hat(0.05, tone=11000, decay=0.009, metal=0.55)
    p["ohat"] = I.hat(0.30, tone=8200, decay=0.075, metal=0.45)
    p["ohat_long"] = I.hat(0.55, tone=7600, decay=0.16, metal=0.5)
    p["clap"] = I.clap(0.5, tone=1600, body=0.35)
    p["snare"] = I.snare(0.32, tune=196, snap=0.78)
    p["snare_t"] = I.snare(0.16, tune=230, snap=0.85)
    p["rim"] = I.rim(0.12, 1750)
    p["ride"] = I.ride(1.0, tone=5400, decay=0.36)
    p["tom_h"] = I.tom(0.35, tune=150, decay=0.1)
    p["tom_l"] = I.tom(0.45, tune=96, decay=0.15)

    for m in (42, 45, 47, 49, 52, 54, 57):
        p[f"hoov{m}"] = I.hoover(m, 1.5, detune=27, sweep_from=1.5, glide=0.12,
                                 cutoff=(600, 5600), res=0.74, drive=5.5)
        p[f"hoovS{m}"] = I.hoover(m, 0.42, detune=22, sweep_from=1.28, glide=0.05,
                                  cutoff=(900, 6200), res=0.68, drive=6.5)
        p[f"stab{m}"] = I.stab(m, 0.26, detune=20, cutoff=3000, res=0.72, drive=6.0)

    p["scr_up"] = I.screech(1.6, base=520, top=5200, wobble=6.5, res=0.93, drive=9, direction="up", seed=1)
    p["scr_dn"] = I.screech(1.2, base=4600, top=420, wobble=9.0, res=0.9, drive=8, direction="down", seed=2)
    p["scr_ud"] = I.screech(2.2, base=700, top=6400, wobble=5.0, res=0.95, drive=11, direction="updown", seed=3)
    p["scr_st"] = I.screech(0.34, base=1500, top=4800, wobble=14.0, res=0.9, drive=10, direction="up", seed=4)
    p["scr_st2"] = I.screech(0.28, base=5200, top=1800, wobble=16.0, res=0.9, drive=10, direction="down", seed=5)
    p["scr_long"] = I.screech(3.2, base=380, top=7200, wobble=4.0, res=0.96, drive=12, direction="up", seed=6)

    p["riser_n"] = I.riser_noise(6.4, 260, 12000, q=2.2, swell=1.8, seed=4)
    p["riser_n2"] = I.riser_noise(12.8, 180, 13000, q=2.0, swell=2.4, seed=8)
    p["riser_t"] = I.riser_tone(6.4, 34, 76, detune=22, drive=3.5)
    p["riser_t2"] = I.riser_tone(12.8, 30, 80, detune=26, drive=4.0)
    p["down"] = I.downlifter(2.4, 2600, 55)
    p["impact"] = I.impact(3.0, tune=44)
    p["rev_swell"] = I.reverse_swell(3.2, seed=9)
    p["rev_swell2"] = I.reverse_swell(1.6, seed=10, bright=9000)
    p["zap"] = I.zap(0.35)

    p["moan_a"] = I.moan(2.6, root=54, seed=0, breath=0.30, up=True)
    p["moan_b"] = I.moan(3.2, root=49, seed=1, breath=0.36, up=False)
    p["moan_c"] = I.moan(2.0, root=57, seed=2, breath=0.26, up=True)
    p["chat_a"] = I.vocal_chatter(3.0, syllables=7, root=45, seed=1)
    p["chat_b"] = I.vocal_chatter(2.2, syllables=5, root=50, seed=4)
    p["whisper"] = I.whisper(4.0, seed=3)
    p["whisper2"] = I.whisper(2.6, seed=6, tone=1.15)
    p["vstab"] = I.vocal_stab(0.40, root=57, vowel="eh", drive=7)
    p["vstab2"] = I.vocal_stab(0.32, root=62, vowel="ah", drive=8)
    p["vstab_low"] = pitch_shift_naive(I.vocal_stab(0.5, root=50, vowel="oh", drive=5), 1.45)

    # cries for help, at varying distances
    p["help_far"] = I.scream_help(1.9, pitch=0.85, seed=7, drive=5.0, effort=0.85)
    p["help_mid"] = I.scream_help(1.6, pitch=1.0, seed=0, drive=6.5, effort=1.0)
    p["help_near"] = I.scream_help(1.45, pitch=1.12, seed=11, drive=7.5, effort=1.15)
    p["help_panic"] = I.scream_help(1.25, pitch=1.22, seed=3, drive=8.0, effort=1.2)
    p["cry"] = I.scream(1.7, f0=(230, 560, 300), vowel_path=((0.0, "eh"), (0.5, "ah"), (1.0, "ah")),
                        roughness=0.65, breath=0.5, drive=6.5, seed=5)
    p["cry_short"] = I.scream(0.85, f0=(320, 620, 380), vowel_path=((0.0, "ah"), (1.0, "eh")),
                              roughness=0.7, breath=0.45, drive=7.5, seed=9, effort=1.15)

    p["pad_a"] = I.pad([42, 49, 54, 57], 12.8, detune=16, cutoff=1300, drive=1.5)
    p["pad_b"] = I.pad([40, 47, 52, 59], 12.8, detune=18, cutoff=1100, drive=1.5)
    return p


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

HOOVER_RIFF = [(0, 0, 54), (0, 12, 49), (2, 0, 47), (2, 10, 45),
               (4, 0, 54), (4, 8, 57), (6, 0, 52), (6, 6, 49)]

LEAD_RIFF = [(0, 0, 54, 1.5), (1, 8, 57, 0.75), (2, 0, 52, 1.0), (3, 4, 49, 0.75),
             (4, 0, 57, 1.5), (5, 8, 61, 0.75), (6, 0, 59, 1.0), (7, 2, 54, 1.25)]


# ---------------------------------------------------------------- sections

def lay_kicks(s, p, b0, b1, key="kick", gain=1.0, ghosts=False, rolls=True,
              sub_gain=1.0, every=4):
    for b in range(b0, b1):
        for st in range(0, 16, every):
            g = gain * (1.0 if st == 0 else 0.97)
            s.place("kick", p[key], b, st, gain=g)
            s.mark_kick(b, st)
            if sub_gain:
                s.place("sub", p["sub"], b, st, gain=0.85 * sub_gain)
        if ghosts and (b - b0) % 4 == 3:
            s.place("kick", p["kick_short"], b, 14, gain=gain * 0.55)
            s.mark_kick(b, 14)
        if rolls and (b - b0) % 16 == 15:
            for st in (12, 13, 14, 15):
                s.place("kick", p["kick_short"], b, st, gain=gain * (0.5 + 0.13 * (st - 12)))
                s.mark_kick(b, st)


def lay_hats(s, p, b0, b1, density=16, open_off=True, gain=0.5, jitter=0.0012):
    for b in range(b0, b1):
        stepset = range(0, 16, 16 // density) if density else []
        for st in stepset:
            acc = 1.0 if st % 4 == 2 else (0.72 if st % 4 == 0 else 0.6)
            k = "hat2" if st % 8 == 4 else "hat"
            s.place("drums", p[k], b, st + RNG.normal(0, jitter) / s.step,
                    gain=gain * acc * RNG.uniform(0.85, 1.05),
                    pan_=RNG.uniform(-0.28, 0.28))
        if open_off:
            for st in (2, 6, 10, 14):
                s.place("drums", p["ohat"], b, st, gain=gain * 0.85, pan_=RNG.uniform(-0.15, 0.15))


def lay_perc(s, p, b0, b1, gain=1.0, claps=True, rides=False, rims=True):
    for b in range(b0, b1):
        i = b - b0
        if claps and i % 2 == 1:
            s.place("drums", p["clap"], b, 8, gain=0.52 * gain, pan_=0.08)
        if rims:
            for st in (3, 11, 15):
                if (i + st) % 3:
                    s.place("drums", p["rim"], b, st, gain=0.22 * gain, pan_=RNG.uniform(-0.6, 0.6))
        if rides:
            for st in range(0, 16, 2):
                s.place("drums", p["ride"], b, st, gain=0.16 * gain, pan_=RNG.uniform(-0.4, 0.4))
        if i % 8 == 7:
            s.place("drums", p["tom_l"], b, 10, gain=0.4 * gain, pan_=-0.3)
            s.place("drums", p["tom_h"], b, 13, gain=0.36 * gain, pan_=0.35)


def snare_roll(s, p, b0, bars=2, start=8, end=64, gain=0.8):
    """Accelerating roll: hits per bar double as it climbs."""
    total_steps = bars * 16
    t = 0.0
    idx = 0
    while t < total_steps:
        frac = t / total_steps
        div = 4 - 3 * frac          # 16ths -> 64ths
        key = "snare_t" if frac > 0.45 else "snare"
        g = gain * (0.45 + 0.75 * frac) * RNG.uniform(0.9, 1.05)
        s.place("drums", p[key], b0 + int(t // 16), t % 16, gain=g,
                pan_=RNG.uniform(-0.2, 0.2))
        t += max(0.25, div)
        idx += 1


def lay_acid(s, p, b0, b1, pattern, gain=0.5, **kw):
    line = I.acid(pattern, BPM, **kw)
    for b in range(b0, b1):
        s.place("bass", line, b, 0, gain=gain)


def lay_hoover(s, p, b0, b1, gain=0.5, riff=HOOVER_RIFF, stabs=True):
    for b in range(b0, b1):
        i = (b - b0) % 8
        for rb, rs, m in riff:
            if rb == i:
                key = f"hoov{m}"
                s.place("lead", p[key], b, rs, gain=gain, pan_=RNG.uniform(-0.12, 0.12))
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


def far(x, hz, rolloff=2):
    """Distance cue: air and walls eat the top end long before the level."""
    return biquad(x, "lp", hz, 0.7, stages=rolloff)


def build(verbose=True):
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    log("rendering sound palette...")
    p = build_palette()
    s = Session(BPM, BARS, tail=6.0)

    # ---------------------------------------------------------- 000-015 intro
    log("intro (warehouse)")
    s.place("air", I.room_tone(s.dur, level=0.09), 0, 0, gain=1.0)
    for b in range(0, 16):
        if b % 4 == 0:
            s.place("air", p["rev_swell"], b, 12, gain=0.16, pan_=RNG.uniform(-0.5, 0.5))
    # distant muffled kick from bar 4
    for b in range(4, 16):
        for st in range(0, 16, 4):
            g = 0.22 + 0.030 * (b - 4)
            s.place("kickfar", p["kick_soft"], b, st, gain=min(0.58, g))
            if b >= 8:
                s.mark_kick(b, st)
    lay_hats(s, p, 10, 16, density=8, open_off=False, gain=0.18)
    s.place("voice", p["chat_a"], 2, 4, gain=0.30, pan_=-0.4)
    # someone is shouting for help somewhere in the building, getting closer
    s.place("scream", far(p["help_far"], 1200), 2, 10, gain=0.34, pan_=-0.45)
    s.place("scream", far(p["cry"], 2100), 5, 6, gain=0.46, pan_=0.42)
    s.place("scream", far(p["help_mid"], 4200), 9, 8, gain=0.66, pan_=-0.18)
    s.place("scream", far(p["help_panic"], 6800), 13, 10, gain=0.78, pan_=0.22)
    s.place("voice", p["whisper"], 6, 0, gain=0.24, pan_=0.45)
    s.place("voice", p["chat_b"], 12, 6, gain=0.26, pan_=0.3)
    s.place("fx", p["scr_dn"], 7, 8, gain=0.18, pan_=0.2)
    s.place("fx", p["impact"], 8, 0, gain=0.5)
    s.place("fx", p["zap"], 14, 10, gain=0.2, pan_=-0.5)

    # -------------------------------------------------------- 016-031 build 1
    log("build 1")
    lay_kicks(s, p, 16, 32, key="kick", gain=0.92, ghosts=True, rolls=False)
    lay_hats(s, p, 16, 24, density=8, gain=0.34)
    lay_hats(s, p, 24, 32, density=16, gain=0.44)
    lay_perc(s, p, 20, 32, gain=0.7, claps=True, rims=True)
    lay_acid(s, p, 22, 32, ACID_A, gain=0.36, cutoff=380, env_mod=2400, res=0.8, drive=5)
    s.place("voice", p["vstab_low"], 23, 12, gain=0.3, pan_=-0.2)
    s.place("fx", p["riser_n2"], 24, 0, gain=0.26)
    s.place("fx", p["rev_swell"], 27, 12, gain=0.22, pan_=0.3)
    s.place("lead", p["scr_up"], 30, 8, gain=0.26, pan_=-0.25)

    # ------------------------------------------------------- 032-039 pre-drop
    log("pre-drop")
    lay_kicks(s, p, 32, 36, key="kick", gain=1.0, ghosts=True, rolls=False)
    lay_hats(s, p, 32, 36, density=16, gain=0.46)
    lay_perc(s, p, 32, 36, gain=0.8)
    lay_acid(s, p, 32, 38, ACID_A, gain=0.44, cutoff=520, env_mod=3200, res=0.84, drive=6)
    s.place("fx", p["riser_n"], 36, 0, gain=0.42)
    s.place("fx", p["riser_t"], 36, 0, gain=0.30)
    snare_roll(s, p, 36, bars=3, gain=0.62)
    s.place("lead", p["scr_long"], 37, 0, gain=0.34, pan_=0.1)
    s.place("fx", p["rev_swell"], 38, 0, gain=0.38)
    # last beat: everything gone
    s.place("fx", p["down"], 39, 12, gain=0.18)

    # --------------------------------------------------------- 040-071 drop 1
    log("DROP 1")
    s.place("fx", p["impact"], 40, 0, gain=0.72)
    lay_kicks(s, p, 40, 72, key="kick", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 40, 72, density=16, gain=0.5)
    lay_perc(s, p, 40, 72, gain=1.0, claps=True, rides=False)
    lay_acid(s, p, 40, 56, ACID_A, gain=0.5, cutoff=520, env_mod=3600, res=0.85, drive=7)
    lay_acid(s, p, 56, 72, ACID_B, gain=0.52, cutoff=600, env_mod=4200, res=0.87, drive=8)
    lay_hoover(s, p, 40, 72, gain=0.34)
    for b in range(40, 72, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.30, pan_=RNG.uniform(-0.3, 0.3))
    for b in range(44, 72, 8):
        s.place("voice", p["vstab"], b, 12, gain=0.34, pan_=0.15)
        s.place("voice", p["vstab2"], b + 2, 6, gain=0.26, pan_=-0.25)
    s.place("lead", p["scr_ud"], 55, 8, gain=0.34, pan_=-0.15)
    s.place("fx", p["rev_swell2"], 63, 12, gain=0.3)
    s.place("fx", p["impact"], 56, 0, gain=0.4)
    s.place("voice", p["chat_b"], 68, 4, gain=0.22, pan_=-0.35)
    s.place("fx", p["down"], 71, 12, gain=0.3)

    # ----------------------------------------------------- 072-087 breakdown
    log("breakdown (voices)")
    s.place("pad", p["pad_a"], 72, 0, gain=0.40)
    s.place("pad", p["pad_b"], 80, 0, gain=0.34)
    s.place("voice", p["moan_a"], 73, 4, gain=0.42, pan_=-0.22)
    s.place("voice", p["moan_b"], 76, 0, gain=0.40, pan_=0.26)
    s.place("voice", p["whisper"], 74, 8, gain=0.26, pan_=0.4)
    s.place("scream", far(p["help_far"], 1700), 75, 4, gain=0.42, pan_=0.45)
    s.place("scream", far(p["cry"], 2200), 86, 8, gain=0.38, pan_=-0.4)
    s.place("voice", p["moan_c"], 78, 8, gain=0.36, pan_=-0.3)
    s.place("voice", p["chat_a"], 79, 0, gain=0.24, pan_=0.2)
    s.place("voice", p["moan_a"], 82, 0, gain=0.38, pan_=0.3)
    s.place("voice", p["whisper2"], 84, 4, gain=0.3, pan_=-0.4)
    s.place("voice", p["moan_b"], 85, 8, gain=0.34, pan_=-0.1)
    s.place("fx", p["scr_dn"], 72, 0, gain=0.3, pan_=0.1)
    s.place("lead", p["scr_up"], 79, 8, gain=0.2, pan_=-0.2)
    for b in range(76, 88):
        s.place("drums", p["rim"], b, 6, gain=0.14, pan_=RNG.uniform(-0.6, 0.6))
        if b >= 80:
            s.place("drums", p["ohat"], b, 2, gain=0.18)
            s.place("drums", p["ohat"], b, 10, gain=0.18)
    # heartbeat kick returns
    for b in range(80, 88):
        for st in range(0, 16, 8 if b < 84 else 4):
            s.place("kickfar", p["kick_soft"], b, st, gain=0.55 + 0.04 * (b - 80))
            s.mark_kick(b, st)

    # --------------------------------------------------------- 088-103 build 2
    log("build 2")
    lay_kicks(s, p, 88, 100, key="kick", gain=0.96, ghosts=True, rolls=False)
    lay_hats(s, p, 88, 96, density=8, gain=0.38)
    lay_hats(s, p, 96, 103, density=16, gain=0.5)
    lay_perc(s, p, 88, 103, gain=0.85)
    lay_acid(s, p, 90, 103, ACID_B, gain=0.46, cutoff=460, env_mod=3800, res=0.86, drive=7)
    lay_hoover(s, p, 96, 100, gain=0.26, stabs=False)
    s.place("fx", p["riser_n2"], 96, 0, gain=0.34)
    s.place("fx", p["riser_t2"], 96, 0, gain=0.26)
    s.place("fx", p["riser_n"], 100, 0, gain=0.46)
    s.place("fx", p["riser_t"], 100, 0, gain=0.34)
    snare_roll(s, p, 100, bars=3, gain=0.7)
    s.place("lead", p["scr_long"], 101, 0, gain=0.36, pan_=-0.1)
    s.place("voice", p["vstab"], 99, 8, gain=0.34)
    s.place("fx", p["rev_swell"], 102, 0, gain=0.42)

    # -------------------------------------------------------- 104-135 drop 2
    log("DROP 2")
    s.place("fx", p["impact"], 104, 0, gain=0.8)
    lay_kicks(s, p, 104, 136, key="kick_hard", gain=1.0, ghosts=True, rolls=True)
    lay_hats(s, p, 104, 136, density=16, gain=0.54)
    lay_perc(s, p, 104, 136, gain=1.0, claps=True, rides=True)
    lay_acid(s, p, 104, 120, ACID_B, gain=0.54, cutoff=640, env_mod=4400, res=0.88, drive=9)
    lay_acid(s, p, 120, 136, ACID_C, gain=0.56, cutoff=700, env_mod=4800, res=0.9, drive=10)
    lay_hoover(s, p, 104, 136, gain=0.34)
    lay_lead_screech(s, p, 112, 128, gain=0.34)
    for b in range(104, 136, 8):
        s.place("lead", p["scr_up"], b + 7, 8, gain=0.32, pan_=RNG.uniform(-0.3, 0.3))
        s.place("voice", p["vstab2"], b + 3, 12, gain=0.28, pan_=RNG.uniform(-0.3, 0.3))
    s.place("voice", p["vstab_low"], 111, 8, gain=0.34, pan_=-0.2)
    s.place("scream", far(p["cry_short"], 7000), 119, 14, gain=0.55, pan_=0.25)
    s.place("voice", p["moan_c"], 118, 0, gain=0.26, pan_=0.35)
    s.place("lead", p["scr_ud"], 127, 8, gain=0.36, pan_=0.15)
    s.place("fx", p["impact"], 120, 0, gain=0.45)
    s.place("voice", p["vstab"], 131, 4, gain=0.3)
    s.place("fx", p["down"], 135, 12, gain=0.32)

    # --------------------------------------------------------- 136-151 outro
    log("outro")
    lay_kicks(s, p, 136, 148, key="kick", gain=0.95, ghosts=True, rolls=False)
    lay_hats(s, p, 136, 146, density=16, gain=0.4)
    lay_perc(s, p, 136, 144, gain=0.7, claps=True)
    lay_acid(s, p, 136, 144, ACID_A, gain=0.36, cutoff=420, env_mod=2600, res=0.82, drive=5)
    s.place("voice", p["chat_a"], 140, 0, gain=0.24, pan_=-0.3)
    s.place("voice", p["moan_b"], 144, 4, gain=0.3, pan_=0.3)
    s.place("lead", p["scr_dn"], 147, 8, gain=0.28, pan_=0.1)
    s.place("fx", p["impact"], 148, 0, gain=0.55)
    s.place("voice", p["whisper"], 148, 8, gain=0.26, pan_=-0.35)
    s.place("fx", p["rev_swell"], 150, 0, gain=0.2)
    s.place("scream", far(p["help_far"], 1100), 149, 8, gain=0.40, pan_=-0.35)

    return s, p


# ------------------------------------------------------------------- render

# Where each bus should sit relative to the kick, and in which band it is
# measured. Low-frequency buses are compared against the kick's low band,
# everything else against the mid band it actually competes in.
TARGETS = {
    "kick":    (0.0, "low"),
    "kickfar": (-11.0, "low"),
    "sub":     (-6.0, "low"),
    "rumble":  (-7.0, "low"),
    "drums":   (1.0, "mid"),
    "bass":    (0.0, "mid"),
    "lead":    (4.0, "mid"),
    "voice":   (0.0, "mid"),
    "scream":  (-2.0, "mid"),
    "fx":      (0.0, "mid"),
    "pad":     (-7.0, "mid"),
    "air":     (-16.0, "mid"),
}


# Automation curves, in bars. This is what makes a drop land: during a build
# the kick is smaller and darker and the sub/rumble are held back, so bar 40
# arrives with a real jump in weight rather than just more elements.
LOW_WEIGHT = [(0, 0.10), (16, 0.26), (24, 0.34), (32, 0.44), (35.9, 0.48),
              (36, 0.30), (39.9, 0.20), (40, 1.0), (71.9, 1.0), (72, 0.12),
              (80, 0.22), (88, 0.40), (96, 0.55), (99.9, 0.62), (100, 0.34),
              (103.9, 0.22), (104, 1.0), (135.9, 1.0), (136, 0.86), (146, 0.8),
              (149, 0.45), (152, 0.2)]

KICK_GAIN = [(0, 0.5), (16, 0.52), (20, 0.60), (24, 0.70), (28, 0.72), (32, 0.78),
             (36, 0.82), (39.9, 0.82), (40, 1.0), (71.9, 1.0), (72, 0.6), (88, 0.70),
             (96, 0.84), (103.9, 0.90), (104, 1.0), (135.9, 1.0), (136, 0.92),
             (147, 0.88), (150, 0.6), (152, 0.3)]

KICK_TONE = [(0, 1500), (16, 1700), (24, 2400), (30, 3400), (36, 4600), (39.9, 4600),
             (40, 20000), (71.9, 20000), (72, 2600), (88, 2800), (96, 4200),
             (103.9, 5200), (104, 20000), (143, 20000), (148, 9000), (152, 2000)]


def process_buses(s, p, verbose=True):
    """Everything up to the balance stage: sends, space, dirt, sidechain."""
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    n = s.n
    for nm in ("kick", "kickfar", "sub", "drums", "bass", "lead", "voice", "scream",
               "fx", "pad", "air"):
        s.bus(nm)

    log("rumble bus")
    s.buses["rumble"] = I.rumble_from(s.buses["sub"][0], rt60=2.0, cut=200.0, drive=2.8)

    log("bus processing")
    # the distant kick in intro/breakdown: muffled, roomy
    s.buses["kickfar"] = biquad(s.buses["kickfar"], "lp", 420, 0.9)
    s.buses["kickfar"] = send_reverb(s.buses["kickfar"], 0.55, rt60=3.0, damp=0.8, hp=90, seed=21)

    # opening filter sweep on the whole intro
    sweep_env = np.ones(n)
    i0, i1 = s.i(0), s.i(16)
    sweep_env[:i1] = np.interp(np.arange(i1), [0, s.i(8), i1], [300.0, 900.0, 18000.0])
    sweep_env[i1:] = 18000.0
    for nm in ("kickfar", "drums", "air"):
        s.buses[nm] = sweep(s.buses[nm], "lp", sweep_env, 0.8, block=512)

    log("arrangement automation")
    low_w = s.ramp(LOW_WEIGHT)
    s.buses["rumble"] *= low_w
    s.buses["sub"] *= low_w
    s.buses["kick"] *= s.ramp(KICK_GAIN)
    s.buses["kick"] = sweep(s.buses["kick"], "lp", s.ramp(KICK_TONE), 0.8, block=512)
    # the build's kick also loses its low-end weight, not just its top
    s.buses["kick"] = s.buses["kick"] * (0.55 + 0.45 * low_w) + \
        biquad(s.buses["kick"], "hp", 110, 0.7) * (1.0 - low_w) * 0.45

    s.buses["drums"] = tanh_drive(s.buses["drums"], 2.0)
    s.buses["drums"] = send_reverb(s.buses["drums"], 0.30, rt60=1.5, damp=0.6, hp=400,
                                   predelay=0.012, seed=31)
    s.buses["drums"] = widen(s.buses["drums"], 0.35, 9.0)

    s.buses["bass"] = tanh_drive(s.buses["bass"], 1.8)
    s.buses["bass"] = send_delay(s.buses["bass"], 0.22, s.step * 3, feedback=0.30)
    s.buses["bass"] = send_reverb(s.buses["bass"], 0.16, rt60=1.4, damp=0.7, hp=300, seed=41)
    s.buses["bass"] = biquad(s.buses["bass"], "hp", 60, 0.7)

    log("lead: delay + reverb + dirt")
    s.buses["lead"] = tanh_drive(s.buses["lead"], 2.2)
    s.buses["lead"] = send_delay(s.buses["lead"], 0.34, s.step * 3, feedback=0.42, damp=4200)
    s.buses["lead"] = send_reverb(s.buses["lead"], 0.55, rt60=2.8, damp=0.5, hp=240,
                                  predelay=0.026, seed=51)
    s.buses["lead"] = widen(s.buses["lead"], 0.55, 13.0)
    s.buses["lead"] = biquad(s.buses["lead"], "hp", 170, 0.7)

    log("voices: pitched down, drenched")
    s.buses["voice"] = send_delay(s.buses["voice"], 0.30, s.step * 6, feedback=0.40, damp=3200)
    s.buses["voice"] = send_reverb(s.buses["voice"], 0.85, rt60=3.6, damp=0.62, hp=180,
                                   predelay=0.034, seed=61)
    s.buses["voice"] = widen(s.buses["voice"], 0.5, 15.0)
    s.buses["voice"] = biquad(s.buses["voice"], "hp", 130, 0.7)
    s.buses["voice"] = tanh_drive(s.buses["voice"], 1.5)

    log("screams: the room answers them")
    s.buses["scream"] = tanh_drive(s.buses["scream"], 1.6)
    s.buses["scream"] = send_delay(s.buses["scream"], 0.34, s.step * 6, feedback=0.46, damp=2800)
    s.buses["scream"] = send_reverb(s.buses["scream"], 1.15, rt60=4.4, damp=0.6, hp=210,
                                    predelay=0.045, seed=101)
    s.buses["scream"] = widen(s.buses["scream"], 0.45, 15.0)
    s.buses["scream"] = biquad(s.buses["scream"], "hp", 160, 0.7)

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
    s.apply_duck(["bass", "pad"], mid)
    s.apply_duck(["lead", "voice", "scream", "fx", "drums", "air"], light)

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
