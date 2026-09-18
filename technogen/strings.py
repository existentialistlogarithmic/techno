"""Bowed strings: a violin that can carry a melody, and a section behind it.

A bowed string is close to a sawtooth (Helmholtz motion), but a plain saw
sounds like a synth. What makes it read as a violin is everything around the
saw: the resonances of a wooden box, a bow that takes time to grab the string,
vibrato that arrives after the note starts, and the fact that no two notes and
no two players are ever quite identical.
"""

import numpy as np

from .dsp import (SR, TWO_PI, n_samples, as_array, saw, noise, biquad, sweep,
                  env_curve, normalize, fit, tanh_drive, drive_os, peaking,
                  reverb, widen, lfilter)

# Body resonances of a violin: the air mode, the main wood modes, and the
# broad "bridge hill" around 2-3 kHz that gives the instrument its bite.
VIOLIN_BODY = [(275, 14, 0.85), (460, 20, 0.80), (530, 26, 0.62), (700, 45, 0.52),
               (880, 70, 0.50), (1180, 110, 0.52), (1600, 150, 0.58),
               (2200, 380, 0.95), (3000, 520, 0.80), (4200, 800, 0.45)]

# A cello's box is bigger, so everything sits lower.
CELLO_BODY = [(105, 8, 1.00), (180, 14, 0.90), (230, 18, 0.70), (310, 30, 0.50),
              (430, 55, 0.40), (620, 90, 0.34), (950, 150, 0.28),
              (1500, 400, 0.34), (2100, 500, 0.22)]


def note_hz(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _body(x, table, direct=0.55):
    out = direct * x
    for f, bw, g in table:
        out += g * biquad(x, "bp", f, max(0.7, f / bw))
    return out


def bowed(midi, dur, vel=0.75, body=VIOLIN_BODY, seed=0, glide_from=None,
          vib_rate=5.7, vib_depth=0.0055, vib_delay=0.28, bow_noise=0.10,
          attack=0.075, release=0.30, bright=1.0, tremolo=0.0, sul_pont=0.0,
          porta=0.09, swell=0.0, sob=0.0, vib_growth=0.0, strain=0.0):
    """One bowed note.

    The expressive parameters are what separate a sad note from a crying one:
    `porta` slides into the note instead of arriving at it, `vib_growth` widens
    the vibrato as the note is held, `swell` leans into the middle of the bow,
    `sob` breaks the tone into catches, and `strain` pushes slightly sharp at
    the peak the way a player does when pressing into a phrase.
    """
    n = n_samples(dur)
    if n < 64:
        return np.zeros(max(1, n))
    t = np.arange(n) / SR
    r = np.random.default_rng(seed * 977 + int(midi))

    f0 = note_hz(midi)
    track = np.full(n, f0)
    if glide_from is not None and porta > 0:
        g = min(n, n_samples(porta))
        if g > 1:
            # an S-curve, not a ramp: a finger leaves slowly and arrives fast
            k = np.linspace(0.0, 1.0, g)
            k = k * k * (3.0 - 2.0 * k)
            track[:g] = note_hz(glide_from) * (f0 / note_hz(glide_from)) ** k

    # vibrato arrives after the note has spoken, as a player's does
    onset = np.clip((t - vib_delay) / max(0.05, 0.35), 0.0, 1.0) ** 1.5
    depth = vib_depth * (1.0 + vib_growth * np.clip(t / max(0.2, dur * 0.7), 0, 1.4))
    vib = np.sin(TWO_PI * vib_rate * t + r.random() * TWO_PI) * depth * onset
    # small pitch drift: the finger is never perfectly still
    drift = biquad(r.standard_normal(n), "lp", 3.5, 0.7)
    drift /= (np.max(np.abs(drift)) + 1e-9)
    push = strain * 0.004 * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    f = track * (1.0 + vib + push + 0.0016 * drift)

    src = saw(f, n)
    # bow pressure sets how many harmonics survive
    cutoff = np.clip(f0 * (10.0 + 30.0 * vel * bright), 1200, 15000)
    src = biquad(src, "lp", cutoff, 0.7)

    # the bow itself: friction noise, loudest while the string is being grabbed
    grab = np.exp(-t / max(0.01, attack * 0.8))
    bn = biquad(r.standard_normal(n), "bp", 2800 + 4000 * sul_pont, 0.8)
    bn += 0.5 * biquad(r.standard_normal(n), "hp", 5000, 0.7)
    src = src + bow_noise * (0.35 + 1.5 * grab) * bn

    x = _body(src, body, direct=0.55 + 0.22 * sul_pont)
    x = peaking(x, 2800, 0.8, 3.5)          # bridge hill: the violin's bite

    amp = env_curve([(0.0, 0.0), (min(0.5, attack / dur), 1.0),
                     (max(0.55, 1.0 - release / dur), 0.88), (1.0, 0.0)], n)
    # bow speed wavers; that waver is most of what "expressive" means
    amp *= 1.0 + 0.09 * biquad(r.standard_normal(n), "lp", 2.2, 0.7) / 3.0
    if swell:
        # lean into the middle of the bow and fall away
        amp *= 1.0 + swell * (np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.6 - 0.35)
    if sob:
        # the catch in the voice, arriving only once the note is established
        catch = np.clip((t - 0.25 * dur) / max(0.1, 0.35 * dur), 0, 1)
        amp *= 1.0 - sob * 0.38 * catch * (0.5 + 0.5 * np.sin(TWO_PI * 5.2 * t))
    if tremolo:
        amp *= 1.0 - tremolo * 0.5 * (0.5 + 0.5 * np.sin(TWO_PI * 7.5 * t))
    x = x * amp * (0.35 + 0.85 * vel)
    x = biquad(x, "hp", note_hz(midi) * 0.55, 0.7)
    return normalize(x, 0.9) * (0.4 + 0.6 * vel)


def phrase(notes, bpm, beat_unit=1.0, legato=1.12, vel=0.75, body=VIOLIN_BODY,
           seed=0, glide=True, rubato=0.0, **kw):
    """Play a melody. notes is [(midi|None, beats), ...].

    `rubato` lets the phrase breathe: a player does not place notes on a grid,
    and a metronomic lament sounds like a sequencer.
    """
    beat = 60.0 / bpm * beat_unit
    rr = np.random.default_rng(seed + 77)
    total = sum(b for _, b in notes) * beat
    n = n_samples(total) + n_samples(1.5)
    out = np.zeros(n)
    pos = 0.0
    prev = None
    for midi, beats in notes:
        ln = beats * beat * (1.0 + rubato * rr.normal(0, 0.055))
        if midi is not None:
            x = bowed(midi, ln * legato, vel=vel, body=body, seed=seed,
                      glide_from=prev if (glide and prev is not None) else None, **kw)
            i = n_samples(pos)
            k = min(len(x), n - i)
            if k > 0:
                out[i:i + k] += x[:k]
            prev = midi
        else:
            prev = None
        pos += ln
    return normalize(out, 0.9)


def ensemble(notes, bpm, players=4, spread_cents=6.0, timing_ms=18.0,
             body=VIOLIN_BODY, vel=0.68, seed=0, **kw):
    """A section, not a chorus effect: each player has their own vibrato rate,
    tuning and entry time."""
    parts = []
    r = np.random.default_rng(seed + 313)
    for i in range(players):
        det = r.normal(0, spread_cents) / 1200.0
        shifted = [(None if m is None else m + det * 12.0, b) for m, b in notes]
        x = phrase(shifted, bpm, body=body, vel=vel * r.uniform(0.85, 1.05),
                   seed=seed * 31 + i, vib_rate=5.2 + r.uniform(0, 1.4),
                   vib_depth=0.0045 + r.uniform(0, 0.003),
                   vib_delay=0.22 + r.uniform(0, 0.18), **kw)
        d = n_samples(abs(r.normal(0, timing_ms / 1000.0)))
        parts.append(np.concatenate([np.zeros(d), x]))
    n = max(len(x) for x in parts)
    out = np.zeros(n)
    for x in parts:
        out[:len(x)] += x
    return normalize(out / players, 0.9)


def chords(seq, bpm, body=VIOLIN_BODY, players=3, vel=0.6, seed=0, **kw):
    """seq is [([midi, ...], beats), ...] - sustained harmony."""
    beat = 60.0 / bpm
    total = sum(b for _, b in seq) * beat
    n = n_samples(total) + n_samples(2.5)
    out = np.zeros(n)
    pos = 0.0
    for voicing, beats in seq:
        for j, m in enumerate(voicing):
            x = ensemble([(m, beats)], bpm, players=players, body=body,
                         vel=vel, seed=seed * 7 + j, **kw)
            i = n_samples(pos)
            k = min(len(x), n - i)
            if k > 0:
                out[i:i + k] += x[:k] / max(1, len(voicing)) ** 0.5
        pos += beats * beat
    return normalize(out, 0.9)


def desecrate(x, amount=1.0, cutoff=2600.0, drive=6.0):
    """Put the strings through the machine: what the rave does to the lament."""
    y = biquad(x, "lp", cutoff, 0.8)
    y = drive_os(y * (1.0 + 3.0 * amount), drive)
    y = y + amount * 0.5 * biquad(y, "bp", 900, 1.2)
    y = biquad(y, "hp", 110, 0.7)
    return normalize((1.0 - amount) * x + amount * y, 0.9)
