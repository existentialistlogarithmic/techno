"""Industrial metal, intimate breath, processed speech and transition effects.

The sound-design layer that turns a working techno arrangement into a record:
struck metal instead of sampled percussion, breath between the drums, a voice
that has been through a room, and the effects that move between sections.
"""

import numpy as np

from .dsp import (SR, TWO_PI, n_samples, load_wav, noise, sine, saw, square, supersaw,
                  biquad, sweep, ladder, comb, vocoder, env_exp, env_ar, env_curve,
                  tanh_drive, drive_os, waveshape, soft_clip, foldback, reverb, delay,
                  widen, normalize, pitch_shift_naive, varispeed, tape_stop, gate,
                  transient_shape, compress, env_follow, resample_poly, fit)

# struck-metal partial ratios: deliberately inharmonic, which is what makes
# it read as metal rather than as a pitched instrument
BAR_RATIOS = (1.0, 2.756, 5.404, 8.933, 13.34, 18.64)
PLATE_RATIOS = (1.0, 1.412, 1.933, 2.571, 3.114, 4.231, 5.107, 6.44)
CAN_RATIOS = (1.0, 1.21, 1.87, 2.34, 3.02, 3.77, 4.91)


# ============================================================ INDUSTRIAL

def metal(dur=1.1, tune=520.0, ratios=PLATE_RATIOS, decay=0.45, noise_amt=0.35,
          bright=1.0, seed=0, drive=3.0):
    """Struck metal: inharmonic partials with per-partial decay, plus a
    noise transient shaped by the same resonances."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    r = np.random.default_rng(seed + 7)
    x = np.zeros(n)
    for i, ratio in enumerate(ratios):
        f = tune * ratio * (1.0 + r.uniform(-0.012, 0.012))
        if f > SR * 0.47:
            continue
        g = bright ** i / (1.0 + 0.55 * i)
        d = decay / (1.0 + 0.42 * i)          # highs die first, as they do
        x += g * np.sin(TWO_PI * f * t + r.random() * TWO_PI) * np.exp(-t / d)
    if noise_amt:
        nz = noise(n, seed=seed + 19) * env_exp(n, 0.004, 0.0002)
        for ratio in ratios[:4]:
            f = tune * ratio
            if f < SR * 0.45:
                x += noise_amt * 0.5 * biquad(nz, "bp", f, 14.0)
    x = biquad(x, "hp", 180, 0.7)
    return normalize(drive_os(x, drive), 0.85)


def clang(dur=2.6, tune=155.0, seed=0, drive=5.0):
    """Big industrial impact: a struck plate over a low boom."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    body = metal(dur, tune, PLATE_RATIOS, decay=0.85, noise_amt=0.45, seed=seed)
    low = np.sin(TWO_PI * np.cumsum(tune * 0.32 + 180 * np.exp(-t / 0.035)) / SR)
    low *= np.exp(-t / 0.30)
    hit = biquad(noise(n, seed=seed + 3), "bp", 2600, 0.6) * env_exp(n, 0.02, 0.0004)
    x = 0.85 * body + 0.9 * low + 0.5 * hit
    return normalize(waveshape(x, drive, sym=0.1), 0.92)


def anvil(dur=0.7, tune=1150.0, seed=0):
    return metal(dur, tune, BAR_RATIOS, decay=0.22, noise_amt=0.5, bright=0.85,
                 seed=seed, drive=4.5)


def chain(dur=1.0, hits=16, tune=2400.0, seed=0, spread=0.9):
    """Rattling chain: a scatter of tiny metal ticks."""
    n = n_samples(dur)
    r = np.random.default_rng(seed + 41)
    x = np.zeros(n)
    for i in range(hits):
        pos = n_samples(r.uniform(0, dur * spread))
        ln = n_samples(r.uniform(0.02, 0.09))
        if pos + ln >= n:
            continue
        seg = metal(ln / SR, tune * r.uniform(0.7, 1.5), CAN_RATIOS,
                    decay=0.03, noise_amt=0.6, seed=seed + i, drive=2.0)
        x[pos:pos + len(seg)] += seg * r.uniform(0.3, 1.0)
    return normalize(biquad(x, "hp", 900, 0.7), 0.8)


def steam(dur=1.6, seed=0, pitch=1.0):
    """Pressure release: noise through a closing constriction."""
    n = n_samples(dur)
    p = np.arange(n) / n
    f = (5200 * pitch) * (0.35 + 0.65 * np.exp(-p * 2.2))
    x = sweep(noise(n, seed=seed + 55), "bp", f, 1.4, stages=2)
    x += 0.4 * sweep(noise(n, seed=seed + 56), "hp", f * 1.4, 0.7)
    x *= env_curve([(0, 0.0), (0.04, 1.0), (0.45, 0.65), (1.0, 0.0)], n) ** 1.2
    return normalize(x, 0.75)


def machine(dur, rate=6.25, seed=0, tune=92.0, grit=0.6):
    """A motor in the next room: a pitched drone, a rhythmic thump and the
    rattle of something loose."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    drone = saw(tune, n) + 0.6 * saw(tune * 1.005, n) + 0.4 * square(tune * 0.5, n, 0.4)
    drone = ladder(drone, 320 + 120 * np.sin(TWO_PI * 0.07 * t), 0.5)
    pulse = np.maximum(0.0, np.sin(TWO_PI * rate * t)) ** 6
    knock = biquad(noise(n, seed=seed + 61), "bp", 220, 2.2) * pulse
    rattle = biquad(noise(n, seed=seed + 62), "bp", 3100, 3.0) * (pulse ** 2) * grit
    x = 0.7 * drone * (0.55 + 0.45 * pulse) + 0.8 * knock + 0.35 * rattle
    x = biquad(x, "hp", 45, 0.7)
    return normalize(drive_os(x, 2.4), 0.7)


def scrape(dur=0.9, seed=0, f_from=800.0, f_to=3400.0):
    """Metal dragged across metal: noise through a sliding resonator."""
    n = n_samples(dur)
    p = np.arange(n) / n
    src = noise(n, seed=seed + 71) * (0.5 + 0.5 * np.abs(np.sin(TWO_PI * 47 * p * dur)))
    f = f_from * (f_to / f_from) ** p
    x = np.zeros(n)
    for k in (1.0, 1.61, 2.29):
        x += sweep(src, "bp", np.clip(f * k, 100, 15000), 9.0)
    x *= env_curve([(0, 0), (0.1, 1), (0.8, 0.7), (1, 0)], n)
    return normalize(drive_os(x, 3.0), 0.7)


def conveyor(dur, bpm, seed=0, tune=140.0):
    """A looping mechanical texture built to sit under the groove."""
    step = 60.0 / bpm / 4.0
    n = n_samples(dur)
    x = np.zeros(n)
    r = np.random.default_rng(seed + 83)
    pat = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    for i in range(int(dur / step)):
        if not pat[i % len(pat)]:
            continue
        pos = n_samples(i * step)
        seg = metal(0.16, tune * r.uniform(0.85, 2.4), CAN_RATIOS, decay=0.045,
                    noise_amt=0.55, seed=seed + i, drive=2.5)
        ln = min(len(seg), n - pos)
        if ln > 0:
            x[pos:pos + ln] += seg[:ln] * r.uniform(0.35, 0.8)
    return normalize(biquad(x, "hp", 400, 0.7), 0.75)


# ============================================================ INTIMATE

def breath(dur=1.2, kind="out", seed=0, intensity=0.6, tone=1.0):
    """Breath: noise through a relaxed vocal tract, no voicing."""
    n = n_samples(dur)
    src = biquad(noise(n, seed=seed + 91), "lp", 6000, 0.7)
    x = np.zeros(n)
    for f, bw, g in ((520 * tone, 140, 1.0), (1100 * tone, 190, 0.55),
                     (2500 * tone, 300, 0.30), (3800 * tone, 400, 0.14)):
        x += g * biquad(src, "bp", f, max(0.7, f / bw))
    if kind == "in":
        env = env_curve([(0, 0.0), (0.35, 0.5), (0.8, 1.0), (1.0, 0.0)], n)
        x = x * env
    else:
        env = env_curve([(0, 0.0), (0.12, 1.0), (0.55, 0.7), (1.0, 0.0)], n)
        x = x * env
    x *= 0.6 + 0.4 * intensity
    return normalize(x, 0.7)


def sigh(dur=1.8, root=55, seed=0, breathiness=0.55):
    """A voiced exhale that falls away - the breath between the drums."""
    from .instruments import voice_morph, note_hz
    f = note_hz(root)
    x = voice_morph(dur, [(0, f * 1.05), (0.3, f), (1.0, f * 0.84)],
                    [(0.0, "ah"), (0.45, "uh"), (1.0, "oo")],
                    breath=breathiness, vib=(4.6, 0.018), seed=seed, drive=1.15)
    n = len(x)
    x *= env_curve([(0, 0.0), (0.14, 1.0), (0.5, 0.75), (1.0, 0.0)], n) ** 1.2
    x += 0.35 * fit(breath(dur, "out", seed=seed + 5, intensity=0.7), n)
    return normalize(biquad(x, "hp", 120, 0.7), 0.8)


# ============================================================ SPEECH

def speech_layers(path, shift=1.0, seed=0):
    """Take a spoken line and build the layers a record would use: a dark
    close voice, a whispered double, a vocoded choir and a broken-radio bed."""
    dry = load_wav(path)
    if shift != 1.0:
        dry = pitch_shift_naive(dry, shift)
    dry = normalize(dry, 0.9)
    dry = compress(dry[None, :], thresh_db=-20, ratio=3.0, attack=0.006,
                   release=0.12, makeup_db=4.0)[0]
    n = len(dry)

    close = biquad(dry, "lp", 7000, 0.7)
    close = biquad(close, "hp", 95, 0.7)
    close = close + 0.25 * biquad(close, "bp", 190, 0.8)        # chest
    close = normalize(drive_os(close, 1.5), 0.85)

    whisper = vocoder(dry, noise(n, seed=seed + 3), bands=26, lo=200, hi=9500,
                      q=8.0, attack=0.003, release=0.028)
    whisper = normalize(biquad(whisper, "hp", 350, 0.7), 0.85)

    car = supersaw(82.4, n, voices=9, detune_cents=24) + 0.5 * square(41.2, n, 0.35)
    choir = vocoder(dry, car, bands=24, lo=160, hi=7000, q=7.0)
    choir = normalize(drive_os(choir, 2.2), 0.85)

    radio = biquad(biquad(dry, "hp", 520, 0.8), "lp", 2900, 0.8)
    radio = normalize(waveshape(radio * 2.2, 6.0, sym=0.2), 0.8)

    return {"close": close, "whisper": whisper, "choir": choir, "radio": radio}


# ============================================================ TRANSITIONS

def reverse_tail(x, rt60=2.6, damp=0.5, gain=1.0):
    """Reverse reverb: the room arrives before the sound does."""
    wet = reverb(x[::-1] if x.ndim == 1 else x[:, ::-1], rt60=rt60, damp=damp,
                 hp=200, predelay=0.0, seed=131)
    wet = wet[:, ::-1]
    return normalize(wet, 0.85) * gain


def sub_drop(dur=3.2, f_from=110.0, f_to=24.0, drive=2.2):
    n = n_samples(dur)
    p = np.arange(n) / n
    f = f_from * (f_to / f_from) ** (p ** 0.55)
    x = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-p * 1.6)
    return normalize(drive_os(x, drive), 0.95)


def noise_fall(dur=2.4, f_from=9000.0, f_to=220.0, seed=0, q=1.8):
    n = n_samples(dur)
    p = np.arange(n) / n
    f = f_from * (f_to / f_from) ** (p ** 0.8)
    x = sweep(noise(n, seed=seed + 101), "bp", f, q, stages=2)
    x *= np.exp(-p * 1.4)
    return normalize(x, 0.8)


def vinyl_brake(x, start=0.35, end_ratio=0.05):
    return tape_stop(x, start=start, end_ratio=end_ratio, curve=1.8)
