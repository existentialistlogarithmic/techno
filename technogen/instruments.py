"""Sound design: every voice used in the track, all synthesised from scratch."""

import numpy as np
from .dsp import (SR, TWO_PI, n_samples, t_axis, db, fit, as_array, sine, saw, square, tri,
                  noise, supersaw, env_ar, env_exp, env_adsr, env_curve, biquad, sweep,
                  ladder, tanh_drive, soft_clip, hard_clip, foldback, bitcrush, waveshape,
                  reverb, delay, chorus, phaser, widen, pan, stereoize, normalize,
                  pitch_shift_naive, compress, drive_os, clip_os, oversampled)


def note_hz(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def formant_sweep(x, tracks, bws, gains):
    """Parallel resonators with time-varying centre frequencies."""
    out = np.zeros_like(x)
    for f, bw, g in zip(tracks, bws, gains):
        fa = as_array(f, x.shape[-1])
        out += g * sweep(x, "bp", fa, np.maximum(0.7, fa / bw))
    return out


# ============================================================ DRUMS

def kick(dur=0.62, tune=48.0, punch=1.0, drive=7.0, decay=0.13, click=1.0, dirt=0.6):
    """Hard techno kick: steep pitch drop, saturated body, crunchy transient."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    f = (tune
         + (330.0 * punch) * np.exp(-t / 0.011)
         + 34.0 * np.exp(-t / 0.055)
         + 9.0 * np.exp(-t / 0.22))
    body = np.sin(TWO_PI * np.cumsum(f) / SR)
    amp = np.exp(-t / decay) * (1 - np.exp(-t / 0.0007))
    body *= amp
    body += 0.28 * np.sin(2 * TWO_PI * np.cumsum(f) / SR) * np.exp(-t / 0.02)

    # transient
    tr = noise(n, seed=21) * env_exp(n, 0.0032, 0.0002)
    tr = biquad(tr, "hp", 2200, 0.8)
    tr += 0.5 * np.sin(TWO_PI * 1350 * t) * env_exp(n, 0.0025, 0.0002)
    tr += 0.35 * np.sin(TWO_PI * 620 * t) * env_exp(n, 0.008, 0.0004)

    x = body + click * 0.45 * tr
    x = drive_os(x, drive)
    x = x + dirt * 0.35 * oversampled(lambda v: foldback(v * 1.5, 0.85), x, 4)
    x = clip_os(x * 1.25, 0.78)
    x = biquad(x, "hp", 33, 0.7)
    x = biquad(x, "lp", 12500, 0.7)
    x = biquad(x, "bp", 68, 1.1) * 0.45 + x            # low-end weight
    return normalize(x, 0.95)


def kick_sub(dur=0.62, tune=46.0, decay=0.2):
    """Clean low sine used to feed the rumble bus."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    f = tune + 90.0 * np.exp(-t / 0.02)
    return np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-t / decay) * (1 - np.exp(-t / 0.001))


def hat(dur=0.06, tone=9000.0, decay=0.014, metal=0.5):
    n = n_samples(dur)
    t = np.arange(n) / SR
    x = noise(n, seed=None)
    if metal > 0:
        m = np.zeros(n)
        for r in (1.0, 1.4728, 1.9285, 2.4409, 3.0341, 3.8324):
            m += square(tone * 0.42 * r, n)
        x = (1 - metal) * x + metal * m / 6.0
    x = biquad(x, "hp", tone, 0.8)
    x = biquad(x, "bp", tone * 1.35, 0.9) * 0.6 + x
    x *= env_exp(n, decay, 0.0004)
    return normalize(drive_os(x, 2.2), 0.8)


def clap(dur=0.5, tone=1500.0, spread=0.011, body=0.3):
    n = n_samples(dur)
    x = np.zeros(n)
    for i, g in enumerate((0.7, 1.0, 0.85, 0.6)):
        d = n_samples(i * spread * (1 + 0.15 * i))
        if d < n:
            burst = noise(n - d, seed=100 + i) * env_exp(n - d, 0.0055, 0.0004)
            x[d:] += g * burst
    tail = noise(n, seed=77) * env_exp(n, 0.115, 0.004) * body
    x = x + tail
    x = biquad(x, "bp", tone, 0.85)
    x = biquad(x, "hp", 600, 0.7)
    return normalize(drive_os(x, 2.0), 0.85)


def snare(dur=0.34, tune=190.0, snap=0.75):
    n = n_samples(dur)
    t = np.arange(n) / SR
    tone = (np.sin(TWO_PI * tune * t) + 0.7 * np.sin(TWO_PI * tune * 1.48 * t)) * env_exp(n, 0.055)
    nz = biquad(noise(n, seed=9), "hp", 1400, 0.7) * env_exp(n, 0.085, 0.0008)
    x = (1 - snap) * tone + snap * nz + 0.3 * tone
    x = biquad(x, "bp", 2300, 0.5) * 0.4 + x
    return normalize(drive_os(x, 3.0), 0.85)


def ride(dur=1.1, tone=5200.0, decay=0.42):
    n = n_samples(dur)
    m = np.zeros(n)
    for r in (1.0, 1.341, 1.7285, 2.1409, 2.7341, 3.4324, 4.2):
        m += square(tone * 0.3 * r, n)
    x = m / 7.0 + 0.5 * noise(n, seed=31)
    x = biquad(x, "hp", 4200, 0.7)
    x *= env_exp(n, decay, 0.001)
    return normalize(x, 0.7)


def tom(dur=0.45, tune=110.0, decay=0.13):
    n = n_samples(dur)
    t = np.arange(n) / SR
    f = tune + tune * 1.1 * np.exp(-t / 0.03)
    x = np.sin(TWO_PI * np.cumsum(f) / SR) * env_exp(n, decay)
    x += 0.25 * noise(n, seed=44) * env_exp(n, 0.012)
    return normalize(drive_os(x, 3.5), 0.85)


def rim(dur=0.12, tune=1700.0):
    n = n_samples(dur)
    t = np.arange(n) / SR
    x = (np.sin(TWO_PI * tune * t) + np.sin(TWO_PI * tune * 1.61 * t)) * env_exp(n, 0.009)
    x += 0.4 * noise(n, seed=12) * env_exp(n, 0.004)
    return normalize(biquad(x, "hp", 900, 0.7), 0.7)


# ============================================================ SYNTHS

def hoover(midi, dur, detune=26.0, sweep_from=1.55, glide=0.11,
           cutoff=(700, 5200), res=0.72, drive=5.0, phase_rate=0.3, pwm=0.55):
    """Dominator-style hoover: detuned saw stack + swept phaser + dirt."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    f0 = note_hz(midi)
    bend = 1.0 + (sweep_from - 1.0) * np.exp(-t / glide)
    f = f0 * bend
    x = supersaw(f, n, voices=9, detune_cents=detune)
    x += pwm * square(f * 0.5, n, duty=0.5 + 0.35 * np.sin(TWO_PI * 0.7 * t))
    x += 0.4 * saw(f * 2.0, n)
    env = env_ar(n, 0.006, dur * 0.55, curve=2.2)
    co = cutoff[0] + (cutoff[1] - cutoff[0]) * env_curve([(0, 0.15), (0.06, 1.0), (1.0, 0.25)], n)
    x = ladder(x, co, res)
    x = phaser(x, rate=phase_rate, stages=6, depth=(300, 2800), mix=0.85)
    x = waveshape(x, drive, sym=0.12)
    x = biquad(x, "hp", 150, 0.7)
    x *= env
    return normalize(x, 0.9)


def screech(dur, base=900.0, top=4200.0, wobble=7.0, res=0.93, drive=9.0,
            direction="up", seed=0, grit=0.5):
    """The scream: hyper-resonant swept band, self-oscillating and distorted."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    p = t / max(1e-6, dur)
    ramp = p if direction == "up" else (1 - p)
    if direction == "updown":
        ramp = 1 - np.abs(2 * p - 1)
    f = base * (top / base) ** ramp
    f *= 1.0 + 0.09 * np.sin(TWO_PI * wobble * t) + 0.04 * np.sin(TWO_PI * wobble * 2.7 * t)
    src = supersaw(f, n, voices=7, detune_cents=34, spread_seed=seed + 3)
    src += 0.6 * square(f * 0.5, n, duty=0.35)
    src += grit * 0.35 * noise(n, seed=seed + 11)
    co = f * 2.1
    x = sweep(src, "bp", np.clip(co, 120, 15000), 3.0 + 14.0 * res, stages=2)
    x = drive_os(x * 3.0, drive)
    x = sweep(x, "bp", np.clip(co * 1.02, 120, 16000), 2.0 + 8.0 * res)
    x = biquad(x, "hp", 300, 0.7)
    x *= env_ar(n, 0.004, dur * 0.4, curve=2.0)
    return normalize(x, 0.85)


def acid(seq, bpm, cutoff=430.0, env_mod=3200.0, res=0.82, drive=6.0, decay=0.22,
         wave="saw", glide=0.055):
    """TB-303-ish line. seq = [(midi|None, steps, accent, slide), ...] in 16ths."""
    step = 60.0 / bpm / 4.0
    total = sum(s[1] for s in seq) * step
    n = n_samples(total)
    freqs = np.zeros(n)
    gate = np.zeros(n)
    accent = np.zeros(n)
    pos = 0
    last_f = note_hz(seq[0][0] if seq[0][0] else 36)
    for midi, steps, acc, slide in seq:
        ln = n_samples(steps * step)
        if pos >= n:
            break
        ln = min(ln, n - pos)
        f = note_hz(midi) if midi is not None else last_f
        g = n_samples(glide) if slide else n_samples(0.004)
        g = max(1, min(g, ln))
        freqs[pos:pos + ln] = f
        freqs[pos:pos + g] = np.linspace(last_f, f, g)
        if midi is not None:
            gate[pos:pos + ln] = env_exp(ln, decay if not slide else decay * 2.2, 0.002)
            accent[pos:pos + ln] = 1.0 if acc else 0.0
        last_f = f
        pos += ln
    freqs = np.maximum(freqs, 20.0)
    osc = saw(freqs, n) if wave == "saw" else square(freqs, n, 0.5)
    osc += 0.35 * saw(freqs * 0.5, n)
    co = cutoff + env_mod * gate * (1 + 1.1 * accent) + 0.0
    x = ladder(osc * (0.75 + 0.45 * accent), np.clip(co, 60, 13000), res)
    x = drive_os(x * 1.6, drive)
    x = biquad(x, "hp", 55, 0.7)
    x *= (gate > 0) * (0.7 + 0.5 * accent)
    return normalize(x, 0.9)


def sub_bass(seq, bpm, drive=2.0, decay=0.45):
    step = 60.0 / bpm / 4.0
    total = sum(s[1] for s in seq) * step
    n = n_samples(total)
    out = np.zeros(n)
    pos = 0
    for midi, steps in seq:
        ln = min(n_samples(steps * step), n - pos)
        if ln <= 0:
            break
        if midi is not None:
            t = np.arange(ln) / SR
            f = note_hz(midi)
            e = env_ar(ln, 0.008, decay, 2.5)
            out[pos:pos + ln] += np.sin(TWO_PI * f * t) * e
        pos += ln
    out = tanh_drive(out, drive)
    return biquad(out, "lp", 180, 0.7)


def stab(midi, dur=0.28, detune=18.0, cutoff=2600, res=0.7, drive=5.0):
    n = n_samples(dur)
    f = note_hz(midi)
    x = supersaw(f, n, voices=7, detune_cents=detune)
    x += 0.5 * square(f, n, 0.3)
    env = env_exp(n, dur * 0.28, 0.002)
    co = cutoff * env_curve([(0, 1.0), (0.25, 0.45), (1, 0.3)], n)
    x = ladder(x, co, res) * env
    return normalize(drive_os(x, drive), 0.85)


def pad(midis, dur, detune=14.0, cutoff=1500.0, movement=0.12, drive=1.4):
    n = n_samples(dur)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for i, m in enumerate(midis):
        f = note_hz(m) * (1 + 0.0009 * i)
        x += supersaw(f, n, voices=5, detune_cents=detune, spread_seed=i + 2)
    x /= len(midis)
    co = cutoff * (1 + movement * np.sin(TWO_PI * 0.07 * t + 1.0))
    x = ladder(x, co, 0.45)
    x = drive_os(x, drive)
    x *= env_ar(n, dur * 0.25, dur * 0.5, curve=1.6)
    return normalize(x, 0.7)


def rumble_from(sub_bus, rt60=1.9, cut=190.0, drive=2.6):
    """Kick -> big reverb -> lowpass: the low-end churn under hard techno."""
    wet = reverb(sub_bus, rt60=rt60, predelay=0.004, damp=0.85, hp=28.0, seed=17, width=0.35)
    wet = biquad(wet, "lp", cut, 0.8)
    wet = biquad(wet, "hp", 34, 0.7)
    wet = drive_os(wet, drive)
    return wet


# ============================================================ VOICE

VOWELS = {
    "ah": ([730, 1090, 2440, 3400], [90, 110, 170, 250], [1.0, 0.55, 0.30, 0.12]),
    "oh": ([570, 840, 2410, 3300], [80, 100, 170, 250], [1.0, 0.45, 0.16, 0.06]),
    "oo": ([300, 870, 2240, 3300], [70, 90, 170, 250], [1.0, 0.30, 0.10, 0.05]),
    "eh": ([530, 1840, 2480, 3500], [80, 100, 170, 250], [1.0, 0.60, 0.35, 0.10]),
    "ee": ([270, 2290, 3010, 3600], [70, 110, 180, 260], [1.0, 0.45, 0.30, 0.10]),
    "uh": ([640, 1190, 2390, 3300], [85, 105, 170, 250], [1.0, 0.50, 0.25, 0.08]),
    "l":  ([360, 1200, 2700, 3300], [70, 110, 180, 250], [1.0, 0.35, 0.20, 0.06]),
}


def _glottal(f0_arr, n, breath=0.2, seed=0):
    src = saw(f0_arr, n)
    src = biquad(src, "lp", 2600, 0.7)
    src = biquad(src, "lp", 4200, 0.7)
    src += breath * biquad(noise(n, seed=seed + 5), "bp", 2000, 0.5)
    return src


def voice_morph(dur, f0_pts, vowel_path, breath=0.22, vib=(5.2, 0.022),
                seed=0, drive=1.3, growl=0.0):
    """Formant-morphing vocal tone - the engine behind moans and vocal fragments."""
    n = n_samples(dur)
    t = np.arange(n) / SR
    f0 = env_curve(f0_pts, n)
    f0 = f0 * (1 + vib[1] * np.sin(TWO_PI * vib[0] * t) * np.clip(t / (0.25 * dur), 0, 1))
    if growl:
        f0 = f0 * (1 + growl * 0.06 * np.sin(TWO_PI * 31.0 * t))
    src = _glottal(f0, n, breath, seed)
    tracks, bws, gains = [], [], []
    ks = [v[1] for v in vowel_path]
    for i in range(4):
        pts = [(p, VOWELS[v][0][i]) for p, v in vowel_path]
        tracks.append(env_curve(pts, n))
        bws.append(VOWELS[ks[0]][1][i] * 1.15)
        gains.append(np.mean([VOWELS[v][2][i] for _, v in vowel_path]))
    x = formant_sweep(src, tracks, bws, gains)
    x += 0.10 * biquad(src, "hp", 4200, 0.7)
    x = tanh_drive(x, drive)
    return normalize(x, 0.85)


def moan(dur=2.4, root=54, seed=0, breath=0.3, up=True):
    """Breathy, pitch-gliding vocal swell: oo -> ah -> oo."""
    f = note_hz(root)
    f0 = ([(0, f * 0.88), (0.25, f), (0.6, f * 1.10), (1.0, f * 0.94)] if up
          else [(0, f * 1.06), (0.35, f * 0.98), (1.0, f * 0.86)])
    path = [(0.0, "oo"), (0.35, "ah"), (0.7, "ah"), (1.0, "oo")]
    x = voice_morph(dur, f0, path, breath=breath, vib=(5.0 + 0.6 * (seed % 2), 0.03),
                    seed=seed, drive=1.25)
    n = len(x)
    x *= env_curve([(0, 0.0), (0.22, 1.0), (0.62, 0.9), (1.0, 0.0)], n) ** 1.3
    x = biquad(x, "hp", 110, 0.7)
    return normalize(x, 0.8)


def vocal_chatter(dur, syllables=6, root=50, seed=1, breath=0.35, scatter=0.5):
    """Unintelligible distant voice: vowel babble with syllabic gating."""
    n = n_samples(dur)
    r = np.random.default_rng(seed)
    out = np.zeros(n)
    pos = 0
    vs = list(VOWELS.keys())
    for i in range(syllables):
        ln = n_samples(r.uniform(0.09, 0.26))
        if pos + ln >= n:
            break
        f = note_hz(root + r.integers(-3, 5))
        v1, v2 = vs[r.integers(0, len(vs))], vs[r.integers(0, len(vs))]
        seg = voice_morph(ln / SR, [(0, f * 1.02), (1, f * 0.93)],
                          [(0.0, v1), (1.0, v2)], breath=breath, vib=(6.0, 0.01),
                          seed=seed * 7 + i)
        seg *= env_ar(len(seg), 0.02, 0.07, 2.5)
        out[pos:pos + len(seg)] += seg * r.uniform(0.6, 1.0)
        pos += ln + n_samples(r.uniform(0.02, 0.18) * scatter)
    return normalize(out, 0.8)


def vocal_stab(dur=0.42, root=57, vowel="eh", drive=6.0, seed=2):
    """Short shouted hit - pitched down and mangled."""
    f = note_hz(root)
    x = voice_morph(dur, [(0, f * 1.25), (0.12, f), (1.0, f * 0.9)],
                    [(0.0, "ah"), (0.4, vowel), (1.0, vowel)], breath=0.18,
                    vib=(6.5, 0.015), seed=seed, drive=2.0)
    n = len(x)
    x *= env_ar(n, 0.006, dur * 0.4, 2.8)
    x = tanh_drive(x * 1.8, drive)
    x = biquad(x, "bp", 1200, 0.5) * 0.5 + x
    return normalize(x, 0.85)


def scream(dur=1.5, f0=(250.0, 520.0, 340.0), vowel_path=((0.0, "eh"), (1.0, "ah")),
           roughness=0.55, breath=0.45, drive=6.0, seed=0, effort=1.0,
           shout=0.25, rasp=0.12, hiss=0.15, edge=0.60):
    """A shouted, distressed voice.

    A scream is not just a loud vowel. It needs pitch an octave above speech,
    irregular pitch (jitter), a subharmonic rattle where the vocal folds stop
    tracking, turbulent noise from the constriction, and the nonlinearity of a
    vocal tract being driven far past its linear range. All four, or it reads
    as singing.
    """
    n = n_samples(dur)
    t = np.arange(n) / SR
    r = np.random.default_rng(seed + 991)

    contour = env_curve([(0.0, f0[0]), (0.10, f0[1]), (0.55, f0[1] * 0.97),
                         (1.0, f0[2])], n)
    jitter = biquad(r.standard_normal(n), "lp", 11.0, 0.7)
    jitter /= (np.max(np.abs(jitter)) + 1e-9)
    contour *= 1.0 + 0.045 * effort * jitter
    contour *= 1.0 + 0.022 * np.sin(TWO_PI * 5.6 * t) * np.clip(t / (0.3 * dur), 0, 1)

    ph = np.cumsum(contour) / SR
    src = 2.0 * (ph % 1.0) - 1.0                       # bright saw, no smoothing
    src += 0.55 * np.where((ph % 1.0) < 0.22, 1.0, -1.0)
    # diplophonia: the fold rattle that makes a shout sound strained
    src += roughness * 0.45 * (2.0 * ((ph * 0.5) % 1.0) - 1.0)
    shimmer = biquad(r.standard_normal(n), "lp", 26.0, 0.7)
    shimmer /= (np.max(np.abs(shimmer)) + 1e-9)
    src *= 1.0 + roughness * 0.30 * shimmer
    src = biquad(src, "lp", 8500, 0.7)
    # turbulence at the constriction
    src += breath * 1.4 * biquad(noise(n, seed=seed + 5), "bp", 2600, 0.35)

    tracks, bws, gains = [], [], []
    for i in range(4):
        pts = [(pos, VOWELS[v][0][i]) for pos, v in vowel_path]
        tracks.append(env_curve(pts, n))
        bws.append(VOWELS[vowel_path[0][1]][1][i] * 1.45)
        gains.append(np.mean([VOWELS[v][2][i] for _, v in vowel_path]))
    src = normalize(src, 0.9)
    x = normalize(formant_sweep(src, tracks, bws, gains), 0.9)

    # A screamed vowel at 600 Hz has its formants sitting between harmonics,
    # so the resonators alone just ring on the fundamental. The upper bands
    # have to be built from the source: the shout formant near 3 kHz, plus a
    # distorted high band for the rasp.
    x += shout * effort * normalize(biquad(src, "bp", 3050, 3.5), 0.9)
    x += rasp * effort * normalize(biquad(src, "bp", 4600, 2.5), 0.9)
    x += hiss * normalize(biquad(noise(n, seed=seed + 61), "hp", 3500, 0.7), 0.9)
    x = normalize(x, 0.9)

    x = waveshape(x * 3.0, drive, sym=0.18)
    x = x + edge * effort * tanh_drive(biquad(x, "hp", 1600, 0.7) * 5.0, 8.0)
    x = biquad(x, "hp", 190, 0.7)
    x = biquad(x, "lp", 13000, 0.7)
    return normalize(x, 0.9)


def scream_help(dur=1.6, pitch=1.0, seed=0, drive=6.5, effort=1.0):
    """The word shaped as h-eh-l-p: aspiration, the vowel, the lateral,
    a lip closure and the release burst."""
    n = n_samples(dur)
    voiced = scream(dur, f0=(250 * pitch, 520 * pitch, 350 * pitch),
                    vowel_path=((0.0, "eh"), (0.45, "eh"), (0.72, "l"), (1.0, "l")),
                    roughness=0.6, breath=0.4, drive=drive, seed=seed, effort=effort,
                    shout=0.38, rasp=0.18, edge=0.75)
    # amplitude: /h/ swell, vowel, lateral, then the closure gap before /p/
    amp = env_curve([(0.0, 0.0), (0.05, 0.55), (0.12, 1.0), (0.45, 0.92),
                     (0.62, 0.68), (0.78, 0.30), (0.82, 0.0), (1.0, 0.0)], n)
    x = voiced * amp

    a = n_samples(dur * 0.06)                          # /h/ aspiration
    asp = biquad(noise(a, seed=seed + 31), "bp", 1900, 0.7) * env_curve(
        [(0, 0.2), (0.4, 1.0), (1, 0.3)], a)
    x[:a] += 0.30 * asp

    b0 = n_samples(dur * 0.88)                         # /p/ release
    bl = min(n - b0, n_samples(0.035))
    if bl > 0:
        burst = biquad(noise(bl, seed=seed + 47), "bp", 1200, 0.5) * env_exp(bl, 0.006, 0.0004)
        x[b0:b0 + bl] += 0.45 * burst
    return normalize(x, 0.9)


def whisper(dur, seed=3, tone=1.0):
    """Noise-only vocal tract: breathy warehouse ghost."""
    n = n_samples(dur)
    src = biquad(noise(n, seed=seed), "lp", 5000, 0.7)
    r = np.random.default_rng(seed)
    tracks, bws, gains = [], [], []
    vs = ["uh", "ah", "ee", "oo", "eh"]
    picks = [vs[r.integers(0, len(vs))] for _ in range(4)]
    for i in range(4):
        pts = [(j / 3.0, VOWELS[picks[j]][0][i] * tone) for j in range(4)]
        tracks.append(env_curve(pts, n))
        bws.append(130.0)
        gains.append([1.0, 0.6, 0.35, 0.15][i])
    x = formant_sweep(src, tracks, bws, gains)
    x *= env_curve([(0, 0), (0.15, 1), (0.5, 0.6), (0.8, 0.9), (1, 0)], n)
    x *= (0.5 + 0.5 * np.abs(np.sin(TWO_PI * 2.3 * np.arange(n) / SR)))
    return normalize(x, 0.7)


# ============================================================ FX

def riser_noise(dur, f_from=300.0, f_to=11000.0, q=2.5, swell=2.0, seed=4):
    n = n_samples(dur)
    p = np.arange(n) / n
    f = f_from * (f_to / f_from) ** (p ** 1.6)
    x = noise(n, seed=seed)
    x = sweep(x, "bp", f, q, stages=2)
    x += 0.4 * sweep(noise(n, seed=seed + 1), "hp", f * 0.6, 0.7)
    x *= p ** swell
    return normalize(x, 0.8)


def riser_tone(dur, midi_from=38, midi_to=74, detune=20.0, drive=3.0):
    n = n_samples(dur)
    p = np.arange(n) / n
    f = note_hz(midi_from) * (note_hz(midi_to) / note_hz(midi_from)) ** (p ** 1.4)
    x = supersaw(f, n, voices=7, detune_cents=detune)
    x = ladder(x, np.clip(f * 6, 300, 14000), 0.7)
    x = drive_os(x, drive)
    x *= p ** 1.8
    return normalize(x, 0.8)


def downlifter(dur=2.2, f_from=2400.0, f_to=60.0, seed=6):
    n = n_samples(dur)
    p = np.arange(n) / n
    f = f_from * (f_to / f_from) ** (p ** 0.7)
    x = np.sin(TWO_PI * np.cumsum(f) / SR)
    x += 0.5 * sweep(noise(n, seed=seed), "bp", f * 2.5, 3.0)
    x *= np.exp(-p * 2.4)
    return normalize(drive_os(x, 2.5), 0.85)


def impact(dur=2.6, tune=44.0, seed=8):
    n = n_samples(dur)
    t = np.arange(n) / SR
    f = tune + 260 * np.exp(-t / 0.03)
    x = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-t / 0.45)
    x += 0.55 * biquad(noise(n, seed=seed), "lp", 2200, 0.7) * np.exp(-t / 0.16)
    x += 0.3 * biquad(noise(n, seed=seed + 1), "hp", 4000, 0.7) * np.exp(-t / 0.05)
    return normalize(tanh_drive(x, 3.0), 0.95)


def reverse_swell(dur=2.0, seed=9, bright=7000.0):
    n = n_samples(dur)
    x = noise(n, seed=seed) * env_exp(n, dur * 0.35)
    x = biquad(x, "hp", 1200, 0.7)
    x = biquad(x, "bp", bright, 0.6) * 0.6 + x
    return normalize(x[::-1], 0.8)


def room_tone(dur, level=1.0, seed=13):
    n = n_samples(dur)
    t = np.arange(n) / SR
    x = noise(n, seed=seed)
    x = biquad(x, "lp", 900, 0.7)
    x = biquad(x, "hp", 60, 0.7)
    x *= 0.6 + 0.4 * np.sin(TWO_PI * 0.05 * t + 0.7)
    hum = 0.35 * np.sin(TWO_PI * 50 * t) + 0.12 * np.sin(TWO_PI * 100 * t)
    return (x * 0.9 + hum * 0.4) * level


def zap(dur=0.35, f_from=5200.0, f_to=180.0, seed=15):
    n = n_samples(dur)
    p = np.arange(n) / n
    f = f_from * (f_to / f_from) ** p
    x = square(f, n, 0.4) + 0.5 * noise(n, seed=seed)
    x = sweep(x, "bp", f * 1.6, 6.0)
    x *= np.exp(-p * 4.0)
    return normalize(drive_os(x, 5.0), 0.8)
