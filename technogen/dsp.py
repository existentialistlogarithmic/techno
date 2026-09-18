"""Core DSP primitives: oscillators, envelopes, filters, distortion, space."""

import numpy as np
from scipy.signal import lfilter, oaconvolve, resample_poly

SR = 44100
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- utilities

def n_samples(dur):
    return int(round(dur * SR))


def t_axis(dur):
    return np.arange(n_samples(dur)) / SR


def db(x):
    return 10.0 ** (x / 20.0)


def fit(x, n):
    """Trim or zero-pad a signal (mono or stereo) to exactly n samples."""
    if x.ndim == 1:
        if len(x) >= n:
            return x[:n]
        return np.concatenate([x, np.zeros(n - len(x))])
    if x.shape[1] >= n:
        return x[:, :n]
    return np.concatenate([x, np.zeros((x.shape[0], n - x.shape[1]))], axis=1)


def as_array(v, n):
    if np.isscalar(v):
        return np.full(n, float(v))
    v = np.asarray(v, dtype=float)
    return fit(v, n)


# -------------------------------------------------------------- oscillators

def phasor(freq, n, phase0=0.0):
    f = as_array(freq, n)
    return (np.cumsum(f) / SR + phase0) % 1.0


def sine(freq, n, phase0=0.0):
    return np.sin(TWO_PI * phasor(freq, n, phase0))


def _polyblep(t, dt):
    """Correction around a phase discontinuity, so the step is band-limited."""
    out = np.zeros_like(t)
    lo = t < dt
    if np.any(lo):
        u = t[lo] / dt[lo]
        out[lo] = u + u - u * u - 1.0
    hi = t > 1.0 - dt
    if np.any(hi):
        u = (t[hi] - 1.0) / dt[hi]
        out[hi] = u * u + u + u + 1.0
    return out


def saw(freq, n, phase0=0.0, blep=True):
    p = phasor(freq, n, phase0)
    y = 2.0 * p - 1.0
    if blep:
        dt = np.clip(as_array(freq, n) / SR, 1e-7, 0.45)
        y -= _polyblep(p, dt)
    return y


def square(freq, n, duty=0.5, phase0=0.0, blep=True):
    p = phasor(freq, n, phase0)
    d = as_array(duty, n)
    y = np.where(p < d, 1.0, -1.0)
    if blep:
        dt = np.clip(as_array(freq, n) / SR, 1e-7, 0.45)
        y += _polyblep(p, dt)
        y -= _polyblep((p - d) % 1.0, dt)
    return y


def tri(freq, n, phase0=0.0):
    p = phasor(freq, n, phase0)
    return 4.0 * np.abs(p - 0.5) - 1.0


_rng = np.random.default_rng(1312)


def noise(n, seed=None):
    r = _rng if seed is None else np.random.default_rng(seed)
    return r.standard_normal(n)


def supersaw(freq, n, voices=7, detune_cents=22.0, spread_seed=7, wave=saw):
    """Detuned oscillator stack - the guts of every hoover and screech."""
    r = np.random.default_rng(spread_seed)
    out = np.zeros(n)
    f = as_array(freq, n)
    for i in range(voices):
        k = (i - (voices - 1) / 2.0) / max(1.0, (voices - 1) / 2.0)
        ratio = 2.0 ** (k * detune_cents / 1200.0)
        out += wave(f * ratio, n, phase0=r.random())
    return out / voices


# --------------------------------------------------------------- envelopes

def env_ar(n, attack, release, curve=3.0):
    a = max(1, n_samples(attack))
    t = np.arange(n)
    e = np.ones(n)
    ai = min(a, n)
    e[:ai] = (np.arange(ai) / a) ** 0.6
    if release > 0:
        e *= np.exp(-curve * np.maximum(0.0, (t - ai) / max(1.0, n_samples(release))))
    return e


def env_exp(n, tau, attack=0.002):
    """Percussive: fast attack, exponential tail (tau = decay time constant)."""
    t = np.arange(n) / SR
    e = np.exp(-t / max(1e-5, tau))
    a = max(1, n_samples(attack))
    e[:a] *= np.linspace(0.0, 1.0, a) ** 0.5
    return e


def env_adsr(n, a, d, s, r):
    ai, di, ri = n_samples(a), n_samples(d), n_samples(r)
    si = max(0, n - ai - di - ri)
    segs = [
        np.linspace(0, 1, max(1, ai)),
        np.linspace(1, s, max(1, di)),
        np.full(si, s),
        np.linspace(s, 0, max(1, ri)),
    ]
    return fit(np.concatenate(segs), n)


def env_curve(points, n):
    """Piecewise-linear envelope from [(pos 0..1, value), ...]."""
    xs = np.array([p[0] for p in points]) * (n - 1)
    ys = np.array([p[1] for p in points])
    return np.interp(np.arange(n), xs, ys)


# ----------------------------------------------------------------- filters

def _biquad(mode, fc, q):
    fc = float(np.clip(fc, 15.0, SR * 0.47))
    q = float(max(0.1, q))
    w0 = TWO_PI * fc / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    if mode == "lp":
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif mode == "hp":
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif mode == "bp":
        b = np.array([alpha, 0.0, -alpha])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif mode == "notch":
        b = np.array([1.0, -2 * cw, 1.0])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif mode == "ap":
        b = np.array([1 - alpha, -2 * cw, 1 + alpha])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    else:
        raise ValueError(mode)
    return b / a[0], a / a[0]


def peaking(x, fc, q, gain_db):
    """RBJ peaking EQ - used by the master tilt correction."""
    if abs(gain_db) < 0.05:
        return x
    A = 10.0 ** (gain_db / 40.0)
    w0 = TWO_PI * float(np.clip(fc, 20.0, SR * 0.45)) / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * max(0.1, q))
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return lfilter(b / a[0], a / a[0], x, axis=-1)


def low_shelf(x, fc, gain_db, q=0.707):
    if abs(gain_db) < 0.05:
        return x
    A = 10.0 ** (gain_db / 40.0)
    w0 = TWO_PI * fc / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1 / A) * (1 / q - 1) + 2)
    tsa = 2 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) - (A - 1) * cw + tsa),
                  2 * A * ((A - 1) - (A + 1) * cw),
                  A * ((A + 1) - (A - 1) * cw - tsa)])
    a = np.array([(A + 1) + (A - 1) * cw + tsa,
                  -2 * ((A - 1) + (A + 1) * cw),
                  (A + 1) + (A - 1) * cw - tsa])
    return lfilter(b / a[0], a / a[0], x, axis=-1)


_bq_cache = {}


def _biquad_cached(mode, fc, q):
    key = (mode, int(round(np.log2(max(15.0, fc)) * 96)), int(round(q * 8)))
    hit = _bq_cache.get(key)
    if hit is None:
        hit = _biquad(mode, fc, q)
        _bq_cache[key] = hit
    return hit


def biquad(x, mode, fc, q=0.707, stages=1):
    """Static filter."""
    b, a = _biquad(mode, fc, q)
    y = x
    for _ in range(stages):
        y = lfilter(b, a, y, axis=-1)
    return y


def sweep(x, mode, fc, q=0.707, stages=1, block=96):
    """Time-varying filter, block-interpolated. fc may be scalar or array."""
    n = x.shape[-1]
    fcs = as_array(fc, n)
    qs = as_array(q, n)
    y = np.zeros_like(x)
    zis = [np.zeros(x.shape[:-1] + (2,)) for _ in range(stages)]
    for i in range(0, n, block):
        j = min(n, i + block)
        b, a = _biquad_cached(mode, float(fcs[i:j].mean()), float(qs[i:j].mean()))
        seg = x[..., i:j]
        for s in range(stages):
            seg, zis[s] = lfilter(b, a, seg, axis=-1, zi=zis[s])
        y[..., i:j] = seg
    return y


def ladder(x, fc, res=0.5, block=96):
    """Two cascaded LP biquads: the second carries the resonance."""
    n = x.shape[-1]
    fcs = as_array(fc, n)
    q1 = 0.707
    q2 = 0.6 + 7.0 * np.clip(as_array(res, n), 0, 1) ** 2
    y = sweep(x, "lp", fcs, q1, block=block)
    return sweep(y, "lp", fcs, q2, block=block)


def formant(x, freqs, bws, gains):
    """Parallel resonator bank - vocal tract."""
    out = np.zeros_like(x)
    for f, bw, g in zip(freqs, bws, gains):
        out += g * biquad(x, "bp", f, max(0.5, f / bw))
    return out


# ------------------------------------------------------------- distortion

def oversampled(fn, x, factor=4, axis=-1):
    """Run a nonlinearity at a higher rate so its harmonics land above
    Nyquist and get filtered out instead of folding back as aliasing."""
    if factor <= 1:
        return fn(x)
    n = x.shape[axis]
    up = resample_poly(x, factor, 1, axis=axis)
    y = fn(up)
    down = resample_poly(y, 1, factor, axis=axis)
    return np.take(down, np.arange(n), axis=axis)


def tanh_drive(x, amount=3.0, comp=True):
    y = np.tanh(x * amount)
    return y / np.tanh(amount) if comp else y


def soft_clip(x, thresh=0.7):
    a = np.abs(x)
    return np.sign(x) * np.where(a < thresh, a, thresh + (1 - thresh) * np.tanh((a - thresh) / (1 - thresh)))


def hard_clip(x, thresh=1.0):
    return np.clip(x, -thresh, thresh)


def foldback(x, thresh=0.8):
    y = x.copy()
    for _ in range(3):
        over = np.abs(y) > thresh
        if not np.any(over):
            break
        y[over] = np.sign(y[over]) * (2 * thresh - np.abs(y[over]))
    return y


def bitcrush(x, bits=8, hold=1):
    q = 2.0 ** (bits - 1)
    y = np.round(x * q) / q
    if hold > 1:
        n = x.shape[-1]
        idx = (np.arange(n) // hold) * hold
        y = y[..., idx]
    return y


def waveshape(x, drive=4.0, sym=0.0, os=4):
    """Asymmetric drive: adds even harmonics = extra dirt."""
    return oversampled(lambda v: tanh_drive(v + sym * v * v, drive), x, os)


def drive_os(x, amount=3.0, os=4):
    """Anti-aliased saturation - the default for anything with harmonics."""
    return oversampled(lambda v: tanh_drive(v, amount), x, os)


def clip_os(x, thresh=0.8, os=4):
    return oversampled(lambda v: soft_clip(v, thresh), x, os)


# ----------------------------------------------------------------- dynamics

def env_follow(x, attack=0.005, release=0.12):
    mono = x if x.ndim == 1 else np.max(np.abs(x), axis=0)
    a = np.exp(-1.0 / (attack * SR))
    r = np.exp(-1.0 / (release * SR))
    fast = lfilter([1 - a], [1, -a], np.abs(mono))
    slow = lfilter([1 - r], [1, -r], np.abs(mono))
    return np.maximum(fast, slow)


def compress(x, thresh_db=-14.0, ratio=4.0, attack=0.005, release=0.12, makeup_db=0.0):
    e = env_follow(x, attack, release) + 1e-9
    e_db = 20 * np.log10(e)
    over = np.maximum(0.0, e_db - thresh_db)
    gain = db(-over * (1 - 1 / ratio) + makeup_db)
    return x * gain


def limiter(x, ceiling=0.97, lookahead=0.003, release=0.05):
    n = x.shape[-1]
    la = n_samples(lookahead)
    mono = np.max(np.abs(x), axis=0) if x.ndim > 1 else np.abs(x)
    # sliding max over the lookahead window
    pad = np.concatenate([mono, np.zeros(la)])
    peak = np.maximum.reduceat(
        pad, np.clip(np.arange(0, len(pad), max(1, la // 2)), 0, len(pad) - 1))
    peak = np.repeat(peak, max(1, la // 2))[:n]
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-6))
    r = np.exp(-1.0 / (release * SR))
    need = -lfilter([1 - r], [1, -r], -need)  # smooth, fast down / slow up
    need = np.minimum(need, ceiling / np.maximum(mono, 1e-6))
    need = np.minimum(need, 1.0)
    return x * need


# -------------------------------------------------------------------- space

def make_ir(rt60=2.4, predelay=0.012, damp=0.55, hp=120.0, seed=5, width=1.0):
    n = n_samples(rt60)
    t = np.arange(n) / SR
    r = np.random.default_rng(seed)
    tail = r.standard_normal((2, n))
    if width < 1.0:
        m = tail.mean(axis=0)
        tail = width * tail + (1 - width) * np.stack([m, m])
    decay = np.exp(-6.9078 * t / rt60)
    build = np.clip(t / 0.02, 0, 1) ** 2
    ir = tail * decay * build
    # sparse early reflections
    for k in range(14):
        d = n_samples(r.uniform(0.004, 0.09))
        g = r.uniform(-0.55, 0.55) * np.exp(-d / (0.05 * SR))
        if d < n:
            ir[r.integers(0, 2), d] += g
    ir = biquad(ir, "lp", 1000 + 9000 * (1 - damp), 0.6)
    ir = biquad(ir, "hp", hp, 0.7)
    ir[:, 0] += 0.0
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9) * 1.0, n_samples(predelay)


_ir_cache = {}


def reverb(x, rt60=2.4, predelay=0.012, damp=0.55, hp=120.0, seed=5, width=1.0):
    key = (round(rt60, 3), round(predelay, 4), round(damp, 3), round(hp, 1), seed, round(width, 2))
    if key not in _ir_cache:
        _ir_cache[key] = make_ir(rt60, predelay, damp, hp, seed, width)
    ir, pre = _ir_cache[key]
    if x.ndim == 1:
        x = np.stack([x, x])
    n = x.shape[1]
    wet = np.stack([oaconvolve(x[c], ir[c])[:n + pre] for c in range(2)])
    if pre:
        wet = np.concatenate([np.zeros((2, pre)), wet], axis=1)
    return fit(wet, n)


def delay(x, time, feedback=0.45, mix=0.35, damp_hz=5200.0, pingpong=True):
    """Exact feedback delay computed block-by-block at the delay length."""
    stereo = x.ndim > 1
    if not stereo:
        x = np.stack([x, x])
    n = x.shape[1]
    d = max(1, n_samples(time))
    buf = x.copy()
    for i in range(d, n, d):
        j = min(n, i + d)
        src = buf[:, i - d:i - d + (j - i)]
        if pingpong:
            src = src[::-1]
        buf[:, i:j] += feedback * biquad(src, "lp", damp_hz, 0.7)
    wet = buf - x
    return x + mix * wet if stereo else (x + mix * wet)


def chorus(x, rate=0.35, depth_ms=6.0, mix=0.5, voices=3, seed=3):
    if x.ndim == 1:
        x = np.stack([x, x])
    n = x.shape[1]
    t = np.arange(n) / SR
    r = np.random.default_rng(seed)
    out = np.zeros_like(x)
    for v in range(voices):
        ph = r.random()
        mod = (depth_ms / 1000.0) * SR * (0.5 + 0.5 * np.sin(TWO_PI * rate * (1 + 0.3 * v) * t + TWO_PI * ph))
        idx = np.arange(n) - mod
        i0 = np.clip(np.floor(idx).astype(int), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        fr = np.clip(idx - i0, 0, 1)
        for c in range(2):
            out[c] += (x[c][i0] * (1 - fr) + x[c][i1] * fr) * (1 if (v + c) % 2 else 0.85)
    out /= voices
    return (1 - mix) * x + mix * out


def phaser(x, rate=0.25, stages=6, depth=(280.0, 2400.0), feedback=0.6, mix=0.8, phase0=0.0):
    """The hoover's signature: swept allpass notches."""
    n = x.shape[-1]
    t = np.arange(n) / SR
    lfo = 0.5 + 0.5 * np.sin(TWO_PI * rate * t + phase0)
    fc = depth[0] * (depth[1] / depth[0]) ** lfo
    y = x.copy()
    fb = np.zeros_like(x)
    y = y + feedback * fb
    for _ in range(stages):
        y = sweep(y, "ap", fc, 0.7)
    return (1 - mix) * x + mix * y


def widen(x, amount=0.6, delay_ms=11.0):
    if x.ndim == 1:
        x = np.stack([x, x])
    d = n_samples(delay_ms / 1000.0)
    out = x.copy()
    out[1] = np.concatenate([np.zeros(d), x[1][:-d]]) if d else x[1]
    mid = (x[0] + x[1]) * 0.5
    side = (out[0] - out[1]) * 0.5 * (1 + amount)
    return np.stack([mid + side, mid - side])


def pan(x, p=0.0):
    """p: -1 left .. +1 right. Returns stereo."""
    if x.ndim > 1:
        return x
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a) * 1.414 * 0.707, x * np.sin(a) * 1.414 * 0.707])


def stereoize(x):
    return x if x.ndim > 1 else np.stack([x, x])


def normalize(x, peak=0.99):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def pitch_shift_naive(x, ratio):
    """Resample (changes length + formants) - fine for grimy FX."""
    n = x.shape[-1]
    idx = np.arange(0, n, ratio)
    i0 = np.clip(np.floor(idx).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    fr = idx - i0
    if x.ndim == 1:
        return x[i0] * (1 - fr) + x[i1] * fr
    return np.stack([x[c][i0] * (1 - fr) + x[c][i1] * fr for c in range(2)])


# ------------------------------------------------------------------ samples

def load_wav(path, target_sr=SR, mono=True):
    """Read a wav and resample to the engine rate."""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    x = data.astype(np.float64)
    if x.dtype == np.int16 or np.max(np.abs(x)) > 2.0:
        x = x / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1) if mono else x.T
    if sr != target_sr:
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g, axis=-1)
    return x


# ----------------------------------------------------------------- vocoder

def vocoder(mod, car, bands=22, lo=140.0, hi=8500.0, q=7.0,
            attack=0.004, release=0.035, tilt=0.0):
    """Classic channel vocoder: the modulator's spectral envelope imposed on
    a carrier. Noise carrier gives whispered speech; a saw stack gives the
    industrial choir."""
    n = min(mod.shape[-1], car.shape[-1])
    mod, car = mod[..., :n], car[..., :n]
    edges = np.geomspace(lo, hi, bands + 1)
    out = np.zeros(n)
    for i in range(bands):
        fc = float(np.sqrt(edges[i] * edges[i + 1]))
        m = biquad(mod, "bp", fc, q)
        c = biquad(car, "bp", fc, q)
        env = env_follow(m, attack, release)
        out += c * env * (10.0 ** (tilt * np.log2(fc / lo) / 20.0))
    return out


# ------------------------------------------------------------- time effects

def varispeed(x, speed):
    """Resample with a per-sample playback rate (tape)."""
    n = x.shape[-1]
    sp = as_array(speed, n)
    pos = np.cumsum(sp)
    pos = pos[pos < n - 2]
    if len(pos) < 2:
        return x[..., :1] * 0.0
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    if x.ndim == 1:
        return x[i0] * (1 - fr) + x[i0 + 1] * fr
    return np.stack([x[c][i0] * (1 - fr) + x[c][i0 + 1] * fr for c in range(2)])


def tape_stop(x, start=0.55, end_ratio=0.03, curve=2.2, fade=True, max_stretch=3.0):
    """The motor cuts: pitch and time wind down together. Slowing down means
    more output samples than input, so the ramp is laid out in output time."""
    n = x.shape[-1]
    m = int(n * max_stretch)
    t = np.linspace(0.0, 1.0, m)
    k = np.clip((t - start) / max(1e-6, 1.0 - start), 0.0, 1.0)
    speed = 1.0 + (end_ratio - 1.0) * k ** curve
    pos = np.cumsum(speed)
    keep = pos < n - 2
    pos = pos[keep]
    if len(pos) < 2:
        return x * 0.0
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    y = (x[i0] * (1 - fr) + x[i0 + 1] * fr) if x.ndim == 1 else \
        np.stack([x[c][i0] * (1 - fr) + x[c][i0 + 1] * fr for c in range(2)])
    if fade:
        m = y.shape[-1]
        tail = max(1, int(m * 0.25))
        env = np.ones(m)
        env[-tail:] = np.linspace(1.0, 0.0, tail) ** 1.5
        y = y * env
    return y


# -------------------------------------------------------------- rhythm & res

def gate(x, step_sec, pattern, smooth=0.008, floor=0.0):
    """Trance gate. pattern is a list of 0..1 values, one per step."""
    n = x.shape[-1]
    steps = int(np.ceil(n / (step_sec * SR)))
    vals = np.array([pattern[i % len(pattern)] for i in range(steps)], dtype=float)
    env = np.repeat(vals, int(round(step_sec * SR)))[:n]
    if len(env) < n:
        env = np.concatenate([env, np.full(n - len(env), env[-1] if len(env) else 1.0)])
    a = np.exp(-1.0 / max(1.0, smooth * SR))
    env = lfilter([1 - a], [1, -a], env)
    env = floor + (1 - floor) * env
    return x * env


def comb(x, freq, feedback=0.85, mix=1.0, damp=7000.0):
    """Tuned resonator - what turns a noise burst into struck metal."""
    n = x.shape[-1]
    d = max(2, int(round(SR / max(20.0, freq))))
    y = x.copy()
    for i in range(d, n, d):
        j = min(n, i + d)
        y[..., i:j] += feedback * biquad(y[..., i - d:i - d + (j - i)], "lp", damp, 0.7)
    return (1 - mix) * x + mix * y


def transient_shape(x, attack=1.0, sustain=1.0, fast=0.003, slow=0.055):
    """Separate hit from tail and re-weight them."""
    mono = np.max(np.abs(x), axis=0) if x.ndim > 1 else np.abs(x)
    af = np.exp(-1.0 / (fast * SR))
    as_ = np.exp(-1.0 / (slow * SR))
    ef = lfilter([1 - af], [1, -af], mono)
    es = lfilter([1 - as_], [1, -as_], mono)
    diff = ef - es
    g = 1.0 + attack * np.maximum(diff, 0) / (es + 1e-6) * 0.5 \
        - (1.0 - sustain) * np.maximum(-diff, 0) / (es + 1e-6) * 0.5
    g = np.clip(g, 0.2, 4.0)
    return x * g


def granular_stretch(x, factor=3.0, grain=0.085, overlap=4, jitter=0.25, seed=0):
    """Lengthen a sound without changing its pitch.

    Overlap-add of short windowed grains whose read position advances slower
    than the write position. Keeps the formants where they are, which is what
    lets a half-second vocal note become a sustained one and still sound like
    a person rather than a slowed-down tape."""
    n_in = x.shape[-1]
    g = max(64, n_samples(grain))
    hop_out = max(1, g // overlap)
    hop_in = hop_out / max(0.05, factor)
    n_out = int(n_in * factor) + g
    out = np.zeros(n_out)
    win = np.hanning(g)
    norm = np.zeros(n_out)
    r = np.random.default_rng(seed + 811)
    pos_in = 0.0
    for i in range(0, n_out - g, hop_out):
        j = int(pos_in + r.normal(0, jitter * hop_in))
        j = int(np.clip(j, 0, max(0, n_in - g - 1)))
        out[i:i + g] += x[j:j + g] * win
        norm[i:i + g] += win
        pos_in += hop_in
    return out / np.maximum(norm, 1e-6)


def vibrato(x, rate=5.0, depth_ms=2.2, onset=0.35):
    """Pitch wobble via a modulated delay - a player's vibrato, not an LFO."""
    n = x.shape[-1]
    t = np.arange(n) / SR
    ramp = np.clip((t / max(0.05, onset)), 0, 1) ** 1.5
    mod = (depth_ms / 1000.0) * SR * ramp * (0.5 + 0.5 * np.sin(TWO_PI * rate * t))
    idx = np.arange(n) - mod
    i0 = np.clip(np.floor(idx).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    fr = np.clip(idx - i0, 0, 1)
    return x[i0] * (1 - fr) + x[i1] * fr
