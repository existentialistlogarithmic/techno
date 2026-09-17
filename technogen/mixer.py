"""Timeline, buses, sidechain and the master chain."""

import numpy as np
from .dsp import (SR, n_samples, db, fit, biquad, sweep, reverb, delay, widen,
                  tanh_drive, soft_clip, compress, limiter, stereoize, normalize,
                  peaking, low_shelf)


class Session:
    """A simple multibus timeline. Positions are given in bars + 16th steps."""

    def __init__(self, bpm=150.0, bars=152, tail=6.0):
        self.bpm = bpm
        self.bars = bars
        self.step = 60.0 / bpm / 4.0          # one 16th note
        self.bar = self.step * 16
        self.dur = bars * self.bar + tail
        self.n = n_samples(self.dur)
        self.buses = {}
        self.kick_times = []

    # -- time -------------------------------------------------------------
    def t(self, bar, step=0.0):
        return bar * self.bar + step * self.step

    def i(self, bar, step=0.0):
        return n_samples(self.t(bar, step))

    def ramp(self, points, default=None):
        """Per-sample automation curve from [(bar, value), ...]."""
        xs = [self.i(b) for b, _ in points]
        ys = [float(v) for _, v in points]
        return np.interp(np.arange(self.n), xs, ys)

    # -- routing ----------------------------------------------------------
    def bus(self, name):
        if name not in self.buses:
            self.buses[name] = np.zeros((2, self.n))
        return self.buses[name]

    def place(self, name, x, bar, step=0.0, gain=1.0, pan_=0.0, reverse=False):
        if gain == 0.0:
            return
        x = np.asarray(x)
        if reverse:
            x = x[..., ::-1]
        if x.ndim == 1:
            a = (pan_ + 1) * np.pi / 4
            x = np.stack([x * np.cos(a), x * np.sin(a)]) * 1.414
        b = self.bus(name)
        i0 = self.i(bar, step)
        if i0 >= self.n:
            return
        ln = min(x.shape[1], self.n - i0)
        b[:, i0:i0 + ln] += x[:, :ln] * gain

    def mark_kick(self, bar, step=0.0):
        self.kick_times.append(self.t(bar, step))

    # -- sidechain --------------------------------------------------------
    def duck_envelope(self, depth=0.75, attack=0.004, hold=0.02, release=0.14):
        env = np.ones(self.n)
        a, h = n_samples(attack), n_samples(hold)
        r = n_samples(release)
        shape = np.concatenate([
            np.linspace(1.0, 1.0 - depth, max(1, a)),
            np.full(max(1, h), 1.0 - depth),
            1.0 - depth * np.exp(-np.linspace(0, 4.0, max(1, r))),
        ])
        for t in self.kick_times:
            i0 = n_samples(t)
            ln = min(len(shape), self.n - i0)
            if ln > 0:
                env[i0:i0 + ln] = np.minimum(env[i0:i0 + ln], shape[:ln])
        return env

    def apply_duck(self, names, env):
        for nm in names:
            if nm in self.buses:
                self.buses[nm] *= env


def send_reverb(x, amount, rt60=2.6, damp=0.55, hp=160.0, predelay=0.02, seed=5, width=1.0):
    if amount <= 0:
        return x
    return x + amount * reverb(x, rt60=rt60, predelay=predelay, damp=damp, hp=hp,
                               seed=seed, width=width)


def send_delay(x, amount, time, feedback=0.42, damp=5200.0, pingpong=True):
    if amount <= 0:
        return x
    wet = delay(x, time, feedback=feedback, mix=1.0, damp_hz=damp, pingpong=pingpong) - x
    return x + amount * wet


def tape(x, drive=1.6, wow=0.0008, hiss=0.0006, seed=23):
    n = x.shape[-1]
    y = tanh_drive(x, drive, comp=True)
    y = biquad(y, "lp", 17500, 0.7)
    if hiss:
        r = np.random.default_rng(seed)
        y = y + hiss * biquad(r.standard_normal(x.shape), "hp", 2000, 0.7)
    return y


BANDS = {
    "low": lambda x: biquad(biquad(x, "lp", 150.0, 0.7), "lp", 150.0, 0.7),
    "mid": lambda x: biquad(biquad(x, "hp", 250.0, 0.7), "hp", 250.0, 0.7),
    "full": lambda x: x,
}


def active_loudness(x, band="full", sr=SR, win=0.35, top_frac=0.25):
    """RMS of the loudest blocks in a band, so silent bars don't skew a bus."""
    y = BANDS[band](x)
    mono = y.mean(axis=0) if y.ndim > 1 else y
    w = max(1, int(win * sr))
    nb = len(mono) // w
    if nb < 2:
        return float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
    blocks = mono[:nb * w].reshape(nb, w)
    e = np.sqrt(np.mean(blocks ** 2, axis=1))
    k = max(1, int(nb * top_frac))
    return float(np.mean(np.sort(e)[-k:]) + 1e-12)


def balance(buses, targets, reference="kick", verbose=True):
    """Scale each bus to sit `target` dB under the reference *in its own band*.

    targets maps bus -> (target_db, band). A rumble is meaningless above
    250 Hz and a hi-hat is meaningless below it, so one broadband number
    cannot balance both against the same kick.
    """
    refs = {b: active_loudness(buses[reference], b) for b in ("low", "mid", "full")}
    gains, report = {}, []
    for nm, (target_db, band) in targets.items():
        if nm not in buses:
            continue
        m = active_loudness(buses[nm], band)
        g = db(target_db) * refs[band] / m
        gains[nm] = g
        report.append((nm, band, target_db, 20 * np.log10(m / refs[band]), 20 * np.log10(g)))
    if verbose:
        print("  bus         band   target  measured     gain")
        for nm, band, t, m, g in report:
            print(f"    {nm:<9s} {band:<5s} {t:+6.1f}  {m:+8.1f}  {g:+7.1f} dB")
    return gains


# Octave-band target curve for a hard techno master, dB relative to 40-80 Hz.
TILT_TARGET = [
    (20, 40, -15.0), (40, 80, 0.0), (80, 160, -4.5), (160, 320, -9.0),
    (320, 640, -10.5), (640, 1280, -12.5), (1280, 2560, -15.0),
    (2560, 5120, -17.0), (5120, 10240, -21.0), (10240, 20000, -26.0),
]


def loud_sections(x, frac=0.4, win=0.5, sr=SR):
    """The loudest blocks, stitched together.

    Tonal balance has to be judged where the track is loud. Averaged over a
    whole arrangement, the quiet sections - which are all midrange and no
    bass - drag the measurement up and the correction then scoops the
    midrange out of the drops."""
    mono = x.mean(axis=0) if x.ndim > 1 else x
    w = max(1, int(win * sr))
    nb = len(mono) // w
    if nb < 4:
        return mono
    blocks = mono[:nb * w].reshape(nb, w)
    e = np.sqrt(np.mean(blocks ** 2, axis=1))
    keep = np.argsort(e)[-max(1, int(nb * frac)):]
    return blocks[np.sort(keep)].reshape(-1)


def measure_bands(x, sr=SR, loud_only=True):
    from scipy.signal import welch
    mono = loud_sections(x) if loud_only else (x.mean(axis=0) if x.ndim > 1 else x)
    f, P = welch(mono, sr, nperseg=16384)
    return [P[(f >= a) & (f < b)].sum() + 1e-18 for a, b, _ in TILT_TARGET]


def tilt_match(x, max_db=6.0, verbose=True):
    """Bounded corrective EQ that pulls the mix toward TILT_TARGET.

    Synthesised drums are far more consistent than sampled ones, so a fixed
    EQ curve never fits; measuring and correcting does.
    """
    e = measure_bands(x)
    ref_idx = 1                                   # the 40-80 Hz band
    have = [10 * np.log10(v / e[ref_idx]) for v in e]
    y = x
    if verbose:
        print("  band          have  target   corr")
    for i, (a, b, target) in enumerate(TILT_TARGET):
        corr = float(np.clip(target - have[i], -max_db, max_db))
        if i == ref_idx:
            corr = 0.0
        fc = np.sqrt(a * b)
        y = peaking(y, fc, 0.9, corr)
        if verbose:
            print(f"    {a:5d}-{b:<6d} {have[i]:+6.1f}  {target:+6.1f}  {corr:+6.1f}")
    return y


def set_loudness(x, target_rms_db=-8.5, win=0.4, top_frac=0.3):
    """Gain so the busy parts land on a target RMS, leaving the limiter to
    catch peaks rather than to manufacture loudness."""
    mono = x.mean(axis=0) if x.ndim > 1 else x
    w = max(1, int(win * SR))
    nb = len(mono) // w
    e = np.sqrt(np.mean(mono[:nb * w].reshape(nb, w) ** 2, axis=1))
    k = max(1, int(nb * top_frac))
    loud = float(np.mean(np.sort(e)[-k:]) + 1e-12)
    return x * (db(target_rms_db) / loud)


def master(mix, headroom_db=-1.0, target_rms_db=-7.8, glue=True, verbose=True):
    x = mix
    x = biquad(x, "hp", 27, 0.7)
    x = tilt_match(x, max_db=9.0, verbose=verbose)
    if glue:
        x = compress(x, thresh_db=-18.0, ratio=2.0, attack=0.015, release=0.18, makeup_db=1.5)
    x = tape(x, drive=1.25)
    x = x + 0.20 * biquad(x, "bp", 2600, 0.5)        # presence: leads bite through
    x = peaking(x, 5800, 0.9, 2.5)                   # screech bite
    x = peaking(x, 95, 0.8, 1.5)                     # fill the 63-125 scoop
    x = x + 0.05 * biquad(x, "hp", 8500, 0.7)        # air
    x = set_loudness(x, target_rms_db)
    x = soft_clip(x * 1.02, 0.90)
    x = limiter(x, ceiling=db(headroom_db))
    if verbose:
        red = 20 * np.log10(np.max(np.abs(x)) / max(1e-9, np.max(np.abs(mix))))
        print(f"  master: peak {20 * np.log10(np.max(np.abs(x))):+.2f} dBFS")
    return x


def to_mp3(path, x, sr=SR, bitrate=256):
    """Optional convenience export. Needs `pip install lameenc`."""
    try:
        import lameenc
    except ImportError:
        return False
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(2)
    enc.set_quality(2)
    pcm = (np.clip(x, -1.0, 1.0).T * 32767.0).astype(np.int16)
    data = enc.encode(pcm.tobytes()) + enc.flush()
    with open(path, "wb") as f:
        f.write(bytes(data))
    return True


def to_wav(path, x, sr=SR, bits=16):
    x = np.clip(x, -1.0, 1.0)
    if bits == 16:
        data = (x.T * 32767.0).astype(np.int16)
    else:
        data = (x.T * 2147483647.0).astype(np.int32)
    from scipy.io import wavfile
    wavfile.write(path, sr, data)
