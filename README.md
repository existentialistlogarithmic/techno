# concrete cathedral

A six and a half minute track that begins as a string lament and is taken
apart by an industrial hard techno rave. Synthesised entirely in Python. No DAW, no samples, no audio libraries beyond `numpy` and `scipy` —
every kick, screech, hi-hat, struck metal plate, breath and scream is generated
from oscillators, filters and noise. The only recorded-sounding thing in it is
the spoken line, and that is neural TTS rendered to `assets/` and then put
through a room.

```
pip install numpy scipy
python3 render.py concrete_cathedral.wav --mp3
```

Renders in about five minutes. `--mp3` needs `pip install lameenc`.

## The track

150 BPM, F# minor, 6:06, 224 bars.

| time | bars | section |
|---|---|---|
| 0:00 | 0–14 | lament — a violin crying, the confession, the room draining away |
| **0:24** | 15 | **BOOM. BOOM.** two hits into silence |
| 0:25 | 16–31 | the machine — full kick, metal, the lament coming apart over the top |
| 0:51 | 32–47 | groove |
| 1:16 | 48–63 | build 1 |
| 1:42 | 64–71 | tension |
| **1:55** | 72–103 | **drop 1** |
| 2:46 | 104–111 | interlude — tape brake, machine room |
| 2:59 | 112–127 | build 2 |
| **3:24** | 128–159 | **drop 2** |
| 4:16 | 160–175 | breakdown — the violin returns whole |
| 4:41 | 176–183 | build 3 |
| **4:54** | 184–215 | **drop 3** — the lament played by the machine |
| 5:45 | 216–223 | outro |

Every drop is built by `drop_core()`: one locked pattern that changes on the
eight and takes a fill on the bar before the change. Nothing happens once.
Drops differ by intensity — kick model, acid pattern, whether the screech lead
runs — not by content.

## How it is built

```
technogen/
  dsp.py           oscillators, envelopes, biquads, distortion, reverb,
                   delay, vocoder, tape effects, gating
  instruments.py   drums, hoover, screech, acid, pads, vocals, screams
  texture.py       industrial metal, breath, speech processing, transitions
  mixer.py         timeline, buses, sidechain, balancing, master chain
  track.py         the arrangement
render.py          CLI
make_voice.py      regenerates the spoken assets (only needed to change words)
```

### Sound design notes

**Kick** — a sine whose frequency drops from ~380 Hz to 45 Hz in about 11 ms,
with two slower pitch tails underneath, a noise-and-blip transient, then
oversampled saturation, foldback distortion and soft clipping.

**Rumble** — the clean sub layer fed through a 2-second reverb, lowpassed to
200 Hz and sidechained hard. That churning low end under the kick cannot be
made with an EQ.

**Screech** — a detuned saw stack whose pitch is swept and wobbled, pushed
through a bandpass at Q≈17 that tracks the sweep, distorted, then bandpassed
again. The second filter pass *after* the distortion is what turns fizz into a
scream.

**Struck metal** — deliberately inharmonic partial ratios, each partial with
its own decay so the highs die first, plus a noise transient filtered by the
same resonances. Harmonic ratios sound like a bell; inharmonic ones sound like
something industrial being hit.

**Making a violin cry** — a sad melody is not a crying one. Four things do the
work, and `bowed()` takes them as parameters: `porta` slides into each note on
an S-curve instead of arriving at it (a finger leaves slowly and lands fast);
`vib_growth` widens the vibrato as a note is held, measured going from ±17 to
±41 cents across one bow, which is the expressive range without tipping into
seasick; `swell` leans into the middle of the bow and falls away, giving 20 dB
of movement inside a single phrase; and `sob` puts a catch in the tone that
only arrives once the note has spoken. `strain` pushes very slightly sharp at
the peak, the way players do when they lean on a phrase. `rubato` in `phrase()`
keeps it off the grid. The melody itself is built on falling semitone pairs —
the sigh figure — and the second phrase climbs instead of falling, which is
what turns grief into panic.

**The hit** — `boom()` is a sub that falls from where it can be heard to where
it can only be felt, under driven noise and a struck plate. Two of them into a
bar of near-silence: the bar before measures −18.7 dB, the hit −7.6 dB. An
11 dB slam is the whole trick; it only works because the bar before it is empty.

**Violin** — a bowed string is close to a sawtooth, but a plain saw sounds
like a synth. What makes it read as a violin is everything around the saw: a
bank of body resonances (the air mode near 275 Hz, the main wood modes, and the
broad "bridge hill" at 2–3 kHz that gives the instrument its bite), vibrato
that arrives *after* the note starts as a player's does, friction noise loudest
while the bow is still grabbing the string, and a slow pitch drift because no
finger is ever still. `ensemble()` is not a chorus effect — each player gets
their own vibrato rate, tuning offset and entry time. `desecrate()` is what the
rave does to it.

**What is deliberately not here** — an earlier version had synthesised moans,
whispers, vowel-babble chatter and screams, all built on the same formant
engine. Formant synthesis without a real glottal model does not sound like a
person, it sounds like a theremin, and in a techno context it reads as a
cartoon ghost. Pitch-swept sirens, zaps and downlifters have the same problem.
All of it is gone. The only voice left is real recorded-style speech from
neural TTS, used three times. Everything else is drums, metal, noise and
strings.

The mix is also much drier than a pop mix on purpose. Hard techno lives close
to the speaker; long tails on every bus is what turns a busy arrangement to
mud. Only the rumble, the strings and the speech get a real room.

**Screams (removed, kept for reference)** — a scream is not a loud vowel. It needs pitch an octave above
speech, jitter from a lowpassed random walk, a subharmonic rattle where the
folds stop tracking, turbulence, and a tract driven past its linear range.
`scream_help()` shapes the word as `h-eh-l-p`: aspiration, vowel, lateral,
silent lip closure, release burst. At that pitch the formants fall between
harmonics, so the 2–6 kHz bands are built from the source rather than from the
resonators.

**The spoken line** — piper neural TTS, then split into four layers: a dark
close voice, a whispered double made by vocoding the speech onto noise, a
vocoded choir on a saw carrier for the breakdown callback, and a broken-radio
bed. The reverb uses a 75 ms predelay so the words stay in front of the room
instead of inside it.

**Breath and moans** — formant synthesis. A glottal saw through four parallel
resonators gliding between vowel targets. The breath bus is kept close and very
wide with almost no reverb, against the screams on the same formant engine
which are drowned in a 4.4-second hall — same synthesis, opposite treatment,
opposite effect.

## Clarity

Naive digital oscillators alias badly, and distortion folds that aliasing back
down into the audible range as grit that no EQ can remove. Both are fixed:

- PolyBLEP band-limited saw and square. Measured aliasing at 2.2 kHz drops
  from 5.77% of total energy to 0.12%.
- Every saturation stage runs 4× oversampled, so its harmonics land above
  Nyquist and get filtered out rather than folding back.

## Mixing

Fader gains are useless here: a saturated kick carries roughly twenty times the
RMS of a screech. `mixer.balance()` measures each bus's loudness *in the band
that bus occupies* — low buses against the kick's low band, everything else
against its mid band — and solves for the gain that hits a target offset,
counting only the blocks where that bus is actually playing.

`mixer.tilt_match()` then measures the summed mix in octave bands and applies a
bounded corrective EQ toward a target curve. With fifteen buses, no fixed
master EQ fits.

`master()` finishes with glue compression, tape saturation, presence EQ, a
loudness target and a lookahead limiter. The loudness target matters: without
it the limiter manufactures level instead of catching peaks and the kick loses
its transient.

## Checking the result

Four analysis tools, because you cannot mix what you cannot hear:

```
python3 analyze.py    concrete_cathedral.wav          # levels, spectrum, dynamics
python3 arc.py        concrete_cathedral.wav          # level and brightness per bar
python3 spectrum.py   concrete_cathedral.wav 128 160  # vs. a reference master curve
python3 audibility.py buses 16 32                     # is each bus actually audible
```

Every one of them caught something real:

- `arc.py` found the first version's build and drop sitting at exactly the same
  level — the drop landed on nothing. The automation curves in `track.py` hold
  the low end back through the builds so each drop arrives with a jump in
  weight.
- `audibility.py` found the screams at 21 dB over everything else with 50% duty
  (continuous screaming, not a distant cry), and later found the whole
  industrial layer 20 dB under in the section named after it.
- `tilt_match` was found overshooting badly: it computed per-band corrections
  independently but applied nine *overlapping* peaking filters, so large cuts
  compounded. It asked for −7 dB and delivered −14. It now measures the
  residual and re-corrects over five passes, landing within 0.7 dB.
- `spectrum.py` found that adding the industrial and intimate layers had tilted
  the mix 1.9 dB down across 200 Hz–2 kHz and 2.4 dB up above 8 kHz — thinner
  and fizzier. Comparing two renders band by band is far more useful than
  comparing one render against an absolute reference.

A note on `TILT_TARGET`: the numbers are not a guess. They are measured off a
master that was checked and signed off, and `tilt_match` runs at the *end* of
the chain, so the target describes the output rather than some intermediate
point. An earlier version ran the tilt mid-chain, which made it impossible to
reason about — every stage after it changed the balance again.

Re-mixing without re-synthesising:

```
python3 render.py out.wav --cache-buses buses
python3 render.py out.wav --from-cache buses     # ~60s instead of ~240s
```
