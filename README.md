# concrete cathedral

A 4-minute hard techno track synthesised entirely in Python. No DAW, no samples,
no audio libraries beyond `numpy` and `scipy` — every kick, screech, hi-hat and
vocal moan is generated from oscillators, filters and noise.

```
pip install numpy scipy
python3 render.py concrete_cathedral.wav --mp3
```

Renders in about 100 seconds. `--mp3` additionally needs `pip install lameenc`.

## The track

150 BPM, F# minor, 4:09, 152 bars.

| bars | section | what happens |
|---|---|---|
| 0–15 | intro | room tone, 50 Hz hum, a voice screaming for help somewhere in the building, muffled kick behind a closed filter |
| 16–31 | build 1 | kick opens up bar by bar, acid line enters, percussion thickens |
| 32–39 | pre-drop | accelerating snare roll, noise + tone risers, everything cuts |
| 40–71 | **drop 1** | full kick, rumble, hoover riff, screeches, vocal stabs |
| 72–87 | breakdown | pads, moans, whispers, reverb wash, heartbeat kick returns |
| 88–103 | build 2 | climbs harder, second roll, hard cut |
| 104–135 | **drop 2** | harder kick, screech lead riff, faster acid |
| 136–151 | outro | strips back, filters down, last impact |

## How it is built

```
technogen/
  dsp.py           oscillators, envelopes, biquads, distortion, reverb, delay
  instruments.py   every voice: drums, hoover, screech, acid, vocals, FX
  mixer.py         timeline, buses, sidechain, balancing, master chain
  track.py         the arrangement
render.py          CLI
```

### Sound design notes

**Kick** — a sine whose frequency drops from ~380 Hz to 45 Hz in about 11 ms,
with two slower pitch tails underneath, a noise-and-blip transient, then
`tanh` saturation, foldback distortion and soft clipping. The distortion is
what makes it hard techno rather than house.

**Rumble** — the clean sub layer is fed through a 2-second reverb, lowpassed to
200 Hz and sidechained hard. That churning low end under the kick is the
genre's signature and it cannot be made with an EQ.

**Screech** — a detuned saw stack whose pitch is swept and wobbled, pushed
through a bandpass at Q≈17 that tracks the sweep, distorted, then bandpassed
again. The second filter pass after the distortion is what turns fizz into a
scream.

**Hoover** — nine detuned saws plus a PWM square an octave down, a resonant
ladder filter, then a six-stage swept allpass phaser.

**Acid** — a 303 model: saw through a resonant lowpass whose cutoff is driven
by the note envelope, with per-step accent and glide.

**Screams** — a scream is not just a loud vowel, and building it as one gives
you opera. It needs four things at once: pitch an octave above speech, *jitter*
(irregular pitch, from a lowpassed random walk), a subharmonic rattle where the
folds stop tracking cleanly, turbulent noise from the constriction, and real
nonlinearity from a vocal tract driven past its linear range.

The word is shaped as `h-eh-l-p`: an aspiration burst, the vowel, a lateral
with its own formant target, a silent lip closure, then the release burst.
At a scream's pitch the formants fall *between* harmonics, so the resonators
alone just ring on the fundamental — the 2–6 kHz bands have to be built from
the source with a shout formant near 3 kHz and a distorted high band. Each
scream is then filtered for distance before it hits the bus, because air and
walls eat the top end long before they eat the level.

**Voices** — formant synthesis. A glottal saw source runs through four parallel
resonators whose centre frequencies move between vowel targets, so a moan is a
pitch glide with an `oo → ah → oo` formant path and the chatter is random
vowel pairs with syllabic gating. Drenched in reverb and delay, it reads as a
sampled voice in a warehouse.

## Mixing

Fader gains are useless here: a saturated kick carries roughly twenty times the
RMS of a screech, so hand-set gains put everything either inaudible or on top
of each other. Instead `mixer.balance()` measures each bus's loudness *in the
band that bus actually occupies* — low buses against the kick's low band,
everything else against its mid band — and solves for the gain that hits a
target offset. Measurement only counts the loudest blocks, so a bus that plays
in eight bars out of thirty-two is not penalised for the silence.

`mixer.tilt_match()` then measures the summed mix in octave bands and applies a
bounded corrective EQ toward a target curve for a club master. Synthesised
drums vary too much from render to render for a fixed master EQ to fit.

`master()` finishes with glue compression, tape saturation, presence EQ, a
loudness target and a lookahead limiter. The loudness target matters: without
it the limiter manufactures level instead of catching peaks, and the kick
loses its transient.

## Checking the result

Four analysis tools, because you cannot mix what you cannot measure:

```
python3 analyze.py    concrete_cathedral.wav        # levels, spectrum, dynamics
python3 arc.py        concrete_cathedral.wav        # level and brightness per bar
python3 spectrum.py   concrete_cathedral.wav 104 136  # vs. reference master curve
python3 audibility.py buses 40 72                   # is each bus actually audible
```

`audibility.py` caught the screams being 21 dB above everything else at 50%
duty — not a distant cry, just continuous screaming. `arc.py` caught the worst bug in this track: the first version had the build
and the drop at exactly the same level, because the kick and sub were already
at full size before bar 40. The drop landed on nothing. The automation curves
in `track.py` hold the low end back through the builds so bar 40 arrives with a
real jump in weight.

Re-mixing without re-synthesising:

```
python3 render.py out.wav --cache-buses buses
python3 render.py out.wav --from-cache buses     # ~30s instead of ~100s
```
