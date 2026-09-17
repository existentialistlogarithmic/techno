# concrete cathedral

A five and a half minute industrial hard techno track synthesised entirely in
Python. No DAW, no samples, no audio libraries beyond `numpy` and `scipy` —
every kick, screech, hi-hat, struck metal plate, breath and scream is generated
from oscillators, filters and noise. The only recorded-sounding thing in it is
the spoken line, and that is neural TTS rendered to `assets/` and then put
through a room.

```
pip install numpy scipy
python3 render.py concrete_cathedral.wav --mp3
```

Renders in about four minutes. `--mp3` needs `pip install lameenc`.

## The track

150 BPM, F# minor, 5:39, 208 bars.

| bars | section | what happens |
|---|---|---|
| 0–15 | intro / confession | *"Father… forgive me. For all my sins… and for all I have seen."* over a machine room, scrapes, steam and a voice screaming for help in the distance |
| 16–31 | industrial groove | struck metal and a conveyor loop carry the section; the kick is still small and dark |
| 32–47 | build 1 | kick opens bar by bar, acid enters |
| 48–55 | pre-drop | accelerating snare roll, risers, steam, cut |
| 56–87 | **drop 1** | kick, rumble, hoover riff, screeches, metal, vocal stabs |
| 88–95 | transition | the groove is braked like tape, a sub drop, then breath, sighs and a gated pad in the space it leaves |
| 96–111 | sultry mid-section | swung shakers, a slower acid line, pulsing pad, moans and breath close and wide |
| 112–127 | build 2 | industrial percussion returns, second roll |
| 128–159 | **drop 2** | harder kick, second hoover riff, screech lead, vocoded choir |
| 160–171 | breakdown | the confession returns as a vocoded choir over dark pads |
| 172–179 | build 3 | short and fast |
| 180–195 | **drop 3** | the hardest kick, everything at once |
| 196–207 | outro | strips back to the machine room and *"forgive me"* |

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

**Screams** — a scream is not a loud vowel. It needs pitch an octave above
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
- `spectrum.py` found that adding the industrial and intimate layers had tilted
  the mix 1.9 dB down across 200 Hz–2 kHz and 2.4 dB up above 8 kHz — thinner
  and fizzier. Comparing two renders band by band is far more useful than
  comparing one render against an absolute reference.

Re-mixing without re-synthesising:

```
python3 render.py out.wav --cache-buses buses
python3 render.py out.wav --from-cache buses     # ~60s instead of ~240s
```
