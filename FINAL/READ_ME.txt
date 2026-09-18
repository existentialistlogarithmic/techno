CONCRETE CATHEDRAL - final
==========================

concrete_cathedral.mp3        the finished track, 6:06, 150 BPM, F# minor
CONCRETE_CATHEDRAL.als        Ableton project, 11 MIDI tracks laid out
CONCRETE_CATHEDRAL_arrangement.mid   same arrangement as plain MIDI
..._arrangement_map.txt       which MIDI note loads which sample
PLACEMENT_GUIDE.txt           every sound, by bar and step
sounds_mp3/                   all 137 isolated sounds
midi_patterns/                14 single patterns to build up from

WHAT CHANGED IN THIS PASS
-------------------------
- Moans and breaths are back, but built differently. The old ones were
  formant synthesis, which is why they sounded like a cartoon ghost.
  These start as sustained vowels sung by a neural TTS voice, granular
  stretched so the formants stay put. Measured 56-80% of their energy in
  the voiced band, against 26% for the old ones.
- A breath now lifts into every drop.
- The interlude and the breakdown were thinned out: fewer elements, so
  the quiet parts are actually quiet. Their crest factor is 9.2-9.6 dB
  against 7.1 in the drops.
- Tonal balance is within 0.45 dB of target across the whole spectrum.

THE ARC (rms, dBFS)
-------------------
  0:00 lament     -13.8      2:46 interlude  -10.0
  0:24 BOOM        -7.6      2:59 build 2    -10.7
  0:25 machine    -10.2      3:24 DROP 2      -7.9
  0:51 groove     -10.8      4:16 breakdown  -10.5
  1:16 build 1    -10.3      4:41 build 3    -10.5
  1:42 tension    -12.1      4:54 DROP 3      -7.9
  1:55 DROP 1      -8.6      5:45 outro      -11.7

The WAV sample pack (no mp3 encoder delay - use it for drums) is in the
two zips sent earlier, regenerate with: python3 export_sounds.py
