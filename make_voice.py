#!/usr/bin/env python3
"""Regenerate the spoken assets with piper (neural TTS).

The rendered results are committed under assets/, so rendering the track
needs no TTS engine. Only run this to change the words.

    pip install piper-tts
    python3 make_voice.py
"""
import os
import subprocess
import sys
import urllib.request

VOICE = "en_GB-alan-medium"
BASE = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "en/en_GB/alan/medium/" + VOICE)

# A second, female voice, used only for sustained vowel tones. These become
# the breath and moan layer: neural TTS has real vocal timbre, so stretching
# a held vowel gives a voice. Synthesising the same thing from formants gives
# a theremin.
VOICE_F = "en_GB-jenny_dioco-medium"
BASE_F = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
          "en/en_GB/jenny_dioco/medium/" + VOICE_F)

LINES = {
    "confession": ("Father... forgive me. For all my sins... "
                   "and for all I have seen.", 1.42),
    "forgive": ("Forgive me.", 1.55),
    "sins": ("For all my sins.", 1.5),
}

TONES = {
    "vox_ah": ("aaaaaah", 2.6),
    "vox_oh": ("oooooh", 2.6),
    "vox_mm": ("mmmmmmm", 2.6),
    "vox_ha": ("haaaaa aaah", 2.6),
    "vox_uh": ("uuuuuh", 2.6),
}


def ensure_model(d, name=VOICE, base=BASE):
    os.makedirs(d, exist_ok=True)
    for ext in (".onnx", ".onnx.json"):
        p = os.path.join(d, name + ext)
        if not os.path.exists(p):
            print("downloading", name + ext)
            urllib.request.urlretrieve(base + ext, p)
    return os.path.join(d, name + ".onnx")


def main():
    model = ensure_model(os.environ.get("PIPER_DIR", ".piper"))
    os.makedirs("assets", exist_ok=True)
    for name, (text, scale) in LINES.items():
        out = f"assets/{name}.wav"
        subprocess.run([sys.executable, "-m", "piper", "--model", model,
                        "--output_file", out, "--length-scale", str(scale)],
                       input=text.encode(), check=True)
        print("wrote", out)

    model_f = ensure_model(os.environ.get("PIPER_DIR", ".piper"), VOICE_F, BASE_F)
    for name, (text, scale) in TONES.items():
        out = f"assets/{name}.wav"
        subprocess.run([sys.executable, "-m", "piper", "--model", model_f,
                        "--output_file", out, "--length-scale", str(scale)],
                       input=text.encode(), check=True)
        print("wrote", out)


if __name__ == "__main__":
    main()
