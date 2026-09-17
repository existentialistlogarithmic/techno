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

LINES = {
    "confession": ("Father... forgive me. For all my sins... "
                   "and for all I have seen.", 1.42),
    "forgive": ("Forgive me.", 1.55),
    "sins": ("For all my sins.", 1.5),
}


def ensure_model(d):
    os.makedirs(d, exist_ok=True)
    for ext in (".onnx", ".onnx.json"):
        p = os.path.join(d, VOICE + ext)
        if not os.path.exists(p):
            print("downloading", ext)
            urllib.request.urlretrieve(BASE + ext, p)
    return os.path.join(d, VOICE + ".onnx")


def main():
    model = ensure_model(os.environ.get("PIPER_DIR", ".piper"))
    os.makedirs("assets", exist_ok=True)
    for name, (text, scale) in LINES.items():
        out = f"assets/{name}.wav"
        subprocess.run([sys.executable, "-m", "piper", "--model", model,
                        "--output_file", out, "--length-scale", str(scale)],
                       input=text.encode(), check=True)
        print("wrote", out)


if __name__ == "__main__":
    main()
