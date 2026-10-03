"""Transcribe the voiced lines back and list those whose words differ from the script.

    python -u scripts/verify_voice.py V [--model small.en]

Reads narration.json and the clips V/voice/<line id>.wav that voice.py wrote into V, a voice's folder, transcribes
each clip with Whisper, and compares its words with the line's text, ignoring case and punctuation. The spoken forms in
the script ("Pie-Bads", "Bads") and Whisper's ways of writing what it hears ("PyBADS", "pie bads", "BADS", "bads")
count as the same. Writes every transcript to V/whisper.txt, prints the lines that differ, and exits with status 1
when any does. A line can differ because Whisper mishears a short line out of its context; listen to it before voicing
it again. Needs faster-whisper, which fetches its model from the Hugging Face hub into HF_HOME on first use.
"""
import argparse
import json
import re
import sys
from pathlib import Path

FILM = Path(__file__).resolve().parent.parent  # dev/film
NAME = re.compile(r"\b(?:pie|pi|py)\s*bads\b")


def words(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " "))
    return NAME.sub("pybads", " ".join(text.split())).split()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v", type=Path, help="the folder of a voice's media")
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    spec = json.loads((FILM / "narration.json").read_text(encoding="utf-8"))
    lines = [ln for sc in spec["scenes"] for ln in sc["lines"]]
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    heard, differ = {}, []
    for ln in lines:
        audio, sr = sf.read(
            args.v / "voice" / f"{ln['id']}.wav", dtype="float32"
        )  # decoded here: faster-whisper's own
        audio = resample_poly(audio, 16000, sr).astype(
            np.float32
        )  # decoder needs a matching PyAV
        segments, _ = model.transcribe(audio, beam_size=5, language="en")
        heard[ln["id"]] = " ".join(seg.text.strip() for seg in segments)
        if words(heard[ln["id"]]) != words(ln["text"]):
            differ.append(ln)
    with open(args.v / "whisper.txt", "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(f"{ln['id']:4s} {heard[ln['id']]}\n")
    for ln in differ:
        print(f"{ln['id']:4s} script: {ln['text']}", flush=True)
        print(f"     heard : {heard[ln['id']]}", flush=True)
    print(
        f"{len(lines) - len(differ)} of {len(lines)} lines as written; transcripts in {args.v / 'whisper.txt'}",
        flush=True,
    )
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
