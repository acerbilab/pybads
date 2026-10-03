"""Voice the narration with Kokoro, and time the film from it.

    python -u scripts/voice.py V [--only ID,...] [--all]

Reads narration.json. Writes film_timeline.js, the timeline that film.html plays (window.BADS_FILM: each scene's start
and length, and each line's start, length and caption, in seconds), and into V, the folder of this voice's media:

    voice/<line id>.wav   one 48 kHz mono clip per line
    narration.wav         the clips placed on the timeline
    narration.srt         the captions

A scene lasts lead + its lines and the gaps before them + tail. A line's "text" is what the voice says; its "caption",
when given, is what the screen shows (the voice reads "Pie-Bads" for PyBADS, and "search" and "poll" in lower case,
so as not to spell them). The narration's "speed" sets the voice's pace, and a scene's "speed" overrides it. Kokoro voices each line on its own and does not give the same take twice, so voicing a line
again moves the timeline; a line is voiced when V has no clip of it, when --only names it or its scene, or with
--all. Needs kokoro (0.9.4 or later), soundfile and SciPy; Kokoro fetches its weights from the Hugging Face hub into
HF_HOME.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FILM = Path(__file__).resolve().parent.parent  # dev/film

SR_TTS, SR_OUT = 24000, 48000
TARGET_RMS_DB, PEAK_DB = -20.0, -1.0
PRE_ROLL, POST_ROLL = 0.05, 0.12


def db(x):
    return 10.0 ** (x / 20.0)


def trim_and_level(audio, sr):
    """Trim the silence at both ends and level the voiced part to TARGET_RMS_DB."""
    win = int(0.02 * sr)
    env = np.sqrt(np.convolve(audio**2, np.ones(win) / win, mode="same"))
    thresh = max(env.max(), 1e-9) * db(-40.0)
    voiced = np.nonzero(env > thresh)[0]
    if voiced.size == 0:
        return audio.astype(np.float32)
    a, b = max(0, voiced[0] - int(PRE_ROLL * sr)), min(
        audio.size, voiced[-1] + int(POST_ROLL * sr)
    )
    clip = audio[a:b].astype(np.float64)
    rms = np.sqrt(np.mean(clip[env[a:b] > thresh] ** 2))
    clip *= db(TARGET_RMS_DB) / max(rms, 1e-9)
    clip *= min(1.0, db(PEAK_DB) / np.abs(clip).max())
    fade = int(0.005 * sr)
    clip[:fade] *= np.linspace(0, 1, fade)
    clip[-fade:] *= np.linspace(1, 0, fade)
    return clip.astype(np.float32)


def synthesize(pipeline, text, voice, speed):
    chunks = [
        np.asarray(
            r.audio.detach().cpu() if hasattr(r.audio, "detach") else r.audio
        )
        for r in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        if r.audio is not None
    ]
    if not chunks:
        raise RuntimeError(f"Kokoro returned no audio for {text!r}")
    return np.concatenate(chunks)


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v", type=Path, help="the folder of this voice's media")
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    spec = json.loads((FILM / "narration.json").read_text(encoding="utf-8"))
    dflt, only = spec["defaults"], {s for s in args.only.split(",") if s}
    clips = args.v / "voice"
    clips.mkdir(parents=True, exist_ok=True)
    todo = [
        (ln, clips / f"{ln['id']}.wav", sc.get("speed", spec["speed"]))
        for sc in spec["scenes"]
        for ln in sc["lines"]
        if args.all
        or ln["id"] in only
        or sc["id"] in only
        or not (clips / f"{ln['id']}.wav").exists()
    ]
    if todo:
        from kokoro import KPipeline

        pipeline = KPipeline(
            lang_code=spec["voice"][0], repo_id="hexgrad/Kokoro-82M"
        )
        for ln, path, speed in todo:
            clip = trim_and_level(
                synthesize(pipeline, ln["text"], spec["voice"], speed), SR_TTS
            )
            sf.write(
                path,
                resample_poly(clip, SR_OUT // SR_TTS, 1).astype(np.float32),
                SR_OUT,
                subtype="PCM_16",
            )
            print(
                f"  {ln['id']:4s} {sf.info(path).duration:5.2f}s  {ln['text']}",
                flush=True,
            )
    timeline = {
        "voice": f"kokoro:{spec['voice']}",
        "speed": spec["speed"],
        "scenes": [],
    }
    t0, placed, captions = 0.0, [], []
    for sc in spec["scenes"]:
        t, lines = sc.get("lead", dflt["lead"]), []
        for i, ln in enumerate(sc["lines"]):
            if i:
                t += ln.get("gap", dflt["gap"])
            path = clips / f"{ln['id']}.wav"
            dur = sf.info(path).duration
            cap = ln.get("caption", ln["text"])
            lines.append(
                {
                    "id": ln["id"],
                    "caption": cap,
                    "start": round(t, 3),
                    "seconds": round(dur, 3),
                }
            )
            placed.append((t0 + t, path))
            captions.append((t0 + t, t0 + t + dur, cap))
            t += dur
        t += sc.get("tail", dflt["tail"])
        length = max(t, sc.get("min", 0.0))
        timeline["scenes"].append(
            {
                "id": sc["id"],
                "start": round(t0, 3),
                "seconds": round(length, 3),
                "lines": lines,
            }
        )
        t0 += length
    timeline["seconds"] = round(t0, 3)
    (FILM / "film_timeline.js").write_text(
        "window.BADS_FILM="
        + json.dumps(timeline, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    mix = np.zeros(int(np.ceil(t0 * SR_OUT)) + SR_OUT, np.float32)
    for start, path in placed:
        clip = sf.read(path, dtype="float32")[0]
        a = int(round(start * SR_OUT))
        mix[a : a + clip.size] += clip
    sf.write(args.v / "narration.wav", mix, SR_OUT, subtype="PCM_16")
    with open(args.v / "narration.srt", "w", encoding="utf-8") as f:
        for k, (a, b, text) in enumerate(captions, 1):
            f.write(f"{k}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")
    print(f"\n{timeline['voice']} at {spec['speed']}: {t0:.1f} s", flush=True)
    for s in timeline["scenes"]:
        print(
            f"  {s['id']:10s} {s['start']:6.1f}  +{s['seconds']:5.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
