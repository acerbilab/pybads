"""Time the words of each line's clip, and map the film's motion onto another voice's words.

    python -u scripts/words.py V --design     # V: the voice the motion was designed on
    python -u scripts/words.py V              # V: the voice that film_timeline.js now holds

film.html's motion was designed on the takes of the draft voice, Kokoro's am_michael: a shot's moments fall on its
line's words as that voice said them. Another voice says them at other times, and the stretch of a shot over its
line's span keeps them only in proportion. This script transcribes each clip V/voice/<line id>.wav with
faster-whisper, with word timestamps, and matches Whisper's words to the line's words as the screen writes them.
The words are known and only their times are wanted, so the line is Whisper's prompt: without it, Whisper left out
the second sentence of line 3.3 on the design voice's take.

With --design, V is the voice the motion was designed on, and film_timeline.js its timeline. The script writes
design_words.json: for each line, its span on that timeline (from its start to the next line's start), its clip's
length, its caption's caption_at, and the time of each matched word, keyed by the word's character offset in the line.
It also writes film_warp.js as null, so that film.html plays the design timing.

Without it, V is another voice, and film_timeline.js its timeline (scripts/voice.py V comes first). The script pairs
the times of the words that both voices' clips matched and writes film_warp.js (window.BADS_WARP): for each line, its
design span, and the design time at every STEP seconds of the line's span on the new timeline, read off a monotone
smooth curve (SciPy's PCHIP) through the line's start, the paired words, the ends of the clips and the ends of the
spans. Pairs closer than MIN_GAP seconds are thinned, so that the motion's pace does not change at every word.
film.html plays each shot at the design time of the moment, and the score's events follow, since the page places them.
For a caption in parts, it prints where each part's start falls in the new take, the caption_at for narration.json.

Runs after scripts/voice.py V, and again after any new take. Needs faster-whisper, soundfile and SciPy;
faster-whisper fetches its model into HF_HOME. Adapted from the PyIBS film's words.py (acerbilab/pyibs, dev/film/).
"""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import PchipInterpolator
from scipy.signal import resample_poly

FILM = Path(__file__).resolve().parent.parent  # dev/film
WORD = re.compile(r"[A-Za-z0-9’']+")
STEP = 0.04  # seconds between the samples of film_warp.js: one frame at 25 fps
MIN_GAP = 0.4  # seconds: paired words closer than this to the last one kept are dropped


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def written(ln):
    """The line as the screen writes it, without its footnote's asterisk."""
    cap = ln.get("caption", ln["text"])
    return (" ".join(cap) if isinstance(cap, list) else cap).replace("*", "")


def line_words(line):
    """The line's words, split at hyphens, each with its character offset."""
    return [
        (m.start(), norm(m.group()))
        for m in WORD.finditer(line)
        if norm(m.group())
    ]


def heard_words(segments):
    """Whisper's words, split at hyphens and spaces, each with its start; a word split in n parts spreads them over
    its own span."""
    out = []
    for seg in segments:
        for w in seg.words or []:
            parts = [norm(p) for p in re.split(r"[\s\-–]+", w.word.strip())]
            parts = [p for p in parts if p]
            for j, p in enumerate(parts):
                out.append((p, w.start + (w.end - w.start) * j / len(parts)))
    return out


def timeline():
    """The voice and, for each line, its span and clip length on film_timeline.js."""
    js = (FILM / "film_timeline.js").read_text(encoding="utf-8")
    film = json.loads(js[js.index("{") : js.rindex("}") + 1])
    starts = [
        (ln["id"], sc["start"] + ln["start"], ln["seconds"])
        for sc in film["scenes"]
        for ln in sc["lines"]
    ]
    ends = [s for _, s, _ in starts[1:]] + [film["seconds"]]
    return film["voice"], {
        i: {"span": round(e - s, 3), "seconds": sec}
        for (i, s, sec), e in zip(starts, ends)
    }


def time_words(model, v, lines):
    """For each line, [character offset, seconds into the clip] of each word matched, in time order."""
    out = {}
    for lid, text in lines.items():
        clip = v / "voice" / f"{lid}.wav"
        if not clip.exists():
            raise SystemExit(f"no take of line {lid}: {clip}")
        audio, sr = sf.read(clip, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            word_timestamps=True,
            initial_prompt=text,
        )
        heard, mine = heard_words(list(segments)), line_words(text)
        sm = SequenceMatcher(
            a=[w for _, w in mine], b=[w for w, _ in heard], autojunk=False
        )
        anchors = sorted(
            [mine[a + j][0], round(heard[b + j][1], 3)]
            for a, b, n in sm.get_matching_blocks()
            for j in range(n)
        )
        kept = (
            []
        )  # in time order: a word matched out of order would run the motion backwards
        for c, s in anchors:
            if not kept or s > kept[-1][1]:
                kept.append([c, s])
        out[lid] = kept
        print(f"  {lid:4s} {len(kept):2d} of {len(mine):2d} words", flush=True)
    return out


def warp(design, now, words_now):
    """The design time at every STEP seconds of the line's span on the new timeline."""
    span, sec = now["span"], now["seconds"]
    d_span, d_sec = design["span"], design["seconds"]
    then = dict(map(tuple, design["words"]))
    pairs = sorted((u, then[c]) for c, u in words_now if c in then)
    pts = [(0.0, 0.0)]
    for u, tau in pairs:
        last_u, last_tau = pts[-1]
        if (
            u - last_u >= MIN_GAP
            and tau > last_tau + 0.05
            and u < sec - 0.2
            and tau < d_sec - 0.05
        ):
            pts.append((u, tau))
    pts += [(sec, d_sec), (span, d_span)]
    u, tau = np.array(pts).T
    curve = PchipInterpolator(u, tau, extrapolate=True)
    grid = np.arange(int(np.ceil(span / STEP)) + 1) * STEP
    return [round(float(x), 4) for x in np.clip(curve(grid), 0.0, d_span)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v", type=Path, help="the folder of a voice's media")
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()
    from faster_whisper import WhisperModel

    spec = json.loads((FILM / "narration.json").read_text(encoding="utf-8"))
    lns = [ln for sc in spec["scenes"] for ln in sc["lines"]]
    lines = {ln["id"]: written(ln) for ln in lns}
    voice, spans = timeline()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    found = time_words(model, args.v, lines)
    out_js = FILM / "film_warp.js"
    head = "// Written by scripts/words.py; do not edit by hand.\n"
    if args.design:
        design = {
            "voice": voice,
            "lines": {
                ln["id"]: {
                    **spans[ln["id"]],
                    **(
                        {"caption_at": ln["caption_at"]}
                        if "caption_at" in ln
                        else {}
                    ),
                    "words": found[ln["id"]],
                }
                for ln in lns
            },
        }
        (FILM / "design_words.json").write_text(
            json.dumps(design, indent=1) + "\n", encoding="utf-8"
        )
        out_js.write_text(
            head + "window.BADS_WARP = null;\n", encoding="utf-8"
        )
        print(f"wrote design_words.json for {voice}, and a null film_warp.js")
        return
    design = json.loads(
        (FILM / "design_words.json").read_text(encoding="utf-8")
    )
    if voice == design["voice"]:
        raise SystemExit(
            f"film_timeline.js holds the design voice, {voice}: run scripts/voice.py on the new voice first"
        )
    table = {
        i: {
            "span": design["lines"][i]["span"],
            "t": warp(design["lines"][i], spans[i], found[i]),
        }
        for i in lines
    }
    data = {
        "voice": voice,
        "design": design["voice"],
        "step": STEP,
        "lines": table,
    }
    out_js.write_text(
        head
        + "window.BADS_WARP = "
        + json.dumps(data, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote film_warp.js: {voice} on {design['voice']}'s motion")
    for i, d in design["lines"].items():
        if "caption_at" in d:
            t = np.array(table[i]["t"])
            at = [
                round(float(np.interp(a, t, np.arange(t.size) * STEP)), 2)
                for a in d["caption_at"]
            ]
            print(
                f"  {i}: caption_at {at} (the design voice's {d['caption_at']})"
            )


if __name__ == "__main__":
    main()
