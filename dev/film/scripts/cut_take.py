"""Cut a narrator's read-through into one clip per line, for voice.py's recorded engine.

The narrator reads the whole script into one file, with a pause between lines and a line read again whenever a
reading goes wrong. Whisper transcribes the take with word timings, and each line's clip is its last complete reading,
wherever it falls: a reader goes back over one line or over a pair of lines. A pause of more than RESTART seconds
inside a reading is a false start when what follows the pause, transcribed on its own, is the whole line; the clip
then begins after the pause. The cuts follow the voiced stretches of the take, and where a reading runs on into the
next words without a pause, the cut falls at the quietest 10 ms between the two words.

The take is high-passed at 75 Hz (room rumble) and every clip gets one gain, which brings the median line to
TARGET_RMS_DB unless that would take a peak above PEAK_DB, so that the lines keep the take's relative levels.

The lines are matched as the screen writes them. NAMES maps each name, as the script writes it and as Whisper writes
it, to one word, and DIGITS reads the numbers that Whisper writes in digits as words. A form of a name that NAMES
misses costs the matcher a word; the forms come from a test reading. RESTART may need tuning for a reader who pauses
longer inside lines.

Whisper's words are cached in V/take_words.json, with the take's name and size; a new take is transcribed again.
Needs numpy, scipy, soundfile and faster-whisper, which fetches its model into HF_HOME on first use. voice.py imports
it from the same folder: cut(take, lines, voice_dir, out).

A false start is tested at every long pause of a reading, from the last, so that a line read once, paused over and
read again whole (line 7.1) keeps its second reading alone; what follows the pause counts as the whole line when
Whisper's words for it come within COMPLETE of the line, or within REREAD when they open with the line's first two
words, since Whisper mishears a few words of a whole reading ("Aput 20" for "up to twenty"). A reading's first or last
words that Whisper mishears ("low" for "slow", "BADZ pulls" for "BADS polls") fall outside the match, so as many words
as the line misses at that end, up to EDGE, that run on from it with a pause under RUN_ON and belong to no other
line's reading, are kept with it. A reading shorter than MIN_WORD seconds a word is dropped: Whisper wrote "Join
them." on the click that ended a take.

From the PyVBMC film's cut_take.py (acerbilab/pyvbmc, branch feat-3d-animation,
docsrc/source/_static/vbmc3d/scripts/), which had one name, no digits, a false-start test at a line's last long pause
alone, and none of the rest of the paragraph above, which this film's readings needed.
"""

import json
import re
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

SR_OUT, SR_WHISPER = 48000, 16000
TARGET_RMS_DB, PEAK_DB = -20.0, -1.0
PRE, POST = 0.06, 0.15  # seconds kept before and after the voiced stretch
VOICED_DB = (
    18.0  # a 10 ms frame is voiced this far above the take's noise floor
)
BRIDGE = 0.25  # voiced stretches closer than this are one
RESTART = 0.8  # a longer pause inside a reading may follow a false start
COMPLETE = 0.9  # a reading this close to the line's words is complete
REREAD = 0.75  # or this close, when it opens with the line's first two words
EDGE, RUN_ON = 2, 0.5  # unmatched words at a reading's ends, and their pause
MIN_WORD = 0.08  # seconds per word below which a reading was never spoken
MODEL = "small.en"

# Each name, as the script writes it and as Whisper writes it, in normalized words ("PyBADS", "pie bads", "Pibads",
# "PyBAD's", and from the narrator's readings "Piebuds", "pie bus", "Buzz" and "BADZ"), and the one word it becomes.
# The longer name comes first, so that "bads" in "pie bads" stays in it.
NAMES = [
    (
        "pybads",
        re.compile(
            r"\b(?:pie|pi|py|by|bi)\s?(?:ba[dt]{1,2}\s?s|bu(?:ds|zz|s))\b"
        ),
    ),
    ("bads", re.compile(r"\bba[dt]{1,2}\s?s\b|\bbadz\b|\bbu(?:ds|zz)\b")),
]
DIGITS = dict(
    enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty".split()
    )
)
DIGITS = {str(k): w for k, w in DIGITS.items()} | {"100": "hundred"}


def canon(text):
    """Each name in normalized words made one word, and digits made words."""
    for word, pattern in NAMES:
        text = pattern.sub(word, text)
    return " ".join(DIGITS.get(w, w) for w in text.split())


def words(text):
    """Lower-case words without punctuation, each name one word, the digits words."""
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " "))
    return canon(" ".join(text.split())).split()


def transcribe(model, take, cache):
    stamp = {"take": take.name, "bytes": take.stat().st_size}
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if {k: cached.get(k) for k in stamp} == stamp:
            return cached["words"]
    segments, _ = model.transcribe(
        str(take), beam_size=5, language="en", word_timestamps=True
    )
    found = [[w.word, w.start, w.end] for s in segments for w in s.words]
    cache.write_text(json.dumps({**stamp, "words": found}), encoding="utf-8")
    return found


def stream_of(found):
    """The take's words, normalized, each with the time span it came from."""
    toks = [[t, a, b] for w, a, b in found for t in words(w)]
    out, i = [], 0
    while i < len(toks):
        for k in (5, 4, 3, 2, 1):  # a name split across Whisper's words
            seg = " ".join(t[0] for t in toks[i : i + k])
            hit = i + k <= len(toks) and next(
                (w for w, pattern in NAMES if pattern.fullmatch(seg)), None
            )
            if hit:
                out.append([hit, toks[i][1], toks[i + k - 1][2]])
                i += k
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def readings(line, stream):
    """(first word, end, ratio) of each reading of a line, overlaps merged.

    A window is scored by how many of the line's words it matches, then by
    how close it is to the line; it starts on one of the line's first words.
    """
    n, found = len(line), []
    for s in range(len(stream)):
        if stream[s][0] not in line[:3]:
            continue
        best = (0, 0.0, None)
        for m in range(max(1, n - 3), n + 3):
            seg = [t[0] for t in stream[s : s + m]]
            sm = SequenceMatcher(None, line, seg)
            key = (sum(b.size for b in sm.get_matching_blocks()), sm.ratio())
            if key > best[:2]:
                best = (*key, s + len(seg))
        e = best[2]
        # Whisper can write words on a click (a "Join them." of 0.02 s at
        # the end of a take); a reading that short was never spoken.
        if best[1] >= 0.6 and stream[e - 1][2] - stream[s][1] >= MIN_WORD * (
            e - s
        ):
            found.append((s, e, best[1]))
    merged = []
    for c in found:
        if merged and c[0] < merged[-1][1]:
            if c[2] > merged[-1][2]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def pick(cands):
    complete = [c for c in cands if c[2] >= COMPLETE]
    if complete:
        return complete[-1]
    return max(cands, key=lambda c: (round(c[2], 2), c[0]))


def choose(lines, stream):
    """The reading of each line, as (first word, end, ratio)."""
    cands = [readings(words(l["text"]), stream) for l in lines]
    for l, c in zip(lines, cands):
        if not c:
            raise SystemExit(
                f"no reading of {l['id']} in the take: {l['text']}"
            )
    chosen = [pick(c) for c in cands]
    # A match inside another line's reading ("evaluate" in "where to evaluate
    # next") is not a reading of its own.
    for i in range(len(lines)):
        for j in range(len(lines)):
            ci, cj = chosen[i], chosen[j]
            if (
                i != j
                and cj[0] <= ci[0]
                and ci[1] <= cj[1]
                and ci[1] - ci[0] < cj[1] - cj[0]
            ):
                chosen[i] = pick(
                    [
                        c
                        for c in cands[i]
                        if not (cj[0] <= c[0] and c[1] <= cj[1])
                    ]
                )
    return chosen, cands


def voiced_runs(edb, thr):
    runs, a0 = [], None
    for i, v in enumerate(np.append(edb > thr, False)):
        if v and a0 is None:
            a0 = i
        if not v and a0 is not None:
            if i - a0 >= 5:
                runs.append([a0 / 100, i / 100])
            a0 = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < BRIDGE:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def cut(take, lines, voice_dir, out):
    """Write voice_dir/<id>.wav for every line; print what was cut."""
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    stream = stream_of(transcribe(model, take, out / "take_words.json"))
    chosen, cands = choose(lines, stream)

    x, sr = sf.read(take, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = sosfiltfilt(butter(4, 75, btype="highpass", fs=sr, output="sos"), x)
    hop = sr // 100
    env = np.sqrt(np.convolve(x**2, np.ones(hop) / hop, mode="same"))[::hop]
    edb = 20 * np.log10(env + 1e-12)
    thr = np.percentile(edb, 10) + VOICED_DB
    runs = voiced_runs(edb, thr)

    def complete(line, a, b):
        piece = x[int(a * sr) : int(b * sr)]
        segs, _ = model.transcribe(
            resample_poly(piece, SR_WHISPER, sr).astype(np.float32),
            beam_size=5,
            language="en",
        )
        heard = " ".join(s.text for s in segs)
        mine, got = words(line), words(heard)
        ratio = SequenceMatcher(None, mine, got).ratio()
        # A restart repeats the line's opening, which a reading that Whisper
        # mishears in places ("Aput 20" for "up to twenty") still shows.
        return ratio >= COMPLETE or (ratio >= REREAD and got[:2] == mine[:2])

    def quietest(t0, t1):
        i0, i1 = int(max(t0, 0) * 100), int(min(t1, len(edb) / 100) * 100)
        return (i0 + int(np.argmin(edb[i0:i1]))) / 100 if i1 > i0 else None

    spans = []
    starts = {c[0] for c in chosen}
    for line, (s, e, _), cs in zip(lines, chosen, cands):
        note = f"reading {[c[0] for c in cs].index(s) + 1} of {len(cs)}"
        # A misheard first or last word ("low" for "slow") falls outside the
        # match. As many words as the line misses at each end, up to EDGE,
        # that run on with a pause under RUN_ON and belong to no other line's
        # reading, are the line's own.
        mine_w = words(line["text"])
        got = SequenceMatcher(
            None, mine_w, [t[0] for t in stream[s:e]]
        ).get_matching_blocks()[:-1]
        head = got[0].a if got else 0
        tail = len(mine_w) - (got[-1].a + got[-1].size) if got else 0
        for _ in range(min(EDGE, tail)):
            if (
                e < len(stream)
                and e not in starts
                and stream[e][1] - stream[e - 1][2] < RUN_ON
            ):
                e += 1
        for _ in range(min(EDGE, head)):
            if (
                s > 0
                and not any(c[0] <= s - 1 < c[1] for c in chosen)
                and stream[s][1] - stream[s - 1][2] < RUN_ON
            ):
                s -= 1
        w0, w1 = stream[s][1], stream[e - 1][2]
        mine = [r for r in runs if w0 <= (r[0] + r[1]) / 2 <= w1 + 0.1]
        mine = mine or [[w0, w1]]
        # A false start is followed by a long pause and the whole line again;
        # the latest pause after which the whole line follows marks it.
        for k in range(len(mine) - 1, 0, -1):
            if mine[k][0] - mine[k - 1][1] > RESTART and complete(
                line["text"], mine[k][0], mine[-1][1]
            ):
                a0, a1 = mine[0][0], mine[k - 1][1]
                note += f", false start {a0:.1f}-{a1:.1f} s dropped"
                mine = mine[k:]
                break
        a, b = mine[0][0], mine[-1][1]
        prev_end = max([r[1] for r in runs if r[1] <= a] or [0.0])
        next_start = min([r[0] for r in runs if r[0] >= b] or [x.size / sr])
        lo = max(a - PRE, (prev_end + a) / 2)
        hi = min(b + POST, (b + next_start) / 2)
        if s > 0 and a < stream[s - 1][2]:
            q = quietest(stream[s - 1][2] - 0.3, stream[s][1] + 0.1)
            if q is not None and q > lo:
                lo = q
        if e < len(stream) and b > stream[e][1]:
            q = quietest(stream[e - 1][2] - 0.1, stream[e][1] + 0.3)
            if q is not None and q < hi:
                hi = q
        spans.append((lo, hi, note))

    clips = [x[int(a * sr) : int(b * sr)] for a, b, _ in spans]
    level = [
        np.sqrt(np.mean(c[np.abs(c) > 10 ** (thr / 20)] ** 2)) for c in clips
    ]
    gain = min(
        10 ** (TARGET_RMS_DB / 20) / np.median(level),
        10 ** (PEAK_DB / 20) / max(np.abs(c).max() for c in clips),
    )
    fade = int(0.005 * sr)
    for line, clip, (a, b, note) in zip(lines, clips, spans):
        clip = clip * gain
        clip[:fade] *= np.linspace(0, 1, fade)
        clip[-fade:] *= np.linspace(1, 0, fade)
        if sr != SR_OUT:
            clip = resample_poly(clip, SR_OUT, sr)
        sf.write(
            voice_dir / f"{line['id']}.wav",
            clip.astype(np.float32),
            SR_OUT,
            subtype="PCM_16",
        )
        print(f"  {line['id']:4s} {a:6.2f}-{b:6.2f} s  {note}", flush=True)
    print(f"  gain {20 * np.log10(gain):+.1f} dB", flush=True)
