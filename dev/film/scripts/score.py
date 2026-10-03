#!/usr/bin/env python3
"""The film's score, and the narration mixed with it.

    python -u scripts/score.py V

The score is the style frame's (synth.py, whose instruments, levels and mix
this file imports): a tracker-style piece in E
minor near 96 BPM, with a few voices, a pulsing bass, a cold pad and a
sparse lead. Here it runs the length of the film and follows what the film
shows. The full arrangement plays while the film trusts a surrogate
(Bayesian optimization, the SEARCH); a low drone and dry taps while it
steps without one (direct search, the POLL); the flat second of the
Phrygian mode colours the surrogate being fooled; a build leads to the
search that lands near the bottom of the trench, where the harmony lifts
to C major 7 and the lead enters; and the lift returns for the record and
resolves to E minor as the version pops on the end card.

The tempo flexes from section to section so that the search near the
bottom of the trench, the reveal of the true landscape and the version's
pop, which a ritardando reaches, fall on a downbeat, and so does the start
of every scene but the sixth, whose section starts at the reveal. Every
event that the film shows sounds on the nearest thirty-second note: an
evaluation as a glassy tick pitched by the height of its ground, a search
that succeeds as a bell pitched by how low it lands, one that fails as a
thud with a dissonant stab, a step of a poll as a dry tap, an octave lower
while the mesh is large. The score ducks under the voice.

The sections start at lines and events of the film, so they follow a new
take of the voice; the bars inside them were fitted to the draft voice's
lines. ``check_form`` warns when a switch to the POLL, a return to the
SEARCH, the lesson or a fooled surrogate falls in a bar of another kind:
re-fit SONG where it does, with score_plan.txt, which lists every bar with
the events in it.

V is the folder of a voice's media. Reads V/events.json (``node
scripts/record.mjs V/events.json --film``) and V/narration.wav (voice.py).
Writes into V:

    score.wav       the score as it sits under the voice, ducked
    mix.wav         the narration with the score
    stems/*.wav     the groups of instruments, before the ducking
    score_plan.txt  every bar: its time, tempo, chord and patterns, and the
                    events that sound in it

``mux.py`` puts mix.wav under the recorded video at -16 LUFS.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import synth as ms
from scipy.signal import butter, fftconvolve, sosfiltfilt

SR = ms.SR = 48000  # the narration's rate

# ══════════════════════════════════════════════════════════════════════════
# The form
# ══════════════════════════════════════════════════════════════════════════

# The sections, in order. Each starts at an anchor of the film and ends where
# the next starts: a line's start (the shot changes there), or an event.
ANCHORS = {
    "problem": 0.0,
    "bayesopt": ("line", "2.1"),
    "direct": ("line", "3.1"),
    "bads": ("line", "4.1"),
    "run": ("line", "5.1"),
    "settle": ("event", "big"),  # the search near the bottom of the trench
    "close": ("event", "reveal"),  # the true landscape
    "record": ("line", "7.1"),
    "join": ("line", "7.4"),  # "Join them."
    "end": ("event", "pop"),  # the version pops on the end card
}
# The sections whose beats slow down, by this fraction of the first beat per
# beat; the others keep an even tempo.
RITARDANDO = {"join": 0.08}

# The levels that a bar's mode sets, which a bar can change: the pad's level
# and the cutoff of its low-pass, the same for the arpeggio, the bass's level
# and how far its filter opens, and the drone's level. A number holds over
# the bar, a pair goes from the first to the second across it, and a list of
# (fraction of the bar, value) places points within it.
MODES = {
    # the drone and a dark pad: the openings
    "intro": dict(pad=0.3, pcut=420, arp=0, acut=500, bass=0, bright=0.2, drone=1.0),
    # the kick and the bass pulse: something starts
    "pulse": dict(pad=0.55, pcut=1000, arp=0.5, acut=1400, bass=0.55, bright=0.35, drone=0.6),
    # trusting a surrogate: the whole arrangement
    "search": dict(pad=0.9, pcut=2000, arp=0.9, acut=2900, bass=1.0, bright=0.85, drone=0),
    # the surrogate fooled: louder and brighter
    "fooled": dict(pad=1.0, pcut=2400, arp=1.05, acut=3300, bass=1.0, bright=1.1, drone=0),
    # into a POLL: everything closes into the drone
    "break": dict(pad=(1.0, 0), pcut=(2400, 160), arp=(1.0, 0), acut=(2900, 250), bass=(1.0, 0), bright=(1.0, 0.1), drone=(0, 1.0)),
    # stepping without a surrogate: the drone alone, under the taps
    "poll": dict(pad=0, pcut=300, arp=0, acut=450, bass=0, bright=0.1, drone=1.0),
    "build": dict(pad=(0.35, 1.0), pcut=(500, 2600), arp=0, acut=500, bass=(0.55, 1.0), bright=(0.5, 1.3), drone=0),
    "arrive": dict(pad=1.0, pcut=3000, arp=0, acut=500, bass=0.9, bright=0.75, drone=0),
    "lift": dict(pad=1.0, pcut=2800, arp=1.0, acut=3400, bass=1.0, bright=1.05, drone=0),
    "out": dict(pad=(1.0, 0.55), pcut=(2800, 700), arp=(0.8, 0), acut=(3400, 600), bass=0.9, bright=0.7, drone=(0, 1.0)),
    "reveal": dict(pad=1.0, pcut=3400, arp=0, acut=500, bass=0.8, bright=0.6, drone=0.3),
    "calm": dict(pad=0.8, pcut=2400, arp=0.6, acut=2400, bass=0.6, bright=0.5, drone=0),
    "final": dict(pad=1.0, pcut=3000, arp=0, acut=500, bass=0.8, bright=0.6, drone=(0, 0.6)),
}  # fmt: skip

# One row per bar: mode, chord, drums, bass, arpeggio (None: silent), and
# what it changes: "beats" when it has fewer than four, levels of MODES.
# fmt: off
SONG = {
    "problem": [  # 1.1-1.5: the question, the landscape, the first evaluation
        ("intro", "Em", None, None, None, dict(pad=(0, 0.28), pcut=(300, 400), drone=(0, 1.0))),
        ("intro", "Em", None, None, None, dict(pad=0.3)),
        ("intro", "Em", None, None, None, dict(pad=(0.3, 0.38), pcut=(420, 560))),
        ("pulse", "Em", "pulse", "hint", None, dict(pad=0.4, pcut=(560, 750), bass=0.4, bright=0.2, drone=1.0)),
        ("pulse", "Em", "pulse", "hint", None, dict(pad=(0.5, 0.85), pcut=(800, 1700), bass=0.5, bright=0.3, drone=(1.0, 0.7))),
        ("pulse", "Em", "pulse", "hint", None, dict(pad=0.85, pcut=1700, bass=0.5, bright=0.3, drone=0.7)),
        ("intro", "Em", None, None, None, dict(pad=(0.85, 0.4), pcut=(1700, 450), drone=(0.7, 1.0))),
        ("pulse", "Em", "pickup", "hint", "pickup", dict(pad=0.4, pcut=(450, 900), arp=0.5, acut=(900, 1700), bass=0.45, bright=0.3, drone=(1.0, 0.5))),
    ],
    "bayesopt": [  # 2.1-2.4: a surrogate, trusted, then fooled
        ("pulse", "Em", "pulse", "eighths", None, dict(pad=0.6, pcut=1100, bass=0.7, bright=0.45, drone=(0.5, 0))),
        ("search", "Em", "A", "eighths", "roll", dict(pad=(0.7, 0.9), pcut=(1300, 1900), arp=(0.6, 0.85), acut=(1800, 2700), bright=0.7)),
        ("search", "Em", "B", "eighths", "roll", {}),
        ("search", "Em", "A", "eighths", "roll", {}),
        ("search", "G", "A", "eighths", "roll", {}),
        ("search", "Dsus2", "B", "eighths", "roll", {}),
        ("fooled", "F/E", "A16", "lean", "climb", dict(arp=(0.95, 1.1), acut=(2900, 3400), bright=(1.0, 1.1))),
        ("fooled", "Fmaj7#11", "B16", "sixteenths", "climb", dict(arp=(1.1, 1.25), acut=(3400, 3900), bright=(1.1, 1.25))),
        ("break", "Em", "break", "sixteenths", "roll", {}),
    ],
    "direct": [  # 3.1-3.4: no surrogate; a clock, and the taps of the steps
        ("poll", None, None, None, None, {}),
        ("poll", None, "clock", None, None, {}),
        ("poll", None, "clock", None, None, {}),
        ("poll", None, "clock", None, None, {}),
        ("poll", None, "clock", None, None, {}),
        ("poll", None, "clock", None, None, {}),
        ("poll", "Em", "clock", None, None, dict(pad=(0, 0.3), pcut=400)),
        ("poll", "Em", "clock", None, None, dict(pad=(0.3, 0.5), pcut=(400, 900))),
    ],
    "bads": [  # 4.1-4.5: the two stages, alternating
        ("pulse", "Em", "pulse", "hint", None, dict(pad=(0.5, 0.8), pcut=(900, 1800), bass=0.5, bright=0.35, drone=(1.0, 0.4))),
        ("pulse", "Em", "pulse", "eighths", "pickup", dict(pad=0.8, pcut=1800, bass=0.7, bright=0.5, arp=0.7, acut=2000, drone=(0.4, 0))),
        ("search", "Em", "A", "eighths", "roll", {}),
        ("search", "Em", "Ahalf", "half", "rollhalf", dict(
            pad=[(0, 0.9), (0.5, 0.9), (0.65, 0)], pcut=[(0, 2000), (0.5, 2000), (0.65, 300)],
            drone=[(0, 0), (0.5, 0), (0.7, 1.0)])),
        ("poll", None, None, None, None, {}),
        ("search", "G", "A", "eighths", "roll", dict(pad=(0.6, 0.9), drone=[(0, 1.0), (0.1, 0)])),
        ("break", "F/E", "break", "lean", "climb", {}),
        ("poll", None, None, None, None, {}),
        ("poll", None, None, None, None, {}),
        ("poll", None, None, None, None, {}),
        ("build", "B7sus4", "build", "build", None, {}),
        ("search", "Em", "A16", "sixteenths", "roll", {}),
    ],
    "run": [  # 5.1-5.6: the real run
        ("intro", "Em", None, None, None, dict(pad=0.35, pcut=(700, 450), drone=(0.6, 1.0))),
        ("pulse", "Em", "pulse", "hint", None, dict(pad=0.45, pcut=(500, 900), bass=0.45, bright=0.3, drone=(1.0, 0.5))),
        ("search", "Em", "A", "eighths", "roll", dict(pad=(0.7, 0.9), pcut=(1500, 2000), arp=(0.7, 0.9), drone=(0.5, 0))),
        ("search", "G", "B", "eighths", "roll", {}),
        ("fooled", "F/E", "A16", "lean", "climb", dict(arp=(0.95, 1.1), acut=(2900, 3400), bright=(1.0, 1.1))),
        ("fooled", "Fmaj7#11", "B16", "sixteenths", "climb", dict(arp=(1.1, 1.25), acut=(3400, 3900), bright=(1.1, 1.25))),
        ("break", "Em", "break", "sixteenths", "roll", {}),
        ("poll", None, None, None, None, {}),
        ("poll", None, None, None, None, {}),
        ("poll", None, "pickup", None, None, {}),
        ("build", "B7sus4", "build1", "build", None, dict(pad=(0.3, 0.6), pcut=(450, 1200), bass=(0.4, 0.7), bright=(0.4, 0.8))),
        ("build", "B7sus4", "build", "build", None, dict(pad=(0.6, 1.0), pcut=(1200, 2600), bass=(0.7, 1.0), bright=(0.8, 1.3))),
    ],
    "settle": [  # 5.6-6.1: the arrival, the lift, the settling
        ("arrive", "Cmaj7", "half", "soft", None, {}),
        ("lift", "G", "A16", "sixteenths", "roll", {}),
        ("lift", "Dsus2", "B16", "sixteenths", "roll", {}),
        ("out", "Em", "out", "long", "out", {}),
    ],
    "close": [  # 6.1-6.2: the true landscape; the two stages summed up
        ("reveal", "Cmaj7", None, "long", None, {}),
        ("calm", "G", None, "pedal", "roll", {}),
        ("poll", "Em", "clock", None, None, dict(beats=3, pad=(0.5, 0.3), pcut=600)),
    ],
    "record": [  # 7.1-7.3: the record
        ("lift", "Em", "A", "eighths", "roll", {}),
        ("lift", "Cmaj7", "A", "eighths", "roll", {}),
        ("lift", "G", "B", "eighths", "roll", {}),
        ("lift", "Dsus2", "A16", "sixteenths", "roll", {}),
        ("lift", "Em", "A16", "sixteenths", "roll", {}),
        ("lift", "Cmaj7", "B16", "sixteenths", "roll", {}),
        ("lift", "G", "A16", "sixteenths", "roll", {}),
        ("lift", "Dsus2", "build", "build", "roll", {}),
    ],
    "join": [  # 7.4: "Join them.", slowing
        ("arrive", "Cmaj7", None, "long", None, {}),
    ],
    "end": [  # the end card
        ("final", "Em", None, "long", None, {}),
    ],
}
# fmt: on

# The style frame's chords, and the arpeggio over C major 7, which it never
# arpeggiates: the chord's upper tones, the bass holding the root.
CHORDS = dict(
    ms.CHORDS,
    Cmaj7=(*ms.CHORDS["Cmaj7"][:2], ["E4", "G4", "B4", "E5", "G5", "B5"]),
)

# Patterns that the style frame does not have.
DRUMS = dict(
    ms.DRUMS,
    clock=("................", "................", "2...2...2...2..."),
    Ahalf=("9...............", "....9...........", "4.7.4.7.2......."),
    build1=("3...4...4...5...", "................", "................"),
)
BASS = dict(ms.BASS, half="0.0.0.+.........", pedal="0.......0.......")
ARP = dict(ms.ARP, rollhalf=ms.ARP["roll"][:8] + [None] * 8)

# The lead: section, bar, sixteenth, length in sixteenths, note.
LEAD = [
    ("settle", 0, 0, 6, "E5"),
    ("settle", 0, 6, 2, "D5"),
    ("settle", 0, 8, 7, "B4"),
    ("join", 0, 6, 2, "A4"),
    ("join", 0, 8, 4, "B4"),
    ("join", 0, 12, 4, "D5"),
    ("end", 0, 0, 15, "E5"),
]

# ══════════════════════════════════════════════════════════════════════════
# The mix
# ══════════════════════════════════════════════════════════════════════════

# Changes to the style frame's levels (ms.LEVEL_DB), for a score under a
# voice: the drums, the arpeggio and the stab of a miss, which compete with
# speech, are softer.
LEVEL_DB = dict(
    ms.LEVEL_DB,
    kick=-10.0,
    snare=-11.0,
    hat=-20.0,
    arp=-13.0,
    lead=-17.0,
    miss_thud=-14.0,
    miss_stab=-14.5,
    tap=-14.0,
    landing=-21.0,
    tick=-27.0,  # a step of a fast run
    bloom=-27.0,  # the breath of a surface rising
    fooled=-17.0,  # the held stab of a surrogate fooled
    name=-19.0,  # the bell of a label
)
# The score's loudness under the voice, in LU from the voice's own: the score
# as it sounds after the ducking, over the whole film.
SCORE_LU = -17.0
# How far the score dips while the voice speaks: the music, and the events
# and effects, which keep more of their level.
DUCK = {"music": 0.45, "events": 0.25}
MUSIC = ("drums", "bass", "pad", "arp", "lead")
FADE_OUT = 1.6  # seconds, to the end of the film

# ══════════════════════════════════════════════════════════════════════════
# The time: anchors, beats, bars
# ══════════════════════════════════════════════════════════════════════════


def anchors(ev):
    """The start of every section, and the end of the film. An anchor names
    a line, whose start it takes, or a kind of event, whose first event it
    takes."""
    first = {}
    for t, kind, _, _ in ev["events"]:
        first.setdefault(kind, t)
    out = {}
    for name, a in ANCHORS.items():
        if isinstance(a, tuple):
            table = ev["lines"] if a[0] == "line" else first
            if a[1] not in table:
                raise SystemExit(
                    f"section {name}: the film has no {a[0]} {a[1]}"
                )
            a = table[a[1]][0] if a[0] == "line" else table[a[1]]
        out[name] = float(a)
    end = float(ev["seconds"])
    starts = list(out.items())
    for (name, t0), t1 in zip(starts, [t for _, t in starts[1:]] + [end]):
        if not t0 < t1:
            raise SystemExit(
                f"section {name} starts at {t0:.3f} s, not before the next at {t1:.3f} s"
            )
    return out, end


class Grid:
    """The beats of the film: every beat's time, from the sections' anchors
    and their bars. A bar is (section, index, first beat, beats, row)."""

    def __init__(self, ev):
        starts, end = anchors(ev)
        names = list(SONG)
        assert names == list(
            ANCHORS
        ), "SONG and ANCHORS name the same sections"
        beats, self.bars, self.section = [], [], {}
        for i, name in enumerate(names):
            t0 = starts[name]
            t1 = starts[names[i + 1]] if i + 1 < len(names) else end
            rows = SONG[name]
            nb = sum(row[5].get("beats", 4) for row in rows)
            lengths = 1.0 + RITARDANDO.get(name, 0.0) * np.arange(nb)
            lengths *= (t1 - t0) / lengths.sum()
            first = len(beats)
            beats.extend(t0 + np.concatenate([[0.0], np.cumsum(lengths)[:-1]]))
            self.section[name] = (first, nb, t0, t1)
            k = first
            for j, row in enumerate(rows):
                n = row[5].get("beats", 4)
                self.bars.append((name, j, k, n, row))
                k += n
        beats.append(end)
        self.beats = np.array(beats)
        self.end = end

    def time(self, b):
        """The time of beat b, a global index that may have a fraction."""
        k = min(int(np.floor(b)), len(self.beats) - 2)
        return self.beats[k] + (b - k) * (self.beats[k + 1] - self.beats[k])

    def at(self, name, bar, beat=0.0):
        """The time of a beat of a bar of a section."""
        first = [b for b in self.bars if b[0] == name][bar][2]
        return self.time(first + beat)

    def snap(self, t, div=8):
        """The time t moved to the nearest 1/div of a beat."""
        k = int(
            np.clip(
                np.searchsorted(self.beats, t, "right") - 1,
                0,
                len(self.beats) - 2,
            )
        )
        d = self.beats[k + 1] - self.beats[k]
        return self.beats[k] + round((t - self.beats[k]) / d * div) / div * d

    def sixteenths(self, bar):
        _, _, first, n, _ = bar
        return np.array([self.time(first + s / 4.0) for s in range(4 * n)])

    def bar_of(self, t):
        for bar in self.bars:
            if self.time(bar[2]) <= t < self.time(bar[2] + bar[3]):
                return bar
        return self.bars[-1]


def lanes(grid):
    """The automation lanes of the style frame's mix, from the bars' levels."""
    names = {"pad": "pad_gain", "pcut": "pad_cutoff", "arp": "arp_gain", "acut": "arp_cutoff",
             "bass": "bass_gain", "bright": "bass_bright", "drone": "drone_gain"}  # fmt: skip
    out = {lane: [] for lane in names.values()}
    out["echo_gate"] = []
    for _, _, first, n, row in grid.bars:
        mode, levels = row[0], dict(
            MODES[row[0]], **{k: v for k, v in row[5].items() if k != "beats"}
        )
        t0, t1 = grid.time(first), grid.time(first + n)
        for key, lane in names.items():
            v = levels[key]
            if isinstance(v, list):
                points = [(t0 + f * (t1 - t0), x) for f, x in v]
            else:
                a, b = v if isinstance(v, tuple) else (v, v)
                points = [(t0 + 0.03, a), (t1 - 0.03, b)]
            out[lane] += [
                (t, max(x, 1.0) if lane.endswith("cutoff") else x)
                for t, x in points
            ]
        # The arpeggio's echoes stop where a POLL starts, so that it starts dry.
        gate = 0.0 if mode == "poll" else 1.0
        out["echo_gate"] += [(t0 + 0.02, gate), (t1 - 0.02, gate)]
    return out


# ══════════════════════════════════════════════════════════════════════════
# The sounds that the style frame does not have
# ══════════════════════════════════════════════════════════════════════════


def tick(f0):
    """A step of a fast run: a sine of a few milliseconds and a click."""
    n = int(0.045 * SR)
    t = np.arange(n) / SR
    x = np.sin(ms.TAU * f0 * t) * np.exp(-t / 0.008)
    noise = ms.bandpass(ms.rng("tick").standard_normal(n), 2500.0, 7000.0)
    x += 0.4 * ms.unit(noise) * np.exp(-t / 0.0008)
    return ms.unit(x * ms.fade_out(n, 0.01))


def bloom(seconds):
    """A breath of noise that swells and fades, its low-pass opening: a
    surface rising."""
    n = int(seconds * SR)
    u = np.arange(n) / n
    noise = ms.bandpass(ms.rng("bloom").standard_normal((n, 2)), 600.0, 7000.0)
    x = ms.sweep_lowpass(
        noise, 700.0 * (5000.0 / 700.0) ** np.sin(0.5 * np.pi * u)
    )
    return ms.unit(x * (np.sin(np.pi * u) ** 2)[:, None])


def mesh_glide(f_from, octaves):
    """The mesh changing size: the style frame's falling note, which here
    glides by `octaves` (down when negative) from f_from."""
    n = int(2.4 * SR)
    t = np.arange(n) / SR
    u = np.clip((t - 0.06) / 0.26, 0.0, 1.0)
    u = u * u * (3.0 - 2.0 * u)
    out = np.empty((n, 2))
    for channel, cents in enumerate((-4.0, 4.0)):
        f = f_from * 2.0 ** (octaves * u + cents * u / 1200.0)
        out[:, channel] = ms.osc(f, n, "tri", 2600.0) + 0.25 * ms.osc(
            f, n, "saw", 1400.0
        )
    env = np.minimum(t / 0.004, 1.0) * np.exp(-t / 0.6) * ms.fade_out(n, 0.3)
    return ms.unit(out * env[:, None])


# ══════════════════════════════════════════════════════════════════════════
# The film's events
# ══════════════════════════════════════════════════════════════════════════

LANDING_SCALE = ms.LANDING_SCALE  # E4 to A5: the lower the ground, the lower
HIT_SCALE = ("G5", "A5", "B5", "D6", "E6")  # the lower the point, the higher
TICK_SCALE = ("E5", "G5", "A5", "B5", "D6", "E6")
NAME_NOTES = ("E5", "G5", "A5", "B5")  # the four fields of line 7.3, rising


def cues(ev, grid):
    """The events as the score plays them: dicts with the time on the grid,
    the time in the picture, the kind, the value, the shot, an accent in dB
    and the rate of the taps (0.5 an octave lower, while the mesh is
    large). Fast runs keep one event per thirty-second note."""
    out, taken, rate, scene = [], set(), 1.0, None
    rows = [
        dict(t=grid.snap(t), pic=t, kind=k, v=v, shot=s, accent=0.0)
        for t, k, v, s in ev["events"]
    ]
    for c in rows:
        if c["shot"][0] != scene:  # each scene's mesh starts at its own size
            scene, rate = c["shot"][0], 1.0
        c["rate"] = rate
        if c["kind"] == "grow":
            rate = 0.5
        elif c["kind"] == "shrink":
            rate = 1.0
        if c["kind"] in ("tick", "eval") and c["shot"] in ("3.4", "5.7"):
            key = round(c["t"], 4)
            if key in taken:
                continue
            taken.add(key)
        out.append(c)
    # What each shot's events say, as accents and a few changes of kind.
    by = {}
    for c in out:
        by.setdefault((c["shot"], c["kind"]), []).append(c)
    for i, c in enumerate(
        by.get(("2.4", "eval"), [])
    ):  # pointing to the wrong places
        c["kind"], c["accent"] = "miss", -5.0 + i
    for c in by.get(("4.3", "miss"), []) + by.get(("5.2", "miss"), []):
        c["accent"] = -4.0  # the SEARCH heads downhill all the same
    for c in by.get(("5.2", "tap"), []):
        c["accent"] = -3.0
    misses = by.get(("5.3", "miss"), [])
    for i, c in enumerate(misses):  # three misses, then four fails in a row
        c["accent"] = (
            -4.0 if i < len(misses) - 4 else -2.0 + (i - len(misses) + 4)
        )
    for shot in ("3.4", "5.7"):  # the fast runs settle
        run = [
            c
            for c in out
            if c["shot"] == shot and c["kind"] in ("tick", "eval")
        ]
        for i, c in enumerate(run):
            c["accent"] = -1.0 - 6.0 * i / max(1, len(run) - 1)
    for c in by.get(("5.4", "taphit"), []):
        c["kind"] = "taphit_big"  # far lower than any point before
    return out


def height(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def note_of(scale, h):
    return scale[min(int(h * len(scale)), len(scale) - 1)]


def play_film_events(film_cues, grid, events, drums, fx):
    """The sounds of the events, into the stems of the events, the drums and
    the effects. Returns (time, strength) of the events that duck the pad and
    the arpeggio."""
    crash = ms.make_crash()
    thud, stab = ms.make_miss()
    tap = ms.make_tap()
    tap_dull = ms.unit(ms.lowpass(tap, 1100.0))
    bells = ms.Stem()
    evals = [c["v"] for c in film_cues if c["kind"] in ("eval", "deep", "big")]
    lo, hi = min(evals), max(evals)
    names = iter(NAME_NOTES)
    pan = ms.rng("film landings").uniform(-0.4, 0.4, len(film_cues))
    ducks = []
    L = LEVEL_DB

    def bell(t0, note, level, tau=0.4, index=0.8):
        bells.put(
            t0,
            ms.bell(
                ms.hz(note) if isinstance(note, str) else note, tau, index
            ),
            level,
            send=ms.SEND["bell"],
        )

    for i, c in enumerate(film_cues):
        t0, kind, v, a, rate = (
            c["t"],
            c["kind"],
            c["v"],
            c["accent"],
            c["rate"],
        )
        if kind == "eval":
            h = height(v, lo, hi)
            x = ms.landing_tick(ms.hz(note_of(LANDING_SCALE, h)), False)
            events.put(
                t0, x, L["landing"] + a, pan=pan[i], send=ms.SEND["landing"]
            )
        elif kind == "deep":
            x = ms.landing_tick(ms.hz(LANDING_SCALE[0]), True)
            events.put(
                t0, x, L["landing_deep"] + a, send=2.0 * ms.SEND["landing"]
            )
            ducks.append((t0, 1.0))
        elif kind == "tick":
            f = ms.hz(note_of(TICK_SCALE, height(v, lo, hi)))
            events.put(
                t0, tick(f), L["tick"] + a, pan=pan[i], send=ms.SEND["tap"]
            )
        elif kind == "hit":
            note = (
                "D6" if v == 0 else note_of(HIT_SCALE, 1.0 - height(v, lo, hi))
            )
            bell(t0, note, L["small_hit"] + a, 0.32, 0.7)
            ducks.append((t0, 1.0))
        elif kind == "miss":
            events.put(t0, thud, L["miss_thud"] + a, send=ms.SEND["miss"])
            events.put(t0, stab, L["miss_stab"] + a, send=ms.SEND["miss"])
            ducks.append((t0, 1.0))
        elif kind == "tap":
            events.put(
                t0,
                ms.at_rate(tap_dull, rate),
                L["tap"] - 2.0 + a,
                send=ms.SEND["tap"],
            )
        elif kind in ("taphit", "taphit_big"):
            events.put(
                t0, ms.at_rate(tap, rate), L["tap"] + a, send=ms.SEND["tap"]
            )
            bell(
                t0,
                ms.hz(ms.POLL_HIT_NOTE) * rate,
                L["bell"] + a,
                0.40 if rate == 1.0 else 0.45,
            )
            if kind == "taphit_big":  # and the fifth above it, brighter
                bell(t0, ms.hz("E6") * rate, L["bell"] + 3.0 + a, 0.7, 0.9)
            ducks.append((t0, 1.0))
        elif kind in ("grow", "shrink"):
            f = ms.hz(ms.MESH_FROM) * rate
            octaves = -1.0 if kind == "grow" else 1.0
            events.put(
                t0,
                ms.at_rate(tap, rate),
                L["tap"] - 1.0 + a,
                send=ms.SEND["tap"],
            )
            events.put(
                t0,
                mesh_glide(f, octaves),
                L["mesh"] - (0 if kind == "grow" else 3) + a,
                send=ms.SEND["mesh"],
            )
            ducks.append((t0, 1.0))
        elif kind == "fooled":
            events.put(t0, stab, L["fooled"] + a, send=0.45)
        elif kind == "poll":  # a powerdown that lands on the POLL
            fx.put(t0 - 1.35, ms.make_powerdown(1.35), L["powerdown"])
        elif kind in ("rise", "lesson"):
            fx.put(t0, bloom(2.2), L["bloom"] + a, send=ms.SEND["fx"])
        elif kind == "named":
            note = "E5" if c["shot"] == "1.4" else next(names)
            bell(t0, note, L["name"] + a, 0.6, 0.6)
        elif kind == "whoosh":  # the points travel back to the SEARCH
            later = [
                d["t"]
                for d in film_cues
                if d["kind"] == "search" and d["t"] > t0
            ]
            seconds = (later[0] if later else t0 + 1.8) - t0
            fx.put(
                t0,
                ms.make_riser(seconds),
                L["riser"] - 3.0,
                send=ms.SEND["fx"],
            )
        elif kind == "search":
            pass  # the groove returns on the bar: see effects()
        elif kind == "big":
            level = L["big_hit"] + a
            for name, lower in zip(ms.BIG_HIT_NOTES, (0.0, 2.0)):
                bell(t0, name, level - lower, 0.95, 0.9)
            bell(t0, ms.BIG_HIT_LOW, level - 6.0, 0.8, 0.5)
            drums.put(t0, crash, L["crash"] - 3.0, send=ms.SEND["crash"])
            ducks.append((t0, 1.0))
        elif kind == "reveal":
            for name, lower in (("E5", 2.0), ("B5", 0.0), ("E6", 4.0)):
                bell(t0, name, L["big_hit"] - 2.0 - lower, 1.2, 0.6)
            drums.put(t0, crash, L["crash"], send=ms.SEND["crash"])
            ducks.append((t0, 1.0))
        elif kind == "you":
            bell(t0, "B5", L["name"] - 1.0 + a, 0.6, 0.6)
        elif kind == "pop":
            for name, lower in (("E6", 0.0), ("B5", 3.0), ("E5", 6.0)):
                bell(t0, name, L["big_hit"] - 2.0 - lower, 1.0, 0.7)
            drums.put(t0, crash, L["crash"] - 4.0, send=ms.SEND["crash"])
            ducks.append((t0, 1.0))
        else:
            raise ValueError(
                f"no sound for the event {kind!r} of shot {c['shot']}"
            )

    events.dry += bells.dry
    events.send += bells.send
    echoes = ms.echoes(bells.dry, "bell")
    events.dry += echoes
    events.send += ms.SEND["bell"] * echoes
    return ducks


def effects(grid, fx, drums, events):
    """The swells, crashes and falls at the changes of section, and the
    steady taps of line 6.2."""
    at = grid.at
    swell, crash = ms.make_swell(ms.BEAT), ms.make_crash()
    L = LEVEL_DB
    for t0, level in [
        (at("bayesopt", 1), -4.0),  # the surrogate rises
        (at("bads", 0), -2.0),
        (at("bads", 5), -5.0),
        (at("bads", 11), 0.0),  # the next SEARCH
        (at("run", 2), -5.0),
        (at("close", 0), 0.0),  # the reveal
        (at("record", 0), -2.0),
        (at("end", 0), -3.0),  # the pop
    ]:
        fx.put(t0 - ms.BEAT, swell, L["swell"] + level, send=ms.SEND["fx"])
    for t0, level in [
        (at("bads", 0), -4.0),
        (at("bads", 11), -3.0),
        (at("run", 0), -6.0),
        (at("record", 0), -2.0),
    ]:
        drums.put(t0, crash, L["crash"] + level, send=ms.SEND["crash"])
    # The falls into a POLL: the breakdown's noise, and a powerdown where no
    # event of the film brings one.
    for t0 in (at("bayesopt", 8), at("run", 6), at("bads", 6)):
        fx.put(
            t0, ms.make_downlifter(2.2), L["downlifter"], send=ms.SEND["fx"]
        )
    fx.put(at("direct", 0) - 1.35, ms.make_powerdown(1.35), L["powerdown"])
    fx.put(
        at("bads", 3, 2.6) - 1.35,
        ms.make_powerdown(1.35),
        L["powerdown"] - 2.0,
    )
    # The build: a riser into the arrival.
    fx.put(
        at("run", 11),
        ms.make_riser(at("settle", 0) - at("run", 11)),
        L["riser"],
        send=ms.SEND["fx"],
    )
    # "Slow but steady": a tap on each beat of the last bar of line 6.2.
    tap_dull = ms.unit(ms.lowpass(ms.make_tap(), 1100.0))
    for beat in range(3):
        events.put(
            at("close", 2, beat), tap_dull, L["tap"] - 4.0, send=ms.SEND["tap"]
        )


# ══════════════════════════════════════════════════════════════════════════
# The sequencer
# ══════════════════════════════════════════════════════════════════════════


def play_bars(grid, stems):
    """The drums, the bass and the arpeggio, bar by bar. Returns the kicks
    as (time, strength), for the sidechain."""
    kick, snare, hat = ms.make_kick(), ms.make_snare(), ms.make_hat()
    drums, bass, arp = stems["drums"], stems["bass"], stems["arp"]
    bass_ref = np.abs(
        ms.bass_note(ms.hz("E2"), 0.72 * ms.STEP, 1.0, 1.0)
    ).max()
    arp_ref = np.abs(ms.arp_note(ms.hz("E4"), 1.0, 1500.0)).max()
    L = LEVEL_DB
    kicks = []
    for bar in grid.bars:
        _, _, first, n, (mode, chord, drum_name, bass_name, arp_name, _) = bar
        times = grid.sixteenths(bar)
        step = (grid.time(first + n) - grid.time(first)) / (4 * n)
        if drum_name:
            for t0, k, s, h in zip(times, *DRUMS[drum_name]):
                if k != ".":
                    drums.put(t0, kick * (int(k) / 9.0), L["kick"])
                    kicks.append((t0, int(k) / 9.0))
                if s != ".":
                    drums.put(
                        t0,
                        snare * (int(s) / 9.0),
                        L["snare"],
                        send=ms.SEND["snare"],
                    )
                if h != ".":
                    drums.put(
                        t0,
                        hat * (int(h) / 9.0),
                        L["hat"],
                        pan=0.25,
                        send=ms.SEND["hat"],
                    )
        if bass_name:
            root = ms.note_number(CHORDS[chord][0])
            row = BASS[bass_name][: len(times)]
            slow = bass_name == "long"
            for s, (t0, mark) in enumerate(zip(times, row)):
                gain = ms.lane("bass_gain", t0)
                if mark == "." or gain < 0.02:
                    continue
                later = [k for k in range(s + 1, len(row)) if row[k] != "."]
                length = min(later[0] - s if later else len(row) - s, 2)
                gate = (
                    min(1.9, 4 * n * step - 0.1)
                    if slow
                    else 0.72 * length * step
                )
                vel = (1.0, 0.68, 0.84, 0.68)[s % 4]
                f0 = ms.hz(root + {"0": 0, "+": 12, "b": 1}[mark])
                x = ms.bass_note(
                    f0, gate, vel, ms.lane("bass_bright", t0), slow
                )
                bass.put(t0, x * (gain / bass_ref), L["bass"])
        if arp_name:
            tones = CHORDS[chord][2]
            for s, (t0, index) in enumerate(zip(times, ARP[arp_name])):
                gain = ms.lane("arp_gain", t0)
                if index is None or gain < 0.02:
                    continue
                vel = (1.0, 0.6, 0.8, 0.6)[s % 4] * gain
                x = ms.arp_note(
                    ms.hz(tones[index]), vel, ms.lane("arp_cutoff", t0)
                )
                arp.put(
                    t0,
                    x / arp_ref,
                    L["arp"],
                    pan=(-0.35, 0.35, 0.35, -0.35)[s % 4],
                )
    return kicks


def pads(grid):
    """The pad's chords: the bars with one chord in a row make one."""
    out = []
    for name, _, first, n, row in grid.bars:
        chord = row[1]
        if out and out[-1][2] == chord and out[-1][3] == first:
            out[-1][1], out[-1][3] = grid.time(first + n), first + n
        elif chord:
            out.append(
                [grid.time(first), grid.time(first + n), chord, first + n]
            )
    return [(t0, t1, chord) for t0, t1, chord, _ in out]


def play(grid, film_cues):
    """Play the film's score. Returns the stems with their reverb, and the
    kicks."""
    stems = {name: ms.Stem() for name in ms.STEMS}
    drums, bass, pad, arp, lead, events, fx = (
        stems[name] for name in ms.STEMS
    )
    t = np.arange(ms.NW) / SR

    kicks = play_bars(grid, stems)
    drone = (
        ms.make_drone() * ms.lane("drone_gain", t) * ms.db(LEVEL_DB["drone"])
    )
    bass.dry += ms.stereo(drone)

    for t0, t1, chord in pads(grid):
        random = ms.rng(f"pad at {t0:.3f}")
        x = ms.pad_chord(CHORDS[chord][1], t1 - t0, 0.4, 0.4, random)
        pad.put(t0, x, LEVEL_DB["pad"])
    pad.dry = ms.sweep_lowpass(pad.dry, ms.lane("pad_cutoff", t))
    pad.dry = ms.highpass(pad.dry, 140.0) * ms.lane("pad_gain", t)[:, None]
    mid = pad.dry.mean(axis=1, keepdims=True)
    pad.dry = mid + ms.PAD_WIDTH * (pad.dry - mid)
    pad.dry /= np.sqrt(0.5 + 0.5 * ms.PAD_WIDTH**2)
    pad.send = ms.SEND["pad"] * pad.dry

    arp.dry += ms.echoes(arp.dry, "arp") * ms.lane("echo_gate", t)[:, None]
    arp.send = ms.SEND["arp"] * arp.dry

    lead_ref = np.abs(ms.lead_note(ms.hz("E5"), 0.5)).max()
    for name, bar, s, length, note in LEAD:
        first = [b for b in grid.bars if b[0] == name][bar][2]
        t0, t1 = grid.time(first + s / 4.0), grid.time(
            first + (s + length) / 4.0
        )
        lead.put(
            t0,
            ms.lead_note(ms.hz(note), t1 - t0 - 0.03) / lead_ref,
            LEVEL_DB["lead"],
        )
    lead.dry += ms.echoes(lead.dry, "lead")
    lead.send = ms.SEND["lead"] * lead.dry

    ducks = play_film_events(film_cues, grid, events, drums, fx)
    effects(grid, fx, drums, events)

    for name, depth in ms.SIDECHAIN.items():
        stems[name].scale(ms.duck_curve(kicks, depth))
    for name, depth in ms.EVENT_DUCK.items():
        stems[name].scale(ms.duck_curve(ducks, depth, 0.004, 0.2))

    ir = ms.reverb_ir()
    out = {}
    for name, stem in stems.items():
        out[name] = stem.dry
        if stem.send.any():
            wet = [
                fftconvolve(stem.send[:, c], ir[:, c])[: ms.NW] for c in (0, 1)
            ]
            out[name] = stem.dry + np.stack(wet, axis=1)
    return out, kicks


# ══════════════════════════════════════════════════════════════════════════
# The mix with the voice
# ══════════════════════════════════════════════════════════════════════════


def duck(ev, n, depth):
    """A gain that dips by `depth` while the voice speaks, smoothed."""
    d = np.zeros(n)
    for a, b in ev["lines"].values():
        d[max(int((a - 0.15) * SR), 0) : int((b + 0.3) * SR)] = 1.0
    d = sosfiltfilt(butter(1, 2.5, fs=SR, output="sos"), d)
    return 1.0 - depth * np.clip(d, 0.0, 1.0)


def speech_mask(ev, n):
    m = np.zeros(n, bool)
    for a, b in ev["lines"].values():
        m[int(a * SR) : int(b * SR)] = True
    return m


def write(path, x):
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


# Where the film's moments must fall: the kinds of bar that may hold each
# (a switch may also land on the bar that follows it).
MOMENTS = {
    "poll": {"poll"},
    "search": {"search"},
    "lesson": {"build"},
    "fooled": {"fooled", "break"},
}


def check_form(grid, film_cues):
    """Warnings where a moment of the film falls in a bar of another kind."""
    out = []
    for c in film_cues:
        want = MOMENTS.get(c["kind"])
        if not want:
            continue
        bar = grid.bar_of(c["t"])
        i = grid.bars.index(bar)
        later = grid.bars[min(i + 1, len(grid.bars) - 1)]
        if bar[4][0] not in want and later[4][0] not in want:
            out.append(
                f"warning: {c['kind']} of line {c['shot']} at {c['t']:.2f} s falls in "
                f"a {bar[4][0]} bar ({bar[0]} {bar[1]}), not in a {' or '.join(sorted(want))} bar"
            )
    return out


def check_song(film_cues):
    """Stop before the synthesis when the form names what does not exist."""
    problems = []
    for name, rows in SONG.items():
        for j, (mode, chord, drums, bass, arp, levels) in enumerate(rows):
            where = f"{name} bar {j}"
            if mode not in MODES:
                problems.append(f"{where}: no mode {mode}")
            if chord is not None and chord not in CHORDS:
                problems.append(f"{where}: no chord {chord}")
            if drums and drums not in DRUMS:
                problems.append(f"{where}: no drum pattern {drums}")
            if bass and (bass not in BASS or chord is None):
                problems.append(f"{where}: no bass {bass} over {chord}")
            if arp and (
                arp not in ARP
                or chord not in CHORDS
                or CHORDS[chord][2] is None
            ):
                problems.append(f"{where}: no arpeggio {arp} over {chord}")
            unknown = set(levels) - set(MODES["intro"]) - {"beats"}
            if unknown:
                problems.append(f"{where}: unknown levels {sorted(unknown)}")
    named = sum(c["kind"] == "named" and c["shot"] != "1.4" for c in film_cues)
    if named > len(NAME_NOTES):
        problems.append(
            f"{named} named fields, {len(NAME_NOTES)} notes for them"
        )
    if problems:
        raise SystemExit("the form does not hold:\n  " + "\n  ".join(problems))


def plan(grid, film_cues):
    """Each bar with its time, its tempo, what plays and the events in it."""
    lines = []
    for bar in grid.bars:
        name, j, first, n, (mode, chord, d, b, a, _) = bar
        t0, t1 = grid.time(first), grid.time(first + n)
        bpm = 60.0 * n / (t1 - t0)
        inside = [c for c in film_cues if t0 <= c["t"] < t1]
        kinds = {}
        for c in inside:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        what = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in kinds.items())
        lines.append(
            f"{name:8s} {j:2d}  {t0:7.2f}-{t1:7.2f}  {bpm:5.1f} BPM  {n}/4  "
            f"{mode:7s} {chord or '-':9s} {d or '-':7s} {b or '-':10s} {a or '-':8s} {what}"
        )
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v", type=Path, help="the folder of a voice's media")
    out = ap.parse_args().v
    start = time.time()
    ev = json.loads((out / "events.json").read_text(encoding="utf-8"))
    grid = Grid(ev)
    ms.N = int(round(grid.end * SR))
    ms.NW = ms.N + int(ms.TAIL * SR)
    ms.DURATION = grid.end
    ms.AUTOMATION = lanes(grid)
    ms.LEVEL_DB = LEVEL_DB
    film_cues = cues(ev, grid)
    offsets = np.array([c["t"] - c["pic"] for c in film_cues])
    (out / "score_plan.txt").write_text(
        "\n".join(plan(grid, film_cues)) + "\n", encoding="utf-8"
    )
    tempi = []
    for name, (first, nb, t0, t1) in grid.section.items():
        tempi.append(f"{name} {60.0 * nb / (t1 - t0):.1f}")
    print(
        f"{len(grid.bars)} bars; BPM by section: {', '.join(tempi)}",
        flush=True,
    )
    print(
        f"{len(film_cues)} events sound (of {len(ev['events'])}); on the grid they move by at most "
        f"{1000 * np.abs(offsets).max():.0f} ms, {1000 * np.abs(offsets).mean():.0f} ms on average",
        flush=True,
    )
    for warning in check_form(grid, film_cues):
        print(warning, flush=True)
    check_song(film_cues)

    print("playing ...", flush=True)
    stems, kicks = play(grid, film_cues)
    n = ms.N
    fades = ms.fade_in(n, 0.02) * ms.fade_out(n, FADE_OUT)
    stems = {
        k: ms.lowpass(ms.highpass(x[:n], 25.0), 14000.0) * fades[:, None]
        for k, x in stems.items()
    }
    print(f"  {len(kicks)} kicks; {time.time() - start:.0f} s", flush=True)

    voice, rate = sf.read(str(out / "narration.wav"), dtype="float64")
    assert rate == SR, rate
    voice = np.pad(voice, (0, max(0, n - len(voice))))[:n]
    voice = np.stack([voice, voice], axis=1)
    music = sum(stems[k] for k in MUSIC) * duck(ev, n, DUCK["music"])[:, None]
    other = (stems["events"] + stems["fx"]) * duck(ev, n, DUCK["events"])[
        :, None
    ]
    score = music + other
    lv, ls = ms.loudness(voice), ms.loudness(score)
    gain = ms.db(lv + SCORE_LU - ls)
    score *= gain
    mix = voice + score
    peak = np.abs(mix).max()
    if peak > 0.89:  # -1 dBFS at most; mux.py sets the loudness
        mix *= 0.89 / peak
    write(out / "score.wav", score)
    write(out / "mix.wav", mix)
    (out / "stems").mkdir(exist_ok=True)
    for k, x in stems.items():
        write(out / "stems" / f"{k}.wav", x * gain)

    m = speech_mask(ev, n)
    gaps = ~m & (np.arange(n) / SR > 0.5)
    print(
        f"voice {lv:.1f} LUFS; score {ms.loudness(score):.1f} LUFS over the film "
        f"({ms.loudness(score) - lv:+.1f} LU); under the voice {ms.loudness(score[m], False):.1f}, "
        f"between the lines {ms.loudness(score[gaps], False):.1f} (ungated)",
        flush=True,
    )
    for k in stems:
        x = stems[k] * gain
        print(
            f"  {k:7s} {ms.loudness(x, False):6.1f} LUFS ungated, peak {ms.to_db(np.abs(x).max()):6.1f} dBFS",
            flush=True,
        )
    print(
        f"mix {ms.loudness(mix):.1f} LUFS, peak {ms.to_db(np.abs(mix).max()):.1f} dBFS, {n / SR:.3f} s; "
        f"wrote {out}; {time.time() - start:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
