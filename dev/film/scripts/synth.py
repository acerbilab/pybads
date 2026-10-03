#!/usr/bin/env python3
"""Synthesize the score of the PyBADS style frame.

The style frame is a 40-second test scene of a film about PyBADS, with the
look and the mood of the cyberpunk games of the 1990s. Its score is a
tracker-style piece: a few voices, a pulsing bass, a cold pad, a sparse
lead. It is in E minor, with the flat second of the Phrygian mode as its
colour of tension, at 96 BPM in 4/4: a beat lasts 0.625 s, a bar 2.5 s and
the 16 bars 40 s.

The scene shows the algorithm alternating between two modes, and the score
follows its cue sheet (``CUES``): every evaluation makes a sound, and lands
on a beat of the music, or in bar 0 on a sixteenth note.

- Bar 0 recalls the sixteen evaluations made before the scene: one quiet
  tick per sixteenth note, pitched by the height of the ground on which
  the evaluation lands.
- SEARCH trusts the surrogate. It plays the whole arrangement: drums, the
  pulsing bass, the pad and a fast arpeggio, which stands for the search
  thinking before each try. A try that fails sounds a dull thud with a
  dissonant stab, one that succeeds a bell.
- POLL steps along the axes without the surrogate. It strips the music down
  to a low drone and dry taps, one per step; a step that succeeds adds a
  bell. When the mesh doubles, a note falls one octave, and the taps that
  follow are an octave lower.
- When the surrogate has learned the shape, a build leads to an arrival
  where the harmony lifts from E minor to C major 7 and the lead enters.

Everything is computed here from NumPy and SciPy, with fixed seeds, so the
same interpreter and libraries write the same samples.

The film's score (score.py) imports this module for its instruments, its
levels and its mix, and gives the piece its own form.

    python -u scripts/synth.py OUT            synthesize, write and measure
    python -u scripts/synth.py OUT --analyze  measure and plot the files on disk

It writes into the folder OUT:

    score.wav               stereo, 44100 Hz, 16-bit PCM, 40.000 s
    score.mp3               the same at 160 kbit/s (score.flac if the
                            installed soundfile cannot write MP3; neither
                            without soundfile)
    stems/*.wav             the groups of instruments before the limiter;
                            their sum is the mix that enters it
    score_levels.txt        levels, loudness (also as a small speaker
                            plays the piece), band energies of the whole
                            file and of each bar, the onsets of the events
    score_spectrogram.png   log-frequency spectrogram with the cues marked
    score_onsets.png        the events' waveform around each cue

The two figures need matplotlib and are skipped without it; ``--analyze``
under an interpreter that has it draws them from the files on disk.

The file reads from top to bottom as: the cue sheet, the piece as data
(chords, patterns, automation), the mix parameters, the signal helpers, the
instruments, the sequencer, the master bus, the files, the analysis.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import (
    butter,
    fftconvolve,
    resample_poly,
    sosfilt,
    spectrogram,
)

try:
    import soundfile as sf
except ImportError:  # the WAV files are then written by SciPy
    sf = None

TAU = 2.0 * np.pi

# ══════════════════════════════════════════════════════════════════════════
# 1. The frame and the cue sheet
# ══════════════════════════════════════════════════════════════════════════

SR = 44100
BPM = 96.0
BEAT = 60.0 / BPM  # 0.625 s
STEP = BEAT / 4.0  # a sixteenth note
BAR = 4.0 * BEAT  # 2.5 s
N_BARS = 16
DURATION = N_BARS * BAR  # 40.0 s
N = int(round(DURATION * SR))  # 1 764 000 samples per channel

# The sixteen evaluations made before the scene land one per sixteenth note
# over bar 0, the last one on the downbeat of bar 1. These are the values
# of the objective at them, in order: the lower the ground, the lower the
# tick. The lowest, the fourteenth, is the point that the search found.
LANDING_VALUES = [
    6.80, 6.80, 4.06, 1.63, 7.76, 18.76, 2.85, 11.89,
    0.44, -1.03, 0.09, 7.36, 0.58, -7.16, -0.34, -6.82,
]  # fmt: skip

# The scene's events: (time in seconds, kind, accent in dB). After the
# landings every time is a beat. The accent raises or lowers that one
# event: the misses of a round grow a little louder from the first to the
# fourth, and the first landings rise with the picture, which fades in over
# 0.45 s.
CUES = [
    (1 * STEP, "landing", -4.5),
    (2 * STEP, "landing", -1.5),
    *[(k * STEP, "landing", 0.0) for k in range(3, 17)],
    (7.5, "miss", -1.5),
    (10.625, "miss", -0.5),
    (11.875, "miss", 0.5),
    (13.75, "miss", 1.0),
    (18.75, "poll_hit", 0.0),
    (19.375, "poll_miss", 0.0),
    (20.0, "mesh_doubles", 0.0),
    (22.5, "miss", 0.0),
    (23.125, "miss", 0.3),
    (23.75, "miss", 0.6),
    (24.375, "miss", 1.0),
    (25.625, "poll_hit", 0.0),
    (26.25, "poll_miss", 0.0),
    (26.875, "poll_miss", 0.0),
    (33.75, "big_hit", 0.0),
    (35.0, "miss", -1.5),
    (35.625, "miss", -1.5),
    (36.25, "small_hit", 0.0),
    (36.875, "small_hit", 1.0),
]
MESH_TIME = 20.0  # the poll's taps are an octave lower after it

# The sections of the scene: first bar, last bar, name.
SECTIONS = [
    (0, 1, "establishing"),
    (2, 5, "SEARCH, fooled"),
    (6, 6, "breakdown"),
    (7, 7, "POLL 1"),
    (8, 8, "mesh doubles"),
    (9, 9, "SEARCH, fooled"),
    (10, 10, "POLL 2"),
    (11, 12, "the lesson"),
    (13, 14, "SEARCH, right"),
    (15, 15, "outro"),
]
SEARCH_BARS = (2, 3, 4, 5, 9, 13, 14)
POLL_BARS = (7, 10)

# ══════════════════════════════════════════════════════════════════════════
# 2. The piece
# ══════════════════════════════════════════════════════════════════════════

# Equal temperament from E1 = 41.2 Hz (A4 = 439.96 Hz).
TUNING_E1 = 41.2

# Each chord: the bass's root, the pad's voicing, the tones that the
# arpeggio cycles. E minor is home. F over E and F major 7 (#11) carry the
# flat second, the colour of the search being fooled; the pad voices them
# without a low F, which would grind against the bass's E. B7sus4 is the
# build, and C major 7, G and Dsus2 are the lift that returns to E minor.
CHORDS = {
    "Em": ("E2", ["E3", "B3", "E4", "G4", "B4"], ["E4", "G4", "B4", "E5"]),
    "F/E": (
        "E2",
        ["A3", "C4", "F4", "A4"],
        ["F4", "A4", "C5", "E5", "F5", "A5"],
    ),
    "Fmaj7#11": (
        "F2",
        ["A3", "C4", "E4", "B4"],
        ["F4", "A4", "B4", "E5", "F5", "A5"],
    ),
    "B7sus4": ("B1", ["F#3", "B3", "E4", "F#4", "A4"], None),
    "Cmaj7": ("C2", ["G3", "C4", "E4", "G4", "B4"], None),
    "G": ("G2", ["G3", "D4", "G4", "B4"], ["G4", "B4", "D5", "G5"]),
    "Dsus2": ("D2", ["A3", "D4", "E4", "A4"], ["A4", "D5", "E5", "A5"]),
}

# One row per bar: its chord, and the pattern that the drums, the bass and
# the arpeggio play in it (None: silent). The pad and the lead have their
# own lists below. In bar 8 only the pickup of its last beat sounds the
# chord.
# fmt: off
SONG = [
    # chord       drums     bass          arpeggio
    ("Em",        None,     None,         None),      # 0   establishing
    ("Em",        "pulse",  "hint",       "pickup"),  # 1
    ("Em",        "A",      "eighths",    "roll"),    # 2   SEARCH, fooled
    ("Em",        "B",      "eighths",    "roll"),    # 3
    ("F/E",       "A16",    "lean",       "climb"),   # 4
    ("Fmaj7#11",  "B16",    "sixteenths", "climb"),   # 5
    ("Em",        "break",  "sixteenths", "roll"),    # 6   breakdown
    (None,        None,     None,         None),      # 7   POLL 1
    ("F/E",       "pickup", None,         "pickup"),  # 8   the mesh doubles
    ("F/E",       "A16",    "sixteenths", "climb"),   # 9   SEARCH, fooled
    (None,        None,     None,         None),      # 10  POLL 2
    ("B7sus4",    "build",  "build",      None),      # 11  the lesson: build
    ("Cmaj7",     "half",   "soft",       None),      # 12  the lesson: arrival
    ("G",         "A16",    "sixteenths", "roll"),    # 13  SEARCH, right
    ("Dsus2",     "B16",    "sixteenths", "roll"),    # 14
    ("Em",        "out",    "long",       "out"),     # 15  outro
]
# fmt: on

# Drum patterns, one character per sixteenth: a digit is a hit and its
# strength (9 is full), a dot a rest. Kick, snare, closed hat.
DRUMS = {
    "pulse": ("4.......4.......", "................", "..2.2.2.3.3.3.4."),
    "A": ("9.......9..5....", "....9.......9...", "4.7.4.7.4.7.4.7."),
    "B": ("9.......9.....6.", "....9.......9...", "4.7.4.7.4.7.4.74"),
    "A16": ("9.......9..5....", "....9.......9...", "4273427342734273"),
    "B16": ("9.......9.....6.", "....9.......9...", "4273427342734274"),
    "break": ("9...............", "................", "4.6.3.4.2......."),
    "pickup": ("................", "..............35", "............2.3."),
    "build": ("6...7...8...9...", "........3.4.5567", "................"),
    "half": ("9.........5.....", "........8.....24", "..5...5...5...5."),
    "out": ("8...............", "................", "4.3.2..........."),
}

# Bass patterns, one character per sixteenth: 0 is the chord's root, + the
# octave above it, b the semitone above it (the flat second), a dot a rest.
BASS = {
    "hint": "........0.0.0.0.",
    "eighths": "0.0.0.+.0.0.0.+.",
    "sixteenths": "000+00+0000+00+0",
    "lean": "000+00+0000+00bb",
    "build": "0.0.0.0.00000000",
    "soft": "0.0.0.0.0.0.0.+.",
    "long": "0...............",
}

# Arpeggio patterns: for each sixteenth, an index into the chord's tones.
ARP = {
    "roll": [0, 2, 1, 2, 3, 2, 1, 2, 0, 2, 1, 2, 3, 2, 1, 2],
    "climb": [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 3, 1],
    "pickup": [None] * 12 + [0, 1, 2, 3],
    "out": [0, 2, 1, 2, 3, 2] + [None] * 10,
}

# The pad: start and length in bars, chord, attack and release in seconds.
PAD = [
    (0.0, 4.0, "Em", 0.5, 0.4),
    (4.0, 1.0, "F/E", 0.4, 0.4),
    (5.0, 1.0, "Fmaj7#11", 0.4, 0.4),
    (6.0, 1.0, "Em", 0.4, 0.3),
    (8.75, 1.25, "F/E", 0.05, 0.05),
    (11.0, 1.0, "B7sus4", 0.3, 0.12),
    (12.0, 1.0, "Cmaj7", 0.04, 0.4),
    (13.0, 1.0, "G", 0.4, 0.4),
    (14.0, 1.0, "Dsus2", 0.4, 0.4),
    (15.0, 1.0, "Em", 0.4, 0.1),
]

# The lead: bar, sixteenth, length in sixteenths, note. All its notes are in
# the minor pentatonic scale of E.
LEAD = [
    (12, 0, 6, "E5"),
    (12, 6, 2, "D5"),
    (12, 8, 7, "B4"),
    (13, 0, 4, "G4"),
    (13, 4, 2, "A4"),
    (13, 6, 2, "B4"),
    (13, 8, 7, "D5"),
    (14, 8, 4, "A4"),
    (14, 12, 4, "B4"),
    (15, 0, 8, "E5"),
]

# The notes of the events. The range of the landings' values is cut into as
# many equal steps as LANDING_SCALE has notes, the lowest ground on the
# first: with these values the two deepest points sound the tonic.
LANDING_SCALE = ("E4", "G4", "A4", "B4", "D5", "E5", "G5", "A5")
TAP_NOTE = "E5"  # the pitch of a poll tap before the mesh doubles
MISS_STAB = ("F3", "B3")  # the flat second and the note a tritone above it
POLL_HIT_NOTE = "B5"  # the fifth; an octave lower after the mesh doubles
BIG_HIT_NOTES = ("G5", "D6")  # root and fifth of the chord of bar 13
BIG_HIT_LOW = "G4"
SMALL_HIT_NOTES = ("D6", "E6")  # rising to the tonic
MESH_FROM = "E5"  # the note that falls one octave

# Automation: (time in seconds, value), linear between the points and flat
# outside them. A lane named "..._cutoff" is in Hz and interpolated in
# log-frequency.
# fmt: off
AUTOMATION = {
    # The pad's level, and the cutoff of the low-pass on it. The pad stands
    # for the surrogate's map: a faint bed under the landings, it swells as
    # the map fades in, from 2.5 to 3.6 s; its filter closes over the
    # breakdown, where the map is switched off, and opens over the build.
    "pad_gain": [
        (0, 0), (0.45, 0.28), (2.5, 0.3), (3.6, 0.8), (5.0, 0.9),
        (10.0, 1.0), (15.0, 1.0), (17.3, 0),
        (21.875, 0), (22.5, 0.9), (25.0, 1.0), (25.04, 0),
        (27.5, 0), (27.51, 0.35), (30.0, 1.0),
        (37.5, 1.0), (38.6, 0.75), (40.0, 0),
    ],
    "pad_cutoff": [
        (0, 350), (2.5, 450), (3.6, 1500), (5.0, 1850),
        (10.0, 2000), (15.0, 2400), (17.3, 160),
        (21.875, 300), (22.5, 2300), (25.0, 2400),
        (27.5, 500), (29.8, 2600), (30.0, 3000),
        (37.5, 2800), (40.0, 500),
    ],
    # The arpeggio grows louder and brighter while the search is fooled.
    "arp_gain": [
        (4.375, 0.3), (5.0, 0.9), (10.0, 0.9), (12.5, 1.05), (15.0, 1.25),
        (15.001, 1.0), (16.9, 0),
        (21.875, 0.3), (22.5, 1.1), (25.0, 1.25),
        (32.5, 1.0), (37.5, 1.0), (37.501, 0.8), (38.44, 0.15),
    ],
    "arp_cutoff": [
        (4.375, 500), (5.0, 2900), (10.0, 2900), (15.0, 3900),
        (15.001, 2900), (17.0, 250),
        (21.875, 450), (22.5, 3400), (25.0, 3900),
        (32.5, 3400), (37.5, 3400), (38.44, 600),
    ],
    # The bass's level, and how far its filter opens on each note.
    "bass_gain": [
        (3.75, 0.35), (5.0, 0.9), (5.001, 1.0), (15.0, 1.0), (16.9, 0),
        (22.499, 0), (22.5, 1.0), (25.0, 1.0),
        (27.5, 0.55), (30.0, 1.0), (30.001, 0.9), (32.5, 0.9),
        (32.501, 1.0), (37.5, 1.0), (37.501, 0.9),
    ],
    "bass_bright": [
        (3.75, 0.15), (5.0, 0.6), (5.001, 0.85), (10.0, 0.85), (15.0, 1.25),
        (15.001, 1.0), (16.9, 0.1),
        (22.499, 0.1), (22.5, 1.2), (25.0, 1.3),
        (27.5, 0.5), (30.0, 1.3), (30.001, 0.75), (32.5, 0.75),
        (32.501, 1.05), (37.5, 1.05), (37.501, 0.7),
    ],
    # The drone sounds where the bass does not: at the start, under the
    # poll and at the end.
    "drone_gain": [
        (0, 0), (0.45, 0.7), (1.5, 1.0), (4.4, 1.0), (5.1, 0),
        (15.6, 0), (17.5, 1.0), (22.4, 1.0), (22.6, 0),
        (24.98, 0), (25.06, 1.0), (27.5, 1.0), (28.1, 0),
        (37.4, 0), (37.8, 1.0), (38.8, 0.8), (40.0, 0),
    ],
    # The arpeggio's echoes are cut where the search hands over to the
    # poll, so that the poll starts dry.
    "echo_gate": [
        (17.5, 1), (17.75, 0), (21.8, 0), (21.875, 1),
        (25.0, 1), (25.2, 0), (27.4, 0), (27.5, 1),
    ],
}
# fmt: on

# Soft noise crashes at the starts of the groove and at the arrival, and
# reverse swells into the returns of the groove: (time, dB).
CRASHES = [(5.0, 0.0), (22.5, -2.0), (30.0, 0.0), (32.5, -3.0), (37.5, -5.0)]
SWELLS = [(5.0, 0.0), (22.5, 0.0), (32.5, -4.0)]

# ══════════════════════════════════════════════════════════════════════════
# 3. The mix: what to adjust after listening
# ══════════════════════════════════════════════════════════════════════════

# The mix is dark, and balanced to carry on a small speaker, which plays
# little below 200 Hz: the kick, the bass and the drone each have harmonics
# between 250 Hz and 1 kHz, above their weight in the sub-bass. The report
# gives the loudness of the piece through such a speaker (SMALL_SPEAKER)
# beside that of the full band.

# Levels in dB before the master gain, which then sets the loudness of the
# whole: only their differences matter. For the pad and the drone the level
# is that of the RMS, for every other sound that of its peak (for the bass,
# the arpeggio and the lead, the peak of a full-strength note).
LEVEL_DB = {
    "kick": -8.0,
    "snare": -6.5,
    "hat": -18.0,
    "crash": -24.0,
    "bass": -9.5,
    "drone": -27.0,
    "pad": -22.5,
    "arp": -10.0,
    "lead": -16.0,
    "landing": -23.0,  # a landing tick of bar 0
    "landing_deep": -19.0,  # the one on the point that the search found
    "miss_thud": -11.0,  # the low thud of a search miss
    "miss_stab": -8.5,  # its dissonant stab
    "tap": -12.0,  # a poll tap; the tap of a poll miss is 2 dB lower
    "bell": -21.0,  # the bell of a poll hit
    "big_hit": -13.5,  # the loudest bell of the search's big hit
    "small_hit": -15.5,  # the bell of a small hit
    "mesh": -18.0,  # the note that falls when the mesh doubles
    "riser": -17.0,  # the noise that rises over the build
    "downlifter": -31.0,  # the noise that falls over the breakdown
    "powerdown": -22.0,  # the tone that falls into the drone there
    "swell": -29.0,  # the reverse swells
}
# How much of each sound goes to the reverb: the level of its reverberation
# against the sound itself, for a sustained sound in the reverb's band.
SEND = {
    "snare": 0.28,
    "hat": 0.08,
    "crash": 0.35,
    "pad": 0.30,
    "arp": 0.20,
    "lead": 0.32,
    "landing": 0.20,
    "miss": 0.15,
    "tap": 0.05,
    "bell": 0.40,
    "mesh": 0.25,
    "fx": 0.35,
}
# The reverb: the decay time of its lowest band and its pre-delay, in
# seconds.
REVERB = {"rt60": 1.5, "predelay": 0.015}
# Echoes in time with the music. Each train: the delay in sixteenths, the
# feedback from one echo to the next, the level of the first echo, and the
# positions (-1 left, 1 right) that the echoes take in turn. The arpeggio
# and the bells bounce from side to side on dotted eighths; the lead has a
# dotted eighth on the left and a quarter on the right.
ECHO = {
    "arp": [(3, 0.35, 0.28, (0.7, -0.7))],
    "lead": [(3, 0.42, 0.27, (-0.7,)), (4, 0.42, 0.27, (0.7,))],
    "bell": [(3, 0.40, 0.20, (-0.7, 0.7))],
}
# The pad's two channels are made of different detuned saws. The width is
# how much of their difference is kept: 1 is all of it, 0 is mono.
PAD_WIDTH = 0.7
# The dip of each stem under a kick and under an event (the landings
# excepted), as a fraction.
SIDECHAIN = {"pad": 0.35, "bass": 0.30, "arp": 0.20}
EVENT_DUCK = {"pad": 0.25, "arp": 0.30}
MASTER = {
    "lufs": -16.0,  # integrated loudness
    "ceiling_db": -2.3,  # of the sample peak; the true peak is a little above
    "lowcut": 25.0,
    "highcut": 14000.0,
    "fade_in": 0.02,
    "fade_out": 0.8,  # the picture fades to black from 39.2 s
}
SEED = 1993

STEMS = ("drums", "bass", "pad", "arp", "lead", "events", "fx")
TAIL = 4.0  # seconds rendered past the end, for the echoes and the reverb
NW = N + int(TAIL * SR)

# ══════════════════════════════════════════════════════════════════════════
# 4. Signal helpers
# ══════════════════════════════════════════════════════════════════════════


def db(x):
    """A level in dB as a gain."""
    return 10.0 ** (np.asarray(x, float) / 20.0)


def to_db(x):
    """A gain as a level in dB."""
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-10))


def power_db(p):
    """A power as a level in dB."""
    return 10.0 * np.log10(np.maximum(p, 1e-12))


def rng(name):
    """The random generator of one sound, whatever is made before it."""
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def note_number(name):
    """The MIDI number of a note such as "E2", "F#3" or "Bb3" (C4 = 60)."""
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    pitch, octave = name[:-1], int(name[-1])
    shift = pitch.count("#") - pitch.count("b")
    return 12 * (octave + 1) + semitones[pitch[0]] + shift


def hz(note):
    """The frequency of a note name or a MIDI number."""
    m = note_number(note) if isinstance(note, str) else note
    return TUNING_E1 * 2.0 ** ((m - 28) / 12.0)


def detuned(f, cents):
    """The frequency f raised by a number of cents."""
    return f * 2.0 ** (cents / 1200.0)


def lane(name, t):
    """The value of an automation lane at the times t."""
    points = np.asarray(AUTOMATION[name], float)
    if name.endswith("cutoff"):
        return np.exp(np.interp(t, points[:, 0], np.log(points[:, 1])))
    return np.interp(t, points[:, 0], points[:, 1])


def lowpass(x, cutoff, order=2):
    sos = butter(order, cutoff, "low", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def highpass(x, cutoff, order=2):
    sos = butter(order, cutoff, "high", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def bandpass(x, low, high, order=2):
    sos = butter(order, [low, high], "band", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def unit(x):
    """x scaled to a peak of 1."""
    return x / np.abs(x).max()


def fade_in(n, seconds):
    w = np.ones(n)
    k = min(n, int(seconds * SR))
    w[:k] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, k)))
    return w


def fade_out(n, seconds):
    return fade_in(n, seconds)[::-1]


def stereo(mono, pan=0.0):
    """A mono signal placed between left (-1) and right (1), at constant
    power; in the centre each channel carries the signal unchanged."""
    a = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4.0
    channels = [mono * np.cos(a), mono * np.sin(a)]
    return np.stack(channels, axis=1) * np.sqrt(2.0)


def envelope(n, attack, decay, sustain, gate, release):
    """A linear attack, an exponential decay towards the sustain level, and
    from the gate time a raised-cosine release."""
    t = np.arange(n) / SR
    decayed = np.exp(-np.maximum(t - attack, 0.0) / decay)
    env = sustain + (1.0 - sustain) * decayed
    env *= np.minimum(t / attack, 1.0)
    released = np.clip((t - gate) / release, 0.0, 1.0)
    env *= 0.5 * (1.0 + np.cos(np.pi * released))
    return env


def partials(theta, weights):
    """The sum over k of weights[k - 1] * sin(k * theta), by the recurrence
    sin((k + 1) x) = 2 cos(x) sin(k x) - sin((k - 1) x). The weights are one
    number per harmonic, or one row of samples per harmonic."""
    twice_cos = 2.0 * np.cos(theta)
    previous = np.zeros_like(theta)
    current = np.sin(theta)
    out = weights[0] * current
    for w in weights[1:]:
        previous, current = current, twice_cos * current - previous
        if np.any(w):
            out += w * current
    return out


def lowpass_gain(f, cutoff, q=0.707):
    """The gain at the frequencies f of a four-pole low-pass: a two-pole
    section of quality q, whose resonance it is, and a two-pole Butterworth
    section."""
    r2 = (f / cutoff) ** 2
    return 1.0 / np.sqrt(((1.0 - r2) ** 2 + r2 / q**2) * (1.0 + r2 * r2))


def spectrum_of(shape, k):
    """The amplitudes of the harmonics k of a wave of peak 1, of the shape
    "saw", "square" or "tri". A pulse is made of two saws."""
    odd = k % 2 == 1
    if shape == "square":
        return np.where(odd, (4.0 / np.pi) / k, 0.0)
    if shape == "tri":
        sign = (-1.0) ** ((k - 1) // 2)
        return np.where(odd, (8.0 / np.pi**2) * sign / k**2, 0.0)
    return (2.0 / np.pi) / k


def osc(freq, n, shape="saw", cutoff=None, q=0.707, duty=0.25, phase=0.0):
    """n samples of an oscillator through a low-pass filter.

    The wave, of the shape "saw", "square", "tri" or "pulse" (the last with
    its duty cycle), is the sum of its harmonics, each scaled by the
    filter's gain at its frequency. So the pitch and the cutoff, numbers or
    arrays of n samples, can move freely, and no harmonic is made above
    0.45 SR: nothing aliases.
    """
    freq = np.asarray(freq, float)
    f = np.broadcast_to(freq, (n,))
    theta = phase + 2.0 * np.pi * (np.cumsum(f) - f) / SR
    limit = 0.45 * SR
    top = limit if cutoff is None else min(limit, 4.0 * float(np.max(cutoff)))
    k = np.arange(1, max(1, int(top / f.min())) + 1)
    amp = spectrum_of(shape, k)
    shift = 2.0 * np.pi * duty

    if freq.ndim == 0 and np.ndim(cutoff) == 0:  # nothing moves
        w = amp if cutoff is None else amp * lowpass_gain(k * f[0], cutoff, q)
        w = np.where(k * f[0] <= limit, w, 0.0)
        out = partials(theta, w)
        return out - partials(theta + shift, w) if shape == "pulse" else out

    if cutoff is not None:
        cutoff = np.broadcast_to(np.asarray(cutoff, float), (n,))
    out = np.empty(n)
    for a in range(0, n, 8192):
        b = min(a + 8192, n)
        fk = k[:, None] * f[None, a:b]
        w = amp[:, None]
        if cutoff is not None:
            w = w * lowpass_gain(fk, cutoff[None, a:b], q)
        w = np.where(fk <= limit, w, 0.0)
        out[a:b] = partials(theta[a:b], w)
        if shape == "pulse":
            out[a:b] -= partials(theta[a:b] + shift, w)
    return out


def sweep_lowpass(x, cutoff):
    """x through a low-pass whose cutoff (one value per sample) moves: a
    crossfade between fixed filters a quarter of an octave apart."""
    grid = np.geomspace(60.0, 16000.0, 33)
    cutoff = np.clip(cutoff, grid[0], grid[-1])
    position = np.interp(np.log(cutoff), np.log(grid), np.arange(len(grid)))
    low = np.minimum(position.astype(int), len(grid) - 2)
    frac = position - low
    out = np.zeros_like(x)
    for i, f in enumerate(grid):
        w = np.where(low == i, 1.0 - frac, 0.0)
        w = w + np.where(low + 1 == i, frac, 0.0)
        if w.any():
            out += lowpass(x, f) * (w[:, None] if x.ndim == 2 else w)
    return out


def at_rate(x, rate):
    """The sample x played at another speed, as a tracker plays one sample
    at another note: at rate 0.5 it is an octave lower and twice as long."""
    ratio = Fraction(rate).limit_denominator(64)
    if ratio == 1:
        return x
    return resample_poly(x, ratio.denominator, ratio.numerator, axis=0)


def echoes(x, name, repeats=6, cutoff=2400.0):
    """The echoes of the stereo signal x, by the trains of ECHO[name]: each
    echo is darker and softer than the one before."""
    out = np.zeros_like(x)
    for steps, feedback, level, pans in ECHO[name]:
        echo = x.mean(axis=1)
        d = int(round(steps * STEP * SR))
        for k in range(1, repeats + 1):
            if k * d >= len(x):
                break
            echo = lowpass(echo, cutoff, 1) * (feedback if k > 1 else 1.0)
            pan = pans[(k - 1) % len(pans)]
            out[k * d :] += stereo(echo[: len(x) - k * d], pan) * level
    return out


def duck_curve(hits, depth, attack=0.006, release=0.14):
    """A gain that dips by the fraction `depth` at each hit, (time,
    strength), and recovers."""
    n = int((4.0 * attack + 6.0 * release) * SR)
    t = np.arange(n) / SR
    shape = unit((1.0 - np.exp(-t / attack)) * np.exp(-t / release))
    dip = np.zeros(NW)
    for t0, strength in hits:
        i = int(round(t0 * SR))
        j = min(i + n, NW)
        dip[i:j] = np.maximum(dip[i:j], depth * strength * shape[: j - i])
    return 1.0 - dip


def reverb_ir():
    """The impulse response of a dark room: decaying noise between 220 Hz
    and 3.8 kHz whose higher bands are softer and die away sooner. Its power
    gain, averaged from 300 Hz to 3 kHz, is 1."""
    rt60 = REVERB["rt60"]
    n = int(1.4 * rt60 * SR)
    t = np.arange(n) / SR
    noise = rng("reverb").standard_normal((n, 2))
    ir = np.zeros((n, 2))
    # low edge, high edge, decay time as a fraction of rt60, level
    for low, high, k, level in (
        (220.0, 600.0, 1.0, 1.0),
        (600.0, 1700.0, 0.8, 0.9),
        (1700.0, 3800.0, 0.5, 0.7),
    ):
        decay = np.exp(-6.91 * t / (k * rt60))
        band = bandpass(noise, low, high, 3) * decay[:, None]
        ir += band * (level / np.sqrt(k))
    ir *= (1.0 - np.exp(-t / 0.006))[:, None]
    size = 2**18
    f = np.fft.rfftfreq(size, 1.0 / SR)
    power = np.abs(np.fft.rfft(ir, n=size, axis=0)) ** 2
    ir /= np.sqrt(power[(f >= 300.0) & (f <= 3000.0)].mean(axis=0))
    return np.vstack([np.zeros((int(REVERB["predelay"] * SR), 2)), ir])


class Stem:
    """A stereo track and what it sends to the reverb."""

    def __init__(self):
        self.dry = np.zeros((NW, 2))
        self.send = np.zeros((NW, 2))

    def put(self, t0, x, level_db=0.0, pan=0.0, send=0.0):
        """Add the sound x, mono or stereo, with its first sample at t0."""
        i = int(round(t0 * SR))
        if x.ndim == 1:
            x = stereo(x, pan)
        j = min(i + len(x), NW)
        x = x[: j - i] * db(level_db)
        self.dry[i:j] += x
        if send:
            self.send[i:j] += send * x

    def scale(self, gain):
        self.dry *= gain[:, None]
        self.send *= gain[:, None]


# ══════════════════════════════════════════════════════════════════════════
# 5. The instruments
# ══════════════════════════════════════════════════════════════════════════
# The drums and the events are made once, like a tracker's samples, and
# played many times.


def make_kick():
    """A sine whose pitch drops from 287 Hz, fast and then slowly, to 47 Hz,
    with a little of its second and third harmonics and a click, saturated.
    The knock of the fast drop, the harmonics and the click are what a
    small speaker plays of it."""
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    # The pitch is 47 Hz plus two decaying terms, of 150 Hz over 10 ms and
    # of 90 Hz over 40 ms; the phase holds the integral of each.
    swept = 150.0 * 0.010 * (1.0 - np.exp(-t / 0.010))
    swept += 90.0 * 0.040 * (1.0 - np.exp(-t / 0.040))
    phase = TAU * (47.0 * t + swept)
    body = np.sin(phase) * np.exp(-t / 0.15)
    body += 0.45 * np.sin(2.0 * phase) * np.exp(-t / 0.07)
    body += 0.25 * np.sin(3.0 * phase) * np.exp(-t / 0.05)
    noise = lowpass(rng("kick").standard_normal(n), 3500.0)
    click = unit(noise) * np.exp(-t / 0.004)
    x = np.tanh(3.0 * (body + 0.6 * click))
    return unit(x * fade_out(n, 0.05))


def make_snare():
    """A soft snare: band-passed noise over a short tone near 190 Hz, and
    under them a narrower band of noise, between 1.2 and 3.2 kHz, with a
    longer tail, like a clap."""
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    white = rng("snare").standard_normal((n, 2))
    noise = unit(bandpass(white[:, 0], 800.0, 6500.0))
    clap = unit(bandpass(white[:, 1], 1200.0, 3200.0))
    f = 185.0 + 70.0 * np.exp(-t / 0.015)
    tone = np.sin(2.0 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.045)
    x = 0.9 * noise * np.exp(-t / 0.09) + 0.6 * tone
    x += 0.45 * clap * np.exp(-t / 0.12)
    x = x * np.minimum(t / 0.0008, 1.0) * fade_out(n, 0.05)
    return unit(lowpass(x, 7500.0))


def make_hat():
    """A closed hat: a tick of noise between 5.5 and 9 kHz."""
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    noise = highpass(rng("hat").standard_normal(n), 5500.0, 4)
    noise = lowpass(noise, 9000.0, 4)
    return unit(noise * np.exp(-t / 0.014) * fade_out(n, 0.02))


def make_crash():
    """A soft, dark crash: a long burst of noise between 2.5 and 8.5 kHz."""
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    noise = bandpass(rng("crash").standard_normal((n, 2)), 2500.0, 8500.0, 3)
    env = np.minimum(t / 0.004, 1.0) * np.exp(-t / 0.45) * fade_out(n, 0.3)
    return unit(noise * env[:, None])


def make_swell(seconds):
    """Noise that rises into the instant at which it stops."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    noise = bandpass(rng("swell").standard_normal((n, 2)), 800.0, 6000.0)
    env = np.exp((t - seconds) / 0.2) * fade_out(n, 0.006)
    return unit(noise * env[:, None])


def make_riser(seconds):
    """Noise under a low-pass that opens from 300 Hz to 6 kHz, growing."""
    n = int(seconds * SR)
    u = np.arange(n) / n
    noise = highpass(rng("riser").standard_normal((n, 2)), 300.0)
    x = lowpass(sweep_lowpass(noise, 300.0 * (6000.0 / 300.0) ** u), 9000.0)
    return unit(x * (u**2 * fade_out(n, 0.01))[:, None])


def make_downlifter(seconds):
    """Noise under a low-pass that closes from 3.5 kHz to 250 Hz, fading."""
    n = int(seconds * SR)
    u = np.arange(n) / n
    noise = highpass(rng("downlifter").standard_normal((n, 2)), 200.0)
    x = sweep_lowpass(noise, 3500.0 * (250.0 / 3500.0) ** u)
    env = np.minimum(u * seconds / 0.3, 1.0) * (1.0 - u) ** 1.5
    return unit(x * env[:, None])


def make_powerdown(seconds):
    """A dark saw that falls two octaves, from E3 to the drone's E1."""
    n = int(seconds * SR)
    u = np.arange(n) / n
    f = hz("E3") * 2.0 ** (-2.0 * u * u * (3.0 - 2.0 * u))
    x = osc(f, n, "saw", 500.0 * (200.0 / 500.0) ** u, q=1.0)
    return unit(x * fade_in(n, 0.3) * fade_out(n, 0.25))


# The drone's partials: multiple of E1, level, detuning in Hz. E1 and E2
# are the sub-bass; E3, B3, E4, B4 and E5 above them, octaves and fifths of
# the same harmonic series, are what a small speaker plays of the drone.
# A second E2 and a second E4 beat against the first, once and twice per
# bar.
DRONE = [
    (1, 1.00, 0.0), (2, 0.40, 0.0), (2, 0.20, 1.0 / BAR),
    (4, 0.25, 0.0), (6, 0.18, 0.0), (8, 0.16, 0.0), (8, 0.06, 2.0 / BAR),
    (12, 0.08, 0.0), (16, 0.04, 0.0),
]  # fmt: skip


def make_drone():
    """The drone, at RMS 1: the partials of DRONE, with a soft swell on
    every beat."""
    t = np.arange(NW) / SR
    f = hz("E1")
    x = np.zeros(NW)
    for multiple, level, detuning in DRONE:
        x += level * np.sin(TAU * (multiple * f + detuning) * t)
    x *= lowpass(0.72 + 0.28 * np.exp(-np.mod(t, BEAT) / 0.22), 12.0)
    return x / np.sqrt(np.mean(x**2))


def bass_note(f0, gate, vel, bright, slow=False):
    """A pluck of the bass: a saw and a square, and a saw an octave above
    them, under a low-pass that closes after the attack to a cutoff of 300
    to 700 Hz, the higher the brighter. The octave and the harmonics that
    the filter leaves between 250 Hz and 1 kHz are what a small speaker
    plays of it. `slow` is the long last note."""
    n = int((gate + 0.03) * SR)
    t = np.arange(n) / SR
    sweep = 2200.0 * bright * (0.5 + 0.5 * vel)
    rest = 300.0 + 400.0 * min(bright, 1.0)
    cutoff = rest + sweep * np.exp(-t / (0.6 if slow else 0.09))
    saw = osc(f0, n, "saw", cutoff, q=1.3)
    square = osc(f0, n, "square", cutoff, q=1.3)
    octave = osc(2.0 * f0, n, "saw", 1.6 * cutoff)
    x = 0.7 * saw + 0.45 * square + 0.7 * octave
    if slow:
        return vel * x * envelope(n, 0.004, 0.6, 0.0, gate, 0.03)
    return vel * x * envelope(n, 0.004, 0.10, 0.5, gate, 0.03)


def arp_note(f0, vel, cutoff):
    """A pluck of the arpeggio: a thin pulse wave under a low-pass."""
    gate = 0.105
    n = int((gate + 0.03) * SR)
    t = np.arange(n) / SR
    fc = cutoff * (1.0 + 0.9 * vel * np.exp(-t / 0.025))
    x = osc(f0, n, "pulse", fc, q=1.0, duty=0.25)
    return vel * x * envelope(n, 0.002, 0.06, 0.2, gate, 0.03)


def lead_note(f0, gate):
    """A note of the lead: a pulse wave and a triangle, with a vibrato that
    comes in after the attack."""
    n = int((gate + 0.2) * SR)
    t = np.arange(n) / SR
    depth = 9.0 * np.clip((t - 0.18) / 0.35, 0.0, 1.0)  # cents
    f = detuned(f0, depth * np.sin(TAU * 5.2 * t))
    fc = 2600.0 + 1800.0 * np.exp(-t / 0.25)
    x = 0.45 * osc(f, n, "tri", fc) + 0.55 * osc(f, n, "pulse", fc, duty=0.3)
    return x * envelope(n, 0.02, 0.4, 0.75, gate, 0.2)


PAD_FILTER = 3200.0  # the pad's own low-pass; the automation closes it more


def pad_chord(notes, seconds, attack, release, random):
    """A chord of the pad, at RMS 1: per note two detuned saws on the left
    and two others on the right, low-passed."""
    n = int((seconds + release) * SR)
    out = np.zeros((n, 2))
    for name in notes:
        m = note_number(name)
        weight = 1.0 / (1.0 + (m - 52) / 20.0)  # the low notes a bit louder
        for channel, cents in ((0, -9.0), (0, 5.0), (1, -5.0), (1, 9.0)):
            phase = random.uniform(0.0, 2.0 * np.pi)
            saw = osc(detuned(hz(m), cents), n, "saw", PAD_FILTER, phase=phase)
            out[:, channel] += weight * saw
    out /= np.sqrt(np.mean(out**2))
    env = fade_in(n, attack)
    k = int(release * SR)
    env[n - k :] *= fade_out(k, release)
    return out * env[:, None]


def landing_tick(f0, deep):
    """A landing: a short glassy ping, a sine with two soft overtones. The
    deep one rings longer and brighter, over an octave below."""
    n = int((0.45 if deep else 0.085) * SR)
    t = np.arange(n) / SR
    tau = 0.085 if deep else 0.02

    def partial(multiple, level, decay):
        tone = np.sin(TAU * multiple * f0 * t)
        return level * tone * np.exp(-t / (decay * tau))

    x = partial(1.0, 1.0, 1.0)
    x += partial(2.0, 0.5 if deep else 0.3, 0.6)
    x += partial(4.0, 0.25 if deep else 0.12, 0.3)
    if deep:
        x += partial(0.5, 0.5, 1.0)
    return unit(x * np.minimum(t / 0.001, 1.0) * fade_out(n, 0.5 * tau))


def make_miss():
    """The two parts of a search miss, each at peak 1: a muted low thud,
    and a short dull stab on two notes a tritone apart."""
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    f = 66.0 + 75.0 * np.exp(-t / 0.022)
    thud = np.sin(2.0 * np.pi * (np.cumsum(f) - f) / SR) * np.exp(-t / 0.07)
    rumble = unit(lowpass(rng("miss").standard_normal(n), 350.0))
    thud += 0.5 * rumble * np.exp(-t / 0.025)
    thud = lowpass(np.tanh(1.4 * thud), 700.0) * fade_out(n, 0.05)

    cutoff = 450.0 + 900.0 * np.exp(-t / 0.06)
    stab = np.zeros(n)
    for i, name in enumerate(MISS_STAB):
        # The two saws of a note start in phase with each other, so that its
        # fundamental is whole at the attack; the notes start a quarter of a
        # cycle apart, so that their edges do not pile up.
        for cents in (-7.0, 7.0):
            f = detuned(hz(name), cents)
            stab += osc(f, n, "saw", cutoff, q=0.9, phase=0.5 * np.pi * i)
    stab *= envelope(n, 0.006, 0.075, 0.0, 0.26, 0.12)
    stab = np.tanh(1.3 * unit(stab))  # rounds its first peaks
    return unit(thud), unit(stab)


def make_tap():
    """A dry woodblock-like tap: two short partials and a click of noise."""
    n = int(0.10 * SR)
    t = np.arange(n) / SR
    f = hz(TAP_NOTE)
    body = np.sin(TAU * f * t) * np.exp(-t / 0.016)
    body += 0.45 * np.sin(TAU * 2.42 * f * t) * np.exp(-t / 0.006)
    click = unit(bandpass(rng("tap").standard_normal(n), 1800.0, 5200.0))
    x = body + 0.5 * click * np.exp(-t / 0.0012)
    return unit(x * fade_out(n, 0.02))


def bell(f0, tau, index):
    """A bell, stereo, at peak about 1: a sine modulated in frequency at 3.5
    times its own, brightest at the attack. `tau` is its decay time and
    `index` its brightness."""
    n = int(min(7.0 * tau, 3.5) * SR)
    t = np.arange(n) / SR
    depth = index * np.exp(-t / 0.16)
    out = np.empty((n, 2))
    for channel, cents in enumerate((-2.5, 2.5)):
        f = detuned(f0, cents)
        out[:, channel] = np.sin(
            TAU * f * t + depth * np.sin(TAU * 3.5 * f * t)
        )
    env = (1.0 - np.exp(-t / 0.0015)) * np.exp(-t / tau) * fade_out(n, 0.05)
    return lowpass(out * env[:, None], 7000.0)


def make_mesh_drop():
    """The note of the mesh doubling: it holds for a moment, glides down one
    octave, and its two channels drift apart in pitch as it lands, which
    widens it."""
    n = int(2.4 * SR)
    t = np.arange(n) / SR
    u = np.clip((t - 0.06) / 0.26, 0.0, 1.0)
    u = u * u * (3.0 - 2.0 * u)
    out = np.empty((n, 2))
    for channel, cents in enumerate((-4.0, 4.0)):
        f = hz(MESH_FROM) * 2.0 ** (-u + cents * u / 1200.0)
        saw = osc(f, n, "saw", 1400.0)
        out[:, channel] = osc(f, n, "tri", 2600.0) + 0.25 * saw
    env = np.minimum(t / 0.004, 1.0) * np.exp(-t / 0.6) * fade_out(n, 0.3)
    return unit(out * env[:, None])


# ══════════════════════════════════════════════════════════════════════════
# 6. The sequencer
# ══════════════════════════════════════════════════════════════════════════


def play_patterns(stems):
    """The drums, the bass and the arpeggio, bar by bar, from SONG. Returns
    the kicks as (time, strength), for the sidechain."""
    kick, snare, hat = make_kick(), make_snare(), make_hat()
    drums, bass, arp = stems["drums"], stems["bass"], stems["arp"]
    # The levels of the bass and of the arpeggio refer to these notes.
    bass_ref = np.abs(bass_note(hz("E2"), 0.72 * STEP, 1.0, 1.0)).max()
    arp_ref = np.abs(arp_note(hz("E4"), 1.0, 1500.0)).max()
    kicks = []

    for bar, (chord, drum_name, bass_name, arp_name) in enumerate(SONG):
        times = bar * BAR + np.arange(16) * STEP
        if drum_name:
            rows = DRUMS[drum_name]
            for t0, k, s, h in zip(times, *rows):
                if k != ".":
                    drums.put(t0, kick * (int(k) / 9.0), LEVEL_DB["kick"])
                    kicks.append((t0, int(k) / 9.0))
                if s != ".":
                    x = snare * (int(s) / 9.0)
                    drums.put(t0, x, LEVEL_DB["snare"], send=SEND["snare"])
                if h != ".":
                    x, send = hat * (int(h) / 9.0), SEND["hat"]
                    drums.put(t0, x, LEVEL_DB["hat"], pan=0.25, send=send)

        if bass_name:
            root = note_number(CHORDS[chord][0])
            row = BASS[bass_name]
            slow = bass_name == "long"
            for s, (t0, mark) in enumerate(zip(times, row)):
                gain = lane("bass_gain", t0)
                if mark == "." or gain < 0.02:
                    continue
                # A note lasts until the next one or the end of the bar,
                # two sixteenths at most.
                later = [k for k in range(s + 1, 16) if row[k] != "."]
                length = min(later[0] - s if later else 16 - s, 2)
                gate = 1.9 if slow else 0.72 * length * STEP
                vel = (1.0, 0.68, 0.84, 0.68)[s % 4]
                f0 = hz(root + {"0": 0, "+": 12, "b": 1}[mark])
                x = bass_note(f0, gate, vel, lane("bass_bright", t0), slow)
                bass.put(t0, x * (gain / bass_ref), LEVEL_DB["bass"])

        if arp_name:
            tones = CHORDS[chord][2]
            for s, (t0, index) in enumerate(zip(times, ARP[arp_name])):
                gain = lane("arp_gain", t0)
                if index is None or gain < 0.02:
                    continue
                vel = (1.0, 0.6, 0.8, 0.6)[s % 4] * gain
                x = arp_note(hz(tones[index]), vel, lane("arp_cutoff", t0))
                pan = (-0.35, 0.35, 0.35, -0.35)[s % 4]
                arp.put(t0, x / arp_ref, LEVEL_DB["arp"], pan=pan)
    return kicks


def play_events(events, drums):
    """The sounds of the cues. Returns the events alone, without echo or
    reverb, for the check of their onsets."""
    crash = make_crash()
    thud, stab = make_miss()
    tap = make_tap()
    tap_dull = unit(lowpass(tap, 1100.0))
    bells = Stem()  # the bells go through their own echo
    small_hits = iter(SMALL_HIT_NOTES)
    # Each landing has its height between the lowest value (0) and the
    # highest (1), which picks its note, and a place a little to the left or
    # to the right.
    values = np.array(LANDING_VALUES)
    heights = (values - values.min()) / (values.max() - values.min())
    pans = rng("landing").uniform(-0.4, 0.4, len(values))
    landings = iter(zip(heights, values == values.min(), pans))
    top = len(LANDING_SCALE) - 1

    for t0, kind, accent in CUES:
        rate = 1.0 if t0 < MESH_TIME else 0.5  # the speed of the poll's taps
        if kind == "landing":
            height, deep, pan = next(landings)
            step = min(int(height * (top + 1)), top)
            x = landing_tick(hz(LANDING_SCALE[step]), deep)
            level_db = LEVEL_DB["landing_deep" if deep else "landing"] + accent
            send = SEND["landing"] * (2.0 if deep else 1.0)
            events.put(t0, x, level_db, pan=pan, send=send)
        elif kind == "miss":
            send = SEND["miss"]
            events.put(t0, thud, LEVEL_DB["miss_thud"] + accent, send=send)
            events.put(t0, stab, LEVEL_DB["miss_stab"] + accent, send=send)
        elif kind == "poll_miss":
            x = at_rate(tap_dull, rate)
            events.put(t0, x, LEVEL_DB["tap"] - 2.0 + accent, send=SEND["tap"])
        elif kind == "poll_hit":
            x = at_rate(tap, rate)
            events.put(t0, x, LEVEL_DB["tap"] + accent, send=SEND["tap"])
            tau = 0.40 if rate == 1.0 else 0.45
            x = bell(hz(POLL_HIT_NOTE) * rate, tau, 0.8)
            bells.put(t0, x, LEVEL_DB["bell"] + accent, send=SEND["bell"])
        elif kind == "mesh_doubles":
            level_db = LEVEL_DB["tap"] - 1.0 + accent
            events.put(t0, tap, level_db, send=SEND["tap"])
            x = make_mesh_drop()
            events.put(t0, x, LEVEL_DB["mesh"] + accent, send=SEND["mesh"])
        elif kind == "big_hit":
            # Two long bright bells over a softer one an octave below, and a
            # crash, which goes with the drums: the stem of the events holds
            # no noise.
            level_db = LEVEL_DB["big_hit"] + accent
            for name, lower in zip(BIG_HIT_NOTES, (0.0, 2.0)):
                x = bell(hz(name), 0.95, 0.9)
                bells.put(t0, x, level_db - lower, send=SEND["bell"])
            x = bell(hz(BIG_HIT_LOW), 0.8, 0.5)
            bells.put(t0, x, level_db - 6.0, send=SEND["bell"])
            level_db = LEVEL_DB["crash"] - 3.0 + accent
            drums.put(t0, crash, level_db, send=SEND["crash"])
        elif kind == "small_hit":
            x = bell(hz(next(small_hits)), 0.32, 0.7)
            level_db = LEVEL_DB["small_hit"] + accent
            bells.put(t0, x, level_db, send=SEND["bell"])
        else:
            raise ValueError(f"unknown kind of cue: {kind}")

    events.dry += bells.dry
    events.send += bells.send
    alone = events.dry[:N].copy()
    bell_echoes = echoes(bells.dry, "bell")
    events.dry += bell_echoes
    events.send += SEND["bell"] * bell_echoes
    return alone


def play():
    """Play the song. Returns the stems before the master bus, the events
    alone without echo or reverb, and the kicks as (time, strength)."""
    stems = {name: Stem() for name in STEMS}
    drums, bass, pad, arp, lead, events, fx = (stems[name] for name in STEMS)
    t = np.arange(NW) / SR

    kicks = play_patterns(stems)
    crash = make_crash()
    for t0, level_db in CRASHES:
        level_db = LEVEL_DB["crash"] + level_db
        drums.put(t0, crash, level_db, send=SEND["crash"])

    # The drone shares the bass's stem.
    drone = make_drone() * lane("drone_gain", t) * db(LEVEL_DB["drone"])
    bass.dry += stereo(drone)

    for start, length, chord, attack, release in PAD:
        # Each chord draws its saws' phases from a generator of its own, so
        # that a change to one chord leaves the others as they were.
        random = rng(f"pad at bar {start:g}")
        notes = CHORDS[chord][1]
        x = pad_chord(notes, length * BAR, attack, release, random)
        pad.put(start * BAR, x, LEVEL_DB["pad"])
    pad.dry = sweep_lowpass(pad.dry, lane("pad_cutoff", t))
    pad.dry = highpass(pad.dry, 140.0) * lane("pad_gain", t)[:, None]
    # Narrowed to PAD_WIDTH, at the level that its unrelated channels had.
    mid = pad.dry.mean(axis=1, keepdims=True)
    pad.dry = mid + PAD_WIDTH * (pad.dry - mid)
    pad.dry /= np.sqrt(0.5 + 0.5 * PAD_WIDTH**2)
    pad.send = SEND["pad"] * pad.dry

    arp.dry += echoes(arp.dry, "arp") * lane("echo_gate", t)[:, None]
    arp.send = SEND["arp"] * arp.dry

    lead_ref = np.abs(lead_note(hz("E5"), 0.5)).max()
    for bar, s, length, name in LEAD:
        x = lead_note(hz(name), length * STEP - 0.03) / lead_ref
        lead.put(bar * BAR + s * STEP, x, LEVEL_DB["lead"])
    lead.dry += echoes(lead.dry, "lead")
    lead.send = SEND["lead"] * lead.dry

    events_alone = play_events(events, drums)

    # The breakdown falls, the build rises, and a swell leads into each
    # return of the groove.
    send = SEND["fx"]
    fx.put(15.0, make_downlifter(2.2), LEVEL_DB["downlifter"], send=send)
    fx.put(16.25, make_powerdown(1.35), LEVEL_DB["powerdown"])
    fx.put(27.5, make_riser(BAR), LEVEL_DB["riser"], send=send)
    swell = make_swell(BEAT)
    for t0, level_db in SWELLS:
        fx.put(t0 - BEAT, swell, LEVEL_DB["swell"] + level_db, send=send)

    for name, depth in SIDECHAIN.items():
        stems[name].scale(duck_curve(kicks, depth))
    cue_hits = [(t0, 1.0) for t0, kind, _ in CUES if kind != "landing"]
    for name, depth in EVENT_DUCK.items():
        stems[name].scale(duck_curve(cue_hits, depth, 0.004, 0.2))

    ir = reverb_ir()
    out = {}
    for name, stem in stems.items():
        out[name] = stem.dry
        if stem.send.any():
            wet = [fftconvolve(stem.send[:, c], ir[:, c])[:NW] for c in (0, 1)]
            out[name] = stem.dry + np.stack(wet, axis=1)
    return out, events_alone, kicks


# ══════════════════════════════════════════════════════════════════════════
# 7. The master bus
# ══════════════════════════════════════════════════════════════════════════


def k_weight(x):
    """x through the K-weighting of ITU-R BS.1770: a high shelf and a
    high-pass, their coefficients computed for this sample rate."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / SR)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / SR)
    a0 = 1.0 + k / q + k * k
    a1, a2 = 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0
    high = [1.0, -2.0, 1.0, 1.0, a1, a2]
    return sosfilt(np.array([shelf, high]), x, axis=0)


def loudness(x, gated=True):
    """The loudness of the stereo signal x in LUFS, after ITU-R BS.1770:
    K-weighted, and with `gated` integrated over the 400 ms blocks that pass
    the standard's two gates."""
    y = k_weight(x)
    if not gated:
        return -0.691 + 10.0 * np.log10(np.mean(y**2, axis=0).sum() + 1e-12)
    block, hop = int(0.4 * SR), int(0.1 * SR)
    total = np.concatenate([np.zeros((1, 2)), np.cumsum(y**2, axis=0)])
    starts = np.arange(0, len(y) - block + 1, hop)
    power = ((total[starts + block] - total[starts]) / block).sum(axis=1)
    level = -0.691 + 10.0 * np.log10(power + 1e-12)
    keep = level > -70.0
    relative = -0.691 + 10.0 * np.log10(power[keep].mean()) - 10.0
    keep &= level > relative
    return -0.691 + 10.0 * np.log10(power[keep].mean())


def limiter(x, ceiling, lookahead=0.003, release=80.0):
    """x with its gain lowered around every peak so that no sample passes
    the ceiling. The gain starts to fall `lookahead` seconds before a peak
    and recovers at `release` dB per second. Returns the limited signal and
    the gain in dB."""
    peak = np.abs(x).max(axis=1)
    need = np.minimum(0.0, 20.0 * np.log10(ceiling / np.maximum(peak, 1e-9)))
    width = 2 * int(lookahead * SR) + 1
    held = minimum_filter1d(need, width, mode="nearest")
    ramp = (release / SR) * np.arange(len(x))
    released = np.minimum.accumulate(held - ramp) + ramp
    gain_db = uniform_filter1d(released, width, mode="nearest")
    return x * db(gain_db)[:, None], gain_db


def premaster(x):
    """What the master bus does to a stem or to the mix before the gain and
    the limiter: the band limits and the fades at the two ends."""
    x = lowpass(highpass(x[:N], MASTER["lowcut"]), MASTER["highcut"])
    fades = fade_in(N, MASTER["fade_in"]) * fade_out(N, MASTER["fade_out"])
    return x * fades[:, None]


def master(mix):
    """The mix at the target loudness under the ceiling. Returns it with
    the gain applied before the limiter and the limiter's gain in dB."""
    ceiling = db(MASTER["ceiling_db"])
    gain = db(MASTER["lufs"] - loudness(mix))
    for _ in range(8):
        out, reduction = limiter(mix * gain, ceiling)
        error = MASTER["lufs"] - loudness(out)
        if abs(error) < 0.01:
            break
        gain *= db(error)
    return out, float(gain), reduction


# ══════════════════════════════════════════════════════════════════════════
# 8. Files
# ══════════════════════════════════════════════════════════════════════════


def to_pcm16(x):
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def write_wav(path, x):
    """Write x as 16-bit PCM."""
    pcm = to_pcm16(x)
    if sf is not None:
        sf.write(str(path), pcm, SR, subtype="PCM_16")
    else:
        wavfile.write(str(path), SR, pcm)
    return pcm


def read_wav(path):
    """The samples of a 16-bit WAV file, as floats, and its sample rate."""
    rate, pcm = wavfile.read(str(path))
    return pcm.astype(np.float64) / 32768.0, rate


def write_compressed(out_dir, pcm):
    """The mix as MP3 at 160 kbit/s, or as FLAC if soundfile cannot write
    MP3. Returns the path, or None without soundfile."""
    if sf is None:
        return None
    data = pcm.astype(np.float32) / 32768.0
    if "MP3" in sf.available_formats():
        path = out_dir / "score.mp3"
        # libsndfile maps the compression level 0..1 onto 320..32 kbit/s.
        options = {
            "format": "MP3",
            "subtype": "MPEG_LAYER_III",
            "compression_level": (320.0 - 160.0) / (320.0 - 32.0) - 1e-4,
            "bitrate_mode": "CONSTANT",
        }
        try:
            sf.write(str(path), data, SR, **options)
            return path
        except Exception as error:  # an older soundfile or libsndfile
            print(f"  MP3 not written ({error}); writing FLAC", flush=True)
            path.unlink(missing_ok=True)
    path = out_dir / "score.flac"
    sf.write(str(path), pcm, SR, format="FLAC", subtype="PCM_16")
    return path


# ══════════════════════════════════════════════════════════════════════════
# 9. Analysis
# ══════════════════════════════════════════════════════════════════════════

# The bands of the report: name, span as written, edges in Hz.
BANDS = [
    ("sub", "< 60 Hz", 0.0, 60.0),
    ("bass", "60-250 Hz", 60.0, 250.0),
    ("low-mid", "250-1000 Hz", 250.0, 1000.0),
    ("mid", "1-4 kHz", 1000.0, 4000.0),
    ("high", "4-10 kHz", 4000.0, 10000.0),
    ("air", "> 10 kHz", 10000.0, SR / 2.0 + 1.0),
]
# A small speaker, for the report: it plays the band between a fourth-order
# high-pass and a fourth-order low-pass at these frequencies.
SMALL_SPEAKER = (200.0, 8000.0)
# The band in which each kind of event is compared with the music.
EVENT_BANDS = {
    "landing": (250.0, 6000.0),
    "miss": (120.0, 900.0),
    "poll_hit": (250.0, 6000.0),
    "poll_miss": (250.0, 6000.0),
    "mesh_doubles": (150.0, 3000.0),
    "big_hit": (500.0, 6000.0),
    "small_hit": (500.0, 6000.0),
}
# The names of the kinds of event in the figures.
EVENT_NAMES = {
    "landing": "landing",
    "miss": "search miss",
    "big_hit": "search hit",
    "small_hit": "search hit",
    "poll_hit": "poll tap, hit",
    "poll_miss": "poll tap, miss",
    "mesh_doubles": "mesh doubles",
}


def band_powers(x):
    """The power of x in each band of BANDS, as the mean of the channels;
    the powers add up to the mean square of x."""
    spectrum = np.abs(np.fft.rfft(x, axis=0)) ** 2
    spectrum[1:] *= 2.0
    if len(x) % 2 == 0:
        spectrum[-1] /= 2.0
    power = spectrum.mean(axis=1) / len(x) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    bands = [(f >= low) & (f < high) for _, _, low, high in BANDS]
    return np.array([power[band].sum() for band in bands])


def true_peak_db(x):
    """The peak of each channel after 4x oversampling, in dBFS."""
    return to_db(np.abs(resample_poly(x, 4, 1, axis=0)).max(axis=0))


def small_speaker(x):
    """x as a small speaker plays it: the band of SMALL_SPEAKER."""
    low, high = SMALL_SPEAKER
    return lowpass(highpass(x, low, 4), high, 4)


def bar_loudness(x):
    """The loudness of each bar of the stereo signal x: K-weighted, not
    gated, in LUFS."""
    bars = x.reshape(N_BARS, N // N_BARS, 2)
    return np.array([loudness(seg, gated=False) for seg in bars])


def find_onset(x, i, order=16):
    """The sample near index i at which a new sound starts in the stereo
    signal x, where an earlier sound may still ring. A linear predictor is
    fitted to the 15 ms that end 5 ms before i; the onset is the first
    sample, from there to 10 ms after i, that it fails to predict. None if
    it predicts them all."""
    mono = x.mean(axis=1)
    a, b, c = i - int(0.02 * SR), i - int(0.005 * SR), i + int(0.01 * SR)
    seg = mono[a - order : c]
    lags = [seg[order - k - 1 : len(seg) - k - 1] for k in range(order)]
    past, now = np.stack(lags, axis=1), seg[order:]
    coef = np.linalg.lstsq(past[: b - a], now[: b - a], rcond=None)[0]
    error = np.abs(now - past @ coef)
    floor = max(8.0 * error[: b - a].max(), 4.0 / 32768.0)
    late = np.nonzero(error[b - a :] > floor)[0]
    return b + int(late[0]) if len(late) else None


def section_of(bar):
    return next(name for a, b, name in SECTIONS if a <= bar <= b)


def analyze(out_dir):
    """Measure score.wav and the stems on disk; write score_levels.txt and,
    with matplotlib, the two figures. Returns the lines of the report and
    the checks that failed."""
    path = out_dir / "score.wav"
    with wave.open(str(path), "rb") as w:
        shape = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        frames = w.getnframes()
    assert shape == (2, 2, SR) and frames == N, (shape, frames)
    x, _ = read_wav(path)
    pcm = np.round(x * 32768.0).astype(int)
    names = [*STEMS, "events_dry"]
    paths = {name: out_dir / "stems" / f"{name}.wav" for name in names}
    stems = {k: read_wav(p)[0] for k, p in paths.items() if p.exists()}
    events_dry = stems.pop("events_dry", None)
    out, failed = [], []
    add = out.append

    def pair(label, values, form="7.2f"):
        left, right = (format(v, form) for v in values)
        add(f"  {label:36s} {left}   {right}")

    def single(label, value, unit_, form="7.2f"):
        add(f"  {label:36s} {format(value, form)} {unit_}".rstrip())

    def check(ok, text):
        add(f"  {'ok    ' if ok else 'FAILED'}  {text}")
        if not ok:
            failed.append(text)

    peak = to_db(np.abs(x).max(axis=0))
    true_peak = true_peak_db(x)
    rms = power_db(np.mean(x**2))
    lufs = loudness(x)
    first, last = (tuple(int(v) for v in pcm[i]) for i in (0, -1))
    add(f"score.wav   2 channels, {SR} Hz, 16-bit PCM, {N} frames")
    add(f"  = {N / SR:.6f} s; first sample (L, R) {first}, last {last}")
    add("")
    add(f"{'Levels':38s}    left     right")
    pair("sample peak, dBFS", peak)
    pair("true peak (4x oversampled), dBFS", true_peak)
    pair("RMS, dBFS", power_db(np.mean(x**2, axis=0)))
    pair("DC offset, fraction of full scale", x.mean(axis=0), "+.1e")
    pair("samples at full scale", (np.abs(pcm) >= 32767).sum(axis=0), "7d")
    add("")
    single("RMS of both channels", rms, "dBFS")
    single("crest factor (peak / RMS)", peak.max() - rms, "dB")
    single("integrated loudness", lufs, "LUFS")
    single("true peak", true_peak.max(), "dBFS")
    single("L/R correlation", np.corrcoef(x.T)[0, 1], "", "7.3f")
    add("  The loudness is K-weighted and gated as in ITU-R BS.1770, by this")
    add("  script's own meter.")
    add("")

    whole = band_powers(x)
    add("Band energies of the whole file: RMS level in dBFS (the mean of the")
    add("two channels) and share of the power")
    for (name, span, _, _), p in zip(BANDS, whole):
        share = 100.0 * p / whole.sum()
        add(f"  {name:8s} {span:12s} {power_db(p):7.2f}   {share:6.2f} %")
    add("")

    add("Per bar of 2.5 s: RMS and band levels in dBFS, loudness (K-weighted,")
    add("ungated) in LUFS, the level above 250 Hz, and the L/R correlation")
    head = "".join(f"{name:>9s}" for name, *_ in BANDS)
    add(f"  bar  start  section           RMS    LUFS {head}   >250Hz   corr")
    bars = x.reshape(N_BARS, N // N_BARS, 2)
    per_bar = np.array([band_powers(seg) for seg in bars])
    above = per_bar[:, 2:].sum(axis=1)
    full = bar_loudness(x)
    for bar, seg in enumerate(bars):
        cells = "".join(f"{power_db(p):9.1f}" for p in per_bar[bar])
        add(
            f"  {bar:3d} {bar * BAR:6.1f}  {section_of(bar):15s} "
            f"{power_db(np.mean(seg**2)):6.1f}  {full[bar]:6.1f} {cells} "
            f"{power_db(above[bar]):8.1f} {np.corrcoef(seg.T)[0, 1]:6.2f}"
        )
    add("")

    # The same piece on a small speaker.
    low, high = SMALL_SPEAKER
    played = small_speaker(x)
    small_lufs, small = loudness(played), bar_loudness(played)
    add("On a small speaker, taken as a fourth-order high-pass at")
    add(f"{low:.0f} Hz and a fourth-order low-pass at {high:.0f} Hz: the")
    add("loudness in LUFS (integrated for the whole file, not gated for a")
    add("bar), and how far it is below that of the full band")
    add("                                  full band   small speaker   below")
    add(
        f"  whole file, integrated          {lufs:9.2f} {small_lufs:15.2f} "
        f"{lufs - small_lufs:7.2f}"
    )
    for bar, (wide, narrow) in enumerate(zip(full, small)):
        add(
            f"  {bar:3d} {bar * BAR:6.1f}  {section_of(bar):15s}    "
            f"{wide:9.1f} {narrow:15.1f} {wide - narrow:7.1f}"
        )
    add("")

    # How sparse the POLL bars are: the energy of the bar above 250 Hz, and
    # its typical level there, the median over the bar's windows of 50 ms.
    high = sosfilt(butter(4, 250.0, "high", fs=SR, output="sos"), x, axis=0)
    windows = (high**2).reshape(N_BARS, -1, int(0.05 * SR), 2)
    typical = np.median(power_db(windows.mean(axis=(2, 3))), axis=1)
    search = list(SEARCH_BARS)
    mean_db, typical_db = (
        power_db(above[search].mean()),
        typical[search].mean(),
    )
    add("Checks")
    add(f"  Above 250 Hz the SEARCH bars {SEARCH_BARS} have a mean")
    add(f"  level of {mean_db:.1f} dBFS and a typical level, the median over")
    add(f"  windows of 50 ms, of {typical_db:.1f} dBFS.")
    for bar in POLL_BARS:
        level, drop = power_db(above[bar]), mean_db - power_db(above[bar])
        check(
            drop > 10.0,
            f"POLL bar {bar}: {level:.1f} dBFS above 250 Hz, "
            f"{drop:.1f} dB below the SEARCH bars",
        )
        level, drop = typical[bar], typical_db - typical[bar]
        check(
            drop > 15.0,
            f"POLL bar {bar}: typically {level:.1f} dBFS above 250 Hz, "
            f"{drop:.1f} dB below the SEARCH bars",
        )
    # On the small speaker the piece must lose little, the POLL bars must
    # stay well below the SEARCH bars, and the drone, which the bass's stem
    # holds alone in the POLL bars, must be heard there, quietly: 15 to 28
    # LU below the SEARCH bars.
    drop = lufs - small_lufs
    text = f"on a small speaker the file is {drop:.1f} LU quieter, 5 at most"
    check(drop <= 5.0, text)
    groove = power_db(np.mean(10.0 ** (small[search] / 10.0)))
    if "bass" in stems:
        drone = bar_loudness(small_speaker(stems["bass"]))
    for bar in POLL_BARS:
        drop = groove - small[bar]
        check(
            drop > 8.0,
            f"POLL bar {bar} on a small speaker: {small[bar]:.1f} LUFS, "
            f"{drop:.1f} LU below the SEARCH bars",
        )
        if "bass" in stems:
            drop = groove - drone[bar]
            check(
                15.0 <= drop <= 28.0,
                f"the drone of POLL bar {bar} on a small speaker: "
                f"{drone[bar]:.1f} LUFS, {drop:.1f} LU below the SEARCH bars",
            )
    check(not pcm[[0, -1]].any(), "the file starts and ends on a zero sample")
    check(np.abs(pcm).max() < 32767, "no sample at full scale")
    check(true_peak.max() < -1.5, "the true peak is below -1.5 dBFS")
    check(np.abs(x.mean(axis=0)).max() < 1e-4, "no DC offset")
    add("")

    if events_dry is not None and len(stems) == len(STEMS):
        report_events(stems, events_dry, add, check)
    text = "\n".join(out) + "\n"
    (out_dir / "score_levels.txt").write_text(text, encoding="utf-8")
    draw_figures(out_dir, x, stems, events_dry, lufs, true_peak.max())
    return out, failed


def report_events(stems, events_dry, add, check):
    """The events in the report: where each starts, and how loud it is
    against the music."""
    events = stems["events"]
    music = sum(stems.values()) - events
    add("Events: the onset found in the stem of the events alone")
    add("(stems/events_dry.wav), and the level of the event against the rest")
    add("of the mix over its first 250 ms, in the band given")
    add(
        "    cue (s)  kind           onset (s)  offset (ms)"
        "  event - music (dB)  band (Hz)"
    )
    filtered, worst = {}, 0.0
    for t0, kind, _ in CUES:
        i = int(round(t0 * SR))
        onset = find_onset(events_dry, i)
        if onset is None:
            check(False, f"no onset found at the cue of {t0} s")
            continue
        offset = 1000.0 * (onset - i) / SR
        worst = max(worst, abs(offset))
        low, high = EVENT_BANDS[kind]
        if (low, high) not in filtered:
            sos = butter(4, [low, high], "band", fs=SR, output="sos")
            pair = (sosfilt(sos, events, axis=0), sosfilt(sos, music, axis=0))
            filtered[(low, high)] = pair
        j = i + int(0.25 * SR)
        band = filtered[(low, high)]
        ev, mu = (power_db(np.mean(s[i:j] ** 2)) for s in band)
        add(
            f"  {t0:9.3f}  {kind:13s} {onset / SR:10.4f}  {offset:+11.3f}"
            f"  {ev - mu:+18.1f}  {low:.0f}-{high:.0f}"
        )
    text = f"the farthest onset is {worst:.3f} ms from its cue, less than 1 ms"
    check(worst < 1.0, text)
    add("")


def log_spectrogram(mono, nfft=4096, hop=1024, bands=360):
    """The spectrogram of a mono signal on log-spaced bands from 20 Hz to
    20 kHz, in dB below a full-scale sine. Returns the edges of the time
    frames, the edges of the bands and the levels (bands by frames)."""
    padding = np.zeros(nfft // 2)
    f, t, power = spectrogram(
        np.concatenate([padding, mono, padding]),
        fs=SR,
        window="hann",
        nperseg=nfft,
        noverlap=nfft - hop,
        scaling="spectrum",
        detrend=False,
    )
    level = 10.0 * np.log10(np.maximum(power / 0.5, 1e-14))
    edges = np.geomspace(20.0, 20000.0, bands + 1)
    low, high = np.searchsorted(f, edges[:-1]), np.searchsorted(f, edges[1:])
    image = np.empty((bands, len(t)))
    for i, centre in enumerate(np.sqrt(edges[:-1] * edges[1:])):
        if high[i] > low[i]:  # the strongest of the bins in the band
            image[i] = level[low[i] : high[i]].max(axis=0)
        else:  # no bin in the band: between the two around it
            j = np.searchsorted(f, centre)
            w = (centre - f[j - 1]) / (f[j] - f[j - 1])
            image[i] = (1.0 - w) * level[j - 1] + w * level[j]
    t = t - (nfft // 2) / SR  # the centres of the frames
    return np.append(t - hop / 2 / SR, t[-1] + hop / 2 / SR), edges, image


def draw_figures(out_dir, x, stems, events_dry, lufs, peak):
    """The spectrogram with the cues, and the events around their cues."""
    try:
        import matplotlib
    except ImportError:
        print("  no matplotlib, no figures: run --analyze where it is")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # A dark surface. One hue for the level, the lighter the louder; two
    # for the two lines of the level panel.
    surface, grid, axis_ink = "#1a1a19", "#2c2c2a", "#383835"
    ink, ink2, muted = "#ffffff", "#c3c2b7", "#898781"
    blue, orange, stripe = "#3987e5", "#d95926", "#232322"
    centred = {"ha": "center", "va": "center"}
    small = {"color": muted, "fontsize": 8}
    faint = {"color": ink, "fontsize": 8, "alpha": 0.75}
    ramp = (
        "#1a1a19 #0d366b #104281 #184f95 #1c5cab #256abf #2a78d6 "
        "#3987e5 #5598e7 #6da7ec #86b6ef #9ec5f4 #b7d3f6 #cde2fb"
    ).split()
    plt.rcParams.update(
        {
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "text.color": ink2,
            "axes.labelcolor": ink2,
            "axes.edgecolor": axis_ink,
            "axes.facecolor": surface,
            "axes.linewidth": 0.8,
            "figure.facecolor": surface,
            "savefig.facecolor": surface,
            "xtick.color": muted,
            "ytick.color": muted,
            "xtick.labelcolor": ink2,
            "ytick.labelcolor": ink2,
        }
    )

    fig = plt.figure(figsize=(16, 10.4), dpi=125)
    layout = fig.add_gridspec(
        3, 2, height_ratios=[1.5, 5.2, 1.9], width_ratios=[60, 1]
    )
    layout.update(hspace=0.07, wspace=0.02)
    ax_cue = fig.add_subplot(layout[0, 0])
    ax = fig.add_subplot(layout[1, 0], sharex=ax_cue)
    ax_level = fig.add_subplot(layout[2, 0], sharex=ax_cue)

    # The cue lane: the sections and the bars, and one row per kind of
    # event.
    rows = list(dict.fromkeys(EVENT_NAMES.values()))
    bottom = len(rows) + 0.75
    ax_cue.set_ylim(len(rows) + 1.3, -1.25)
    for bar in range(N_BARS):
        if bar % 2:
            ax_cue.axvspan(bar * BAR, (bar + 1) * BAR, color=stripe, lw=0)
        ax_cue.text((bar + 0.5) * BAR, bottom, str(bar), **centred, **small)
    for first, last, name in SECTIONS:
        if first == last and len(name) > 10:
            name = name.replace(" ", "\n", 1)
        middle = (first + last + 1) * BAR / 2
        ax_cue.text(middle, -0.35, name, **centred, color=ink)
    for t0, kind, _ in CUES:
        y = rows.index(EVENT_NAMES[kind]) + 1
        if kind == "landing":  # sixteen within one bar: a rug
            ax_cue.plot(t0, y, "|", ms=9, color=ink2, mew=1.3)
        else:
            ax_cue.plot(t0, y, "o", ms=6.5, color=ink2, mec=surface, mew=1.2)
    ax_cue.set_yticks(range(1, len(rows) + 1), rows)
    ax_cue.tick_params(axis="y", length=0)
    ax_cue.tick_params(axis="x", labelbottom=False, length=0)
    ax_cue.text(-0.4, bottom, "bar", ha="right", va="center", **small)
    for y in range(1, len(rows) + 1):
        ax_cue.axhline(y, color=grid, lw=0.8, zorder=0)

    times, edges, image = log_spectrogram(x.mean(axis=1))
    cmap = LinearSegmentedColormap.from_list("level", ramp)
    shown = {"cmap": cmap, "vmin": -100.0, "vmax": -15.0, "rasterized": True}
    mesh = ax.pcolormesh(times, edges, image, **shown)
    ax.set_yscale("log")
    ax.set_ylim(20.0, 20000.0)
    ticks = [20, 60, 100, 250, 500, 1000, 2000, 4000, 10000, 20000]
    labels = [f"{v // 1000} k" if v >= 1000 else str(v) for v in ticks]
    ax.set_yticks(ticks, labels)
    ax.minorticks_off()
    ax.set_ylabel("frequency, Hz (log)")
    ax.tick_params(axis="x", labelbottom=False)
    for name, _, low, high in BANDS:
        if low > 0:
            ax.axhline(low, color=ink, lw=0.5, alpha=0.22)
        y = np.sqrt(max(low, 20.0) * min(high, 20000.0))
        ax.text(DURATION - 0.12, y, name, ha="right", va="center", **faint)
    bar = fig.colorbar(mesh, cax=fig.add_subplot(layout[1, 1]))
    bar.set_label("level, dB re a full-scale sine")
    bar.outline.set_edgecolor(axis_ink)

    # Levels over 50 ms: the mix and, if it is there, the stem of the events.
    def short_level(y):
        n = int(0.05 * SR)
        power = (y**2).reshape(len(y) // n, n, 2).mean(axis=(1, 2))
        return (np.arange(len(power)) + 0.5) * n / SR, power_db(power)

    ax_level.plot(*short_level(x), color=blue, lw=1.3, label="the mix")
    if "events" in stems:
        t, level = short_level(stems["events"])
        label = "the events alone (before the limiter)"
        ax_level.plot(t, level, color=orange, lw=1.3, label=label)
    ax_level.set_ylim(-80.0, 0.0)
    ax_level.set_yticks([-80, -60, -40, -20, 0])
    ax_level.grid(axis="y", color=grid, lw=0.8)
    ax_level.set_axisbelow(True)
    ax_level.set_ylabel("RMS over 50 ms, dBFS")
    ax_level.set_xlabel("time, s (one bar is 2.5 s)")
    ax_level.legend(loc="lower left", ncols=2, frameon=False, labelcolor=ink2)
    ax_level.set_xticks(np.arange(0.0, DURATION + 0.1, BAR))
    ax_level.set_xlim(0.0, DURATION)

    for a in (ax_cue, ax, ax_level):
        # The lines show better over the spectrogram than over the surface.
        strong, weak = (0.5, 0.3) if a is ax else (0.3, 0.18)
        for first, _, _ in SECTIONS[1:]:
            a.axvline(first * BAR, color=ink, lw=0.8, alpha=strong)
        for t0, _, _ in CUES:
            a.axvline(t0, color=ink, lw=0.5, alpha=weak)
        a.spines[["top", "right"]].set_visible(False)
    title = (
        "score.wav: spectrogram of the mono sum, with the scene's cues       "
        f"integrated loudness {lufs:.1f} LUFS, true peak {peak:.1f} dBFS"
    )
    fig.suptitle(title, x=0.065, y=0.925, ha="left", color=ink, fontsize=12)
    path = out_dir / "score_spectrogram.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {path.name}", flush=True)
    if events_dry is None:
        return

    # The events alone around each cue: the sound must start on the line.
    fig, axes = plt.subplots(5, 7, figsize=(18, 10.5), dpi=110, sharey=True)
    before, after = int(0.02 * SR), int(0.06 * SR)
    for a, (t0, kind, _) in zip(axes.flat, CUES):
        i = int(round(t0 * SR))
        seg = events_dry[i - before : i + after].mean(axis=1)
        ms = (np.arange(len(seg)) - before) / SR * 1000.0
        a.plot(ms, seg, color=blue, lw=0.9)
        a.axvline(0.0, color=ink, lw=0.8, alpha=0.6)
        title = f"{t0:.3f} s   {EVENT_NAMES[kind]}"
        a.set_title(title, color=ink2, fontsize=9, loc="left")
        a.grid(axis="y", color=grid, lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    for a in axes.flat[len(CUES) :]:
        a.set_visible(False)
    for a in axes[-1]:
        a.set_xlabel("time from the cue, ms")
    for a in axes[:, 0]:
        a.set_ylabel("amplitude")
    title = (
        "The events alone (no echo, no reverb) around each cue: "
        "the sound starts on the line"
    )
    fig.suptitle(title, x=0.06, y=0.97, ha="left", color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "score_onsets.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# 10. Main
# ══════════════════════════════════════════════════════════════════════════


def synthesize(out_dir):
    """Play, master and write the score, its stems and its compressed copy;
    print the balance of the stems and what the limiter did."""
    start = time.time()
    print("playing the song ...", flush=True)
    stems, events_dry, kicks = play()
    stems = {name: premaster(x) for name, x in stems.items()}
    mix = sum(stems.values())
    seconds = time.time() - start
    print(
        f"  {len(kicks)} kicks, {len(CUES)} cues; {seconds:.1f} s", flush=True
    )

    out, gain, reduction = master(mix)
    share = 100.0 * (reduction < -0.1).mean()
    print(
        f"master: gain {to_db(gain):+.2f} dB; the limiter takes off at most "
        f"{-reduction.min():.2f} dB, and more than 0.1 dB during "
        f"{share:.2f} % of the time",
        flush=True,
    )
    edges = np.round(np.arange(4 * N_BARS + 1) * BEAT * SR).astype(int)
    beats = [-reduction[a:b].min() for a, b in zip(edges[:-1], edges[1:])]
    deepest = np.argsort(beats)[::-1][:8]
    listed = ", ".join(
        f"{b * BEAT:.3f} s ({beats[b]:.1f} dB)" for b in deepest
    )
    print(f"  the beats on which it takes off most: {listed}", flush=True)

    # The balance of the stems, at the level at which they enter the
    # limiter: the peak over the piece, and the groove of bars 13 and 14.
    a, b = int(13 * BAR * SR), int(15 * BAR * SR)
    print("stem      peak dBFS   bars 13-14: RMS dBFS   LUFS", flush=True)
    for name, x in [*stems.items(), ("mix", mix)]:
        y = x * gain
        peak, rms = to_db(np.abs(y).max()), power_db(np.mean(y[a:b] ** 2))
        lufs = loudness(y[a:b], gated=False)
        print(f"  {name:7s} {peak:8.1f} {rms:22.1f} {lufs:6.1f}", flush=True)

    (out_dir / "stems").mkdir(parents=True, exist_ok=True)
    pcm = write_wav(out_dir / "score.wav", out)
    for name, x in stems.items():
        write_wav(out_dir / "stems" / f"{name}.wav", x * gain)
    write_wav(
        out_dir / "stems" / "events_dry.wav", premaster(events_dry) * gain
    )
    print(f"wrote score.wav and {len(stems) + 1} stems", flush=True)
    path = write_compressed(out_dir, pcm)
    if path is not None:
        data, size = sf.read(str(path))[0], path.stat().st_size
        print(
            f"wrote {path.name}: {size} bytes, "
            f"{8 * size / DURATION / 1000:.0f} kbit/s, {len(data)} frames "
            f"decoded, peak {to_db(np.abs(data).max()):.2f} dBFS",
            flush=True,
        )
    print(f"  {time.time() - start:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="do not synthesize: measure and plot the files on disk",
    )
    parser.add_argument("out", type=Path, help="the folder to write into")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    version = "none" if sf is None else sf.__version__
    print(
        f"python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"soundfile {version}",
        flush=True,
    )
    if not args.analyze:
        synthesize(args.out)
    print("measuring ...", flush=True)
    lines, failed = analyze(args.out)
    print("\n".join(lines), flush=True)
    if failed:
        sys.exit("FAILED: " + "; ".join(failed))


if __name__ == "__main__":
    main()
