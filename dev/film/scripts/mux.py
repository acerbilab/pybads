"""Put the film's sound under its recorded video, at -16 LUFS.

    python scripts/mux.py VIDEO AUDIO OUT

Two passes of ffmpeg's EBU R128 loudness normalization bring the sound to
-16 LUFS integrated and -1.5 dBTP true peak, the level that video
platforms play at. The video stream is copied, and the film ends with the
shorter of the two streams. ffmpeg is FFMPEG, else the one that imageio-ffmpeg installs
into the environment that runs this script, else ffmpeg on the PATH.
"""

import argparse
import json
import os
import subprocess

TARGET = "I=-16:TP=-1.5:LRA=11"


def ffmpeg_exe():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def measure(ffmpeg, path):
    p = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            path,
            "-vn",
            "-af",
            f"loudnorm={TARGET}:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise SystemExit(
            f"ffmpeg could not measure {path}:\n{p.stderr[-2000:]}"
        )
    return json.loads(
        p.stderr[p.stderr.rindex("{") : p.stderr.rindex("}") + 1]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("audio")
    ap.add_argument("out")
    args = ap.parse_args()
    ffmpeg = ffmpeg_exe()
    m = measure(ffmpeg, args.audio)
    print(f"measured {m['input_i']} LUFS, {m['input_tp']} dBTP", flush=True)
    loudnorm = (
        f"loudnorm={TARGET}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
        f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
        f"offset={m['target_offset']}:linear=true"
    )
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", args.video, "-i", args.audio,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", loudnorm, "-ar", "48000",
         "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", args.out],
        check=True,
    )  # fmt: skip
    after = measure(ffmpeg, args.out)
    print(
        f"wrote {args.out}: {after['input_i']} LUFS, {after['input_tp']} dBTP",
        flush=True,
    )


if __name__ == "__main__":
    main()
