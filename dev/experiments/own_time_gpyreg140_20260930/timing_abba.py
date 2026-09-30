"""PyBADS's own time under gpyreg 1.3.3 (A) and gpyreg 3e56dce, 1.4.0's
commit (B): the same PyBADS (the worktree at bef26ec2), the `profile` suite,
plain runs of profile_run.py, one at a time, each pinned to one performance
core with Windows's power throttling off, the arms alternating ABBA over the
repetitions of each configuration and seed.

    python timing_abba.py OUT_DIR [--cpu 12] [--seeds 0-2] [--reps 2]

Each repetition of each arm goes to OUT_DIR/<arm>_rep<r>, a campaign that
profile_compare.py reads.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WT = ROOT / "dev/scripts/runs/worktrees/winref_bef26ec2"
ARMS = {
    "A": ROOT / "dev/scripts/runs/gpyreg/v1.3.3",
    "B": ROOT / "dev/scripts/runs/gpyreg/main_3e56dce",
}
PROFILE = [
    "ellipsoid_D3",
    "ellipsoid_D10",
    "rosenbrock_D6",
    "ackley_D6",
    "multisensory_s1_D6_homo",
    "ellipsoid_D3_homo",
    "sphere_D3_hetero",
]
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

k = ctypes.WinDLL("kernel32", use_last_error=True)


class PowerThrottlingState(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


def pin(handle, cpu):
    h = wintypes.HANDLE(int(handle))
    assert k.SetProcessAffinityMask(h, ctypes.c_size_t(1 << cpu))
    state = PowerThrottlingState(1, 1, 0)  # EXECUTION_SPEED, off
    k.SetProcessInformation(h, 4, ctypes.byref(state), ctypes.sizeof(state))


def seeds_of(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def run(arm, label, seed, rep, out, cpu):
    camp = out / f"{arm}_rep{rep}"
    camp.mkdir(parents=True, exist_ok=True)
    tag = f"{label}_seed{seed}_plain"
    if (camp / tag / "summary.json").exists():
        return
    env = dict(os.environ, PYTHONPATH=str(ARMS[arm]))
    env.update({v: "1" for v in THREADS})
    cmd = [
        sys.executable,
        "-u",
        str(WT / "dev/scripts/profile_run.py"),
        "--config",
        label,
        "--seed",
        str(seed),
        "--out",
        str(camp),
        "--tag",
        tag,
    ]
    t0 = time.time()
    with open(camp / f"{tag}.log", "w", encoding="utf-8") as fh:
        p = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env
        )
        pin(p._handle, cpu)
        rc = p.wait()
    print(
        f"{time.strftime('%H:%M:%S')} {arm} rep{rep} {tag} rc={rc}"
        f" {time.time() - t0:.1f} s",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cpu", type=int, default=12)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    for arm, g in ARMS.items():
        print(f"[timing] {arm}: {g}", flush=True)
    i = 0
    for label in PROFILE:
        for seed in seeds_of(a.seeds):
            for rep in range(1, a.reps + 1):
                order = "AB" if (i + rep) % 2 else "BA"
                for arm in order:
                    run(arm, label, seed, rep, out, a.cpu)
            i += 1
    print("[timing] done", flush=True)


if __name__ == "__main__":
    main()
