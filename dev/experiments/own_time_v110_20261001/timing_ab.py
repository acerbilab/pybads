"""PyBADS 1.1.0 with gpyreg 1.3.3 (A) against PyBADS at 1ecfeb61 with
gpyreg 1.4.0 (B): the `profile` suite, plain runs of profile_run.py, one at
a time, each pinned to one performance core with Windows's power throttling
off, the arms alternating AB and BA over the configurations and seeds.

    python timing_ab.py OUT_DIR [--cpu 12] [--seeds 0-29]

The runs of arm A go to OUT_DIR/A, those of arm B to OUT_DIR/B, one
directory per run as profile_run.py writes it, with ``machine.json`` beside
its ``summary.json``: the load of the whole machine while the run ran (the
busy fraction of all logical CPUs, from GetSystemTimes), which shows a run
that another process disturbed. A run already recorded is skipped, so the
campaign resumes after an interruption.

Arm A runs the profile_run.py, population.py, benchmark_targets.py and
harness.py of B's commit, copied into a worktree at v1.1.0, so that both
arms run the same problems, start points and noise.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "dev/scripts/runs"
ARMS = {
    "A": (RUNS / "worktrees/timing_v110", RUNS / "gpyreg/v1.3.3"),
    "B": (RUNS / "worktrees/timing_1ecfeb61", RUNS / "gpyreg/v1.4.0"),
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


def system_times():
    """Idle and total time of all logical CPUs, in 100 ns units."""
    idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
    assert k.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    )

    def ticks(ft):
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    # the kernel time includes the idle time
    return ticks(idle), ticks(kernel) + ticks(user)


def seeds_of(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def run(arm, label, seed, out, cpu):
    worktree, gpyreg = ARMS[arm]
    camp = out / arm
    camp.mkdir(parents=True, exist_ok=True)
    tag = f"{label}_seed{seed}_plain"
    if (camp / tag / "summary.json").exists():
        return
    env = dict(os.environ, PYTHONPATH=str(gpyreg))
    env.update({v: "1" for v in THREADS})
    cmd = [
        sys.executable,
        "-u",
        str(worktree / "dev/scripts/profile_run.py"),
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
    idle0, total0 = system_times()
    with open(camp / f"{tag}.log", "w", encoding="utf-8") as fh:
        p = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env
        )
        pin(p._handle, cpu)
        rc = p.wait()
    idle1, total1 = system_times()
    busy = 1 - (idle1 - idle0) / max(total1 - total0, 1)
    if (camp / tag).is_dir():
        (camp / tag / "machine.json").write_text(
            json.dumps(
                {"busy": busy, "cpus": os.cpu_count(), "pinned_cpu": cpu}
            )
        )
    print(
        f"{time.strftime('%H:%M:%S')} {arm} {tag} rc={rc}"
        f" {time.time() - t0:.1f} s, machine busy {100 * busy:.1f} %",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cpu", type=int, default=12)
    ap.add_argument("--seeds", default="0-29")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    for arm, (worktree, gpyreg) in ARMS.items():
        print(f"[timing] {arm}: {worktree} with {gpyreg}", flush=True)
    print(f"[timing] {os.cpu_count()} logical CPUs", flush=True)
    i = 0
    for seed in seeds_of(a.seeds):
        for label in PROFILE:
            for arm in "AB" if i % 2 == 0 else "BA":
                run(arm, label, seed, out, a.cpu)
            i += 1
    print("[timing] done", flush=True)


if __name__ == "__main__":
    main()
