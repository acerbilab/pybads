"""Tabulate the stage profiles written by profile_periodic.py, one row per
configuration and seed:

    python tabulate.py DIR TAG [TAG2]

reads DIR/TAG_<label>_s<seed>_{on,off}.json and, with TAG2, the "on" profiles
of DIR/TAG2_...; the kernel time sums the kernel and gpyreg's periodic
helpers, and the periodic excess is that time less the same calls without
periods, as a share of the run."""
import glob
import json
import sys
from collections import defaultdict

d = sys.argv[1]
tags = sys.argv[2:]
KERNEL = ("kernel_", "sq_dist", "sq_diff")


def load(tag, arm):
    out = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/{tag}_periodic*_s*_{arm}.json")):
        r = json.load(open(f))
        out[r["label"]].append(r)
    return out


def kern(r):
    return sum(v for k, v in r["tot"].items() if k.startswith(KERNEL))


def cf(r):
    return sum(v for k, v in r["tot"].items() if k.startswith("cf_"))


base_on, base_off = load(tags[0], "on"), load(tags[0], "off")
new_on = load(tags[1], "on") if len(tags) > 1 else {}
print(
    "| configuration | seed | evals on / off | run s on / off | ms/eval on / off"
    " | kernel s (on) | same calls without periods s | periodic excess, % of run"
    + (
        " | new: evals | new: run s | new: kernel s | new: saving % |"
        if new_on
        else " |"
    )
)
print("|---" * (12 if new_on else 8) + "|")
for lab in base_on:
    for r in base_on[lab]:
        o = [x for x in base_off.get(lab, []) if x["seed"] == r["seed"]]
        o = o[0] if o else None
        k, c = kern(r), cf(r)
        row = (
            f"| {lab} | {r['seed']} | {r['func_count']} / {o['func_count'] if o else '-'}"
            f" | {r['run_s']:.2f} / {o['run_s'] if o else float('nan'):.2f}"
            f" | {1e3 * r['run_s'] / r['func_count']:.1f} / "
            f"{1e3 * o['run_s'] / o['func_count'] if o else float('nan'):.1f}"
            f" | {k:.2f} | {c:.2f} | {100 * (k - c) / r['run_s']:.0f} %"
        )
        if new_on:
            n = [x for x in new_on.get(lab, []) if x["seed"] == r["seed"]]
            if n:
                n = n[0]
                row += (
                    f" | {n['func_count']} | {n['run_s']:.2f} | {kern(n):.2f}"
                    f" | {100 * (1 - n['run_s'] / r['run_s']):.0f} %"
                )
            else:
                row += " | - | - | - | - "
        print(row + " |")
