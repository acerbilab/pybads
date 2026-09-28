"""W2-36's comparison, per configuration and pooled: the runs that the
variant changes, the fraction solved in each arm with an exact McNemar test
on the discordant pairs and a bootstrap interval of the paired difference,
and the paired log10 error ratio over the changed runs.

    python w236_pairs.py REF_DIR NEW_DIR [RERUN_DIR]

REF_DIR and NEW_DIR are the population directories of the two arms, paired
by (label, seed). With RERUN_DIR, runs of REF_DIR's code made again, it
also reports how many of them equal the records of REF_DIR exactly (every
field of ``final`` but the wall time), and lists those that differ.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

FIELDS = (
    "x",
    "fval",
    "fsd",
    "true_error",
    "func_count",
    "iterations",
    "message",
    "crashed",
)
N_BOOT = 10000


def load(d):
    out = {}
    for p in Path(d).glob("*_seed*.json"):
        r = json.loads(p.read_text())
        out[(r["label"], r["seed"])] = r
    return out


def same(a, b):
    return all(a["final"].get(k) == b["final"].get(k) for k in FIELDS)


def solved(r):
    e = r["final"]["true_error"]
    return not r["final"]["crashed"] and e is not None and e < r["tolerance"]


def boot_diff(ref, new, keys, labels, rng):
    """Bootstrap interval of the paired difference in the fraction solved
    (NEW minus REF), resampling the pairs within each configuration."""
    d = {
        lab: np.array(
            [solved(new[k]) - solved(ref[k]) for k in keys if k[0] == lab],
            dtype=float,
        )
        for lab in labels
    }
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        parts = [x[rng.integers(0, len(x), len(x))] for x in d.values()]
        means[i] = np.mean(np.concatenate(parts))
    return np.percentile(means, [2.5, 97.5])


def main():
    ref, new = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(ref) & set(new))
    labels = sorted({k[0] for k in keys})
    rng = np.random.default_rng(0)
    print(
        "| config | pairs | changed | solved ref | solved new | new only |"
        " ref only | McNemar p | paired difference [95% CI] |"
        " median log10 error ratio, changed runs |"
        " signed-rank p, changed runs |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label in labels + ["all"]:
        labs = labels if label == "all" else [label]
        ks = [k for k in keys if k[0] in labs]
        changed = [k for k in ks if not same(ref[k], new[k])]
        s_ref = sum(solved(ref[k]) for k in ks) / len(ks)
        s_new = sum(solved(new[k]) for k in ks) / len(ks)
        b = sum(solved(new[k]) and not solved(ref[k]) for k in ks)
        c = sum(solved(ref[k]) and not solved(new[k]) for k in ks)
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        lo, hi = boot_diff(ref, new, ks, labs, rng)
        r = np.array(
            [
                np.log10(
                    new[k]["final"]["true_error"]
                    / ref[k]["final"]["true_error"]
                )
                for k in changed
            ]
        )
        med = f"{np.median(r):+.3f}" if len(r) else "—"
        w = f"{wilcoxon(r).pvalue:.3f}" if len(r) >= 10 else "—"
        print(
            f"| {label} | {len(ks)} | {len(changed)} | {s_ref:.2f}"
            f" | {s_new:.2f} | {b} | {c} | {p:.3f}"
            f" | {s_new - s_ref:+.3f} [{lo:+.3f}, {hi:+.3f}] | {med} | {w} |"
        )
    print(
        "\nThe paired difference is NEW minus REF; its interval is the"
        f" percentile interval of {N_BOOT} bootstrap resamples of the pairs,"
        " within each configuration."
    )

    if len(sys.argv) > 3:
        rerun = load(sys.argv[3])
        common = sorted(k for k in rerun if k in ref)
        differ = [
            k
            for k in common
            if any(
                rerun[k]["final"][f] != ref[k]["final"][f]
                for f in rerun[k]["final"]
                if f != "wall_s"
            )
        ]
        print(
            f"\nReproduced: {len(common) - len(differ)} of {len(common)}"
            " runs of RERUN_DIR equal the records of REF_DIR (every field of"
            " final but wall_s)."
        )
        for k in differ:
            print(f"- differs: {k[0]} seed {k[1]}")


if __name__ == "__main__":
    main()
