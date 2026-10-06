"""The investigation of W0-1 on ellipsoid_D3_homo: each variant against the
base over paired seeds (median error, evaluations, solved, termination,
paired signed-rank test of log10 error and of evaluations).

Run from the repository root:
python dev/experiments/port_review_20260925/w01_investigation/w01_analysis.py
"""

import collections
import glob
import json
import os

import numpy as np
from scipy import stats

# The records beside this script; run from the repository root
P = os.path.dirname(os.path.abspath(__file__))


def load(d):
    out = {}
    for p in glob.glob(os.path.join(P, d, "*.json")):
        r = json.load(open(p))
        out[r["seed"]] = r["final"]
    return out


ref = {}
for p in glob.glob(
    "dev/experiments/population_gpfixes_20260925/ellipsoid_D3_homo_seed*.json"
):
    r = json.load(open(p))
    ref[r["seed"]] = r["final"]
base = load("w01_base_ellhomo")
same = sum(
    ref[s]["x"] == base[s]["x"]
    and ref[s]["func_count"] == base[s]["func_count"]
    for s in ref
)
print(
    f"base (010eeb4) vs the Windows reference, seeds 0-29: {same} of {len(ref)} with the same x and evaluations"
)

tol = 0.1  # the configuration's tolerance for "solved"


def summary(name, runs, seeds):
    e = np.array([runs[s]["true_error"] for s in seeds])
    f = np.array([runs[s]["func_count"] for s in seeds])
    stall = sum(
        "change in the function value" in runs[s]["message"] for s in seeds
    )
    return (
        e,
        f,
        (
            f"{name:28s} median err {np.median(e):.4f}  [q25 {np.percentile(e, 25):.4f}, q75 {np.percentile(e, 75):.4f}]"
            f"  solved {np.mean(e < tol):.2f}  evals {np.median(f):.0f}  stall-ends {stall}/{len(seeds)}"
        ),
    )


def paired(e0, e1, f0, f1):
    d = np.log10(e1) - np.log10(e0)
    p_err = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    p_ev = stats.wilcoxon(f1 - f0).pvalue if np.any(f1 != f0) else 1.0
    return f"    vs base: median log10 ratio {np.median(d):+.3f}, signed-rank p (error) {p_err:.3g}, p (evals) {p_ev:.3g}, identical runs {int(np.sum((d == 0) & (f1 == f0)))}"


seeds = list(range(90))
e0, f0, line = summary("base (010eeb4)", base, seeds)
print(line)
for name, d in [
    ("a: swap on a deep copy", "w01_a_ellhomo"),
    ("b: no swap, old re-eval", "w01_b_ellhomo"),
    ("c: W0-1 (MATLAB)", "w01_c_ellhomo"),
]:
    runs = load(d)
    e1, f1, line = summary(name, runs, seeds)
    print(line)
    print(paired(e0, e1, f0, f1))
    for lo, hi in [(0, 30), (30, 90)]:
        s = slice(lo, hi)
        dd = np.log10(e1[s]) - np.log10(e0[s])
        print(
            f"      seeds {lo}-{hi - 1}: median err {np.median(e0[s]):.4f} -> {np.median(e1[s]):.4f}, median log10 ratio {np.median(dd):+.3f}"
        )

b_ns, c_ns = load("w01_base_nostall"), load("w01_c_nostall")
for lo, hi in [(0, 30), (30, 90), (0, 90)]:
    ss = list(range(lo, hi))
    print(
        f"\nstall criterion off (tol_stall_iters 100000), seeds {lo}-{hi - 1}:"
    )
    e0n, f0n, line = summary("base, no stall", b_ns, ss)
    print(line)
    e1n, f1n, line = summary("c, no stall", c_ns, ss)
    print(line)
    print(paired(e0n, e1n, f0n, f1n))
    term = collections.Counter(c_ns[s]["message"][:45] for s in ss)
    print("    c, no stall, terminations:", dict(term))
