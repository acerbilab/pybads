"""ellipsoid_D3_homo: W4-21's runs (W4-1's code with the design's seed
forced to 967, no draw; checked against 86512c9 itself), W4-1's runs (the
wave-4 reference, W4-6 leaving the noisy runs as they were; checked against
efe5e95 itself), the wave-3 reference, and W4-1's code with the same design
as W4-21 but W4-1's draw (forced 967 after the draw)."""
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load(p):
    return {r["seed"]: r for r in map(json.loads, open(p))}


base = Path("/home/user/pybads-review/dev/experiments")


def ref(name):
    out = {}
    for s in range(30):
        r = json.loads(
            (base / name / f"ellipsoid_D3_homo_seed{s}.json").read_text()
        )
        out[s] = dict(r["final"], seed=s)
    return out


w3 = ref("population_linux_wave3_20260927")
w4 = ref("population_linux_wave4_20260927")
w421_direct = load("edh_w421.jsonl")
f967 = load("edh_w41_force967.jsonl")
w41_direct = load("edh_w41.jsonl")
fd967 = load("edh_w41_forcedraw967.jsonl")
tol = 0.1
same = (
    lambda a, b: a["func_count"] == b["func_count"]
    and a["x"] == b["x"]
    and a["fval"] == b["fval"]
)
print(
    "forced 967 (no draw) == 86512c9 run:",
    [same(w421_direct[s], f967[s]) for s in sorted(w421_direct)],
)
print(
    "efe5e95 run == wave-4 reference record:",
    [same(w41_direct[s], w4[s]) for s in sorted(w41_direct)],
)


def summ(name, d, seeds):
    e = np.array([d[s]["true_error"] for s in seeds])
    fc = np.array([d[s]["func_count"] for s in seeds])
    print(
        f"{name:28s} n={len(seeds)} solved {np.mean(e <= tol):.2f} median err {np.median(e):.3g} "
        f"quartiles {np.quantile(e, .25):.3g}/{np.quantile(e, .75):.3g} mean fc {fc.mean():.1f}"
    )
    return e


S30, S24 = list(range(30)), sorted(fd967)
print("--- over seeds 0-29")
e3 = summ("wave3 ref (a14524d)", w3, S30)
e21 = summ("W4-21 (forced 967)", f967, S30)
e1 = summ("W4-1 (wave4 ref)", w4, S30)
print(f"--- over seeds {S24[0]}-{S24[-1]}")
summ("W4-21 (forced 967)", f967, S24)
summ("W4-1 (wave4 ref)", w4, S24)
summ("967 + W4-1's draw", fd967, S24)
summ("wave3 ref", w3, S24)


def paired(a, b, seeds, label):
    r = np.log10(
        np.array([b[s]["true_error"] for s in seeds])
        / np.array([a[s]["true_error"] for s in seeds])
    )
    w = stats.wilcoxon(r)
    flips = sum(
        (a[s]["true_error"] <= tol) != (b[s]["true_error"] <= tol)
        for s in seeds
    )
    print(
        f"{label:34s} median log10 ratio {np.median(r):+.3f}, signed-rank p {w.pvalue:.3f}, runs whose solved flag flips {flips}, changed {sum(not same(a[s], b[s]) for s in seeds)}"
    )


paired(w3, f967, S30, "W4-21 vs wave3")
paired(f967, w4, S30, "W4-1 vs W4-21")
paired(w3, w4, S30, "pass (wave4 vs wave3)")
paired(f967, fd967, S24, "967+draw vs W4-21 (same design)")
paired(fd967, w4, S24, "W4-1 vs 967+draw (design only)")
