"""Check the meta of the two committed Linux references, their times, and the
per-configuration medians and fractions solved, for comparison with the
chain of the pass's *_changed.txt files."""
import collections
import glob
import json
import os

import gpyreg

import pybads

print(pybads.__file__, gpyreg.__file__)
import numpy as np

R = "/home/user/pybads-review/dev/experiments"
for name in [
    "population_linux_wave3_20260927",
    "population_linux_wave4_20260927",
]:
    d = os.path.join(R, name)
    files = sorted(glob.glob(os.path.join(d, "*_seed*.json")))
    metas = collections.Counter()
    starts, ends = [], []
    err = collections.defaultdict(list)
    solved = collections.defaultdict(list)
    crashed = 0
    for f in files:
        r = json.load(open(f))
        m = r["meta"]
        key = (
            m["git"]["sha"],
            m["git"]["dirty"],
            m["pybads"],
            m["pybads_source"]["git"]["sha"],
            m["pybads_source"]["git"]["dirty"],
            m["pybads_source"]["path"].split("worktrees/")[-1],
            m["gpyreg"],
            m["gpyreg_source"]["git"]["sha"],
            m["python"],
            m["numpy"],
            m["scipy"],
            m["platform"],
            tuple(sorted(m["threads"].items())),
        )
        metas[key] += 1
        starts.append(m["started"])
        ends.append(m["finished"])
        lab = r["label"]
        err[lab].append(r["final"]["true_error"])
        solved[lab].append(
            r["final"]["true_error"] <= r["tolerance"]
            if r["final"]["true_error"] is not None
            else False
        )
        crashed += bool(r["final"]["crashed"])
    print(name, len(files), "records; crashed", crashed)
    for k, v in metas.items():
        print("  meta", v, k)
    print("  started", min(starts), "finished", max(ends))
    for lab in sorted(err):
        print(
            f"  {lab}: n={len(err[lab])} median error {np.median(err[lab]):.3g} solved {np.mean(solved[lab]):.2f}"
        )
