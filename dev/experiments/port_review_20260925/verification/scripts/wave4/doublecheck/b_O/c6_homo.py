"""ellipsoid_D3_homo in the two committed Linux references: the errors and
the tolerance of 'solved', seed by seed."""
import glob
import json
import re

import numpy as np

base = "/home/user/pybads-review/dev/experiments/"
for pop in [
    "population_linux_wave3_20260927",
    "population_linux_wave4_20260927",
]:
    recs = {}
    for f in glob.glob(base + pop + "/ellipsoid_D3_homo_seed*.json"):
        r = json.load(open(f))
        s = int(re.search(r"seed(\d+)", f).group(1))
        recs[s] = r
    tol = recs[0]["tolerance"]
    err = np.array([recs[s]["final"]["true_error"] for s in range(30)])
    fc = np.array([recs[s]["final"]["func_count"] for s in range(30)])
    print(
        pop,
        "tolerance",
        tol,
        "solved",
        np.sum(err < tol),
        "/30 median err",
        np.median(err),
        "meta",
        {
            k: recs[0]["meta"].get(k)
            for k in ("commit", "pybads_commit", "git_commit")
            if k in recs[0].get("meta", {})
        },
    )
    print("  errors sorted:", np.round(np.sort(err), 4).tolist())
    print(
        "  within x1.5 of tol:",
        int(np.sum((err > tol / 1.5) & (err < tol * 1.5))),
    )
