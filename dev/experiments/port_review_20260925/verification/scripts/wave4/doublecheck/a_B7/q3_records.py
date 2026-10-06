"""ellipsoid_D3_homo in the two committed references: per seed, the error,
the evaluations, the message, fsd; the tolerance; solved = error <= tol."""
import json
from pathlib import Path

import numpy as np

base = Path("/home/user/pybads-review/dev/experiments")
refs = {
    "wave3": base / "population_linux_wave3_20260927",
    "wave4": base / "population_linux_wave4_20260927",
}
recs = {}
for k, p in refs.items():
    recs[k] = {}
    for s in range(30):
        r = json.loads((p / f"ellipsoid_D3_homo_seed{s}.json").read_text())
        recs[k][s] = r
r0 = recs["wave3"][0]
print(
    "tolerance",
    r0["tolerance"],
    "meta3",
    r0["meta"].get("git"),
    "| meta4",
    recs["wave4"][0]["meta"].get("git"),
)
print("final keys", sorted(r0["final"].keys()))
tol = r0["tolerance"]
solved = {
    k: np.mean([recs[k][s]["final"]["true_error"] <= tol for s in range(30)])
    for k in recs
}
print("fraction solved", solved)
print("seed  err3      err4      fc3  fc4  it3 it4  fsd3    fsd4    msg3/msg4")
flip_down = flip_up = 0
for s in range(30):
    a, b = recs["wave3"][s]["final"], recs["wave4"][s]["final"]
    sa, sb = a["true_error"] <= tol, b["true_error"] <= tol
    flip_down += sa and not sb
    flip_up += sb and not sa
    print(
        f"{s:4d}  {a['true_error']:.3e} {b['true_error']:.3e} {a['func_count']:4d} {b['func_count']:4d} {a['iterations']:3d} {b['iterations']:3d}  {a['fsd']:.3f}  {b['fsd']:.3f}  {a['message'][27:45]}/{b['message'][27:45]}  {'S' if sa else '-'}{'S' if sb else '-'}"
    )
e3 = np.array([recs["wave3"][s]["final"]["true_error"] for s in range(30)])
e4 = np.array([recs["wave4"][s]["final"]["true_error"] for s in range(30)])
print("flips solved->not", flip_down, "not->solved", flip_up)
print(
    "median err",
    np.median(e3),
    np.median(e4),
    "quantiles3",
    np.quantile(e3, [0.25, 0.5, 0.75]),
    "quantiles4",
    np.quantile(e4, [0.25, 0.5, 0.75]),
)
print(
    "errors within a factor 2 of tol: wave3",
    np.sum((e3 > tol / 2) & (e3 < 2 * tol)),
    "wave4",
    np.sum((e4 > tol / 2) & (e4 < 2 * tol)),
)
fc3 = np.array([recs["wave3"][s]["final"]["func_count"] for s in range(30)])
fc4 = np.array([recs["wave4"][s]["final"]["func_count"] for s in range(30)])
print("func_count mean", fc3.mean(), fc4.mean())
w3 = np.array(
    [recs["wave3"][s]["final"].get("wall_s", np.nan) for s in range(30)]
)
w4 = np.array(
    [recs["wave4"][s]["final"].get("wall_s", np.nan) for s in range(30)]
)
print(
    "wall_s mean",
    np.nanmean(w3),
    np.nanmean(w4),
    "max",
    np.nanmax(w3),
    np.nanmax(w4),
)
fsd3 = np.array([recs["wave3"][s]["final"]["fsd"] for s in range(30)])
fsd4 = np.array([recs["wave4"][s]["final"]["fsd"] for s in range(30)])
print("fsd median", np.median(fsd3), np.median(fsd4))
from collections import Counter

print(
    "messages3",
    Counter(recs["wave3"][s]["final"]["message"][27:60] for s in range(30)),
)
print(
    "messages4",
    Counter(recs["wave4"][s]["final"]["message"][27:60] for s in range(30)),
)
print(
    "x0 equal across refs:",
    all(recs["wave3"][s]["x0"] == recs["wave4"][s]["x0"] for s in range(30)),
)
