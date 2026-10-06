"""edgesphere_D2, seeds 0-29, with W4-1's code (run from its worktree) and the
seed of the initial design's scrambling held fixed: is the reference's narrow
spread of evaluations that of one shared design?

Modes: "old948" forces the seed 948 that init_sobol derived from any start
inside the plausible box at D = 2 before W4-1, without W4-1's draw from the
run's generator (so that the run should equal W4-21's); "948+draw" forces 948
after W4-1's draw (the same design, W4-1's later draws); "fixed<k>" forces
another seed k, without the draw; "W4-1" is the code as it is, one design per
seed."""

import json
import sys
from pathlib import Path

import numpy as np

WT = Path(sys.argv[1])
sys.path.insert(0, str(WT / "dev" / "scripts"))
import population  # noqa: E402  (puts the worktree's checkout first)

import pybads  # noqa: E402
import pybads.init_functions  # noqa: E402

# the package re-exports the function under the module's name
init_sobol_module = sys.modules["pybads.init_functions.init_sobol"]

print(pybads.__file__, flush=True)
real_get_rng = init_sobol_module.get_rng


class _Forced:
    def __init__(self, rng, seed, draw):
        self.rng, self.seed, self.draw = rng, seed, draw

    def integers(self, *args, **kwargs):
        if self.draw:
            real_get_rng(self.rng).integers(*args, **kwargs)
        return self.seed


def run(mode, seed_value=None, draw=False):
    if seed_value is None:
        init_sobol_module.get_rng = real_get_rng
    else:
        init_sobol_module.get_rng = lambda rng=None: _Forced(
            rng, seed_value, draw
        )
    out = Path(sys.argv[2]) / mode
    out.mkdir(parents=True, exist_ok=True)
    fc, err = [], []
    for s in range(30):
        row = population.run_task("edgesphere_D2", s, {}, 1.0, out)
        fc.append(row["func_count"])
        err.append(row["true_error"])
    fc, err = np.array(fc), np.array(err)
    print(
        f"{mode:>10}: evaluations mean {fc.mean():5.1f}, range {fc.min()}-{fc.max()}, "
        f"distinct {len(set(fc))}; error median {np.median(err):.2g}, max {err.max():.2g}",
        flush=True,
    )
    return fc, err


results = {}
results["old948"] = run("old948", 948, draw=False)
results["948+draw"] = run("948+draw", 948, draw=True)
for k in (1, 2, 3, 500, 997, 12345):
    results[f"fixed{k}"] = run(f"fixed{k}", k, draw=False)
results["W4-1"] = run("W4-1")

# old948 against W4-21's records, run by run
ref = Path("/home/user/pybads/dev/scripts/runs/population/geo_w421_86512c9")
same = 0
for s in range(30):
    a = json.loads((ref / f"edgesphere_D2_seed{s}.json").read_text())["final"]
    b = json.loads(
        (
            Path(sys.argv[2]) / "old948" / f"edgesphere_D2_seed{s}.json"
        ).read_text()
    )["final"]
    same += a["func_count"] == b["func_count"] and a["x"] == b["x"]
print(
    f"old948 equals W4-21's run in {same} of 30 seeds (func_count and x)",
    flush=True,
)
