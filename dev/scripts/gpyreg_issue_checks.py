"""Two PyBADS-side GP issues, under the gpyreg that ``PYTHONPATH`` selects.

1. The known-noise path: one run of ``ellipsoid_D3`` (seed 0) with
   ``fit_lik=False``; records how it ends and where it raises.
2. ``_robust_gp_fit_``: for each call, the number of ``gp.fit`` calls that
   raise ``LinAlgError`` before one succeeds, and how the call ends; over
   the ``default`` suite of ``benchmark_targets.py``, seeds 0-9, in-process
   (``gpyreg.GP.fit`` and ``_robust_gp_fit_`` are wrapped for counting).

The survey (``dev/results/2026-09-23-codebase-survey.md``) cites its runs
under gpyreg 1.3.1 and 1.3.2 (2026-09-24: the suite's first 15
configurations, draws through NumPy's global stream) and under 1.3.3
(2026-09-25: all 18 configurations, draws through ``bads.rng``). Usage,
from the repository root::

    PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 python -u dev/scripts/gpyreg_issue_checks.py OUT.json
"""
import json
import sys
import traceback

import numpy as np

sys.path.insert(0, "dev/scripts")
import benchmark_targets as bt  # noqa: E402
import gpyreg  # noqa: E402

import pybads.bads.gaussian_process_train as gtrain  # noqa: E402
from pybads import BADS  # noqa: E402

print(gpyreg.__file__, flush=True)
out = {"gpyreg": gpyreg.__file__}

# 1. fit_lik=False
prob = bt.find_config("ellipsoid_D3").make(seed=0)
args, options = prob.bads_args()
options["fit_lik"] = False
try:
    r = BADS(*args, options=options).optimize()
    out["fit_lik_false"] = {"outcome": "finished", "fval": float(r["fval"])}
except Exception as e:  # noqa: BLE001
    tb = traceback.extract_tb(e.__traceback__)
    out["fit_lik_false"] = {
        "outcome": f"{type(e).__name__}: {e}",
        "raised_in": f"{tb[-1].filename.split('site-packages')[-1]}:{tb[-1].lineno} {tb[-1].name}",
        "pybads_frame": next(
            (
                f"{f.filename.split('pybads')[-1]}:{f.lineno} {f.name}"
                for f in reversed(tb)
                if "pybads" in f.filename and "gpyreg" not in f.filename
            ),
            None,
        ),
    }
print("fit_lik=False:", out["fit_lik_false"], flush=True)

# 2. _robust_gp_fit_ failure counts
orig_fit = gpyreg.GP.fit
orig_robust = gtrain._robust_gp_fit_
current = {"call": None}
calls = []


def counting_fit(self, *a, **k):
    try:
        return orig_fit(self, *a, **k)
    except np.linalg.LinAlgError:
        if current["call"] is not None:
            current["call"]["fails"] += 1
        raise


def counting_robust(*a, **k):
    current["call"] = {"fails": 0, "outcome": "ok"}
    try:
        return orig_robust(*a, **k)
    except Exception as e:  # noqa: BLE001
        current["call"]["outcome"] = f"{type(e).__name__}: {str(e)[:120]}"
        raise
    finally:
        calls.append(current["call"])
        current["call"] = None


gpyreg.GP.fit = counting_fit
gtrain._robust_gp_fit_ = counting_robust

runs = []
for cfg in bt.suite_configs("default"):
    for seed in range(10):
        n0 = len(calls)
        args, options = cfg.make(seed=seed).bads_args()
        try:
            BADS(*args, options=options).optimize()
            outcome = "finished"
        except Exception as e:  # noqa: BLE001
            outcome = f"{type(e).__name__}: {str(e)[:120]}"
        run_calls = calls[n0:]
        runs.append(
            {
                "label": cfg.label,
                "seed": seed,
                "outcome": outcome,
                "robust_calls": len(run_calls),
                "max_fails": max((c["fails"] for c in run_calls), default=0),
                "call_outcomes": sorted({c["outcome"] for c in run_calls}),
            }
        )
        print(
            cfg.label,
            seed,
            outcome[:60],
            "calls",
            len(run_calls),
            "max_fails",
            runs[-1]["max_fails"],
            flush=True,
        )

hist = {}
for c in calls:
    hist[c["fails"]] = hist.get(c["fails"], 0) + 1
out["robust_fit"] = {
    "n_calls": len(calls),
    "fails_histogram": {str(k): v for k, v in sorted(hist.items())},
    "call_outcomes": sorted({c["outcome"] for c in calls}),
    "runs_not_finished": [r for r in runs if r["outcome"] != "finished"],
    "runs": runs,
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(
    "histogram of LinAlgError per _robust_gp_fit_ call:",
    out["robust_fit"]["fails_histogram"],
    flush=True,
)
print(
    "runs not finished:",
    len(out["robust_fit"]["runs_not_finished"]),
    flush=True,
)
