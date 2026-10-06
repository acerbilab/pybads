"""B1 verifier: options. tol_noise (K4, C-F3); descriptions cut at '=' or
':' (K6); options without a description and stray quotes (K7);
search_n_try's type (K8); display levels (I-F11, C-F11); user None and
MATLAB-style strings (I-F13, C-F8); MaxFunEvals and ImprovementQuantile
checks (C-F10)."""
import logging
import os

import common  # noqa: F401  (banner)
import numpy as np

from pybads import BADS

ini_dir = "/home/user/pybads-review/pybads/bads/option_configs"


def make(opts=None, D=2, x0=None):
    o = {"display": "off", "random_seed": 0}
    if opts:
        o.update(opts)
    return BADS(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        np.zeros(D) + 0.3 if x0 is None else x0,
        -np.ones(D) * 5,
        np.ones(D) * 5,
        -np.ones(D) * 2,
        np.ones(D) * 2,
        options=o,
    )


b = make()
print(
    "tol_fun =",
    b.options["tol_fun"],
    " tol_noise =",
    b.options["tol_noise"],
    " MATLAB sqrt(eps)*TolFun =",
    np.sqrt(np.spacing(1.0)) * 1e-3,
)

# K6: descriptions against the last comment line above each option
print("\n-- K6: Options.descriptions against the raw comment line")
raw = {}
for f in ("basic_bads_options.ini", "advanced_bads_options.ini"):
    last = ""
    for line in open(os.path.join(ini_dir, f)):
        s = line.strip()
        if s.startswith("#"):
            last = s.lstrip("#").strip()
        elif "=" in s and not s.startswith("["):
            k = s.split("=")[0].strip()
            raw[k] = last
            last = ""
        elif s == "":
            pass
cut = [
    (k, raw[k], b.options.descriptions.get(k))
    for k in raw
    if raw[k] and raw[k].strip("# ") != b.options.descriptions.get(k)
]
for k, r, d in cut:
    print(f"  {k}:\n    ini : {r}\n    desc: {d}")
print(
    "  str(options) line for noise_size:",
    [l for l in str(b.options).splitlines() if l.startswith("noise_size")],
)

# K7: options without a description, and descriptions ending in a quote
print("\n-- K7")
nodesc = [k for k in raw if not raw[k]]
print("  options with no description line:", nodesc)
quoted = [k for k in raw if raw[k].endswith("'") or raw[k].endswith("';")]
print(f"  descriptions ending in a stray quote: {len(quoted)}:", quoted)

# K8
print("\n-- K8: search_n_try at D = 1, 2, 6, 20")
for D in (1, 2, 6, 20):
    bb = make(D=D)
    v = bb.options["search_n_try"]
    print(
        f"  D={D}: {v!r} type {type(v).__name__}; optim_state['search_count']"
        f" = {bb.optim_state['search_count']!r}"
    )

# display levels
print("\n-- display levels -> BADS logger level (INFO=20, WARN=30, DEBUG=10)")
for d in ("off", "iter", "full", "final", "notify", "none", "OFF", "Iter"):
    bb = make({"display": d})
    print(f"  display={d!r}: level {bb.logger.level}")

# user None and MATLAB-style strings
print("\n-- user None / strings")
for k, v in [
    ("nonlinear_scaling", None),
    ("nonlinear_scaling", "off"),
    ("uncertainty_handling", "off"),
    ("uncertainty_handling", "no"),
    ("tol_mesh", None),
    ("max_fun_evals", None),
    ("complete_poll", "off"),
]:
    try:
        bb = make({k: v}, D=1, x0=np.array([2.0]))
        bb.lower_bounds  # noqa
        note = ""
        if k == "nonlinear_scaling":
            note = f"log flag {bb.var_transf.apply_log_t.ravel()}"
        if k == "uncertainty_handling":
            note = (
                "uncertainty level after init: "
                f"{bb.optim_state['uncertainty_handling_level']}"
            )
        if k in ("max_fun_evals", "complete_poll"):
            bb.options["max_fun_evals"] = (
                bb.options["max_fun_evals"] if v is not None else None
            )
            try:
                r = bb.optimize()
                note = f"optimize ran, func_count {r['func_count']}"
            except Exception as e:
                note = f"optimize raised {type(e).__name__}: {str(e)[:60]}"
        print(f"  {k}={v!r}: constructed; {note}")
    except Exception as e:
        print(f"  {k}={v!r}: {type(e).__name__}: {str(e)[:70]}")

# nonlinear_scaling on a log-eligible variable, default vs None
for v in ("default", None, "off", False):
    o = {} if v == "default" else {"nonlinear_scaling": v}
    bb = BADS(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        np.array([2.0]),
        np.array([0.1]),
        np.array([100.0]),
        np.array([0.5]),
        np.array([50.0]),
        options={"display": "off", "random_seed": 0, **o},
    )
    print(
        f"  nonlinear_scaling={v!r}: log flag "
        f"{bb.var_transf.apply_log_t.ravel()}"
    )

# C-F10
print("\n-- C-F10: max_fun_evals and improvement_quantile checks")
for k, v in [
    ("max_fun_evals", 0),
    ("max_fun_evals", -5),
    ("max_fun_evals", 30.5),
    ("improvement_quantile", 0.9),
]:
    try:
        bb = make({k: v})
        logging.getLogger("BADS").setLevel(logging.ERROR)
        r = bb.optimize()
        print(
            f"  {k}={v}: ran, func_count {r['func_count']}, "
            f"message {r['message'][:60]!r}"
        )
    except Exception as e:
        print(f"  {k}={v}: {type(e).__name__}: {str(e)[:70]}")
