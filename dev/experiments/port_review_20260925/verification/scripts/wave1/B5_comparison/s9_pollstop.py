"""How often the poll's stop decision would change if the reliability flag
were MATLAB's (gppredcheck on the same stats, same SD) instead of the
port's, along default runs."""
import matlab_ref as M
import numpy as np
import s4_runs as R

from pybads.bads.bads import BADS

orig_refit = BADS._is_gp_refit_time_
orig_stop = BADS._is_poll_stop_
last = {}
cnt = {"steps": 0, "differ": 0, "differ_n": {}}


def is_refit(self, alpha):
    st = self.gp_stats
    f = [] if st.get("iter_gp") is None else [float(v) for v in st.get("fval")]
    m = [] if st.get("iter_gp") is None else [float(v) for v in st.get("ymu")]
    s = [] if st.get("iter_gp") is None else [float(v) for v in st.get("ys")]
    try:
        u = M.gppredcheck(f, m, s, alpha)
    except Exception:
        u = True
    r, up = orig_refit(self, alpha)
    last["py"], last["mat"], last["n"] = bool(up), (False if r else u), len(f)
    return r, up


def logic(self, good, unrel, p_less, poll_count):
    if good:
        return True if unrel else p_less > 1 - self.options["tol_poi"]
    return (
        not unrel
        and (
            self.options["consecutive_skipping"]
            or self.last_skipped < self.optim_state["iter"] - 1
        )
        and poll_count >= self.options["min_failed_poll_steps"]
        and p_less > 1 - self.options["tol_poi"]
    )


def stop(self, good, unrel, p_less, poll_count):
    cnt["steps"] += 1
    if bool(unrel) == last["py"] and last["py"] != last["mat"]:
        a = logic(self, good, last["py"], p_less, poll_count)
        b = logic(self, good, last["mat"], p_less, poll_count)
        if a != b:
            cnt["differ"] += 1
            cnt["differ_n"][last["n"]] = cnt["differ_n"].get(last["n"], 0) + 1
    return orig_stop(self, good, unrel, p_less, poll_count)


BADS._is_gp_refit_time_ = is_refit
BADS._is_poll_stop_ = stop
for name, f, D, x0 in (
    ("rosen3", R.rosen, 3, -1.5),
    ("ellip4", R.ellip, 4, 2.0),
):
    for seed in range(3):
        for k in ("steps", "differ"):
            cnt[k] = 0
        cnt["differ_n"] = {}
        b = BADS(
            f,
            np.full((1, D), x0),
            np.full((1, D), -5.0),
            np.full((1, D), 5.0),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 200,
            },
        )
        b.optimize()
        print(
            f"{name} seed={seed}: poll-stop checks={cnt['steps']}, decision would differ with MATLAB's flag: {cnt['differ']} (by n stats: {cnt['differ_n']})"
        )
