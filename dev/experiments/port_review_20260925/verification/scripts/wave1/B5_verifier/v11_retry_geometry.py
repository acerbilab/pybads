"""F11 (internal) / F10 (comparison) / B5-R1: local_gp_fitting after the posterior update of a refit fails once."""
import copy

import common  # noqa
import gpyreg as gpr
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt


def f(x):
    x = np.ravel(x)
    return float(np.sum((x - 0.3) ** 2 * np.array([1.0, 30.0, 400.0])))


D = 3
b = BADS(
    f,
    np.array([[1.0, -1.0, 0.5]]),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options={"random_seed": 8, "display": "off", "max_fun_evals": 60},
)
b.optimize()
orig_update = gpr.GP.update


def run(n_fail_first_updates, refit):
    gp = copy.deepcopy(b.iteration_history["gp"][b.optim_state["iter"]])
    gp.temporary_data["len_scale"] = np.ones(
        D
    )  # make the entry geometry distinguishable
    entry_hyp = gp.get_hyperparameters(as_array=True).copy()
    C = {"n": 0}

    def upd(
        self,
        X_new=None,
        y_new=None,
        s2_new=None,
        hyp=None,
        compute_posterior=True,
    ):
        if (
            self is gp
            and compute_posterior
            and X_new is None
            and hyp is not None
        ):
            C["n"] += 1
            if C["n"] <= n_fail_first_updates:
                raise np.linalg.LinAlgError("injected")
        return orig_update(self, X_new, y_new, s2_new, hyp, compute_posterior)

    gpr.GP.update = upd
    try:
        out, flag = gpt.local_gp_fitting(
            gp,
            b.u,
            b.function_logger,
            b.options,
            b.optim_state,
            b.iteration_history,
            refit,
            rng=np.random.default_rng(2),
        )
    finally:
        gpr.GP.update = orig_update
    held = out.get_hyperparameters()[0]
    ll_held = np.exp(held["covariance_log_lengthscale"])
    # poll_scale recomputed from the hyperparameters the GP holds (the formula of local_gp_fitting)
    l = b.options["gp_rescale_poll"] * held["covariance_log_lengthscale"]
    ps_held = np.minimum(
        np.maximum(np.exp(l - np.mean(l)), b.optim_state["search_mesh_size"]),
        (b.optim_state["ub"] - b.optim_state["lb"]).ravel()
        / b.optim_state["scale"],
    )
    a = np.exp(held["covariance_log_shape"])[0]
    print(
        f"--- refit={refit}, failing posterior updates={n_fail_first_updates}: exit flag {flag}, "
        f"hyp back to entry: {np.allclose(out.get_hyperparameters(as_array=True), entry_hyp)}, "
        f"markers {[k for k in ('needs_rebuild', 'needs_refit') if k in out.temporary_data]}, calls {C['n']}"
    )
    print(
        f"    held lengthscale   {np.round(ll_held, 4)}   temporary_data len_scale {np.round(out.temporary_data['len_scale'], 4)}"
    )
    print(
        f"    poll_scale from held hyp {np.round(ps_held, 4)}   temporary_data poll_scale {np.round(out.temporary_data['poll_scale'], 4)}"
    )
    print(
        f"    eff. radius from held hyp {np.sqrt(a * (np.exp(1 / a) - 1)):.4f}   temporary_data {np.ravel(out.temporary_data['effective_radius'])}"
    )
    print(
        f"    training rows {out.X.shape[0]} (entry {b.iteration_history['gp'][b.optim_state['iter']].X.shape[0]}), posterior ok: {out.posteriors[0].L is not None}"
    )


run(0, True)
run(1, True)
run(1, False)
run(2, False)
