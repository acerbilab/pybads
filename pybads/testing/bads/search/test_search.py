import gpyreg as gpr
import numpy as np
import pytest

import pybads.bads.bads as bads_module
from pybads import BADS
from pybads.bads.gaussian_process_train import get_grid_search_neighbors
from pybads.bads.option_configs import get_pybads_option_dir_path
from pybads.bads.options import Options
from pybads.function_examples import rosenbrocks_fcn
from pybads.function_logger import FunctionLogger, contraints_check
from pybads.search.es_search import ESSearchELL, ESSearchWM, ucov
from pybads.search.search_hedge import ESSearchHedge


def test_incumbent_constraint_check():
    D = 3
    U = np.random.normal(size=(10, D))
    # check duplicates
    U = np.unique(U, axis=0)
    lb = np.array([[-5] * D]) * 100
    ub = np.array([[5] * D]) * 100
    f = FunctionLogger(rosenbrocks_fcn, D, False, 0)
    for i in range(len(U)):
        y, y_sd, idx_y = f(U[i])

    U = np.vstack((U, U[-1]))  # add duplicate
    # Every row of U is already evaluated, and contraints_check removes none
    # of them, only the duplicate: MATLAB's uCheck would remove them all (a
    # candidate defect, in dev/results/2026-09-23-codebase-survey.md).
    U_new = contraints_check(U, lb, ub, 1e-6, f, True)
    assert U_new.size != U.size
    assert U_new.shape[0] == U.shape[0] - 1

    # Check outliers and project them
    lb = np.array([[-0.5] * D])
    ub = np.array([[0.5] * D])
    U = np.vstack((U, np.array([[1] * D, [1] * D])))
    outbounds = (U < lb) | (U > ub)
    assert np.any(outbounds)
    U_new = contraints_check(U, lb, ub, 1e-6, f, True)
    inbounds = np.all(U_new >= lb) & np.all(U_new <= ub)
    assert inbounds


def load_options(D, path_dir):
    """Load basic and advanced options and validate the names"""
    pybads_path = path_dir
    basic_path = pybads_path + "/basic_bads_options.ini"
    options = Options(
        basic_path,
        evaluation_parameters={"D": D},
        user_options=None,
    )
    advanced_path = pybads_path + "/advanced_bads_options.ini"
    options.load_options_file(
        advanced_path,
        evaluation_parameters={"D": D},
    )
    options.validate_option_names([basic_path, advanced_path])
    return options


def test_search():
    x0 = np.array([[0, 0, 0]])
    # Starting point
    lb = np.array([[-20, -20, -20]])  # Lower bounds
    ub = np.array([[20, 20, 20]])  # Upper bounds
    plb = np.array([[-5, -5, -5]])  # Plausible lower bounds
    pub = np.array([[5, 5, 5]])  # Plausible upper bounds
    D = 3
    bads = BADS(
        rosenbrocks_fcn, x0, lb, ub, plb, pub, options={"random_seed": 0}
    )
    bads.options["fun_eval_start"] = 10
    gp, Ns_gp, sn2hpd, hyp_dict = bads._init_optimization_()

    es_iter = bads.options["n_search_iter"]
    mu = int(bads.options["n_search"] / es_iter)
    lamb = mu
    search_es = ESSearchWM(mu, lamb, bads.options, rng=bads.rng)
    us, z = search_es(
        bads.u, lb, ub, bads.function_logger, gp, bads.optim_state, True, None
    )

    assert us.size == 3 and (np.isscalar(z) or z.size == 1)
    assert np.all(gp.y >= z)

    search_es = ESSearchELL(mu, lamb, bads.options, rng=bads.rng)
    us, z = search_es(
        bads.u, lb, ub, bads.function_logger, gp, bads.optim_state, True, None
    )
    assert us.size == 3 and (np.isscalar(z) or z.size == 1)
    assert np.all(gp.y >= z)


def test_search_selection_mask():
    D = 3
    mu = 1
    lamb = 2048
    options = load_options(
        D,
        get_pybads_option_dir_path(),
    )
    search_es = ESSearchWM(mu, lamb, options, rng=np.random.default_rng(0))
    mask = search_es._get_selection_idx_mask_(mu, lamb)
    assert np.sum(mask) == 885072
    assert np.min(mask + 1) == 1


def test_search_hedge():
    x0 = np.array([[0, 0, 0]])
    # Starting point
    lb = np.array([[-20, -20, -20]])  # Lower bounds
    ub = np.array([[20, 20, 20]])  # Upper bounds
    plb = np.array([[-5, -5, -5]])  # Plausible lower bounds
    pub = np.array([[5, 5, 5]])  # Plausible upper bounds
    D = 3

    bads = BADS(
        rosenbrocks_fcn, x0, lb, ub, plb, pub, options={"random_seed": 0}
    )
    bads.options["fun_eval_start"] = 10
    gp, Ns_gp, sn2hpd, hyp_dict = bads._init_optimization_()

    search_hedge = ESSearchHedge(
        bads.options["search_method"], bads.options, rng=bads.rng
    )

    us, z = search_hedge(
        bads.u, lb, ub, bads.function_logger, gp, bads.optim_state
    )
    print(search_hedge.chosen_search_fun)
    assert us.size == 3 and (np.isscalar(z) or z.size == 1)
    assert np.all(gp.y >= z)


def test_u_cov():
    U = np.array(
        [
            [0, 0, 0],
            [0.1172, 0.1328, 0.6641],
            [0, 0, 1],
            [0, 0, -1],
            [0.6172, -0.3672, 0.1641],
        ]
    )
    u0 = np.array([[0.0, 0.0, 0.0]])
    ub = np.array([[4.0, 4.0, 4.0]])
    lb = -ub
    w = np.array([0.4563, 0.2708, 0.1622, 0.0852, 0.0255])
    C = ucov(U, u0, w, ub, lb, 1)
    assert C.shape == (U.shape[1], U.shape[1])


def test_grid_search_neighbors():
    x0 = np.array([[0, 0]])
    # Starting point
    lb = np.array([[-20, -20]])  # Lower bounds
    ub = np.array([[20, 20]])  # Upper bounds
    plb = np.array([[-5, -5]])  # Plausible lower bounds
    pub = np.array([[5, 5]])  # Plausible upper bounds
    D = 2

    bads = BADS(
        rosenbrocks_fcn, x0, lb, ub, plb, pub, options={"random_seed": 0}
    )
    bads.options["fun_eval_start"] = 10
    gp, Ns_gp, sn2hpd, hyp_dict = bads._init_optimization_()
    gp.X = np.array([[0, 0], [-0.1055, 0.4570], [-0.3555, -0.7930]])
    f = FunctionLogger(rosenbrocks_fcn, D, False, 0)
    f.X = gp.X.copy()
    gp.y = np.array([1, 405.1637, 5.082e3])
    f.Y = gp.y.copy()

    f.X_max_idx = 3

    gp.temporary_data["len_scale"] = 1.0
    bads.optim_state["scale"] = 1.0
    bads.options["gp_radius"] = 3

    result = get_grid_search_neighbors(
        f, np.array([[0, 0]]), gp, bads.options, bads.optim_state
    )[0]
    assert (
        result[0, 0] == 0.0
        and np.isclose(result[1, 0], -0.1055, 1e-3)
        and np.isclose(result[2, 0], -0.3555, 1e-3)
    )


def test_last_search_of_a_round_adds_no_point_to_the_gp(monkeypatch):
    """A search adds its point to the GP, except the last search of a round,
    before the poll rebuilds the GP, as MATLAB BADS does: the round's
    `search_count` decides, not the hedge's count over the run."""
    searches = []
    original_step = BADS._search_step_
    original_add = bads_module.add_and_update_gp

    def step(self, gp):
        func_count = self.function_logger.func_count
        searches.append({"added": False})
        try:
            return original_step(self, gp)
        finally:
            searches[-1]["evaluated"] = (
                self.function_logger.func_count > func_count
            )
            searches[-1]["count"] = self.optim_state["search_count"]

    def add(*args, **kwargs):
        if searches and "count" not in searches[-1]:
            searches[-1]["added"] = True
        return original_add(*args, **kwargs)

    monkeypatch.setattr(BADS, "_search_step_", step)
    monkeypatch.setattr(bads_module, "add_and_update_gp", add)
    D = 3
    bads = BADS(
        rosenbrocks_fcn,
        np.zeros((1, D)),
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={"random_seed": 0, "display": "off", "max_fun_evals": 100},
    )
    bads.optimize()
    n_try = bads.options["search_n_try"]
    evaluated = [s for s in searches if s["evaluated"]]
    assert any(s["count"] == n_try for s in evaluated)
    assert any(s["count"] < n_try for s in evaluated)
    for s in evaluated:
        assert s["added"] == (s["count"] < n_try)


def test_failed_searches_floor_the_search_factor():
    """A failed search shrinks the search factor by `search_scale_failure`,
    but not below `search_factor_min`, as in MATLAB BADS; a successful or
    incremental search scales it with no floor, and the end of a round
    resets it to 1."""
    D = 6
    bads = BADS(
        rosenbrocks_fcn,
        np.zeros((1, D)),
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={"random_seed": 0, "display": "off"},
    )
    options = bads.options
    n_try = int(options["search_n_try"])
    assert n_try == 6
    # A round of failed searches, and the factor each of them runs at
    for count in range(1, n_try + 1):
        bads.optim_state["search_count"] = count
        bads._update_search_stats_("failure", 0.0)
    factors = np.exp(bads.optim_state["search_stats"]["log_search_factor"])
    assert np.allclose(
        factors,
        np.maximum(
            options["search_factor_min"],
            options["search_scale_failure"] ** np.arange(n_try),
        ),
    )
    assert np.isclose(factors[-1], options["search_factor_min"])
    assert bads.optim_state["search_factor"] == 1

    bads.optim_state["search_count"] = 1
    bads.optim_state["search_factor"] = 0.25
    bads._update_search_stats_("success", 0.0)
    assert np.isclose(
        bads.optim_state["search_factor"],
        0.25 * options["search_scale_success"],
    )
    bads.optim_state["search_factor"] = 0.125
    bads._update_search_stats_("incremental", 0.0)
    assert np.isclose(
        bads.optim_state["search_factor"],
        0.125 * options["search_scale_incremental"],
    )
