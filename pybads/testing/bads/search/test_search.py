import logging

import gpyreg as gpr
import numpy as np
import pytest

import pybads.bads.bads as bads_module
import pybads.search.es_search as es_search_module
from pybads import BADS
from pybads.acquisition_functions import acq_fcn_lcb
from pybads.bads.gaussian_process_train import get_grid_search_neighbors
from pybads.bads.option_configs import get_pybads_option_dir_path
from pybads.bads.options import Options
from pybads.function_examples import rosenbrocks_fcn
from pybads.function_logger import FunctionLogger, contraints_check
from pybads.search.es_search import ESSearchELL, ESSearchWM, ucov
from pybads.search.grid_functions import force_to_grid
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


def _es_update_select_mask(mu, lamb):
    """The selection mask of MATLAB's ESupdate.m, transcribed: its 1-based
    parent indices, and the numbers of offspring of the parents."""
    tot = mu + lamb
    s = 1.0 / np.sqrt(np.arange(1, tot + 1))
    w = np.ceil(s / np.sum(s) * lamb).astype(int)
    nonzero = np.sum(w > 0)
    while np.sum(w) - lamb > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = np.sum(w) - lamb
    last = np.flatnonzero(w > 0)[-1] + 1  # find(w > 0, 1, 'last')
    w[max(1, last - delta + 1) - 1 : last] -= 1
    cw = np.cumsum(w) - w + 1
    idx = np.zeros(np.max(cw), dtype=int)
    idx[cw - 1] = 1  # idx(cw) = 1
    return np.cumsum(idx[:-1]), w


@pytest.mark.parametrize(
    "mu, lamb", [(1, 2048), (2048, 2048), (100, 2048), (7, 20), (3, 3)]
)
def test_search_selection_mask(mu, lamb):
    """Each parent has the offspring that MATLAB's ESupdate.m gives it: the
    selection mask is MATLAB's, 1-based, minus one."""
    D = 3
    options = load_options(
        D,
        get_pybads_option_dir_path(),
    )
    search_es = ESSearchWM(mu, lamb, options, rng=np.random.default_rng(0))
    mask = search_es._get_selection_idx_mask_(mu, lamb)
    select_mask, w = _es_update_select_mask(mu, lamb)
    assert np.array_equal(np.bincount(mask, minlength=w.size), w)
    assert np.array_equal(mask, select_mask - 1)


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


def test_hedge_gamma_zero_scores_each_search_at_the_search_point(
    monkeypatch,
):
    """With `hedge_gamma = 0` the hedge rewards every search, the searches not
    chosen at the GP's prediction at the search point, taken as a row (MATLAB
    BADS's intent): the run completes."""
    points = []
    original_update = ESSearchHedge.update_hedge

    def update(self, u_search, *args, **kwargs):
        points.append(u_search)
        return original_update(self, u_search, *args, **kwargs)

    monkeypatch.setattr(ESSearchHedge, "update_hedge", update)
    D = 3
    bads = BADS(
        rosenbrocks_fcn,
        np.zeros((1, D)),
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={
            "random_seed": 0,
            "display": "off",
            "max_fun_evals": 60,
            "hedge_gamma": 0,
        },
    )
    result = bads.optimize()
    assert len(points) > 1
    assert np.isfinite(result["fval"])
    assert np.all(np.isfinite(bads.search_es_hedge.g))


class _FixedGP:
    """A stand-in for a GP, with fixed predictions at any point."""

    f_mu = np.array([[1.0], [2.0]])
    f_s2 = np.array([[4.0], [0.25]])

    def predict(self, x):
        return self.f_mu.copy(), self.f_s2.copy()


@pytest.mark.parametrize(
    "sqrt_beta",
    [2.0, 2, np.float64(2.0), np.int64(2), np.array(2.0), np.array([2.0])],
    ids=["float", "int", "np.float64", "np.int64", "0-d", "1-element"],
)
def test_lcb_accepts_a_positive_number_as_sqrt_beta(sqrt_beta):
    """`sqrt_beta` is a positive finite real number: a Python or NumPy
    scalar, or an array of one element."""
    xi = np.zeros((2, 3))
    z, f_mu, f_s = acq_fcn_lcb(xi, 9, _FixedGP(), sqrt_beta)
    assert np.array_equal(f_s, np.array([[2.0], [0.5]]))
    assert np.array_equal(z, np.array([[-3.0], [1.0]]))


def test_lcb_sqrt_beta_schedule_and_callable():
    """`sqrt_beta=None` is the schedule of MATLAB BADS's acqLCB, and a
    callable is called with the evaluation count plus one and D."""
    xi = np.zeros((2, 3))
    z, _, _ = acq_fcn_lcb(xi, 9, _FixedGP())
    t, n_vars = 10, 3
    sqrt_beta = np.sqrt(0.4 * np.log(n_vars * t**2 * np.pi**2 / 0.6))
    assert np.allclose(z, _FixedGP.f_mu - sqrt_beta * np.sqrt(_FixedGP.f_s2))
    calls = []

    def schedule(t, n_vars):
        calls.append((t, n_vars))
        return 1.5

    z, _, _ = acq_fcn_lcb(xi, 9, _FixedGP(), schedule)
    assert calls == [(10, 3)]
    assert np.array_equal(z, np.array([[-2.0], [1.25]]))


@pytest.mark.parametrize(
    "sqrt_beta",
    [
        0.0,
        -1.0,
        np.float64(-1.0),
        np.inf,
        np.nan,
        "acq_schedule",
        np.array([1.0, 2.0]),
        np.array([]),
        True,
        np.complex128(2.0),
    ],
    ids=[
        "zero",
        "negative",
        "np.float64 negative",
        "inf",
        "nan",
        "name",
        "2 elements",
        "empty",
        "bool",
        "complex",
    ],
)
def test_lcb_refuses_other_values_of_sqrt_beta(sqrt_beta):
    """Any other `sqrt_beta` is refused with a message that says what is
    accepted."""
    with pytest.raises(ValueError, match="positive finite real number"):
        acq_fcn_lcb(np.zeros((2, 3)), 9, _FixedGP(), sqrt_beta)


def test_force_to_grid_rounds_halves_away_from_zero():
    """`force_to_grid` rounds as MATLAB's `round` in `force2grid.m` does:
    halves away from zero, where `np.round` takes them to the even
    integer."""
    q = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49, 0.51, -0.51, 3.0])
    expected = np.array([1, 2, 3, -1, -2, -3, 0, 1, -1, 3])
    tol = 2.0**-10
    assert np.array_equal(force_to_grid(q * tol, tol) / tol, expected)
    assert np.array_equal(force_to_grid(q, 0.1, tol=1.0), expected)
    # The largest double below one half, which floor(|q| + 0.5) takes to 1
    below_half = np.nextafter(0.5, 0.0)
    assert np.array_equal(
        force_to_grid(np.array([below_half, -below_half]), 1.0), [0, 0]
    )


def test_start_on_a_half_goes_to_matlabs_grid_point():
    """`x0 = [1, 3]` in the plausible box `[-2048, 2048]` lies at 0.5 and
    1.5 search meshes from the origin, and starts at `[2, 4]`, as in MATLAB
    BADS, not at `[0, 4]`."""
    bads = BADS(
        lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
        np.array([1.0, 3.0]),
        np.full(2, -4096.0),
        np.full(2, 4096.0),
        np.full(2, -2048.0),
        np.full(2, 2048.0),
        options={"display": "off", "random_seed": 0},
    )
    search_mesh_size = bads.optim_state["search_mesh_size"]
    u_x0 = bads.var_transf(np.array([1.0, 3.0]))
    np.testing.assert_allclose(
        np.ravel(u_x0) / search_mesh_size, [0.5, 1.5], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.ravel(bads.u) / search_mesh_size, [1.0, 2.0], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.ravel(bads.var_transf.inverse_transf(np.atleast_2d(bads.u))),
        [2.0, 4.0],
        rtol=1e-12,
    )


def _initial_state(D=3, **options):
    """A BADS object and its GP after the initial design, for a unit call of
    the ES search."""
    bads = BADS(
        rosenbrocks_fcn,
        np.zeros((1, D)),
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={"random_seed": 0, "display": "off", **options},
    )
    bads.options["fun_eval_start"] = 10
    gp, _, _, _ = bads._init_optimization_()
    return bads, gp


def test_es_scale_follows_the_fraction_of_new_candidates(monkeypatch):
    """From the second generation of the ES search, the scale follows the
    fraction of the generation's candidates among the best ntest of the
    candidates kept before it and its own, ntest the smaller of their two
    numbers, as in MATLAB's searchES."""
    n_search_iter = 4
    bads, gp = _initial_state(n_search_iter=n_search_iter)
    values = np.random.default_rng(1)
    generations = []

    def lcb(u, *args, **kwargs):
        # Each generation is better than the one before, on average
        z = values.normal(-0.3 * len(generations), 1.0, size=(len(u), 1))
        generations.append(z.ravel())
        return z, z, np.zeros_like(z)

    monkeypatch.setattr(es_search_module, "acq_fcn_lcb", lcb)
    mu = int(bads.options["n_search"] / n_search_iter)
    search_es = ESSearchWM(mu, mu, bads.options, rng=bads.rng)
    search_es(
        bads.u,
        None,
        None,
        bads.function_logger,
        gp,
        bads.optim_state,
        True,
        None,
    )
    assert len(generations) == n_search_iter

    kept = np.empty(0)
    log_scale = 0.0
    for i, z_new in enumerate(generations):
        ntest = min(z_new.size, kept.size)
        pool = np.concatenate((kept, z_new))
        best = np.sort(pool)[:ntest]
        if 0 < i < n_search_iter - 1:
            frac = np.sum(np.isin(best, z_new)) / ntest
            log_scale += bads.options["es_beta"] * (frac - 0.2)
        kept = np.sort(pool)[:mu]
    assert np.isclose(
        search_es.scale, bads.options["es_start"] * np.exp(log_scale)
    )


def test_es_search_adds_no_handler_to_the_root_logger():
    """Constructing an ES search leaves the root logger as it was, which
    BADS.__init__ configures."""
    root = logging.getLogger()
    handlers = root.handlers[:]
    for handler in handlers:
        root.removeHandler(handler)
    try:
        ESSearchWM(
            1,
            2048,
            load_options(3, get_pybads_option_dir_path()),
            rng=np.random.default_rng(0),
        )
        added = root.handlers[:]
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
    assert added == []


def test_es_wcm_takes_one_best_point_per_weight(monkeypatch):
    """ES-wcm computes its covariance from the floor(mu) best training
    points, mu half their number, one per weight, as in MATLAB's
    searchES."""
    bads, gp = _initial_state()
    calls = []
    original_ucov = es_search_module.ucov

    def ucov(U, u, w, *args, **kwargs):
        calls.append((U.copy(), w.copy()))
        return original_ucov(U, u, w, *args, **kwargs)

    monkeypatch.setattr(es_search_module, "ucov", ucov)
    search_es = ESSearchWM(1, 1, bads.options, rng=bads.rng)
    search_es._initialize_(bads.u, gp, bads.optim_state, True)
    ((U_best, weights),) = calls
    n_best = int(np.floor(0.5 * gp.X.shape[0]))
    assert U_best.shape[0] == n_best == weights.size
    best = np.argsort(gp.y.ravel(), kind="stable")[:n_best]
    assert np.array_equal(U_best, gp.X[best])
