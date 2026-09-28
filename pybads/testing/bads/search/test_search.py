import logging
import sys
from types import SimpleNamespace

import gpyreg as gpr
import numpy as np
import pytest
from scipy.stats import norm

import pybads.bads.bads as bads_module
import pybads.search.es_search as es_search_module
from pybads import BADS
from pybads.acquisition_functions import acq_fcn_lcb
from pybads.bads.gaussian_process_train import get_grid_search_neighbors
from pybads.bads.option_configs import get_pybads_option_dir_path
from pybads.bads.options import Options
from pybads.function_examples import rosenbrocks_fcn
from pybads.function_logger import FunctionLogger, contraints_check
from pybads.function_logger.constraints_check import _lexsort_rows
from pybads.rounding import round_half_away
from pybads.search.es_search import ESSearchELL, ESSearchWM, ucov
from pybads.search.grid_functions import force_to_grid, udist
from pybads.search.search_hedge import ESSearchHedge


def test_incumbent_constraint_check():
    D = 3
    U = np.random.default_rng(0).normal(size=(10, D))
    # check duplicates
    U = np.unique(U, axis=0)
    lb = np.array([[-5] * D]) * 100
    ub = np.array([[5] * D]) * 100
    f = FunctionLogger(rosenbrocks_fcn, D, False, 0)
    for i in range(len(U)):
        y, y_sd, idx_y = f(U[i])

    U = np.vstack((U, U[-1]))  # add duplicate
    # Every row of U is already evaluated: contraints_check removes them
    # all, as MATLAB's uCheck
    U_new = contraints_check(U, lb, ub, 1e-6, f, True)
    assert U_new.shape == (0, D)

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


@pytest.mark.parametrize(
    "n_search_iter, ns",
    [(2, [1024, 1024]), (3, [683, 682]), (7, [293, 292]), (4096, [1, 0])],
)
def test_es_search_splits_its_first_population_as_matlab(n_search_iter, ns):
    """The ES search splits its first population between its two scales with
    MATLAB's round, as searchES.m does, which takes a half away from zero:
    the 1365 points of `n_search_iter = 3` give 683 and 682."""
    options = load_options(3, get_pybads_option_dir_path())
    mu = int(options["n_search"] / n_search_iter)
    search_es = ESSearchWM(mu, mu, options, rng=np.random.default_rng(0))
    assert np.array_equal(search_es.ns, ns)
    assert np.array_equal(np.ravel(search_es.vec), np.repeat(search_es.w, ns))


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


def _brute_udist(U, u2, len_scale, lb, ub, mask):
    """Squared distances, the periodic differences the shorter way round,
    one pair at a time."""
    period = (ub - lb).ravel()
    out = np.zeros((U.shape[0], u2.shape[0]))
    for i, a in enumerate(U):
        for j, b in enumerate(u2):
            d = np.abs(a - b)
            d[mask] = np.minimum(d[mask], period[mask] - d[mask])
            out[i, j] = np.sum((d / len_scale) ** 2)
    return out


def test_udist_periodic_takes_the_shorter_way_round():
    """Along a periodic variable `udist` takes each difference the shorter
    way round the period, per coordinate, in units of the length scales,
    as MATLAB's udist.m; the other coordinates are as without periodic
    variables."""
    rng = np.random.default_rng(0)
    lb = np.array([[-1.0, -3.0, -2.0]])
    ub = np.array([[1.0, 3.0, 2.0]])
    mask = np.array([True, False, True])
    U = rng.uniform(lb, ub, size=(7, 3))
    u2 = rng.uniform(lb, ub, size=(4, 3))
    len_scale = np.array([0.5, 2.0, 1.5])
    dist = udist(U, u2, len_scale, lb, ub, 1.0, mask[None, :])
    np.testing.assert_allclose(
        dist, _brute_udist(U, u2, len_scale, lb, ub, mask), rtol=1e-12
    )
    # Two points close across the bounds are close
    a = np.array([[-0.95, 0.0, 1.9]])
    b = np.array([[0.95, 0.0, -1.9]])
    np.testing.assert_allclose(
        udist(a, b, 1, lb, ub, 1.0, mask[None, :]), [[0.1**2 + 0.2**2]]
    )
    # Without periodic variables, the plain squared distances
    np.testing.assert_allclose(
        udist(U, u2, len_scale, lb, ub, 1.0, np.zeros((1, 3), dtype=bool)),
        _brute_udist(U, u2, len_scale, lb, ub, np.zeros(3, dtype=bool)),
        rtol=1e-12,
    )


def test_ucov_periodic_shifts_the_shorter_way_round():
    """`ucov` takes a periodic coordinate relative to the centre, the
    shorter way round its period, as MATLAB's ucov.m: points that lie close
    to the centre across the bounds give the covariance of the same points
    unwrapped."""
    lb = np.array([[-1.0, -4.0]])
    ub = np.array([[1.0, 4.0]])
    u0 = np.array([[0.95, 0.5]])
    offsets = np.array([[0.1, 0.2], [-0.05, -0.3], [0.2, 0.1]])
    w = np.array([0.5, 0.3, 0.2])
    unwrapped = u0 + offsets
    wrapped = unwrapped.copy()
    wrapped[:, 0] = np.where(
        wrapped[:, 0] >= 1.0, wrapped[:, 0] - 2.0, wrapped[:, 0]
    )
    C_periodic = ucov(wrapped, u0, w, ub, lb, 1, np.array([[True, False]]))
    C_plain = ucov(unwrapped, u0, w, ub, lb, 1)
    np.testing.assert_allclose(C_periodic, C_plain, rtol=1e-12, atol=1e-15)
    # The centre itself is left as it was
    np.testing.assert_array_equal(u0, [[0.95, 0.5]])


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


def test_search_scores_its_point_with_the_search_sqrt_beta(monkeypatch):
    """The search scores its chosen point with the LCB of `search_acq_fcn`,
    its `sqrt_beta` included, as MATLAB BADS applies `SearchAcqFcn`
    (`bads.m:578`)."""
    received = []
    original_lcb = bads_module.acq_fcn_lcb

    def lcb(xi, func_count, gp, sqrt_beta=None):
        if sys._getframe(1).f_code.co_name == "_search_step_":
            received.append(sqrt_beta)
        return original_lcb(xi, func_count, gp, sqrt_beta)

    monkeypatch.setattr(bads_module, "acq_fcn_lcb", lcb)
    D = 3
    sqrt_beta = lambda t, n_vars: 2.0
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
            "search_acq_fcn": ("acq_LCB", sqrt_beta),
        },
    )
    bads.optimize()
    assert len(received) > 0
    assert all(value is sqrt_beta for value in received)


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


@pytest.mark.parametrize(
    "value",
    [-1.0, 0, np.nan, np.inf, np.array([1.0, 2.0]), "1.5", None, True],
    ids=[
        "negative",
        "zero",
        "nan",
        "inf",
        "2 elements",
        "string",
        "None",
        "bool",
    ],
)
def test_lcb_refuses_a_callable_sqrt_beta_of_another_value(value):
    """A callable `sqrt_beta` returns a positive finite real number: any
    other value is refused at the call, with the arguments it was called
    with."""
    with pytest.raises(
        ValueError,
        match=r"sqrt_beta\(t, n_vars\) needs to return a positive finite "
        r"real number, not .* \(t = 10, n_vars = 3\)",
    ):
        acq_fcn_lcb(np.zeros((2, 3)), 9, _FixedGP(), lambda t, n_vars: value)


@pytest.mark.parametrize(
    "value",
    [1.5, np.float64(1.5), np.array(1.5), np.array([1.5])],
    ids=["float", "np.float64", "0-d", "1-element"],
)
def test_lcb_accepts_a_callable_sqrt_beta_of_a_positive_number(value):
    xi = np.zeros((2, 3))
    z, _, _ = acq_fcn_lcb(xi, 9, _FixedGP(), lambda t, n_vars: value)
    assert np.array_equal(z, np.array([[-2.0], [1.25]]))


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


def test_es_wcm_ranks_tied_training_points_in_their_order(monkeypatch):
    """ES-wcm ranks training points of equal value in their order in the
    training set, as MATLAB's stable sort does, when it takes the best."""
    bads, gp = _initial_state()
    n = gp.X.shape[0]
    y = np.random.default_rng(0).integers(0, 3, size=(n, 1)).astype(float)
    calls = []
    original_ucov = es_search_module.ucov

    def ucov(U, u, w, *args, **kwargs):
        calls.append(U.copy())
        return original_ucov(U, u, w, *args, **kwargs)

    monkeypatch.setattr(es_search_module, "ucov", ucov)
    search_es = ESSearchWM(1, 1, bads.options, rng=bads.rng)
    search_es._initialize_(
        bads.u, SimpleNamespace(X=gp.X, y=y), bads.optim_state, True
    )
    (U_best,) = calls
    # Python's sort is stable
    order = sorted(range(n), key=lambda k: y[k, 0])
    assert np.array_equal(U_best, gp.X[order[: U_best.shape[0]]])


def test_es_search_ranks_tied_candidates_in_their_order(monkeypatch):
    """The ES search ranks candidates of equal acquisition value in their
    order in the pool, as MATLAB's stable sort does: it returns the first
    of the best."""
    bads, gp = _initial_state()
    values = np.random.default_rng(0)
    generations = []

    def lcb(u, *args, **kwargs):
        # Two values, so that most candidates tie
        z = -(values.normal(size=(len(u), 1)) > 1.0).astype(float)
        generations.append((u.copy(), z.ravel()))
        return z, z, np.zeros_like(z)

    monkeypatch.setattr(es_search_module, "acq_fcn_lcb", lcb)
    mu = int(bads.options["n_search"] / bads.options["n_search_iter"])
    search_es = ESSearchWM(mu, mu, bads.options, rng=bads.rng)
    us, z = search_es(
        bads.u,
        None,
        None,
        bads.function_logger,
        gp,
        bads.optim_state,
        True,
        None,
    )
    U = np.vstack([u for u, _ in generations])
    Z = np.concatenate([z for _, z in generations])
    assert z == np.min(Z)
    # np.argmin returns the first of the minima
    assert np.array_equal(us, U[np.argmin(Z)])


def test_constraint_check_removes_evaluated_points_as_matlab():
    """contraints_check removes the candidates whose bin, of half tol_mesh,
    holds an evaluated point, and keeps one candidate per bin, the bins
    sorted, as MATLAB's uCheck with setdiff(u1, u2, 'rows') does."""
    D = 2
    tol_mesh = 2.0**-19
    X_eval = np.array([[0.5, 0.25], [0.0, 0.0], [-0.25, 0.75]])
    function_logger = SimpleNamespace(
        X=np.vstack((X_eval, np.full((5, D), np.nan))), X_max_idx=2
    )
    U = np.array(
        [
            [0.5, 0.25],  # evaluated
            [0.75, 0.0],
            [0.0, 0.0],  # evaluated
            [0.0, 0.0],  # duplicate
            [0.0, 1e-8],  # in the bin of [0, 0], evaluated
            [0.25, -0.25],
            [2.0, 0.0],  # projected onto [1, 0]
            [-0.25, 0.75 + 3e-7],  # in the bin of [-0.25, 0.75], evaluated
        ]
    )
    U_new = contraints_check(
        U, -np.ones((1, D)), np.ones((1, D)), tol_mesh, function_logger, True
    )
    # The rows that uCheck.m returns
    assert np.array_equal(U_new, [[0.25, -0.25], [0.75, 0.0], [1.0, 0.0]])


def _u_check(U, lb, ub, tol_mesh, X_eval):
    """MATLAB BADS's uCheck.m with projection, transcribed: the sorted unique
    rows of U, binned with MATLAB's round, which takes halves away from
    zero, and setdiff(u1, u2, 'rows'), the first row of each bin that holds
    no evaluated point, the bins sorted."""

    def matlab_round(q):
        frac, r = np.modf(q)
        return r + np.sign(frac) * (np.abs(frac) >= 0.5)

    U = np.unique(np.maximum(np.minimum(U, ub), lb), axis=0)
    tol = tol_mesh / 2
    evaluated = {tuple(r) for r in matlab_round(X_eval / tol)}
    first = {}
    for i, r in enumerate(matlab_round(U / tol)):
        if tuple(r) not in evaluated:
            first.setdefault(tuple(r), i)
    return U[[first[k] for k in sorted(first)]].reshape(-1, U.shape[1])


def test_constraint_check_rounds_halves_of_a_bin_away_from_zero():
    """contraints_check bins with MATLAB's round, as uCheck.m does, which
    takes a half of a bin away from zero: a candidate half a bin from an
    evaluated point, on either side of zero, is removed or kept, and two
    candidates half a bin apart share a bin or not, as in MATLAB BADS."""
    D = 2
    tol_mesh = 2.0**-19
    tol = tol_mesh / 2  # The width of a bin
    X_eval = tol * np.array([[0, 0], [-3, 4], [3, 5], [4.5, -6]])
    function_logger = SimpleNamespace(
        X=np.vstack((X_eval, np.full((5, D), np.nan))), X_max_idx=3
    )
    U = tol * np.array(
        [
            [0.5, 0],  # half a bin from [0, 0], in the bin [1, 0]
            [-0.5, 0],  # half a bin from [0, 0], in the bin [-1, 0]
            [-2.5, 4],  # in the bin of [-3, 4]
            [2.5, 5],  # in the bin of [3, 5]
            [5, -6],  # in the bin of [4.5, -6], [5, -6]
            [2.5, 2],  # in the bin [3, 2], with the next
            [3, 2],
            [0, 3],  # in the bin [0, 3], the next in [1, 3]
            [0.5, 3],
            [-1, -2],  # in the bin [-1, -2], with the next
            [-0.5, -2],
        ]
    )
    lb, ub = -np.ones((1, D)), np.ones((1, D))
    U_new = contraints_check(U, lb, ub, tol_mesh, function_logger, True)
    assert np.array_equal(U_new, _u_check(U, lb, ub, tol_mesh, X_eval))
    assert np.array_equal(
        U_new / tol,
        [[-1, -2], [-0.5, 0], [0, 3], [0.5, 0], [0.5, 3], [2.5, 2]],
    )


@pytest.mark.parametrize("D", [1, 2, 3, 6])
def test_constraint_check_keeps_the_first_candidate_of_each_bin(D):
    """Of the candidates that share a bin, duplicates included,
    contraints_check keeps the first in their order, where uCheck.m, whose
    unique sorts them, keeps the smallest, a difference within a bin (KD-B3-10
    in pybads/bads/README.md). The bins come out sorted, without those that
    hold an evaluated point."""
    rng = np.random.default_rng(D)
    tol_mesh = 2.0**-19
    tol = tol_mesh / 2  # The width of a bin
    lb, ub = -np.ones((1, D)), np.ones((1, D))
    for _ in range(50):
        # Candidates within a quarter of a bin of a few bins' centres, in a
        # random order, so that many share a bin or repeat; evaluated points
        # at some of the centres and elsewhere
        centres = tol * rng.integers(-3, 4, size=(rng.integers(1, 10), D))
        n = rng.integers(1, 80)
        U = centres[rng.integers(0, len(centres), n)]
        U = U + tol / 4 * rng.integers(-1, 2, size=(n, D))
        X_eval = np.vstack(
            (
                centres[: rng.integers(0, 3)],
                tol * rng.integers(-3, 4, size=(rng.integers(0, 3), D)),
            )
        )
        function_logger = SimpleNamespace(
            X=np.vstack((X_eval, np.full((3, D), np.nan))),
            X_max_idx=len(X_eval) - 1,
        )
        U_new = contraints_check(U, lb, ub, tol_mesh, function_logger, True)
        evaluated = {tuple(r) for r in round_half_away(X_eval / tol)}
        first = {}
        for i, r in enumerate(round_half_away(U / tol)):
            if tuple(r) not in evaluated:
                first.setdefault(tuple(r), i)
        assert np.array_equal(U_new, U[[first[k] for k in sorted(first)]])


@pytest.mark.parametrize("D", [1, 2, 3, 6])
def test_lexsort_rows_is_numpys_lexsort(D):
    """_lexsort_rows, which orders the bins of contraints_check, gives the
    stable order of np.lexsort, the first column first, on rows with many
    ties, signed zeros, infinities and NaNs."""
    rng = np.random.default_rng(D)
    for _ in range(200):
        A = rng.integers(-2, 3, size=(rng.integers(1, 60), D)).astype(float)
        A[A == 0] = rng.choice([0.0, -0.0], size=np.sum(A == 0))
        special = rng.random(A.shape)
        A[special < 0.05] = np.nan
        A[(special >= 0.05) & (special < 0.08)] = np.inf
        A[(special >= 0.08) & (special < 0.1)] = -np.inf
        assert np.array_equal(_lexsort_rows(A), np.lexsort(A.T[::-1]))


@pytest.mark.parametrize(
    "f, fs", [(1.0, 2.0), (-0.5, 0.25), (3.0, 1.0), (0.0, 1e-3)]
)
def test_hedge_reward_is_the_expected_improvement(f, fs):
    """The reward of the chosen search is sigma * (gamma * Phi(gamma) +
    phi(gamma)), with gamma = (fval_old - f) / sigma, as in MATLAB BADS's
    acqPortfolio.m, where phi is the standard normal density."""
    D = 3
    options = load_options(D, get_pybads_option_dir_path())
    hedge = ESSearchHedge(
        options["search_method"], options, rng=np.random.default_rng(0)
    )
    hedge.g = np.zeros(hedge.n_funs)
    hedge.chosen_hedge = np.array([0])
    hedge.phat = np.array([1.0, np.inf])
    fval_old = 0.0
    hedge.update_hedge(np.zeros(D), fval_old, f, fs, None, 1.0)
    gamma = (fval_old - f) / fs
    reward = fs * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    assert np.isclose(hedge.g[0], reward, rtol=1e-12, atol=0)
    assert hedge.g[1] == 0
